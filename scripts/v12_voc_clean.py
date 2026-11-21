#!/usr/bin/env python3
"""v12: clean up Sakhiya-V1's vocoder with REAL voices, verified by ear-proxy after every chunk (never by loss alone).

Why: the v11 acoustic fine-tune improved every internal metric and still sounded worse (UTMOS 3.72 -> 2.28). Here the acoustic
model is untouched; only the vocoder (what makes the waveform) learns, and every ~10 min chunk is exported and scored on the full
forensic gate using the SHIPPED V1 graphs with only vocoder.int8.onnx swapped, against the shipped V1 itself (same clips).

Data
  * real speech (waveform targets): mined Nepali news / conversational / storytelling segments with UTMOS >= 3.0, SNR >= 20 dB,
    0 clicks, no clipping, speaker-pure; plus gold Vedic chanting. 75 % of each batch is female (median F0 >= 165 Hz, measured
    with V1's own pitch tracker), as in V1's original vocoder recipe;
  * Sakhiya anchor: crops of V1's training mels through the frozen V1 vocoder. The anchor matches only the coarse mel envelope
    and the pitch / voicing track (keeps Sakhiya's timbre and pitch), NOT the fine structure, so clicks / buzz / roughness can be
    learned away from real voices.
Training: V1 vocoder init, lr 2e-5, MPD+MRD hinge GAN (discriminator warm-up first), input = exactly what the shipped graph
feeds the vocoder (bins 98/99 <- 97; decoder comb-sharpening on half the batch).
Safety: step-50 throughput gate; per chunk: export -> gate -> keep / stop (2 chunks in a row not OK, or one clearly worse);
hard total cap (default 110 min incl. gates); full resumable state on disk (mirrored to HF by hf_backup.py)."""
import argparse, copy, json, math, os, random, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, soundfile as sf, soxr, torch, torch.nn.functional as F
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, REPO)
from sakhiya.v2_vocoder import VocoderV2, mel_spec_24k, ola64, SR  # noqa: E402
from scripts.train_v2_voc import DiscV2, mr_complex_stft_loss  # noqa: E402
from scripts.train_v3_voc import resamp16  # noqa: E402
from scripts.v2_full_eval import load_utmos  # noqa: E402
R = "/work/tts"
LOGF = None


def log(*x):
    s = " ".join(str(v) for v in x); print(s, flush=True)
    if LOGF:
        with open(LOGF, "a") as f: f.write(s + "\n")


def jl(p):
    out = []
    if not os.path.exists(p): return out
    for l in open(p):
        try: out.append(json.loads(l))
        except Exception: pass
    return out


def score(f): return (f.get("utmos_15") or 0) - 0.5 * (f.get("word_drop_count") or 0) - 0.3 * (f.get("stutter_count") or 0) - 0.02 * (f.get("clicks") or 0) / 10.0


def summ(f):
    st = {}
    for c in f.get("per_clip", []): st.setdefault(c["style"], []).append(c["utmos"])
    return dict(score=round(score(f), 3), utmos=f.get("utmos_15"), drops=f.get("word_drop_count") or 0, stutters=f.get("stutter_count") or 0,
                clicks=f.get("clicks") or 0, sv_cos=f.get("sv_cos_ne_sa"), styles={k: round(float(np.mean(v)), 3) for k, v in st.items()})


def vin(mel, sharpen_mask=None):
    """What the shipped vocoder graph receives: decoder comb sharpening (optional, per sample) then bins 98/99 <- bin 97."""
    if sharpen_mask is not None:
        env = F.avg_pool1d(F.pad(mel.transpose(1, 2), (3, 3), mode="replicate"), 7, stride=1).transpose(1, 2)
        kb = torch.arange(100, device=mel.device, dtype=mel.dtype); wk = ((64.0 - kb) / 18.0).clamp(0.0, 1.0)[None, :, None]
        sh = env + (mel - env) * (1.0 + 0.12 * wk)
        mel = torch.where(sharpen_mask[:, None, None], sh, mel)
    return torch.cat([mel[:, :98], mel[:, 97:98], mel[:, 97:98]], dim=1)


def load_real(a):
    seg = {x["path"]: x for x in jl(f"{R}/asr/asr.jsonl")}
    P = [p for p, x in seg.items() if x.get("cat") in ("news", "conversational", "storytelling") and (x.get("utmos") or 0) >= a.min_utmos
         and (x.get("snr") or 0) >= a.min_snr and not x.get("clicks") and (x.get("clip") or 0) < 0.001 and (x.get("spk_min_sim") or 0) >= a.min_spk
         and 1.0 <= (x.get("dur") or 0) <= 20.0]
    nv = 0
    if os.path.exists(f"{R}/vedic/gold.json"):
        g = json.load(open(f"{R}/vedic/gold.json")); V = [p for p, v in g.items() if v.get("tier") == "gold" and 1.0 <= (v.get("dur") or 0) <= 20.0]
        nv = len(V); P += V
    random.Random(0).shuffle(P); P = P[: a.max_clips]

    def rd(p):
        try:
            y, sr = sf.read(p, dtype="float32", always_2d=False)
            if y.ndim > 1: y = y.mean(1)
            if sr != SR: y = soxr.resample(y, sr, SR)
            r = float(np.sqrt(np.mean(y ** 2)))
            if r < 1e-4 or len(y) < SR: return None
            y = np.clip(y * (0.062 / r), -0.97, 0.97)
            return p, torch.from_numpy((y * 32767.0).astype(np.int16))
        except Exception:
            return None
    with ThreadPoolExecutor(16) as ex: out = [o for o in ex.map(rd, P) if o]
    log(f"[v12] real clips selected {len(P)} (vedic gold {nv}) loaded {len(out)} hours {sum(len(w) for _, w in out) / SR / 3600:.2f}")
    return out


def main():
    global LOGF
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc_init", default=f"{R}/v11/in/v5_c1/voc.pt"); ap.add_argument("--am", default=f"{R}/v11/in/v5_c1/am.pt")
    ap.add_argument("--sak", default=f"{R}/v11/in/feats/v6_expressive_vedic_24k.pt"); ap.add_argument("--shipped", default=f"{REPO}/docs/models_v2")
    ap.add_argument("--out", default=f"{R}/v12"); ap.add_argument("--chunk-min", type=float, default=10.0); ap.add_argument("--max-chunks", type=int, default=6)
    ap.add_argument("--max-minutes", type=float, default=110.0); ap.add_argument("--gate-limit", type=int, default=30); ap.add_argument("--max-clips", type=int, default=12000)
    ap.add_argument("--min_utmos", type=float, default=3.0); ap.add_argument("--min_snr", type=float, default=20.0); ap.add_argument("--min_spk", type=float, default=0.8)
    ap.add_argument("--bs", type=int, default=24); ap.add_argument("--bs_anchor", type=int, default=16); ap.add_argument("--seg", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--d_warm", type=int, default=400); ap.add_argument("--w_anchor", type=float, default=0.5)
    ap.add_argument("--p_female", type=float, default=0.75); ap.add_argument("--inloop_every", type=int, default=1000)
    a = ap.parse_args(); assert a.max_minutes <= 115, "stages are capped under 2 h"
    T0 = time.time(); dev = "cuda"; os.makedirs(a.out, exist_ok=True); LOGF = f"{a.out}/train.log"
    random.seed(0); torch.manual_seed(0)
    base_json = f"{a.out}/gate_base.json"; base_proc = None
    if not os.path.exists(base_json):
        bs_dir = f"{a.out}/stage_base"; shutil.rmtree(bs_dir, ignore_errors=True); shutil.copytree(a.shipped, bs_dir)
        base_proc = subprocess.Popen([sys.executable, "scripts/forensic_audio_gate.py", "--model", bs_dir, "--tiny", "--limit", str(a.gate_limit), "--out", base_json],
                                     cwd=REPO, stdout=open(f"{a.out}/gate_base.log", "w"), stderr=subprocess.STDOUT)
        log("[v12] baseline gate (shipped V1 int8 graphs) started in background")
    t_voc = VocoderV2().to(dev)
    ck = torch.load(a.voc_init, map_location=dev, weights_only=False)
    t_voc.load_state_dict({k: v.float() for k, v in ck.get("G_ema", ck.get("G", ck)).items()}); t_voc.train(False)
    for p in t_voc.parameters(): p.requires_grad_(False)
    G = copy.deepcopy(t_voc); G.train(True)
    for p in G.parameters(): p.requires_grad_(True)
    G_ema = copy.deepcopy(t_voc); D = DiscV2().to(dev)
    oG = torch.optim.AdamW(G.parameters(), lr=a.lr, betas=(0.8, 0.99), weight_decay=1e-5)
    oD = torch.optim.AdamW(D.parameters(), lr=1e-4, betas=(0.8, 0.99))
    step, chunk, hist, train_s = 0, 0, [], 0.0
    st_p = f"{a.out}/state.pt"
    if os.path.exists(st_p):
        s = torch.load(st_p, map_location=dev, weights_only=False)
        G.load_state_dict(s["G"]); G_ema.load_state_dict(s["G_ema"]); D.load_state_dict(s["D"]); oG.load_state_dict(s["oG"]); oD.load_state_dict(s["oD"])
        step, chunk, hist, train_s = s["step"], s["chunk"], s["hist"], s["train_s"]; log(f"[v12] resumed from state: step {step} chunk {chunk}")
    real = load_real(a)
    nva = max(24, len(real) // 50); val_real, real = real[:nva], real[nva:]
    fem = []
    with torch.no_grad():
        for _, w in real:
            try:
                m = mel_spec_24k(w[: SR * 6].float().div(32767.0).to(dev), dev); _, _, f0, uv = t_voc.src(vin(m))
                f0, uv = f0.flatten(), uv.flatten(); n = min(len(f0), len(uv)); v = uv[:n] > 0.5
                fem.append(bool(v.sum() > 10 and float(f0[:n][v].median()) >= 165.0))
            except Exception:
                fem.append(False)
    SEGS = a.seg * 256
    lens = torch.tensor([len(w) for _, w in real], device=dev); starts = torch.cumsum(lens, 0) - lens
    WAV = torch.cat([w for _, w in real]).to(dev); ar = torch.arange(SEGS, device=dev)
    okl = (lens > SEGS + 256).cpu().tolist()
    fem_idx = torch.tensor([i for i in range(len(real)) if fem[i] and okl[i]] or [0], device=dev)
    all_idx = torch.tensor([i for i in range(len(real)) if okl[i]], device=dev)
    log(f"[v12] train clips {len(real)} (female {int(sum(fem))}) val {len(val_real)} | GPU wav {WAV.numel() * 2 / 1e9:.2f} GB")
    S = torch.load(a.sak, weights_only=False)
    sak = [x["mel"].to(dev) for x in S["items"] if x.get("split") == "train" and x["mel"].shape[1] >= a.seg + 2]
    sak_val = [x["mel"].to(dev) for x in S["items"] if x.get("split") == "val" and x["mel"].shape[1] >= 48][:24]
    del S
    utmos = load_utmos(dev)

    def ut(w):
        return float(utmos(resamp16(w.float()), torch.full((w.shape[0],), 16000, device=dev)).mean())

    def inloop(model):
        with torch.no_grad():
            s1 = float(np.mean([ut(model(vin(m[None].float(), torch.ones(1, dtype=torch.bool, device=dev)))) for m in sak_val]))
            s2 = []
            for _, w in val_real[:24]:
                x = w[: SR * 8].float().div(32767.0).to(dev)[None]; s2.append(ut(model(vin(mel_spec_24k(x, dev)))))
        return dict(sakhiya_mels=round(s1, 3), real_copysynth=round(float(np.mean(s2)), 3))

    base_in = inloop(t_voc); log(f"[v12] in-loop audio check, V1 vocoder: {json.dumps(base_in)}")

    def batch():
        pf = torch.rand(a.bs, device=dev) < a.p_female
        i = torch.where(pf, fem_idx[torch.randint(0, len(fem_idx), (a.bs,), device=dev)], all_idx[torch.randint(0, len(all_idx), (a.bs,), device=dev)])
        f = (torch.rand(a.bs, device=dev) * (lens[i] - SEGS - 1).float()).long()
        x_b = WAV[(starts[i] + f)[:, None] + ar[None]].float() / 32767.0
        mel = mel_spec_24k(x_b, dev)[:, :, : a.seg]
        ms = []
        for _ in range(a.bs_anchor):
            m = sak[random.randrange(len(sak))]; o = random.randint(0, m.shape[1] - a.seg); ms.append(m[:, o:o + a.seg].float())
        return x_b, mel, torch.stack(ms)

    total_train = a.chunk_min * 60.0 * a.max_chunks
    best, bad_streak, stop_reason = None, 0, None
    for c in hist:
        if c["ok"] and (best is None or c["cand"]["score"] > best["cand"]["score"]): best = c
    while chunk < a.max_chunks:
        if (time.time() - T0) / 60.0 + a.chunk_min + 6.0 > a.max_minutes: stop_reason = "time cap"; break
        chunk += 1; c0 = time.time(); s0 = step; early = None
        log(f"[v12] chunk {chunk} start step {step} t={(time.time() - T0) / 60:.1f}m")
        while time.time() - c0 < a.chunk_min * 60.0:
            step += 1; frac = min(1.0, (train_s + time.time() - c0) / total_train)
            lr = a.lr * min(1.0, step / 200.0) * (0.5 * (1 + math.cos(math.pi * frac)) * 0.7 + 0.3)
            for g_ in oG.param_groups: g_["lr"] = lr
            x_b, mel, ma = batch()
            sm_r = torch.rand(a.bs, device=dev) < 0.5; sm_a = torch.rand(a.bs_anchor, device=dev) < 0.5
            x_r, x_a = vin(mel, sm_r), vin(ma, sm_a)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                re_t, im_t, f0_t, uv_t = t_voc.spectrum_and_f0(x_a)
            wav_at = ola64(torch.complex(re_t.float(), im_t.float()), t_voc.win)[:, :SEGS]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                re_s, im_s, f0_s, uv_s = G.spectrum_and_f0(torch.cat([x_r, x_a]))
            wav = ola64(torch.complex(re_s.float(), im_s.float()), G.win)[:, :SEGS]
            w_r, w_a = wav[: a.bs], wav[a.bs:]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                dr, _ = D(x_b[:, None]); df, _ = D(w_r.detach()[:, None])
                l_d = sum(F.relu(1.0 - r.float()).mean() + F.relu(1.0 + q.float()).mean() for r, q in zip(dr, df)) / len(dr)
            oD.zero_grad(set_to_none=True); l_d.backward(); oD.step()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                l_mel_r = F.l1_loss(mel_spec_24k(w_r, dev)[:, :, : a.seg].float(), mel.float())
                l_stft_r = mr_complex_stft_loss(w_r, x_b, dev)
                l_mel_a = F.l1_loss(mel_spec_24k(w_a, dev)[:, :, : a.seg].float(), mel_spec_24k(wav_at, dev)[:, :, : a.seg].float())
            fa, ua = f0_s[a.bs:].float(), uv_s[a.bs:].float(); wv = uv_t.detach().float()
            l_f0 = ((torch.log2(fa.clamp(min=60)) - torch.log2(f0_t.float().clamp(min=60))).abs() * wv).sum() / wv.sum().clamp(min=1.0)
            l_uv = F.l1_loss(ua, wv)
            loss = 45.0 * l_mel_r + 2.5 * l_stft_r + a.w_anchor * (45.0 * l_mel_a + 12.0 * l_f0 + 4.0 * l_uv)
            l_adv = l_fm = torch.zeros((), device=dev)
            if step > a.d_warm:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    with torch.no_grad(): _, fr = D(x_b[:, None])
                    dg, fg = D(w_r[:, None])
                    l_adv = sum(F.relu(1.0 - q.float()).mean() for q in dg) / len(dg)
                    l_fm = sum(F.l1_loss(y.float(), z.float()) for A, B in zip(fr, fg) for z, y in zip(A, B)) / len(fr)
                loss = loss + 1.0 * l_adv + 2.0 * l_fm
            oG.zero_grad(set_to_none=True); loss.backward()
            if torch.isfinite(torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)): oG.step()
            with torch.no_grad():
                for pe, p in zip(G_ema.parameters(), G.parameters()): pe.lerp_(p, 0.002)
            if step == 50 and s0 == 0:
                sps = 50.0 / (time.time() - c0)
                log(f"[GATE] step-50 throughput {sps:.2f} steps/s -> ~{sps * a.chunk_min * 60 / 1000:.1f}k steps per {a.chunk_min:.0f}-min chunk")
                if sps < 2.0: log("[GATE] too slow, aborting"); sys.exit(3)
            if step % 200 == 0:
                log(f"[v12] step={step} {(step - s0) / (time.time() - c0):.1f} st/s lr={lr:.2e} mel_real={l_mel_r.item():.3f} stft={l_stft_r.item():.3f} "
                    f"anchor_mel={l_mel_a.item():.3f} f0={l_f0.item():.4f} uv={l_uv.item():.3f} adv={l_adv.item():.3f} fm={l_fm.item():.3f} d={l_d.item():.3f}")
            if step % a.inloop_every == 0:
                cur = inloop(G_ema); log(f"[v12] in-loop audio check step {step}: {json.dumps(cur)}  (V1 {json.dumps(base_in)})")
                if cur["sakhiya_mels"] < base_in["sakhiya_mels"] - 0.4: early = "in-loop collapse"; log("[v12] in-loop audio collapsed -> ending chunk early"); break
        train_s += time.time() - c0
        cv = f"{a.out}/c{chunk}_voc.pt"; torch.save(dict(G_ema=G_ema.state_dict(), step=step, chunk=chunk), cv)
        exp, stg = f"{a.out}/exp_c{chunk}", f"{a.out}/stage_c{chunk}"
        r = subprocess.run([sys.executable, "scripts/export_v2_onnx.py", "--tier", "v25", "--am", a.am, "--voc", cv, "--out", exp], cwd=REPO, capture_output=True, text=True)
        if r.returncode: log("[v12] export failed", r.stderr[-600:]); stop_reason = "export failed"; break
        shutil.rmtree(stg, ignore_errors=True); shutil.copytree(a.shipped, stg); shutil.copyfile(f"{exp}/vocoder.int8.onnx", f"{stg}/vocoder.int8.onnx")
        if base_proc is not None: base_proc.wait(); base_proc = None
        gj = f"{a.out}/gate_c{chunk}.json"
        subprocess.run([sys.executable, "scripts/forensic_audio_gate.py", "--model", stg, "--tiny", "--limit", str(a.gate_limit), "--out", gj], cwd=REPO, capture_output=True, text=True)
        if not os.path.exists(gj) or not os.path.exists(base_json): log("[v12] gate failed"); stop_reason = "gate failed"; break
        b, cd = summ(json.load(open(base_json))), summ(json.load(open(gj)))
        ok = cd["drops"] <= b["drops"] and cd["stutters"] <= b["stutters"] and cd["score"] >= b["score"] - 0.03
        rec = dict(chunk=chunk, step=step, t_min=round((time.time() - T0) / 60, 1), ok=ok, base=b, cand=cd, inloop=inloop(G_ema), early=early)
        hist.append(rec)
        with open(f"{a.out}/chunks.jsonl", "a") as f: f.write(json.dumps(rec) + "\n")
        torch.save(dict(G=G.state_dict(), G_ema=G_ema.state_dict(), D=D.state_dict(), oG=oG.state_dict(), oD=oD.state_dict(), step=step, chunk=chunk, hist=hist, train_s=train_s), st_p + ".tmp")
        os.replace(st_p + ".tmp", st_p)
        if ok and (best is None or cd["score"] > best["cand"]["score"]): best = rec
        log(f"[v12] CHUNK {chunk} step {step}: V1 score {b['score']} utmos {b['utmos']} | chunk score {cd['score']} utmos {cd['utmos']} drops {cd['drops']} stutters {cd['stutters']} "
            f"clicks {cd['clicks']} voice {cd['sv_cos']} -> {'OK' if ok else 'NOT OK'} | best chunk {best['chunk'] if best else None}")
        bad_streak = 0 if ok else bad_streak + 1
        if cd["score"] < b["score"] - 0.15: stop_reason = f"chunk {chunk} clearly worse than V1"; break
        if bad_streak >= 2: stop_reason = "2 chunks in a row not OK"; break
    json.dump(dict(best_chunk=best["chunk"] if best else None, stop_reason=stop_reason or "max chunks", chunks=len(hist), steps=step,
                   minutes=round((time.time() - T0) / 60, 1), inloop_v1=base_in), open(f"{a.out}/summary.json", "w"), indent=1)
    log(f"[v12] DONE best_chunk={best['chunk'] if best else None} stop={stop_reason or 'max chunks'} steps={step} t={(time.time() - T0) / 60:.1f}m")


if __name__ == "__main__":
    main()
