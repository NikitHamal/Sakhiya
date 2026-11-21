#!/usr/bin/env python3
"""v13 EXPERIMENT: a brand-new acoustic model trained FROM SCRATCH (random init, no V1 weights) on the mined real data.

Data: mined Nepali news / conversational / storytelling (AssemblyAI > Gemini transcripts) + verse-matched Vedic, 520+ real
speakers in speaker slots 1..15, plus the Sakhiya corpus as speaker 0 so the website voice can stay Sakhiya (no single YouTube
speaker's voice is cloned). Style ids = the runtime ids (news 0, conversational 1, storytelling 2, expressive 4, vedic 5).
Safety (lessons from v11): loss numbers are only logged; progress is judged by AUDIO. Every --chunk-min minutes the EMA weights are
exported (V1's graph, V1 vocoder), scored on the forensic gate (int8, --gate-limit clips) and rendered to listenable samples, in a
background process so training keeps going. Stops if the audio score falls two chunks in a row after chunk 2. Hard cap <= 110 min,
step-50 throughput gate."""
import argparse, copy, json, math, os, random, subprocess, sys, time
from collections import Counter
import torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, REPO)
from sakhiya.cfm import SakhiyaAM  # noqa: E402
from sakhiya.ctc_aligner import MelCTCAligner  # noqa: E402
from sakhiya.symbols import LANG2ID  # noqa: E402
from scripts.train_v3_am import _batch_prosody  # noqa: E402
from scripts.v11_am_finetune import sak_style  # noqa: E402
R = "/work/tts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sak", default=f"{R}/feats/v6_expressive_vedic_24k.pt"); ap.add_argument("--mined", default=f"{R}/v13/mined_24k.pt")
    ap.add_argument("--ctc", default=f"{R}/feats/v31_ctc_recognizer.pt"); ap.add_argument("--voc", default=f"{R}/runs/v5_c1/voc.pt")
    ap.add_argument("--out", default=f"{R}/v13/am_scratch"); ap.add_argument("--max-minutes", type=float, default=110.0)
    ap.add_argument("--chunk-min", type=float, default=20.0); ap.add_argument("--gate-limit", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--p_sak", type=float, default=0.4); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lambda_grl", type=float, default=0.2); ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args(); assert a.max_minutes <= 115
    dev = "cuda"; os.makedirs(a.out, exist_ok=True); random.seed(0); torch.manual_seed(0)
    log_p = f"{a.out}/train.log"
    def log(s):
        print(s, flush=True); open(log_p, "a").write(s + "\n")
    S = torch.load(a.sak, weights_only=False); M = torch.load(a.mined, weights_only=False)
    mean, std = float(S.get("mean", -6.176)), float(S.get("std", 2.493))
    sak = [x for x in S["items"] if len(x["ids"]) >= 3]; mined = list(M["items"])
    for x in sak: x["spk"], x["studio"], x["sty_t"] = 0, 1, sak_style(x)
    for x in mined: x["studio"], x["sty_t"] = 0, int(x["v3_style_id"])
    for x in sak + mined: x["mel"] = x["mel"].to(dev)
    tr_s = [x for x in sak if x.get("split") == "train"]; tr_m = [x for x in mined if x.get("split") == "train"]
    tr_m_ne = [x for x in tr_m if x["lang"] == "ne"]; tr_m_sa = [x for x in tr_m if x["lang"] == "sa"]
    hrs = lambda L: round(sum(x["mel"].shape[1] for x in L) * 256 / 24000 / 3600, 2)
    log(f"[v13-am] FROM SCRATCH | sakhiya {len(tr_s)} clips {hrs(tr_s)} h | mined ne {len(tr_m_ne)} clips {hrs(tr_m_ne)} h, sa {len(tr_m_sa)} clips {hrs(tr_m_sa)} h | "
        f"mined speakers {M.get('report', {}).get('speakers')} | styles {dict(Counter(x['sty_t'] for x in tr_s + tr_m))}")
    net = SakhiyaAM().to(dev); net.stats.copy_(torch.tensor([mean, std]))
    n_par = sum(p.numel() for p in net.parameters())
    ctc = MelCTCAligner().to(dev); cc = torch.load(a.ctc, map_location=dev, weights_only=False); ctc.load_state_dict(cc.get("model", cc))
    for p in ctc.parameters(): p.requires_grad_(False)
    for m in ctc.modules():
        if isinstance(m, torch.nn.Dropout): m.p = 0.0
        if isinstance(m, torch.nn.RNNBase): m.dropout = 0.0
    ctc.train(True); net.ctc_rec = ctc
    ema = copy.deepcopy(net).train(False); ema.ctc_rec = ctc
    params = [p for n, p in net.named_parameters() if not n.startswith("ctc_rec.") and p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.98), weight_decay=1e-2)
    log(f"[v13-am] params {n_par:,} (random init, nothing loaded from V1)")

    def batch(xs):
        L = max(len(x["ids"]) for x in xs); T = max(x["mel"].shape[1] for x in xs); B = len(xs)
        ids = torch.zeros(B, L, dtype=torch.long, device=dev); dur = torch.zeros(B, L, dtype=torch.long, device=dev); y = torch.zeros(B, 100, T, device=dev)
        for k, x in enumerate(xs):
            Lk, Tm = len(x["ids"]), x["mel"].shape[1]
            ids[k, :Lk] = x["ids"].to(dev); dur[k, :Lk] = x["dur_ctc"].to(dev).long(); y[k, :, :Tm] = (x["mel"].float() - mean) / std
        il = torch.tensor([len(x["ids"]) for x in xs], device=dev); yl = torch.tensor([x["mel"].shape[1] for x in xs], device=dev)
        spk = torch.tensor([int(x["spk"]) for x in xs], device=dev); studio = torch.tensor([int(x["studio"]) for x in xs], device=dev)
        lang = torch.tensor([0 if (x["lang"] == "sa" and random.random() < 0.85) else LANG2ID[x["lang"]] for x in xs], device=dev)
        sty = torch.tensor([int(x["sty_t"]) for x in xs], device=dev)
        return ids, il, y, yl, spk, lang, sty, studio, _batch_prosody(y, yl, il), dur

    t0, budget, step, chunk, last_chunk_t = time.time(), a.max_minutes * 60.0, 0, 0, time.time()
    gates, procs = [], []
    n_s = int(round(a.bs * a.p_sak))
    def launch_check(ck_path, k):
        stg = f"{a.out}/stage_c{k}"; gj = f"{a.out}/gate_c{k}.json"; smp = f"{R}/v13/samples"
        cmd = (f"cd {REPO} && {sys.executable} scripts/export_v2_onnx.py --tier v25 --am {ck_path} --voc {a.voc} --out {stg} > {a.out}/export_c{k}.log 2>&1 && "
               f"{sys.executable} scripts/forensic_audio_gate.py --model {stg} --tiny --limit {a.gate_limit} --out {gj} > {a.out}/gate_c{k}.log 2>&1 && "
               f"{sys.executable} scripts/v13_samples.py --model {stg} --name scratchAM_c{k} --out {smp} > {a.out}/samples_c{k}.log 2>&1")
        procs.append((k, gj, subprocess.Popen(cmd, shell=True)))
    def collect():
        for k, gj, p in list(procs):
            if p.poll() is not None:
                procs.remove((k, gj, p))
                try:
                    g = json.load(open(gj)); s = (g.get("utmos_15") or 0) - 0.5 * (g.get("word_drop_count") or 0) - 0.3 * (g.get("stutter_count") or 0)
                    gates.append(dict(chunk=k, utmos=g.get("utmos_15"), drops=g.get("word_drop_count"), stutters=g.get("stutter_count"), clicks=g.get("clicks"), score=round(s, 3)))
                    open(f"{a.out}/gates.jsonl", "a").write(json.dumps(gates[-1]) + "\n")
                    log(f"[v13-am] AUDIO CHECK chunk {k}: UTMOS {g.get('utmos_15')} drops {g.get('word_drop_count')} stutters {g.get('stutter_count')} clicks {g.get('clicks')} score {s:.3f} (V1 3.717)")
                except Exception as e:
                    log(f"[v13-am] audio check chunk {k} failed rc={p.returncode} {e!r}")
    while time.time() - t0 < budget:
        frac = (time.time() - t0) / budget; step += 1
        lr = a.lr * min(1.0, step / 1000.0) * (0.5 * (1 + math.cos(math.pi * frac)) * 0.95 + 0.05)
        for g_ in opt.param_groups: g_["lr"] = lr
        if step == 50:
            sps = 50.0 / (time.time() - t0)
            log(f"[GATE] step-50 throughput {sps:.2f} steps/s -> ~{sps * budget / 1000:.1f}k steps inside the {a.max_minutes:.0f} min cap")
            if sps < 1.5: log("[GATE] too slow, aborting"); sys.exit(3)
        n_sa = max(1, (a.bs - n_s) // 4)
        xs = random.sample(tr_s, n_s) + random.sample(tr_m_ne, a.bs - n_s - n_sa) + random.sample(tr_m_sa, n_sa)
        ids, il, y, yl, spk, lang, sty, studio, pros, dur = batch(xs)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, parts = net.loss(ids, il, y, yl, spk, lang, sty, studio, None, None, pros, dur, spk_true=spk, lambda_ctc=0.22, lambda_grl=a.lambda_grl)
        opt.zero_grad(set_to_none=True); loss.backward()
        if torch.isfinite(torch.nn.utils.clip_grad_norm_(params, 1.0)): opt.step()
        with torch.no_grad():
            k_ema = min(0.999, (1 + step) / (10 + step))
            for pe, p in zip(ema.parameters(), net.parameters()): pe.lerp_(p, 1.0 - k_ema)
        if step % 200 == 0:
            log(f"[v13-am] step={step} {step / (time.time() - t0):.1f} st/s t={(time.time() - t0) / 60:.1f}m lr={lr:.2e} " + " ".join(f"{k}={float(v):.3f}" for k, v in parts.items()))
        collect()
        smoke_done = a.smoke and step >= a.smoke
        if time.time() - last_chunk_t >= a.chunk_min * 60.0 or smoke_done:
            chunk += 1; last_chunk_t = time.time()
            sd = {k: v for k, v in ema.state_dict().items() if not k.startswith("ctc_rec.")}
            ckp = f"{a.out}/c{chunk}.pt"
            torch.save(dict(ema=sd, step=step, mean=mean, std=std, pre_baked_v25=True, version="13.0-scratch"), ckp)
            torch.save(dict(ema=sd, step=step, mean=mean, std=std, pre_baked_v25=True, version="13.0-scratch"), f"{a.out}/last.pt")
            log(f"[v13-am] chunk {chunk} saved at step {step}; audio check started in background")
            launch_check(ckp, chunk)
            sc = [g["score"] for g in sorted(gates, key=lambda g: g["chunk"])]
            if len(sc) >= 4 and sc[-1] < sc[-2] < sc[-3]: log("[v13-am] audio score fell two chunks in a row -> stopping"); break
            if smoke_done: break
    sd = {k: v for k, v in ema.state_dict().items() if not k.startswith("ctc_rec.")}
    torch.save(dict(ema=sd, step=step, mean=mean, std=std, pre_baked_v25=True, version="13.0-scratch"), f"{a.out}/final.pt")
    log(f"[v13-am] training stopped at step {step}, t={(time.time() - t0) / 60:.1f}m; waiting for audio checks")
    for _, _, p in procs: p.wait()
    collect()
    json.dump(dict(steps=step, params=n_par, gates=gates), open(f"{a.out}/summary.json", "w"), indent=1)
    log(f"[v13-am] DONE steps={step} gates={json.dumps(gates)}")


if __name__ == "__main__":
    main()
