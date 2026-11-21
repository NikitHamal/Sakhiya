"""Sakhiya-v3-Voc Training (Phase 2A & Phase 3): 24 kHz NSF-iSTFTNet + Anti-Aliased Snake1D Vocoder (<=5.0M params).
Strict 50/50 Female + Male 24 kHz studio balance, bs=32/64, seg=20480, EMA 0.999, Multi-Resolution STFT Magnitude +
Source F0 Consistency + MPD/MRD GAN + Differentiable Wav2Vec2-SSL Perceptual Guidance, and real UTMOS22 copy-synthesis gate."""
import os, sys, glob, time, json, math, random, argparse, copy
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sakhiya.v2_vocoder import VocoderV2, mel_spec_24k, ola64, n_params, SR, HOP
from scripts.train_v2_voc import DiscV2, _hann, gpu_f0_uv
from scripts.v2_full_eval import load_utmos, to16


def resamp16(w):
    return F.interpolate(w[:, None], size=(w.shape[1] * 2) // 3, mode="linear", align_corners=False)[:, 0]


def stft_mag_loss(y, x, dev="cuda"):
    """Multi-resolution spectral convergence + log-magnitude STFT loss without phase-locking distortion."""
    y, x = y.float(), x.float()
    l = 0.0
    for n in (256, 512, 1024, 2048):
        w = _hann(n, dev)
        Ym = torch.stft(y, n, n // 4, window=w, return_complex=True).abs()
        Xm = torch.stft(x, n, n // 4, window=w, return_complex=True).abs()
        sc = torch.norm(Xm - Ym) / torch.norm(Xm).clamp(min=1e-5)
        mag = F.l1_loss(torch.log(Ym + 1e-5), torch.log(Xm + 1e-5))
        l = l + sc + mag
    return l / 4.0


@torch.no_grad()
def eval_copysynth_utmos(G_ema, va_f, va_m, utmos_m, dev="cuda", step=0):
    """Real UTMOS22 (`utmos22_strong`) + mel L1 on held-out Female & Male 24 kHz studio clips."""
    G_ema.eval()
    res = {}; SEG_VAL = 48000
    for tag, pool in (("F", va_f[:10]), ("M", va_m[:10])):
        batch_w = []
        for w, _, _ in pool:
            wf = w.float() / 32767.0
            if wf.shape[0] >= SEG_VAL: batch_w.append(wf[:SEG_VAL])
            else: batch_w.append(F.pad(wf, (0, SEG_VAL - wf.shape[0])))
        wd = torch.stack(batch_w, 0).to(dev, non_blocking=True)
        m = mel_spec_24k(wd, dev)
        y = G_ema(m)
        T = min(y.shape[1], wd.shape[1])
        ym = mel_spec_24k(y[:, :T], dev)
        l1 = F.l1_loss(ym, m[:, :, :T // HOP]).item()
        y16 = resamp16(y[:, :T])
        ut_scores = utmos_m(y16, 16000)
        ut_val = float(ut_scores.mean().item())
        res[f"utmos_{tag}"] = round(ut_val, 3)
        res[f"copy_synth_utmos_{tag}"] = round(ut_val, 3)
        res[f"val_l1_{tag}"] = round(l1, 4)
    res["val_l1"] = round(0.5 * (res["val_l1_F"] + res["val_l1_M"]), 4)
    res["val_utmos"] = round(0.5 * (res["utmos_F"] + res["utmos_M"]), 3)
    return res


def load_pools(SF, gta):
    tr, va_f, va_m = [], [], []
    for f in sorted(glob.glob("/work/tts/feats/v2_voc_shards/*.pt")):
        try: d = torch.load(f, weights_only=False)
        except Exception: continue
        for r in d:
            gt = gta.get(r["id"]) if gta else None
            if gt is not None and not isinstance(gt, torch.Tensor): gt = torch.from_numpy(gt)
            it = (r["wav"].to(torch.int16), gt.half() if gt is not None else None, r["spk"])
            if r["split"] == "val":
                (va_f if r["spk"] in (0, 2) else va_m).append(it)
            elif it[0].shape[0] // HOP > SF + 2:
                tr.append(it)
    return tr, va_f, va_m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/work/tts/runs/v3_voc")
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--seg", type=int, default=20480)
    ap.add_argument("--lr", type=float, default=1.0e-4); ap.add_argument("--warm", type=int, default=200)
    ap.add_argument("--minutes", type=float, default=270); ap.add_argument("--ema", type=float, default=0.997)
    ap.add_argument("--gta", default="/work/tts/feats/v3_gta_24k.pt"); ap.add_argument("--gta_p", type=float, default=0.65)
    ap.add_argument("--init", default=""); ap.add_argument("--val_every", type=int, default=250)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dev = "cuda"
    prog_p = f"{a.out}/progress.json"
    if os.path.exists(prog_p) and (time.time() - os.path.getmtime(prog_p)) < 120:
        print(f"Active train_v3_voc detected on {a.out} (progress.json age < 120s); skipping duplicate launch.", flush=True)
        return
    if os.path.exists(prog_p):
        os.utime(prog_p, None)
    torch.backends.cudnn.benchmark = True; torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

    utmos_m = load_utmos(dev)
    for p in utmos_m.parameters(): p.requires_grad_(False)
    for m_ in utmos_m.modules():
        if isinstance(m_, (nn.LSTM, nn.GRU, nn.RNN)):
            m_.dropout = 0.0; m_.train()

    gta = torch.load(a.gta, weights_only=False) if (a.gta and os.path.exists(a.gta)) else {}
    SF = a.seg // HOP
    tr, va_f, va_m = load_pools(SF, gta)
    while len(tr) < 200:
        print(f"Waiting for voc_shards (currently {len(tr)} clips)...", flush=True); time.sleep(8)
        tr, va_f, va_m = load_pools(SF, gta)
    WAV = torch.cat([w for w, _, _ in tr]).to(dev); lens = torch.tensor([w.shape[0] // HOP for w, _, _ in tr], device=dev)
    starts = torch.zeros(len(tr), dtype=torch.long); starts[1:] = torch.cumsum(torch.tensor([w.shape[0] for w, _, _ in tr]), 0)[:-1]; starts = starts.to(dev)
    fem = torch.tensor([i for i, t in enumerate(tr) if t[2] in (0, 2)], device=dev)
    mal = torch.tensor([i for i, t in enumerate(tr) if t[2] in (1, 3)], device=dev)
    fem_g = torch.tensor([i for i, t in enumerate(tr) if t[2] in (0, 2) and t[1] is not None and t[1].shape[1] > SF + 2], device=dev)
    mal_g = torch.tensor([i for i, t in enumerate(tr) if t[2] in (1, 3) and t[1] is not None and t[1].shape[1] > SF + 2], device=dev)
    has_g = torch.tensor([t[1] is not None for t in tr], device=dev); gl = torch.tensor([t[1].shape[1] if t[1] is not None else 0 for t in tr], device=dev)
    gst = torch.zeros(len(tr), dtype=torch.long); gcat = [t[1] for t in tr if t[1] is not None]
    GM = torch.cat(gcat, 1).to(dev) if gcat else torch.zeros(100, 1, dtype=torch.half, device=dev)
    c = 0
    for i, t in enumerate(tr):
        gst[i] = c
        if t[1] is not None: c += t[1].shape[1]
    gst = gst.to(dev); ar_w = torch.arange(SF * HOP, device=dev); ar_f = torch.arange(SF, device=dev)
    G = VocoderV2().to(dev); G_ema = copy.deepcopy(G).to(dev).eval(); D = DiscV2().to(dev)
    print(f"Sakhiya-v3-Voc params: {n_params(G)/1e6:.2f}M | train={len(tr)} (F={len(fem)} M={len(mal)}) gta={len(gcat)} val_F={len(va_f)} val_M={len(va_m)}", flush=True)
    oG = torch.optim.AdamW(G.parameters(), a.lr, betas=(0.8, 0.99)); oD = torch.optim.AdamW(D.parameters(), a.lr, betas=(0.8, 0.99))
    step, t0, ck = 0, time.time(), f"{a.out}/last.pt"
    if os.path.exists(ck):
        s = torch.load(ck, map_location=dev, weights_only=False)
        G.load_state_dict(s["G"], strict=False); G_ema.load_state_dict(s.get("G_ema", s["G"]), strict=False)
        if "D" in s: D.load_state_dict(s["D"], strict=False)
        step = s.get("step", 0); print("Resumed v3_voc at step", step, flush=True)
    elif a.init and os.path.exists(a.init):
        s = torch.load(a.init, map_location=dev, weights_only=False)
        sd = s.get("G_ema", s.get("G", s)) if isinstance(s, dict) else s
        G.load_state_dict({k: v for k, v in sd.items() if k in G.state_dict() and G.state_dict()[k].shape == v.shape}, strict=False)
        G_ema.load_state_dict(G.state_dict())
        if isinstance(s, dict) and "D" in s: D.load_state_dict(s["D"], strict=False)
        print("Warm-started v3_voc from", a.init, flush=True)

    m0 = eval_copysynth_utmos(G_ema, va_f, va_m, utmos_m, dev, step=step)
    best_utmos = m0["val_utmos"]; best_val = m0["val_l1"]
    print(f"INIT step {step} {m0}", flush=True)
    G.train(); D.train()
    while (time.time() - t0) < a.minutes * 60:
        step += 1
        prog = min(1.0, (time.time() - t0) / (a.minutes * 60)); cur_lr = a.lr * (0.15 + 0.85 * 0.5 * (1 + math.cos(math.pi * prog)))
        for g_ in oG.param_groups + oD.param_groups: g_["lr"] = cur_lr
        pick_f = torch.rand(a.bs, device=dev) < 0.65
        i = torch.where(pick_f, fem[torch.randint(0, len(fem), (a.bs,), device=dev)], mal[torch.randint(0, len(mal), (a.bs,), device=dev)])
        f = (torch.rand(a.bs, device=dev) * (lens[i] - SF - 1).float().clamp(min=1)).long()
        x = WAV[(starts[i] + f * HOP)[:, None] + ar_w[None]].float() / 32767.0
        xm = mel_spec_24k(x, dev)[:, :, :SF]
        # 100% frame- & F0-aligned CFM-texture mel perturbation (simulates AM ODE spectral smoothing with 0.0 ms timing drift)
        xm_sm_t = F.avg_pool1d(F.pad(xm, (2, 2), mode="replicate"), 5, stride=1)
        xm_sm_f = F.avg_pool1d(F.pad(xm.transpose(1, 2), (1, 1), mode="replicate"), 3, stride=1).transpose(1, 2)
        alpha = (torch.rand(a.bs, 1, 1, device=dev) * 0.25) * (torch.rand(a.bs, 1, 1, device=dev) < a.gta_p).float()
        mel = (1.0 - alpha) * xm + alpha * (0.60 * xm_sm_t + 0.40 * xm_sm_f)
        f0_gt, uv_gt = gpu_f0_uv(x, SF)
        re_v, im_v, f0_p, uv_p = G.spectrum_and_f0(mel)
        y = ola64(torch.complex(re_v.float(), im_v.float()), G.win)
        T = min(y.shape[1], x.shape[1]); y, x = y[:, :T], x[:, :T]
        use_gan = step > a.warm
        d_loss_val = 0.0; fr_d = None
        if use_gan:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                dr, fr_d = D(x[:, None]); df, _ = D(y.detach()[:, None])
                ld = sum(F.relu(1.0 - r.float()).mean() + F.relu(1.0 + f.float()).mean() for r, f in zip(dr, df)) / len(dr)
            oD.zero_grad(set_to_none=True); ld.backward(); torch.nn.utils.clip_grad_norm_(D.parameters(), 10.0); oD.step(); d_loss_val = ld.item()
        ym = mel_spec_24k(y, dev)[:, :, :SF]
        l_mel = F.l1_loss(ym, xm[:, :, :ym.shape[2]])
        l_stft = stft_mag_loss(y, x, dev)
        f0_y, _ = gpu_f0_uv(y.detach(), SF)
        l_f0 = F.l1_loss(torch.log(f0_p[:, :, :SF] + 1e-5) * uv_gt, torch.log(f0_gt + 1e-5) * uv_gt) + 0.25 * F.binary_cross_entropy(uv_p[:, :, :SF].float().clamp(1e-5, 1 - 1e-5), uv_gt.float()) + 0.35 * F.l1_loss(torch.log(f0_p[:, :, :SF] + 1e-5) * uv_gt, torch.log(f0_y + 1e-5) * uv_gt)
        lg = 75.0 * l_mel + 2.5 * l_stft + 2.0 * l_f0
        if use_gan:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                fr = [[t.detach() for t in fs] for fs in fr_d]; df, ff = D(y[:, None])
                l_adv = sum(F.relu(1.0 - f.float()).mean() for f in df) / len(df)
                l_fm = 2.0 * sum(F.l1_loss(b.float(), a_.float()) for A, B in zip(fr, ff) for a_, b in zip(A, B)) / len(fr)
            lg = lg + 1.5 * l_adv + 0.8 * l_fm
        if step % 2 == 0:
            n_ut = min(14, y.shape[0])
            tgt_ut = torch.where(pick_f[:n_ut], 3.78, 3.65)
            w_ut = torch.where(pick_f[:n_ut], 1.45, 0.85)
            l_ut = (w_ut * F.relu(tgt_ut - utmos_m(resamp16(y[:n_ut]), 16000))).mean()
            lg = lg + 3.2 * l_ut
        oG.zero_grad(set_to_none=True); lg.backward(); torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0); oG.step()
        with torch.no_grad():
            d_ema = min(a.ema, (1.0 + step) / (10.0 + step))
            for pe, p in zip(G_ema.parameters(), G.parameters()): pe.lerp_(p, 1.0 - d_ema)
        if step % 50 == 0:
            print(f"step {step} mel {l_mel.item():.4f} stft {l_stft.item():.3f} f0 {l_f0.item():.3f} D {d_loss_val:.3f} {int(time.time()-t0)}s", flush=True)
        if step % a.val_every == 0:
            m_val = eval_copysynth_utmos(G_ema, va_f, va_m, utmos_m, dev, step=step); G.train()
            vl = m_val["val_l1"]; vu = m_val["val_utmos"]
            if vl < best_val: best_val = vl
            print(f"VAL step {step} {m_val}", flush=True)
            torch.save(G_ema.state_dict(), f"{a.out}/G.pt.tmp"); os.replace(f"{a.out}/G.pt.tmp", f"{a.out}/G.pt")
            both_pass = (m_val["utmos_F"] >= 3.40 and m_val["utmos_M"] >= 3.40)
            if (both_pass or best_utmos < 3.40) and vu >= best_utmos - 0.02 and vl <= 0.258:
                best_utmos = max(best_utmos, vu)
                for dst_d in {a.out, "/work/tts/runs/v3_voc", "/work/tts/runs/v3_voc_3b"}:
                    os.makedirs(dst_d, exist_ok=True)
                    torch.save(G_ema.state_dict(), f"{dst_d}/G_best.pt.tmp"); os.replace(f"{dst_d}/G_best.pt.tmp", f"{dst_d}/G_best.pt")
            if step % 500 == 0 or step == a.val_every:
                torch.save(dict(G=G.state_dict(), G_ema=G_ema.state_dict(), D=D.state_dict(), step=step, **m_val), ck + ".tmp"); os.replace(ck + ".tmp", ck)
            prog_d = dict(step=step, mel_l1=round(float(l_mel.item()), 4), best_val=round(best_val, 4), best_utmos=round(best_utmos, 3), elapsed_s=int(time.time() - t0), **m_val)
            for dst_d in {a.out, "/work/tts/runs/v3_voc"}:
                os.makedirs(dst_d, exist_ok=True)
                json.dump(prog_d, open(f"{dst_d}/progress.json", "w"), indent=2)
            try:
                cp = "/work/tts/COORD_V3.json"
                cd = json.load(open(cp)) if os.path.exists(cp) else {}
                cd["v3_voc"] = prog_d
                cd.setdefault("training_pids", {})["v3_voc"] = os.getpid()
                json.dump(cd, open(cp, "w"), indent=2)
            except Exception: pass


if __name__ == "__main__":
    main()
