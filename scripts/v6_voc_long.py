#!/usr/bin/env python3
"""Long, resumable, wall-clock-budgeted student-vocoder distillation for v6 Voca/Sano (GPU-saturating fast track).

Stage 1 (steps 1..600): Multi-Resolution STFT + Mel warmup.
Stage 2 (steps >600): Drops phase-canceling raw complex L1 (`l_ph`) and enables `DiscV2` (MPD+MRD adversarial + feature matching)
+ Frozen `WavLM-SLM` Perceptual Discriminator (`SLMDiscriminator`) against Teacher `VocoderV2` 24 kHz audio `wav_t`.
Saves `voc.pt` whenever validation copy-synthesis UTMOS improves.
"""
import argparse, copy, math, os, random, sys, time
import torch, torch.nn.functional as F

R = "/work/tts"
REPO = f"{R}/repo"
sys.path.insert(0, REPO)
from sakhiya.v2_vocoder import VocoderV2, mel_spec_24k, ola64  # noqa: E402
from sakhiya.v4_slm import SLMDiscriminator, WavLMSLMHelper  # noqa: E402
from scripts.export_v2_onnx import TIER_CFG  # noqa: E402
from scripts.train_v2_voc import DiscV2, mr_complex_stft_loss, phase_loss  # noqa: E402
from scripts.train_v3_voc import stft_mag_loss, resamp16  # noqa: E402
from scripts.v2_full_eval import load_utmos  # noqa: E402


def warm_from_teacher(s_voc, t_sd):
    with torch.no_grad():
        for k, p in s_voc.state_dict().items():
            if k not in t_sd or not torch.is_floating_point(p): continue
            s = t_sd[k].float().to(p.device)
            if s.shape == p.shape:
                p.copy_(s)
            elif s.ndim == p.ndim and all(a >= b for a, b in zip(s.shape, p.shape)):
                sl = tuple(slice(0, b) for b in p.shape)
                sc = math.sqrt(s.shape[1] / p.shape[1]) if s.ndim >= 2 and s.shape[1] > p.shape[1] else 1.0
                p.copy_(s[sl] * min(sc, 1.25))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, choices=["voca", "sano"])
    ap.add_argument("--minutes", type=float, default=14.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher_voc", default=f"{R}/runs/v5_c1/voc.pt")
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--seg", type=int, default=64)
    a = ap.parse_args()
    dev = "cuda"
    os.makedirs(a.out, exist_ok=True)
    c = TIER_CFG[a.tier]
    D = torch.load(f"{R}/feats/v6_expressive_vedic_24k.pt", weights_only=False)
    pool = [x["mel"] for x in D["items"] if x.get("split") == "train" and x["mel"].shape[1] >= max(a.seg, 160)]
    val_mels = torch.stack([m[:, :160].float() for m in pool[:8]]).to(dev)
    print(f"[{a.tier}] pool={len(pool)} mel clips", flush=True)

    t_voc = VocoderV2().to(dev).eval()
    vck = torch.load(a.teacher_voc, map_location=dev, weights_only=False)
    t_sd = vck.get("G_ema", vck.get("G", vck))
    t_voc.load_state_dict({k: v.float() for k, v in t_sd.items()}, strict=False)
    s_voc = VocoderV2(ch=c["v_ch"], res_kernels=c["v_rk"]).to(dev)
    step0 = 0
    if os.path.exists(f"{a.out}/voc.pt"):
        ck = torch.load(f"{a.out}/voc.pt", map_location=dev, weights_only=False)
        s_voc.load_state_dict(ck.get("G_ema", ck["G"]), strict=False); step0 = int(ck.get("step", 0))
    else:
        warm_from_teacher(s_voc, t_sd)
    s_voc.src.f0_net.load_state_dict(t_voc.src.f0_net.state_dict())
    for p in s_voc.src.f0_net.parameters(): p.requires_grad_(False)
    ema = copy.deepcopy(s_voc).eval()

    D_gan = DiscV2().to(dev)
    D_slm = SLMDiscriminator().to(dev)
    slm = WavLMSLMHelper(dev)
    utmos = load_utmos(dev)
    base_lr = 1.2e-4 if step0 >= 5000 else 3.0e-4
    opt = torch.optim.AdamW([p for p in s_voc.parameters() if p.requires_grad], lr=base_lr, betas=(0.8, 0.99), weight_decay=1e-4)
    oD = torch.optim.AdamW(list(D_gan.parameters()) + list(D_slm.parameters()), lr=base_lr * 0.7, betas=(0.8, 0.99))

    with torch.no_grad():
        wv0 = ema(val_mels)
        best_utmos = float(utmos(resamp16(wv0), torch.full((8,), 16000, device=dev)).mean())
    print(f"[{a.tier}-VocLong] resume step={step0} initial_utmos={best_utmos:.3f}", flush=True)

    t_start, budget, step = time.time(), a.minutes * 60.0, step0
    while (time.time() - t_start) < budget:
        frac = min(1.0, (time.time() - t_start) / max(budget, 1.0))
        lr = base_lr * (0.5 * (1 + math.cos(math.pi * frac)) * 0.88 + 0.12)
        for g in opt.param_groups: g["lr"] = lr
        step += 1
        mels = torch.stack([m[:, s:s + a.seg] for m in random.sample(pool, a.bs) for s in [random.randint(0, m.shape[1] - a.seg)]]).to(dev).float()
        with torch.no_grad():
            wav_t = t_voc(mels)[:, :a.seg * 256]
            mel_t = mel_spec_24k(wav_t, dev)[:, :, :a.seg]
        use_gan = (step > 600)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            wav_s = s_voc(mels)[:, :a.seg * 256]
            if use_gan:
                dr, fr_d = D_gan(wav_t[:, None]); df, _ = D_gan(wav_s.detach()[:, None])
                l_d = sum(F.relu(1.0 - r.float()).mean() + F.relu(1.0 + f.float()).mean() for r, f in zip(dr, df)) / len(dr)
                if step % 2 == 0:
                    w_r16, w_h16 = resamp16(wav_t), resamp16(wav_s)
                    with torch.no_grad(): ssl_r, _ = slm.extract(w_r16)
                    ssl_h, _ = slm.extract(w_h16)
                    d_r, _ = D_slm(ssl_r.detach()); d_f, _ = D_slm(ssl_h.detach())
                    l_d = l_d + torch.mean((d_r - 1.0) ** 2) + torch.mean(d_f ** 2)
        if use_gan:
            oD.zero_grad(set_to_none=True); l_d.backward(); oD.step()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            mel_s = mel_spec_24k(wav_s, dev)[:, :, :a.seg]
            l_mel = F.l1_loss(mel_s, mel_t)
            l_stft = mr_complex_stft_loss(wav_s, wav_t, dev) + 0.8 * phase_loss(wav_s, wav_t, 512, dev)
            loss = 45.0 * l_mel + 2.5 * l_stft
            if use_gan and (step - step0) > 40:
                fr = [[t.detach() for t in fs] for fs in fr_d]; df_g, ff_g = D_gan(wav_s[:, None])
                l_adv = sum(F.relu(1.0 - f.float()).mean() for f in df_g) / len(df_g)
                l_fm = 2.0 * sum(F.l1_loss(b_.float(), a_.float()) for A, B in zip(fr, ff_g) for a_, b_ in zip(A, B)) / len(fr)
                loss = loss + 1.5 * l_adv + 1.2 * l_fm
                if step % 2 == 0:
                    _, fm_r = D_slm(ssl_r.detach()); d_g, fm_g = D_slm(ssl_h)
                    l_slm = 0.8 * torch.mean((d_g - 1.0) ** 2) + 2.5 * sum(F.l1_loss(x_, y_.detach()) for x_, y_ in zip(fm_g, fm_r))
                    loss = loss + l_slm
        opt.zero_grad(set_to_none=True); loss.backward()
        if torch.isfinite(torch.nn.utils.clip_grad_norm_(s_voc.parameters(), 1.0)): opt.step()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), s_voc.parameters()): pe.lerp_(p, 0.015)
        if (step - step0) % 150 == 0:
            with torch.no_grad():
                wv = ema(val_mels)
                cur_u = float(utmos(resamp16(wv), torch.full((8,), 16000, device=dev)).mean())
            if cur_u >= best_utmos - 0.05:
                best_utmos = max(best_utmos, cur_u)
                tmp_p = f"{a.out}/voc.pt.tmp"
                torch.save(dict(G_ema=ema.state_dict(), G=s_voc.state_dict(), step=step, tier=a.tier, cfg=c, utmos=cur_u), tmp_p)
                os.replace(tmp_p, f"{a.out}/voc.pt")
            print(f"[{a.tier}-VocLong] step={step} mel={l_mel.item():.3f} utmos={cur_u:.3f} (best={best_utmos:.3f}) t={time.time()-t_start:.0f}s", flush=True)
    print(f"[{a.tier}-VocLong] DONE step={step} best_utmos={best_utmos:.3f}", flush=True)


if __name__ == "__main__":
    main()
