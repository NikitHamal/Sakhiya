#!/usr/bin/env python3
"""Distill Sakhiya-Pro v3 + V1/Sano Teacher into each Shrink-Ladder tier (t30/t15/t10/t5/t1).

Combines:
1. Warm-initialization from pre-trained Sakhiya Acoustic Models & Harmonic-iSTFT Vocoders (`/work/tts/ck/*.pt`).
2. High-speed Acoustic + Prosody Distillation against Teacher AM (`v5_c1_am.pt` + `v9_voca_am.pt` + `v9_sano_am.pt`) AND
   Sakhiya-Pro v3 Native-Nepal corpus (`/work/sp/distill/cache_train.pt`: 2,928 single-voice utterances,
   calibrating duration predictor, F0 contour, and downward Nepali sentence closures).
3. End-to-End Differentiable Harmonic-iSTFT Vocoder-in-the-Loop Co-Distillation (`mrstft` + 33-bin complex
   phase/group-delay + 100-band log-mel L1 + Frozen `WavLM-base-plus-sv` perceptual loss every 2nd step).
4. Straight-Through Estimator (STE) Per-Channel INT8 QAT (`--qat_start 0.35`).
5. Step-50 stdout throughput watchdog + resumable checkpointing (`ema_latest.pt`, `state.pt`).
"""
import argparse, copy, json, math, os, random, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for p in (HERE, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from student import build, set_qat, n_params, SR, N_FFT, HOP, N_MEL, TIERS, _calibrate_styles, ste_int8_weight, snap_f0_int8, snap_am_int8  # noqa: E402
from sakhiya.cfm import SakhiyaAM  # noqa: E402
from sakhiya.v2_vocoder import VocoderV2  # noqa: E402

LR = dict(t30=2.0e-5, t15=2.5e-5, t10=3.0e-5, t5=3.5e-5, t1=1.2e-4)


class Mel(nn.Module):
    def __init__(s):
        super().__init__()
        import librosa
        s.register_buffer("fb", torch.tensor(librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MEL, fmin=0, fmax=12000), dtype=torch.float32))
        s.register_buffer("win", torch.hann_window(N_FFT))

    def forward(s, w):
        pad = (N_FFT - HOP) // 2
        x = F.pad(w.float()[:, None], (pad, pad), mode="reflect")[:, 0]
        S = torch.stft(x, N_FFT, HOP, N_FFT, window=s.win, center=False, return_complex=True).abs()
        return torch.log(torch.clamp(s.fb @ S, min=1e-5))


def mrstft(y, yh):
    tot = 0.0
    for n in (512, 1024, 2048):
        win = torch.hann_window(n, device=y.device)
        a = torch.stft(y.float(), n, n // 4, window=win, return_complex=True).abs()
        b = torch.stft(yh.float(), n, n // 4, window=win, return_complex=True).abs()
        tot = tot + torch.norm(a - b) / (torch.norm(a) + 1e-7) + F.l1_loss(torch.log(b + 1e-5), torch.log(a + 1e-5))
    return tot


def spec_phase_gd_loss(re_s, im_s, re_t, im_t):
    mag_s = torch.sqrt(re_s.float() ** 2 + im_s.float() ** 2 + 1e-7)
    mag_t = torch.sqrt(re_t.float() ** 2 + im_t.float() ** 2 + 1e-7)
    w = (mag_t / (mag_t.mean(dim=(1, 2), keepdim=True) + 1e-5)).clamp(max=3.0).detach()
    nr, ni = re_s.float() / (mag_s + 1e-5), im_s.float() / (mag_s + 1e-5)
    tr, ti = re_t.float() / (mag_t + 1e-5), im_t.float() / (mag_t + 1e-5)
    l_cos = ((1.0 - (nr * tr + ni * ti)) * w).mean()
    if_sr = nr[:, :, 1:] * nr[:, :, :-1] + ni[:, :, 1:] * ni[:, :, :-1]
    if_si = ni[:, :, 1:] * nr[:, :, :-1] - nr[:, :, 1:] * ni[:, :, :-1]
    if_tr = tr[:, :, 1:] * tr[:, :, :-1] + ti[:, :, 1:] * ti[:, :, :-1]
    if_ti = ti[:, :, 1:] * tr[:, :, :-1] - tr[:, :, 1:] * ti[:, :, :-1]
    l_if = ((1.0 - (if_sr * if_tr + if_si * if_ti)) * w[:, :, 1:]).mean()
    l_mag = F.l1_loss(torch.log(mag_s + 1e-5), torch.log(mag_t + 1e-5))
    return l_mag + 0.35 * (l_cos + l_if)


def install_ste_qat_hooks(model):
    hooks = []
    for name, m in model.named_modules():
        if name.startswith("voc.src.f0_net") or hasattr(m, "qat_enabled"):
            continue
        if isinstance(m, (nn.Conv1d, nn.Linear)) and m.weight.numel() >= 256:
            if name.endswith(("vel.out", "enc.mu", "dur.o", "voc.post")):
                continue
            def _pre(mod, inp):
                if mod.training and getattr(model, "_qat_flag", False):
                    mod._orig_w = mod.weight.data.clone()
                    mod.weight.data.copy_(ste_int8_weight(mod.weight, axis=0, enabled=True).data)
            def _post(mod, inp, out):
                if hasattr(mod, "_orig_w"):
                    mod.weight.data.copy_(mod._orig_w)
                    delattr(mod, "_orig_w")
            hooks.append(m.register_forward_pre_hook(_pre))
            hooks.append(m.register_forward_hook(_post))
    return hooks


def load_data(path, seg, dev):
    D = torch.load(path, weights_only=False)
    tr, dv = [], []
    for x in D["items"]:
        if len(x["wav"]) // HOP < seg + 8 or len(x["ids"]) < 4:
            continue
        x["wav"] = x["wav"].to(dev)
        x["f0"] = x["f0"].to(dev)
        (dv if x["split"] == "dev" else tr).append(x)
    return tr, dv, float(D["f0_mu"]), float(D["f0_sd"])


def make_batch(items, B, dev, mu, sd):
    xs = random.sample(items, B)
    L = max(len(x["ids"]) for x in xs)
    T = max(len(x["f0"]) for x in xs)
    ids = torch.zeros(B, L, dtype=torch.long)
    wav = torch.zeros(B, T * HOP, device=dev)
    f0 = torch.zeros(B, T, device=dev)
    il = torch.tensor([len(x["ids"]) for x in xs])
    tl = torch.tensor([len(x["f0"]) for x in xs])
    for b, x in enumerate(xs):
        ids[b, :il[b]] = x["ids"]
        wav[b, :len(x["wav"])] = x["wav"].float() / 32767.0
        f0[b, :tl[b]] = x["f0"]
    vuv = (f0 > 0).float()
    lf = torch.where(f0 > 0, (torch.log(f0.clamp(min=1.0)) - mu) / sd, torch.zeros_like(f0))
    lang = torch.tensor([x["lang"] for x in xs])
    return ids.to(dev), il.to(dev), tl.to(dev), lang.to(dev), wav, lf, vuv


def _ode_solve(vel_net, mu_seg, g, m_seg, steps):
    B, _, T = mu_seg.shape
    if steps == 1:
        t0 = torch.zeros((B,), device=mu_seg.device, dtype=torch.float32)
        return mu_seg + vel_net(mu_seg, t0, mu_seg, g, m_seg)
    c_idx = torch.arange(100, device=mu_seg.device, dtype=torch.float32)[None, :, None]
    t_idx = torch.arange(T, device=mu_seg.device, dtype=torch.float32)[None, None, :]
    z_init = torch.sin(c_idx * 12.9898 + t_idx * 78.233) * torch.cos(c_idx * 39.346 + t_idx * 11.135) * (0.667 * 1.4142)
    xt = z_init.expand(B, 100, T)
    dt = 1.0 / steps
    for i in range(steps):
        ti = torch.full((B,), i * dt, device=mu_seg.device, dtype=torch.float32)
        k1 = vel_net(xt, ti, mu_seg, g, m_seg)
        k2 = vel_net(xt + 0.5 * dt * k1, ti + 0.5 * dt, mu_seg, g, m_seg)
        xt = xt + dt * k2
    return xt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=120)
    ap.add_argument("--bs", type=int, default=20)
    ap.add_argument("--seg", type=int, default=48)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--gan_start", type=int, default=2500)
    ap.add_argument("--warm_uniform", type=int, default=400)
    ap.add_argument("--qat_start", type=float, default=0.35)
    ap.add_argument("--max_step_s", type=float, default=3.0)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_steps", type=int, default=10 ** 9)
    ap.add_argument("--fresh_clock", action="store_true")
    ap.add_argument("--lr_mult", type=float, default=1.0)
    a = ap.parse_args()
    assert a.minutes <= 135, "hard 135 min cap"

    dev = "cuda"
    os.makedirs(a.out, exist_ok=True)
    random.seed(0)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    logp_ = a.out + "/train.log"

    def log(s):
        print(s, flush=True)
        open(logp_, "a").write(s + "\n")

    tr, dv, mu, sd = load_data(a.data, a.seg, dev)
    G = build(a.tier).to(dev)
    G.f0_stats.copy_(torch.tensor([mu, sd], device=dev))
    G.warm_init("/work/tts/ck")
    install_ste_qat_hooks(G)

    # Load frozen Teacher AM (`sano` for t1 matching d_enc=96, `voca` for t5/t10 matching d_enc=144, `v5_c1` for t15/t30 matching d_enc=192)
    t_cfg = TIERS[a.tier]
    t_am = SakhiyaAM(
        d_enc=t_cfg["d_enc"], L_enc=t_cfg["L_enc"], ff_enc=t_cfg["ff_enc"],
        d_vel=t_cfg["d_vel"], nb_vel=(12 if t_cfg["d_enc"] == 192 else (7 if t_cfg["d_enc"] == 144 else 5)),
        h_dur=t_cfg["h_dur"]
    ).to(dev).eval()
    t_am_p = "/work/tts/ck/v5_c1_am.pt" if t_cfg["d_enc"] == 192 else ("/work/tts/ck/v9_voca_am.pt" if t_cfg["d_enc"] == 144 else "/work/tts/ck/v9_sano_am.pt")
    if os.path.exists(t_am_p):
        ck_t = torch.load(t_am_p, map_location=dev, weights_only=False)
        t_am.load_state_dict({k: v.float() for k, v in ck_t.get("ema", ck_t.get("net", ck_t)).items() if not k.startswith("ctc_rec.")}, strict=False)
        t_am.stats.copy_(torch.tensor([float(ck_t.get("mean", -6.1760635)), float(ck_t.get("std", 2.4932315))], device=dev))
        _calibrate_styles(t_am, pre_baked=bool(ck_t.get("pre_baked_v25", False)))
    for p_ in t_am.parameters():
        p_.requires_grad_(False)

    # Frozen high-capacity V1 Teacher Vocoder (320-ch) for Complex STFT / Phase / WavLM supervision
    t_voc = VocoderV2(ch=320, res_kernels=(3, 7, 11)).to(dev).eval()
    if os.path.exists("/work/tts/ck/v5_c1_voc.pt"):
        vck_t = torch.load("/work/tts/ck/v5_c1_voc.pt", map_location=dev, weights_only=False)
        t_voc.load_state_dict({k: v.float() for k, v in vck_t.get("G_ema", vck_t.get("G", vck_t)).items()}, strict=True)
        t_voc.fuse_weight_norm()
    for p_ in t_voc.parameters():
        p_.requires_grad_(False)

    mel_fn = Mel().to(dev)
    wavlm = None
    try:
        from transformers import WavLMModel
        for wname in ("microsoft/wavlm-base-plus-sv", "microsoft/wavlm-base-plus"):
            try:
                wavlm = WavLMModel.from_pretrained(wname).to(dev).eval()
                for p_ in wavlm.parameters():
                    p_.requires_grad_(False)
                log(f"[wavlm] loaded frozen {wname} (subsampled every 2nd step)")
                break
            except Exception:
                pass
    except Exception as e:
        log(f"[wavlm] skipped: {e}")

    ema = copy.deepcopy(G).eval()
    lr = (a.lr or LR[a.tier]) * a.lr_mult
    voc_lr_ratio = 0.60 if a.tier == "t1" else 0.15
    am_params = [p for n, p in G.named_parameters() if not n.startswith("voc.") and p.requires_grad]
    voc_params = [p for n, p in G.named_parameters() if n.startswith("voc.") and p.requires_grad]
    oG = torch.optim.AdamW([
        {"params": am_params, "lr": lr, "initial_lr": lr},
        {"params": voc_params, "lr": lr * voc_lr_ratio, "initial_lr": lr * voc_lr_ratio},
    ], betas=(0.8, 0.99), weight_decay=1e-4)

    step, spent = 0, 0.0
    if a.resume and os.path.exists(a.out + "/state.pt"):
        try:
            S = torch.load(a.out + "/state.pt", weights_only=False, map_location=dev)
            if S.get("cfg") == G.cfg:
                G.load_state_dict(S["G"], strict=True)
                ema.load_state_dict(S["ema"], strict=True)
                oG.load_state_dict(S["oG"])
                step, spent = S["step"], (0.0 if a.fresh_clock else S["spent"])
                log(f"[resume] step {step} spent {spent / 60:.1f} min")
        except Exception as e:
            log(f"[resume] warm-starting fresh Harmonic-iSTFT graph ({e})")

    log(f"[{a.tier}] Harmonic-iSTFT {G.arch} deployed params {n_params(G) / 1e6:.3f}M | train {len(tr)} dev {len(dv)} utts | lr {lr} bs {a.bs} seg {a.seg}")
    budget = a.minutes * 60
    t0 = time.time() - spent
    tstep = time.time()
    qat_on = False
    last_wlm = 0.0
    ar = torch.arange(a.seg, device=dev)
    ac = lambda: torch.autocast("cuda", dtype=torch.bfloat16)

    while True:
        spent = time.time() - t0
        prog = min(1.0, spent / budget)
        if spent > budget or step >= a.max_steps:
            break
        mult = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog))
        oG.param_groups[0]["lr"] = lr * mult
        oG.param_groups[1]["lr"] = lr * voc_lr_ratio * mult

        if prog >= a.qat_start and not qat_on:
            set_qat(G, True)
            qat_on = True
            log(f"[qat] int8 fake-quant ON at step {step}")

        ids, il, tl, lang, wav, lf, vuv = make_batch(tr, a.bs, dev, mu, sd)
        B, L = ids.shape
        T = lf.shape[1]
        im = torch.arange(L, device=dev)[None] < il[:, None]
        m_l = im[:, None, :].float()
        spk0 = torch.zeros(B, dtype=torch.long, device=dev)
        studio = torch.ones(B, dtype=torch.long, device=dev)
        lang_eff = torch.where(lang == 1, torch.full_like(lang, 4), torch.zeros_like(lang))
        sty = torch.where(lang == 1, torch.full_like(lang, 5), torch.zeros_like(lang))
        pros = G.pros_tbl[sty]
        p_add = (torch.randn(B, L, device=dev) * 0.14).clamp(-0.42, 0.42) * im.float()
        ones_m = torch.ones((B, 1), dtype=torch.float32, device=dev)

        with torch.no_grad(), ac():
            ht, mut, gt = t_am.enc(ids, spk0, lang_eff, sty, studio, im, pitch_add=p_add, prosody=pros, sty_mask=ones_m)
            ldt = t_am.dur.pred_logdur(ids, ht, im)
            mut_t = mut.transpose(1, 2)
            # Gentle duration calibration toward Sakhiya-Pro v3 utterance length tl
            dur_t = torch.exp(ldt).clamp(min=1.0, max=60.0) * im.float()
            scale_v3 = (tl.float() / dur_t.sum(1).clamp(min=1.0)).clamp(0.92, 1.08)[:, None]
            ldt_v3 = torch.log((dur_t * scale_v3).clamp(min=1.0) + 1.0) * im.float()

        with ac():
            G.train()
            hs, mus, gs = G.am.enc(ids, spk0, lang_eff, sty, studio, im, pitch_add=p_add, prosody=pros, sty_mask=ones_m)
            lds = G.am.dur.pred_logdur(ids, hs, im) + G.dur_rate_bias.clamp(-0.12, 0.12) * im.float()
            mus_t = mus.transpose(1, 2)

        l_prior = F.l1_loss(mus[im].float(), mut[im].float()) + 0.4 * F.l1_loss(hs[im].float(), ht[im].float())
        l_dur = F.l1_loss(lds[im].float(), (0.85 * ldt[im] + 0.15 * ldt_v3[im]).float())

        # Expand to frame level using teacher durations for exact frame-synchronous Harmonic-iSTFT waveform distillation
        with torch.no_grad():
            dur_int = torch.clamp(torch.round(torch.exp(ldt) - 1.0), min=3.0, max=40.0) * im.float()
            ends = torch.cumsum(dur_int, 1)
            T_exp = int(max(a.seg + 8, min(320, int(ends[:, -1].max().item()))))
            t_ar = torch.arange(0, T_exp, device=dev, dtype=torch.float32)
            idx = (t_ar[None, :, None] >= ends[:, None, :]).to(torch.float32).sum(-1).to(torch.int64)
            idx = torch.minimum(idx, (il[:, None] - 1).clamp(min=0))
            max_s0 = (ends[:, -1].long() - a.seg - 1).clamp(min=1, max=max(1, T_exp - a.seg - 1))
            s0 = (torch.rand(B, device=dev) * max_s0.float()).long()
            sel = s0[:, None] + ar[None]
            idx_seg = torch.gather(idx, 1, sel)
            mut_seg = torch.gather(mut_t.float(), 2, idx_seg[:, None, :].expand(-1, 100, -1))
            m_seg = torch.ones((B, 1, a.seg), device=dev, dtype=torch.float32)
            xt_t_seg = _ode_solve(t_am.vel, mut_seg, gt.float(), m_seg, G.steps)
            mel_t = xt_t_seg * t_am.stats[1] + t_am.stats[0]
            re_t, im_t, f0_t, _ = t_voc.spectrum_and_f0(mel_t)
            z_t = torch.zeros_like(im_t[:, :1, :])
            spec_t = torch.cat([re_t.float(), torch.cat([z_t, im_t[:, 1:-1, :], z_t], dim=1).float()], dim=1)
            y_t = F.conv_transpose1d(spec_t, G.K_istft, stride=16, padding=24)[:, 0, :]

        with ac():
            mus_seg = torch.gather(mus_t, 2, idx_seg[:, None, :].expand(-1, 100, -1))
            xt_s_seg = _ode_solve(G.am.vel, mus_seg, gs, m_seg, G.steps)
            mel_s = xt_s_seg * G.am.stats[1] + G.am.stats[0]
            y_s, re_s, im_s, f0_s, _ = G.voc_wav(mel_s)

        # Frame-level velocity + spectral sharpness + cepstral harmonic comb losses (no token-level vel distortion)
        l_vel = F.l1_loss(xt_s_seg.float(), xt_t_seg.float())
        l_df = F.l1_loss(mel_s[:, 1:, :].float() - mel_s[:, :-1, :].float(), mel_t[:, 1:, :].float() - mel_t[:, :-1, :].float())
        l_dt = F.l1_loss(mel_s[:, :, 1:].float() - mel_s[:, :, :-1].float(), mel_t[:, :, 1:].float() - mel_t[:, :, :-1].float())
        cep_s = torch.fft.rfft(mel_s.float(), dim=1).abs()[:, 4:48, :]
        cep_t = torch.fft.rfft(mel_t.float(), dim=1).abs()[:, 4:48, :]
        l_cep = F.l1_loss(cep_s, cep_t)
        l_mel = F.l1_loss(mel_s.float(), mel_t.float()) + 0.5 * F.l1_loss(mel_fn(y_s), mel_fn(y_t)) + 0.6 * l_df + 0.3 * l_dt + 0.25 * l_cep
        l_st = mrstft(y_t.float(), y_s.float())
        l_ph = spec_phase_gd_loss(re_s, im_s, re_t, im_t)
        l_pitch = F.l1_loss(G.am.enc.f0_pred(hs)[im].squeeze(-1).float(), t_am.enc.f0_pred(ht)[im].squeeze(-1).float()) * 0.25

        step += 1
        l_wlm = torch.zeros((), device=dev)
        if wavlm is not None and (step % 2 == 0):
            with ac():
                n16 = (y_t.shape[1] * 2) // 3
                y16 = F.interpolate(y_t.float()[:, None], size=n16, mode="linear", align_corners=False)[:, 0]
                yh16 = F.interpolate(y_s.float()[:, None], size=n16, mode="linear", align_corners=False)[:, 0]
                y16 = (y16 - y16.mean(-1, keepdim=True)) / y16.std(-1, keepdim=True).clamp(min=1e-5)
                yh16 = (yh16 - yh16.mean(-1, keepdim=True)) / yh16.std(-1, keepdim=True).clamp(min=1e-5)
                with torch.no_grad():
                    hy = wavlm(y16, output_hidden_states=True).hidden_states
                hyh = wavlm(yh16, output_hidden_states=True).hidden_states
                l_wlm = sum(F.l1_loss(hyh[k].float(), hy[k].detach().float()) for k in (3, 6, 9, 12) if k < len(hy)) / 4.0
                last_wlm = float(l_wlm.item())

        loss = 4.0 * l_mel + 2.5 * l_prior + 2.0 * l_vel + 1.5 * l_dur + 0.5 * l_st + 0.5 * l_ph + l_pitch + 1.2 * l_wlm
        oG.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(G.parameters(), 2.0)
        if torch.isfinite(gn):
            oG.step()

        with torch.no_grad():
            dcy = min(0.998, (1 + step) / (10 + step))
            for pe_, pg in zip(ema.parameters(), G.parameters()):
                pe_.lerp_(pg, 1 - dcy)

        if step == 50:
            sps = (time.time() - tstep) / 50
            if sps > a.max_step_s:
                log(f"WATCHDOG: step-50 {sps:.2f}s/step > {a.max_step_s}s")
                sys.exit(3)

        if step % 50 == 0:
            now = time.time()
            ips = 50 / max(1e-6, now - tstep)
            tstep = now
            log(f"Step {step} | {ips:.2f} it/s | mel {l_mel.item():.3f} st {l_st.item():.3f} ph {l_ph.item():.3f} prior {l_prior.item():.3f} dur {l_dur.item():.3f} pitch {l_vel.item():.3f} g {last_wlm:.3f} fm {gn.item() if torch.isfinite(gn) else 0.0:.3f} | {spent / 60:.1f}/{a.minutes:.0f} min qat={int(qat_on)}")

        if step in (50, 150) or step % a.save_every == 0:
            torch.save(dict(ema=ema.state_dict(), cfg=G.cfg, tier=a.tier, arch=G.arch, step=step, f0=[mu, sd], qat=qat_on), a.out + "/ema_latest.tmp")
            os.replace(a.out + "/ema_latest.tmp", a.out + "/ema_latest.pt")
            if step % (a.save_every * 2) == 0:
                torch.save(dict(G=G.state_dict(), ema=ema.state_dict(), oG=oG.state_dict(), cfg=G.cfg, arch=G.arch, step=step, spent=time.time() - t0), a.out + "/state.tmp")
                os.replace(a.out + "/state.tmp", a.out + "/state.pt")

    torch.save(dict(ema=ema.state_dict(), cfg=G.cfg, tier=a.tier, arch=G.arch, step=step, f0=[mu, sd], qat=qat_on), a.out + "/ema_final.pt")
    log(f"DONE step {step} minutes {(time.time() - t0) / 60:.1f}")


if __name__ == "__main__":
    main()
