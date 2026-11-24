#!/usr/bin/env python3
"""GTA (Ground-Truth-Aligned) Vocoder + Student Co-Adaptation Pipeline (`scripts/sakhiya_pro/distill2/gta_polish.py`).

Eliminates:
1. Faint metallic buzz on 4-step CFM tiers (`t30`, `t15`) by GTA-adapting `v3_voc320.pt` on real 4-step ODE predicted mels
   with smooth C1 740 Hz harmonic anti-aliasing roll-off, `comb_gain=0.0`, `mr_complex_stft_loss`, and restored `DiscV2`.
2. Mid-vowel voicing shutters & AM-to-vocoder mismatch on 1-step tiers (`t10`, `t5` Micro, `t1` Pico) by:
   - Polishing `t1` (`SakhiyaPicoAM`), `t5` (`SakhiyaAM`), `t10`, and `t15` with non-mutating out-of-place STE QAT,
     gentle looped-block cosine learning rates, `MelCTCAligner` pronunciation supervision, `l_f0net`, and `l_vspec`.
   - Extracting exact frame-aligned 1-step GTA predicted mels for all clean Pro-v3 utterances in `gt_feats.pt`.
   - Distilling `v3_voc144.pt` (for `t5` Micro) and `v3_voc144_t1.pt` (for `t1` Pico, initialized via exact SVD from
     converged `v3_voc144.pt`) against frozen `v3_voc320(gt_mel)` complex-iSTFT magnitude/phase/group-delay (`spec_phase_gd_loss`)
     + direct `f0_net(pred_mel) -> f0_net(gt_mel)` voicing/F0 distillation + 24 kHz GT waveform `DiscV2` GAN + STE INT8 QAT.
   - Injecting the GTA-converged vocoders into `/work/sp/distill/runs/{t30,t15,t10,t5,t1}/ema_final.pt` & `ema_latest.pt`.
"""
import argparse, copy, json, math, os, random, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

HERE = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(HERE, "..", "distill")
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for p in (HERE, DIST, REPO, "/work/sp"):
    if p not in sys.path:
        sys.path.insert(0, p)

from student import Student, build, set_qat, snap_f0_int8, snap_am_int8, ste_int8_weight  # noqa: E402
from train_student import install_ste_qat_hooks, _ode_solve, spec_phase_gd_loss  # noqa: E402
from train_student_gt import load_clean_pro_v3_items, _mel_env  # noqa: E402
from sakhiya.cfm import STYLE_PROSODY_TARGETS, _dur_to_attn  # noqa: E402
from sakhiya.v2_vocoder import VocoderV2, mel_spec_24k, ola64, HOP, SR  # noqa: E402
from sakhiya.ctc_aligner import MelCTCAligner  # noqa: E402
from scripts.train_v2_voc import DiscV2, mr_complex_stft_loss, gpu_f0_uv  # noqa: E402


def vin_tier(mel, comb_gain: float):
    """Exact post-ODE mel conditioning used by Student.infer before feeding VocoderV2."""
    if comb_gain > 0.0:
        env = F.avg_pool1d(F.pad(mel.transpose(1, 2), (3, 3), mode="replicate"), 7, stride=1).transpose(1, 2)
        kb = torch.arange(100, device=mel.device, dtype=mel.dtype)
        wk = ((64.0 - kb) / 18.0).clamp(0.0, 1.0)[None, :, None]
        mel = env + (mel - env) * (1.0 + comb_gain * wk)
    return torch.cat([mel[:, :98, :], mel[:, 97:98, :], mel[:, 97:98, :]], dim=1)


def load_or_warm_student(tier: str, dev="cuda") -> Student:
    ck_p = f"/work/sp/distill/runs/{tier}/ema_final.pt"
    if not os.path.exists(ck_p):
        ck_p = f"/work/sp/distill/runs/{tier}/ema_latest.pt"
    if os.path.exists(ck_p):
        ck = torch.load(ck_p, map_location=dev, weights_only=False)
        cfg = ck.get("cfg") or dict(tier=tier)
        G = Student(**cfg).to(dev)
        G.load_state_dict({k: v.float() for k, v in ck.get("ema", ck).items()}, strict=False)
        return G
    G = build(tier).to(dev)
    G.warm_init("/work/tts/ck")
    return G


def polish_am(tier: str, cache, ctc, steps: int, lr_e: float, lr_v: float, bs: int = 44, dev="cuda"):
    out = f"/work/sp/distill/runs/{tier}"
    os.makedirs(out, exist_ok=True)
    logf = f"{out}/train.log"

    def log(s):
        print(s, flush=True)
        open(logf, "a").write(s + "\n")

    G = load_or_warm_student(tier, dev=dev)
    for p in G.voc.parameters():
        p.requires_grad_(False)
    set_qat(G, True)
    install_ste_qat_hooks(G)
    ema = copy.deepcopy(G).eval()

    enc_dur_p = [p for n, p in G.named_parameters() if not n.startswith(("voc.", "am.vel.")) and p.requires_grad]
    vel_p = [p for n, p in G.named_parameters() if n.startswith("am.vel.") and p.requires_grad]
    oG = torch.optim.AdamW([
        dict(params=enc_dur_p, lr=lr_e, weight_decay=1e-3),
        dict(params=vel_p, lr=lr_v, weight_decay=1e-4),
    ], betas=(0.9, 0.99))

    ck_p = f"{out}/ema_latest.pt"
    base_step = 0
    if os.path.exists(ck_p):
        try:
            base_step = int(torch.load(ck_p, map_location="cpu", weights_only=False).get("step", 0))
        except Exception:
            base_step = 0

    log(f"[{tier}-polish] starting {steps} steps from step {base_step} (lr_e={lr_e:.1e}, lr_v={lr_v:.1e}, bs={bs})")
    t0 = time.time()
    tstep = time.time()

    for st in range(1, steps + 1):
        prog = st / max(1, steps)
        mult = 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * prog))
        oG.param_groups[0]["lr"] = lr_e * mult
        oG.param_groups[1]["lr"] = lr_v * mult

        xs = random.sample(cache, bs)
        B = len(xs)
        ids = pad_sequence([x["ids"] for x in xs], batch_first=True)
        dur = pad_sequence([x["dur"] for x in xs], batch_first=True)
        p_enc = pad_sequence([x["p_enc"].float() for x in xs], batch_first=True)
        mu_t = pad_sequence([x["mu"].float() for x in xs], batch_first=True)
        f0_t = pad_sequence([x["f0"].float() for x in xs], batch_first=True)
        eng_t = pad_sequence([x["eng"].float() for x in xs], batch_first=True)
        y_clean = pad_sequence([x["y_clean"].float() for x in xs], batch_first=True).transpose(1, 2).contiguous()
        y_gt = pad_sequence([x["y_gt"].float() for x in xs], batch_first=True).transpose(1, 2).contiguous()
        il = torch.tensor([len(x["ids"]) for x in xs], device=dev)
        yl = torch.tensor([x["y_clean"].shape[0] for x in xs], device=dev)
        L, T = ids.shape[1], y_clean.shape[2]
        im = torch.arange(L, device=dev)[None] < il[:, None]
        ym = torch.arange(T, device=dev)[None] < yl[:, None]
        ymf = ym[:, None].float()
        spk = torch.zeros(B, dtype=torch.long, device=dev)
        studio = torch.ones(B, dtype=torch.long, device=dev)
        lang_eff = torch.tensor([4 if x["lang"] == 1 else 0 for x in xs], device=dev)
        sty = torch.tensor([5 if x["lang"] == 1 else 0 for x in xs], device=dev)
        ones_m = torch.ones((B, 1), device=dev)

        G.train()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            hs, mus, gs = G.am.enc(ids, spk, lang_eff, sty, studio, im, pitch_add=p_enc, prosody=G.pros_tbl[sty], sty_mask=ones_m)
            f0_s = G.am.enc.f0_pred(hs).squeeze(-1)
            eng_s = G.am.enc.energy_pred(hs).squeeze(-1).clamp(-1.2, 1.2)
            l_heads = 0.5 * (F.l1_loss(f0_s[im].float(), f0_t[im].float()) + F.l1_loss(eng_s[im].float(), eng_t[im].float()))
            mus_e = mus + (0.10 * G.pros_tbl[sty][:, 2:3] + 0.06 * eng_s)[..., None] * 0.32
            lds = G.am.dur.pred_logdur(ids, hs, im) + G.dur_rate_bias.clamp(-0.12, 0.12)
            l_dur = F.smooth_l1_loss(lds[im].float(), torch.log(dur.float().clamp(min=1.0))[im])
            l_prior = F.l1_loss(mus[im].float(), mu_t[im].float())

            attn = _dur_to_attn(dur, im, ym)
            mu_y_s = torch.einsum("blt,bld->bdt", attn.to(mus_e.dtype), mus_e) * ymf
            y_pred = _ode_solve(G.am.vel, mu_y_s, gs, ymf, G.steps) * ymf

            denom = (ymf.sum() * 100.0).clamp(min=1.0)
            env_p = _mel_env(y_pred.float(), 7)
            env_c = _mel_env(y_clean.float(), 7)
            env_g = _mel_env(y_gt.float(), 7)
            env_tgt = 0.75 * env_c + 0.25 * env_g
            l_env = ((env_p - env_tgt).abs() * ymf).sum() / denom

            l_mel = ((1.2 * (y_pred.float() - y_clean.float()).abs() + 0.8 * (y_pred.float() - y_clean.float()) ** 2) * ymf).sum() / denom + 2.0 * l_env
            df_p = y_pred[:, 1:, :].float() - y_pred[:, :-1, :].float()
            df_t = y_clean[:, 1:, :].float() - y_clean[:, :-1, :].float()
            l_df = ((df_p - df_t).abs() * ymf).sum() / denom
            l_std = ((y_pred.float().std(dim=1, keepdim=True) - y_clean.float().std(dim=1, keepdim=True)).abs() * ymf).sum() / ymf.sum().clamp(min=1.0)
            ym_t = ymf[:, :, 1:] * ymf[:, :, :-1]
            l_dt = (((y_pred[:, :, 1:].float() - y_pred[:, :, :-1].float()) - (y_clean[:, :, 1:].float() - y_clean[:, :, :-1].float())).abs() * ym_t).sum() / (ym_t.sum() * 100.0 + 1.0)
            cep_p = torch.fft.rfft(y_pred.float(), dim=1).abs()[:, 4:48, :]
            cep_t = torch.fft.rfft(y_clean.float(), dim=1).abs()[:, 4:48, :]
            l_cep = ((cep_p - cep_t).abs() * ymf).sum() / (ymf.sum() * 44.0).clamp(min=1.0)

            l_ctc = torch.zeros((), device=dev)
            if ctc is not None:
                c_val = ctc.ctc_loss(0.25 * mu_y_s.float() + 0.75 * y_pred.float(), yl, ids, il)
                if torch.isfinite(c_val):
                    l_ctc = F.relu(c_val - 0.10)

            mp_full = y_pred.float() * G.am.stats[1] + G.am.stats[0]
            mt_full = y_clean.float() * G.am.stats[1] + G.am.stats[0]
            with torch.no_grad():
                rf_t = G.voc.src.f0_net(mt_full)
            rf_s = G.voc.src.f0_net(mp_full)
            l_f0net = (F.smooth_l1_loss(rf_s[:, 0:1].float(), rf_t[:, 0:1].float(), reduction="none") * ymf).sum() / ymf.sum().clamp(min=1.0) \
                    + 1.8 * (F.l1_loss(torch.sigmoid(rf_s[:, 1:2].float()), torch.sigmoid(rf_t[:, 1:2].float()), reduction="none") * ymf).sum() / ymf.sum().clamp(min=1.0)

            l_vspec = torch.zeros((), device=dev)
            if T >= 36:
                seg_v = 32
                B_v = min(B, 16)
                s_p, s_t = [], []
                for bi in range(B_v):
                    valid_t = max(int(yl[bi].item()), seg_v + 1)
                    st_i = random.randint(0, max(0, valid_t - seg_v)) if valid_t > seg_v else 0
                    s_p.append(y_pred[bi, :, st_i:st_i + seg_v])
                    s_t.append(y_clean[bi, :, st_i:st_i + seg_v])
                mp_seg = torch.stack(s_p, dim=0) * G.am.stats[1] + G.am.stats[0]
                mt_seg = torch.stack(s_t, dim=0) * G.am.stats[1] + G.am.stats[0]
                cg = 0.10 if G.steps == 1 else 0.0
                mp_in = vin_tier(mp_seg, cg)
                mt_in = vin_tier(mt_seg, 0.0)
                with torch.no_grad():
                    re_vt, im_vt, _, _ = G.voc.spectrum_and_f0(mt_in)
                re_vs, im_vs, _, _ = G.voc.spectrum_and_f0(mp_in)
                v_val = spec_phase_gd_loss(re_vs, im_vs, re_vt, im_vt)
                if torch.isfinite(v_val):
                    l_vspec = v_val

            loss = l_mel + 2.4 * l_df + 0.9 * l_std + 1.2 * l_dt + 0.90 * l_cep + 1.5 * l_prior + 0.8 * l_dur + l_heads + 0.45 * l_ctc + 1.5 * l_f0net + 0.60 * l_vspec

        oG.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
        if torch.isfinite(gn):
            oG.step()

        with torch.no_grad():
            dcy = min(0.995, (1 + st) / (12 + st))
            for pe, pg in zip(ema.parameters(), G.parameters()):
                pe.lerp_(pg, 1.0 - dcy)

        if st == 50:
            sps = (time.time() - tstep) / 50.0
            log(f"[WATCHDOG] {tier} step-50 {1.0/max(sps,1e-4):.2f} it/s ({sps:.3f}s/step)")
            if sps > 3.5:
                sys.exit(3)

        if st % 100 == 0 or st == steps:
            now = time.time()
            ips = 100.0 / max(1e-6, now - tstep) if st % 100 == 0 else st / max(1e-6, now - t0)
            tstep = now
            log(f"Step {base_step + st} | {ips:.2f} it/s | mel {l_mel.item():.3f} env {l_env.item():.3f} prior {l_prior.item():.3f} cep {l_cep.item():.3f} dur {l_dur.item():.3f} ctc {l_ctc.item():.3f} | {(now - t0)/60:.1f}/{(steps/max(ips,1.0))/60:.0f} min qat=1")

    new_step = base_step + steps
    ck_out = dict(ema=ema.state_dict(), cfg=G.cfg, tier=tier, arch=G.arch, step=new_step, f0=[5.418, 0.203], qat=True, gt=True)
    torch.save(ck_out, f"{out}/ema_final.tmp")
    os.replace(f"{out}/ema_final.tmp", f"{out}/ema_final.pt")
    torch.save(ck_out, f"{out}/ema_latest.tmp")
    os.replace(f"{out}/ema_latest.tmp", f"{out}/ema_latest.pt")
    log(f"DONE step {new_step} minutes {(time.time() - t0)/60:.1f}")
    return ema


@torch.no_grad()
def extract_gta_mels(student_model: Student, gt_items, dev="cuda"):
    """Pre-compute exact frame-aligned predicted log-mels [100, T] for every item in gt_items using ground-truth durations."""
    student_model.eval()
    pros_tbl = student_model.pros_tbl
    stats = student_model.am.stats
    out_mels = []
    t0 = time.time()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for b0 in range(0, len(gt_items), 64):
            xs = gt_items[b0:b0 + 64]
            B = len(xs)
            ids = pad_sequence([x["ids"].long() for x in xs], batch_first=True).to(dev)
            dur = pad_sequence([x["dur"].long() for x in xs], batch_first=True).to(dev)
            il = torch.tensor([len(x["ids"]) for x in xs], device=dev)
            yl = dur.sum(1)
            T = int(yl.max().item())
            L = ids.shape[1]
            im = torch.arange(L, device=dev)[None] < il[:, None]
            ym = torch.arange(T, device=dev)[None] < yl[:, None]
            ymf = ym[:, None].float()
            spk = torch.zeros(B, dtype=torch.long, device=dev)
            lang_eff = torch.tensor([4 if x["lang"] == 1 else 0 for x in xs], device=dev)
            sty = torch.tensor([5 if x["lang"] == 1 else 0 for x in xs], device=dev)
            studio = torch.ones(B, dtype=torch.long, device=dev)
            pos = torch.arange(L, device=dev, dtype=torch.float32)[None, :]
            L_f = il[:, None].float().clamp(min=2.0)
            is_pause = (ids <= 8).float()
            phase = (pos / L_f).clamp(0.0, 1.0)
            tail_closure = -0.06 * ((phase - 0.72).clamp(min=0.0) / 0.28) * (1.0 - is_pause)
            p_arch = (0.55 * torch.sin(phase * 3.14159265) + 0.48 * torch.sin(phase * 9.424778) + 0.32 * torch.cos(phase * 15.707963) + tail_closure) * (1.0 - is_pause) - 0.28 * is_pause
            p_enc = F.avg_pool1d(F.pad(p_arch[:, None, :], (1, 1), mode="replicate"), 3, stride=1)[:, 0] * im.float()
            ones_m = torch.ones((B, 1), device=dev)
            hs, mus, gs = student_model.am.enc(ids, spk, lang_eff, sty, studio, im, pitch_add=p_enc, prosody=pros_tbl[sty], sty_mask=ones_m)
            eng_s = student_model.am.enc.energy_pred(hs).squeeze(-1).clamp(-1.2, 1.2)
            mus_e = mus + (0.10 * pros_tbl[sty][:, 2:3] + 0.06 * eng_s)[..., None] * 0.32
            attn = _dur_to_attn(dur, im, ym)
            mu_y = torch.einsum("blt,bld->bdt", attn.to(mus_e.dtype), mus_e) * ymf
            y_pred = _ode_solve(student_model.am.vel, mu_y.float(), gs.float(), ymf, student_model.steps) * ymf
            mel_full = y_pred.float() * stats[1] + stats[0]
            for k in range(B):
                Tk = int(yl[k].item())
                out_mels.append(mel_full[k, :, :Tk].half().contiguous())
    print(f"[GTA] Extracted {len(out_mels)} frame-aligned student mels in {time.time()-t0:.1f}s", flush=True)
    return out_mels


def gta_train_voc(
    tag: str,
    voc_model: VocoderV2,
    teacher_voc320: VocoderV2,
    gt_items,
    gta_mels,
    comb_gain: float,
    steps: int = 1500,
    lr: float = 6.0e-5,
    bs: int = 48,
    seg: int = 40,
    save_path: str = None,
    svd_t1: bool = False,
    dev: str = "cuda",
):
    logf = f"/work/tts/logs/{tag}.log"

    def log(s):
        print(s, flush=True)
        open(logf, "a").write(s + "\n")

    valid_idx = [idx for idx, x in enumerate(gt_items) if len(x["wav"]) // HOP > seg + 2 and gta_mels[idx].shape[1] >= len(x["wav"]) // HOP]
    WAV = torch.cat([gt_items[i]["wav"] for i in valid_idx]).to(dev)
    lens = torch.tensor([len(gt_items[i]["wav"]) // HOP for i in valid_idx], device=dev)
    starts = torch.cumsum(torch.tensor([len(gt_items[i]["wav"]) for i in valid_idx], device=dev), 0) - torch.tensor([len(gt_items[i]["wav"]) for i in valid_idx], device=dev)
    MEL_GTA = torch.cat([gta_mels[i][:, :len(gt_items[i]["wav"]) // HOP].to(dev) for i in valid_idx], dim=1)
    m_starts = torch.cumsum(lens, 0) - lens

    ar_w = torch.arange(seg * HOP, device=dev)
    ar_m = torch.arange(seg, device=dev)

    G = voc_model.to(dev).train()
    # Freeze f0_net weights in voc320 (already snapped); for voc144/voc144_t1 keep f0_net locked to voc320.src.f0_net!
    if teacher_voc320 is not None:
        G.src.f0_net.load_state_dict(teacher_voc320.src.f0_net.state_dict())
    snap_f0_int8(G)
    G_ema = copy.deepcopy(G).eval()

    D = DiscV2().to(dev).train()
    dp = "/work/tts/ck/voc_disc_state.pt"
    if os.path.exists(dp):
        try:
            s_d = torch.load(dp, map_location=dev, weights_only=False)
            D.load_state_dict(s_d.get("D", s_d), strict=False)
        except Exception as e:
            log(f"[{tag}] disc init warn: {e}")

    # Install out-of-place STE INT8 QAT on vocoder Conv1d layers (excluding post and f0_net)
    qat_active = [False]
    for name, m in G.named_modules():
        if name.startswith("src.f0_net") or name.endswith("post"):
            continue
        if isinstance(m, nn.Conv1d) and m.weight.numel() >= 256:
            def _pre(mod, inp):
                if mod.training and qat_active[0]:
                    mod._orig_w = mod.weight.data.clone()
                    mod.weight.data.copy_(ste_int8_weight(mod.weight, axis=0, enabled=True).data)
            def _post(mod, inp, out):
                if hasattr(mod, "_orig_w"):
                    mod.weight.data.copy_(mod._orig_w)
                    delattr(mod, "_orig_w")
            m.register_forward_pre_hook(_pre)
            m.register_forward_hook(_post)

    oG = torch.optim.AdamW([p for p in G.parameters() if p.requires_grad], lr=lr, betas=(0.8, 0.99), weight_decay=1e-5)
    oD = torch.optim.AdamW(D.parameters(), lr=lr, betas=(0.8, 0.99))

    t0 = time.time()
    tstep = time.time()
    log(f"[{tag}] starting {steps} GTA+GAN steps on {len(valid_idx)} utts ({WAV.numel()/24000/3600:.2f} h) | comb_gain={comb_gain}")

    for st in range(1, steps + 1):
        prog = st / max(1, steps)
        cur_lr = lr * (0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * prog)))
        for g_ in oG.param_groups + oD.param_groups:
            g_["lr"] = cur_lr
        if prog >= 0.35 and not qat_active[0]:
            qat_active[0] = True
            log(f"[{tag}] STE INT8 QAT enabled at step {st}")

        idx = torch.randint(0, len(valid_idx), (bs,), device=dev)
        f = (torch.rand(bs, device=dev) * (lens[idx] - seg - 1).float()).long()
        x = WAV[(starts[idx] + f * HOP)[:, None] + ar_w[None]].float() / 32767.0
        xm = mel_spec_24k(x, dev)
        mgta = MEL_GTA[:, (m_starts[idx] + f)[:, None] + ar_m[None]].permute(1, 0, 2).float()

        # Use 70% real student GTA predicted mels, 30% ground-truth mels
        use_gta = (torch.rand(bs, 1, 1, device=dev) < 0.70).float()
        mel_raw = use_gta * mgta + (1.0 - use_gta) * xm
        mel_in = vin_tier(mel_raw, comb_gain)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            re_v, im_v, f0_p, uv_p = G.spectrum_and_f0(mel_in)
        y = ola64(torch.complex(re_v.float(), im_v.float()), G.win)
        T = min(y.shape[1], x.shape[1])
        y, x = y[:, :T], x[:, :T]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            dr, fr_d = D(x[:, None])
            df, _ = D(y.detach()[:, None])
            ld = sum(F.relu(1.0 - r.float()).mean() + F.relu(1.0 + q.float()).mean() for r, q in zip(dr, df)) / len(dr)
        oD.zero_grad(set_to_none=True)
        ld.backward()
        torch.nn.utils.clip_grad_norm_(D.parameters(), 10.0)
        oD.step()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            ym = mel_spec_24k(y, dev)
            l_mel = F.l1_loss(ym, xm[:, :, :ym.shape[2]])
            l_st = mr_complex_stft_loss(y, x, dev)
            l_teach = torch.zeros((), device=dev)
            if teacher_voc320 is not None:
                with torch.no_grad():
                    re_t, im_t, _, _ = teacher_voc320.spectrum_and_f0(vin_tier(xm, 0.0))
                l_teach = spec_phase_gd_loss(re_v, im_v, re_t, im_t)
            fr = [[t.detach() for t in fs] for fs in fr_d]
            df, ff = D(y[:, None])
            l_adv = sum(F.relu(1.0 - q.float()).mean() for q in df) / len(df)
            n_fm = sum(len(A) for A in fr)
            l_fm = sum(F.l1_loss(b.float(), a_.float()) for A, B in zip(fr, ff) for a_, b in zip(A, B)) / n_fm
            lg = 45.0 * l_mel + 2.5 * l_st + 2.5 * l_teach + l_adv + 4.0 * l_fm

        oG.zero_grad(set_to_none=True)
        lg.backward()
        gn = torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
        if torch.isfinite(gn):
            oG.step()

        with torch.no_grad():
            for pe, p in zip(G_ema.parameters(), G.parameters()):
                pe.lerp_(p, 1.0 - 0.995)

        if st == 50:
            sps = 50.0 / max(1e-4, time.time() - tstep)
            log(f"[GATE] {tag} step-50 throughput {sps:.2f} steps/s")
            if sps < 0.8:
                sys.exit(3)

        if st % 100 == 0 or st == steps:
            now = time.time()
            ips = 100.0 / max(1e-4, now - tstep)
            tstep = now
            log(f"Step {st} mel {l_mel.item():.4f} stft {l_st.item():.3f} teach {l_teach.item():.3f} D {ld.item():.3f} | {(now-t0)/60:.1f}/{(steps/max(ips,1.0))/60:.0f} min ({ips:.1f} it/s)")

    snap_f0_int8(G_ema)
    if save_path:
        torch.save(
            dict(G_ema={k: v.detach().float().cpu() for k, v in G_ema.state_dict().items()}, step=steps, ch=(144 if "144" in tag else 320), svd_t1=svd_t1),
            save_path + ".tmp",
        )
        os.replace(save_path + ".tmp", save_path)
        log(f"VOC_GT_DONE tag={tag} steps={steps} -> {save_path}")
    return G_ema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_am", action="store_true")
    ap.add_argument("--skip_voc", action="store_true")
    a = ap.parse_args()

    dev = "cuda"
    random.seed(42)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tr_items, _, _, _ = load_clean_pro_v3_items("/work/sp/distill/gt_feats.pt", dev=dev)
    C_data = torch.load("/work/sp/distill/v3_rectified_cache.pt", map_location=dev, weights_only=False)
    cache = C_data["cache"]
    for x in cache:
        for k in ("ids", "dur", "p_enc", "mu", "f0", "eng", "y_clean", "y_gt"):
            x[k] = x[k].to(dev)

    ctc = None
    if os.path.exists("/work/tts/ck/v31_ctc_recognizer.pt"):
        cc = torch.load("/work/tts/ck/v31_ctc_recognizer.pt", map_location=dev, weights_only=False)
        ctc = MelCTCAligner().to(dev).train()
        ctc.load_state_dict(cc.get("model", cc), strict=False)
        for m_ in ctc.modules():
            if isinstance(m_, nn.Dropout):
                m_.p = 0.0
        for p_ in ctc.parameters():
            p_.requires_grad_(False)

    # 1. Polish Student Acoustic Models with non-mutating out-of-place STE QAT + gentle cosine LR
    if not a.skip_am:
        for t_am, st_am, lre, lrv, bs_am, min_done in (
            ("t1",  1600, 1.5e-4, 1.5e-4, 48, 7000),
            ("t5",  1000, 8.0e-5, 8.0e-5, 48, 5900),
            ("t10", 1000, 7.0e-5, 7.0e-5, 44, 4900),
            ("t15", 500,  2.5e-5, 2.5e-5, 36, 1800),
        ):
            ck_chk_p = f"/work/sp/distill/runs/{t_am}/ema_latest.pt"
            cur_st = int(torch.load(ck_chk_p, map_location="cpu", weights_only=False).get("step", 0)) if os.path.exists(ck_chk_p) else 0
            if cur_st >= min_done:
                print(f"[{t_am}-polish] already at step {cur_st} >= {min_done}, skipping to next stage", flush=True)
                continue
            polish_am(t_am, cache, ctc, steps=st_am, lr_e=lre, lr_v=lrv, bs=bs_am, dev=dev)

    # Ensure t30 has an up-to-date checkpoint in /work/sp/distill/runs/t30/
    os.makedirs("/work/sp/distill/runs/t30", exist_ok=True)
    m30 = load_or_warm_student("t30", dev=dev).eval()

    if not a.skip_voc:
        # 2A. GTA Fine-tune v3_voc320.pt on t30/t15 4-step predicted mels (comb_gain=0.0)
        m15 = load_or_warm_student("t15", dev=dev).eval()
        gta_mels_320 = extract_gta_mels(m15, tr_items, dev=dev)
        voc320_ema = gta_train_voc(
            tag="voc320_gta",
            voc_model=copy.deepcopy(m30.voc),
            teacher_voc320=None,
            gt_items=tr_items,
            gta_mels=gta_mels_320,
            comb_gain=0.0,
            steps=1200,
            lr=5.0e-5,
            bs=48,
            seg=40,
            save_path="/work/tts/ck/v3_voc320.pt",
            svd_t1=False,
            dev=dev,
        )

        # 2B. GTA + voc320-Teacher Distill v3_voc144.pt on t5 1-step predicted mels (comb_gain=0.10)
        m5 = load_or_warm_student("t5", dev=dev).eval()
        gta_mels_t5 = extract_gta_mels(m5, tr_items, dev=dev)
        voc144_ema = gta_train_voc(
            tag="voc144",
            voc_model=copy.deepcopy(m5.voc),
            teacher_voc320=voc320_ema,
            gt_items=tr_items,
            gta_mels=gta_mels_t5,
            comb_gain=0.10,
            steps=1600,
            lr=8.0e-5,
            bs=48,
            seg=40,
            save_path="/work/tts/ck/v3_voc144.pt",
            svd_t1=False,
            dev=dev,
        )

        # 2C. GTA + voc320-Teacher Distill v3_voc144_t1.pt on t1 1-step predicted mels (comb_gain=0.10)
        # Warm-start t1's SVD vocoder directly from the newly converged voc144_ema via exact SVD factorization!
        m1_tmp = Student(tier="t1", r_voc_k7=None, r_voc_k11=None).to(dev)
        m1_tmp.voc.load_state_dict(voc144_ema.state_dict(), strict=True)
        m1_tmp.r_voc_k7, m1_tmp.r_voc_k11 = 50, 31
        m1_tmp._apply_svd_structure()
        m1 = load_or_warm_student("t1", dev=dev).eval()
        gta_mels_t1 = extract_gta_mels(m1, tr_items, dev=dev)
        voc144_t1_ema = gta_train_voc(
            tag="voc144_t1",
            voc_model=copy.deepcopy(m1_tmp.voc),
            teacher_voc320=voc320_ema,
            gt_items=tr_items,
            gta_mels=gta_mels_t1,
            comb_gain=0.10,
            steps=1600,
            lr=8.0e-5,
            bs=48,
            seg=40,
            save_path="/work/tts/ck/v3_voc144_t1.pt",
            svd_t1=True,
            dev=dev,
        )

    # 3. Inject newly converged GTA vocoders into all 5 student tier checkpoints
    v320_sd = torch.load("/work/tts/ck/v3_voc320.pt", map_location="cpu", weights_only=False)["G_ema"]
    v144_sd = torch.load("/work/tts/ck/v3_voc144.pt", map_location="cpu", weights_only=False)["G_ema"]
    v144_t1_sd = torch.load("/work/tts/ck/v3_voc144_t1.pt", map_location="cpu", weights_only=False)["G_ema"]

    # Save t30 checkpoint first
    m30_cpu = load_or_warm_student("t30", dev="cpu").eval()
    m30_cpu.voc.load_state_dict(v320_sd, strict=True)
    snap_f0_int8(m30_cpu.voc)
    ck30 = dict(ema=m30_cpu.state_dict(), cfg=m30_cpu.cfg, tier="t30", arch=m30_cpu.arch, step=2200, f0=[5.418, 0.203], qat=True, gt=True)
    torch.save(ck30, "/work/sp/distill/runs/t30/ema_final.pt")
    torch.save(ck30, "/work/sp/distill/runs/t30/ema_latest.pt")

    for tier, vsd in (("t15", v320_sd), ("t10", v320_sd), ("t5", v144_sd), ("t1", v144_t1_sd)):
        for fn in ("ema_final.pt", "ema_latest.pt"):
            p = f"/work/sp/distill/runs/{tier}/{fn}"
            if not os.path.exists(p):
                continue
            ck = torch.load(p, map_location="cpu", weights_only=False)
            mt = Student(**ck["cfg"]).eval()
            mt.load_state_dict(ck["ema"], strict=False)
            mt.voc.load_state_dict(vsd, strict=True)
            snap_f0_int8(mt.voc)
            ck["ema"] = mt.state_dict()
            ck["qat"] = True
            torch.save(ck, p + ".tmp")
            os.replace(p + ".tmp", p)
        print(f"[inject] Updated {tier} with GTA-converged vocoder", flush=True)

    print("GTA_POLISH_ALL_DONE — launching ladder_daemon --once for all 5 tiers", flush=True)
    import subprocess
    subprocess.run([sys.executable, os.path.join(DIST, "ladder_daemon.py"), "--tiers", "t30,t15,t10,t5,t1", "--once"], check=False)
    print("GTA_LADDER_GATE_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
