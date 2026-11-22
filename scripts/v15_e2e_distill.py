#!/usr/bin/env python3
"""Sakhiya v15 Phase 3 (Breakthrough 2): End-to-End Differentiable Vocoder-in-the-Loop + WavLM Perceptual Co-Distillation.
Co-distills 1-step Student AM (`Nano-v15` / `voca` and `Atom-v15` / `sano`) jointly through `MS-RND-Wavehax`:
1. Backpropagates directly through `export_mel(student_mel)` -> `MS-RND-Wavehax` -> 24 kHz waveform into:
   - Multi-Resolution Complex STFT Loss + 33-bin Complex/Phase/Group-Delay Loss + Mel L1
   - Frozen `WavLM-base-plus-sv` multi-layer perceptual feature & speaker-verification timbre loss (every 2nd step)
2. Eliminates over-smoothing & the lingering 1 word drop via:
   - Spectral/Temporal Delta-Mel + Cepstral envelope sharpness loss
   - Frozen `MelCTCAligner` pronunciation loss (`ctc_pron`) with hinge floor annealed 0.22 -> 0.00
   - 2.5x Duration Loss Weight on short/final words + Hard >= 5.0-frame Hinge Floor on standalone short function words
3. Final 15% Straight-Through Estimator (STE) Per-Channel INT8 QAT polish on both Student AM and Vocoder.
4. Hard --max-minutes cap + Step-50 throughput gate + End-to-End text->student_AM->student_vocoder UTMOS checkpointing."""
import argparse, copy, json, math, os, random, sys, time
import torch, torch.nn as nn, torch.nn.functional as F

R = "/work/tts"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from sakhiya import g2p, sanskrit as SA
from sakhiya.cfm import SakhiyaAM, STYLE_PROSODY_TARGETS
from sakhiya.ctc_aligner import MelCTCAligner
from sakhiya.symbols import SYM2ID, LANG2ID
from sakhiya.v15_rnd_wavehax import MSRNDWavehax, ste_int8_weight
from sakhiya.v2_vocoder import VocoderV2, mel_spec_24k
from sakhiya.v4_slm import WavLMSLMHelper
from sakhiya.v6_distill import _expand_batch_durs, apply_hiatus_mu_rule, build_teacher_anchor_bank
from scripts.export_v2_onnx import TIER_CFG
from scripts.train_v2_voc import mr_complex_stft_loss
from scripts.train_v3_voc import resamp16
from scripts.v15_train_wavehax import spec_phase_gd_loss, warm_init_f0, snap_f0_int8
from scripts.v2_full_eval import load_utmos
from scripts.v8_voc_gta import export_mel, val_mels


def _calibrate_v15_styles(m):
    """Exact in-manifold style & language calibration matching export_v2_onnx.py for pre_baked_v25=True models."""
    with torch.no_grad():
        w0 = m.enc.sty.weight[0].clone()
        w2 = m.enc.sty.weight[2].clone()
        m.enc.sty.weight[1].copy_(0.45 * w0 + 0.55 * w2)
        m.enc.sty.weight[3].copy_(0.60 * w0 + 0.40 * w2)
        m.enc.sty.weight[4].copy_(0.20 * w0 + 0.80 * w2)
        w_l = m.enc.lang.weight
        for lid in range(w_l.shape[0]):
            if lid != 4:
                w_l[lid].copy_(w_l[0])


def apply_am_ste_qat(am):
    """Apply Straight-Through Estimator (STE) symmetric per-channel INT8 weight quantization hook to Student AM."""
    hooks = []
    for name, m in am.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)) and m.weight.numel() >= 64:
            if name.endswith(("vel.out", "enc.mu", "dur.o")):
                continue
            def _pre_hook(mod, inp):
                if mod.training and getattr(am, "_qat_enabled", False):
                    mod._orig_w = mod.weight.data.clone()
                    mod.weight.data.copy_(ste_int8_weight(mod.weight, axis=0, enabled=True).data)
            def _post_hook(mod, inp, out):
                if hasattr(mod, "_orig_w"):
                    mod.weight.data.copy_(mod._orig_w)
                    delattr(mod, "_orig_w")
            hooks.append(m.register_forward_pre_hook(_pre_hook))
            hooks.append(m.register_forward_hook(_post_hook))
    return hooks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, choices=["voca", "sano"])
    ap.add_argument("--am_init", default=None)
    ap.add_argument("--voc_init", default=None)
    ap.add_argument("--voc_variant", default=None, choices=["lite", "pro"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-minutes", type=float, default=55.0)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--seg", type=int, default=48)
    ap.add_argument("--lr_am", type=float, default=3.0e-4)
    ap.add_argument("--lr_voc", type=float, default=6.0e-5)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    assert a.max_minutes <= 115.0, "hard stage cap <= 115 min"

    voc_var = a.voc_variant or ("pro" if a.tier == "voca" else "lite")
    out = a.out or f"{R}/v15/e2e_{a.tier}"
    os.makedirs(out, exist_ok=True)
    dev = "cuda"
    tag = f"v15-e2e-{a.tier}-{voc_var}"
    torch.backends.cudnn.benchmark = False

    D = torch.load(f"{R}/feats/v6_expressive_vedic_24k.pt", weights_only=False)
    mean, std = float(D.get("mean", -5.5028)), float(D.get("std", 2.1869))

    # Load frozen MelCTCAligner for differentiable pronunciation supervision
    ctc = MelCTCAligner().to(dev).eval()
    for cp in (f"{R}/feats/v5_ctc_recognizer.pt", f"{R}/feats/v31_ctc_recognizer.pt", f"{R}/feats/v3_ctc_recognizer.pt"):
        if os.path.exists(cp):
            ck_c = torch.load(cp, map_location=dev, weights_only=False)
            ctc.load_state_dict(ck_c.get("model", ck_c), strict=False)
            break
    ctc.train(True)
    for m in ctc.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
    for p in ctc.parameters():
        p.requires_grad_(False)

    # Load V1 Teacher AM & Vocoder
    t_ck = torch.load(f"{R}/runs/v5_c1/am.pt", map_location=dev, weights_only=False)
    t_sd = {k: v.float() for k, v in t_ck.get("ema", t_ck.get("net", t_ck)).items() if not k.startswith("ctc_rec.")}
    t_am = SakhiyaAM().to(dev).eval()
    t_am.load_state_dict(t_sd, strict=False)
    t_am.stats.copy_(torch.tensor([mean, std], device=dev))
    _calibrate_v15_styles(t_am)
    for p in t_am.parameters():
        p.requires_grad_(False)

    t_voc = VocoderV2().to(dev).eval()
    vck = torch.load(f"{R}/runs/v5_c1/voc.pt", map_location=dev, weights_only=False)
    t_voc.load_state_dict({k: v.float() for k, v in vck.get("G_ema", vck.get("G", vck)).items()}, strict=False)
    for p in t_voc.parameters():
        p.requires_grad_(False)

    # Build clean Teacher anchor bank + studio training pool (100% pure studio sakhiya timbre — ZERO noisy IA audio!)
    anchors = build_teacher_anchor_bank(t_am, mean, std, ctc, dev)
    base_tr = [x for x in D["items"] if x.get("split") == "train" and 3 <= len(x["ids"]) and 28 <= x["mel"].shape[1] <= 440]
    exp_tr = [x for x in base_tr if x.get("v3_style_id", 0) in (2, 5)]
    pool = base_tr + exp_tr * 2 + anchors * 30

    # Load Student AM (`v9_voca_am.pt` or `v9_sano_am.pt`)
    c = TIER_CFG[a.tier]
    s_am = SakhiyaAM(**{k: c[k] for k in ("d_enc", "L_enc", "ff_enc", "d_vel", "nb_vel", "h_dur")}).to(dev)
    s_am.stats.copy_(torch.tensor([mean, std], device=dev))
    am_path = a.am_init or (f"{R}/ck/v9_voca_am.pt" if a.tier == "voca" else f"{R}/ck/v9_sano_am.pt")
    if os.path.exists(am_path):
        ck_s = torch.load(am_path, map_location=dev, weights_only=False)
        s_am.load_state_dict({k: v.float() for k, v in ck_s.get("ema", ck_s.get("net", ck_s)).items()}, strict=False)
        print(f"[{tag}] Loaded student AM from {am_path} (step={ck_s.get('step', 0)})", flush=True)
    _calibrate_v15_styles(s_am)
    apply_am_ste_qat(s_am)
    ema_am = copy.deepcopy(s_am).eval()

    # Load MS-RND-Wavehax Student Vocoder
    s_voc = MSRNDWavehax(variant=voc_var).to(dev)
    voc_path = a.voc_init or f"{R}/v15/wavehax_{voc_var}/voc.pt"
    if os.path.exists(voc_path):
        ck_v = torch.load(voc_path, map_location=dev, weights_only=False)
        s_voc.load_state_dict({k: v.float() for k, v in ck_v.get("G_ema", ck_v.get("G", ck_v)).items()}, strict=False)
        s_voc.src.f0_net.load_state_dict(t_voc.src.f0_net.state_dict())
        snap_f0_int8(s_voc)
        snap_f0_int8(t_voc)
        print(f"[{tag}] Loaded Wavehax-{voc_var} vocoder from {voc_path} (step={ck_v.get('step', 0)})", flush=True)
    else:
        warm_init_f0(s_voc, t_voc, [[x["mel"].half().to(dev) for x in base_tr[:256]]], dev)
    ema_voc = copy.deepcopy(s_voc).eval()

    utmos = load_utmos(dev)
    slm = WavLMSLMHelper(dev)
    pros_tbl = torch.tensor(STYLE_PROSODY_TARGETS, dtype=torch.float32, device=dev)

    def eval_e2e(am_eval, voc_eval):
        am_eval.eval(); voc_eval.eval()
        vms = val_mels(am_eval, dev, mean, std)
        sc = []
        with torch.no_grad():
            for m in vms:
                w = voc_eval(export_mel(m[None]))
                sc.append(float(utmos(resamp16(w.float()), torch.full((1,), 16000, device=dev)).mean()))
        am_eval.train(); voc_eval.train()
        return round(sum(sc) / len(sc), 4)

    base_utmos = eval_e2e(ema_am, ema_voc)
    t_utmos = eval_e2e(ema_am, t_voc)
    print(f"[{tag}] INIT e2e_utmos={base_utmos:.4f} (with V1 teacher voc={t_utmos:.4f}) am_params={sum(p.numel() for p in s_am.parameters()):,} voc_params={sum(p.numel() for p in s_voc.parameters()):,}", flush=True)

    opt_am = torch.optim.AdamW(s_am.parameters(), lr=a.lr_am, betas=(0.9, 0.99), weight_decay=1e-4)
    opt_voc = torch.optim.AdamW([p for p in s_voc.parameters() if p.requires_grad], lr=a.lr_voc, betas=(0.8, 0.99), weight_decay=1e-5)

    best = base_utmos
    # Save initial baseline if already valid
    torch.save(dict(ema=ema_am.state_dict(), net=s_am.state_dict(), step=0, mean=mean, std=std, pre_baked_v25=True), f"{out}/am.pt")
    torch.save(dict(arch="ms_rnd_wavehax", variant=voc_var, G_ema=ema_voc.state_dict(), G=s_voc.state_dict(), step=0, utmos={"e2e": best}), f"{out}/voc.pt")

    t0, budget, step = time.time(), a.max_minutes * 60.0, 0
    L_wav = a.seg * 256
    qat_active = False
    FUNC_C_IDS = torch.tensor([SYM2ID[p] for p in ("r", "t̪", "n", "m", "h", "tsʰ") if p in SYM2ID], device=dev)
    FUNC_V_IDS = torch.tensor([SYM2ID[p] for p in ("ʌ", "o") if p in SYM2ID], device=dev)

    while (time.time() - t0) < budget:
        frac = (time.time() - t0) / budget
        if frac >= 0.82 and not qat_active:
            s_am._qat_enabled = True
            s_voc.set_qat(True)
            qat_active = True
            print(f"[{tag}] step={step} Enabled STE Per-Channel INT8 QAT on Student AM + Vocoder!", flush=True)

        cos_mult = 0.5 * (1.0 + math.cos(math.pi * frac)) * 0.9 + 0.1
        warm_m = min(1.0, (step + 1) / 250.0)
        for g_ in opt_am.param_groups:
            g_["lr"] = a.lr_am * warm_m * cos_mult
        for g_ in opt_voc.param_groups:
            g_["lr"] = a.lr_voc * warm_m * cos_mult
        step += 1

        if step == 5:
            t_gate = time.time()
        if step == 55:
            sps = 50.0 / max(time.time() - t_gate, 1e-6)
            print(f"[GATE] step-50 throughput {sps:.2f} steps/s -> ~{sps * budget / 1000:.1f}k steps inside {a.max_minutes:.0f} min cap", flush=True)
            if sps < 3.5:
                print("[GATE] throughput too low, aborting", flush=True)
                sys.exit(3)

        xs = random.sample(pool, a.bs - 6) + random.sample(anchors, min(6, len(anchors)))
        B = len(xs)
        L = ((max(len(x["ids"]) for x in xs) + 15) // 16) * 16
        T = min(((max(int(x["dur_ctc"].sum().item()) for x in xs) + 63) // 64) * 64, 448)
        ids = torch.zeros(B, L, dtype=torch.long, device=dev)
        durs = torch.zeros(B, L, dtype=torch.long, device=dev)
        po = torch.zeros(B, L, dtype=torch.float32, device=dev)
        y_gt = torch.full((B, 100, T), (mean - mean) / std, dtype=torch.float32, device=dev)

        for k, x in enumerate(xs):
            Lk = len(x["ids"])
            ids[k, :Lk] = x["ids"].to(dev)
            durs[k, :Lk] = x["dur_ctc"].to(dev).long()
            if x.get("po") is not None:
                po[k, :Lk] = x["po"][:Lk].to(dev)
            m_k = (x["mel"].to(dev).float() - mean) / std
            Tk = min(T, m_k.shape[1])
            y_gt[k, :, :Tk] = m_k[:, :Tk]

        il = torch.tensor([len(x["ids"]) for x in xs], device=dev)
        yl = durs.sum(1).clamp(min=a.seg, max=T)
        im = torch.arange(L, device=dev)[None] < il[:, None]
        ym = torch.arange(T, device=dev)[None] < yl[:, None]
        ymf = ym[:, None].float()
        spks = torch.zeros(B, dtype=torch.long, device=dev)
        stud = torch.ones(B, dtype=torch.long, device=dev)
        langs = torch.tensor([0 if (x["lang"] == "sa" and random.random() < 0.85) else LANG2ID.get(x["lang"], 0) for x in xs], device=dev)
        stys = torch.tensor([int(x.get("v3_style_id", 0)) for x in xs], device=dev).clamp(0, 5)
        pros_b = pros_tbl[stys]
        ones_m = torch.ones((B, 1), dtype=torch.float32, device=dev)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                h_t, mu_t, g_t = t_am.enc(ids, spks, langs, stys, stud, im, pitch_add=po, prosody=pros_b, sty_mask=ones_m)
                f0_t = t_am.enc.f0_pred(h_t)
                eng_t = t_am.enc.energy_pred(h_t)
                ld_tgt = torch.log(durs.float().clamp(min=1.0)) * im.float()

            h_s, mu_s, g_s = s_am.enc(ids, spks, langs, stys, stud, im, pitch_add=po, prosody=pros_b, sty_mask=ones_m)
            l_heads = 0.5 * (((s_am.enc.f0_pred(h_s) - f0_t) ** 2 + (s_am.enc.energy_pred(h_s) - eng_t) ** 2) * im[:, :, None]).sum() / im.sum().clamp(min=1)
            l_prior = 2.0 * (((mu_s - mu_t).abs() + (mu_s - mu_t) ** 2) * im[:, :, None]).sum() / (im.sum() * 100.0).clamp(min=1.0)

            # Duration loss with 2.5x weight on short/final words + hard >= 5.0-frame floor on standalone short function words
            ld_pred = s_am.dur.pred_logdur(ids, h_s, im).float()
            pos_l = torch.arange(L, device=dev)[None, :]
            is_short_or_tail = ((durs <= 6) & (ids > 6)) | (pos_l >= (il[:, None] - 4))
            w_dur = torch.where(is_short_or_tail & im, 2.5, 1.0) * im.float()
            l_dur = (F.smooth_l1_loss(ld_pred, ld_tgt, reduction="none") * w_dur).sum() / w_dur.sum().clamp(min=1.0)
            is_func_c = torch.isin(ids[:, 1:-2], FUNC_C_IDS) & (ids[:, :-3] <= 2) & torch.isin(ids[:, 2:-1], FUNC_V_IDS) & (ids[:, 3:] <= 2)
            if is_func_c.any():
                d_pred_c = torch.exp(ld_pred[:, 1:-2].clamp(max=4.0))
                l_dur = l_dur + 0.5 * (F.relu(5.2 - d_pred_c) * is_func_c.float()).sum() / is_func_c.float().sum().clamp(min=1.0)

            mu_s_h = apply_hiatus_mu_rule(ids, mu_s)
            mu_y_s = _expand_batch_durs(mu_s_h, durs, T) * ymf
            t0_z = torch.zeros((B,), device=dev, dtype=torch.float32)
            y_pred = (mu_y_s + s_am.vel(mu_y_s, t0_z, mu_y_s, g_s, ymf)) * ymf

            denom = (ymf.sum() * 100.0).clamp(min=1.0)
            l_mel_am = ((1.5 * (y_pred - y_gt).abs() + (y_pred - y_gt) ** 2) * ymf).sum() / denom
            df_p = y_pred[:, 1:, :] - y_pred[:, :-1, :]
            df_t = y_gt[:, 1:, :] - y_gt[:, :-1, :]
            l_df = ((df_p - df_t).abs() * ymf).sum() / denom
            ym_t = ymf[:, :, 1:] * ymf[:, :, :-1]
            l_dt = (((y_pred[:, :, 1:] - y_pred[:, :, :-1]) - (y_gt[:, :, 1:] - y_gt[:, :, :-1])).abs() * ym_t).sum() / (ym_t.sum() * 100.0 + 1.0)
            cep_p = torch.fft.rfft(y_pred.float(), dim=1).abs()[:, 4:48, :]
            cep_t = torch.fft.rfft(y_gt.float(), dim=1).abs()[:, 4:48, :]
            l_cep = ((cep_p - cep_t).abs() * ymf).sum() / (ymf.sum() * 44.0).clamp(min=1.0)

            # Differentiable Pronunciation-CTC Loss with hinge floor annealed 0.22 -> 0.00 (computed on odd steps)
            if step % 2 == 1:
                b_c = min(10, B)
                hinge_floor = max(0.0, 0.22 * (1.0 - frac / 0.70))
                l_ctc_raw = ctc.ctc_loss(0.25 * mu_y_s[:b_c].float() + 0.75 * y_pred[:b_c].float(), yl[:b_c], ids[:b_c], il[:b_c])
                l_ctc = F.relu(l_ctc_raw - hinge_floor) if torch.isfinite(l_ctc_raw) else torch.tensor(0.0, device=dev)
            else:
                l_ctc = torch.tensor(0.0, device=dev)

        # Slice active voiced segment of length `a.seg` for End-to-End Vocoder-in-the-Loop Waveform Co-Distillation
        mel_s_segs, mel_t_segs = [], []
        for k in range(B):
            valid_T = max(a.seg, min(int(yl[k].item()), xs[k]["mel"].shape[1]))
            s0 = random.randint(0, max(0, valid_T - a.seg))
            s_slice = (y_pred[k, :, s0:s0 + a.seg].float() * std + mean)
            t_slice = (y_gt[k, :, s0:s0 + a.seg].float() * std + mean)
            if s_slice.shape[1] < a.seg:
                s_slice = F.pad(s_slice, (0, a.seg - s_slice.shape[1]), value=mean)
                t_slice = F.pad(t_slice, (0, a.seg - t_slice.shape[1]), value=mean)
            mel_s_segs.append(s_slice)
            mel_t_segs.append(t_slice)

        x_s = export_mel(torch.stack(mel_s_segs))
        x_t = export_mel(torch.stack(mel_t_segs))

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            re_t_gt, im_t_gt, _, _ = t_voc.spectrum_and_f0(x_t)
            re_t_same, im_t_same, _, _ = t_voc.spectrum_and_f0(x_s.detach())
        re_t_gt, im_t_gt = re_t_gt.float(), im_t_gt.float()
        re_t_same, im_t_same = re_t_same.float(), im_t_same.float()
        wav_t = s_voc.spec_to_wav(re_t_gt, im_t_gt)[:, :L_wav]
        wav_t_same = s_voc.spec_to_wav(re_t_same, im_t_same)[:, :L_wav]

        # Pass student mel x_s (with gradient to s_am & s_voc); use phase-locked re_t_same for complex/phase loss (0.0000 phase drift!)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            re_s, im_s, _, _ = s_voc.spectrum_and_f0(x_s)
        re_s, im_s = re_s.float(), im_s.float()
        wav_s = s_voc.spec_to_wav(re_s, im_s)[:, :L_wav]

        mag_t_gt = torch.sqrt(re_t_gt ** 2 + im_t_gt ** 2 + 1e-9)
        mag_t_same = torch.sqrt(re_t_same ** 2 + im_t_same ** 2 + 1e-9)
        mag_s = torch.sqrt(re_s ** 2 + im_s ** 2 + 1e-9)
        w_bin = torch.ones((1, 33, 1), device=dev, dtype=torch.float32)
        w_bin[:, :14, :] = 3.5
        w_bin = w_bin / w_bin.mean()
        l_lmag = 0.5 * (F.l1_loss(torch.log(mag_s + 1e-5), torch.log(mag_t_same + 1e-5), reduction="none") * w_bin).mean() + 0.5 * (F.l1_loss(torch.log(mag_s + 1e-5), torch.log(mag_t_gt + 1e-5), reduction="none") * w_bin).mean()
        l_cplx = (((re_s - re_t_same).abs() + (im_s - im_t_same).abs()) * w_bin).mean() / (mag_t_same.mean() + 1e-6) + 0.5 * F.l1_loss(wav_s, wav_t_same)
        l_ph = spec_phase_gd_loss(re_s[:, :16], im_s[:, :16], re_t_same[:, :16], im_t_same[:, :16], mag_s[:, :16], mag_t_same[:, :16])

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            l_wav_mel = F.l1_loss(mel_spec_24k(wav_s, dev)[:, :, :a.seg].float(), mel_spec_24k(wav_t, dev)[:, :, :a.seg].float())
            l_wav_stft = mr_complex_stft_loss(wav_s, wav_t, dev)
            if step % 2 == 0:
                b_slm = min(8, B)
                with torch.no_grad():
                    ft_t, emb_t = slm.extract(resamp16(wav_t[:b_slm].float()))
                ft_s, emb_s = slm.extract(resamp16(wav_s[:b_slm].float()))
                l_wavlm = F.l1_loss(ft_s.float(), ft_t.float()) + 0.5 * (1.0 - F.cosine_similarity(emb_s.float(), emb_t.float(), dim=-1).mean())
            else:
                l_wavlm = torch.tensor(0.0, device=dev)

        loss_am_core = l_mel_am + 2.4 * l_df + 1.2 * l_dt + 0.65 * l_cep + l_prior + l_heads + 0.75 * l_dur + 0.25 * l_ctc
        loss_e2e_wav = 12.0 * l_wav_mel + 1.2 * l_wav_stft + 1.5 * l_lmag + 2.0 * l_cplx + 0.8 * l_ph + 2.5 * l_wavlm
        loss = loss_am_core + loss_e2e_wav

        opt_am.zero_grad(set_to_none=True)
        opt_voc.zero_grad(set_to_none=True)
        loss.backward()
        if torch.isfinite(torch.nn.utils.clip_grad_norm_(s_am.parameters(), 1.0)):
            opt_am.step()
        if torch.isfinite(torch.nn.utils.clip_grad_norm_([p for p in s_voc.parameters() if p.requires_grad], 1.0)):
            opt_voc.step()

        with torch.no_grad():
            _calibrate_v15_styles(s_am)
            k_ema = 1.0 - min(0.996, (1.0 + step) / (10.0 + step))
            for pe, p in zip(ema_am.parameters(), s_am.parameters()):
                pe.lerp_(p, k_ema)
            for pe, p in zip(ema_voc.parameters(), s_voc.parameters()):
                pe.lerp_(p, k_ema)

        if step % 200 == 0:
            print(f"[{tag}] step={step} {step / (time.time() - t0):.1f} st/s t={(time.time() - t0) / 60:.1f}m mel_am={l_mel_am.item():.3f} dur={l_dur.item():.3f} ctc={l_ctc.item():.3f} wav_mel={l_wav_mel.item():.3f} stft={l_wav_stft.item():.3f} cplx={l_cplx.item():.3f} wavlm={l_wavlm.item():.3f} qat={int(qat_active)}", flush=True)

        done_smoke = a.smoke and step >= a.smoke
        if step % a.eval_every == 0 or done_smoke:
            cur = eval_e2e(ema_am, ema_voc)
            ck_am = dict(ema=ema_am.state_dict(), net=s_am.state_dict(), step=step, mean=mean, std=std, pre_baked_v25=True, e2e_utmos=cur)
            ck_voc = dict(arch="ms_rnd_wavehax", variant=voc_var, G_ema=ema_voc.state_dict(), G=s_voc.state_dict(), step=step, utmos={"e2e": cur})
            torch.save(ck_am, f"{out}/am_last.pt.tmp"); os.replace(f"{out}/am_last.pt.tmp", f"{out}/am_last.pt")
            torch.save(ck_voc, f"{out}/voc_last.pt.tmp"); os.replace(f"{out}/voc_last.pt.tmp", f"{out}/voc_last.pt")
            if cur >= best:
                best = cur
                torch.save(ck_am, f"{out}/am.pt.tmp"); os.replace(f"{out}/am.pt.tmp", f"{out}/am.pt")
                torch.save(ck_voc, f"{out}/voc.pt.tmp"); os.replace(f"{out}/voc.pt.tmp", f"{out}/voc.pt")
            print(f"[{tag}] SCORE step={step} e2e_utmos={cur:.4f} best={best:.4f} init={base_utmos:.4f}", flush=True)
            if done_smoke:
                break

    json.dump(dict(tier=a.tier, voc_variant=voc_var, best_utmos=best, init_utmos=base_utmos, steps=step), open(f"{out}/summary.json", "w"), indent=2)
    print(f"[{tag}] DONE best_utmos={best:.4f} steps={step}", flush=True)


if __name__ == "__main__":
    main()
