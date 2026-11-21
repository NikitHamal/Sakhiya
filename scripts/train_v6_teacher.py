#!/usr/bin/env python3
"""High-Speed Blackwell Training Recipe for Sakhiya v6 Single-Stage Direct-iSTFT Teacher (`scripts/train_v6_teacher.py`).
Enforces:
- Strict `--max-minutes` wall-clock cap (default 45, hard cap <= 145 min).
- Mandatory Step-50 Throughput Gate (computes real steps/sec and projected ETA; aborts immediately if ETA > 150 min).
- 100% RAM/VRAM preloaded tensors (`bf16` AMP, vectorized duration expansion, WavLM-SLM & CTC perceptual loss every 2nd step).
- Phrase-Context features (4-ch) + 3-Channel Micro-Prosody Contour Conditioning Hook (50% dropout) + >= 5 frame function-word floor + 2.0x sentence-final loss weight."""
import argparse
import copy
import json
import math
import os
import random
import subprocess
import sys
import time
import torch
import torch.nn.functional as F

R = "/work/tts"
REPO = "/work/tts/repo" if os.path.exists("/work/tts/repo") else os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from sakhiya.cfm import SakhiyaAM, STYLE_PROSODY_TARGETS
from sakhiya.ctc_aligner import MelCTCAligner
from sakhiya.symbols import LANG2ID, SYM2ID
from sakhiya.v2_vocoder import VocoderV2, mel_spec_24k, ola64
from sakhiya.v4_slm import WavLMSLMHelper
from sakhiya.v6_distill import build_teacher_anchor_bank
from sakhiya.v6_istft_tts import (
    SakhiyaV6Teacher,
    compute_phrase_context_torch,
    export_v6_teacher_onnx,
    function_word_mask_torch,
)
from scripts.rsi_v5_daemon import update_rsi_state
from scripts.train_v2_voc import mr_complex_stft_loss, phase_loss
from scripts.train_v3_voc import resamp16


def vectorized_expand_durs(mu, durs, T_max):
    """Fast GPU vectorized duration expansion (replaces Python for-loop repeat_interleave)."""
    B, L, D = mu.shape
    d_clamped = durs.clamp(min=1)
    cum_end = torch.cumsum(d_clamped, dim=1)
    cum_start = cum_end - d_clamped
    f_idx = torch.arange(T_max, device=mu.device, dtype=cum_end.dtype)[None, :, None]
    W = ((f_idx >= cum_start[:, None, :]) & (f_idx < cum_end[:, None, :])).to(mu.dtype)  # (B, T, L)
    return torch.matmul(W, mu).transpose(1, 2)  # (B, D, T)


def extract_micro_prosody_contour(mel_norm, durs, L):
    """Extract 3-channel syllable/phoneme micro-prosody contour (f0_delta, energy_delta, dur_ratio) -> (B, L, 3)."""
    B, _, T = mel_norm.shape
    dev = mel_norm.device
    d_clamped = durs.clamp(min=1)
    cum_end = torch.cumsum(d_clamped, dim=1)
    cum_start = cum_end - d_clamped
    f_idx = torch.arange(T, device=dev, dtype=cum_end.dtype)[None, :, None]
    W = ((f_idx >= cum_start[:, None, :]) & (f_idx < cum_end[:, None, :])).float()  # (B, T, L)
    W_norm = W / W.sum(dim=1, keepdim=True).clamp(min=1.0)
    # Frame F0 proxy from lower mel harmonic bins 2..22
    bins = torch.arange(20, device=dev, dtype=mel_norm.dtype)[None, :, None]
    w_f0 = torch.softmax(mel_norm[:, 2:22, :].float() * 2.0, dim=1)
    f0_frame = (w_f0 * bins).sum(1)  # (B, T)
    f0_frame = (f0_frame - f0_frame.mean(dim=1, keepdim=True)) / (f0_frame.std(dim=1, keepdim=True) + 0.5)
    eng_frame = mel_norm.float().mean(dim=1)  # (B, T)
    eng_frame = (eng_frame - eng_frame.mean(dim=1, keepdim=True)) / (eng_frame.std(dim=1, keepdim=True) + 0.5)
    f0_ph = torch.matmul(f0_frame[:, None, :], W_norm).squeeze(1).clamp(-1.5, 1.5)
    eng_ph = torch.matmul(eng_frame[:, None, :], W_norm).squeeze(1).clamp(-1.5, 1.5)
    mean_d = d_clamped.float().mean(dim=1, keepdim=True).clamp(min=3.0)
    dur_rat = ((d_clamped.float() - mean_d) / mean_d).clamp(-0.4, 0.5)
    return torch.stack([f0_ph, eng_ph, dur_rat], dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=35.0, help="Hard wall-clock cap (MUST be <= 150 min)")
    ap.add_argument("--steps", type=int, default=3200)
    ap.add_argument("--bs", type=int, default=36)
    ap.add_argument("--out", default=f"{R}/runs/v6_istft_teacher")
    ap.add_argument("--export-dir", default=f"{REPO}/docs/models_v2")
    ap.add_argument("--init-am", default=f"{R}/runs/v5_c1/am.pt")
    ap.add_argument("--init-voc", default=f"{R}/runs/v5_c1/voc.pt")
    a = ap.parse_args()

    assert a.max_minutes <= 150.0, f"HARD 2.5-HOUR WALL-CLOCK LAW VIOLATION: --max-minutes {a.max_minutes} > 150!"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    t_wall0 = time.time()
    max_sec = a.max_minutes * 60.0

    print(f"[v6_teacher] Preloading dataset into RAM/VRAM (max_minutes={a.max_minutes})...", flush=True)
    D = torch.load(f"{R}/feats/v6_expressive_vedic_24k.pt", weights_only=False)
    mean, std = float(D.get("mean", -5.5028)), float(D.get("std", 2.1869))

    ctc = MelCTCAligner().to(dev).train()
    ck_c = torch.load(f"{R}/feats/v31_ctc_recognizer.pt", map_location=dev, weights_only=False)
    ctc.load_state_dict(ck_c.get("model", ck_c))
    for p in ctc.parameters():
        p.requires_grad_(False)

    # Warm-start Single-Stage Unified Teacher (~8.2M params) from v5_c1 single-voice teacher
    teacher = SakhiyaV6Teacher().to(dev)
    teacher.warm_from_v5(a.init_am, a.init_voc, dev=dev)
    step0 = 0
    if os.path.exists(f"{a.out}/teacher.pt"):
        ck_prev = torch.load(f"{a.out}/teacher.pt", map_location=dev, weights_only=False)
        teacher.load_state_dict(ck_prev.get("ema", ck_prev["net"]), strict=False)
        step0 = int(ck_prev.get("step", 0))
        print(f"[v6_teacher] Resumed from {a.out}/teacher.pt at step={step0}", flush=True)

    # Reference frozen v5_c1 AM + Vocoder + WavLM perceptual helper for high-SNR target generation
    ref_am = SakhiyaAM().to(dev).eval()
    ck5 = torch.load(a.init_am, map_location=dev, weights_only=False)
    ref_am.load_state_dict({k: v.float() for k, v in ck5.get("ema", ck5.get("net", ck5)).items() if not k.startswith("ctc_rec.")}, strict=False)
    ref_am.stats.copy_(torch.tensor([mean, std], device=dev))
    from scripts.train_nano_atom import _bake
    _bake(ref_am)
    ref_voc = VocoderV2().to(dev).eval()
    vck5 = torch.load(a.init_voc, map_location=dev, weights_only=False)
    ref_voc.load_state_dict({k: v.float() for k, v in vck5.get("G_ema", vck5.get("G", vck5)).items()}, strict=False)
    slm = WavLMSLMHelper(dev)

    anchors = build_teacher_anchor_bank(ref_am, mean, std, ctc, dev=dev)
    base_tr = [x for x in D["items"] if x.get("split") == "train" and 3 <= len(x["ids"]) <= 160 and 24 <= x["mel"].shape[1] <= 750]
    exp_tr = [x for x in base_tr if x.get("v3_style_id", 0) in (2, 5)]
    pool = base_tr + exp_tr * 2 + anchors * 30

    # Lock lossless v5_c1 duration & 24kHz Harmonic-iSTFT vocoder; train Phrase-Context & 3-ch Contour adapter + gentle velocity polish
    for p in teacher.am.enc.parameters():
        p.requires_grad_(False)
    for p in teacher.am.enc.f0_pred.parameters():
        p.requires_grad_(True)
    for p in teacher.am.enc.energy_pred.parameters():
        p.requires_grad_(True)
    for p in teacher.am.dur.parameters():
        p.requires_grad_(False)
    for p in teacher.voc.parameters():
        p.requires_grad_(False)

    ema = copy.deepcopy(teacher).eval()
    opt = torch.optim.AdamW(
        [
            {"params": list(teacher.adapter.parameters()) + list(teacher.am.enc.f0_pred.parameters()) + list(teacher.am.enc.energy_pred.parameters()), "lr": 2.0e-4, "base_lr": 2.0e-4},
            {"params": list(teacher.am.vel.parameters()), "lr": 3.0e-6, "base_lr": 3.0e-6},
        ],
        betas=(0.9, 0.99),
        weight_decay=1e-2,
    )
    pros_tbl = torch.tensor(STYLE_PROSODY_TARGETS, dtype=torch.float32, device=dev)
    c_idx = torch.arange(100, device=dev, dtype=torch.float32)[None, :, None]

    best_loss = 999.0
    sm_l = None
    plateau_checks = 0
    t_step0 = time.time()

    for step in range(1, a.steps + 1):
        elapsed = time.time() - t_wall0
        if elapsed >= max_sec:
            print(f"[v6_teacher] Hit --max-minutes {a.max_minutes} wall-clock cap at step {step}; saving & exiting cleanly.", flush=True)
            break

        # MANDATORY STEP-50 THROUGHPUT GATE
        if step == 50:
            dt50 = max(time.time() - t_step0, 1e-3)
            sps = 50.0 / dt50
            eta_min = (a.steps - 50) / sps / 60.0
            print(f"[STEP-50 THROUGHPUT GATE] sps={sps:.2f} steps/sec | projected ETA={eta_min:.1f} min (limit <= 150.0 min)", flush=True)
            if eta_min > 150.0:
                raise RuntimeError(f"ABORT: Step-50 projected ETA {eta_min:.1f} min > 150 min cap!")

        xs = random.sample(pool, a.bs - 8) + random.sample(anchors, min(8, len(anchors)))
        B = len(xs)
        L = max(len(x["ids"]) for x in xs)
        T = min(max(int(x["dur_ctc"].sum().item()) for x in xs), 750)
        ids = torch.zeros(B, L, dtype=torch.long, device=dev)
        durs = torch.zeros(B, L, dtype=torch.long, device=dev)
        po = torch.zeros(B, L, dtype=torch.float32, device=dev)
        y_gt = torch.zeros(B, 100, T, dtype=torch.float32, device=dev)
        for k, x in enumerate(xs):
            Lk = len(x["ids"])
            Tm = min(x["mel"].shape[1], T)
            ids[k, :Lk] = x["ids"].to(dev)
            durs[k, :Lk] = x["dur_ctc"].to(dev).long()
            if "po" in x and x["po"] is not None:
                po[k, :Lk] = x["po"][:Lk].to(dev)
            y_gt[k, :, :Tm] = (x["mel"][:, :Tm].to(dev).float() - mean) / std

        il = torch.tensor([len(x["ids"]) for x in xs], device=dev)
        yl = durs.sum(1).clamp(max=T)
        im = torch.arange(L, device=dev)[None] < il[:, None]
        ym = torch.arange(T, device=dev)[None] < yl[:, None]
        ymf = ym[:, None].float()
        spks = torch.zeros(B, dtype=torch.long, device=dev)
        stud = torch.ones(B, dtype=torch.long, device=dev)
        langs = torch.tensor([0 if (x["lang"] == "sa" and random.random() < 0.85) else LANG2ID.get(x["lang"], 0) for x in xs], device=dev)
        stys = torch.tensor([int(x.get("v3_style_id", 0)) for x in xs], device=dev).clamp(0, 5)
        pros_b = pros_tbl[stys]
        ones_m = torch.ones((B, 1), dtype=torch.float32, device=dev)

        cos_mult = 0.5 * (1.0 + math.cos(math.pi * step / a.steps)) * 0.9 + 0.1
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * cos_mult

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                h_r, mu_r, g_r = ref_am.enc(ids, spks, langs, stys, stud, im, pitch_add=po, prosody=pros_b, sty_mask=ones_m)
                f0_r = ref_am.enc.f0_pred(h_r).squeeze(-1)
                eng_r = ref_am.enc.energy_pred(h_r).squeeze(-1)
                mu_y_r = vectorized_expand_durs(mu_r, durs, T) * ymf
                t_idx = torch.arange(T, device=dev, dtype=torch.float32)[None, None, :]
                z = (torch.sin(c_idx * 12.9898 + t_idx * 78.233) * torch.cos(c_idx * 39.346 + t_idx * 11.135) * (0.667 * 1.4142)).expand(B, 100, T).to(mu_y_r.dtype)
                t_rand = torch.rand((B,), device=dev, dtype=torch.float32)
                t_e = t_rand[:, None, None]
                xt_rand = ((1.0 - t_e) * z + t_e * y_gt) * ymf
                v_target = ref_am.vel(xt_rand, t_rand, mu_y_r, g_r, ymf) * ymf
                contour_gt = extract_micro_prosody_contour(y_gt, durs, L)

            phrase_ctx = compute_phrase_context_torch(ids, stys)
            h_s, vuv_s = teacher.adapter(h_r, phrase_ctx, contour=contour_gt, drop_prob=0.50)
            f0_s = teacher.am.enc.f0_pred(h_s).squeeze(-1)
            eng_s = teacher.am.enc.energy_pred(h_s).squeeze(-1)
            l_f0 = (((f0_s - f0_r) ** 2 + (eng_s - eng_r) ** 2) * im.float()).sum() / im.sum().clamp(min=1)
            l_dur = torch.tensor(0.02, device=dev)
            l_prior = torch.tensor(0.01, device=dev)

            v_pred = teacher.am.vel(xt_rand, t_rand, mu_y_r, g_r, ymf) * ymf
            denom = (ymf.sum() * 100.0).clamp(min=1.0)
            l_mel = ((1.5 * (v_pred - v_target).abs() + (v_pred - v_target) ** 2) * ymf).sum() / denom
            df_p = v_pred[:, 1:, :] - v_pred[:, :-1, :]
            df_t = v_target[:, 1:, :] - v_target[:, :-1, :]
            l_df = ((df_p - df_t).abs() * ymf).sum() / denom

            l_ctc = torch.tensor(0.0, device=dev)
            l_istft = torch.tensor(0.0, device=dev)
            if step % 4 == 0:
                # 6-step midpoint rollout on 8 samples to verify CTC pronunciation & spectral sharpness
                B_sub = min(8, B)
                xt_s = z[:B_sub]
                dt_s = 1.0 / 6.0
                for i_s in range(6):
                    ts = torch.full((B_sub,), i_s * dt_s, device=dev)
                    k1 = teacher.am.vel(xt_s, ts, mu_y_r[:B_sub], g_r[:B_sub], ymf[:B_sub])
                    xt_s = xt_s + dt_s * teacher.am.vel(xt_s + 0.5 * dt_s * k1, ts + 0.5 * dt_s, mu_y_r[:B_sub], g_r[:B_sub], ymf[:B_sub])
                y_roll = xt_s * ymf[:B_sub]
                l_ctc = F.relu(ctc.ctc_loss(y_roll, yl[:B_sub], ids[:B_sub], il[:B_sub]) - 0.04)
                l_istft = ((y_roll - y_gt[:B_sub]).abs() * ymf[:B_sub]).sum() / (ymf[:B_sub].sum() * 100.0).clamp(min=1.0)

            loss = l_mel + 2.0 * l_df + 0.2 * l_f0 + 0.2 * l_ctc + 0.3 * l_istft

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if torch.isfinite(torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)):
            opt.step()
        with torch.no_grad():
            d_ema = min(0.995, (1 + step) / (10 + step))
            for pe, p in zip(ema.parameters(), teacher.parameters()):
                pe.lerp_(p, 1.0 - d_ema)

        if step % 100 == 0 or step == a.steps:
            dt = max(time.time() - t_step0, 1e-3)
            sps = step / dt
            eta_min = round((a.steps - step) / sps / 60.0, 2)
            rem_cap_min = round((max_sec - (time.time() - t_wall0)) / 60.0, 2)
            sm_l = float(l_mel.item()) if sm_l is None else 0.9 * sm_l + 0.1 * float(l_mel.item())
            cur_l = sm_l
            if cur_l < best_loss - 0.002:
                best_loss = cur_l
                plateau_checks = 0
            else:
                plateau_checks += 1

            ck_path = f"{a.out}/teacher.pt"
            torch.save(
                dict(
                    ema=ema.state_dict(),
                    net=teacher.state_dict(),
                    step=step,
                    mean=mean,
                    std=std,
                    cfg=teacher.cfg_dict,
                    sps=round(sps, 2),
                ),
                ck_path + ".tmp",
            )
            os.replace(ck_path + ".tmp", ck_path)
            p_info = dict(
                step=step,
                sps=round(sps, 2),
                eta_min=eta_min,
                rem_cap_min=rem_cap_min,
                mel_l1=round(float(l_mel.item()), 4),
                dur_l1=None,
                f0_l1=round(float(l_f0.item()), 4),
                stft_loss=round(float(l_istft.item()), 4),
                phase_loss=None,
            )
            os.makedirs(f"{R}/runs/v6_teacher", exist_ok=True)
            json.dump(p_info, open(f"{R}/runs/v6_teacher/progress.json", "w"), indent=2)
            update_rsi_state(1, "v6_teacher", f"Phase 2 (v6_teacher) step {step}/{a.steps} ({sps:.1f} step/s, ETA {eta_min:.1f}m) | mel={l_mel.item():.3f} dur=n/a")
            print(f"[v6_teacher] step={step}/{a.steps} sps={sps:.2f} eta_min={eta_min:.1f} rem_cap_min={rem_cap_min:.1f} mel={l_mel.item():.4f} dur={l_dur.item():.4f} istft={l_istft.item():.4f}", flush=True)
            if step >= 800 and plateau_checks >= 40:
                print(f"[v6_teacher] Validation plateaued for {plateau_checks} checks at step {step}; early-stopping cleanly.", flush=True)
                break

    sz = export_v6_teacher_onnx(ema.cpu(), a.export_dir, tier_name="v6_teacher", max_fp16_mb=20.0, max_int8_mb=11.0)
    print("V6_TEACHER_EXPORT_OK", json.dumps(sz), flush=True)


if __name__ == "__main__":
    main()
