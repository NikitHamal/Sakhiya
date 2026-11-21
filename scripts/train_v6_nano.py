#!/usr/bin/env python3
"""Phase 3 (`v6_nano`): True Sub-3 MB `Sakhiya-Nano` via `sanoTTS` + `Paradee` Sequence-Level Teacher Distillation.
Enforces:
- Step 3A: Spotless 12-15 Hour (~11,169 utterances) single-voice Teacher Distillation Corpus (`phonemes, teacher_dur, teacher_f0, teacher_energy, teacher_mel, teacher_wav_24k`).
- Step 3B: Train `Sakhiya-Nano` (`sakhiya/v6_nano.py`, ~1.88M params: ~425k Looped Weight-Tied Encoder + ~1.45M Decoder-Heavy Harmonic-iSTFT) with strict `--max-minutes 120` cap and Mandatory Step-50 Throughput Gate.
- Step 3C: `sanoTTS`-Style Quantization Fidelity Gate (`corr >= 0.98` between INT8 and FP32/FP16 renders, `size <= 3.0 MB`, zero `ConvInteger`)."""
import argparse
import copy
import json
import math
import os
import random
import sys
import time
import torch
import torch.nn.functional as F

R = "/work/tts"
REPO = "/work/tts/repo" if os.path.exists("/work/tts/repo") else os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from sakhiya.cfm import SakhiyaAM, STYLE_PROSODY_TARGETS
from sakhiya.ctc_aligner import MelCTCAligner
from sakhiya.symbols import LANG2ID
from sakhiya.v2_vocoder import VocoderV2, ola64
from sakhiya.v6_distill import build_teacher_anchor_bank
from sakhiya.v6_istft_tts import compute_phrase_context_torch, function_word_mask_torch
from sakhiya.v6_nano import SakhiyaV6Nano, export_v6_nano_onnx
from scripts.rsi_v5_daemon import update_rsi_state
from scripts.train_v2_voc import mr_complex_stft_loss, phase_loss
from scripts.train_v6_teacher import extract_micro_prosody_contour, vectorized_expand_durs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=35.0, help="Hard wall-clock cap (MUST be <= 120 min)")
    ap.add_argument("--am-steps", type=int, default=2200)
    ap.add_argument("--voc-steps", type=int, default=1200)
    ap.add_argument("--bs", type=int, default=40)
    ap.add_argument("--out", default=f"{R}/runs/v6_nano")
    ap.add_argument("--export-dir", default=f"{REPO}/docs/models_nano")
    ap.add_argument("--teacher-ckpt", default=f"{R}/runs/v6_istft_teacher/teacher.pt")
    ap.add_argument("--fallback-am", default=f"{R}/runs/v5_c1/am.pt")
    ap.add_argument("--fallback-voc", default=f"{R}/runs/v5_c1/voc.pt")
    a = ap.parse_args()

    assert a.max_minutes <= 120.0, f"HARD 120-MIN WALL-CLOCK CAP VIOLATION: --max-minutes {a.max_minutes} > 120!"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    t_wall0 = time.time()
    max_sec = a.max_minutes * 60.0

    print(f"[v6_nano] Step 3A: Loading 12-15h single-voice Teacher Distillation Corpus into RAM/VRAM...", flush=True)
    D = torch.load(f"{R}/feats/v6_expressive_vedic_24k.pt", weights_only=False)
    mean, std = float(D.get("mean", -5.5028)), float(D.get("std", 2.1869))

    ctc = MelCTCAligner().to(dev).train()
    ck_c = torch.load(f"{R}/feats/v31_ctc_recognizer.pt", map_location=dev, weights_only=False)
    ctc.load_state_dict(ck_c.get("model", ck_c))
    for p in ctc.parameters():
        p.requires_grad_(False)

    t_am = SakhiyaAM().to(dev).eval()
    ck_a = torch.load(a.fallback_am, map_location=dev, weights_only=False)
    t_am.load_state_dict({k: v.float() for k, v in ck_a.get("ema", ck_a.get("net", ck_a)).items() if not k.startswith("ctc_rec.")}, strict=False)
    t_am.stats.copy_(torch.tensor([mean, std], device=dev))
    from scripts.train_nano_atom import _bake
    _bake(t_am)

    t_voc = VocoderV2().to(dev).eval()
    ck_v = torch.load(a.fallback_voc, map_location=dev, weights_only=False)
    t_vsd = ck_v.get("G_ema", ck_v.get("G", ck_v))
    t_voc.load_state_dict({k: v.float() for k, v in t_vsd.items()}, strict=False)

    anchors = build_teacher_anchor_bank(t_am, mean, std, ctc, dev=dev)
    base_tr = [x for x in D["items"] if x.get("split") == "train" and 3 <= len(x["ids"]) <= 160 and 24 <= x["mel"].shape[1] <= 750]
    exp_tr = [x for x in base_tr if x.get("v3_style_id", 0) in (2, 5)]
    pool = base_tr + exp_tr * 2 + anchors * 35
    total_hours = sum(int(x["mel"].shape[1]) * 256 / 24000.0 for x in base_tr) / 3600.0
    print(f"[v6_nano] Step 3A Complete: {len(base_tr)} clean single-voice utterances ({total_hours:.2f} hours) + {len(anchors)} multi-style anchors in RAM!", flush=True)

    nano = SakhiyaV6Nano().to(dev)
    nano.stats.copy_(torch.tensor([mean, std], device=dev))
    t_sd = torch.load(a.teacher_ckpt, map_location=dev, weights_only=False)["ema"] if os.path.exists(a.teacher_ckpt) else t_am.state_dict()
    nano.warm_from_teacher(t_sd, t_vsd)
    # Also warm-load pre-distilled student vocoder if available
    for vp_cand in (f"{R}/runs/v7_sano_voc/voc.pt", f"{R}/runs/v6_sano_voclong/voc.pt", f"{R}/runs/v6_voca_voclong/voc.pt"):
        if os.path.exists(vp_cand):
            ck_sv = torch.load(vp_cand, map_location=dev, weights_only=False)
            svd = ck_sv.get("G_ema", ck_sv.get("G", ck_sv))
            with torch.no_grad():
                for k, p in nano.voc.state_dict().items():
                    if k in svd and torch.is_floating_point(p):
                        s = svd[k].float().to(p.device)
                        if s.shape == p.shape:
                            p.copy_(s)
                        elif s.ndim == p.ndim and all(x1 >= x2 for x1, x2 in zip(s.shape, p.shape)):
                            sl = tuple(slice(0, b) for b in p.shape)
                            p.copy_(s[sl])
            break
    nano.voc.src.f0_net.load_state_dict(t_voc.src.f0_net.state_dict())
    for p in nano.voc.src.f0_net.parameters():
        p.requires_grad_(False)

    ema = copy.deepcopy(nano).eval()
    opt = torch.optim.AdamW([p for p in nano.parameters() if p.requires_grad], lr=7.0e-4, betas=(0.9, 0.99), weight_decay=1e-2)
    pros_tbl = torch.tensor(STYLE_PROSODY_TARGETS, dtype=torch.float32, device=dev)
    c_idx = torch.arange(100, device=dev, dtype=torch.float32)[None, :, None]

    t_step0 = time.time()
    for step in range(1, a.am_steps + 1):
        if (time.time() - t_wall0) >= max_sec * 0.70:
            print(f"[v6_nano] Reached AM time budget at step {step}; proceeding to decoder iSTFT polish.", flush=True)
            break

        if step == 50:
            dt50 = max(time.time() - t_step0, 1e-3)
            sps = 50.0 / dt50
            eta_min = ((a.am_steps + a.voc_steps) - 50) / sps / 60.0
            print(f"[STEP-50 THROUGHPUT GATE (Nano)] sps={sps:.2f} steps/sec | projected ETA={eta_min:.1f} min (limit <= 120.0 min)", flush=True)
            if eta_min > 120.0:
                raise RuntimeError(f"ABORT: Nano Step-50 ETA {eta_min:.1f} min > 120 min cap!")

        xs = random.sample(pool, a.bs - 10) + random.sample(anchors, min(10, len(anchors)))
        B = len(xs)
        L = max(len(x["ids"]) for x in xs)
        T = min(max(int(x["dur_ctc"].sum().item()) for x in xs), 750)
        ids = torch.zeros(B, L, dtype=torch.long, device=dev)
        durs = torch.zeros(B, L, dtype=torch.long, device=dev)
        po = torch.zeros(B, L, dtype=torch.float32, device=dev)
        for k, x in enumerate(xs):
            Lk = len(x["ids"])
            ids[k, :Lk] = x["ids"].to(dev)
            durs[k, :Lk] = x["dur_ctc"].to(dev).long()
            if "po" in x and x["po"] is not None:
                po[k, :Lk] = x["po"][:Lk].to(dev)
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

        lr = max(7.0e-4 * 0.5 * (1.0 + math.cos(math.pi * step / a.am_steps)), 5.0e-5)
        for g in opt.param_groups:
            g["lr"] = lr

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                h_t, mu_t, g_t = t_am.enc(ids, spks, langs, stys, stud, im, pitch_add=po, prosody=pros_b, sty_mask=ones_m)
                f0_t = t_am.enc.f0_pred(h_t).squeeze(-1)
                eng_t = t_am.enc.energy_pred(h_t).squeeze(-1)
                mu_y_t = vectorized_expand_durs(mu_t, durs, T) * ymf
                t_idx = torch.arange(T, device=dev, dtype=torch.float32)[None, None, :]
                xt = (torch.sin(c_idx * 12.9898 + t_idx * 78.233) * torch.cos(c_idx * 39.346 + t_idx * 11.135) * 0.7071).expand(B, 100, T).to(mu_y_t.dtype)
                for i_o in range(3):
                    to = torch.full((B,), i_o / 3.0, device=dev)
                    k1 = t_am.vel(xt, to, mu_y_t, g_t, ymf)
                    xt = xt + (1.0 / 3.0) * t_am.vel(xt + (1.0 / 6.0) * k1, to + (1.0 / 6.0), mu_y_t, g_t, ymf)
                y_clean = xt * ymf
                contour_gt = extract_micro_prosody_contour(y_clean, durs, L)

            phrase_ctx = compute_phrase_context_torch(ids, stys)
            h_s, mu_s, g_s = nano.enc(ids, spks, langs, stys, stud, im, pitch_add=po, prosody=pros_b, sty_mask=ones_m)
            h_s, vuv_s = nano.adapter(h_s, phrase_ctx, contour=contour_gt, drop_prob=0.50)

            f0_s = nano.enc.f0_pred(h_s).squeeze(-1)
            eng_s = nano.enc.energy_pred(h_s).squeeze(-1)
            l_heads = 0.5 * (((f0_s - f0_t) ** 2 + (eng_s - eng_t) ** 2) * im.float()).sum() / im.sum().clamp(min=1)

            fn_mask = function_word_mask_torch(ids)
            d_tgt_fl = torch.maximum(durs.float().clamp(min=1.0), fn_mask * 5.5)
            ld_tgt = torch.log(d_tgt_fl) * im.float()
            ld_pred = nano.dur.pred_logdur(ids, h_s, im)
            pos_l = torch.arange(L, device=dev)[None, :]
            tail_w = torch.where((pos_l >= (il[:, None] - 3)) & im, torch.full_like(ld_tgt, 2.0), torch.ones_like(ld_tgt))
            l_dur = (F.smooth_l1_loss(ld_pred * im.float(), ld_tgt, reduction="none") * tail_w).sum() / im.sum().clamp(min=1)

            mu_y_s = vectorized_expand_durs(mu_s, durs, T) * ymf
            l_prior = 2.2 * (((mu_s - mu_t).abs() + (mu_s - mu_t) ** 2) * im[:, :, None]).sum() / (im.sum() * 100.0)
            t0_z = torch.zeros((B,), device=dev, dtype=torch.float32)
            y_pred = (mu_y_s + nano.vel(mu_y_s, t0_z, mu_y_s, g_s, ymf)) * ymf
            denom = (ymf.sum() * 100.0).clamp(min=1.0)
            l_mel = ((1.6 * (y_pred - y_clean).abs() + (y_pred - y_clean) ** 2) * ymf).sum() / denom
            df_p = y_pred[:, 1:, :] - y_pred[:, :-1, :]
            df_t = y_clean[:, 1:, :] - y_clean[:, :-1, :]
            l_df = ((df_p - df_t).abs() * ymf).sum() / denom
            cep_p = torch.fft.rfft(y_pred.float(), dim=1).abs()[:, 4:48, :]
            cep_t = torch.fft.rfft(y_clean.float(), dim=1).abs()[:, 4:48, :]
            l_cep = ((cep_p - cep_t).abs() * ymf).sum() / (ymf.sum() * 44.0).clamp(min=1.0)

            l_ctc = torch.tensor(0.0, device=dev)
            if step % 2 == 0:
                l_ctc = F.relu(ctc.ctc_loss(0.3 * mu_y_s + 0.7 * y_pred, yl, ids, il) - 0.05)

            loss = l_mel + 2.4 * l_df + 0.65 * l_cep + l_prior + l_heads + 0.7 * l_dur + 0.25 * l_ctc

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if torch.isfinite(torch.nn.utils.clip_grad_norm_(nano.parameters(), 1.0)):
            opt.step()
        with torch.no_grad():
            d_ema = min(0.996, (1 + step) / (10 + step))
            for pe, p in zip(ema.parameters(), nano.parameters()):
                pe.lerp_(p, 1.0 - d_ema)

        if step % 150 == 0 or step == a.am_steps:
            dt = max(time.time() - t_step0, 1e-3)
            sps = step / dt
            eta_min = round(((a.am_steps + a.voc_steps) - step) / sps / 60.0, 2)
            rem_cap_min = round((max_sec - (time.time() - t_wall0)) / 60.0, 2)
            p_info = dict(step=step, sps=round(sps, 2), eta_min=eta_min, rem_cap_min=rem_cap_min, mel_l1=round(float(l_mel.item()), 4), dur_l1=round(float(l_dur.item()), 4))
            json.dump(p_info, open(f"{a.out}/progress.json", "w"), indent=2)
            if step % 1500 == 0:
                torch.save(dict(ema=ema.state_dict(), net=nano.state_dict(), cfg=nano.cfg_dict, mean=mean, std=std, step=step), f"{a.out}/nano.pt.tmp")
                os.replace(f"{a.out}/nano.pt.tmp", f"{a.out}/nano.pt")
            update_rsi_state(2, "v6_nano", f"Phase 3 (v6_nano AM) step {step}/{a.am_steps} ({sps:.1f} step/s, ETA {eta_min:.1f}m) | mel={l_mel.item():.3f} dur={l_dur.item():.3f}")
            print(f"[v6_nano-AM] step={step}/{a.am_steps} sps={sps:.2f} eta_min={eta_min:.1f} mel={l_mel.item():.4f} dur={l_dur.item():.4f} ctc={l_ctc.item():.4f}", flush=True)

    # Decoder-Heavy Harmonic-iSTFT distillation stage
    v_opt = torch.optim.AdamW([p for p in nano.voc.parameters() if p.requires_grad], lr=5.0e-4, betas=(0.8, 0.99), weight_decay=1e-4)
    seg_f = 64
    t_voc0 = time.time()
    for step in range(1, a.voc_steps + 1):
        if (time.time() - t_wall0) >= max_sec:
            break
        b_xs = [x for x in random.sample(pool, 64) if x["mel"].shape[1] >= seg_f][:32]
        mels = torch.stack([x["mel"][:, s:(s + seg_f)] for x in b_xs for s in [random.randint(0, x["mel"].shape[1] - seg_f)]]).to(dev).float()
        with torch.no_grad():
            re_t, im_t = t_voc.spectrum(mels)
            wav_t = ola64(torch.complex(re_t.float(), im_t.float()), t_voc.win)
            mag_t = torch.sqrt(re_t.float() ** 2 + im_t.float() ** 2 + 1e-7)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            re_s, im_s = nano.voc.spectrum(mels)
            wav_s = ola64(torch.complex(re_s.float(), im_s.float()), nano.voc.win)
            mag_s = torch.sqrt(re_s.float() ** 2 + im_s.float() ** 2 + 1e-7)
            l_stft = mr_complex_stft_loss(wav_s, wav_t, dev) + 0.6 * phase_loss(wav_s, wav_t, 512, dev)
            l_mag = F.l1_loss(mag_s, mag_t) + 0.5 * F.l1_loss(torch.log(mag_s + 1e-5), torch.log(mag_t + 1e-5))
            v_loss = 2.0 * l_stft + 4.0 * l_mag
        v_opt.zero_grad(set_to_none=True)
        v_loss.backward()
        if torch.isfinite(torch.nn.utils.clip_grad_norm_(nano.voc.parameters(), 1.0)):
            v_opt.step()
        with torch.no_grad():
            for pe, p in zip(ema.voc.parameters(), nano.voc.parameters()):
                pe.lerp_(p, 0.015)
        if step % 200 == 0 or step == a.voc_steps:
            sps_v = step / max(time.time() - t_voc0, 1e-3)
            print(f"[v6_nano-iSTFT] step={step}/{a.voc_steps} sps={sps_v:.2f} stft={l_stft.item():.4f} mag={l_mag.item():.4f}", flush=True)

    ck_path = f"{a.out}/nano.pt"
    torch.save(dict(ema=ema.state_dict(), net=nano.state_dict(), cfg=nano.cfg_dict, mean=mean, std=std), ck_path)
    sz = export_v6_nano_onnx(ema.cpu(), a.export_dir, max_fp16_mb=4.5, max_int8_mb=3.0)
    print("V6_NANO_EXPORT_OK", json.dumps(sz), flush=True)


if __name__ == "__main__":
    main()
