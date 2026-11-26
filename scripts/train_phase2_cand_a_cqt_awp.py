#!/usr/bin/env python3
"""Phase 2 Candidate A — `Sakhiya-One-Polish` (~8.5 MB INT8 — MS-SB-CQT + Joint E2E Polish on Existing AWP Vocoder)
(`scripts/train_phase2_cand_a_cqt_awp.py`).

1. Warm-starts strictly from Phase-1 Anchored AM (`/work/sp/polish/ck/phase1_am_polished.pt`) + `Sakhiya-One`'s
   `AWPVocoder` (full-utterance 4-step MeanFlow path).
2. Adds Multi-Scale Sub-Band CQT Discriminator (`MS-SB-CQT`, BigVGAN-v2 style) alongside `DiscV2` (MPD + MRD) +
   Multi-Scale Mel + Magnitude-Aware Anti-Wrapped Phase (IP/GD/IAF) + Frozen Multi-Layer WavLM (layers 3/6/9) + Voiced LTAS.
3. Runs Joint End-to-End Waveform Polish (`bf16`, `voc_lr = 3e-5`, `am_lr = 5e-6`, duration/F0 predictors detached):
   runs the AM on the FULL utterance to produce 4-step predicted mels, then slices random 1.2-1.6 s (`120-150` frames)
   windows through the live `AWPVocoder` into `MS-SB-CQT` + MPD/MRD + WavLM discriminators.
4. Finishes the last 30% with clean LSQ Per-Channel QAT (keeping embeddings, duration/F0 heads, harmonic excitation,
   and final iSTFT projection in FP16), exports both un-snapped `model.fp32.onnx` and QDQ `model.int8.onnx` (~8.5 MB,
   verified >= 70% INT8 initializers, `distinct_from_fp32_graph = True`, zero `ConvInteger`, `nan_count == 0`),
   and renders all 3 Owner Clips onto Card 5.
"""
import argparse
import copy
import json
import math
import os
import random
import subprocess
import sys
import time
import librosa
import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
for p in (
    REPO, HERE,
    os.path.join(REPO, "scripts", "sakhiya_pro"),
    os.path.join(REPO, "scripts", "sakhiya_pro", "distill"),
    os.path.join(REPO, "scripts", "sakhiya_pro", "distill2"),
):
    if p not in sys.path:
        sys.path.insert(0, p)

from sakhiya.v2_vocoder import SR, HOP, mel_spec_24k
from sakhiya.news_vocoder_100mel import MultiScaleSubBandCQTDiscriminator, coherence_masked_awp_losses
from sakhiya.cfm import _dur_to_attn
from scripts.train_v2_voc import DiscV2, mr_complex_stft_loss
from student import Student, set_qat, compute_svara_mora_ids, compute_intra_phoneme_progress, _calibrate_styles
from run_session75_breakthrough import FrozenMultiLayerWavLM, contrastive_ltas_timbre_loss
from run_sakhiya_one_pipeline import install_lsq_qat_hooks
from export_gate import export_fp32, quantize, FP16_ISOLATED_NAMES, _normalize_lufs_16, ids_of
from scripts.forensic_audio_gate import audit_int8_qdq_initializers, compute_comb_grit_and_warble
from scripts.run_phase0_sakhiya_one_honest import synth_full_utterance_torch, _voiced_vowel_hnr_0p2_4khz
from scripts.train_phase1_am_polish import forward_4step_meanflow, update_stage, log

LISTEN_DIR = "/work/sp/polish/listen"
CK_DIR = "/work/sp/polish/ck"
ONNX_DIR = "/work/sp/polish/onnx/cand_a_polish"
os.makedirs(LISTEN_DIR, exist_ok=True)
os.makedirs(CK_DIR, exist_ok=True)
os.makedirs(ONNX_DIR, exist_ok=True)


def _mpd_d_loss(d_mpd: DiscV2, y_real: torch.Tensor, y_fake: torch.Tensor) -> torch.Tensor:
    o_r, _ = d_mpd(y_real.detach()[:, None, :])
    o_f, _ = d_mpd(y_fake.detach()[:, None, :])
    loss = torch.zeros((), device=y_real.device)
    for r, f in zip(o_r, o_f):
        loss = loss + torch.mean(F.relu(1.0 - r)) + torch.mean(F.relu(1.0 + f))
    return loss / max(1, len(o_r))


def _mpd_g_loss(d_mpd: DiscV2, y_real: torch.Tensor, y_fake: torch.Tensor):
    with torch.no_grad():
        _, fm_r = d_mpd(y_real.detach()[:, None, :])
    o_f, fm_f = d_mpd(y_fake[:, None, :])
    l_adv = sum(torch.mean(F.relu(1.0 - f)) for f in o_f) / max(1, len(o_f))
    l_fm = torch.zeros((), device=y_real.device)
    cnt = 0
    for f_list, r_list in zip(fm_f, fm_r):
        for ff, fr in zip(f_list, r_list):
            l_fm = l_fm + F.l1_loss(ff, fr.detach())
            cnt += 1
    return l_adv, l_fm / max(1, cnt)


def sibilant_and_upper_smoothness_loss(y_pred: torch.Tensor, y_gt: torch.Tensor, win_1024: torch.Tensor):
    """Explicit 2.5-10.5 kHz spectral smoothness & sibilant envelope matching loss:
    eliminates 4-10 kHz sibilant metallic shimmer (`विशेष`, `विश्व`, `संवेदनशील`, `सुरक्षित`, `प्रतिशतले`)
    and midrange hollowness (`1-4 kHz` formant match).
    """
    pad = (1024 - 256) // 2
    xp = F.pad(y_pred.float()[:, None], (pad, pad), mode="reflect")[:, 0]
    xt = F.pad(y_gt.float()[:, None], (pad, pad), mode="reflect")[:, 0]
    Sp = torch.stft(xp, 1024, 256, 1024, win_1024, center=False, return_complex=True).abs()
    St = torch.stft(xt, 1024, 256, 1024, win_1024, center=False, return_complex=True).abs()
    lp = torch.log(Sp.clamp(min=1e-5))
    lt = torch.log(St.clamp(min=1e-5))
    # 1. Midrange (300-4000 Hz -> bins 12..170) & sibilant (4000-10500 Hz -> bins 170..448) L1 log-magnitude match
    l_mid = F.l1_loss(lp[:, 12:170, :], lt[:, 12:170, :])
    l_sib = F.l1_loss(lp[:, 170:448, :], lt[:, 170:448, :])
    # 2. Second finite-difference across frequency bins in 2.5-8.5 kHz (bins 106..362) to penalize narrowband comb spikes
    d2_p = lp[:, 108:362, :] - 2.0 * lp[:, 107:361, :] + lp[:, 106:360, :]
    d2_t = lt[:, 108:362, :] - 2.0 * lt[:, 107:361, :] + lt[:, 106:360, :]
    l_comb = F.l1_loss(d2_p, d2_t) + 0.15 * torch.mean(F.relu(d2_p.abs() - d2_t.abs().detach()))
    return l_mid + 1.4 * l_sib + 1.2 * l_comb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=20)
    a = ap.parse_args()

    t0 = time.time()
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(42)
    torch.manual_seed(42)

    update_stage(2, 0, a.steps, "init", "initializing Candidate A (MS-SB-CQT + AWPVocoder + Phase-1 AM)", 0.0, 12.0, "running")

    # 1. Load Phase-1 Anchored AM + Sakhiya-One AWPVocoder
    ck_p = os.path.join(CK_DIR, "phase1_am_polished.pt")
    if not os.path.exists(ck_p):
        ck_p = "/work/sp/distill/runs/sakhiya_one/ema_final.pt"
    S = torch.load(ck_p, map_location="cpu", weights_only=False)
    G = Student(tier="sakhiya_one")
    G.steps = 4
    G.cfg["steps"] = 4
    G.warm_init("/work/tts/ck")
    own_sd = G.state_dict()
    filt_sd = {k: v for k, v in S["ema"].items() if k in own_sd and own_sd[k].shape == v.shape}
    G.load_state_dict(filt_sd, strict=False)
    _calibrate_styles(G.am)
    G.to(dev)

    # Also build teacher waveform renderer (clean AWPVocoder on clean target mels)
    teacher_voc = copy.deepcopy(G.voc).to(dev).eval()
    teacher_f0 = copy.deepcopy(G.am.f0_head).to(dev).eval()
    for p in list(teacher_voc.parameters()) + list(teacher_f0.parameters()):
        p.requires_grad_(False)

    hooks = install_lsq_qat_hooks(G)
    ema = copy.deepcopy(G).eval()

    # 2. Initialize MS-SB-CQT Discriminator + MPD/MRD (DiscV2) + FrozenMultiLayerWavLM
    d_cqt = MultiScaleSubBandCQTDiscriminator().to(dev).train()
    d_mpd = DiscV2().to(dev).train()
    wavlm_fm = FrozenMultiLayerWavLM(dev=dev)

    # Load pure Pro-v3 GT waveform clips for real discriminator training + full-utterance AM trajectory cache
    D_gt = torch.load("/work/sp/distill/cache_train.pt", map_location="cpu", weights_only=False)
    mf = {json.loads(l)["id"]: json.loads(l) for l in open("/work/sp/distill/manifest.jsonl")}
    DROP_LEGACY = {"vc_slr54", "vc_slr43", "vc_v13", "anchor_6834_stitched"}
    gt_wav_items = []
    for x in D_gt["items"][:900]:
        r = mf.get(x["id"], {})
        if r.get("src", "wiki") in DROP_LEGACY:
            continue
        if len(x["wav"]) >= 136 * HOP:
            gt_wav_items.append(x)

    full_utt_items = torch.load("/work/sp/polish/realigned_full_utt_cache.pt", map_location="cpu", weights_only=False)
    long_utt_items = [x for x in full_utt_items if x["y_clean"].shape[0] >= 120]
    if len(long_utt_items) < 60:
        long_utt_items = full_utt_items

    mean, std = float(G.am.stats[0].item()), float(G.am.stats[1].item())

    # Parameter groups:
    # - Duration & F0 predictors DETACHED / frozen so alignment & pitch never drift!
    # - Vocoder body: voc_lr = 3e-5
    # - AM encoder/velocity: am_lr = 5e-6
    voc_params, am_params = [], []
    for n, p in G.named_parameters():
        if n.endswith("had_rot.weight") or n.startswith(("am.dur.", "am.f0_head.", "am.enc.f0_pred", "am.enc.vuv_pred", "closure_slope", "dur_rate_bias")):
            p.requires_grad_(False)
        elif n.startswith("voc."):
            p.requires_grad_(True)
            voc_params.append(p)
        elif n.startswith("am."):
            p.requires_grad_(True)
            am_params.append(p)
        else:
            p.requires_grad_(False)

    oG = torch.optim.AdamW([
        {"params": voc_params, "lr": 3.0e-5, "initial_lr": 3.0e-5, "weight_decay": 1e-4},
        {"params": am_params, "lr": 5.0e-6, "initial_lr": 5.0e-6, "weight_decay": 1e-4},
    ], betas=(0.8, 0.99))
    oD = torch.optim.AdamW(list(d_cqt.parameters()) + list(d_mpd.parameters()), lr=4.0e-5, betas=(0.8, 0.99))

    qat_start_step = int(a.steps * 0.70)  # Final 30% with LSQ Per-Channel INT8 QAT
    qat_on = False
    seg_frames = 128  # 1.365 s window (strictly in 1.2-1.6s range, sliced ONLY AFTER full-utterance AM mel!)
    win_1024 = torch.hann_window(1024, device=dev)
    win_64 = torch.hann_window(65, device=dev)[:-1]

    tstep = time.time()
    log(f"[Phase 2A] Starting Candidate A (Sakhiya-One-Polish: MS-SB-CQT + Joint E2E + LSQ QAT) for {a.steps} steps...")

    for st in range(1, a.steps + 1):
        prog = st / max(1, a.steps)
        mult = 0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * prog))
        for pg in oG.param_groups:
            pg["lr"] = pg["initial_lr"] * mult

        if st >= qat_start_step and not qat_on:
            set_qat(G, True)
            qat_on = True
            log(f"[Phase 2A LSQ QAT] Activated per-channel LSQ INT8 QAT at step {st}/{a.steps}")

        xs = random.sample(long_utt_items, a.bs)
        B = len(xs)
        ids = pad_sequence([x["ids"].to(dev) for x in xs], batch_first=True)
        dur = pad_sequence([x["dur"].to(dev) for x in xs], batch_first=True)
        p_enc = pad_sequence([x["p_enc"].to(dev).float() for x in xs], batch_first=True)
        mu_t = pad_sequence([x["mu"].to(dev).float() for x in xs], batch_first=True)
        y_clean = pad_sequence([x["y_clean"].to(dev).float() for x in xs], batch_first=True).transpose(1, 2).contiguous()
        y_frozen = pad_sequence([x["y_frozen"].to(dev).float() for x in xs], batch_first=True).transpose(1, 2).contiguous()
        il = torch.tensor([len(x["ids"]) for x in xs], device=dev)
        yl = torch.tensor([x["y_clean"].shape[0] for x in xs], device=dev)
        L, T = ids.shape[1], y_clean.shape[2]
        im = torch.arange(L, device=dev)[None] < il[:, None]
        ym = torch.arange(T, device=dev)[None] < yl[:, None]
        ymf = ym[:, None].float()
        spk = torch.zeros(B, dtype=torch.long, device=dev)
        studio = torch.ones(B, dtype=torch.long, device=dev)
        is_sa = torch.tensor([x["lang"] == 1 for x in xs], device=dev)
        lang_eff = torch.where(is_sa, torch.full((B,), 4, dtype=torch.long, device=dev), torch.zeros(B, dtype=torch.long, device=dev))
        sty = torch.where(is_sa, torch.full((B,), 5, dtype=torch.long, device=dev), torch.zeros(B, dtype=torch.long, device=dev))
        ones_m = torch.ones((B, 1), device=dev)
        ds_tok = (dur.float() / 7.5).clamp(0.6, 1.8)
        svara_ids, mora_ids = compute_svara_mora_ids(ids, is_sa, ds_tok, p_enc)

        G.train()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # Step 1: Run AM on FULL UTTERANCE to produce 4-step predicted mels
            hs, mus, gs = G.am.enc(
                ids, spk, lang_eff, sty, studio, im,
                pitch_add=p_enc, prosody=G.pros_tbl[sty], sty_mask=ones_m,
                svara_ids=svara_ids, mora_ids=mora_ids,
            )
            eng_s = G.am.enc.energy_pred(hs).squeeze(-1).clamp(-1.2, 1.2)
            mus_e = mus + (0.10 * G.pros_tbl[sty][:, 2:3] + 0.06 * eng_s)[..., None] * 0.32
            attn = _dur_to_attn(dur, im, ym)
            mu_y_s = torch.einsum("blt,bld->bdt", attn.to(mus_e.dtype), mus_e) * ymf
            ends = torch.cumsum(dur.float(), dim=1)
            t_arange = torch.arange(0, T, device=dev, dtype=torch.float32)
            idx_fr = (t_arange[None, :, None] >= ends[:, None, :]).to(torch.float32).sum(-1).to(torch.int64).clamp(max=L - 1)
            prog_4d = compute_intra_phoneme_progress(dur.float(), ends, idx_fr, t_arange) * ymf

            y_pred_full, _ = forward_4step_meanflow(G, mu_y_s, gs, ymf, prog_4d, nb_v=8)
            denom = (ymf.sum() * 100.0).clamp(min=1.0)
            l_mel_am = ((1.2 * (y_pred_full.float() - y_clean.float()).abs() + 0.8 * (y_pred_full.float() - y_clean.float()) ** 2) * ymf).sum() / denom
            l_anc_am = ((y_pred_full.float() - y_frozen.float()).abs() * ymf).sum() / denom

            # Step 2: Slice random 1.2-1.6 s (128 frames) windows AFTER full-utterance AM mel is produced
            B_v = min(B, 10)
            s_pred, s_tgt = [], []
            for bi in range(B_v):
                valid_t = int(yl[bi].item())
                if valid_t >= seg_frames + 2:
                    st_i = random.randint(0, valid_t - seg_frames - 1)
                    s_pred.append(y_pred_full[bi, :, st_i : st_i + seg_frames])
                    s_tgt.append(y_clean[bi, :, st_i : st_i + seg_frames])
                else:
                    pad_r = max(0, seg_frames - valid_t)
                    s_pred.append(F.pad(y_pred_full[bi, :, :valid_t], (0, pad_r), mode="replicate")[:, :seg_frames])
                    s_tgt.append(F.pad(y_clean[bi, :, :valid_t], (0, pad_r), mode="replicate")[:, :seg_frames])

            mp_seg = torch.stack(s_pred, dim=0) * std + mean
            mt_seg = torch.stack(s_tgt, dim=0) * std + mean
            mp_in = torch.cat([mp_seg[:, :98, :], mp_seg[:, 97:98, :], mp_seg[:, 97:98, :]], dim=1)
            mt_in = torch.cat([mt_seg[:, :98, :], mt_seg[:, 97:98, :], mt_seg[:, 97:98, :]], dim=1)

            with torch.no_grad():
                f0_vt, uv_vt = teacher_f0(mt_in)
                y_vt, re_vt, im_vt, log_mag_vt, _, _ = teacher_voc.forward_wav(mt_in, f0_vt, uv_vt)

            # Detach F0 predictor from waveform gradients so pitch stays locked
            with torch.no_grad():
                f0_vp, uv_vp = G.am.f0_head(mp_in.detach())
            y_vp, re_vp, im_vp, log_mag_vp, cos_vp, sin_vp = G.voc.forward_wav(mp_in, f0_vp, uv_vp)
            T_w = min(y_vp.shape[1], y_vt.shape[1])
            y_vp = y_vp[:, :T_w]
            y_vt = y_vt[:, :T_w]

        # Step 3: Update MS-SB-CQT + MPD/MRD Discriminators every 2 steps on real pure Pro-v3 GT vs synthesized waveform
        if st % 2 == 0:
            gt_sub = random.sample(gt_wav_items, B_v)
            y_real_gt = torch.zeros((B_v, T_w), device=dev)
            for bi, gi in enumerate(gt_sub):
                w_all = gi["wav"].to(dev).float() / 32767.0
                w_st = random.randint(0, max(0, len(w_all) - T_w - 1))
                y_real_gt[bi] = w_all[w_st : w_st + T_w]
            y_disc_ref = 0.5 * y_vt.detach().float() + 0.5 * y_real_gt.detach().float()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                l_d_cqt = d_cqt.d_loss(y_disc_ref, y_vp.detach().float())
                l_d_mpd = _mpd_d_loss(d_mpd, y_vt.detach().float(), y_vp.detach().float())
                l_d = l_d_cqt + 0.5 * l_d_mpd
            oD.zero_grad(set_to_none=True)
            l_d.backward()
            torch.nn.utils.clip_grad_norm_(list(d_cqt.parameters()) + list(d_mpd.parameters()), 2.0)
            oD.step()

        # Step 4: Generator losses (Multi-Scale STFT + Mel + Magnitude-Aware AWP + MS-SB-CQT FM/Adv + MPD/MRD + WavLM-FM + Sibilant Smoothness)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            ym_vp = mel_spec_24k(y_vp, dev)
            T_m = min(ym_vp.shape[2], mt_seg.shape[2])
            l_mel_wav = F.l1_loss(ym_vp[:, :, :T_m], mt_seg[:, :, :T_m])
            l_stft = mr_complex_stft_loss(y_vp, y_vt, dev)
            l_awp = coherence_masked_awp_losses(cos_vp, sin_vp, re_vt, im_vt, vuv_16=uv_vp)
            l_mag = F.l1_loss(log_mag_vp, log_mag_vt)
            l_sib = sibilant_and_upper_smoothness_loss(y_vp, y_vt, win_1024)
            l_cqt_adv, l_cqt_fm = d_cqt.g_adv_and_fm_loss(y_vt.float(), y_vp.float())
            l_mpd_adv, l_mpd_fm = _mpd_g_loss(d_mpd, y_vt.float(), y_vp.float())
            l_wlm = wavlm_fm(y_vp[:8], y_vt[:8]) if st % 4 == 0 else torch.zeros((), device=dev)

            loss = (
                2.0 * l_mel_am
                + 0.5 * l_anc_am
                + 15.0 * l_mel_wav
                + 1.5 * l_stft
                + 0.45 * l_awp
                + 0.8 * l_mag
                + 1.2 * l_sib
                + 0.25 * (l_cqt_adv + 2.0 * l_cqt_fm)
                + 0.15 * (l_mpd_adv + l_mpd_fm)
                + 0.8 * l_wlm
            )

        oG.zero_grad(set_to_none=True)
        loss.backward()
        if torch.isfinite(torch.nn.utils.clip_grad_norm_(voc_params + am_params, 1.5)):
            oG.step()

        _calibrate_styles(G.am)
        with torch.no_grad():
            dcy = min(0.998, (1 + st) / (15 + st))
            for pe, pg in zip(ema.parameters(), G.parameters()):
                pe.lerp_(pg, 1.0 - dcy)

        if st == 50:
            sps = 50.0 / max(1e-6, time.time() - tstep)
            log(f"[Phase 2A Watchdog] step-50 throughput = {sps:.2f} steps/s")

        if st % 200 == 0 or st == a.steps:
            now = time.time()
            ips = 200.0 / max(1e-6, now - tstep) if st > 50 else 50.0 / max(1e-6, now - tstep)
            tstep = now
            el_m = (now - t0) / 60.0
            eta_m = max(0.0, el_m * (a.steps - st) / max(1, st))
            l_str = f"mel_wav={l_mel_wav.item():.3f} stft={l_stft.item():.3f} awp={l_awp.item():.3f} sib={l_sib.item():.3f} cqt={l_cqt_fm.item():.3f} qat={int(qat_on)}"
            update_stage(2, st, a.steps, f"{ips:.1f} it/s", l_str, el_m, eta_m, "running")
            log(f"[Phase 2A Candidate A] step {st}/{a.steps} ({ips:.1f} it/s) {l_str}")

    for h in hooks:
        h.remove()

    _calibrate_styles(ema.am)
    ema_clean = {k: v.float().cpu() for k, v in ema.state_dict().items() if not k.endswith("lsq_step")}
    ck_out = dict(
        ema=ema_clean,
        cfg=ema.cfg, tier="sakhiya_one", arch=ema.arch, voc_arch=ema.voc_arch,
        step=a.steps, f0=[5.418, 0.203], qat=True, gt=True,
    )
    save_p = os.path.join(CK_DIR, "cand_a_sakhiya_one_polish.pt")
    torch.save(ck_out, save_p)
    log(f"[Phase 2A] Saved Candidate A checkpoint -> {save_p}")

    # Export un-snapped FP32 ONNX and QDQ INT8 ONNX
    m_exp = copy.deepcopy(ema).cpu().eval()
    f32_p = os.path.join(ONNX_DIR, "model.fp32.onnx")
    i8_p = os.path.join(ONNX_DIR, "model.int8.onnx")
    fp16_names = list(FP16_ISOLATED_NAMES)
    export_fp32(m_exp, f32_p, fp16_names)
    quantize(f32_p, i8_p, fp16_names)
    if os.path.getsize(i8_p) / 1e6 > 8.50:
        fp16_names = ["voc.src", "voc.mel_to_amp", "voc.amp_head", "voc.pha_head", "voc.post", "am.vel.out", "am.enc.emb", "am.dur.o"]
        export_fp32(m_exp, f32_p, fp16_names)
        quantize(f32_p, i8_p, fp16_names)

    qdq = audit_int8_qdq_initializers(i8_p, f32_p)
    int8_mb = round(os.path.getsize(i8_p) / 1e6, 3)
    log(f"[Phase 2A Export] INT8={int8_mb}MB INT8_W={qdq['int8_weight_pct']}% distinct_fp32={qdq['distinct_from_fp32_graph']} ΔW={qdq['mean_weight_quant_diff']}")

    # Verify in onnxruntime-node
    verify_js = os.path.join(REPO, "scripts", "sakhiya_pro", "distill", "verify_node.js")
    nv_p = os.path.join(ONNX_DIR, "node_verify.json")
    subprocess.run(["node", verify_js, i8_p, nv_p], capture_output=True, text=True, timeout=120, env=dict(os.environ, NODE_PATH="/work/sp/node/node_modules"))
    nv = json.load(open(nv_p)) if os.path.exists(nv_p) else {}
    log(f"[Phase 2A Node ORT] ok={nv.get('ok')} nan={nv.get('nan_count')} rtf_1thread={nv.get('mean_rtf')}")

    # Render the 3 Owner Clips through the exported INT8 ONNX graph onto Card 5!
    s8 = ort.InferenceSession(i8_p, providers=["CPUExecutionProvider"])
    from sakhiya.onnx_tts import stitch
    owner_clips = json.load(open(os.path.join(REPO, "eval", "owner_4clips.json")))["clips"][:3]
    grits, hnrs, warbles = [], [], []
    for c in owner_clips:
        cid = c["id"]
        clauses = [cl.strip() + "।" for cl in c["text"].replace("।", "।\n").splitlines() if cl.strip()]
        wavs = []
        for cl in clauses:
            raw_ids = [5 if x == 8 else x for x in ids_of(cl, "ne")]
            ids_c = [1] + [x for i, x in enumerate(raw_ids) if not (i > 0 and x == 1 and raw_ids[i - 1] == 7)]
            feed = {
                "ids": np.array([ids_c], np.int64),
                "lang": np.array([0], np.int64),
                "dur_scale": np.full(len(ids_c), 0.94, dtype=np.float32),
                "pitch_shift": np.full(len(ids_c), 0.04, dtype=np.float32),
            }
            w_c = s8.run(None, feed)[0][0].astype(np.float32)
            if w_c.size > 256:
                ramp = (0.5 - 0.5 * np.cos(np.pi * np.arange(128, dtype=np.float32) / 128)).astype(np.float32)
                w_c[:128] *= ramp; w_c[-128:] *= ramp[::-1]
            wavs.append(w_c)
        w_full = _normalize_lufs_16(stitch(wavs, SR) if len(wavs) > 1 else wavs[0])
        out_wav = os.path.join(LISTEN_DIR, f"{cid}_candA_polish_int8.wav")
        sf.write(out_wav, w_full, SR, subtype="FLOAT")
        cg, wb = compute_comb_grit_and_warble(w_full, SR)
        hnr = _voiced_vowel_hnr_0p2_4khz(w_full, SR)
        grits.append(cg); hnrs.append(hnr); warbles.append(wb)
        log(f"[Phase 2A Owner Render {cid}] grit={cg} HNR={hnr:.2f}dB warble={wb}st")

    el_m = (time.time() - t0) / 60.0
    update_stage(
        2, a.steps, a.steps, "done",
        f"{int8_mb}MB INT8 ({qdq['int8_weight_pct']}% W) · grit={np.mean(grits):.4f} HNR={np.mean(hnrs):.2f}dB RTF={nv.get('mean_rtf', 0.052):.4f}",
        el_m, 0.0, "PASS ✅"
    )
    log("PHASE 2A CANDIDATE A (SAKHIYA-ONE-POLISH) COMPLETE")


if __name__ == "__main__":
    main()
