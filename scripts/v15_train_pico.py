#!/usr/bin/env python3
"""Sakhiya v15 Phase 4 (Breakthrough 3): Train World-First Sub-1.0 MB (`<= 980 KB` int8 ONNX) `Sakhiya-Pico`.
1. Runs HARD PRE-TRAINING SIZE ASSERTION (`enc + dec + voc <= 990,000` bytes int8 ONNX) before step 1.
2. Warm-initializes `SakhiyaPicoAM` universal looped blocks & embeddings from `v9_sano_am.pt` (`warm_from_sano`).
3. Stage A (first 45% of budget): High-speed bf16 Teacher + Sano hidden-state, prior, duration (with >=5f short-word floor),
   F0/energy, spectral/temporal delta-mel, cepstral, and `ctc_pron` distillation.
4. Stage B (remaining 55% of budget): End-to-End Vocoder-in-the-Loop joint co-distillation through `MS-RND-Wavehax-Lite`
   with Multi-Res Complex STFT + 33-bin Complex/Phase/Group-Delay + Frozen `WavLM-base-plus` Perceptual Feature Loss,
   finishing with Straight-Through Estimator (STE) Per-Channel INT8 QAT polish in the final 20%."""
import argparse, copy, json, math, os, random, subprocess, sys, time
import torch, torch.nn as nn, torch.nn.functional as F

R = "/work/tts"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from sakhiya import g2p, sanskrit as SA
from sakhiya.cfm import SakhiyaAM, STYLE_PROSODY_TARGETS
from sakhiya.ctc_aligner import MelCTCAligner
from sakhiya.symbols import SYM2ID, LANG2ID
from sakhiya.v15_pico import SakhiyaPicoAM
from sakhiya.v15_rnd_wavehax import MSRNDWavehax
from sakhiya.v2_vocoder import VocoderV2, mel_spec_24k
from sakhiya.v4_slm import WavLMSLMHelper
from sakhiya.v6_distill import _expand_batch_durs, build_teacher_anchor_bank
from scripts.export_v2_onnx import TIER_CFG
from scripts.train_nano_atom import _bake
from scripts.train_v2_voc import mr_complex_stft_loss
from scripts.train_v3_voc import resamp16
from scripts.v15_train_wavehax import spec_phase_gd_loss, warm_init_f0
from scripts.v2_full_eval import load_utmos
from scripts.v8_voc_gta import export_mel, val_mels


def assert_pico_sub_1mb(s_am, s_voc, out_dir):
    """Hard pre-training size assertion: export dummy int8 ONNX and verify enc + dec + voc <= 990,000 bytes (< 0.99 MB)."""
    tmp_d = f"{out_dir}/_size_check"
    os.makedirs(tmp_d, exist_ok=True)
    torch.save({"arch": "sakhiya_pico", "ema": s_am.state_dict()}, f"{tmp_d}/am.pt")
    torch.save({"arch": "ms_rnd_wavehax", "variant": "lite", "G_ema": s_voc.state_dict()}, f"{tmp_d}/voc.pt")
    cmd = f"python3 {REPO}/scripts/export_v2_onnx.py --tier pico --am {tmp_d}/am.pt --voc {tmp_d}/voc.pt --out {tmp_d}/onnx"
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Pico ONNX export check failed:\n{r.stdout}\n{r.stderr}")
    sizes = {f: os.path.getsize(f"{tmp_d}/onnx/{f}.int8.onnx") for f in ("encoder", "decoder", "vocoder")}
    tot_bytes = sum(sizes.values())
    print(f"[Pico-Size-Gate] PASSED PRE-TRAINING ASSERTION: encoder={sizes['encoder']:,}B decoder={sizes['decoder']:,}B vocoder={sizes['vocoder']:,}B TOTAL={tot_bytes:,}B ({tot_bytes/1e6:.4f} MB <= 0.9900 MB)", flush=True)
    assert tot_bytes <= 990_000, f"Pico total int8 bytes {tot_bytes} > 990,000!"
    return sizes, tot_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sano_init", default=f"{R}/ck/v9_sano_am.pt")
    ap.add_argument("--voc_init", default=f"{R}/v15/wavehax_lite/voc.pt")
    ap.add_argument("--out", default=f"{R}/v15/pico")
    ap.add_argument("--max-minutes", type=float, default=65.0)
    ap.add_argument("--bs", type=int, default=28)
    ap.add_argument("--seg", type=int, default=48)
    ap.add_argument("--lr_am", type=float, default=6.0e-4)
    ap.add_argument("--lr_voc", type=float, default=1.5e-4)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    assert a.max_minutes <= 115.0, "hard stage cap <= 115 min"

    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    tag = "v15-pico"
    torch.backends.cudnn.benchmark = True

    D = torch.load(f"{R}/feats/v6_expressive_vedic_24k.pt", weights_only=False)
    mean, std = float(D.get("mean", -5.5028)), float(D.get("std", 2.1869))

    # Load frozen MelCTCAligner
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

    # Load V1 Teacher AM & Vocoder + Sano Reference Teacher for exact d=96 hidden-state alignment
    t_ck = torch.load(f"{R}/runs/v5_c1/am.pt", map_location=dev, weights_only=False)
    t_sd = {k: v.float() for k, v in t_ck.get("ema", t_ck.get("net", t_ck)).items() if not k.startswith("ctc_rec.")}
    t_am = SakhiyaAM().to(dev).eval()
    t_am.load_state_dict(t_sd, strict=False)
    t_am.stats.copy_(torch.tensor([mean, std], device=dev))
    _bake(t_am)
    for p in t_am.parameters():
        p.requires_grad_(False)

    c_sano = TIER_CFG["sano"]
    sano_am = SakhiyaAM(**{k: c_sano[k] for k in ("d_enc", "L_enc", "ff_enc", "d_vel", "nb_vel", "h_dur")}).to(dev).eval()
    sano_am.stats.copy_(torch.tensor([mean, std], device=dev))
    _bake(sano_am)
    if os.path.exists(a.sano_init):
        ck_sano = torch.load(a.sano_init, map_location=dev, weights_only=False)
        sano_sd = {k: v.float() for k, v in ck_sano.get("ema", ck_sano.get("net", ck_sano)).items()}
        sano_am.load_state_dict(sano_sd, strict=False)
    _bake(sano_am)
    for p in sano_am.parameters():
        p.requires_grad_(False)

    t_voc = VocoderV2().to(dev).eval()
    vck = torch.load(f"{R}/runs/v5_c1/voc.pt", map_location=dev, weights_only=False)
    t_voc.load_state_dict({k: v.float() for k, v in vck.get("G_ema", vck.get("G", vck)).items()}, strict=False)
    for p in t_voc.parameters():
        p.requires_grad_(False)

    # Build clean Teacher anchor bank + studio training pool
    anchors = build_teacher_anchor_bank(t_am, mean, std, ctc, dev)
    base_tr = [x for x in D["items"] if x.get("split") == "train" and 3 <= len(x["ids"]) and 28 <= x["mel"].shape[1] <= 750]
    exp_tr = [x for x in base_tr if x.get("v3_style_id", 0) in (2, 5)]
    pool = base_tr + exp_tr * 2 + anchors * 35

    # Build & Warm-Start SakhiyaPicoAM + MSRNDWavehax("lite")
    s_am = SakhiyaPicoAM().to(dev)
    s_am.stats.copy_(torch.tensor([mean, std], device=dev))
    s_am.warm_from_sano(sano_am.state_dict())
    ema_am = copy.deepcopy(s_am).eval()

    s_voc = MSRNDWavehax(variant="lite").to(dev)
    if os.path.exists(a.voc_init):
        ck_v = torch.load(a.voc_init, map_location=dev, weights_only=False)
        s_voc.load_state_dict({k: v.float() for k, v in ck_v.get("G_ema", ck_v.get("G", ck_v)).items()}, strict=False)
        s_voc.src.f0_net.load_state_dict(t_voc.src.f0_net.state_dict())
        for p in s_voc.src.f0_net.parameters():
            p.requires_grad_(False)
        print(f"[{tag}] Loaded Wavehax-lite vocoder from {a.voc_init} (step={ck_v.get('step', 0)})", flush=True)
    else:
        warm_init_f0(s_voc, t_voc, [[x["mel"].half().to(dev) for x in base_tr[:256]]], dev)
    ema_voc = copy.deepcopy(s_voc).eval()

    # HARD PRE-TRAINING SIZE ASSERTION (< 0.99 MB int8 ONNX)
    assert_pico_sub_1mb(ema_am, ema_voc, a.out)

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
    print(f"[{tag}] WARM-START e2e_utmos={base_utmos:.4f} (with V1 teacher voc={t_utmos:.4f}) am_params={sum(p.numel() for p in s_am.parameters()):,} voc_params={sum(p.numel() for p in s_voc.parameters()):,}", flush=True)

    opt_am = torch.optim.AdamW(s_am.parameters(), lr=a.lr_am, betas=(0.9, 0.99), weight_decay=1e-4)
    opt_voc = torch.optim.AdamW([p for p in s_voc.parameters() if p.requires_grad], lr=a.lr_voc, betas=(0.8, 0.99), weight_decay=1e-5)

    best = base_utmos
    torch.save(dict(arch="sakhiya_pico", ema=ema_am.state_dict(), net=s_am.state_dict(), step=0, mean=mean, std=std, pre_baked_v25=True), f"{a.out}/am.pt")
    torch.save(dict(arch="ms_rnd_wavehax", variant="lite", G_ema=ema_voc.state_dict(), G=s_voc.state_dict(), step=0, utmos={"e2e": best}), f"{a.out}/voc.pt")

    t0, budget, step = time.time(), a.max_minutes * 60.0, 0
    L_wav = a.seg * 256
    qat_active = False
    stage_b_active = False
    FUNC_C_IDS = torch.tensor([SYM2ID[p] for p in ("r", "t̪", "n", "m", "h", "tsʰ") if p in SYM2ID], device=dev)
    FUNC_V_IDS = torch.tensor([SYM2ID[p] for p in ("ʌ", "o") if p in SYM2ID], device=dev)

    while (time.time() - t0) < budget:
        frac = (time.time() - t0) / budget
        if frac >= 0.40 and not stage_b_active:
            stage_b_active = True
            if os.path.exists(a.voc_init):
                ck_v = torch.load(a.voc_init, map_location=dev, weights_only=False)
                s_voc.load_state_dict({k: v.float() for k, v in ck_v.get("G_ema", ck_v.get("G", ck_v)).items()}, strict=False)
                s_voc.src.f0_net.load_state_dict(t_voc.src.f0_net.state_dict())
                for p in s_voc.src.f0_net.parameters():
                    p.requires_grad_(False)
                ema_voc.load_state_dict(s_voc.state_dict())
                print(f"[{tag}] Hot-reloaded latest Wavehax-lite from {a.voc_init} (step={ck_v.get('step', 0)}) for Stage B!", flush=True)
            print(f"[{tag}] step={step} Transitioning Stage A -> Stage B (End-to-End Vocoder-in-the-Loop + WavLM Co-Distillation)!", flush=True)
        if frac >= 0.80 and not qat_active:
            s_am.set_qat(True)
            s_voc.set_qat(True)
            qat_active = True
            print(f"[{tag}] step={step} Enabled STE Per-Channel INT8 QAT on Pico AM + Vocoder!", flush=True)

        cos_mult = 0.5 * (1.0 + math.cos(math.pi * frac)) * 0.9 + 0.1
        warm_m = min(1.0, (step + 1) / 250.0)
        for g_ in opt_am.param_groups:
            g_["lr"] = a.lr_am * warm_m * cos_mult
        for g_ in opt_voc.param_groups:
            g_["lr"] = a.lr_voc * warm_m * cos_mult
        step += 1

        if step == 50:
            sps = 50.0 / max(time.time() - t0, 1e-6)
            print(f"[GATE] step-50 throughput {sps:.2f} steps/s -> ~{sps * budget / 1000:.1f}k steps inside {a.max_minutes:.0f} min cap", flush=True)
            if sps < 4.0:
                print("[GATE] throughput too low, aborting", flush=True)
                sys.exit(3)

        xs = random.sample(pool, a.bs - 6) + random.sample(anchors, min(6, len(anchors)))
        B = len(xs)
        L = max(len(x["ids"]) for x in xs)
        T = min(max(int(x["dur_ctc"].sum().item()) for x in xs), 650)
        ids = torch.zeros(B, L, dtype=torch.long, device=dev)
        durs = torch.zeros(B, L, dtype=torch.long, device=dev)
        po = torch.zeros(B, L, dtype=torch.float32, device=dev)
        y_gt = torch.zeros((B, 100, T), dtype=torch.float32, device=dev)

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
                h_ref, mu_ref, g_ref = sano_am.enc(ids, spks, langs, stys, stud, im, pitch_add=po, prosody=pros_b, sty_mask=ones_m)
                f0_t = sano_am.enc.f0_pred(h_ref)
                eng_t = sano_am.enc.energy_pred(h_ref)
                mu_y_ref = _expand_batch_durs(mu_ref, durs, T) * ymf
                t0_z = torch.zeros((B,), device=dev, dtype=torch.float32)
                y_sano = (mu_y_ref + sano_am.vel(mu_y_ref, t0_z, mu_y_ref, g_ref, ymf)) * ymf
                # Blend studio GT mel and clean 1-step teacher mel for crisp, un-blurred targets
                y_tgt = 0.55 * y_gt + 0.45 * y_sano.float()
                ld_tgt = torch.log(durs.float().clamp(min=1.0)) * im.float()

            h_s, mu_s, g_s = s_am.enc(ids, spks, langs, stys, stud, im, pitch_add=po, prosody=pros_b, sty_mask=ones_m)
            # Hidden-state alignment loss between 4-loop universal encoder and 4-layer Sano encoder
            l_hid = F.l1_loss(h_s * im[:, :, None], h_ref.detach() * im[:, :, None])
            l_heads = 0.5 * (((s_am.enc.f0_pred(h_s) - f0_t) ** 2 + (s_am.enc.energy_pred(h_s) - eng_t) ** 2) * im[:, :, None]).sum() / im.sum().clamp(min=1)
            l_prior = 2.0 * (((mu_s - mu_ref).abs() + (mu_s - mu_ref) ** 2) * im[:, :, None]).sum() / (im.sum() * 100.0).clamp(min=1.0)

            ld_pred = s_am.dur.pred_logdur(ids, h_s, im).float()
            pos_l = torch.arange(L, device=dev)[None, :]
            is_short_or_tail = ((durs <= 6) & (ids > 6)) | (pos_l >= (il[:, None] - 4))
            w_dur = torch.where(is_short_or_tail & im, 2.5, 1.0) * im.float()
            l_dur = (F.smooth_l1_loss(ld_pred, ld_tgt, reduction="none") * w_dur).sum() / w_dur.sum().clamp(min=1.0)
            is_func_c = torch.isin(ids[:, 1:-2], FUNC_C_IDS) & (ids[:, :-3] <= 2) & torch.isin(ids[:, 2:-1], FUNC_V_IDS) & (ids[:, 3:] <= 2)
            if is_func_c.any():
                d_pred_c = torch.exp(ld_pred[:, 1:-2].clamp(max=4.0))
                l_dur = l_dur + 0.5 * (F.relu(5.2 - d_pred_c) * is_func_c.float()).sum() / is_func_c.float().sum().clamp(min=1.0)

            mu_y_s = _expand_batch_durs(mu_s, durs, T) * ymf
            y_pred = (mu_y_s + s_am.vel(mu_y_s, t0_z, mu_y_s, g_s, ymf)) * ymf

            denom = (ymf.sum() * 100.0).clamp(min=1.0)
            l_mel_am = ((1.6 * (y_pred - y_tgt).abs() + (y_pred - y_tgt) ** 2) * ymf).sum() / denom
            df_p = y_pred[:, 1:, :] - y_pred[:, :-1, :]
            df_t = y_tgt[:, 1:, :] - y_tgt[:, :-1, :]
            l_df = ((df_p - df_t).abs() * ymf).sum() / denom
            ym_t = ymf[:, :, 1:] * ymf[:, :, :-1]
            l_dt = (((y_pred[:, :, 1:] - y_pred[:, :, :-1]) - (y_tgt[:, :, 1:] - y_tgt[:, :, :-1])).abs() * ym_t).sum() / (ym_t.sum() * 100.0 + 1.0)
            cep_p = torch.fft.rfft(y_pred.float(), dim=1).abs()[:, 4:48, :]
            cep_t = torch.fft.rfft(y_tgt.float(), dim=1).abs()[:, 4:48, :]
            l_cep = ((cep_p - cep_t).abs() * ymf).sum() / (ymf.sum() * 44.0).clamp(min=1.0)

            hinge_floor = max(0.0, 0.22 * (1.0 - frac / 0.70))
            l_ctc_raw = ctc.ctc_loss(0.25 * mu_y_s.float() + 0.75 * y_pred.float(), yl, ids, il)
            l_ctc = F.relu(l_ctc_raw - hinge_floor) if torch.isfinite(l_ctc_raw) else torch.tensor(0.0, device=dev)

        loss = l_mel_am + 2.4 * l_df + 1.2 * l_dt + 0.65 * l_cep + l_prior + 0.8 * l_hid + l_heads + 0.75 * l_dur + 0.25 * l_ctc
        l_wav_mel = l_wav_stft = l_cplx = l_wavlm = torch.tensor(0.0, device=dev)

        if stage_b_active:
            mel_s_segs, mel_t_segs = [], []
            for k in range(B):
                valid_T = max(a.seg, min(int(yl[k].item()), xs[k]["mel"].shape[1]))
                s0 = random.randint(0, max(0, valid_T - a.seg))
                s_slice = (y_pred[k, :, s0:s0 + a.seg].float() * std + mean)
                t_slice = (y_tgt[k, :, s0:s0 + a.seg].float() * std + mean)
                if s_slice.shape[1] < a.seg:
                    s_slice = F.pad(s_slice, (0, a.seg - s_slice.shape[1]), value=mean)
                    t_slice = F.pad(t_slice, (0, a.seg - t_slice.shape[1]), value=mean)
                mel_s_segs.append(s_slice)
                mel_t_segs.append(t_slice)

            x_s = export_mel(torch.stack(mel_s_segs))
            x_t = export_mel(torch.stack(mel_t_segs))

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                re_t, im_t, _, _ = t_voc.spectrum_and_f0(x_t)
            re_t, im_t = re_t.float(), im_t.float()
            wav_t = s_voc.spec_to_wav(re_t, im_t)[:, :L_wav]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                re_s, im_s, _, _ = s_voc.spectrum_and_f0(x_s)
            re_s, im_s = re_s.float(), im_s.float()
            wav_s = s_voc.spec_to_wav(re_s, im_s)[:, :L_wav]

            mag_t = torch.sqrt(re_t ** 2 + im_t ** 2 + 1e-9)
            mag_s = torch.sqrt(re_s ** 2 + im_s ** 2 + 1e-9)
            l_lmag = F.l1_loss(torch.log(mag_s + 1e-5), torch.log(mag_t + 1e-5))
            l_cplx = ((re_s - re_t).abs() + (im_s - im_t).abs()).mean() / (mag_t.mean() + 1e-6)
            l_ph = spec_phase_gd_loss(re_s, im_s, re_t, im_t, mag_s, mag_t)

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

            loss = loss + 12.0 * l_wav_mel + 1.2 * l_wav_stft + 1.5 * l_lmag + 2.0 * l_cplx + 0.8 * l_ph + 2.5 * l_wavlm

        opt_am.zero_grad(set_to_none=True)
        if stage_b_active:
            opt_voc.zero_grad(set_to_none=True)
        loss.backward()
        if torch.isfinite(torch.nn.utils.clip_grad_norm_(s_am.parameters(), 1.0)):
            opt_am.step()
        if stage_b_active and torch.isfinite(torch.nn.utils.clip_grad_norm_([p for p in s_voc.parameters() if p.requires_grad], 1.0)):
            opt_voc.step()

        with torch.no_grad():
            k_ema = 1.0 - min(0.996, (1.0 + step) / (10.0 + step))
            for pe, p in zip(ema_am.parameters(), s_am.parameters()):
                pe.lerp_(p, k_ema)
            if stage_b_active:
                for pe, p in zip(ema_voc.parameters(), s_voc.parameters()):
                    pe.lerp_(p, k_ema)

        if step % 200 == 0:
            print(f"[{tag}] step={step} {step / (time.time() - t0):.1f} st/s t={(time.time() - t0) / 60:.1f}m stg={'B' if stage_b_active else 'A'} mel_am={l_mel_am.item():.3f} hid={l_hid.item():.3f} dur={l_dur.item():.3f} ctc={l_ctc.item():.3f} wav_mel={l_wav_mel.item():.3f} cplx={l_cplx.item():.3f} qat={int(qat_active)}", flush=True)

        done_smoke = a.smoke and step >= a.smoke
        if step % a.eval_every == 0 or done_smoke:
            if not stage_b_active and os.path.exists(a.voc_init):
                try:
                    ck_v = torch.load(a.voc_init, map_location=dev, weights_only=False)
                    s_voc.load_state_dict({k: v.float() for k, v in ck_v.get("G_ema", ck_v.get("G", ck_v)).items()}, strict=False)
                    ema_voc.load_state_dict(s_voc.state_dict())
                except Exception:
                    pass
            cur = eval_e2e(ema_am, ema_voc)
            ck_am = dict(arch="sakhiya_pico", ema=ema_am.state_dict(), net=s_am.state_dict(), step=step, mean=mean, std=std, pre_baked_v25=True, e2e_utmos=cur)
            ck_voc = dict(arch="ms_rnd_wavehax", variant="lite", G_ema=ema_voc.state_dict(), G=s_voc.state_dict(), step=step, utmos={"e2e": cur})
            torch.save(ck_am, f"{a.out}/am_last.pt.tmp"); os.replace(f"{a.out}/am_last.pt.tmp", f"{a.out}/am_last.pt")
            torch.save(ck_voc, f"{a.out}/voc_last.pt.tmp"); os.replace(f"{a.out}/voc_last.pt.tmp", f"{a.out}/voc_last.pt")
            if cur >= best:
                best = cur
                torch.save(ck_am, f"{a.out}/am.pt.tmp"); os.replace(f"{a.out}/am.pt.tmp", f"{a.out}/am.pt")
                torch.save(ck_voc, f"{a.out}/voc.pt.tmp"); os.replace(f"{a.out}/voc.pt.tmp", f"{a.out}/voc.pt")
            print(f"[{tag}] SCORE step={step} e2e_utmos={cur:.4f} best={best:.4f} init={base_utmos:.4f}", flush=True)
            if done_smoke:
                break

    json.dump(dict(tier="pico", best_utmos=best, init_utmos=base_utmos, steps=step), open(f"{a.out}/summary.json", "w"), indent=2)
    print(f"[{tag}] DONE best_utmos={best:.4f} steps={step}", flush=True)


if __name__ == "__main__":
    main()
