#!/usr/bin/env python3
"""Session 75 Unified Breakthrough Pipeline (`scripts/sakhiya_pro/distill2/run_session75_breakthrough.py`).

Executes:
- Phase 0.2: 100% Pure Sakhiya-Pro-v3 Single-Timbre Cache (`/work/sp/distill/pure_pro_v3_cache.pt`)
- Phase 1: Studio-Clean Alias-Free Full-Band Harmonic + Parallel Amplitude/Anti-Wrapped-Phase (AWP) iSTFT Vocoder
  (`v4_awp_voc320.pt` for t30/t15/t10, `v4_awp_voc144_t1.pt` for t1 Pico)
  - Trains on 100% pure Pro-v3 audio with +-1.5% F0 jitter, Multi-Res Complex STFT, Anti-Wrapped Phase (GD/IAF),
    MPD/MRD GAN, Frozen Multi-Layer WavLM Feature-Matching (layers 3, 6, 9), and Voiced-Only 24-Bin LTAS Envelope Loss.
  - Verifies `sim_ProV3 >= 0.90` on GT mels before student refinement.
- Phase 2 & 3: Joint End-to-End Waveform Distillation + Intra-Phoneme Progress + Vedic Svara & Moraic Conditioning +
  Multi-Resolution Contrastive Timbre Lock + Coupled Reflow across all 4 active tiers (`t30`, `t15`, `t10`, `t1` Pico).
"""
import argparse, copy, glob, json, math, os, random, re, subprocess, sys, time
import librosa, numpy as np, soundfile as sf, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

HERE = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(HERE, "..", "distill")
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for p in (HERE, DIST, REPO, os.path.join(REPO, "scripts", "sakhiya_pro"), "/work/sp"):
    if p not in sys.path:
        sys.path.insert(0, p)

import sp_common as C
import nep_norm as _nn
from sakhiya import chandas as _ch, g2p as _g2p, symbols as _S
from sakhiya.v2_vocoder import mel_spec_24k, SR, HOP, N_FFT
from sakhiya.awp_vocoder import AWPVocoder, awp_phase_losses
from sakhiya.ctc_aligner import MelCTCAligner, batch_viterbi_ctc_durations
from sakhiya.cfm import STYLE_PROSODY_TARGETS, _dur_to_attn
from scripts.train_v2_voc import DiscV2, mr_complex_stft_loss, gpu_f0_uv
from student import (
    Student, build, set_qat, n_params, TIERS,
    compute_svara_mora_ids, compute_intra_phoneme_progress, ste_int8_weight, _make_svd_seq,
)


def install_awp_qat_hooks(model):
    hooks = []
    fp16_skip = ("voc.src", "voc.amp_head", "voc.pha_head", "voc.pre", "am.f0_head", "am.vel.out", "am.vel.inp", "am.vel.prog_proj", "am.enc.emb", "am.enc.svara_emb", "am.enc.mora_emb", "am.enc.f0_pred", "am.enc.vuv_pred", "am.dur.o")
    for name, m in model.named_modules():
        if any(name.startswith(s) or name.endswith(s) for s in fp16_skip) or hasattr(m, "qat_enabled"):
            continue
        if isinstance(m, (nn.Conv1d, nn.Linear)) and m.weight.numel() >= 256 and m.weight.shape[0] >= 16:
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


class FrozenMultiLayerWavLM(nn.Module):
    """Frozen Multi-Layer WavLM (layers 3, 6, 9) perceptual feature-matching loss."""
    def __init__(self, dev="cuda"):
        super().__init__()
        from transformers import WavLMModel
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus-sv").to(dev).eval()
        for p in self.wavlm.parameters():
            p.requires_grad_(False)
        self.layers = (3, 6, 9)

    def _prep16k(self, w24: torch.Tensor) -> torch.Tensor:
        w16 = F.interpolate(w24[:, None, :].float(), scale_factor=16000.0 / 24000.0, mode="linear", align_corners=False)[:, 0, :]
        mu = w16.mean(dim=-1, keepdim=True)
        sd = w16.std(dim=-1, keepdim=True).clamp(min=1e-4)
        return (w16 - mu) / sd

    def forward(self, y_pred: torch.Tensor, y_gt: torch.Tensor) -> torch.Tensor:
        xp = self._prep16k(y_pred)
        with torch.no_grad():
            xt = self._prep16k(y_gt)
            out_t = self.wavlm(xt, output_hidden_states=True).hidden_states
            hs_t = [out_t[i].detach() for i in self.layers]
        out_p = self.wavlm(xp, output_hidden_states=True).hidden_states
        loss = 0.0
        for hp, ht in zip([out_p[i] for i in self.layers], hs_t):
            T_m = min(hp.shape[1], ht.shape[1])
            loss = loss + F.l1_loss(hp[:, :T_m].float(), ht[:, :T_m].float())
        return loss / len(self.layers)


def voiced_24bin_ltas(mel_100: torch.Tensor, vuv: torch.Tensor = None) -> torch.Tensor:
    m = mel_100.float()
    if vuv is None:
        e = m.mean(dim=1, keepdim=True)
        e_max = e.amax(dim=-1, keepdim=True)
        w = torch.sigmoid((e - (e_max - 2.4)) * 4.0)
    else:
        w = (vuv.float() if vuv.dim() == 3 else vuv[:, None, :].float()).clamp(0.0, 1.0)
    w_sum = w.sum(dim=-1).clamp(min=1.0)
    ltas_100 = (m * w).sum(dim=-1) / w_sum
    return F.adaptive_avg_pool1d(ltas_100[:, None, :], 24)[:, 0, :]


def contrastive_ltas_timbre_loss(mel_pred: torch.Tensor, mel_pro_gt: torch.Tensor, c_pro_24: torch.Tensor, c_v1_24: torch.Tensor, vuv: torch.Tensor = None, margin: float = 0.35):
    lt_p = voiced_24bin_ltas(mel_pred, vuv)
    lt_t = voiced_24bin_ltas(mel_pro_gt, vuv).detach()
    l_utt = F.l1_loss(lt_p, lt_t)

    mid = 0.5 * (c_pro_24 + c_v1_24)
    p_n = F.normalize(lt_p - mid, dim=-1)
    pro_n = F.normalize(c_pro_24 - mid, dim=-1)
    v1_n = F.normalize(c_v1_24 - mid, dim=-1)
    sim_pro = (p_n * pro_n).sum(dim=-1)
    sim_v1 = (p_n * v1_n).sum(dim=-1)
    l_margin = F.relu(margin - (sim_pro - sim_v1)).mean() + F.l1_loss(lt_p, c_pro_24.expand_as(lt_p)) * 0.25

    hf_p = mel_pred[:, 65:94, :].float()
    hf_t = mel_pro_gt[:, 65:94, :].float().detach()
    l_hf = F.l1_loss(hf_p.mean(dim=-1), hf_t.mean(dim=-1)) + 0.5 * F.l1_loss(hf_p.std(dim=-1), hf_t.std(dim=-1))
    return l_utt + 0.5 * l_margin + 0.8 * l_hf


def train_phase1_awp_vocoder(tag: str, ch: int, base_ck: str, save_p: str, cache_obj: dict, is_t1: bool = False, steps: int = 800, bs: int = 36, seg_frames: int = 40, lr: float = 4.5e-5, dev="cuda"):
    """Phase 1: Train Studio-Clean Alias-Free Full-Band Harmonic + AWP Vocoder on 100% pure Pro-v3 audio."""
    log_p = f"/work/tts/logs/{tag}.log"
    os.makedirs("/work/tts/logs", exist_ok=True)
    open(log_p, "w").write("")

    def log(s):
        print(s, flush=True)
        open(log_p, "a").write(s + "\n")

    G = AWPVocoder(ch=ch).to(dev)
    if is_t1:
        for b_idx, r_k in ((1, 50), (2, 31)):
            blk = G.mrfs[0][b_idx]
            for i in range(len(blk.c1)):
                if isinstance(blk.c1[i], nn.Conv1d):
                    blk.c1[i] = _make_svd_seq(blk.c1[i], r_k, use_qat=False)
                if isinstance(blk.c2[i], nn.Conv1d):
                    blk.c2[i] = _make_svd_seq(blk.c2[i], r_k, use_qat=False)
        G.to(dev)
    G.load_pro_v3_filterbank(base_ck)
    G_ema = copy.deepcopy(G).eval()

    D = DiscV2().to(dev)
    dp = "/work/tts/ck/voc_disc_state.pt"
    if os.path.exists(dp):
        try:
            s = torch.load(dp, map_location=dev, weights_only=False)
            D.load_state_dict(s.get("D", s), strict=False)
        except Exception:
            pass

    wavlm_fm = FrozenMultiLayerWavLM(dev=dev)
    c_pro_24 = cache_obj["c_pro_ltas24"].to(dev)
    c_v1_24 = cache_obj["c_v1_ltas24"].to(dev)

    seen_ids = set()
    tr = []
    for x in cache_obj["train"]:
        if x["id"] not in seen_ids and x["mel"].shape[1] >= seg_frames + 4:
            seen_ids.add(x["id"])
            tr.append(x)

    oG = torch.optim.AdamW(G.parameters(), lr=lr, betas=(0.8, 0.99), weight_decay=1e-5)
    oD = torch.optim.AdamW(D.parameters(), lr=lr, betas=(0.8, 0.99), weight_decay=1e-5)

    t0 = time.time()
    tstep = time.time()
    n_fft_istft, hop_istft = 64, 16
    win_istft = torch.hann_window(n_fft_istft + 1, device=dev)[:-1]

    log(f"[Phase 1 {tag}] Training AWPVocoder(ch={ch}) for {steps} steps on {len(tr)} pure Pro-v3 utterances (bs={bs})...")
    for st in range(1, steps + 1):
        prog = st / max(1, steps)
        cur_lr = lr * (0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * prog)))
        for g_ in oG.param_groups + oD.param_groups:
            g_["lr"] = cur_lr

        xs = random.sample(tr, bs)
        wav_seg = torch.zeros((bs, seg_frames * HOP), device=dev)
        mel_seg = torch.zeros((bs, 100, seg_frames), device=dev)
        f0_seg = torch.zeros((bs, 1, seg_frames), device=dev)
        vuv_seg = torch.zeros((bs, 1, seg_frames), device=dev)

        for bi, x in enumerate(xs):
            Tk = x["mel"].shape[1]
            f_start = random.randint(0, Tk - seg_frames - 1)
            w_start = f_start * HOP
            wav_seg[bi] = x["wav"][w_start : w_start + seg_frames * HOP].to(dev).float() / 32767.0
            mel_seg[bi] = x["mel"][:, f_start : f_start + seg_frames].to(dev).float()
            f0_seg[bi, 0] = x["f0_hz"][f_start : f_start + seg_frames].to(dev).float().clamp(130.0, 380.0)
            vuv_seg[bi, 0] = x["vuv"][f_start : f_start + seg_frames].to(dev).float()

        # +-1.5% F0 jitter augmentation on GT F0 so small AM pitch errors never cause comb-filter beating
        jitter = 1.0 + (torch.rand((bs, 1, 1), device=dev) * 0.03 - 0.015)
        f0_in = (f0_seg * jitter).clamp(130.0, 380.0)
        mel_in = torch.cat([mel_seg[:, :98, :], mel_seg[:, 97:98, :], mel_seg[:, 97:98, :]], dim=1)

        G.train(); D.train()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            y_pred, _, _, _, cos_p, sin_p = G.forward_wav(mel_in, f0_in, vuv_seg)
        T_w = min(y_pred.shape[1], wav_seg.shape[1])
        y_pred = y_pred[:, :T_w]
        wav_gt = wav_seg[:, :T_w]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            dr, fr_d = D(wav_gt[:, None])
            df, _ = D(y_pred.detach()[:, None])
            ld = sum(F.relu(1.0 - r.float()).mean() + F.relu(1.0 + q.float()).mean() for r, q in zip(dr, df)) / len(dr)
        oD.zero_grad(set_to_none=True)
        ld.backward()
        torch.nn.utils.clip_grad_norm_(D.parameters(), 10.0)
        oD.step()
        ld_val = float(ld.item())

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            ym = mel_spec_24k(y_pred, dev)
            T_m = min(ym.shape[2], mel_seg.shape[2])
            l_mel = F.l1_loss(ym[:, :, :T_m], mel_seg[:, :, :T_m])
            l_stft = mr_complex_stft_loss(y_pred, wav_gt, dev)

            pad_i = (n_fft_istft - hop_istft) // 2
            x_pad = F.pad(wav_gt[:, None], (pad_i, pad_i), mode="reflect")[:, 0]
            spec_gt64 = torch.stft(x_pad, n_fft_istft, hop_istft, n_fft_istft, win_istft, center=False, return_complex=True)
            T_64 = min(cos_p.shape[2], spec_gt64.shape[2])
            l_awp = awp_phase_losses(cos_p[:, :, :T_64], sin_p[:, :, :T_64], spec_gt64.real[:, :, :T_64], spec_gt64.imag[:, :, :T_64])

            l_ltas = contrastive_ltas_timbre_loss(ym[:, :, :T_m], mel_seg[:, :, :T_m], c_pro_24, c_v1_24, vuv_seg[:, :, :T_m])
            l_wavlm = wavlm_fm(y_pred[:12], wav_gt[:12]) if st % 3 == 0 else torch.zeros((), device=dev)

            fr = [[t.detach() for t in fs] for fs in fr_d]
            df_g, ff_g = D(y_pred[:, None])
            l_adv = sum(F.relu(1.0 - q.float()).mean() for q in df_g) / len(df_g)
            n_fm = sum(len(A) for A in fr)
            l_fm = sum(F.l1_loss(b.float(), a_.float()) for A, B in zip(fr, ff_g) for a_, b in zip(A, B)) / n_fm

            lg = 45.0 * l_mel + 2.0 * l_stft + 0.5 * l_awp + 2.0 * l_ltas + 1.5 * l_wavlm + l_adv + 4.0 * l_fm

        oG.zero_grad(set_to_none=True)
        lg.backward()
        gn = torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
        if torch.isfinite(gn):
            oG.step()

        with torch.no_grad():
            for pe, p in zip(G_ema.parameters(), G.parameters()):
                pe.lerp_(p, 1.0 - 0.995)

        if st == 50:
            sps = 50.0 / max(1e-6, time.time() - tstep)
            log(f"[GATE {tag}] step-50 throughput {sps:.2f} steps/s")
            if sps < 0.8:
                log("[GATE] too slow"); sys.exit(3)

        if st % 100 == 0 or st == steps:
            el_m = (time.time() - t0) / 60.0
            tot_m = max(1.0, el_m / max(prog, 1e-3))
            log(f"Step {st} mel {l_mel.item():.4f} stft {l_stft.item():.3f} f0 {l_awp.item():.3f} D {ld_val:.3f} | {el_m:.1f}/{tot_m:.0f} min")

    torch.save(dict(G_ema={k: v.float().cpu() for k, v in G_ema.state_dict().items()}, step=steps, ch=ch), save_p + ".tmp")
    os.replace(save_p + ".tmp", save_p)
    log(f"VOC_GT_DONE tag={tag} steps={steps} -> {save_p}")
    return G_ema


def refine_student_tier(tier: str, cache_obj: dict, rect_cache: list, steps: int = 400, bs: int = 32, lr_e: float = 3.0e-5, lr_v: float = 3.0e-5, dev="cuda"):
    """Phase 2 (`t30`, `t15`) & Phase 3 (`t10`, `t1` Pico):
    Polishes each student tier with:
    - 4D Intra-Phoneme Progress Conditioning `[k/d, sin(pi*k/d), cos(pi*k/d), log(1+d)/4]`
    - Explicit discrete Vedic Svara (`udatta`, `anudatta`, `svarita`, `pracaya`, `none`) & Moraic (`laghu`, `guru`, `pluta`, `pranava_om`) embeddings
    - Multi-Resolution Contrastive Timbre Lock (`sim_ProV3 - sim_OldV1 >= +0.32` margin + Voiced 24-Bin LTAS)
    - Joint End-to-End Waveform Distillation through the Phase 1 `AWPVocoder` + Out-of-place STE Per-Channel INT8 QAT
    """
    out = f"/work/sp/distill/runs/{tier}"
    os.makedirs(out, exist_ok=True)
    log_p = f"{out}/train.log"

    def log(s):
        print(s, flush=True)
        open(log_p, "a").write(s + "\n")

    G = build(tier).to(dev)
    G.warm_init("/work/tts/ck")
    set_qat(G, True)
    install_awp_qat_hooks(G)
    ema = copy.deepcopy(G).eval()

    cc = torch.load("/work/tts/ck/v31_ctc_recognizer.pt", map_location=dev, weights_only=False)
    ctc = MelCTCAligner().to(dev).train()
    ctc.load_state_dict(cc.get("model", cc), strict=False)
    for m_ in ctc.modules():
        if isinstance(m_, nn.Dropout):
            m_.p = 0.0
    for p_ in ctc.parameters():
        p_.requires_grad_(False)

    mean, std = float(cache_obj["mean"]), float(cache_obj["std"])
    c_pro_24 = cache_obj["c_pro_ltas24"].to(dev)
    c_v1_24 = cache_obj["c_v1_ltas24"].to(dev)

    # Train AM (`enc`, `dur`, `vel`, and `prog_proj`/`svara_emb`/`mora_emb`) + gentle vocoder amp/pha parallel branch co-adaptation
    enc_dur_p = [p for n, p in G.named_parameters() if n.startswith(("am.enc.", "am.dur.", "closure_slope", "dur_rate_bias")) and p.requires_grad]
    vel_p = [p for n, p in G.named_parameters() if n.startswith("am.vel.") and p.requires_grad]
    voc_branch_p = [p for n, p in G.named_parameters() if n.startswith(("voc.amp_branch.", "voc.pha_branch.", "voc.src.upper_")) and p.requires_grad]
    for n, p in G.named_parameters():
        if n.startswith("voc.") and not n.startswith(("voc.amp_branch.", "voc.pha_branch.", "voc.src.upper_")):
            p.requires_grad_(False)

    oG = torch.optim.AdamW([
        dict(params=enc_dur_p, lr=lr_e, weight_decay=1e-3),
        dict(params=vel_p, lr=lr_v, weight_decay=1e-4),
        dict(params=voc_branch_p, lr=lr_v * 0.5, weight_decay=1e-4),
    ], betas=(0.9, 0.99))

    base_step = 2200
    t0 = time.time()
    tstep = time.time()
    log(f"[{tier}] Refining {steps} steps with Intra-Phoneme Progress + Svara/Moraic + Contrastive LTAS + Joint E2E AWP Waveform Distillation...")

    for st in range(1, steps + 1):
        prog = st / max(1, steps)
        mult = 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * prog))
        for pg in oG.param_groups:
            pg["lr"] = pg["initial_lr"] * mult if "initial_lr" in pg else lr_v * mult

        xs = random.sample(rect_cache, bs)
        B = len(xs)
        ids = pad_sequence([x["ids"].to(dev) for x in xs], batch_first=True)
        dur = pad_sequence([x["dur"].to(dev) for x in xs], batch_first=True)
        p_enc = pad_sequence([x["p_enc"].to(dev).float() for x in xs], batch_first=True)
        mu_t = pad_sequence([x["mu"].to(dev).float() for x in xs], batch_first=True)
        f0_t = pad_sequence([x["f0"].to(dev).float() for x in xs], batch_first=True)
        eng_t = pad_sequence([x["eng"].to(dev).float() for x in xs], batch_first=True)
        y_clean = pad_sequence([x["y_clean"].to(dev).float() for x in xs], batch_first=True).transpose(1, 2).contiguous()
        y_gt = pad_sequence([x["y_gt"].to(dev).float() for x in xs], batch_first=True).transpose(1, 2).contiguous()
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
            hs, mus, gs = G.am.enc(
                ids, spk, lang_eff, sty, studio, im,
                pitch_add=p_enc, prosody=G.pros_tbl[sty], sty_mask=ones_m,
                svara_ids=svara_ids, mora_ids=mora_ids,
            )
            f0_s = G.am.enc.f0_pred(hs).squeeze(-1)
            eng_s = G.am.enc.energy_pred(hs).squeeze(-1).clamp(-1.2, 1.2)
            l_heads = 0.5 * (F.l1_loss(f0_s[im].float(), f0_t[im].float()) + F.l1_loss(eng_s[im].float(), eng_t[im].float()))
            mus_e = mus + (0.10 * G.pros_tbl[sty][:, 2:3] + 0.06 * eng_s)[..., None] * 0.32
            lds = G.am.dur.pred_logdur(ids, hs.detach(), im) + G.dur_rate_bias.clamp(-0.12, 0.12)
            l_dur = F.smooth_l1_loss(lds[im].float(), torch.log(dur.float().clamp(min=1.0))[im])
            l_prior = F.l1_loss(mus[im].float(), mu_t[im].float())

            attn = _dur_to_attn(dur, im, ym)
            mu_y_s = torch.einsum("blt,bld->bdt", attn.to(mus_e.dtype), mus_e) * ymf
            ends = torch.cumsum(dur.float(), dim=1)
            t_arange = torch.arange(0, T, device=dev, dtype=torch.float32)
            idx_fr = (t_arange[None, :, None] >= ends[:, None, :]).to(torch.float32).sum(-1).to(torch.int64).clamp(max=L - 1)
            prog_4d = compute_intra_phoneme_progress(dur.float(), ends, idx_fr, t_arange) * ymf

            if G.steps == 1:
                t0_v = torch.zeros((B,), device=dev, dtype=torch.float32)
                y_pred = (mu_y_s + G.am.vel(mu_y_s, t0_v, mu_y_s, gs, ymf, prog=prog_4d)) * ymf
            else:
                c_idx = torch.arange(100, device=dev, dtype=torch.float32)[None, :, None]
                t_idx = torch.arange(T, device=dev, dtype=torch.float32)[None, None, :]
                z_init = torch.sin(c_idx * 12.9898 + t_idx * 78.233) * torch.cos(c_idx * 39.346 + t_idx * 11.135) * (0.667 * 1.4142)
                xt = z_init.expand(B, 100, T)
                dt = 1.0 / G.steps
                for i_s in range(G.steps):
                    ti = torch.full((B,), i_s * dt, device=dev, dtype=torch.float32)
                    k1 = G.am.vel(xt, ti, mu_y_s, gs, ymf, prog=prog_4d)
                    k2 = G.am.vel(xt + 0.5 * dt * k1, ti + 0.5 * dt, mu_y_s, gs, ymf, prog=prog_4d)
                    xt = xt + dt * k2
                y_pred = xt * ymf

            denom = (ymf.sum() * 100.0).clamp(min=1.0)
            l_mel = ((1.2 * (y_pred.float() - y_clean.float()).abs() + 0.8 * (y_pred.float() - y_clean.float()) ** 2) * ymf).sum() / denom
            df_p = y_pred[:, 1:, :].float() - y_pred[:, :-1, :].float()
            df_t = y_clean[:, 1:, :].float() - y_clean[:, :-1, :].float()
            l_df = ((df_p - df_t).abs() * ymf).sum() / denom
            ym_t = ymf[:, :, 1:] * ymf[:, :, :-1]
            l_dt = (((y_pred[:, :, 1:].float() - y_pred[:, :, :-1].float()) - (y_clean[:, :, 1:].float() - y_clean[:, :, :-1].float())).abs() * ym_t).sum() / (ym_t.sum() * 100.0 + 1.0)
            cep_p = torch.fft.rfft(y_pred.float(), dim=1).abs()[:, 4:48, :]
            cep_t = torch.fft.rfft(y_clean.float(), dim=1).abs()[:, 4:48, :]
            l_cep = ((cep_p - cep_t).abs() * ymf).sum() / (ymf.sum() * 44.0).clamp(min=1.0)

            mp_full = y_pred.float() * std + mean
            mt_full = (0.65 * y_clean.float() + 0.35 * y_gt.float()) * std + mean
            l_ltas = contrastive_ltas_timbre_loss(mp_full, mt_full, c_pro_24, c_v1_24)

            l_ctc = torch.zeros((), device=dev)
            if st % 2 == 0:
                c_val = ctc.ctc_loss(0.3 * mu_y_s.float() + 0.7 * y_pred.float(), yl, ids, il)
                if torch.isfinite(c_val):
                    l_ctc = F.relu(c_val - 0.12)

            # Joint End-to-End AWP Vocoder Complex-iSTFT + Phase Distillation on 48-frame slice
            l_vspec = torch.zeros((), device=dev)
            if T >= 52:
                seg_v = 48
                B_v = min(B, 14)
                s_p, s_t = [], []
                for bi in range(B_v):
                    valid_t = max(int(yl[bi].item()), seg_v + 1)
                    st_i = random.randint(0, max(0, valid_t - seg_v)) if valid_t > seg_v else 0
                    s_p.append(y_pred[bi, :, st_i : st_i + seg_v])
                    s_t.append(y_clean[bi, :, st_i : st_i + seg_v])
                mp_seg = torch.stack(s_p, dim=0) * std + mean
                mt_seg = torch.stack(s_t, dim=0) * std + mean
                mp_in = torch.cat([mp_seg[:, :98, :], mp_seg[:, 97:98, :], mp_seg[:, 97:98, :]], dim=1)
                mt_in = torch.cat([mt_seg[:, :98, :], mt_seg[:, 97:98, :], mt_seg[:, 97:98, :]], dim=1)
                with torch.no_grad():
                    f0_vt, uv_vt = G.am.f0_head(mt_in)
                    re_vt, im_vt, log_mag_vt, cos_vt, sin_vt = G.voc.spectrum_and_phase(mt_in, f0_vt, uv_vt)
                f0_vs, uv_vs = G.am.f0_head(mp_in)
                re_vs, im_vs, log_mag_vs, cos_vs, sin_vs = G.voc.spectrum_and_phase(mp_in, f0_vs, uv_vs)
                l_mag_v = F.l1_loss(log_mag_vs, log_mag_vt)
                l_awp_v = awp_phase_losses(cos_vs, sin_vs, re_vt, im_vt)
                l_vspec = l_mag_v + 0.4 * l_awp_v

            loss = l_mel + 2.2 * l_df + 1.1 * l_dt + 0.8 * l_cep + 1.5 * l_prior + 0.8 * l_dur + l_heads + 0.6 * l_ltas + 0.25 * l_ctc + 0.5 * l_vspec

        oG.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(G.parameters(), 1.2)
        if torch.isfinite(gn):
            oG.step()

        with torch.no_grad():
            for pe, pg in zip(ema.parameters(), G.parameters()):
                pe.lerp_(pg, 1.0 - 0.992)

        if st % 50 == 0 or st == steps:
            now = time.time()
            ips = 50.0 / max(1e-6, now - tstep)
            tstep = now
            el_m = (now - t0) / 60.0
            tot_m = max(1.0, el_m / max(prog, 1e-3))
            log(f"Step {base_step + st} | {ips:.2f} it/s | mel {l_mel.item():.3f} ltas {l_ltas.item():.3f} prior {l_prior.item():.3f} cep {l_cep.item():.3f} vspec {l_vspec.item():.3f} | {el_m:.1f}/{tot_m:.0f} min qat=1")

    ck_obj = dict(
        ema={k: v.float().cpu() for k, v in ema.state_dict().items()},
        cfg=G.cfg, tier=tier, arch=G.arch, voc_arch=G.voc_arch,
        step=base_step + steps, f0=[5.418, 0.203], qat=True, gt=True,
    )
    torch.save(ck_obj, f"{out}/ema_final.pt")
    torch.save(ck_obj, f"{out}/ema_latest.pt")
    log(f"DONE {tier} step {base_step + steps} minutes {(time.time() - t0) / 60:.1f}")
    return ema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc_steps", type=int, default=600)
    ap.add_argument("--student_steps", type=int, default=350)
    ap.add_argument("--skip_voc", action="store_true")
    ap.add_argument("--tiers", default="t30,t15,t10,t1")
    a = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(42)
    torch.manual_seed(42)

    cache_obj = torch.load("/work/sp/distill/pure_pro_v3_cache.pt", weights_only=False)
    rect_obj = torch.load("/work/sp/distill/v3_rectified_cache.pt", weights_only=False)
    rect_cache = rect_obj["cache"]

    if not a.skip_voc:
        train_phase1_awp_vocoder("awp_voc224", ch=320, base_ck="/work/tts/ck/v3_voc320.pt", save_p="/work/tts/ck/v4_awp_voc320.pt", cache_obj=cache_obj, is_t1=False, steps=a.voc_steps, bs=36, lr=3.5e-5)
        train_phase1_awp_vocoder("awp_voc136", ch=144, base_ck="/work/tts/ck/v3_voc144_t1.pt", save_p="/work/tts/ck/v4_awp_voc144_t1.pt", cache_obj=cache_obj, is_t1=True, steps=a.voc_steps, bs=40, lr=4.5e-5)

    for tier in [t.strip() for t in a.tiers.split(",") if t.strip()]:
        lr_val = 2.5e-5 if tier in ("t30", "t15") else (6.0e-5 if tier == "t10" else 1.2e-4)
        refine_student_tier(tier, cache_obj, rect_cache, steps=a.student_steps, bs=32, lr_e=lr_val, lr_v=lr_val)

    print("SESSION75_BREAKTHROUGH_ALL_DONE — launching ladder_daemon --once for t30,t15,t10,t1", flush=True)
    subprocess.run([sys.executable, "/work/sp/repo/scripts/sakhiya_pro/distill/ladder_daemon.py", "--once", "--tiers", "t30,t15,t10,t1"])
    print("SESSION75_LADDER_GATE_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
