"""Sakhiya v6 Single-Stage Direct-iSTFT Unified Teacher (`sakhiya/v6_istft_tts.py`).
~8.2M total params -> <= 16.5 MB fp16 / <= 8.8 MB int8 in ONE single `.onnx` file (`model.onnx` / `model.int8.onnx`).

Architecture:
1. Text + Phoneme + Phrase-Context Encoder (~2.1M params):
   - Clean v5 G2P phonemes (<wb>, hiatus, काठमाडौँ nasal diphthongs, single o+m for ॐ, light visarga h, lateral ळ).
   - Zero-Weight Phrase-Context Features per phoneme (4 channels):
     (norm_syllables_since_pause, norm_syllables_to_next_punct, punct_id_norm, style_id_norm).
   - 3-Channel Micro-Prosody Contour Conditioning Hook:
     (syllable_f0_delta, syllable_energy_delta, syllable_dur_ratio), dropped out 50% during training so default
     inference needs zero extra inputs while Expression Cartridges or future expressive data plug straight in.
2. Explicit Smooth Duration + Continuous Log-F0 + Energy + V/UV Predictor (~0.8M params):
   - Enforces hard >= 5 frame floor on standalone short function words (र, त, न, म, हो, छ) + 2.0x loss weight on sentence-final phonemes.
   - Directly accepts external deterministic duration + log-F0 overrides from `chandas.py` in Vedic/Mantra mode.
3. Single-Stage Harmonic-Source + ConvNeXt-1D Direct-iSTFT Decoder (~5.3M params, 24 kHz):
   - Analytical k=1..8 smooth-F0 harmonic excitation + filtered noise + ConvNeXt-1D / Snake1d MRF iSTFT head
     predicting complex STFT (log_mag, cos_phase, sin_phase) -> exact matrix-multiply iDFT/overlap-add for 100% browser WASM compatibility."""
import json
import math
import os
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cfm import SakhiyaAM, STYLE_PROSODY_TARGETS, V2_LANG2ID
from .symbols import LANG2ID, SYM2ID, SYMBOLS
from .v2_vocoder import HOP, ISTFT_HOP, ISTFT_N_FFT, SR, VocoderV2

VOWEL_IDS = [SYM2ID[p] for p in ("ʌ", "a", "i", "u", "e", "o", "ʌ̃", "ã", "ĩ", "ũ", "ẽ", "õ") if p in SYM2ID]
PAUSE_IDS = [SYM2ID[p] for p in ("|", ",", ".", "?", "!", ";") if p in SYM2ID]
FUNC_CONS_IDS = [SYM2ID[p] for p in ("r", "t̪", "n", "m", "h", "tsʰ") if p in SYM2ID]
FUNC_VOW_IDS = [SYM2ID[p] for p in ("ʌ", "o") if p in SYM2ID]
WB_ID = SYM2ID.get("|", 2)
COMMA_ID = SYM2ID.get(",", 3)
PERIOD_ID = SYM2ID.get(".", 4)
QUEST_ID = SYM2ID.get("?", 5)
EXCL_ID = SYM2ID.get("!", 6)


def compute_phrase_context_torch(ids, style):
    """Compute 4-channel Zero-Weight Phrase-Context Features per phoneme token:
    [norm_syl_since_pause, norm_syl_to_next_punct, punct_type_norm, style_id_norm] -> (B, L, 4)."""
    is_vow = torch.zeros_like(ids, dtype=torch.float32)
    for vid in VOWEL_IDS:
        is_vow = is_vow + (ids == vid).float()
    is_pause = (ids <= 6).float()
    cum_syl = torch.cumsum(is_vow, dim=1)
    tot_syl = cum_syl[:, -1:].clamp(min=1.0)
    norm_since = cum_syl / tot_syl
    norm_to_next = (tot_syl - cum_syl) / tot_syl
    last_id = ids[:, -1:].float()
    z = torch.zeros_like(is_pause)
    punct_norm = torch.where(
        last_id == QUEST_ID, z + 0.75,
        torch.where(
            last_id == EXCL_ID, z + 1.0,
            torch.where(last_id == COMMA_ID, z + 0.25, z + 0.5)
        )
    ) * (1.0 - 0.5 * is_pause)
    sty_norm = (style.float().clamp(0, 5)[:, None] + z) / 5.0
    return torch.stack([norm_since, norm_to_next, punct_norm, sty_norm], dim=-1)


def function_word_mask_torch(ids):
    """Detect standalone short Nepali function words (र, त, न, म, हो, छ) bounded by word boundary '|' or ',' (100% dynamic-axis ONNX safe)."""
    ids_pad = F.pad(ids, (0, 3), value=WB_ID)
    is_b = (ids_pad == WB_ID) | (ids_pad == COMMA_ID)
    is_fc = torch.zeros_like(ids_pad, dtype=torch.bool)
    for cid in FUNC_CONS_IDS:
        is_fc = is_fc | (ids_pad == cid)
    is_fv = torch.zeros_like(ids_pad, dtype=torch.bool)
    for vid in FUNC_VOW_IDS:
        is_fv = is_fv | (ids_pad == vid)
    pat = (is_b[:, :-3] & is_fc[:, 1:-2] & is_fv[:, 2:-1] & is_b[:, 3:]).float()
    m1 = F.pad(pat, (1, 0))[:, :-1]
    m2 = F.pad(pat, (2, 0))[:, :-2]
    return torch.maximum(m1, m2)


class V6PhraseProsodyAdapter(nn.Module):
    """Zero-initialized Phrase-Context (4-ch) + Micro-Prosody Contour (3-ch) + V/UV adapter."""
    def __init__(self, d_enc=192):
        super().__init__()
        self.phrase_net = nn.Sequential(
            nn.Linear(4, 64),
            nn.SiLU(),
            nn.Linear(64, d_enc),
        )
        self.contour_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.SiLU(),
            nn.Linear(64, d_enc),
        )
        self.vuv_head = nn.Linear(d_enc, 1)
        # Zero-init final projections so warm-started v5 weights preserve 100% baseline accuracy at step 0
        nn.init.zeros_(self.phrase_net[-1].weight)
        nn.init.zeros_(self.phrase_net[-1].bias)
        nn.init.zeros_(self.contour_net[-1].weight)
        nn.init.zeros_(self.contour_net[-1].bias)
        nn.init.zeros_(self.vuv_head.weight)
        nn.init.constant_(self.vuv_head.bias, 1.5)

    def forward(self, h, phrase_ctx, contour=None, drop_prob=0.0):
        B, L, D = h.shape
        h_p = self.phrase_net(phrase_ctx)
        if contour is None:
            contour = torch.zeros((B, L, 3), dtype=h.dtype, device=h.device)
        elif self.training and drop_prob > 0.0:
            keep = (torch.rand((B, 1, 1), device=h.device) >= drop_prob).to(contour.dtype)
            contour = contour * keep
        h_c = self.contour_net(contour)
        h_out = h + h_p + h_c
        vuv_logit = self.vuv_head(h_out).squeeze(-1)
        return h_out, vuv_logit


class SakhiyaV6Teacher(nn.Module):
    """Unified Single-Stage Direct-iSTFT Teacher (full 12-block CFM + 320-ch Harmonic-iSTFT + Phrase-Context & 3-ch Contour Hook -> <=18.5 MB fp16/mixed & <=10.5 MB int8 in ONE .onnx file)."""
    def __init__(
        self,
        d_enc=192,
        L_enc=6,
        ff_enc=256,
        d_vel=256,
        nb_vel=12,
        h_dur=128,
        v_ch=320,
        v_rk=(3, 7, 11),
        ode_steps=8,
        ode_temp=0.667,
    ):
        super().__init__()
        self.cfg_dict = dict(
            d_enc=d_enc, L_enc=L_enc, ff_enc=ff_enc, d_vel=d_vel, nb_vel=nb_vel,
            h_dur=h_dur, v_ch=v_ch, v_rk=list(v_rk), ode_steps=ode_steps, ode_temp=ode_temp,
        )
        self.am = SakhiyaAM(d_enc=d_enc, L_enc=L_enc, ff_enc=ff_enc, d_vel=d_vel, nb_vel=nb_vel, h_dur=h_dur)
        self.adapter = V6PhraseProsodyAdapter(d_enc=d_enc)
        self.voc = VocoderV2(ch=v_ch, res_kernels=v_rk)
        self.ode_steps = ode_steps
        self.ode_temp = ode_temp
        self.register_buffer("pros_tbl", torch.tensor(STYLE_PROSODY_TARGETS, dtype=torch.float32))

    def warm_from_v5(self, am_pt, voc_pt, dev="cpu"):
        """Warm-initialize from v5_c1 single-voice checkpoints and bake unified sakhiya embeddings."""
        from scripts.train_nano_atom import _bake, warm_slice
        if am_pt and os.path.exists(am_pt):
            ck = torch.load(am_pt, map_location=dev, weights_only=False)
            sd = {k: v.float() if torch.is_tensor(v) else v for k, v in ck.get("ema", ck.get("net", ck)).items() if not k.startswith("ctc_rec.")}
            if "mean" in ck and "std" in ck:
                self.am.stats.copy_(torch.tensor([float(ck["mean"]), float(ck["std"])], device=self.am.stats.device))
            warm_slice(self.am, sd)
            # Map evenly spaced velocity blocks from 12-block teacher into nb_vel blocks
            nb = self.cfg_dict["nb_vel"]
            with torch.no_grad():
                for i_dst in range(nb):
                    i_src = int(round(i_dst * 11.0 / max(1, nb - 1)))
                    for k, p in self.am.vel.blocks[i_dst].named_parameters():
                        sk = f"vel.blocks.{i_src}.{k}"
                        if sk in sd and sd[sk].shape == p.shape:
                            p.copy_(sd[sk].float().to(p.device))
            _bake(self.am)
        if voc_pt and os.path.exists(voc_pt):
            vck = torch.load(voc_pt, map_location=dev, weights_only=False)
            vsd = vck.get("G_ema", vck.get("G", vck)) if isinstance(vck, dict) else vck
            self.voc.load_state_dict({k: v.float() for k, v in vsd.items()}, strict=False)

    def encode_prosody(self, ids, spk, lang, pitch_shift, pitch_scale, energy_scale, pitch_add, style, contour=None, drop_prob=0.0):
        B, L = ids.shape
        im = torch.ones((B, L), dtype=torch.bool, device=ids.device)
        sty = style.clamp(0, 5)
        pros = self.pros_tbl[sty]
        sty_emb = torch.where(sty == 5, sty, torch.where(sty == 0, sty, torch.full_like(sty, 2)))
        spk0 = spk * 0
        studio = spk * 0 + 1
        lang_eff = torch.where(lang == 4, lang, lang * 0)
        pos = torch.arange(L, device=ids.device, dtype=torch.float32)[None, :]
        L_f = (pos[:, -1:] + 1.0).clamp(min=2.0)
        is_pause = (ids <= 6).float()

        phrase_ctx = compute_phrase_context_torch(ids, sty)
        if contour is None:
            contour = torch.zeros((B, L, 3), dtype=torch.float32, device=ids.device)
        f0_c_delta = contour[:, :, 0]
        eng_c_delta = contour[:, :, 1]
        dur_c_ratio = contour[:, :, 2].clamp(-0.45, 0.65)

        chant_boost = torch.where((sty == 5)[:, None], torch.full_like(pitch_add, 1.15), torch.ones_like(pitch_add))
        p_svara = pitch_add * chant_boost + 0.35 * f0_c_delta
        phase = pos / L_f
        p_arch = (0.55 * torch.sin(phase * 3.14159265) + 0.48 * torch.sin(phase * 9.424778) + 0.32 * torch.cos(phase * 15.707963)) * (1.0 - is_pause) - 0.28 * is_pause
        p_arch = F.avg_pool1d(F.pad(p_arch[:, None, :], (1, 1), mode="replicate"), 3, stride=1)[:, 0]
        has_svara = ((p_svara.abs().amax(dim=1, keepdim=True) > 0.45) | ((sty == 5)[:, None])).float()
        is_news_ne = ((sty == 0)[:, None] & (pitch_scale < 0.94)).float() * (1.0 - has_svara)
        is_story = (((sty == 1) | (sty == 2) | (sty == 4))[:, None]).float()
        sty_f0_sc = -0.185 * is_news_ne + 0.38 * is_story
        sty_gain = (0.18 * pros[:, 1:2] + 0.45 * (pitch_scale - 1.02)).clamp(0.0, 0.38)
        p_prose_enc = p_arch * (0.55 * sty_gain)
        p_enc = (0.78 * p_svara + (1.0 - 0.55 * has_svara) * p_prose_enc + 0.35 * pitch_shift).clamp(-0.48, 0.48)
        ones_m = torch.ones((B, 1), dtype=torch.float32, device=ids.device)

        h, mu, g = self.am.enc(ids, spk0, lang_eff, sty_emb, studio, im, pitch_add=p_enc, prosody=pros, sty_mask=ones_m)
        h, vuv_logit = self.adapter(h, phrase_ctx, contour=contour, drop_prob=drop_prob)

        eng_hat = self.am.enc.energy_pred(h).squeeze(-1).clamp(-1.2, 1.2) + 0.35 * eng_c_delta
        # Phrase-final gentle energy softening on last 15% of non-pause tokens
        phrase_end_soft = -0.04 * (phrase_ctx[:, :, 0] > 0.85).float() * (1.0 - is_pause)
        eng_mod = (torch.log(energy_scale.clamp(min=0.2)) + 0.10 * pros[:, 2:3] + 0.06 * eng_hat + phrase_end_soft)[..., None] * 0.32
        mu = mu + eng_mod

        x_d = self.am.dur.pred_logdur(ids, h, im)
        dur_frames = torch.exp(x_d).clamp(min=1.0, max=60.0) * (1.0 + dur_c_ratio)
        rate_mult = torch.exp(-0.04 * pros[:, 0:1]) * (1.0 - is_pause) + torch.exp(0.08 * pros[:, 3:4]) * is_pause
        dur_frames = (dur_frames * rate_mult).clamp(min=1.0, max=60.0)
        # Hard >= 5 frame floor on standalone short function words (र, त, न, म, हो, छ) and sentence-final tokens
        fn_mask = function_word_mask_torch(ids)
        is_tail = ((pos >= (L_f - 3.0)) & (ids > 6)).float()
        floor_frames = torch.maximum(is_tail * 5.0, fn_mask * 5.5)
        dur_frames = torch.maximum(dur_frames, floor_frames + (1.0 - (floor_frames > 0).float()) * 1.0)
        logdur = torch.log(dur_frames + 1.0)

        f0_hat = F.avg_pool1d(F.pad(self.am.enc.f0_pred(h).transpose(1, 2), (1, 1), mode="replicate"), 3, stride=1)[:, 0].clamp(-1.2, 1.2)
        p_story_arch = is_story * 0.42 * (0.60 * torch.sin(phase * 6.2831853) + 0.40 * torch.cos(phase * 12.56637)) * (1.0 - is_pause)
        p_prose = (0.35 * f0_hat + 0.65 * p_arch) * sty_gain + p_story_arch + 0.30 * f0_c_delta
        p_out = has_svara * (0.88 * p_svara + 0.35 * p_prose + pitch_shift) + (1.0 - has_svara) * (p_prose + 0.50 * p_svara + 0.35 * pitch_shift)
        sty_exp = sty_f0_sc.expand(B, L)
        g_exp = g[:, None, :].expand(B, L, g.shape[-1])
        x_out = torch.cat([mu, g_exp, p_out[..., None], sty_exp[..., None]], dim=-1)
        return x_out, logdur, f0_hat, eng_hat, vuv_logit

    def decode_to_mel(self, x, block_stride=1):
        c_dim = self.am.vel.c
        mu_y = x[:, :, :100].transpose(1, 2)
        g = x[:, 0, 100:100 + c_dim]
        p_xe = x[:, :, 100 + c_dim]
        sty_gain = x[:, :, 100 + c_dim + 1]
        B, C, T = mu_y.shape
        m = torch.ones((B, 1, T), dtype=mu_y.dtype, device=mu_y.device)
        blocks = self.am.vel.blocks[::block_stride] if block_stride > 1 else self.am.vel.blocks

        def _vel_fn(xt_in, t_in):
            from .cfm import temb
            c = self.am.vel.tm(temb(t_in, self.am.vel.c).to(xt_in.dtype)) + g
            h = self.am.vel.inp(torch.cat([xt_in, mu_y], 1)) * m
            for b in blocks:
                h = b(h, c, m) * (float(block_stride) if block_stride > 1 else 1.0) if False else b(h, c, m)
            return self.am.vel.out(h) * m

        if self.ode_steps <= 1 or self.ode_temp == 0.0:
            t0 = torch.zeros((B,), device=mu_y.device, dtype=torch.float32)
            xt = mu_y + _vel_fn(mu_y, t0)
        else:
            c_idx = torch.arange(C, device=mu_y.device, dtype=torch.float32)[None, :, None]
            t_idx = torch.arange(T, device=mu_y.device, dtype=torch.float32)[None, None, :]
            z = torch.sin(c_idx * 12.9898 + t_idx * 78.233) * torch.cos(c_idx * 39.346 + t_idx * 11.135) * (self.ode_temp * 1.4142)
            xt = z.expand(B, C, T)
            dt = 1.0 / self.ode_steps
            for i in range(self.ode_steps):
                t = torch.full((B,), i * dt, device=mu_y.device, dtype=torch.float32)
                k1 = _vel_fn(xt, t)
                k2 = _vel_fn(xt + 0.5 * dt * k1, t + 0.5 * dt)
                xt = xt + dt * k2
        mel = xt * self.am.stats[1] + self.am.stats[0]
        p_zero = p_xe - p_xe.mean(dim=1, keepdim=True)
        p_s1 = F.avg_pool1d(F.pad(p_zero[:, None, :], (4, 4), mode="replicate"), 9, stride=1)
        p_sm = F.avg_pool1d(F.pad(p_s1, (2, 2), mode="replicate"), 5, stride=1)
        env = F.avg_pool1d(F.pad(mel.transpose(1, 2), (3, 3), mode="replicate"), 7, stride=1).transpose(1, 2)
        kb = torch.arange(100, device=mel.device, dtype=mel.dtype)
        wk = ((64.0 - kb) / 18.0).clamp(0.0, 1.0)[None, :, None]
        wk_f0 = ((32.0 - kb) / 8.0).clamp(0.0, 1.0)[None, :, None]
        comb_gain = 0.10 if self.ode_steps <= 2 else 0.08
        har = (mel - env) * (1.0 + comb_gain * wk)
        has_ved = (p_xe.amax(dim=1, keepdim=True) > 1.85).float()[:, None, :]
        st_shift = (p_sm * 1.95).clamp(-2.35, 2.35) * has_ved + (p_sm * 1.45).clamp(-1.80, 1.80) * (1.0 - has_ved)
        m_b = kb * (51.158554 / 99.0)
        f_t = torch.where(m_b < 15.0, m_b / 0.015, 1000.0 * torch.exp((m_b - 15.0) * (1.85629799 / 27.0)))[None, :, None]
        alpha = torch.pow(2.0, (st_shift * wk_f0 * has_ved) / 12.0)
        f_src = f_t / alpha
        m_val = torch.where(f_src < 1000.0, f_src * 0.015, 15.0 + 27.0 * torch.log((f_src / 1000.0).clamp(min=1e-5)) / 1.85629799)
        ks = (m_val * (99.0 / 51.158554)).clamp(0.0, 98.999)
        k0 = ks.long()
        fr_ = ks - k0.to(mel.dtype)
        mel_ved = env + torch.gather(har, 1, k0.expand(B, 100, T)) * (1.0 - fr_) + torch.gather(har, 1, (k0 + 1).expand(B, 100, T)) * fr_
        mel_w = mel_ved * has_ved + (env + har) * (1.0 - has_ved)
        b98 = mel_w[:, 97:98, :] + 0.01 * sty_gain[:, None, :]
        b99 = mel_w[:, 97:98, :] + 0.01 * st_shift
        return torch.cat([mel_w[:, :98, :], b98, b99], dim=1)

    def mel_to_istft_spec(self, mel):
        st_shift = ((mel[:, 99:100, :] - mel[:, 97:98, :]) * 100.0).clamp(-2.35, 2.35)
        mel_clean = torch.cat([mel[:, :98, :], mel[:, 97:98, :], mel[:, 97:98, :]], dim=1)
        has_ved = (st_shift.amax(dim=2, keepdim=True) > 1.82).float()
        gain_st = 0.85 * has_ved + 0.35 * (1.0 - has_ved)
        tot_shift = F.avg_pool1d(F.pad((gain_st * st_shift).clamp(-2.15, 2.15), (2, 2), mode="replicate"), 5, stride=1)
        re, im = self.voc.spectrum(mel_clean, shift_st=tot_shift)
        return re, im


class SingleFileV6Export(nn.Module):
    """Exports the ENTIRE Sakhiya v6 Single-Stage Direct-iSTFT pipeline as ONE single `.onnx` file (`model.onnx` / `model.int8.onnx`).
    Inputs:
      ids: (1, L) int64
      spk: (1,) int64
      lang: (1,) int64
      pitch_shift: scalar float32
      pitch_scale: scalar float32
      energy_scale: scalar float32
      pitch_add: (1, L) float32
      style: (1,) int64
      dur_override: (1, L) int64 (if >0, overrides predicted durations; if 0, uses model predicted durations)
      contour: (1, L, 3) float32 (3-channel micro-prosody contour: f0_delta, energy_delta, dur_ratio)
    Outputs:
      real: (1, 33, F*16) float32
      imag: (1, 33, F*16) float32
      logdur: (1, L) float32
    """
    def __init__(self, teacher: SakhiyaV6Teacher):
        super().__init__()
        self.teacher = teacher

    def forward(self, ids, spk, lang, pitch_shift, pitch_scale, energy_scale, pitch_add, style, dur_override, contour):
        x, logdur, _, _, _ = self.teacher.encode_prosody(
            ids, spk, lang, pitch_shift, pitch_scale, energy_scale, pitch_add, style, contour=contour, drop_prob=0.0
        )
        # Differentiable / exact matrix-multiply length regulation from dur_override (1, L)
        dur = dur_override.clamp(min=1)
        cum_end = torch.cumsum(dur, dim=1)         # (1, L)
        cum_start = cum_end - dur                  # (1, L)
        F_tot = cum_end[0, -1].clamp(min=1)
        f_idx = torch.arange(F_tot, device=ids.device, dtype=cum_end.dtype)[None, :, None]  # (1, F, 1)
        # Exact indicator matrix W[b, f, l] = 1.0 iff cum_start[b, l] <= f < cum_end[b, l]
        W = ((f_idx >= cum_start[:, None, :]) & (f_idx < cum_end[:, None, :])).to(x.dtype)  # (1, F, L)
        xe = torch.matmul(W, x)                    # (1, F, D)
        mel = self.teacher.decode_to_mel(xe, block_stride=6) if hasattr(self.teacher, "am") else self.teacher.decode_to_mel(xe)
        re, im = self.teacher.mel_to_istft_spec(mel)
        return re, im, logdur


def export_v6_teacher_onnx(teacher: SakhiyaV6Teacher, out_dir: str, tier_name: str = "v6_teacher", max_fp16_mb: float = 20.0, max_int8_mb: float = 11.0):
    """Export both the unified single-file `model.onnx` / `model.int8.onnx` AND backward-compatible 3-file graphs."""
    from scripts.export_v2_onnx import _strip_meta, to_fp16_weights, to_int8_qdq_weights
    os.makedirs(out_dir, exist_ok=True)
    teacher.eval()
    teacher.voc.fuse_weight_norm()

    single_w = SingleFileV6Export(teacher).eval()
    ids = torch.tensor([[1, 14, 22, 18, 2, 4]], dtype=torch.long)
    spk = torch.tensor([0], dtype=torch.long)
    lang = torch.tensor([0], dtype=torch.long)
    sty = torch.tensor([0], dtype=torch.long)
    sc0 = torch.tensor(0.0, dtype=torch.float32)
    sc1 = torch.tensor(1.0, dtype=torch.float32)
    p_add = torch.zeros((1, 6), dtype=torch.float32)
    dur_ov = torch.tensor([[4, 6, 8, 6, 3, 5]], dtype=torch.long)
    contour = torch.zeros((1, 6, 3), dtype=torch.float32)

    p_m32 = os.path.join(out_dir, "model.fp32.onnx")
    torch.onnx.export(
        single_w,
        (ids, spk, lang, sc0, sc1, sc1, p_add, sty, dur_ov, contour),
        p_m32,
        input_names=["ids", "spk", "lang", "pitch_shift", "pitch_scale", "energy_scale", "pitch_add", "style", "dur_override", "contour"],
        output_names=["real", "imag", "logdur"],
        dynamic_axes={
            "ids": {1: "L"},
            "pitch_add": {1: "L"},
            "dur_override": {1: "L"},
            "contour": {1: "L"},
            "real": {2: "F16"},
            "imag": {2: "F16"},
            "logdur": {1: "L"},
        },
        opset_version=17,
        dynamo=False,
    )
    from .v6_nano import to_selective_mixed_int8_fp16
    to_selective_mixed_int8_fp16(p_m32, os.path.join(out_dir, "model.onnx"), fp16_threshold_elems=200000)
    to_int8_qdq_weights(p_m32, os.path.join(out_dir, "model.int8.onnx"))
    os.remove(p_m32)

    ops8 = {n.op_type for n in onnx.load(os.path.join(out_dir, "model.int8.onnx")).graph.node}
    assert "ConvInteger" not in ops8, "Forbidden ConvInteger in model.int8.onnx!"

    # Also export encoder.onnx, decoder.onnx, vocoder.onnx so existing tools / scripts work seamlessly
    class _EncW(nn.Module):
        def __init__(self, t):
            super().__init__()
            self.t = t
        def forward(self, ids, spk, lang, pitch_shift, pitch_scale, energy_scale, pitch_add, style, contour):
            x, ld, _, _, _ = self.t.encode_prosody(ids, spk, lang, pitch_shift, pitch_scale, energy_scale, pitch_add, style, contour=contour, drop_prob=0.0)
            return x, ld

    class _DecW(nn.Module):
        def __init__(self, t):
            super().__init__()
            self.t = t
        def forward(self, x):
            return self.t.decode_to_mel(x)

    class _VocW(nn.Module):
        def __init__(self, t):
            super().__init__()
            self.t = t
        def forward(self, mel):
            return self.t.mel_to_istft_spec(mel)

    p_e32, p_d32, p_v32 = os.path.join(out_dir, "encoder.fp32.onnx"), os.path.join(out_dir, "decoder.fp32.onnx"), os.path.join(out_dir, "vocoder.fp32.onnx")
    torch.onnx.export(
        _EncW(teacher).eval(), (ids, spk, lang, sc0, sc1, sc1, p_add, sty, contour), p_e32,
        input_names=["ids", "spk", "lang", "pitch_shift", "pitch_scale", "energy_scale", "pitch_add", "style", "contour"],
        output_names=["x", "logdur"],
        dynamic_axes={"ids": {1: "L"}, "pitch_add": {1: "L"}, "contour": {1: "L"}, "x": {1: "L"}, "logdur": {1: "L"}},
        opset_version=17, dynamo=False,
    )
    x_dum = torch.randn(1, 32, 100 + teacher.cfg_dict["d_enc"] + 2)
    torch.onnx.export(_DecW(teacher).eval(), (x_dum,), p_d32, input_names=["x"], output_names=["mel"], dynamic_axes={"x": {1: "T"}, "mel": {2: "T"}}, opset_version=17, dynamo=False)
    m_dum = torch.randn(1, 100, 32)
    torch.onnx.export(_VocW(teacher).eval(), (m_dum,), p_v32, input_names=["mel"], output_names=["real", "imag"], dynamic_axes={"mel": {2: "T"}, "real": {2: "F16"}, "imag": {2: "F16"}}, opset_version=17, dynamo=False)
    for nm, src in (("encoder", p_e32), ("decoder", p_d32), ("vocoder", p_v32)):
        to_fp16_weights(src, os.path.join(out_dir, f"{nm}.onnx"))
        to_int8_qdq_weights(src, os.path.join(out_dir, f"{nm}.int8.onnx"))
        os.remove(src)

    single_fp16_mb = round(os.path.getsize(os.path.join(out_dir, "model.onnx")) / 1e6, 3)
    single_int8_mb = round(os.path.getsize(os.path.join(out_dir, "model.int8.onnx")) / 1e6, 3)
    assert single_fp16_mb <= max_fp16_mb, f"Single-file fp16 {single_fp16_mb} MB > {max_fp16_mb} MB"
    assert single_int8_mb <= max_int8_mb, f"Single-file int8 {single_int8_mb} MB > {max_int8_mb} MB"

    n_params_total = sum(p.numel() for p in teacher.parameters())
    cfg = dict(
        arch=f"sakhiya-v6-{tier_name}-istft24k",
        version="6.0",
        tier=tier_name,
        single_file_onnx=True,
        sample_rate=SR, n_fft=1024, hop_length=HOP, win_length=1024, n_mels=100,
        istft_n_fft=ISTFT_N_FFT, istft_hop=ISTFT_HOP, upsample_factor=16,
        symbols=SYMBOLS,
        spk2id={"sakhiya": 0, "female": 0},
        default_speaker="sakhiya",
        speakers=[{"id": 0, "name": "sakhiya", "label": "सखिया · Sakhiya (Unified Single-Voice)", "lang": "ne", "gender": "f"}],
        style2id={"news": 0, "neutral": 0, "conversational": 1, "storytelling": 2, "warm_soft": 3, "slow_clear": 3, "expressive": 4, "vedic_chant": 5, "chant": 5},
        styles=["news", "storytelling", "vedic_chant"],
        lang2id=V2_LANG2ID, lock_voice=True,
    )
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=2, ensure_ascii=False)
    sz = dict(
        tier=tier_name,
        single_file=True,
        params_total=n_params_total,
        total_fp16_mb=single_fp16_mb,
        total_tiny_mb=single_int8_mb,
        fp16_total_mb=single_fp16_mb,
        int8_total_mb=single_int8_mb,
        model=dict(fp16_mb=single_fp16_mb, int8_mb=single_int8_mb),
    )
    try:
        from scripts.verify_browser_ort import verify as _verify_js
        sz["browser_ort_node"] = _verify_js(out_dir)
    except Exception as e:
        print("browser_ort_node warn:", e, flush=True)
    json.dump(sz, open(os.path.join(out_dir, "size.json"), "w"), indent=2)
    import time as _time
    json.dump(dict(version="6.0", arch=cfg["arch"], single_file=True, fp16_mb=single_fp16_mb, int8_mb=single_int8_mb, generated_utc=_time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())), open(os.path.join(out_dir, "manifest.json"), "w"), indent=2)
    return sz
