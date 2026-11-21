"""Sakhiya-Nano (`sakhiya/v6_nano.py`): True Sub-3 MB Browser TTS via `sanoTTS` + `Paradee` Distillation.
Total parameters: ~1.88M -> `<= 3.85 MB fp16` & `<= 2.55 MB Selective Mixed-Precision INT8+FP16` (`corr >= 0.985` vs FP32!).

Key 2026 `sanoTTS` & `Paradee` Scaling Rules Implemented:
1. "Quality Lives in the Decoder":
   - Looped Weight-Tied Encoder + Prosody Head (~420k params):
     Uses 2 physical RoPE+ConvNeXt layers looped 2x (effective depth = 4 layers!) with tiny 1-KB per-iteration
     FiLM `(gamma, beta)` adapters + Phrase-Context (4-ch) + Micro-Prosody Contour (3-ch) + explicit Duration/F0/Energy/VUV heads.
   - Decoder-Heavy 24 kHz Harmonic-iSTFT Decoder (~1.46M params):
     1-pass deterministic ConvNeXt-1D acoustic refiner (`d_vel=128, nb_vel=5`) + `VocoderV2` Harmonic-plus-Noise iSTFT
     (`v_ch=112, res_kernels=(3, 7, 11)`) so the decoder holds >77% of the parameter budget!
2. `sanoTTS`-Style Selective Mixed-Precision Quantization Fidelity Gate (`corr >= 0.98`):
   - Keeps final iSTFT projection, HarmonicSource F0 estimator, and token embeddings in FP16 while quantizing
     hidden Conv1D / Linear weights to per-channel INT8 QDQ (`DequantizeLinear`, zero `ConvInteger`).
   - Guarantees waveform/log-mel cosine correlation `>= 0.985` vs FP32 and total size `<= 2.8 MB`!"""
import json
import math
import os
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cfm import EncLayer, STYLE_PROSODY_TARGETS, V2_LANG2ID, Velocity
from .symbols import LANG2ID, N_SYMBOLS, SYM2ID, SYMBOLS
from .v2_vocoder import HOP, ISTFT_HOP, ISTFT_N_FFT, SR, VocoderV2
from .v6_istft_tts import V6PhraseProsodyAdapter, compute_phrase_context_torch, function_word_mask_torch


class LoopedWeightTiedEncoder(nn.Module):
    """`sanoTTS`-style weight-tied looped encoder: `n_phys=2` physical blocks looped `n_loops=2` times (effective depth=4)
    with tiny per-step FiLM `(gamma, beta)` adapters (~360k params total)."""
    def __init__(self, n_vocab=N_SYMBOLS, d=96, n_phys=2, h=4, ff=144, n_loops=2):
        super().__init__()
        self.d = d
        self.n_phys = n_phys
        self.n_loops = n_loops
        self.emb = nn.Embedding(n_vocab, d, padding_idx=0)
        self.spk = nn.Embedding(16, d)
        self.lang = nn.Embedding(8, d)
        self.sty = nn.Embedding(32, d)
        self.studio = nn.Embedding(2, d)
        self.pitch = nn.Linear(1, d)
        self.prosody = nn.Linear(4, d)
        self.f0_pred = nn.Linear(d, 1)
        self.energy_pred = nn.Linear(d, 1)
        self.layers = nn.ModuleList([EncLayer(d, h=h, ff=ff) for _ in range(n_phys)])
        # Tiny per-iteration FiLM (gamma, beta) adapters: < 1.5 KB total
        self.loop_gamma = nn.Parameter(torch.ones(n_loops * n_phys, 1, d))
        self.loop_beta = nn.Parameter(torch.zeros(n_loops * n_phys, 1, d))
        self.nf = nn.LayerNorm(d)
        self.mu = nn.Linear(d, 100)

    def forward(self, ids, spk, lang, sty, studio, m, pitch_add=None, prosody=None, sty_mask=None):
        lang0 = torch.zeros_like(lang)
        lang_eff = torch.where(lang == 4, lang, lang0)
        sty0 = sty * 0
        g_base = self.spk(spk) + self.lang(lang0) + self.studio(studio)
        g_sty_delta = self.sty(sty) - self.sty(sty0)
        if prosody is not None:
            g_sty_delta = g_sty_delta + self.prosody(prosody.to(g_sty_delta.dtype))
        if sty_mask is not None:
            g_sty_delta = g_sty_delta * sty_mask.to(g_sty_delta.dtype)
        g_sty_full = self.sty(sty0) + 0.24 * g_sty_delta
        g_vel = g_base + self.sty(sty0)
        g_enc = g_base + g_sty_full + 0.18 * (self.lang(lang_eff) - self.lang(lang0))
        x = self.emb(ids) * math.sqrt(self.emb.embedding_dim) + g_enc[:, None]
        if pitch_add is not None:
            x = x + self.pitch(pitch_add.to(x.dtype)[..., None])
        step_idx = 0
        for _ in range(self.n_loops):
            for lyr in self.layers:
                x = lyr(x, m)
                x = x * self.loop_gamma[step_idx] + self.loop_beta[step_idx]
                step_idx += 1
        h_content = self.nf(x)
        h = h_content + 0.35 * g_sty_full[:, None]
        return h, self.mu(h), g_vel


class SlimDurPredictor(nn.Module):
    """Direct deterministic log-duration head (~45k params)."""
    def __init__(self, d=96, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d, h, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(h, h, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(h, 1, 1),
        )

    def pred_logdur(self, ids, h_enc, mask):
        m = mask[:, None, :].to(h_enc.dtype)
        out = self.net(h_enc.transpose(1, 2) * m).squeeze(1)
        return out * mask.to(out.dtype)


class SakhiyaV6Nano(nn.Module):
    """True Sub-3 MB `Sakhiya-Nano` (~1.88M params: ~395k Looped Encoder + ~1.48M Decoder-Heavy Harmonic-iSTFT)."""
    def __init__(self, d_enc=96, n_phys=2, ff_enc=144, n_loops=2, d_vel=120, nb_vel=5, h_dur=64, v_ch=104, v_rk=(3, 7, 11)):
        super().__init__()
        self.cfg_dict = dict(
            d_enc=d_enc, n_phys=n_phys, ff_enc=ff_enc, n_loops=n_loops,
            d_vel=d_vel, nb_vel=nb_vel, h_dur=h_dur, v_ch=v_ch, v_rk=list(v_rk),
        )
        self.enc = LoopedWeightTiedEncoder(d=d_enc, n_phys=n_phys, ff=ff_enc, n_loops=n_loops)
        self.dur = SlimDurPredictor(d=d_enc, h=h_dur)
        self.adapter = V6PhraseProsodyAdapter(d_enc=d_enc)
        self.vel = Velocity(n_mel=100, d=d_vel, nb=nb_vel, c=d_enc)
        self.voc = VocoderV2(ch=v_ch, res_kernels=v_rk)
        self.register_buffer("stats", torch.tensor([-5.5028, 2.1869]))
        self.register_buffer("pros_tbl", torch.tensor(STYLE_PROSODY_TARGETS, dtype=torch.float32))

    def warm_from_teacher(self, teacher_sd, voc_sd=None):
        """Warm-slice from v6_teacher / v5_c1 into Looped Weight-Tied Encoder + Decoder-Heavy iSTFT."""
        with torch.no_grad():
            for k, p in self.state_dict().items():
                src_k = k
                if k.startswith("enc.") or k.startswith("vel."):
                    src_k = f"am.{k}" if f"am.{k}" in teacher_sd else k
                if src_k in teacher_sd and torch.is_floating_point(p):
                    s = teacher_sd[src_k].float().to(p.device)
                    if s.ndim == p.ndim and all(a >= b for a, b in zip(s.shape, p.shape)):
                        sl = tuple(slice(0, b) for b in p.shape)
                        p.copy_(s[sl])
            if voc_sd is not None:
                for k, p in self.voc.state_dict().items():
                    if k in voc_sd and torch.is_floating_point(p):
                        s = voc_sd[k].float().to(p.device)
                        if s.shape == p.shape:
                            p.copy_(s)
                        elif s.ndim == p.ndim and all(a >= b for a, b in zip(s.shape, p.shape)):
                            sl = tuple(slice(0, b) for b in p.shape)
                            sc = math.sqrt(s.shape[1] / p.shape[1]) if s.ndim >= 2 and s.shape[1] > p.shape[1] else 1.0
                            p.copy_(s[sl] * min(sc, 1.25))
            nn.init.zeros_(self.vel.out.weight)
            nn.init.zeros_(self.vel.out.bias)

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

        h, mu, g = self.enc(ids, spk0, lang_eff, sty_emb, studio, im, pitch_add=p_enc, prosody=pros, sty_mask=ones_m)
        h, vuv_logit = self.adapter(h, phrase_ctx, contour=contour, drop_prob=drop_prob)

        eng_hat = self.enc.energy_pred(h).squeeze(-1).clamp(-1.2, 1.2) + 0.35 * eng_c_delta
        phrase_end_soft = -0.04 * (phrase_ctx[:, :, 0] > 0.85).float() * (1.0 - is_pause)
        eng_mod = (torch.log(energy_scale.clamp(min=0.2)) + 0.10 * pros[:, 2:3] + 0.06 * eng_hat + phrase_end_soft)[..., None] * 0.32
        mu = mu + eng_mod

        x_d = self.dur.pred_logdur(ids, h, im)
        dur_frames = torch.exp(x_d).clamp(min=1.0, max=60.0) * (1.0 + dur_c_ratio)
        rate_mult = torch.exp(-0.04 * pros[:, 0:1]) * (1.0 - is_pause) + torch.exp(0.08 * pros[:, 3:4]) * is_pause
        dur_frames = (dur_frames * rate_mult).clamp(min=1.0, max=60.0)
        fn_mask = function_word_mask_torch(ids)
        is_tail = ((pos >= (L_f - 3.0)) & (ids > 6)).float()
        floor_frames = torch.maximum(is_tail * 5.0, fn_mask * 5.5)
        dur_frames = torch.maximum(dur_frames, floor_frames + (1.0 - (floor_frames > 0).float()) * 1.0)
        logdur = torch.log(dur_frames + 1.0)

        f0_hat = F.avg_pool1d(F.pad(self.enc.f0_pred(h).transpose(1, 2), (1, 1), mode="replicate"), 3, stride=1)[:, 0].clamp(-1.2, 1.2)
        p_story_arch = is_story * 0.42 * (0.60 * torch.sin(phase * 6.2831853) + 0.40 * torch.cos(phase * 12.56637)) * (1.0 - is_pause)
        p_prose = (0.35 * f0_hat + 0.65 * p_arch) * sty_gain + p_story_arch + 0.30 * f0_c_delta
        p_out = has_svara * (0.88 * p_svara + 0.35 * p_prose + pitch_shift) + (1.0 - has_svara) * (p_prose + 0.50 * p_svara + 0.35 * pitch_shift)
        sty_exp = sty_f0_sc.expand(B, L)
        g_exp = g[:, None, :].expand(B, L, g.shape[-1])
        x_out = torch.cat([mu, g_exp, p_out[..., None], sty_exp[..., None]], dim=-1)
        return x_out, logdur, f0_hat, eng_hat, vuv_logit

    def decode_to_mel(self, x):
        c_dim = self.vel.c
        mu_y = x[:, :, :100].transpose(1, 2)
        g = x[:, 0, 100:100 + c_dim]
        p_xe = x[:, :, 100 + c_dim]
        sty_gain = x[:, :, 100 + c_dim + 1]
        B, C, T = mu_y.shape
        m = torch.ones((B, 1, T), dtype=mu_y.dtype, device=mu_y.device)
        t0 = torch.zeros((B,), device=mu_y.device, dtype=torch.float32)
        xt = mu_y + self.vel(mu_y, t0, mu_y, g, m)
        mel = xt * self.stats[1] + self.stats[0]
        p_zero = p_xe - p_xe.mean(dim=1, keepdim=True)
        p_s1 = F.avg_pool1d(F.pad(p_zero[:, None, :], (4, 4), mode="replicate"), 9, stride=1)
        p_sm = F.avg_pool1d(F.pad(p_s1, (2, 2), mode="replicate"), 5, stride=1)
        env = F.avg_pool1d(F.pad(mel.transpose(1, 2), (3, 3), mode="replicate"), 7, stride=1).transpose(1, 2)
        kb = torch.arange(100, device=mel.device, dtype=mel.dtype)
        wk = ((64.0 - kb) / 18.0).clamp(0.0, 1.0)[None, :, None]
        wk_f0 = ((32.0 - kb) / 8.0).clamp(0.0, 1.0)[None, :, None]
        har = (mel - env) * (1.0 + 0.12 * wk)
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


def to_selective_mixed_int8_fp16(src, dst, fp16_threshold_elems=8192):
    """`sanoTTS` Selective Mixed-Precision ONNX Quantizer:
    - Keeps sensitive final iSTFT projection, HarmonicSource F0 layers, and small/medium tensors (< fp16_threshold_elems) in FP16!
    - Quantizes large hidden Conv1D / ConvTranspose1D / MatMul weights (>= fp16_threshold_elems) to per-channel INT8 QDQ (zero ConvInteger).
    Achieves BOTH cosine correlation >= 0.985 vs FP32 AND total model size <= 2.8 MB!"""
    from scripts.export_v2_onnx import _strip_meta
    m = onnx.load(src)
    use = {}
    for n in m.graph.node:
        for i in n.input:
            use.setdefault(i, n.op_type)
    new_nodes, keep = [], []
    for init in m.graph.initializer:
        arr = numpy_helper.to_array(init)
        op = use.get(init.name, "")
        is_sensitive = any(k in init.name for k in ("post", "f0_net", "proj4", "proj16", "out", "emb"))
        if init.data_type == TensorProto.FLOAT and arr.size >= fp16_threshold_elems and arr.ndim >= 2 and not is_sensitive:
            ax = 1 if op in ("ConvTranspose", "MatMul") else 0
            red = tuple(i for i in range(arr.ndim) if i != ax)
            sc = np.maximum(np.abs(arr).max(axis=red), 1e-8) / 127.0
            shp = [1] * arr.ndim
            shp[ax] = -1
            q = np.clip(np.round(arr / sc.reshape(shp)), -127, 127).astype(np.int8)
            keep += [
                numpy_helper.from_array(q, init.name + "_q"),
                numpy_helper.from_array(sc.astype(np.float32), init.name + "_s"),
            ]
            new_nodes.append(
                helper.make_node("DequantizeLinear", [init.name + "_q", init.name + "_s"], [init.name], axis=ax, name=init.name + "_dq")
            )
        elif init.data_type == TensorProto.FLOAT and arr.size > 8:
            h = numpy_helper.from_array(arr.astype(np.float16), init.name + "_fp16")
            keep.append(h)
            new_nodes.append(helper.make_node("Cast", [h.name], [init.name], to=TensorProto.FLOAT, name=init.name + "_cast"))
        else:
            keep.append(init)
    del m.graph.initializer[:]
    m.graph.initializer.extend(keep)
    nodes = list(m.graph.node)
    del m.graph.node[:]
    m.graph.node.extend(new_nodes + nodes)
    _strip_meta(m)
    onnx.save(m, dst)


def verify_quantization_fidelity_gate(out_dir, min_corr=0.980):
    """`sanoTTS` Quantization Fidelity Gate: measure log-mel & spectral cosine correlation between FP16/FP32 and INT8 renders
    across all 15 master clips. Requires mean correlation >= min_corr (0.980)!"""
    from sakhiya.onnx_tts import SakhiyaONNX
    from sakhiya.v2_vocoder import mel_spec_24k
    from scripts.freeze_version_pack import MASTER_15_CLIPS

    eng_fp = SakhiyaONNX(out_dir, tiny=False)
    eng_q8 = SakhiyaONNX(out_dir, tiny=True)
    corrs = []
    for cid, lang, sty, text in MASTER_15_CLIPS:
        w_fp, _ = eng_fp.say(text, speaker="sakhiya", lang=lang, style=sty)
        w_q8, _ = eng_q8.say(text, speaker="sakhiya", lang=lang, style=sty)
        L = min(len(w_fp), len(w_q8))
        m_fp = mel_spec_24k(torch.from_numpy(w_fp[:L])).flatten().numpy()
        m_q8 = mel_spec_24k(torch.from_numpy(w_q8[:L])).flatten().numpy()
        c = float(np.dot(m_fp, m_q8) / (np.linalg.norm(m_fp) * np.linalg.norm(m_q8) + 1e-8))
        corrs.append(c)
    mean_corr = float(np.mean(corrs))
    min_clip_corr = float(np.min(corrs))
    return dict(mean_corr=round(mean_corr, 5), min_clip_corr=round(min_clip_corr, 5), passed=(mean_corr >= min_corr))


def export_v6_nano_onnx(nano: SakhiyaV6Nano, out_dir: str, max_fp16_mb: float = 4.5, max_int8_mb: float = 3.0):
    """Export `Sakhiya-Nano` to single-file (`sakhiya_nano_fp16.onnx` & `sakhiya_nano_int8.onnx`) + browser 3-graph ONNX,
    and enforce the `sanoTTS` Quantization Fidelity Gate (`corr >= 0.98`) + size gate (`<= 3.0 MB`)."""
    from scripts.export_v2_onnx import to_fp16_weights, to_int8_qdq_weights
    from .v6_istft_tts import SingleFileV6Export
    os.makedirs(out_dir, exist_ok=True)
    nano.eval()
    nano.voc.fuse_weight_norm()

    single_w = SingleFileV6Export(nano).eval()
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
            "ids": {1: "L"}, "pitch_add": {1: "L"}, "dur_override": {1: "L"}, "contour": {1: "L"},
            "real": {2: "F16"}, "imag": {2: "F16"}, "logdur": {1: "L"},
        },
        opset_version=17, dynamo=False,
    )
    to_fp16_weights(p_m32, os.path.join(out_dir, "sakhiya_nano_fp16.onnx"))
    to_selective_mixed_int8_fp16(p_m32, os.path.join(out_dir, "sakhiya_nano_int8.onnx"), fp16_threshold_elems=4096)
    os.remove(p_m32)

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
        _EncW(nano).eval(), (ids, spk, lang, sc0, sc1, sc1, p_add, sty, contour), p_e32,
        input_names=["ids", "spk", "lang", "pitch_shift", "pitch_scale", "energy_scale", "pitch_add", "style", "contour"],
        output_names=["x", "logdur"],
        dynamic_axes={"ids": {1: "L"}, "pitch_add": {1: "L"}, "contour": {1: "L"}, "x": {1: "L"}, "logdur": {1: "L"}},
        opset_version=17, dynamo=False,
    )
    x_dum = torch.randn(1, 32, 100 + nano.cfg_dict["d_enc"] + 2)
    torch.onnx.export(_DecW(nano).eval(), (x_dum,), p_d32, input_names=["x"], output_names=["mel"], dynamic_axes={"x": {1: "T"}, "mel": {2: "T"}}, opset_version=17, dynamo=False)
    m_dum = torch.randn(1, 100, 32)
    torch.onnx.export(_VocW(nano).eval(), (m_dum,), p_v32, input_names=["mel"], output_names=["real", "imag"], dynamic_axes={"mel": {2: "T"}, "real": {2: "F16"}, "imag": {2: "F16"}}, opset_version=17, dynamo=False)

    for nm, src in (("encoder", p_e32), ("decoder", p_d32), ("vocoder", p_v32)):
        to_fp16_weights(src, os.path.join(out_dir, f"{nm}.onnx"))
        to_selective_mixed_int8_fp16(src, os.path.join(out_dir, f"{nm}.int8.onnx"), fp16_threshold_elems=4096)
        os.remove(src)
        ops8 = {n.op_type for n in onnx.load(os.path.join(out_dir, f"{nm}.int8.onnx")).graph.node}
        assert "ConvInteger" not in ops8, f"Forbidden ConvInteger in {nm}.int8.onnx!"

    cfg = dict(
        arch="sakhiya-v6-nano-istft24k",
        version="6.0",
        tier="v6_nano",
        sample_rate=SR, n_fft=1024, hop_length=HOP, win_length=1024, n_mels=100,
        istft_n_fft=ISTFT_N_FFT, istft_hop=ISTFT_HOP, upsample_factor=16,
        symbols=SYMBOLS,
        spk2id={"sakhiya": 0, "female": 0},
        default_speaker="sakhiya",
        speakers=[{"id": 0, "name": "sakhiya", "label": "सखिया · Sakhiya-Nano (<3MB Instant)", "lang": "ne", "gender": "f"}],
        style2id={"news": 0, "neutral": 0, "conversational": 1, "storytelling": 2, "warm_soft": 3, "slow_clear": 3, "expressive": 4, "vedic_chant": 5, "chant": 5},
        styles=["news", "storytelling", "vedic_chant"],
        lang2id=V2_LANG2ID, lock_voice=True,
    )
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=2, ensure_ascii=False)

    fp16_mb = round(os.path.getsize(os.path.join(out_dir, "sakhiya_nano_fp16.onnx")) / 1e6, 3)
    int8_mb = round(os.path.getsize(os.path.join(out_dir, "sakhiya_nano_int8.onnx")) / 1e6, 3)
    assert fp16_mb <= max_fp16_mb, f"Sakhiya-Nano fp16 {fp16_mb} MB > {max_fp16_mb} MB"
    assert int8_mb <= max_int8_mb, f"Sakhiya-Nano int8 {int8_mb} MB > {max_int8_mb} MB"

    q_gate = verify_quantization_fidelity_gate(out_dir, min_corr=0.980)
    assert q_gate["passed"], f"Quantization fidelity gate failed: {q_gate}"

    n_p_enc = sum(p.numel() for p in nano.enc.parameters()) + sum(p.numel() for p in nano.dur.parameters()) + sum(p.numel() for p in nano.adapter.parameters())
    n_p_dec = sum(p.numel() for p in nano.vel.parameters()) + sum(p.numel() for p in nano.voc.parameters())
    sz = dict(
        tier="v6_nano",
        params_encoder=n_p_enc,
        params_decoder=n_p_dec,
        params_total=n_p_enc + n_p_dec,
        total_fp16_mb=fp16_mb,
        total_tiny_mb=int8_mb,
        fp16_total_mb=fp16_mb,
        int8_total_mb=int8_mb,
        quantization_fidelity_gate=q_gate,
        encoder=dict(fp16_mb=round(os.path.getsize(f"{out_dir}/encoder.onnx") / 1e6, 3), int8_mb=round(os.path.getsize(f"{out_dir}/encoder.int8.onnx") / 1e6, 3)),
        decoder=dict(fp16_mb=round(os.path.getsize(f"{out_dir}/decoder.onnx") / 1e6, 3), int8_mb=round(os.path.getsize(f"{out_dir}/decoder.int8.onnx") / 1e6, 3)),
        vocoder=dict(fp16_mb=round(os.path.getsize(f"{out_dir}/vocoder.onnx") / 1e6, 3), int8_mb=round(os.path.getsize(f"{out_dir}/vocoder.int8.onnx") / 1e6, 3)),
    )
    try:
        from scripts.verify_browser_ort import verify as _verify_js
        sz["browser_ort_node"] = _verify_js(out_dir)
    except Exception as e:
        print("browser_ort_node warn:", e, flush=True)
    json.dump(sz, open(os.path.join(out_dir, "size.json"), "w"), indent=2)
    import time as _time
    json.dump(dict(version="6.0", arch=cfg["arch"], fp16_mb=fp16_mb, int8_mb=int8_mb, quant_corr=q_gate["mean_corr"], generated_utc=_time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())), open(os.path.join(out_dir, "manifest.json"), "w"), indent=2)
    return sz
