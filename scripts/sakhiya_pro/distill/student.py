"""Sakhiya Single-Graph Harmonic-iSTFT Distilled Students (`scripts/sakhiya_pro/distill/student.py`).

Unifies Acoustic Model (RoPE/Looped-Universal TextEncoder + Deterministic DurFlow + ConvNeXt-1D CFM Velocity Decoder)
with Decoder-Heavy Harmonic-iSTFT Waveform Synthesis (`HarmonicSource` continuous-phase excitation + 16x `Snake1d`
upsampling + 1-op COLA `ConvTranspose1d` iSTFT) in a SINGLE end-to-end ONNX graph:
  inputs:  ids [1, L] int64, lang [1] int64, dur_scale [1] float32, pitch_shift [1] float32
  output:  wav [1, N] float32 (24 kHz mono)

Tiers:
- t30: Full Teacher-Bridge (d_enc=192, L_enc=6, d_vel=256, nb_vel=12, steps=4, voc_ch=320) -> ~15.4 MB INT8 ONNX
- t15: Flagship Compact (d_enc=192, L_enc=6, d_vel=256, nb_vel=9, steps=3, voc_ch=320)     -> ~14.1 MB <= 14.5 MB INT8 ONNX
- t10: Compact 10M (d_enc=144, L_enc=5, d_vel=160, nb_vel=7, steps=2, voc_ch=320)          -> ~8.6 MB <= 9.5 MB INT8 ONNX
- t5:  Hero Sub-5 MB (d_enc=144, L_enc=5, d_vel=160, nb_vel=5, steps=1, voc_ch=144)        -> ~4.4 MB <= 4.8 MB INT8 ONNX
- t1:  Breakthrough Sub-1.5 MB Looped (1 shared Universal Encoder block looped 4x with 1-KB FiLM adapters +
       1 shared Universal Velocity block looped 5x + Decoder-Heavy Harmonic-iSTFT voc_ch=144 with ~74% of weights
       in the waveform decoder)                                                            -> ~1.36 MB <= 1.45 MB INT8 ONNX
"""
import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sakhiya.cfm import SakhiyaAM, STYLE_PROSODY_TARGETS
from sakhiya.v15_pico import SakhiyaPicoAM
from sakhiya.v2_vocoder import VocoderV2, ISTFT_N_FFT, ISTFT_HOP, SR, N_FFT, HOP, NMEL as N_MEL
from sakhiya.v15_rnd_wavehax import _build_cola_istft_kernel, ste_int8_weight

NB = N_FFT // 2 + 1
N_SYM = 64
N_LANG = 2

TIERS = {
    "t30": dict(tier="t30", arch="sakhiya_am", d_enc=192, L_enc=6, ff_enc=256, d_vel=256, nb_vel=12, h_dur=128, voc_ch=320, steps=4),
    "t15": dict(tier="t15", arch="sakhiya_am", d_enc=192, L_enc=6, ff_enc=256, d_vel=256, nb_vel=9,  h_dur=128, voc_ch=320, steps=3),
    "t10": dict(tier="t10", arch="sakhiya_am", d_enc=144, L_enc=5, ff_enc=224, d_vel=160, nb_vel=7,  h_dur=96,  voc_ch=320, steps=2),
    "t5":  dict(tier="t5",  arch="sakhiya_am", d_enc=144, L_enc=5, ff_enc=224, d_vel=160, nb_vel=5,  h_dur=96,  voc_ch=144, steps=1),
    "t1":  dict(tier="t1",  arch="sakhiya_pico", d_enc=96, L_enc=4, ff_enc=144, d_vel=112, nb_vel=5, h_dur=64,  voc_ch=144, steps=1),
}


def _calibrate_styles(am):
    with torch.no_grad():
        w_spk = am.enc.spk.weight.clone()
        w_sty = am.enc.sty.weight.clone()
        if w_sty.shape[0] >= 15 and w_spk.shape[0] >= 14:
            w0 = 0.65 * w_sty[10] + 0.35 * w_sty[3]
            w2 = w0 + 1.15 * ((0.75 * w_sty[3] + 0.25 * w_sty[2]) - w0)
            w5 = 0.60 * w0 + 0.40 * (0.55 * w_sty[3] + 0.45 * w_sty[14])
            am.enc.sty.weight[0].copy_(w0)
            am.enc.sty.weight[1].copy_(0.45 * w0 + 0.55 * w2)
            am.enc.sty.weight[2].copy_(w2)
            am.enc.sty.weight[3].copy_(0.60 * w0 + 0.40 * w2)
            am.enc.sty.weight[4].copy_(0.20 * w0 + 0.80 * w2)
            am.enc.sty.weight[5].copy_(w5)
        else:
            w0 = am.enc.sty.weight[0].clone()
            w2 = am.enc.sty.weight[2].clone()
            am.enc.sty.weight[1].copy_(0.45 * w0 + 0.55 * w2)
            am.enc.sty.weight[3].copy_(0.60 * w0 + 0.40 * w2)
            am.enc.sty.weight[4].copy_(0.20 * w0 + 0.80 * w2)
        w_l = am.enc.lang.weight
        for lid in range(w_l.shape[0]):
            if lid != 4:
                w_l[lid].copy_(w_l[0])


def snap_f0_int8(voc):
    with torch.no_grad():
        for idx in (0, 2):
            w = voc.src.f0_net[idx].weight
            w.copy_(ste_int8_weight(w, axis=0, enabled=True))
        for p in voc.src.f0_net.parameters():
            p.requires_grad_(False)


class Student(nn.Module):
    def __init__(self, tier="t15", arch=None, d_enc=192, L_enc=6, ff_enc=256, d_vel=256, nb_vel=9, h_dur=128, voc_ch=320, steps=3, **kw):
        super().__init__()
        if tier in TIERS and arch is None:
            c = TIERS[tier]
            arch, d_enc, L_enc, ff_enc, d_vel, nb_vel, h_dur, voc_ch, steps = (
                c["arch"], c["d_enc"], c["L_enc"], c["ff_enc"], c["d_vel"], c["nb_vel"], c["h_dur"], c["voc_ch"], c["steps"]
            )
        self.tier = tier
        self.arch = arch or ("sakhiya_pico" if tier == "t1" else "sakhiya_am")
        self.steps = int(steps)
        self.cfg = dict(
            tier=self.tier, arch=self.arch, d_enc=d_enc, L_enc=L_enc, ff_enc=ff_enc,
            d_vel=d_vel, nb_vel=nb_vel, h_dur=h_dur, voc_ch=voc_ch, steps=self.steps
        )
        if self.arch == "sakhiya_pico":
            self.am = SakhiyaPicoAM(d_enc=d_enc, L_enc=L_enc, ff_enc=ff_enc, d_vel=d_vel, nb_vel=nb_vel, h_dur=h_dur)
            # Share universal velocity block across all 5 steps so decoder weights are stored once and ~74% of weights sit in the Harmonic-iSTFT vocoder
            self.am.vel.ublock_late = self.am.vel.ublock_early
        else:
            self.am = SakhiyaAM(d_enc=d_enc, L_enc=L_enc, ff_enc=ff_enc, d_vel=d_vel, nb_vel=nb_vel, h_dur=h_dur)
            # Remove unused training-only speaker adversarial head so exported ONNX has zero dead parameters
            if hasattr(self.am.enc, "spk_adv"):
                self.am.enc.spk_adv = nn.Identity()

        self.voc = VocoderV2(ch=voc_ch, res_kernels=(3, 7, 11))
        self.voc.fuse_weight_norm()

        K_istft, _ = _build_cola_istft_kernel(ISTFT_N_FFT, ISTFT_HOP)
        self.register_buffer("K_istft", torch.from_numpy(K_istft), persistent=False)
        self.register_buffer("pros_tbl", torch.tensor(STYLE_PROSODY_TARGETS, dtype=torch.float32), persistent=False)
        self.register_buffer("f0_stats", torch.tensor([5.418, 0.203]))
        self.register_buffer("mel_stats", torch.tensor([-5.5028, 2.1869]))
        # Learned Nepal-native prosody & spectral tilt adaptation scalars (calibrated on Sakhiya-Pro v3 cache_train.pt)
        self.closure_slope = nn.Parameter(torch.tensor(-0.22))
        self.dur_rate_bias = nn.Parameter(torch.tensor(0.0))
        self._qat_flag = False

    def warm_init(self, ck_dir="/work/tts/ck"):
        """Warm-initialize Student AM and Harmonic-iSTFT Vocoder from pre-trained Sakhiya checkpoints."""
        dev = next(self.parameters()).device
        if self.tier in ("t30", "t15"):
            am_p = os.path.join(ck_dir, "v5_c1_am.pt")
        elif self.tier in ("t10", "t5"):
            am_p = os.path.join(ck_dir, "v9_voca_am.pt")
        else:
            am_p = os.path.join(ck_dir, "v9_sano_am.pt")
        voc_p = os.path.join(ck_dir, "v5_c1_voc.pt" if self.cfg["voc_ch"] == 320 else "v14_voc144.pt")

        if os.path.exists(am_p):
            ck = torch.load(am_p, map_location=dev, weights_only=False)
            sd = {k: v.float() for k, v in ck.get("ema", ck.get("net", ck)).items() if not k.startswith("ctc_rec.")}
            if self.arch == "sakhiya_pico":
                sano_ref = SakhiyaAM(d_enc=96, L_enc=4, ff_enc=144, d_vel=112, nb_vel=5, h_dur=64).to(dev)
                sano_ref.load_state_dict(sd, strict=False)
                _calibrate_styles(sano_ref)
                self.am.warm_from_sano(sano_ref.state_dict())
                with torch.no_grad():
                    # Copy full c2 kernel (k=5) since ff_enc=144 matches sano 1:1
                    if "enc.layers.1.c2.weight" in sd and self.am.enc.ublock.c2.weight.shape == sd["enc.layers.1.c2.weight"].shape:
                        self.am.enc.ublock.c2.weight.copy_(sd["enc.layers.1.c2.weight"].float())
            else:
                self.am.load_state_dict(sd, strict=False)
            self.am.stats.copy_(torch.tensor([-5.5028, 2.1869], device=dev))
            _calibrate_styles(self.am)

        if os.path.exists(voc_p):
            vck = torch.load(voc_p, map_location=dev, weights_only=False)
            vsd = {k: v.float() for k, v in vck.get("G_ema", vck.get("G", vck)).items()}
            self.voc.load_state_dict(vsd, strict=False)
            snap_f0_int8(self.voc)
        return self

    def voc_wav(self, mel, shift_st=None):
        """Differentiable Harmonic-iSTFT waveform synthesis from 100-band log-mel [B, 100, T]."""
        re_s, im_s, f0, uv = self.voc.spectrum_and_f0(mel, shift_st=shift_st)
        z = torch.zeros_like(im_s[:, :1, :])
        spec_cat = torch.cat([re_s.float(), torch.cat([z, im_s[:, 1:-1, :], z], dim=1).float()], dim=1)
        y = F.conv_transpose1d(spec_cat, self.K_istft.to(spec_cat.device), stride=ISTFT_HOP, padding=(ISTFT_N_FFT - ISTFT_HOP) // 2)[:, 0, :]
        return y, re_s, im_s, f0, uv

    @torch.no_grad()
    def infer(self, ids, lang, dur_scale, pitch_shift):
        """Single-graph ONNX inference: ids [1,L] int64, lang [1] int64, dur_scale [1] float, pitch_shift [1] float -> wav [1,N]."""
        B, L = ids.shape
        im = torch.ones((B, L), dtype=torch.bool, device=ids.device)
        spk0 = torch.zeros((B,), dtype=torch.long, device=ids.device)
        studio = torch.ones((B,), dtype=torch.long, device=ids.device)
        is_sa = (lang == 1)
        lang_eff = torch.where(is_sa, torch.full_like(lang, 4), torch.zeros_like(lang))
        sty = torch.where(is_sa, torch.full_like(lang, 5), torch.zeros_like(lang))
        pros = self.pros_tbl[sty]

        pos = torch.arange(L, device=ids.device, dtype=torch.float32)[None, :]
        L_f = (pos[:, -1:] + 1.0).clamp(min=2.0)
        is_pause = (ids <= 6).float()
        phase = pos / L_f
        # Grounded Kathmandu/Nepal-native downward clause ending (eradicates upward floating endings)
        tail_closure = self.closure_slope.clamp(-0.35, -0.08) * ((phase - 0.70).clamp(min=0.0) / 0.30) * (1.0 - is_pause)
        p_arch = (0.48 * torch.sin(phase * 3.14159265) + 0.38 * torch.sin(phase * 9.424778) + tail_closure) * (1.0 - is_pause) - 0.25 * is_pause
        p_arch = F.avg_pool1d(F.pad(p_arch[:, None, :], (1, 1), mode="replicate"), 3, stride=1)[:, 0]
        p_enc = (0.14 * p_arch + 0.35 * pitch_shift[:, None]).clamp(-0.48, 0.48)

        ones_m = torch.ones((B, 1), dtype=torch.float32, device=ids.device)
        h, mu, g = self.am.enc(ids, spk0, lang_eff, sty, studio, im, pitch_add=p_enc, prosody=pros, sty_mask=ones_m)
        if hasattr(self.am.enc, "energy_pred"):
            eng_hat = self.am.enc.energy_pred(h).squeeze(-1).clamp(-1.2, 1.2)
            mu = mu + (0.10 * pros[:, 2:3] + 0.06 * eng_hat)[..., None] * 0.32

        x_d = self.am.dur.pred_logdur(ids, h, im) + self.dur_rate_bias.clamp(-0.12, 0.12)
        dur = torch.clamp(torch.round((torch.exp(x_d) - 1.0) * dur_scale[:, None]), min=3.0, max=60.0)
        # Protect sentence-final tokens and short Nepali function words from frame collapse
        is_tail = ((pos >= (L_f - 3.0)) & (ids > 6)).float()
        dur = torch.maximum(dur, is_tail * 5.0 + (1.0 - is_tail) * 3.0)
        # Resonant Pranava ॐ floor on Sanskrit utterances starting with 'o' (17) + 'm' (39)
        is_om_o = (is_sa[:, None] & (pos == 1.0) & (ids == 17)).float()
        is_om_m = (is_sa[:, None] & (pos == 2.0) & (ids == 39)).float()
        dur = torch.maximum(dur, is_om_o * 36.0 + is_om_m * 18.0)

        ends = torch.cumsum(dur, 1)
        T = ends[0, -1].to(torch.int64)
        t_arange = torch.arange(0, T, device=ids.device, dtype=torch.float32)
        idx = (t_arange[None, :, None] >= ends[:, None, :]).to(torch.float32).sum(-1).to(torch.int64).clamp(max=L - 1)

        mu_t = mu.transpose(1, 2)
        mu_y = torch.gather(mu_t, 2, idx[:, None, :].expand(-1, mu_t.shape[1], -1))
        m_t = torch.ones((B, 1, T), dtype=mu_y.dtype, device=mu_y.device)

        if self.steps == 1:
            t0 = torch.zeros((B,), device=mu_y.device, dtype=torch.float32)
            xt = mu_y + self.am.vel(mu_y, t0, mu_y, g, m_t)
        else:
            c_idx = torch.arange(100, device=mu_y.device, dtype=torch.float32)[None, :, None]
            t_idx = torch.arange(T, device=mu_y.device, dtype=torch.float32)[None, None, :]
            z_init = torch.sin(c_idx * 12.9898 + t_idx * 78.233) * torch.cos(c_idx * 39.346 + t_idx * 11.135) * (0.667 * 1.4142)
            xt = z_init.expand(B, 100, T)
            dt = 1.0 / self.steps
            for i in range(self.steps):
                ti = torch.full((B,), i * dt, device=mu_y.device, dtype=torch.float32)
                k1 = self.am.vel(xt, ti, mu_y, g, m_t)
                k2 = self.am.vel(xt + 0.5 * dt * k1, ti + 0.5 * dt, mu_y, g, m_t)
                xt = xt + dt * k2

        mel = xt * self.am.stats[1] + self.am.stats[0]
        p_tok = (0.16 * p_arch + 0.35 * pitch_shift[:, None]).clamp(-0.65, 0.65)
        p_frame = torch.gather(p_tok[:, None, :], 2, idx[:, None, :])
        p_sm = F.avg_pool1d(F.pad(p_frame, (2, 2), mode="replicate"), 5, stride=1)

        y, _, _, _, _ = self.voc_wav(mel, shift_st=p_sm)
        pk = y.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
        return y * (0.88 / pk)


def set_qat(model, flag):
    model._qat_flag = bool(flag)
    if hasattr(model.am, "set_qat"):
        model.am.set_qat(bool(flag))
    for name, m in model.named_modules():
        if name.startswith("voc.src.f0_net"):
            continue
        if hasattr(m, "qat_enabled"):
            m.qat_enabled = bool(flag)


def build(tier):
    return Student(tier=tier)


def n_params(m, deployed=True):
    return sum(p.numel() for n, p in m.named_parameters() if not n.startswith("am.enc.spk_adv"))


if __name__ == "__main__":
    for t in TIERS:
        m = build(t)
        print(t, f"{n_params(m)/1e6:.3f}M params")
