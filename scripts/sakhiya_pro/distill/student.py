"""Sakhiya Single-Graph Harmonic-iSTFT Distilled Students (`scripts/sakhiya_pro/distill/student.py`).

Unifies Acoustic Model (RoPE/Looped-Universal TextEncoder + Deterministic DurFlow + ConvNeXt-1D CFM Velocity Decoder)
with Decoder-Heavy Harmonic-iSTFT Waveform Synthesis (`HarmonicSource` continuous-phase excitation + 16x `Snake1d`
upsampling + 1-op COLA `ConvTranspose1d` iSTFT) in a SINGLE end-to-end ONNX graph:
  inputs:  ids [1, L] int64, lang [1] int64, dur_scale [1] float32, pitch_shift [1] float32
  output:  wav [1, N] float32 (24 kHz mono)

Tiers:
- t30: Full Teacher-Bridge (d_enc=192, L_enc=6, d_vel=256, nb_vel=12, steps=4, voc_ch=320) -> ~15.6 MB <= 29.6 MB INT8 ONNX
- t15: Flagship Compact (d_enc=192, L_enc=6, d_vel=256, nb_vel=12, steps=4, voc_ch=320, r_enc=96) -> ~14.1 MB <= 14.5 MB INT8 ONNX
- t10: Compact 10M (d_enc=144, L_enc=5, d_vel=160, nb_vel=7, steps=1, voc_ch=320, r_enc=104)      -> ~8.4 MB <= 9.5 MB INT8 ONNX
- t5:  Hero Sub-5 MB (d_enc=144, L_enc=5, d_vel=160, nb_vel=7, steps=1, voc_ch=144, r_enc=88)     -> ~4.4 MB <= 4.8 MB INT8 ONNX
- t1:  Breakthrough Sub-1.5 MB Looped (1 shared Universal Encoder block looped 4x with 1-KB FiLM adapters +
       1 shared Universal Velocity block looped 5x + Decoder-Heavy Harmonic-iSTFT voc_ch=144 with ~79% of weights
       in the waveform decoder)                                                                    -> ~1.36 MB <= 1.45 MB INT8 ONNX
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
from sakhiya.v15_rnd_wavehax import _build_cola_istft_kernel, ste_int8_weight, QATConv1d

NB = N_FFT // 2 + 1
N_SYM = 64
N_LANG = 2

TIERS = {
    "t30": dict(tier="t30", arch="sakhiya_am", d_enc=192, L_enc=6, ff_enc=256, d_vel=256, nb_vel=12, h_dur=128, voc_ch=320, steps=4, r_enc=None, r_voc_k7=None, r_voc_k11=None),
    "t15": dict(tier="t15", arch="sakhiya_am", d_enc=192, L_enc=6, ff_enc=256, d_vel=256, nb_vel=12, h_dur=128, voc_ch=320, steps=4, r_enc=100,  r_voc_k7=None, r_voc_k11=132),
    "t10": dict(tier="t10", arch="sakhiya_am", d_enc=144, L_enc=5, ff_enc=224, d_vel=160, nb_vel=7,  h_dur=96,  voc_ch=320, steps=1, r_enc=104,  r_voc_k7=None, r_voc_k11=None),
    "t5":  dict(tier="t5",  arch="sakhiya_am", d_enc=144, L_enc=5, ff_enc=224, d_vel=160, nb_vel=7,  h_dur=96,  voc_ch=144, steps=1, r_enc=88,   r_voc_k7=None, r_voc_k11=None),
    "t1":  dict(tier="t1",  arch="sakhiya_pico", d_enc=96, L_enc=4, ff_enc=144, d_vel=112, nb_vel=5, h_dur=64,  voc_ch=144, steps=1, r_enc=52,   r_voc_k7=50,   r_voc_k11=31),
}


def _make_svd_seq(conv: nn.Conv1d, rank: int, use_qat: bool = False):
    Co, Ci, K = conv.weight.shape
    r = min(int(rank), Co, Ci * K)
    cls = QATConv1d if use_qat else nn.Conv1d
    c1 = cls(Ci, r, K, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, bias=False)
    c2 = cls(r, Co, 1, bias=(conv.bias is not None))
    with torch.no_grad():
        W = conv.weight.data.float()
        U, S, Vh = torch.linalg.svd(W.reshape(Co, Ci * K), full_matrices=False)
        sq = torch.sqrt(S[:r])
        c1.weight.data.copy_((sq[:, None] * Vh[:r, :]).reshape(r, Ci, K))
        c2.weight.data.copy_((U[:, :r] * sq[None, :]).reshape(Co, r, 1))
        if conv.bias is not None:
            c2.bias.data.copy_(conv.bias.data.float())
    return nn.Sequential(c1, c2)


def _calibrate_styles(am, pre_baked=False):
    with torch.no_grad():
        w_spk = am.enc.spk.weight.clone()
        w_sty = am.enc.sty.weight.clone()
        if not pre_baked and w_sty.shape[0] >= 15 and w_spk.shape[0] >= 14:
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
        if w_l.shape[0] > 4:
            w_l[4].copy_(0.94 * w_l[0] + 0.06 * w_l[4])
        for lid in range(w_l.shape[0]):
            if lid != 4:
                w_l[lid].copy_(w_l[0])


def snap_f0_int8(voc):
    with torch.no_grad():
        for idx in (0, 2):
            w = voc.src.f0_net[idx].weight
            w.copy_(ste_int8_weight(w.float(), axis=0, enabled=True))
        for p in voc.src.f0_net.parameters():
            p.requires_grad_(False)


def snap_am_int8(am):
    """Snap encoder and duration predictor weights to the per-channel INT8 grid so FP32 and INT8 ONNX durations never flip."""
    with torch.no_grad():
        for name, p in am.named_parameters():
            if name.startswith(("enc.", "dur.")):
                if p.ndim >= 2 and p.numel() >= 128 and p.shape[0] > 1:
                    p.copy_(ste_int8_weight(p.float(), axis=0, enabled=True))
                elif p.numel() > 16:
                    p.copy_(p.half().float())


class Student(nn.Module):
    def __init__(self, tier="t15", arch=None, d_enc=192, L_enc=6, ff_enc=256, d_vel=256, nb_vel=12, h_dur=128, voc_ch=320, steps=4, r_enc=None, r_voc_k7=None, r_voc_k11=None, **kw):
        super().__init__()
        if tier in TIERS:
            c = TIERS[tier]
            if arch is None:
                arch = c["arch"]
            d_enc, L_enc, ff_enc, d_vel, nb_vel, h_dur, voc_ch, steps = (
                c["d_enc"], c["L_enc"], c["ff_enc"], c["d_vel"], c["nb_vel"], c["h_dur"], c["voc_ch"], c["steps"]
            )
            if r_enc is None:
                r_enc = c.get("r_enc")
            if r_voc_k7 is None:
                r_voc_k7 = c.get("r_voc_k7")
            if r_voc_k11 is None:
                r_voc_k11 = c.get("r_voc_k11")

        self.tier = tier
        self.arch = arch or ("sakhiya_pico" if tier == "t1" else "sakhiya_am")
        self.steps = int(steps)
        self.r_enc = r_enc
        self.r_voc_k7 = r_voc_k7
        self.r_voc_k11 = r_voc_k11
        self.cfg = dict(
            tier=self.tier, arch=self.arch, d_enc=d_enc, L_enc=L_enc, ff_enc=ff_enc,
            d_vel=d_vel, nb_vel=nb_vel, h_dur=h_dur, voc_ch=voc_ch, steps=self.steps,
            r_enc=self.r_enc, r_voc_k7=self.r_voc_k7, r_voc_k11=self.r_voc_k11,
        )
        self._init_raw_submodules(d_enc, L_enc, ff_enc, d_vel, nb_vel, h_dur, voc_ch)
        self._apply_svd_structure()

        K_istft, _ = _build_cola_istft_kernel(ISTFT_N_FFT, ISTFT_HOP)
        self.register_buffer("K_istft", torch.from_numpy(K_istft), persistent=False)
        self.register_buffer("pros_tbl", torch.tensor(STYLE_PROSODY_TARGETS, dtype=torch.float32), persistent=False)
        self.register_buffer("f0_stats", torch.tensor([5.418, 0.203]))
        self.register_buffer("mel_stats", torch.tensor([-6.1761, 2.4932]))
        # Learned Nepal-native prosody & downward sentence closure adaptation scalars
        self.closure_slope = nn.Parameter(torch.tensor(-0.06))
        self.dur_rate_bias = nn.Parameter(torch.tensor(0.0))
        self._qat_flag = False

    def _init_raw_submodules(self, d_enc, L_enc, ff_enc, d_vel, nb_vel, h_dur, voc_ch):
        if self.arch == "sakhiya_pico":
            self.am = SakhiyaPicoAM(d_enc=d_enc, L_enc=L_enc, ff_enc=ff_enc, d_vel=d_vel, nb_vel=nb_vel, h_dur=h_dur)
        else:
            self.am = SakhiyaAM(d_enc=d_enc, L_enc=L_enc, ff_enc=ff_enc, d_vel=d_vel, nb_vel=nb_vel, h_dur=h_dur)
            if hasattr(self.am, "ctc_rec"):
                self.am.ctc_rec = nn.Identity()
            if hasattr(self.am.enc, "spk_adv"):
                self.am.enc.spk_adv = nn.Identity()

        self.voc = VocoderV2(ch=voc_ch, res_kernels=(3, 7, 11))
        self.voc.fuse_weight_norm()

    def _apply_svd_structure(self):
        if self.arch == "sakhiya_pico":
            self.am.vel.ublock_late = self.am.vel.ublock_early
        if self.r_enc is not None:
            if self.arch == "sakhiya_pico":
                if isinstance(self.am.enc.ublock.c1, nn.Conv1d):
                    self.am.enc.ublock.c1 = _make_svd_seq(self.am.enc.ublock.c1, self.r_enc, use_qat=True)
                    self.am.enc.ublock.c2 = _make_svd_seq(self.am.enc.ublock.c2, self.r_enc, use_qat=True)
            else:
                for lyr in self.am.enc.layers:
                    if isinstance(lyr.c1, nn.Conv1d):
                        lyr.c1 = _make_svd_seq(lyr.c1, self.r_enc, use_qat=False)
                        lyr.c2 = _make_svd_seq(lyr.c2, self.r_enc, use_qat=False)

        for b_idx, r_k in ((1, self.r_voc_k7), (2, self.r_voc_k11)):
            if r_k is not None:
                blk = self.voc.mrfs[0][b_idx]
                for i in range(len(blk.c1)):
                    if isinstance(blk.c1[i], nn.Conv1d):
                        blk.c1[i] = _make_svd_seq(blk.c1[i], r_k, use_qat=False)
                    if isinstance(blk.c2[i], nn.Conv1d):
                        blk.c2[i] = _make_svd_seq(blk.c2[i], r_k, use_qat=False)

    def warm_init(self, ck_dir="/work/tts/ck"):
        """Warm-initialize Student AM and GAN-converged Sakhiya-Pro v3 Harmonic-iSTFT Vocoder."""
        dev = next(self.parameters()).device
        c = self.cfg
        self._init_raw_submodules(c["d_enc"], c["L_enc"], c["ff_enc"], c["d_vel"], c["nb_vel"], c["h_dur"], c["voc_ch"])
        self.to(dev)

        # Prefer Sakhiya-Pro v3 adapted teacher AM if available, with slice-compatible fallback
        v3_t30_p = os.path.join(ck_dir, "v3_t30_am.pt")
        if self.tier in ("t30", "t15"):
            am_p = v3_t30_p if os.path.exists(v3_t30_p) else os.path.join(ck_dir, "v5_c1_am.pt")
        elif self.tier in ("t10", "t5"):
            am_p = os.path.join(ck_dir, "v9_voca_am.pt")
        else:
            am_p = os.path.join(ck_dir, "v9_sano_am.pt")

        if os.path.exists(am_p):
            ck = torch.load(am_p, map_location=dev, weights_only=False)
            pre_baked = bool(ck.get("pre_baked_v25", False))
            sd = {k: v.float() for k, v in ck.get("ema", ck.get("net", ck)).items() if not k.startswith("ctc_rec.")}
            st_vec = torch.tensor([float(ck.get("mean", -6.1760635)), float(ck.get("std", 2.4932315))], device=dev)
            if self.arch == "sakhiya_pico":
                sano_ref = SakhiyaAM(d_enc=96, L_enc=4, ff_enc=144, d_vel=112, nb_vel=5, h_dur=64).to(dev)
                sano_ref.load_state_dict(sd, strict=False)
                _calibrate_styles(sano_ref, pre_baked=pre_baked)
                self.am.warm_from_sano(sano_ref.state_dict())
                with torch.no_grad():
                    if "enc.layers.1.c2.weight" in sd and self.am.enc.ublock.c2.weight.shape == sd["enc.layers.1.c2.weight"].shape:
                        self.am.enc.ublock.c2.weight.copy_(sd["enc.layers.1.c2.weight"].float())
            else:
                # Copy matching or sliceable weights (e.g. ff_enc=168 from 224 in t5, or nb_vel=9 from 12 in t15)
                dst_sd = self.am.state_dict()
                with torch.no_grad():
                    for k, dst_t in dst_sd.items():
                        if k in sd:
                            src_t = sd[k]
                            if dst_t.shape == src_t.shape:
                                dst_t.copy_(src_t)
                            elif dst_t.ndim == src_t.ndim:
                                sl = tuple(slice(0, min(a, b)) for a, b in zip(dst_t.shape, src_t.shape))
                                dst_t[sl].copy_(src_t[sl])
                self.am.load_state_dict(dst_sd, strict=False)
                if not os.path.exists(v3_t30_p) or am_p != v3_t30_p:
                    _calibrate_styles(self.am, pre_baked=pre_baked)
            self.am.stats.copy_(st_vec)
            self.mel_stats.copy_(st_vec)

        # Load GAN-adapted Sakhiya-Pro v3 vocoder if present, else baseline vocoder
        voc_v3_p = os.path.join(ck_dir, "v3_voc320.pt" if c["voc_ch"] == 320 else "v3_voc144.pt")
        voc_base_p = os.path.join(ck_dir, "v5_c1_voc.pt" if c["voc_ch"] == 320 else "v14_voc144.pt")
        voc_p = voc_v3_p if os.path.exists(voc_v3_p) else voc_base_p
        if os.path.exists(voc_p):
            vck = torch.load(voc_p, map_location=dev, weights_only=False)
            vsd = {k: v.float() for k, v in vck.get("G_ema", vck.get("G", vck)).items()}
            vtmp = VocoderV2(ch=c["voc_ch"], res_kernels=(3, 7, 11)).to(dev)
            if any("parametrizations" in k or "weight_g" in k for k in vsd):
                vtmp.load_state_dict(vsd, strict=False)
                vtmp.fuse_weight_norm()
            else:
                vtmp.fuse_weight_norm()
                vtmp.load_state_dict(vsd, strict=False)
            self.voc.load_state_dict(vtmp.state_dict(), strict=True)

        self._apply_svd_structure()
        if self.tier == "t1" and os.path.exists(os.path.join(ck_dir, "t1_am_pretrained.pt")):
            try:
                ck_p1 = torch.load(os.path.join(ck_dir, "t1_am_pretrained.pt"), map_location=dev, weights_only=False)
                sd_p1 = {k: v.float() for k, v in ck_p1.get("ema", ck_p1).items() if k.startswith(("am.", "voc."))}
                self.load_state_dict(sd_p1, strict=False)
            except Exception:
                pass
        # If pre-adapted SVD vocoder exists for t1, load it after _apply_svd_structure
        if self.tier == "t1" and os.path.exists(os.path.join(ck_dir, "v3_voc144_t1.pt")):
            try:
                vck_t1 = torch.load(os.path.join(ck_dir, "v3_voc144_t1.pt"), map_location=dev, weights_only=False)
                self.voc.load_state_dict({k: v.float() for k, v in vck_t1.get("G_ema", vck_t1).items()}, strict=False)
            except Exception:
                pass
        self.to(dev)
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
        # Grounded Kathmandu/Nepal-native downward clause ending + gentle natural phrase arch (active on both Nepali & Sanskrit!)
        tail_closure = self.closure_slope.clamp(-0.30, -0.05) * ((phase - 0.72).clamp(min=0.0) / 0.28) * (1.0 - is_pause)
        p_arch = (0.14 * torch.sin(phase * 3.14159265) + tail_closure) * (1.0 - is_pause) - 0.15 * is_pause
        p_arch = F.avg_pool1d(F.pad(p_arch[:, None, :], (1, 1), mode="replicate"), 3, stride=1)[:, 0]
        p_enc = (p_arch + 0.35 * pitch_shift[:, None]).clamp(-0.48, 0.48)

        ones_m = torch.ones((B, 1), dtype=torch.float32, device=ids.device)
        h, mu, g = self.am.enc(ids, spk0, lang_eff, sty, studio, im, pitch_add=p_enc, prosody=pros, sty_mask=ones_m)
        if hasattr(self.am.enc, "energy_pred"):
            eng_hat = self.am.enc.energy_pred(h).squeeze(-1).clamp(-1.2, 1.2)
            mu = mu + (0.10 * pros[:, 2:3] + 0.06 * eng_hat)[..., None] * 0.32

        x_d = self.am.dur.pred_logdur(ids, h, im) + self.dur_rate_bias.clamp(-0.12, 0.12)
        dur_frames = torch.exp(x_d).clamp(min=1.0, max=60.0)
        rate_mult = torch.exp(-0.04 * pros[:, 0:1]) * (1.0 - is_pause) + torch.exp(0.08 * pros[:, 3:4]) * is_pause
        dur_frames = (dur_frames * rate_mult * dur_scale[:, None]).clamp(min=1.0, max=60.0)
        # Protect sentence-final tokens and short Nepali function words from frame collapse
        is_tail = ((pos >= (L_f - 3.0)) & (ids > 6)).float()
        dur = torch.round(torch.maximum(dur_frames, is_tail * 5.0 + (1.0 - is_tail) * 1.0)).clamp(min=1.0, max=60.0)
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
        env = F.avg_pool1d(F.pad(mel.transpose(1, 2), (3, 3), mode="replicate"), 7, stride=1).transpose(1, 2)
        kb = torch.arange(100, device=mel.device, dtype=mel.dtype)
        wk = ((64.0 - kb) / 18.0).clamp(0.0, 1.0)[None, :, None]
        comb_gain = 0.12 if self.steps == 1 else 0.08
        mel_w = env + (mel - env) * (1.0 + comb_gain * wk)
        if self.arch == "sakhiya_pico":
            mel_mu = mel_w.mean(dim=(1, 2), keepdim=True)
            mel_w = (mel_w - mel_mu) * 1.04 + mel_mu - 0.04
            env_f = F.avg_pool1d(F.pad(mel_w.transpose(1, 2), (2, 2), mode="replicate"), 5, stride=1).transpose(1, 2)
            wf = ((72.0 - kb) / 24.0).clamp(0.0, 1.0)[None, :, None]
            mel_w = mel_w + 0.14 * wf * (mel_w - env_f)
        mel_clean = torch.cat([mel_w[:, :98, :], mel_w[:, 97:98, :], mel_w[:, 97:98, :]], dim=1)
        ext_shift = (0.35 * pitch_shift[:, None, None]).clamp(-0.85, 0.85).expand(B, 1, T)

        y, _, _, _, _ = self.voc_wav(mel_clean, shift_st=ext_shift)
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
    tot = 0
    for name, p in m.named_parameters():
        if deployed and name.startswith(("am.ctc_rec.", "am.enc.spk_adv.")):
            continue
        tot += p.numel()
    return tot
