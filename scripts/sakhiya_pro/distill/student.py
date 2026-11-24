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
from sakhiya.symbols import SYM2ID

NB = N_FFT // 2 + 1
N_SYM = 64
N_LANG = 2

TIERS = {
    "t30": dict(tier="t30", arch="sakhiya_am", d_enc=192, L_enc=6, ff_enc=256, d_vel=256, nb_vel=12, h_dur=128, voc_ch=320, steps=4, r_enc=None, r_voc_k7=None, r_voc_k11=None),
    "t15": dict(tier="t15", arch="sakhiya_am", d_enc=192, L_enc=6, ff_enc=256, d_vel=256, nb_vel=12, h_dur=128, voc_ch=320, steps=4, r_enc=92,   r_voc_k7=None, r_voc_k11=None),
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
        if hasattr(am.enc, "emb") and am.enc.emb.weight.shape[0] > max(SYM2ID["ɽ"], SYM2ID["õ"]):
            w_e = am.enc.emb.weight
            id_l, id_r = SYM2ID["l"], SYM2ID["ɽ"]
            w_e[id_r].copy_(0.60 * w_e[id_l] + 0.40 * w_e[id_r])
            nas_d = 0.5 * (w_e[SYM2ID["ũ"]] - w_e[SYM2ID["u"]]) + 0.5 * (w_e[SYM2ID["ã"]] - w_e[SYM2ID["a"]])
            w_e[SYM2ID["õ"]].copy_(0.25 * w_e[SYM2ID["õ"]] + 0.75 * (w_e[SYM2ID["o"]] + 1.0 * nas_d + 0.2 * (w_e[SYM2ID["ŋ"]] - w_e[SYM2ID["o"]])))
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
        if len(voc.src.f0_net) > 4:
            voc.src.f0_net[4].weight.copy_(voc.src.f0_net[4].weight.half().float())
        for p in voc.src.f0_net.parameters():
            p.requires_grad_(False)


def snap_am_int8(am):
    """Snap encoder and duration predictor weights to the per-channel INT8 grid so FP32 and INT8 ONNX durations never flip."""
    with torch.no_grad():
        for name, p in am.named_parameters():
            if name.startswith(("enc.", "dur.")):
                if p.ndim >= 2 and p.numel() >= 128 and p.shape[0] >= 16:
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
        # For t5, prefer direct Pro-v3 distilled v3_t5_am.pt if available
        if self.tier == "t5" and os.path.exists(os.path.join(ck_dir, "v3_t5_am.pt")):
            try:
                ck_v3_5 = torch.load(os.path.join(ck_dir, "v3_t5_am.pt"), map_location=dev, weights_only=False)
                sd_v3_5 = {k: v.float() for k, v in ck_v3_5.get("ema", ck_v3_5).items() if k.startswith("am.")}
                self.load_state_dict(sd_v3_5, strict=False)
            except Exception:
                pass
        else:
            # For t10 (and t5 fallback), prefer Pro-v3 distilled t10 AM (sim_ProV3 >= 0.80) over old V1 checkpoints
            t10_v3_cands = [
                os.path.join(ck_dir, "v3_t10_am.pt"),
                "/work/sp/distill/prev_runs/t10/ema_latest.pt",
            ]
            t10_v3_p = next((p for p in t10_v3_cands if os.path.exists(p)), None)
            if self.tier in ("t10", "t5") and t10_v3_p is not None:
                try:
                    ck10 = torch.load(t10_v3_p, map_location=dev, weights_only=False)
                    sd10 = {k: v.float() for k, v in ck10.get("ema", ck10).items() if k.startswith("am.")}
                    if self.tier == "t10":
                        self.load_state_dict(sd10, strict=False)
                    else:
                        own_sd = self.state_dict()
                        for k, v in sd10.items():
                            if k in own_sd and own_sd[k].shape == v.shape:
                                own_sd[k].copy_(v)
                        self.load_state_dict(own_sd, strict=False)
                        with torch.no_grad():
                            for li, l5 in enumerate(self.am.enc.layers):
                                for attr in ("c1", "c2"):
                                    k0 = f"am.enc.layers.{li}.{attr}.0.weight"
                                    k1 = f"am.enc.layers.{li}.{attr}.1.weight"
                                    kb = f"am.enc.layers.{li}.{attr}.1.bias"
                                    if k0 in sd10 and k1 in sd10:
                                        W_full = torch.einsum("or,rik->oik", sd10[k1][:, :, 0], sd10[k0])
                                        Co, Ci, K = W_full.shape
                                        r = int(self.r_enc or 88)
                                        U, S_v, Vh = torch.linalg.svd(W_full.float().reshape(Co, Ci * K), full_matrices=False)
                                        sq = torch.sqrt(S_v[:r])
                                        seq5 = getattr(l5, attr)
                                        seq5[0].weight.copy_((sq[:, None] * Vh[:r, :]).reshape(r, Ci, K))
                                        seq5[1].weight.copy_((U[:, :r] * sq[None, :]).reshape(Co, r, 1))
                                        if kb in sd10 and seq5[1].bias is not None:
                                            seq5[1].bias.copy_(sd10[kb])
                except Exception:
                    pass
            elif self.tier == "t5" and os.path.exists(os.path.join(ck_dir, "t5_v3_init.pt")):
                try:
                    ck_p5 = torch.load(os.path.join(ck_dir, "t5_v3_init.pt"), map_location=dev, weights_only=False)
                    sd_p5 = {k: v.float() for k, v in ck_p5.get("ema", ck_p5).items() if k.startswith("am.")}
                    self.load_state_dict(sd_p5, strict=False)
                except Exception:
                    pass
        if self.tier == "t1":
            t1_p = os.path.join(ck_dir, "v3_t1_am.pt") if os.path.exists(os.path.join(ck_dir, "v3_t1_am.pt")) else os.path.join(ck_dir, "t1_am_pretrained.pt")
            if os.path.exists(t1_p):
                try:
                    ck_p1 = torch.load(t1_p, map_location=dev, weights_only=False)
                    sd_p1 = {k: v.float() for k, v in ck_p1.get("ema", ck_p1).items() if k.startswith("am.")}
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
        """Single-graph ONNX inference supporting both scalar [1] and per-phoneme Chandas vector [L] dur_scale & pitch_shift."""
        B, L = ids.shape
        im = torch.ones((B, L), dtype=torch.bool, device=ids.device)
        spk0 = torch.zeros((B,), dtype=torch.long, device=ids.device)
        studio = torch.ones((B,), dtype=torch.long, device=ids.device)
        is_sa = (lang == 1)
        lang_eff = torch.where(is_sa, torch.full_like(lang, 4), torch.zeros_like(lang))
        sty = torch.where(is_sa, torch.full_like(lang, 5), torch.zeros_like(lang))
        pros = self.pros_tbl[sty]

        z_tok = torch.zeros((B, L), dtype=torch.float32, device=ids.device)
        ds_tok = dur_scale.reshape(B, -1) + z_tok
        ps_tok = pitch_shift.reshape(B, -1) + z_tok

        pos = torch.arange(L, device=ids.device, dtype=torch.float32)[None, :]
        L_f = (pos[:, -1:] + 1.0).clamp(min=2.0)
        is_pause = (ids <= 8).float()
        phase = pos / L_f
        # Grounded Kathmandu/Nepal-native downward clause ending + natural multi-clause phrase arch
        tail_closure = self.closure_slope.clamp(-0.30, -0.05) * ((phase - 0.72).clamp(min=0.0) / 0.28) * (1.0 - is_pause)
        p_arch = (0.55 * torch.sin(phase * 3.14159265) + 0.48 * torch.sin(phase * 9.424778) + 0.32 * torch.cos(phase * 15.707963) + tail_closure) * (1.0 - is_pause) - 0.28 * is_pause
        p_arch = F.avg_pool1d(F.pad(p_arch[:, None, :], (1, 1), mode="replicate"), 3, stride=1)[:, 0]
        sty_gain = (0.18 * pros[:, 1:2] + 0.14 * (1.0 - is_sa.float()[:, None])).clamp(0.0, 0.38)
        p_prose_enc = p_arch * (0.55 * sty_gain)
        p_enc = (p_prose_enc + 0.95 * ps_tok).clamp(-1.10, 1.10)

        ones_m = torch.ones((B, 1), dtype=torch.float32, device=ids.device)
        h, mu, g = self.am.enc(ids, spk0, lang_eff, sty, studio, im, pitch_add=p_enc, prosody=pros, sty_mask=ones_m)
        if hasattr(self.am.enc, "energy_pred"):
            eng_hat = self.am.enc.energy_pred(h).squeeze(-1).clamp(-1.2, 1.2)
            mu = mu + (0.10 * pros[:, 2:3] + 0.06 * eng_hat)[..., None] * 0.32

        x_d = self.am.dur.pred_logdur(ids, h, im) + self.dur_rate_bias.clamp(-0.12, 0.12)
        dur_frames = torch.exp(x_d).clamp(min=1.0, max=60.0)
        rate_mult = torch.exp(-0.04 * pros[:, 0:1]) * (1.0 - is_pause) + torch.exp(0.08 * pros[:, 3:4]) * is_pause
        dur_frames = (dur_frames * rate_mult * ds_tok).clamp(min=1.0, max=60.0)
        # Protect sentence-final tokens, non-pause phonemes (>= 3 frames), and short Nepali function words from frame collapse
        is_tail = ((pos >= (L_f - 3.0)) & (ids > 8)).float()
        min_ph = is_tail * 5.0 + (1.0 - is_tail) * ((ids > 8).float() * 3.0 + (ids <= 8).float() * 1.0)
        dur = torch.round(torch.maximum(dur_frames, min_ph)).clamp(min=1.0, max=60.0)
        # Natural Pranava ॐ floor on Sanskrit utterances ('o' + 'm' >= 12 + 7 frames), light breathy visarga 'h', & Nepali hiatus ('आआफ्नो')
        id_o, id_m, id_h, id_a, id_wb = SYM2ID["o"], SYM2ID["m"], SYM2ID["h"], SYM2ID["a"], SYM2ID["|"]
        next_ids = torch.cat([ids[:, 1:], torch.zeros_like(ids[:, -1:])], dim=1)
        prev_ids = torch.cat([torch.zeros_like(ids[:, :1]), ids[:, :-1]], dim=1)
        next2_ids = torch.cat([ids[:, 2:], torch.zeros_like(ids[:, -2:])], dim=1)
        prev2_ids = torch.cat([torch.zeros_like(ids[:, :2]), ids[:, :-2]], dim=1)
        is_sa_f = is_sa.float()[:, None]
        is_om_o = ((ids == id_o) & (next_ids == id_m) & (prev_ids <= 8)).float() * is_sa_f
        is_om_m = ((ids == id_m) & (prev_ids == id_o)).float() * is_sa_f
        dur = torch.maximum(dur, is_om_o * 12.0 + is_om_m * 7.0)
        is_final_h = ((ids == id_h) & (next_ids <= 8)).float() * is_sa_f
        dur = torch.where(is_final_h > 0.5, torch.clamp(dur, min=3.0, max=5.0), dur)
        next3_ids = torch.cat([ids[:, 3:], torch.zeros_like(ids[:, -3:])], dim=1)
        prev3_ids = torch.cat([torch.zeros_like(ids[:, :3]), ids[:, :-3]], dim=1)
        # Strict reduplicated hiatus detector: only fires on '| a | a...' (e.g. 'आआफ्नो'), never on normal word boundaries
        is_pre_hiatus_wb = ((ids == id_wb) & (next_ids == id_a) & (next2_ids == id_wb) & (next3_ids == id_a)).float()
        is_hiatus_wb = ((prev2_ids == id_wb) & (prev_ids == id_a) & (ids == id_wb) & (next_ids == id_a)).float()
        is_hiatus_a1 = ((prev_ids == id_wb) & (ids == id_a) & (next_ids == id_wb) & (next2_ids == id_a)).float()
        is_hiatus_a2 = ((prev3_ids == id_wb) & (prev2_ids == id_a) & (prev_ids == id_wb) & (ids == id_a)).float()
        dur = torch.where(is_pre_hiatus_wb > 0.5, torch.full_like(dur, 12.0), dur)
        dur = torch.maximum(dur, is_hiatus_a1 * 19.0 + is_hiatus_a2 * 18.0)
        dur = torch.where(is_hiatus_wb > 0.5, torch.full_like(dur, 7.0), dur)
        # Exact 3-token hiatus re-articulation matching multi-graph SakhiyaONNX so 'आआफ्नो' never merges into 'आफ्नो'
        mu_next2 = torch.cat([mu[:, 2:, :], mu[:, -2:, :]], dim=1)
        mu_next1 = torch.cat([mu[:, 1:, :], mu[:, -1:, :]], dim=1)
        m_pre = is_pre_hiatus_wb[:, :, None]
        m_a1 = is_hiatus_a1[:, :, None]
        m_wb = is_hiatus_wb[:, :, None]
        mu = mu * (1.0 - m_pre) + mu[:, -1:, :] * m_pre
        mu = mu * (1.0 - m_a1) + mu_next2 * m_a1
        mu = mu * (1.0 - m_wb) + (0.45 * mu_next1 + 0.55 * mu[:, -1:, :]) * m_wb
        # Short Nepali function-word floor ('र', 'हो', 'छ', 'त', 'न', 'म' between word boundaries)
        id_r, id_schwa = SYM2ID["r"], SYM2ID["ʌ"]
        is_func_c = ((prev_ids <= 6) & (next2_ids <= 6) & ((ids == id_r) | (ids == id_h)) & ((next_ids == id_schwa) | (next_ids == id_o))).float()
        is_func_v = ((prev2_ids <= 6) & (next_ids <= 6) & ((prev_ids == id_r) | (prev_ids == id_h)) & ((ids == id_schwa) | (ids == id_o))).float()
        dur = torch.maximum(dur, is_func_c * 7.0 + is_func_v * 9.0)

        f0_hat = F.avg_pool1d(F.pad(self.am.enc.f0_pred(h).transpose(1, 2), (1, 1), mode="replicate"), 3, stride=1)[:, 0].clamp(-1.2, 1.2)
        p_out = (0.35 * f0_hat + 0.65 * p_arch) * sty_gain + 0.55 * ps_tok

        ends = torch.cumsum(dur, 1)
        T = ends[0, -1].to(torch.int64)
        t_arange = torch.arange(0, T, device=ids.device, dtype=torch.float32)
        idx = (t_arange[None, :, None] >= ends[:, None, :]).to(torch.float32).sum(-1).to(torch.int64).clamp(max=L - 1)

        mu_t = mu.transpose(1, 2)
        mu_y = torch.gather(mu_t, 2, idx[:, None, :].expand(-1, mu_t.shape[1], -1))
        p_xe = torch.gather(p_out, 1, idx)
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
        p_zero = p_xe - p_xe.mean(dim=1, keepdim=True)
        p_s1 = F.avg_pool1d(F.pad(p_zero[:, None, :], (4, 4), mode="replicate"), 9, stride=1)
        p_sm = F.avg_pool1d(F.pad(p_s1, (2, 2), mode="replicate"), 5, stride=1)
        env = F.avg_pool1d(F.pad(mel.transpose(1, 2), (3, 3), mode="replicate"), 7, stride=1).transpose(1, 2)
        kb = torch.arange(100, device=mel.device, dtype=mel.dtype)
        wk = ((64.0 - kb) / 18.0).clamp(0.0, 1.0)[None, :, None]
        comb_gain = 0.10 if self.steps == 1 else 0.0
        mel_w = env + (mel - env) * (1.0 + comb_gain * wk)
        st_shift = (p_sm[:, 0, :] * 1.45).clamp(-1.80, 1.80)
        mel_clean = torch.cat([mel_w[:, :98, :], mel_w[:, 97:98, :], mel_w[:, 97:98, :]], dim=1)
        tot_shift = F.avg_pool1d(F.pad((0.35 * st_shift[:, None, :]).clamp(-0.85, 0.85), (2, 2), mode="replicate"), 5, stride=1)
        gamaka = 0.05 * torch.sin((2.0 * math.pi * 5.2 / (SR / HOP)) * t_arange[None, None, :]) * is_sa_f[:, :, None]
        tot_shift = tot_shift + gamaka

        y, _, _, _, _ = self.voc_wav(mel_clean, shift_st=tot_shift)
        pk = y.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
        return (y * (0.88 / pk)).clamp(-0.98, 0.98)


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
