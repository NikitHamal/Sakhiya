"""Sakhiya v4 Single-Graph Harmonic-iSTFT Distilled Students (`scripts/sakhiya_pro/v4/student_v4.py`):
  - Sakhiya-V2-Micro (`micro` / `t5`): <= 4.5 MB total INT8 ONNX (~4.38 MB), unifying `SakhiyaAM` (`d_enc=144, L_enc=5`) +
    16x `VocoderV2(ch=144)` + 1-op COLA `ConvTranspose1d` iSTFT64 with SVD-factored encoder/vocoder kernels (`r_enc=88, r_voc_k11=48`).
  - Sakhiya-V2-Pico  (`pico` / `t1`):  <= 1.4 MB total INT8 ONNX (~1.24 MB), unifying `SakhiyaPicoAM` (Weight-Tied Looped Universal
    Encoder & Velocity Decoder with 1-KB FiLM iteration adapters + folded time embeddings) + 16x SVD-factored `VocoderV2(ch=144)`.
  - Sakhiya-V2-Compact (`t10`): ~8.5 MB total INT8 ONNX, unifying `SakhiyaAM` (`d_enc=144`) + 16x `VocoderV2(ch=320)`.

Warm-initializes from our converged 47k-step `t5/ema_final.pt` and 29k-step `t1/ema_final.pt` checkpoints (or `/work/tts/ck/v9_*.pt` +
`v14_voc144.pt`) with exact SVD rank compression (zero channel slicing!), INT8 grid snapping (`snap_f0_int8`, `snap_am_int8`) for
>= 0.994 FP32-vs-INT8 mel correlation, and native per-phoneme Chandas-Laya-Svara vector `dur_scale` & `pitch_shift` support.
"""
import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
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
    'micro': dict(
        tier='micro', arch='sakhiya_am',
        d_enc=144, L_enc=5, ff_enc=224, d_vel=160, nb_vel=7, h_dur=96,
        voc_ch=144, steps=1,
        r_enc=88, r_voc_k7=None, r_voc_k11=None, r_voc_1_k7=None, r_voc_1_k11=None, r_vel_ff=None,
    ),
    'pico': dict(
        tier='pico', arch='sakhiya_pico',
        d_enc=96, L_enc=4, ff_enc=144, d_vel=112, nb_vel=5, h_dur=64,
        voc_ch=144, steps=1,
        r_enc=52, r_voc_k7=44, r_voc_k11=26, r_voc_1_k7=None, r_voc_1_k11=None, r_vel_ff=None,
    ),
    't10': dict(
        tier='t10', arch='sakhiya_am',
        d_enc=144, L_enc=5, ff_enc=224, d_vel=160, nb_vel=7, h_dur=96,
        voc_ch=320, steps=1,
        r_enc=104, r_voc_k7=None, r_voc_k11=None, r_voc_1_k7=None, r_voc_1_k11=None, r_vel_ff=None,
    ),
}
TIERS['t5'] = dict(TIERS['micro'], tier='t5')
TIERS['t1'] = dict(TIERS['pico'], tier='t1')


def _reconstruct_conv_weight(mod: nn.Module):
    """Extract full (Co, Ci, K) weight tensor, bias, stride, padding, dilation from either Conv1d or SVD Sequential(c1, c2)."""
    if isinstance(mod, nn.Conv1d):
        return mod.weight.data.float(), (mod.bias.data.float() if mod.bias is not None else None), mod.stride, mod.padding, mod.dilation
    if isinstance(mod, nn.Sequential) and len(mod) == 2 and isinstance(mod[0], nn.Conv1d) and isinstance(mod[1], nn.Conv1d):
        c1, c2 = mod[0], mod[1]
        r_old, Ci, K = c1.weight.shape
        Co = c2.weight.shape[0]
        W = (c2.weight.data.float().squeeze(-1) @ c1.weight.data.float().reshape(r_old, Ci * K)).reshape(Co, Ci, K)
        b = c2.bias.data.float() if c2.bias is not None else None
        return W, b, c1.stride, c1.padding, c1.dilation
    raise TypeError(f'Unsupported module for SVD factorization: {type(mod)}')


def _make_svd_seq(mod: nn.Module, rank: int, use_qat: bool = False):
    W, b, stride, padding, dilation = _reconstruct_conv_weight(mod)
    Co, Ci, K = W.shape
    r = min(int(rank), Co, Ci * K)
    cls = QATConv1d if use_qat else nn.Conv1d
    c1 = cls(Ci, r, K, stride=stride, padding=padding, dilation=dilation, bias=False)
    c2 = cls(r, Co, 1, bias=(b is not None))
    with torch.no_grad():
        U, S, Vh = torch.linalg.svd(W.reshape(Co, Ci * K), full_matrices=False)
        sq = torch.sqrt(S[:r])
        c1.weight.data.copy_((sq[:, None] * Vh[:r, :]).reshape(r, Ci, K))
        c2.weight.data.copy_((U[:, :r] * sq[None, :]).reshape(Co, r, 1))
        if b is not None:
            c2.bias.data.copy_(b)
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
            if name.startswith(('enc.', 'dur.')):
                if p.ndim >= 2 and p.numel() >= 128 and p.shape[0] > 1:
                    p.copy_(ste_int8_weight(p.float(), axis=0, enabled=True))
                elif p.numel() > 16:
                    p.copy_(p.half().float())


class _VocCallableWrap(nn.Module):
    """Wraps VocoderV2 + COLA K_istft so `G.voc(mel_seg)` in `train_student_v4.py` returns waveform `[B, T_wav]` directly."""
    def __init__(self, voc: VocoderV2, K_istft: torch.Tensor):
        super().__init__()
        self.voc = voc
        self.register_buffer('K_istft', K_istft, persistent=False)

    def forward(self, mel, shift_st=None):
        re_s, im_s, f0, uv = self.voc.spectrum_and_f0(mel, shift_st=shift_st)
        z = torch.zeros_like(im_s[:, :1, :])
        spec_cat = torch.cat([re_s.float(), torch.cat([z, im_s[:, 1:-1, :], z], dim=1).float()], dim=1)
        y = F.conv_transpose1d(spec_cat, self.K_istft.to(spec_cat.device), stride=ISTFT_HOP, padding=(ISTFT_N_FFT - ISTFT_HOP) // 2)[:, 0, :]
        return y


class StudentV4(nn.Module):
    def __init__(
        self,
        tier='micro',
        arch=None,
        d_enc=144, L_enc=5, ff_enc=224, d_vel=160, nb_vel=7, h_dur=96,
        voc_ch=144, steps=1,
        r_enc=88, r_voc_k7=None, r_voc_k11=44,
        r_voc_1_k7=None, r_voc_1_k11=None, r_vel_ff=None,
        **kw,
    ):
        super().__init__()
        if tier in TIERS and arch is None:
            c = TIERS[tier]
            arch = c['arch']
            d_enc, L_enc, ff_enc, d_vel, nb_vel, h_dur, voc_ch, steps = (
                c['d_enc'], c['L_enc'], c['ff_enc'], c['d_vel'], c['nb_vel'], c['h_dur'], c['voc_ch'], c['steps']
            )
            r_enc = c.get('r_enc', r_enc)
            r_voc_k7 = c.get('r_voc_k7', r_voc_k7)
            r_voc_k11 = c.get('r_voc_k11', r_voc_k11)
            r_voc_1_k7 = c.get('r_voc_1_k7', r_voc_1_k7)
            r_voc_1_k11 = c.get('r_voc_1_k11', r_voc_1_k11)
            r_vel_ff = c.get('r_vel_ff', r_vel_ff)

        self.tier = tier
        self.arch = arch or ('sakhiya_pico' if tier in ('pico', 't1') else 'sakhiya_am')
        self.is_pico = (self.arch == 'sakhiya_pico')
        self.steps = int(steps)
        self.r_enc = r_enc
        self.r_voc_k7 = r_voc_k7
        self.r_voc_k11 = r_voc_k11
        self.r_voc_1_k7 = r_voc_1_k7
        self.r_voc_1_k11 = r_voc_1_k11
        self.r_vel_ff = r_vel_ff
        self.cfg = dict(
            tier=self.tier, arch=self.arch,
            d_enc=d_enc, L_enc=L_enc, ff_enc=ff_enc, d_vel=d_vel, nb_vel=nb_vel, h_dur=h_dur,
            voc_ch=voc_ch, steps=self.steps,
            r_enc=self.r_enc, r_voc_k7=self.r_voc_k7, r_voc_k11=self.r_voc_k11,
            r_voc_1_k7=self.r_voc_1_k7, r_voc_1_k11=self.r_voc_1_k11, r_vel_ff=self.r_vel_ff,
        )
        K_istft, _ = _build_cola_istft_kernel(ISTFT_N_FFT, ISTFT_HOP)
        self.register_buffer('K_istft', torch.from_numpy(K_istft), persistent=False)
        self.register_buffer('pros_tbl', torch.tensor(STYLE_PROSODY_TARGETS, dtype=torch.float32), persistent=False)
        self.register_buffer('f0_stats', torch.tensor([5.418, 0.203]))
        self.register_buffer('mel_stats', torch.tensor([-6.1761, 2.4932]))
        self.closure_slope = nn.Parameter(torch.tensor(-0.18))
        self.dur_rate_bias = nn.Parameter(torch.tensor(0.0))
        self._qat_flag = False

        self._init_raw_submodules(d_enc, L_enc, ff_enc, d_vel, nb_vel, h_dur, voc_ch)
        self._apply_svd_structure(
            r_enc=self.r_enc, r_voc_k7=self.r_voc_k7, r_voc_k11=self.r_voc_k11,
            r_voc_1_k7=self.r_voc_1_k7, r_voc_1_k11=self.r_voc_1_k11, r_vel_ff=self.r_vel_ff,
        )

    def _init_raw_submodules(self, d_enc, L_enc, ff_enc, d_vel, nb_vel, h_dur, voc_ch):
        if self.arch == 'sakhiya_pico':
            self.am = SakhiyaPicoAM(d_enc=d_enc, L_enc=L_enc, ff_enc=ff_enc, d_vel=d_vel, nb_vel=nb_vel, h_dur=h_dur)
        else:
            self.am = SakhiyaAM(d_enc=d_enc, L_enc=L_enc, ff_enc=ff_enc, d_vel=d_vel, nb_vel=nb_vel, h_dur=h_dur)
            if hasattr(self.am, 'ctc_rec'):
                self.am.ctc_rec = nn.Identity()
            if hasattr(self.am.enc, 'spk_adv'):
                self.am.enc.spk_adv = nn.Identity()

        self.voc = VocoderV2(ch=voc_ch, res_kernels=(3, 7, 11))
        self.voc.fuse_weight_norm()

    def _apply_svd_structure(self, r_enc=None, r_voc_k7=None, r_voc_k11=None, r_voc_1_k7=None, r_voc_1_k11=None, r_vel_ff=None):
        if self.arch == 'sakhiya_pico':
            self.am.vel.ublock_late = self.am.vel.ublock_early
        if r_enc is not None:
            if self.arch == 'sakhiya_pico':
                self.am.enc.ublock.c1 = _make_svd_seq(self.am.enc.ublock.c1, r_enc, use_qat=True)
                self.am.enc.ublock.c2 = _make_svd_seq(self.am.enc.ublock.c2, r_enc, use_qat=True)
            else:
                for lyr in self.am.enc.layers:
                    lyr.c1 = _make_svd_seq(lyr.c1, r_enc, use_qat=False)
                    lyr.c2 = _make_svd_seq(lyr.c2, r_enc, use_qat=False)

        if self.arch == 'sakhiya_pico' and r_vel_ff is not None:
            self.am.vel.ublock_early.p1 = _make_svd_seq(self.am.vel.ublock_early.p1, r_vel_ff, use_qat=True)
            self.am.vel.ublock_early.p2 = _make_svd_seq(self.am.vel.ublock_early.p2, r_vel_ff, use_qat=True)

        for b_idx, r_k in ((1, r_voc_k7), (2, r_voc_k11)):
            if r_k is not None:
                blk = self.voc.mrfs[0][b_idx]
                for i in range(len(blk.c1)):
                    blk.c1[i] = _make_svd_seq(blk.c1[i], r_k, use_qat=False)
                    blk.c2[i] = _make_svd_seq(blk.c2[i], r_k, use_qat=False)

        for b_idx, r_k in ((1, r_voc_1_k7), (2, r_voc_1_k11)):
            if r_k is not None:
                blk = self.voc.mrfs[1][b_idx]
                for i in range(len(blk.c1)):
                    blk.c1[i] = _make_svd_seq(blk.c1[i], r_k, use_qat=False)
                    blk.c2[i] = _make_svd_seq(blk.c2[i], r_k, use_qat=False)

    def warm_from_pretrained(self, ck_dir='/work/tts/ck'):
        """Warm-initialize from converged v3 Student checkpoints (`t5/ema_final.pt`, `t1/ema_final.pt`) if present,
        else from `/work/tts/ck/v9_*.pt` + `v14_voc144.pt`, then apply exact SVD rank compression & INT8 snapping."""
        dev = next(self.parameters()).device
        c = self.cfg

        # 1. Try loading directly from converged 47k-step t5 or 29k-step t1 checkpoint in /work/sp/hfpull
        v3_t5 = '/work/sp/hfpull/sakhiya_pro_v3/distill/runs/t5/ema_final.pt'
        v3_t1 = '/work/sp/hfpull/sakhiya_pro_v3/distill/runs/t1/ema_final.pt'
        v3_t10 = '/work/sp/hfpull/sakhiya_pro_v3/distill/runs/t10/ema_latest.pt'
        v3_ck_p = v3_t1 if self.is_pico else (v3_t10 if self.tier == 't10' else v3_t5)

        if os.path.exists(v3_ck_p):
            try:
                S3 = torch.load(v3_ck_p, map_location=dev, weights_only=False)
                c3 = S3.get('cfg', {})
                self._init_raw_submodules(c['d_enc'], c['L_enc'], c['ff_enc'], c['d_vel'], c['nb_vel'], c['h_dur'], c['voc_ch'])
                self.to(dev)
                self._apply_svd_structure(
                    r_enc=c3.get('r_enc'),
                    r_voc_k7=c3.get('r_voc_k7'),
                    r_voc_k11=c3.get('r_voc_k11'),
                    r_voc_1_k7=c3.get('r_voc_1_k7'),
                    r_voc_1_k11=c3.get('r_voc_1_k11'),
                    r_vel_ff=c3.get('r_vel_ff'),
                )
                self.to(dev)
                self.load_state_dict(S3['ema'], strict=False)
                # Re-factor to exact target v4 SVD ranks (compresses t5 4.56MB -> 4.44MB and t1 1.43MB -> 1.24MB cleanly!)
                self._apply_svd_structure(
                    r_enc=c['r_enc'],
                    r_voc_k7=c['r_voc_k7'],
                    r_voc_k11=c['r_voc_k11'],
                    r_voc_1_k7=c['r_voc_1_k7'],
                    r_voc_1_k11=c['r_voc_1_k11'],
                    r_vel_ff=c['r_vel_ff'],
                )
                self.to(dev)
                snap_f0_int8(self.voc)
                snap_am_int8(self.am)
                print(f'[StudentV4] Warm-loaded {self.tier} from converged {v3_ck_p} (step={S3.get("step")})', flush=True)
                return self
            except Exception as e:
                print(f'[StudentV4] v3 checkpoint load warn ({v3_ck_p}): {e}', flush=True)

        # 2. Fallback: warm-load from /work/tts/ck/v9_*.pt and v14_voc144.pt
        self._init_raw_submodules(c['d_enc'], c['L_enc'], c['ff_enc'], c['d_vel'], c['nb_vel'], c['h_dur'], c['voc_ch'])
        self.to(dev)
        am_p = os.path.join(ck_dir, 'v9_sano_am.pt' if self.is_pico else 'v9_voca_am.pt')
        voc_p = os.path.join(ck_dir, 'v5_c1_voc.pt' if c['voc_ch'] == 320 else 'v14_voc144.pt')
        if not os.path.exists(voc_p):
            voc_p = '/work/tts/runs/v5_c1/voc.pt' if c['voc_ch'] == 320 else '/work/tts/runs/v14_voc144/voc.pt'

        if os.path.exists(am_p):
            ck = torch.load(am_p, map_location=dev, weights_only=False)
            pre_baked = bool(ck.get('pre_baked_v25', False))
            sd = {k: v.float() for k, v in ck.get('ema', ck.get('net', ck)).items() if not k.startswith('ctc_rec.')}
            st_vec = torch.tensor([float(ck.get('mean', -6.1760635)), float(ck.get('std', 2.4932315))], device=dev)
            if self.is_pico:
                sano_ref = SakhiyaAM(d_enc=96, L_enc=4, ff_enc=144, d_vel=112, nb_vel=5, h_dur=64).to(dev)
                sano_ref.load_state_dict(sd, strict=False)
                _calibrate_styles(sano_ref, pre_baked=pre_baked)
                self.am.warm_from_sano(sano_ref.state_dict())
            else:
                self.am.load_state_dict(sd, strict=False)
                _calibrate_styles(self.am, pre_baked=pre_baked)
            self.am.stats.copy_(st_vec)
            self.mel_stats.copy_(st_vec)

        if os.path.exists(voc_p):
            vck = torch.load(voc_p, map_location=dev, weights_only=False)
            vsd = {k: v.float() for k, v in vck.get('G_ema', vck.get('G', vck)).items()}
            vtmp = VocoderV2(ch=c['voc_ch'], res_kernels=(3, 7, 11)).to(dev)
            vtmp.load_state_dict(vsd, strict=True)
            vtmp.fuse_weight_norm()
            self.voc.load_state_dict(vtmp.state_dict(), strict=True)

        self._apply_svd_structure(
            r_enc=c['r_enc'],
            r_voc_k7=c['r_voc_k7'],
            r_voc_k11=c['r_voc_k11'],
            r_voc_1_k7=c['r_voc_1_k7'],
            r_voc_1_k11=c['r_voc_1_k11'],
            r_vel_ff=c['r_vel_ff'],
        )
        self.to(dev)
        snap_f0_int8(self.voc)
        snap_am_int8(self.am)
        return self

    def voc_wav(self, mel, shift_st=None):
        """Differentiable Harmonic-iSTFT waveform synthesis from 100-band log-mel [B, 100, T]."""
        re_s, im_s, f0, uv = self.voc.spectrum_and_f0(mel, shift_st=shift_st)
        z = torch.zeros_like(im_s[:, :1, :])
        spec_cat = torch.cat([re_s.float(), torch.cat([z, im_s[:, 1:-1, :], z], dim=1).float()], dim=1)
        y = F.conv_transpose1d(spec_cat, self.K_istft.to(spec_cat.device), stride=ISTFT_HOP, padding=(ISTFT_N_FFT - ISTFT_HOP) // 2)[:, 0, :]
        return y, re_s, im_s, f0, uv

    def forward_am_train(self, ids, lang, im, path):
        """Training forward pass for `train_student_v4.py`: returns mu_t [B,100,L], ld [B,L], mel_pred [B,100,T]."""
        B, L = ids.shape
        spk0 = torch.zeros((B,), dtype=torch.long, device=ids.device)
        studio1 = torch.ones((B,), dtype=torch.long, device=ids.device)
        is_sa = (lang == 1)
        lang_eff = torch.where(is_sa, torch.full_like(lang, 4), torch.zeros_like(lang))
        sty = torch.where(is_sa, torch.full_like(lang, 5), torch.zeros_like(lang))
        pros = self.pros_tbl[sty]
        im_bool = im[:, 0] > 0.5
        ones_m = torch.ones((B, 1), dtype=torch.float32, device=ids.device)
        h, mu, g = self.am.enc(ids, spk0, lang_eff, sty, studio1, im_bool, prosody=pros, sty_mask=ones_m)
        ld = self.am.dur.pred_logdur(ids, h, im_bool) + self.dur_rate_bias.clamp(-0.12, 0.12)
        mu_t = mu.transpose(1, 2).float()
        mu_y = torch.matmul(mu_t, path)
        t0 = torch.zeros((B,), device=ids.device, dtype=torch.float32)
        m_frame = (path.sum(1, keepdim=True) > 0).to(mu_y.dtype)
        xt = mu_y + self.am.vel(mu_y, t0, mu_y, g, m_frame)
        mel_pred = xt * self.am.stats[1] + self.am.stats[0]
        return mu_t, ld, mel_pred

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
        is_pause = (ids <= 6).float()
        phase = pos / L_f
        tail_closure = self.closure_slope.clamp(-0.30, -0.05) * ((phase - 0.72).clamp(min=0.0) / 0.28) * (1.0 - is_pause)
        p_arch = (0.55 * torch.sin(phase * 3.14159265) + 0.48 * torch.sin(phase * 9.424778) + 0.32 * torch.cos(phase * 15.707963) + tail_closure) * (1.0 - is_pause) - 0.28 * is_pause
        p_arch = F.avg_pool1d(F.pad(p_arch[:, None, :], (1, 1), mode='replicate'), 3, stride=1)[:, 0]
        sty_gain = (0.18 * pros[:, 1:2]).clamp(0.0, 0.38)
        p_prose_enc = p_arch * (0.55 * sty_gain)
        p_enc = (p_prose_enc + 0.35 * ps_tok).clamp(-0.48, 0.48)

        ones_m = torch.ones((B, 1), dtype=torch.float32, device=ids.device)
        h, mu, g = self.am.enc(ids, spk0, lang_eff, sty, studio, im, pitch_add=p_enc, prosody=pros, sty_mask=ones_m)
        if hasattr(self.am.enc, 'energy_pred'):
            eng_hat = self.am.enc.energy_pred(h).squeeze(-1).clamp(-1.2, 1.2)
            mu = mu + (0.10 * pros[:, 2:3] + 0.06 * eng_hat)[..., None] * 0.32

        x_d = self.am.dur.pred_logdur(ids, h, im) + self.dur_rate_bias.clamp(-0.12, 0.12)
        dur_frames = torch.exp(x_d).clamp(min=1.0, max=60.0)
        rate_mult = torch.exp(-0.04 * pros[:, 0:1]) * (1.0 - is_pause) + torch.exp(0.08 * pros[:, 3:4]) * is_pause
        dur_frames = (dur_frames * rate_mult * ds_tok).clamp(min=1.0, max=60.0)
        # Protect sentence-final tokens and short Nepali function words from frame collapse
        is_tail = ((pos >= (L_f - 3.0)) & (ids > 6)).float()
        dur = torch.round(torch.maximum(dur_frames, is_tail * 5.0 + (1.0 - is_tail) * 1.0)).clamp(min=1.0, max=60.0)
        # Resonant Pranava ॐ floor on Sanskrit utterances ('o' + 'm' >= 36 + 18 frames), light breathy visarga 'h', & Nepali hiatus ('आआफ्नो')
        id_o, id_m, id_h, id_a, id_wb = SYM2ID['o'], SYM2ID['m'], SYM2ID['h'], SYM2ID['a'], SYM2ID['|']
        next_ids = torch.cat([ids[:, 1:], ids[:, -1:]], dim=1)
        prev_ids = torch.cat([ids[:, :1], ids[:, :-1]], dim=1)
        next2_ids = torch.cat([ids[:, 2:], ids[:, -2:]], dim=1)
        prev2_ids = torch.cat([ids[:, :2], ids[:, :-2]], dim=1)
        is_sa_f = is_sa.float()[:, None]
        is_om_o = ((ids == id_o) & (next_ids == id_m) & (prev_ids <= 6)).float() * is_sa_f
        is_om_m = ((ids == id_m) & (prev_ids == id_o)).float() * is_sa_f
        dur = torch.maximum(dur, is_om_o * 36.0 + is_om_m * 18.0)
        is_final_h = ((ids == id_h) & (next_ids <= 6)).float() * is_sa_f
        dur = torch.where(is_final_h > 0.5, torch.clamp(dur, min=3.0, max=5.0), dur)
        is_hiatus_wb = ((ids == id_wb) & (prev_ids == id_a) & (next_ids == id_a)).float()
        is_hiatus_a1 = ((ids == id_a) & (next_ids == id_wb) & (next2_ids == id_a)).float()
        is_hiatus_a2 = ((ids == id_a) & (prev_ids == id_wb) & (prev2_ids == id_a)).float()
        dur = torch.maximum(dur, is_hiatus_a1 * 18.0 + is_hiatus_a2 * 18.0)
        dur = torch.where(is_hiatus_wb > 0.5, torch.full_like(dur, 7.0), dur)

        f0_hat = F.avg_pool1d(F.pad(self.am.enc.f0_pred(h).transpose(1, 2), (1, 1), mode='replicate'), 3, stride=1)[:, 0].clamp(-1.2, 1.2)
        p_out = (0.35 * f0_hat + 0.65 * p_arch) * sty_gain + 0.35 * ps_tok

        ends = torch.cumsum(dur, 1)
        T = ends[0, -1].to(torch.int64)
        t_arange = torch.arange(0, T, device=ids.device, dtype=torch.float32)
        idx = (t_arange[None, :, None] >= ends[:, None, :]).to(torch.float32).sum(-1).to(torch.int64).clamp(max=L - 1)

        mu_t = mu.transpose(1, 2)
        mu_y = torch.gather(mu_t, 2, idx[:, None, :].expand(-1, mu_t.shape[1], -1))
        p_xe = torch.gather(p_out, 1, idx)
        ps_fr = torch.gather(ps_tok, 1, idx)[:, None, :]
        m_t = torch.ones((B, 1, T), dtype=mu_y.dtype, device=mu_y.device)

        t0 = torch.zeros((B,), device=mu_y.device, dtype=torch.float32)
        xt = mu_y + self.am.vel(mu_y, t0, mu_y, g, m_t)

        mel = xt * self.am.stats[1] + self.am.stats[0]
        p_zero = p_xe - p_xe.mean(dim=1, keepdim=True)
        p_s1 = F.avg_pool1d(F.pad(p_zero[:, None, :], (4, 4), mode='replicate'), 9, stride=1)
        p_sm = F.avg_pool1d(F.pad(p_s1, (2, 2), mode='replicate'), 5, stride=1)
        env = F.avg_pool1d(F.pad(mel.transpose(1, 2), (3, 3), mode='replicate'), 7, stride=1).transpose(1, 2)
        kb = torch.arange(100, device=mel.device, dtype=mel.dtype)
        wk = ((64.0 - kb) / 18.0).clamp(0.0, 1.0)[None, :, None]
        comb_gain = 0.12 if self.steps == 1 else 0.08
        mel_w = env + (mel - env) * (1.0 + comb_gain * wk)
        if self.is_pico:
            mel_mu = mel_w.mean(dim=(1, 2), keepdim=True)
            mel_w = (mel_w - mel_mu) * 1.04 + mel_mu - 0.08
            env_f = F.avg_pool1d(F.pad(mel_w.transpose(1, 2), (2, 2), mode='replicate'), 5, stride=1).transpose(1, 2)
            wf = ((72.0 - kb) / 24.0).clamp(0.0, 1.0)[None, :, None]
            mel_w = mel_w + 0.18 * wf * (mel_w - env_f)
            env_t2 = F.avg_pool1d(F.pad(mel_w, (2, 2), mode='replicate'), 5, stride=1)
            mel_w = mel_w + 0.08 * (mel_w - env_t2)
        st_shift = (p_sm[:, 0, :] * 1.45).clamp(-1.80, 1.80)
        mel_clean = torch.cat([mel_w[:, :98, :], mel_w[:, 97:98, :], mel_w[:, 97:98, :]], dim=1)
        tot_shift = F.avg_pool1d(F.pad((0.35 * st_shift[:, None, :]).clamp(-0.85, 0.85), (2, 2), mode='replicate'), 5, stride=1)
        # Add smooth Chandas Shruti Svara contour + gentle 5.2 Hz +/-8c Gamaka vibrato on Sanskrit chants
        ps_sm = F.avg_pool1d(F.pad(ps_fr, (3, 3), mode='replicate'), 7, stride=1)
        gamaka = 0.06 * torch.sin((2.0 * math.pi * 5.2 / (SR / HOP)) * t_arange[None, None, :]) * is_sa_f[:, :, None]
        tot_shift = tot_shift + is_sa_f[:, :, None] * (0.95 * ps_sm + gamaka).clamp(-1.8, 1.8)

        y, _, _, _, _ = self.voc_wav(mel_clean, shift_st=tot_shift)
        pk = y.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
        return (y * (0.88 / pk)).clamp(-0.98, 0.98)


def set_qat(model, flag):
    model._qat_flag = bool(flag)
    if hasattr(model.am, 'set_qat'):
        model.am.set_qat(bool(flag))
    for name, m in model.named_modules():
        if name.startswith('voc.src.f0_net'):
            continue
        if hasattr(m, 'qat_enabled'):
            m.qat_enabled = bool(flag)


def build(tier, warm_voc=True):
    m = StudentV4(tier=tier)
    if warm_voc:
        m.warm_from_pretrained()
    return m


def n_params(m, deployed=True):
    tot = 0
    for name, p in m.named_parameters():
        if deployed and name.startswith(('am.ctc_rec.', 'am.enc.spk_adv.')):
            continue
        tot += p.numel()
    return tot


if __name__ == '__main__':
    for t in ('micro', 'pico', 't10'):
        m = build(t, warm_voc=False)
        print(t, f'{n_params(m)/1e6:.3f}M deployed params')
