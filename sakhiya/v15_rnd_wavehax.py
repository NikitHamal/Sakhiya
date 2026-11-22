"""Sakhiya v15 MS-RND-Wavehax Vocoder (`sakhiya/v15_rnd_wavehax.py`).
Combines two mathematical speech synthesis breakthroughs into an ultra-compact browser-native 24 kHz vocoder:
1. Analytical Range-Null Space Decomposition (RNDVoC):
   Decomposes the 33-bin short-time spectrum orthogonally into an exact 0-parameter Range-Space envelope
   A^+ y_mel plus a learned Null-Space fine harmonic residual (I - A^+ A) r_theta.
2. Multi-Stream 2D Subband ConvNeXt-V2 + Periodic Snake1d Depthwise-Separable MRF (MS-Wavehax):
   Captures 2D time-frequency harmonic comb structure and phase continuity with 10x fewer parameters
   than 1D C^2*K transposed-conv iSTFTNet.
3. Fused 1-Op COLA iSTFT Direct-Wav Export + Per-Channel INT8 STE Quantization-Aware Training (QAT):
   Exports directly to 24 kHz waveform `wav` (zero ConvInteger, 100% browser WASM native)."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .v2_vocoder import SR, N_FFT, HOP, WIN, NMEL, FMIN, FMAX, ISTFT_N_FFT, ISTFT_HOP, Snake1d, ola64


def _build_rnd_matrices(n_mels=NMEL, n_bins=ISTFT_N_FFT // 2 + 1):
    """Precompute exact analytical Mel Filterbank A (100 x 33), pseudo-inverse A_pinv (33 x 100),
    and Null-Space projector P_null = I_33 - A_pinv @ A (33 x 33)."""
    import librosa
    fb_full = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=n_mels, fmin=FMIN, fmax=FMAX).astype(np.float64)
    # Pool 513 bins of n_fft=1024 into 33 bins of n_fft=64 (stride 16)
    A = np.zeros((n_mels, n_bins), dtype=np.float64)
    for k in range(n_bins):
        c0 = max(0, k * 16 - 8)
        c1 = min(fb_full.shape[1], k * 16 + 9)
        A[:, k] = fb_full[:, c0:c1].sum(axis=1)
    row_sum = np.maximum(A.sum(axis=1, keepdims=True), 1e-8)
    A = A / row_sum
    # Regularized Moore-Penrose pseudo-inverse A^+ = A^T (A A^T + lam I)^{-1}
    lam = 1e-2
    AAt = A @ A.T + lam * np.eye(n_mels, dtype=np.float64)
    A_pinv = A.T @ np.linalg.inv(AAt)
    P_null = np.eye(n_bins, dtype=np.float64) - A_pinv @ A
    return A.astype(np.float32), A_pinv.astype(np.float32), P_null.astype(np.float32)


def _build_cola_istft_kernel(n_fft=ISTFT_N_FFT, hop=ISTFT_HOP):
    """Build exact 1-op fused iDFT + Hann window + 75% COLA (1/1.5) ConvTranspose1d kernel (66, 1, 64)."""
    win = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    k = np.arange(n_fft // 2 + 1, dtype=np.float32)[:, None]
    n = np.arange(n_fft, dtype=np.float32)[None, :]
    ang = 2.0 * np.pi * k * n / float(n_fft)
    c = np.full((n_fft // 2 + 1, 1), 2.0, dtype=np.float32)
    c[0, 0] = 1.0
    c[-1, 0] = 1.0
    # For periodic Hann at 75% overlap (hop = n_fft / 4), sum_t win^2[n - t*hop] === 1.500000 everywhere
    w_cola = win[None, :] / 1.5
    k_re = (c * np.cos(ang) / float(n_fft)) * w_cola
    k_im = (-c * np.sin(ang) / float(n_fft)) * w_cola
    K = np.concatenate([k_re, k_im], axis=0)[:, None, :]  # (66, 1, 64)
    return K.astype(np.float32), win


def ste_int8_weight(w, axis=0, enabled=False):
    """Straight-Through Estimator (STE) symmetric per-channel INT8 fake-quantization matching to_int8_qdq_weights."""
    if not enabled or w.numel() < 64 or w.dim() < 2:
        return w
    red = tuple(i for i in range(w.dim()) if i != axis)
    sc = w.detach().abs().amax(dim=red, keepdim=True).clamp(min=1e-8) / 127.0
    w_q = torch.clamp(torch.round(w / sc), -127.0, 127.0) * sc
    return w + (w_q - w).detach()


class QATConv1d(nn.Conv1d):
    def __init__(s, *args, **kwargs):
        super().__init__(*args, **kwargs)
        s.qat_enabled = False

    def forward(s, x):
        w = ste_int8_weight(s.weight, axis=0, enabled=s.qat_enabled and s.training)
        return F.conv1d(x, w, s.bias, s.stride, s.padding, s.dilation, s.groups)


class QATConvTranspose1d(nn.ConvTranspose1d):
    def __init__(s, *args, **kwargs):
        super().__init__(*args, **kwargs)
        s.qat_enabled = False

    def forward(s, x):
        w = ste_int8_weight(s.weight, axis=1, enabled=s.qat_enabled and s.training)
        return F.conv_transpose1d(x, w, s.bias, s.stride, s.padding, s.output_padding, s.groups, s.dilation)


class QATConv2d(nn.Conv2d):
    def __init__(s, *args, **kwargs):
        super().__init__(*args, **kwargs)
        s.qat_enabled = False

    def forward(s, x):
        w = ste_int8_weight(s.weight, axis=0, enabled=s.qat_enabled and s.training)
        return F.conv2d(x, w, s.bias, s.stride, s.padding, s.dilation, s.groups)


class ConvNeXt2DSubbandBlock(nn.Module):
    """2D Time-Frequency ConvNeXt-V2 block operating on (B, C, F_sub=16, T_4):
    Captures harmonic pitch comb spacing along frequency and inter-frame continuity along time."""
    def __init__(s, c=24, h=48, kf=5, kt=7):
        super().__init__()
        s.dw = QATConv2d(c, c, (kf, kt), padding=(kf // 2, kt // 2), groups=c)
        s.norm = nn.GroupNorm(1, c)
        s.pw1 = QATConv2d(c, h, 1)
        s.grn_g = nn.Parameter(torch.zeros(1, h, 1, 1))
        s.grn_b = nn.Parameter(torch.zeros(1, h, 1, 1))
        s.pw2 = QATConv2d(h, c, 1)

    def forward(s, x):
        r = x
        y = s.norm(s.dw(x))
        y = F.gelu(s.pw1(y))
        gx = torch.sqrt((y.float() ** 2).mean(dim=(2, 3), keepdim=True) + 1e-6).to(y.dtype)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        y = y * (1.0 + s.grn_g.to(y.dtype) * nx) + s.grn_b.to(y.dtype)
        return r + s.pw2(y)


class SnakeMRFConvNeXt1D(nn.Module):
    """Depthwise-Separable Multi-Receptive-Field Periodic Snake1d Block:
    Replaces C^2*K dense ResBlock1 with parallel depthwise dilated kernels (3, 7, 11) + 1x1 pointwise mixing."""
    def __init__(s, ch, h_mult=2.0, kernels=(3, 7, 11), dilations=(1, 3, 5)):
        super().__init__()
        s.act1 = Snake1d(ch)
        s.dws = nn.ModuleList([
            QATConv1d(ch, ch, k, padding=(k * d - d) // 2, dilation=d, groups=ch)
            for k, d in zip(kernels, dilations)
        ])
        s.norm = nn.GroupNorm(1, ch)
        hid = int(round(ch * h_mult))
        s.pw1 = QATConv1d(ch, hid, 1)
        s.act2 = Snake1d(hid)
        s.pw2 = QATConv1d(hid, ch, 1)

    def forward(s, x):
        y = s.act1(x)
        y = (s.dws[0](y) + s.dws[1](y) + s.dws[2](y)) * (1.0 / 3.0)
        y = s.norm(y)
        y = s.pw2(s.act2(s.pw1(y)))
        return x + y


class WavehaxHarmonicSource(nn.Module):
    """NSF Harmonic Source with configurable f0_net width (c_f0=64 for Lite, c_f0=128 for Pro) and shift_st hook."""
    def __init__(s, n_mels=NMEL, c4=96, c16=64, c_f0=64, n_har=8):
        super().__init__()
        s.n_har = n_har
        s.c_f0 = c_f0
        s.f0_net = nn.Sequential(
            QATConv1d(n_mels, c_f0, 5, padding=2),
            nn.SiLU(),
            QATConv1d(c_f0, c_f0, 5, padding=2),
            nn.SiLU(),
            QATConv1d(c_f0, 2, 3, padding=1),
        )
        s.proj4 = QATConv1d(2 * n_har, c4, 4, stride=4)
        s.proj16 = QATConv1d(2 * n_har, c16, 3, padding=1)

    def forward(s, mel, shift_st=None):
        raw = s.f0_net(mel)
        rf = raw.float()
        f0 = 60.0 + 440.0 * torch.sigmoid(rf[:, 0:1])
        uv = torch.sigmoid(rf[:, 1:2])
        w_v = (uv.detach() > 0.35).float() * uv.detach() + 0.04
        lf0_raw = torch.log2(f0.clamp(130.0, 380.0))
        num = F.avg_pool1d(F.pad(lf0_raw * w_v, (3, 3), mode="replicate"), 7, stride=1)
        den = F.avg_pool1d(F.pad(w_v, (3, 3), mode="replicate"), 7, stride=1).clamp(min=1e-3)
        lf0_loc = num / den
        lf0_sm = F.avg_pool1d(F.pad(0.50 * lf0_raw + 0.50 * lf0_loc, (2, 2), mode="replicate"), 5, stride=1)
        f0 = torch.pow(2.0, lf0_sm)
        uv = F.avg_pool1d(F.pad(uv, (1, 1), mode="replicate"), 3, stride=1)
        sh = shift_st if shift_st is not None else getattr(s, "shift_st", None)
        if sh is not None:
            f0 = f0 * torch.pow(2.0, sh.to(f0.dtype) / 12.0)
        f0_16 = F.interpolate(f0.detach(), scale_factor=16.0, mode="linear", align_corners=False)
        uv_16 = F.interpolate(uv.detach(), scale_factor=16.0, mode="linear", align_corners=False)
        cyc = torch.cumsum(f0_16 * (1.0 / (SR / ISTFT_HOP)), dim=-1)
        cyc = cyc - torch.floor(cyc)
        ph = (2.0 * math.pi) * cyc
        ks = torch.arange(1, s.n_har + 1, device=ph.device, dtype=ph.dtype)[None, :, None]
        amp = (1.0 / torch.sqrt(ks)) * (ks * f0_16 < 720.0).float()
        k_ph = ks * ph
        har = torch.cat([torch.sin(k_ph) * amp, torch.cos(k_ph) * amp], dim=1) * uv_16
        t_idx = torch.arange(ph.shape[-1], device=ph.device, dtype=ph.dtype)[None, None, :]
        ap_freqs = torch.tensor([1.41421, 1.73205, 2.23607, 2.64575, 3.14159, 3.60555, 4.12311, 4.58258], device=ph.device, dtype=ph.dtype)[None, :, None]
        ap = torch.cat([torch.sin(t_idx * ap_freqs), torch.cos(t_idx * (ap_freqs * 1.17))], dim=1)
        har = (har + (1.0 - uv_16) * 0.14 * ap).to(raw.dtype)
        return s.proj4(har), s.proj16(har), f0, uv, har


class MSRNDWavehax(nn.Module):
    """MS-RND-Wavehax Vocoder:
    - variant='lite': ~370K params -> ~420 KB int8 ONNX
    - variant='pro':  ~760K params -> ~840 KB int8 ONNX
    """
    def __init__(s, variant="lite", n_mels=NMEL, ch=None, c1=None, c2=None, c_2d=None, c_f0=None, n_2d=None, n_4x=None, n_16x=None):
        super().__init__()
        if variant == "pro":
            ch = ch or 104; c1 = c1 or 88; c2 = c2 or 60; c_2d = c_2d or 24; c_f0 = c_f0 or 128; n_2d = n_2d or 6; n_4x = n_4x or 6; n_16x = n_16x or 6
        else:
            ch = ch or 72; c1 = c1 or 60; c2 = c2 or 44; c_2d = c_2d or 16; c_f0 = c_f0 or 56; n_2d = n_2d or 5; n_4x = n_4x or 4; n_16x = n_16x or 4
        s.variant = variant
        s.n_bins = ISTFT_N_FFT // 2 + 1  # 33
        s.f_sub = 8  # 8 frequency subbands x c_2d channels in the 2D Time-Frequency stream

        # Analytical Range-Null Space Decomposition matrices (0 trainable parameters)
        A, A_pinv, P_null = _build_rnd_matrices(n_mels, s.n_bins)
        K_istft, win = _build_cola_istft_kernel(ISTFT_N_FFT, ISTFT_HOP)
        s.register_buffer("A_pinv", torch.from_numpy(A_pinv))      # (33, 100)
        s.register_buffer("P_null", torch.from_numpy(P_null))      # (33, 33)
        s.register_buffer("K_istft", torch.from_numpy(K_istft), persistent=False)  # (66, 1, 64)
        s.register_buffer("win", torch.from_numpy(win), persistent=False)

        # Calibrated per-bin affine scale/offset on the analytical Range-Space log-magnitude envelope (+3.15 matches n_fft=64 vs 1024 log scale)
        s.range_gain = nn.Parameter(torch.ones(1, s.n_bins, 1))
        s.range_bias = nn.Parameter(torch.full((1, s.n_bins, 1), 3.15))
        s.null_gain = nn.Parameter(torch.full((1, s.n_bins, 1), 0.50))
        s.null_alpha = nn.Parameter(torch.ones(1, s.n_bins, 1))

        s.src = WavehaxHarmonicSource(n_mels, c1, c2, c_f0=c_f0)
        s.pre = QATConv1d(n_mels + s.n_bins, ch, 5, padding=2)

        # Stage 1: 4x temporal upsampling (hop=256 -> hop=64) + 2D Subband ConvNeXt-V2 stream + 1D Snake-MRF stream
        s.up1_act = Snake1d(ch)
        s.up1 = QATConvTranspose1d(ch, c1, 8, 4, padding=2)
        s.to_2d = QATConv1d(c1, c_2d * s.f_sub, 1)
        s.blocks_2d = nn.ModuleList([ConvNeXt2DSubbandBlock(c_2d, c_2d * 2) for _ in range(n_2d)])
        s.from_2d = QATConv1d(c_2d * s.f_sub, c1, 1)
        s.blocks_4x = nn.ModuleList([SnakeMRFConvNeXt1D(c1, h_mult=2.0) for _ in range(n_4x)])

        # Stage 2: 4x temporal upsampling (hop=64 -> hop=16) + 1D Snake-MRF harmonic refinement
        s.up2_act = Snake1d(c1)
        s.up2 = QATConvTranspose1d(c1, c2, 8, 4, padding=2)
        s.blocks_16x = nn.ModuleList([SnakeMRFConvNeXt1D(c2, h_mult=2.0) for _ in range(n_16x)])

        s.post_act = Snake1d(c2)
        s.post = QATConv1d(c2, 2 * s.n_bins, 7, padding=3)
        s.har_phase = QATConv1d(16, s.n_bins, 1)

    def set_qat(s, enabled=True):
        for m in s.modules():
            if hasattr(m, "qat_enabled"):
                m.qat_enabled = bool(enabled)
        return s

    def fuse_weight_norm(s):
        return s

    def spectrum_and_f0(s, mel, shift_st=None):
        # 1. Exact Analytical Range-Space Envelope at frame rate T and upsampled to 16T
        mel_f = mel.float()
        range_env_t = torch.matmul(s.A_pinv.to(mel_f.dtype), mel_f)  # (B, 33, T)
        range_env_16 = F.interpolate(range_env_t, scale_factor=16.0, mode="linear", align_corners=False)

        # 2. Analytic Harmonic Source (f0, uv, 4x & 16x harmonic excitations)
        s4, s16, f0, uv, har16 = s.src(mel, shift_st=shift_st)

        # 3. Pre-conv on concatenated (mel, range_env_t)
        x = s.pre(torch.cat([mel, range_env_t.to(mel.dtype)], dim=1))

        # 4. 4x Upsample + Dual-Stream (2D Frequency-Time Subband ConvNeXt + 1D Snake-MRF)
        x = s.up1(s.up1_act(x)) + s4
        B, C1, T4 = x.shape
        x2d = s.to_2d(x).view(B, -1, s.f_sub, T4)
        for b2 in s.blocks_2d:
            x2d = b2(x2d)
        x = x + s.from_2d(x2d.view(B, -1, T4))
        for b4 in s.blocks_4x:
            x = b4(x)

        # 5. 16x Upsample + Snake-MRF Harmonic Refinement
        x = s.up2(s.up2_act(x)) + s16
        for b16 in s.blocks_16x:
            x = b16(x)

        # 6. Range-Null Space Spectral Synthesis:
        #    log|X| = (range_gain * A^+ y + range_bias) + null_gain * (P_null r_theta) + null_alpha * r_theta
        raw_out = s.post(s.post_act(x)).float()
        r_theta, ph_res = raw_out.chunk(2, dim=1)
        null_proj = torch.matmul(s.P_null.to(r_theta.dtype), r_theta)
        log_mag = (s.range_gain.float() * range_env_16 + s.range_bias.float()) + s.null_gain.float() * null_proj + s.null_alpha.float() * r_theta
        mag = torch.exp(log_mag.clamp(min=-11.0, max=5.0))
        ph = ph_res + s.har_phase(har16).float()
        return mag * torch.cos(ph), mag * torch.sin(ph), f0, uv

    def spec_to_wav(s, re, im):
        """Fast 1-op COLA iSTFT matching ONNX export 1:1."""
        z = torch.zeros_like(im[:, :1, :])
        im_sym = torch.cat([z, im[:, 1:-1, :], z], dim=1)
        spec_cat = torch.cat([re.float(), im_sym.float()], dim=1)
        p = (ISTFT_N_FFT - ISTFT_HOP) // 2
        return F.conv_transpose1d(spec_cat, s.K_istft.to(spec_cat.device), stride=ISTFT_HOP, padding=p)[:, 0, :]

    def spectrum(s, mel, shift_st=None):
        re, im, _, _ = s.spectrum_and_f0(mel, shift_st=shift_st)
        return re, im

    def forward(s, mel, shift_st=None):
        re, im = s.spectrum(mel, shift_st=shift_st)
        return s.spec_to_wav(re, im)

    def forward_wav_onnx(s, mel):
        """Direct 24 kHz waveform ONNX export (extracts st_shift from bin 99 - bin 97 and uses 1-op COLA ConvTranspose1d)."""
        st_shift = ((mel[:, 99:100, :] - mel[:, 97:98, :]) * 100.0).clamp(-1.80, 1.80)
        mel_clean = torch.cat([mel[:, :98, :], mel[:, 97:98, :], mel[:, 97:98, :]], dim=1)
        tot_shift = F.avg_pool1d(F.pad((0.35 * st_shift).clamp(-0.85, 0.85), (2, 2), mode="replicate"), 5, stride=1)
        re, im = s.spectrum(mel_clean, shift_st=tot_shift)
        return s.spec_to_wav(re, im)
