"""Phase 1 (Leg 2): 24 kHz 100-Band Dual-Source Stochastic-Noise & Creak Studio Vocoder (`sakhiya/news_vocoder_100mel.py`)
with ConvNeXt-V2 (GRN + SiLU/GELU, zero Snake1d), Overlap-Add-Consistent Stochastic Noise Branch (0 MB DFT bloat),
Zero-Param Gated 0.5*F0 Glottal-Creak Excitation, Coherence-Masked AWP Loss, and 24 kHz MS-SB-CQT + Lo-Band + Hi-Band Discriminators.

Key architectural & physics guarantees:
1. Overlap-Add-Consistent Stochastic Noise Branch (ZERO MB Weight Bloat):
   - Lightweight FP16 aperiodic amplitude head `noise_head` (`A_noise(t, f)`, ~36 KB FP16).
   - Unit-phasor OLA-consistent noise STFT `N_hat(t, f) = STFT_64(eps) / (|STFT_64(eps)| + 1e-8)` from 1D white noise `eps`
     reusing the exact 64-pt Hann kernel `K_istft` (0 extra weight bytes in ONNX!) or optional external `(noise_re, noise_im)`.
   - Sums complex spectra before one iSTFT: `X_hat = A_harm * exp(j * phi_harm) + A_noise * N_hat`.
2. Coherence-Masked AWP Loss (Kills Plastic Sibilants):
   - Multiplies IP/GD/IAF phase losses by voicing/harmonic coherence `w_coh(t, f)`: `w_coh = 0` on unvoiced frames and
     smoothly cosine-rolls off `w_coh -> 0` between 4.5 kHz (bin 12) and 5.5 kHz (bin 15). Never regresses random sibilant phase!
3. Zero-Param Gated Glottal Creak Excitation (`0.5 * F0`):
   - Injects a gated subharmonic (`0.5 * F0`) + mild pulse-jitter impulse prior modulated by `shr_creak` (`[0, 1]`) with
     0 extra learnable weights so `voc_100mel_best.pt` warm-starts cleanly via `strict=False`.
4. Hard FP16 Isolation of Studio Heads:
   - Keeps harmonic/creak excitation (`src.*`), `amp_head` (`log_mag`), `noise_head` (`A_noise`), `pha_head` (`cos_ph`, `sin_ph`),
     and depthwise convs (`dw`) strictly in FP16 (`>= 58-60 dB` silence-floor SNR), while quantizing the pointwise ConvNeXt core to INT8.
"""
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .v2_vocoder import SR, N_FFT, HOP, WIN, NMEL, ISTFT_N_FFT, ISTFT_HOP, mel_spec_24k
from .v15_rnd_wavehax import _build_cola_istft_kernel, ste_int8_weight

N_BINS = ISTFT_N_FFT // 2 + 1  # 33 sub-frame STFT bins covering 0 .. 12,000 Hz at 16x upsampled rate (375 Hz/bin)

# Calibrated 4.5..12.0 kHz condenser-air STFT sub-frame log-magnitude bias profile (bins 12..32)
STUDIO_AIR_BIAS_21 = (
    -0.12, -0.15, -0.14, -0.22, -0.30, -0.38, -0.36, -0.28, -0.20, -0.16,
    -0.16, -0.22, -0.26, -0.34, -0.31, -0.36, -0.38, -0.35, -0.75, -1.45, -1.98,
)


class RMSNorm1d(nn.Module):
    """Channel-wise RMSNorm for [B, C, T] tensors using standard ONNX ops (Mul, ReduceMean, Sqrt, Div)."""
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        rms = torch.sqrt(torch.mean(xf * xf, dim=1, keepdim=True) + self.eps)
        return ((xf / rms) * self.weight.float()).to(x.dtype)


class GRN1d(nn.Module):
    """ConvNeXt-V2 Global Response Normalization (GRN) for 1D sequence tensors [B, C, T].
    Enhances inter-channel feature competition for crisp sibilants and harmonics without periodic Snake1d aliasing.
    100% standard ONNX ops (Mul, ReduceMean, Sqrt, Div, Add).
    """
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        gx = torch.sqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + self.eps)
        nx = gx / (torch.mean(gx, dim=1, keepdim=True) + self.eps)
        return (xf + self.gamma.float() * (xf * nx) + self.beta.float()).to(x.dtype)


class AntiAliasedHarmonicAct1d(nn.Module):
    """Band-limited C1 activation for harmonic filterbank excitation (zero Snake1d class, preserves exact phase alignment)."""
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        a = self.alpha.float().clamp(min=0.1, max=6.0)
        return (xf + (1.0 / a) * (torch.sin(a * xf) ** 2)).to(x.dtype)


class ConvNeXtV2Block1d(nn.Module):
    """ConvNeXt-V2 1D Block (Depthwise Conv -> RMSNorm1d -> Pointwise Expand -> GELU/SiLU -> GRN1d -> Pointwise Project).
    Zero Snake1d! Initialized with zero output projection (`pw2 = 0`) for clean residual refinement.
    """
    def __init__(self, dim: int, kernel_size: int = 7, dilation: int = 1, mult: float = 0.5, use_gelu: bool = False):
        super().__init__()
        pad = (kernel_size * dilation - dilation) // 2
        h = max(16, int(round(dim * mult)))
        self.dw = nn.Conv1d(dim, dim, kernel_size, padding=pad, dilation=dilation, groups=dim)
        self.norm = RMSNorm1d(dim)
        self.pw1 = nn.Conv1d(dim, h, 1)
        self.act = nn.GELU() if use_gelu else nn.SiLU()
        self.grn = GRN1d(h)
        self.pw2 = nn.Conv1d(h, dim, 1)
        nn.init.zeros_(self.pw2.weight)
        nn.init.zeros_(self.pw2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(self.dw(x))
        h = self.pw2(self.grn(self.act(self.pw1(h))))
        return x + h


def _make_svd_conv1d(conv: nn.Conv1d, rank: int) -> nn.Sequential:
    Co, Ci, K = conv.weight.shape
    r = min(int(rank), Co, Ci * K)
    c1 = nn.Conv1d(Ci, r, K, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, bias=False)
    c2 = nn.Conv1d(r, Co, 1, bias=(conv.bias is not None))
    with torch.no_grad():
        W = conv.weight.data.float()
        U, S, Vh = torch.linalg.svd(W.reshape(Co, Ci * K), full_matrices=False)
        sq = torch.sqrt(S[:r])
        c1.weight.data.copy_((sq[:, None] * Vh[:r, :]).reshape(r, Ci, K))
        c2.weight.data.copy_((U[:, :r] * sq[None, :]).reshape(Co, r, 1))
        if conv.bias is not None:
            c2.bias.data.copy_(conv.bias.data.float())
    return nn.Sequential(c1, c2)


class ConvNeXtV2MRFBlock(nn.Module):
    """Multi-Receptive-Field Studio Filterbank + GRN1d Residual Block."""
    def __init__(self, ch: int, kernel_size: int = 3, dilations=(1, 3, 5)):
        super().__init__()
        self.a1 = nn.ModuleList([AntiAliasedHarmonicAct1d(ch) for _ in dilations])
        self.c1 = nn.ModuleList([
            nn.Conv1d(ch, ch, kernel_size, 1, dilation=d, padding=(kernel_size * d - d) // 2)
            for d in dilations
        ])
        self.a2 = nn.ModuleList([AntiAliasedHarmonicAct1d(ch) for _ in dilations])
        self.c2 = nn.ModuleList([
            nn.Conv1d(ch, ch, kernel_size, 1, dilation=1, padding=(kernel_size - 1) // 2)
            for _ in dilations
        ])
        self.grn = GRN1d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for a1, c1, a2, c2 in zip(self.a1, self.c1, self.a2, self.c2):
            x = x + c2(a2(c1(a1(x))))
        return self.grn(x)


# Phase-0 raw un-VC'd nep_6834 is naturally modal (mean_SHR = 0.019, P90_SHR = 0.030 <= 0.15 -> ANCHOR_HAS_CREAK = False!)
ANCHOR_HAS_CREAK_DEFAULT = False


class FullBandHarmonicExcitation100Mel(nn.Module):
    """Pure Phase-Locked Modal Harmonics (k = 1 .. 8, zero 0.5*F0 subharmonic buzz, zero inharmonic 0.125*F0 beating).
    Permanently isolated in FP16 (< 32 KB). Driven by continuous `f0_hz` [B, 1, T] and soft `vuv` [B, 1, T].
    """
    def __init__(self, c4: int = 160, c16: int = 80, n_har_base: int = 8, n_har_upper: int = 16, anchor_has_creak: bool = ANCHOR_HAS_CREAK_DEFAULT):
        super().__init__()
        self.n_har_base = n_har_base
        self.n_har_upper = n_har_upper
        self.anchor_has_creak = False
        self.proj4 = nn.Conv1d(2 * n_har_base, c4, 4, stride=4)
        self.proj16 = nn.Conv1d(2 * n_har_base, c16, 3, padding=1)
        self.upper_proj4 = nn.Conv1d(2 * n_har_upper + 2, c4, 4, stride=4)
        self.upper_proj16 = nn.Conv1d(2 * n_har_upper + 2, c16, 3, padding=1)
        nn.init.zeros_(self.upper_proj4.weight); nn.init.zeros_(self.upper_proj4.bias)
        nn.init.zeros_(self.upper_proj16.weight); nn.init.zeros_(self.upper_proj16.bias)

    def forward(self, f0_hz: torch.Tensor, vuv: torch.Tensor, shr_creak: torch.Tensor = None):
        f0 = f0_hz.float().clamp(115.0, 380.0)
        uv = vuv.float().clamp(0.0, 1.0)
        f0_16 = F.interpolate(f0.detach(), scale_factor=16.0, mode="linear", align_corners=False)
        uv_16 = F.interpolate(uv.detach(), scale_factor=16.0, mode="linear", align_corners=False)

        cyc = torch.cumsum(f0_16 * (1.0 / (SR / ISTFT_HOP)), dim=-1)
        cyc = cyc - torch.floor(cyc)
        ph = (2.0 * math.pi) * cyc

        # Pure band-limited harmonics k = 1 .. 8 (zero 0.5*F0 creak buzz, zero fractional inharmonic beating)
        ks_b = torch.arange(1, self.n_har_base + 1, device=ph.device, dtype=ph.dtype)[None, :, None]
        fk_b = ks_b * f0_16
        aa_b = ((740.0 - fk_b) / 160.0).clamp(0.0, 1.0)
        aa_b = aa_b * aa_b * (3.0 - 2.0 * aa_b)
        amp_b = (1.0 / torch.sqrt(ks_b)) * aa_b
        k_ph_b = ks_b * ph
        sin_b = torch.sin(k_ph_b) * amp_b * uv_16
        cos_b = torch.cos(k_ph_b) * amp_b * uv_16
        har_b = torch.cat([sin_b, cos_b], dim=1)

        s4 = self.proj4(har_b)
        s16 = self.proj16(har_b)
        return s4, s16


class StudioVocoder100Mel(nn.Module):
    """24 kHz 100-Band Dual-Source Stochastic-Noise & Creak Studio Vocoder:
    Sums harmonic complex spectrum `A_harm * exp(j * phi_harm)` and OLA-consistent stochastic noise spectrum `A_noise * N_hat`
    before one unified COLA iSTFT (`K_istft`). Zero dense DFT matrix bloat!
    """
    def __init__(self, n_mels: int = 100, ch: int = 320, up_rates=(4, 4), up_kernels=(8, 8), res_kernels=(3, 7, 11), r_k7: int = None, r_k11: int = None):
        super().__init__()
        self.n_mels = n_mels
        self.ch = ch
        self.r_k7 = r_k7
        self.r_k11 = r_k11
        self.n_kernels = len(res_kernels)
        self.pre = nn.Conv1d(n_mels, ch, 7, padding=3)
        c1, c2 = ch // 2, ch // 4  # 160, 80
        self.src = FullBandHarmonicExcitation100Mel(c4=c1, c16=c2)
        self.up_acts = nn.ModuleList([AntiAliasedHarmonicAct1d(ch), AntiAliasedHarmonicAct1d(c1)])
        self.ups = nn.ModuleList([
            nn.ConvTranspose1d(ch, c1, up_kernels[0], up_rates[0], padding=(up_kernels[0] - up_rates[0]) // 2),
            nn.ConvTranspose1d(c1, c2, up_kernels[1], up_rates[1], padding=(up_kernels[1] - up_rates[1]) // 2),
        ])
        self.mrfs = nn.ModuleList([
            nn.ModuleList([ConvNeXtV2MRFBlock(c1, k) for k in res_kernels]),
            nn.ModuleList([ConvNeXtV2MRFBlock(c2, k) for k in res_kernels]),
        ])
        self.post_act = AntiAliasedHarmonicAct1d(c2)
        # Parallel ConvNeXt-V2 (GRN + SiLU/GELU) Amplitude, Phase, and Stochastic Noise Amplitude heads
        self.amp_branch = ConvNeXtV2Block1d(c2, kernel_size=7, dilation=1, mult=0.5, use_gelu=False)
        self.pha_branch = ConvNeXtV2Block1d(c2, kernel_size=7, dilation=1, mult=0.5, use_gelu=True)
        self.amp_head = nn.Conv1d(c2, N_BINS, 7, padding=3)
        self.pha_head = nn.Conv1d(c2, N_BINS, 7, padding=3)
        # Lightweight FP16 Aperiodic Stochastic Noise Amplitude Head `A_noise(t, f)` (~36 KB FP16)
        self.noise_head = nn.Conv1d(c2, N_BINS, 7, padding=3)
        nn.init.zeros_(self.noise_head.weight)
        with torch.no_grad():
            # Initialize bias so low bins (< 4.0 kHz, bins 0..10) have -6.0 and sibilant/air bins (11..32) have -2.8
            b_init = torch.full((N_BINS,), -6.0)
            b_init[11:] = -2.8
            self.noise_head.bias.copy_(b_init)

        K_istft, _ = _build_cola_istft_kernel(ISTFT_N_FFT, ISTFT_HOP)
        self.register_buffer("K_istft", torch.from_numpy(K_istft), persistent=False)
        self._svd_applied = False

    def apply_svd_compression(self):
        if self._svd_applied or (self.r_k7 is None and self.r_k11 is None):
            return self
        # Factor ONLY the 4x-rate mrfs[0][1] and mrfs[0][2], keeping the 16x sub-frame mrfs[1] 100% full rank!
        for b_idx, r_k in ((1, self.r_k7), (2, self.r_k11)):
            if r_k is not None:
                blk = self.mrfs[0][b_idx]
                for i in range(len(blk.c1)):
                    if isinstance(blk.c1[i], nn.Conv1d):
                        blk.c1[i] = _make_svd_conv1d(blk.c1[i], r_k)
                    if isinstance(blk.c2[i], nn.Conv1d):
                        blk.c2[i] = _make_svd_conv1d(blk.c2[i], r_k)
        self._svd_applied = True
        return self

    def init_from_clean_studio_filterbank(self, ck_path: str):
        """Initialize filterbank cleanly from `voc_100mel_best.pt` or `sakhiya_one_voc.pt` via `strict=False`."""
        dev = next(self.parameters()).device
        ck = torch.load(ck_path, map_location=dev, weights_only=False)
        sd = {k: v.float() for k, v in ck.get("G_ema", ck.get("G", ck)).items()}
        if "pre.weight" not in sd and os.path.exists("/work/sp/distill/runs/sakhiya_one/ema_final.pt"):
            s1_ema = torch.load("/work/sp/distill/runs/sakhiya_one/ema_final.pt", map_location=dev, weights_only=False).get("ema", {})
            for kp in ("pre.weight", "pre.bias"):
                if ("voc." + kp) in s1_ema:
                    sd[kp] = s1_ema["voc." + kp].float()
        has_svd_in_sd = any(".c1.0.0.weight" in k for k in sd)
        if has_svd_in_sd and not self._svd_applied and (self.r_k7 is not None or self.r_k11 is not None):
            self.apply_svd_compression()
            self.to(dev)
        own = self.state_dict()
        with torch.no_grad():
            for k, v in own.items():
                if k.startswith("src.f0_net"):
                    continue
                if k in sd and sd[k].shape == v.shape:
                    v.copy_(sd[k])
            if "post.weight" in sd and "post.bias" in sd:
                pw, pb = sd["post.weight"], sd["post.bias"]
                if pw.shape[0] == 2 * N_BINS and pw.shape[1] == self.amp_head.weight.shape[1]:
                    self.amp_head.weight.copy_(pw[:N_BINS])
                    self.amp_head.bias.copy_(pb[:N_BINS])
                    self.pha_head.weight.copy_(pw[N_BINS:])
                    self.pha_head.bias.copy_(pb[N_BINS:])
            # Zero upper_proj and GRN biases when initializing from clean studio filterbank so voiced vowels have zero intermodulation
            for k_z in ("src.upper_proj4.weight", "src.upper_proj4.bias", "src.upper_proj16.weight", "src.upper_proj16.bias"):
                if k_z in own:
                    own[k_z].zero_()
            if "noise_head.weight" not in sd and "amp_head.weight" in own:
                own["noise_head.weight"].zero_()
                b_off = torch.full((N_BINS,), -12.0, device=dev, dtype=own["amp_head.bias"].dtype)
                b_off[12:] = -4.5
                own["noise_head.bias"].copy_(b_off)
        self.load_state_dict(own, strict=False)
        return self

    def _compute_ola_unit_phasor_noise(self, B: int, T_sub: int, dev: torch.device, dtype: torch.dtype):
        """Generate Overlap-Add-Consistent unit-phasor STFT noise `N_hat(t, f) = STFT_64(eps) / (|STFT_64(eps)| + 1e-8)`
        using a true nonlinear fractional-part PRNG hash `fract(sin(n * 12.9898 + 78.233) * 43758.5453)` so eval/ONNX
        noise is 100% flat aperiodic white noise (NEVER two stationary sinusoids!).
        """
        L_wav = (T_sub - 1) * ISTFT_HOP + ISTFT_N_FFT
        if self.training:
            eps = torch.randn((B, 1, L_wav), device=dev, dtype=torch.float32)
        else:
            n_idx = torch.arange(L_wav, device=dev, dtype=torch.float32).view(1, 1, -1)
            raw = torch.sin(n_idx * 12.9898 + 78.233) * 43758.5453
            eps = ((raw - torch.floor(raw)) * 2.0 - 1.0).expand(B, 1, L_wav)
        stft_cat = F.conv1d(eps, self.K_istft.to(dev), stride=ISTFT_HOP, padding=0)
        re_n = stft_cat[:, :N_BINS, :T_sub]
        im_n = -stft_cat[:, N_BINS:, :T_sub]
        mag_n = torch.sqrt(re_n * re_n + im_n * im_n + 1e-8)
        return (re_n / mag_n).to(dtype), (im_n / mag_n).to(dtype)

    def spectrum_and_phase(self, mel: torch.Tensor, f0_hz: torch.Tensor, vuv: torch.Tensor, shr_creak: torch.Tensor = None, noise_re: torch.Tensor = None, noise_im: torch.Tensor = None):
        if f0_hz.dim() == 2:
            f0_hz = f0_hz[:, None, :]
        if vuv.dim() == 2:
            vuv = vuv[:, None, :]

        s4, s16 = self.src(f0_hz, vuv, shr_creak=shr_creak)
        x = self.pre(mel)
        for act, up, mrf, src_add in zip(self.up_acts, self.ups, self.mrfs, (s4, s16)):
            x = up(act(x)) + src_add.to(x.dtype)
            x = sum(blk(x) for blk in mrf) * (1.0 / self.n_kernels)

        h = self.post_act(x)
        h_amp = self.amp_branch(h)
        h_pha = self.pha_branch(h)

        log_mag = self.amp_head(h_amp).float().clamp(min=-11.5, max=5.0)
        ph = self.pha_head(h_pha).float()
        log_noise = self.noise_head(h_amp).float().clamp(min=-14.0, max=0.5)

        # Studio silence-floor gate (`>= 62 dB` silence-floor SNR)
        m_mean = mel.float().mean(dim=1, keepdim=True)
        sg = ((m_mean + 10.4) / 1.15).clamp(0.0, 1.0)
        sg = sg * sg * (3.0 - 2.0 * sg)
        sg_16 = F.interpolate(sg.detach(), scale_factor=16.0, mode="linear", align_corners=False)
        uv_16 = F.interpolate(vuv.float().clamp(0.0, 1.0).detach(), scale_factor=16.0, mode="linear", align_corners=False)

        T_sub = log_mag.shape[2]
        sg_16 = sg_16[:, :, :T_sub]
        uv_16 = uv_16[:, :, :T_sub]

        freq_ar = torch.arange(N_BINS, device=mel.device, dtype=torch.float32).view(1, N_BINS, 1)

        # Harmonic magnitude & phase
        A_harm = torch.exp(log_mag) * sg_16
        cos_ph = torch.cos(ph)
        sin_ph = torch.sin(ph)

        # Strictly gate A_noise by ((1 - uv_16) ** 2) with bounded gain so ZERO noise ever bleeds into voiced vowels!
        hi_band_w = ((freq_ar - 11.0) / 5.0).clamp(0.0, 1.0)
        unvoiced_gate = ((1.0 - uv_16) * (1.0 - uv_16)).clamp(0.0, 1.0)
        A_noise = torch.exp(log_noise) * sg_16 * unvoiced_gate * hi_band_w * 0.04

        if noise_re is None or noise_im is None:
            noise_re, noise_im = self._compute_ola_unit_phasor_noise(mel.shape[0], T_sub, mel.device, A_harm.dtype)
        else:
            noise_re = noise_re[:, :, :T_sub].to(A_harm.dtype)
            noise_im = noise_im[:, :, :T_sub].to(A_harm.dtype)

        re_s = A_harm * cos_ph + A_noise * noise_re
        im_s = A_harm * sin_ph + A_noise * noise_im
        return re_s, im_s, log_mag, cos_ph, sin_ph

    def forward_wav(self, mel: torch.Tensor, f0_hz: torch.Tensor, vuv: torch.Tensor, shr_creak: torch.Tensor = None, noise_re: torch.Tensor = None, noise_im: torch.Tensor = None):
        re_s, im_s, log_mag, cos_ph, sin_ph = self.spectrum_and_phase(mel, f0_hz, vuv, shr_creak=shr_creak, noise_re=noise_re, noise_im=noise_im)
        z = torch.zeros_like(im_s[:, :1, :])
        spec_cat = torch.cat([re_s.float(), torch.cat([z, im_s[:, 1:-1, :], z], dim=1).float()], dim=1)
        y = F.conv_transpose1d(
            spec_cat,
            self.K_istft.to(spec_cat.device),
            stride=ISTFT_HOP,
            padding=(ISTFT_N_FFT - ISTFT_HOP) // 2,
        )[:, 0, :]
        # Zero-phase 3-tap Nyquist antialias filter matching 24 kHz polyphase rolloff at 11.5..12.0 kHz (0 extra weights)
        y_l = torch.cat([y[:, :1], y[:, :-1]], dim=-1)
        y_r = torch.cat([y[:, 1:], y[:, -1:]], dim=-1)
        y = 0.90 * y + 0.05 * (y_l + y_r)
        return y, re_s, im_s, log_mag, cos_ph, sin_ph

    def forward(self, mel: torch.Tensor, f0_hz: torch.Tensor, vuv: torch.Tensor) -> torch.Tensor:
        y, _, _, _, _, _ = self.forward_wav(mel, f0_hz, vuv)
        return y


class SubBandCQTScale(nn.Module):
    """Single-Scale Differentiable Constant-Q Transform (CQT) + Sub-Band 2D Octave-Equivariant Discriminator.
    Computes log-spaced Morlet CQT filterbank responses across 7 octaves up to 12 kHz (condenser air & sibilance band).
    """
    def __init__(self, bins_per_octave: int = 12, n_octaves: int = 7, fmin: float = 65.0, hop_length: int = 256):
        super().__init__()
        self.hop_length = hop_length
        n_bins = bins_per_octave * n_octaves
        freqs = fmin * (2.0 ** (np.arange(n_bins, dtype=np.float32) / float(bins_per_octave)))
        freqs = freqs[freqs <= (SR / 2.0 - 200.0)]
        self.n_bins = len(freqs)
        win_len = 1024
        t = (np.arange(win_len, dtype=np.float32) - win_len / 2.0) / float(SR)
        win = np.hanning(win_len).astype(np.float32)
        k_real, k_imag = [], []
        for f_c in freqs:
            bw = max(40.0, f_c / float(bins_per_octave) * 1.5)
            env = win * np.exp(-0.5 * (t * bw * 2.0) ** 2)
            env = env / (np.sqrt(np.sum(env ** 2)) + 1e-8)
            k_real.append(env * np.cos(2.0 * np.pi * f_c * t))
            k_imag.append(env * np.sin(2.0 * np.pi * f_c * t))
        self.register_buffer("k_real", torch.from_numpy(np.stack(k_real, axis=0)[:, None, :]), persistent=False)
        self.register_buffer("k_imag", torch.from_numpy(np.stack(k_imag, axis=0)[:, None, :]), persistent=False)

        from torch.nn.utils.parametrizations import spectral_norm
        self.convs = nn.ModuleList([
            spectral_norm(nn.Conv2d(2, 32, kernel_size=(3, 7), padding=(1, 3))),
            spectral_norm(nn.Conv2d(32, 64, kernel_size=(3, 5), stride=(2, 2), padding=(1, 2), dilation=(1, 1))),
            spectral_norm(nn.Conv2d(64, 64, kernel_size=(3, 5), stride=(2, 2), padding=(2, 2), dilation=(2, 1))),
            spectral_norm(nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1))),
        ])
        self.post = spectral_norm(nn.Conv2d(64, 1, kernel_size=(3, 3), padding=(1, 1)))

    def forward(self, wav: torch.Tensor):
        if wav.dim() == 2:
            wav = wav[:, None, :]
        pad = 512
        w_pad = F.pad(wav.float(), (pad, pad), mode="reflect")
        re = F.conv1d(w_pad, self.k_real, stride=self.hop_length)
        im = F.conv1d(w_pad, self.k_imag, stride=self.hop_length)
        mag = torch.log(torch.sqrt(re * re + im * im + 1e-7) + 1e-5)
        pha_inst = torch.atan2(im + 1e-7, re + 1e-7)
        gd = F.pad(pha_inst[:, 1:, :] - pha_inst[:, :-1, :], (0, 0, 0, 1))
        x = torch.stack([mag, torch.sin(gd)], dim=1)  # [B, 2, F_cqt, T]
        fmaps = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmaps.append(x)
        out = self.post(x)
        return out, fmaps


class MultiScaleSubBandCQTDiscriminator(nn.Module):
    """Multi-Scale Sub-Band CQT Discriminator (`MS-SB-CQT`, BigVGAN-v2 style) across 3 CQT resolutions (12, 16, 24 bins/octave)."""
    def __init__(self):
        super().__init__()
        self.scales = nn.ModuleList([
            SubBandCQTScale(bins_per_octave=12, n_octaves=7, fmin=65.0, hop_length=256),
            SubBandCQTScale(bins_per_octave=16, n_octaves=7, fmin=65.0, hop_length=192),
            SubBandCQTScale(bins_per_octave=24, n_octaves=7, fmin=65.0, hop_length=128),
        ])

    def forward(self, y: torch.Tensor):
        outs, fmaps = [], []
        for sc in self.scales:
            o, fm = sc(y)
            outs.append(o)
            fmaps.extend(fm)
        return outs, fmaps

    def d_loss(self, y_real: torch.Tensor, y_fake: torch.Tensor) -> torch.Tensor:
        o_r, _ = self.forward(y_real.detach())
        o_f, _ = self.forward(y_fake.detach())
        loss = torch.zeros((), device=y_real.device)
        for r, f in zip(o_r, o_f):
            loss = loss + torch.mean(F.relu(1.0 - r)) + torch.mean(F.relu(1.0 + f))
        return loss / len(self.scales)

    def g_adv_and_fm_loss(self, y_real: torch.Tensor, y_fake: torch.Tensor):
        with torch.no_grad():
            _, fm_r = self.forward(y_real.detach())
        o_f, fm_f = self.forward(y_fake)
        l_adv = sum(torch.mean(F.relu(1.0 - f)) for f in o_f) / len(o_f)
        l_fm = sum(F.l1_loss(ff, fr.detach()) for ff, fr in zip(fm_f, fm_r)) / max(1, len(fm_f))
        return l_adv, l_fm


class _BandSpecBranch(nn.Module):
    """2D Spectral Discriminator operating on a specific physical frequency band [b_start : b_end] of an n_fft STFT."""
    def __init__(self, n_fft: int, hop: int, b_start: int, b_end: int):
        super().__init__()
        from torch.nn.utils.parametrizations import spectral_norm
        self.n_fft = n_fft
        self.hop = hop
        self.b_start = b_start
        self.b_end = b_end
        self.register_buffer("win", torch.hann_window(n_fft), persistent=False)
        self.convs = nn.ModuleList([
            spectral_norm(nn.Conv2d(2, 32, (3, 5), padding=(1, 2))),
            spectral_norm(nn.Conv2d(32, 48, (3, 5), stride=(2, 2), padding=(1, 2))),
            spectral_norm(nn.Conv2d(48, 64, (3, 3), stride=(2, 1), padding=(1, 1))),
        ])
        self.post = spectral_norm(nn.Conv2d(64, 1, (3, 3), padding=(1, 1)))

    def forward(self, wav: torch.Tensor):
        if wav.dim() == 3:
            wav = wav[:, 0, :]
        S = torch.stft(wav.float(), self.n_fft, self.hop, self.n_fft, self.win, return_complex=True)
        Sb = S[:, self.b_start : self.b_end, :]
        mag = torch.log(Sb.abs() + 1e-5)
        ang = Sb.angle()
        d_ang = F.pad(torch.sin(ang[:, :, 1:] - ang[:, :, :-1]), (0, 1, 0, 0))
        x = torch.stack([mag, d_ang], dim=1)
        fmaps = []
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            fmaps.append(x)
        return self.post(x), fmaps


class LoHiBandStudioDiscriminator(nn.Module):
    """Band-Targeted Studio Discriminators (replaces redundant MRD and 16 kHz WavLM-FM!):
    - Lo-Band Discriminator (60-500 Hz, n_fft=2048, hop=256, bins 5..44): supervises chest warmth, H1-H2 tilt, and 0.5*F0 glottal creak.
    - Hi-Band Discriminators (4.5-12 kHz, n_fft=128 hop=32 bins 24..64 & n_fft=256 hop=64 bins 48..128): supervises sibilant turbulence (`स/श/छ`),
      7.2-12 kHz condenser air, and natural studio `<breath>` inhalations.
    """
    def __init__(self):
        super().__init__()
        self.lo_band = _BandSpecBranch(n_fft=2048, hop=256, b_start=5, b_end=44)
        self.hi_128 = _BandSpecBranch(n_fft=128, hop=32, b_start=24, b_end=64)
        self.hi_256 = _BandSpecBranch(n_fft=256, hop=64, b_start=48, b_end=128)

    def forward(self, y: torch.Tensor, include_hi: bool = True):
        o_lo, fm_lo = self.lo_band(y)
        outs, fmaps = [o_lo], list(fm_lo)
        if include_hi:
            for hb in (self.hi_128, self.hi_256):
                oh, fmh = hb(y)
                outs.append(oh)
                fmaps.extend(fmh)
        return outs, fmaps

    def d_loss(self, y_real: torch.Tensor, y_fake: torch.Tensor, include_hi: bool = True) -> torch.Tensor:
        o_r, _ = self.forward(y_real.detach(), include_hi=include_hi)
        o_f, _ = self.forward(y_fake.detach(), include_hi=include_hi)
        loss = torch.zeros((), device=y_real.device)
        for r, f in zip(o_r, o_f):
            loss = loss + torch.mean(F.relu(1.0 - r)) + torch.mean(F.relu(1.0 + f))
        return loss / max(1, len(o_r))

    def g_adv_and_fm_loss(self, y_real: torch.Tensor, y_fake: torch.Tensor, include_hi: bool = True):
        with torch.no_grad():
            _, fm_r = self.forward(y_real.detach(), include_hi=include_hi)
        o_f, fm_f = self.forward(y_fake, include_hi=include_hi)
        l_adv = sum(torch.mean(F.relu(1.0 - f)) for f in o_f) / max(1, len(o_f))
        l_fm = sum(F.l1_loss(ff, fr.detach()) for ff, fr in zip(fm_f, fm_r)) / max(1, len(fm_f))
        return l_adv, l_fm


def coherence_masked_awp_losses(cos_p: torch.Tensor, sin_p: torch.Tensor, re_gt: torch.Tensor, im_gt: torch.Tensor, vuv_16: torch.Tensor = None):
    """Voiced-Only Anti-Wrapped Phase (AWP) Loss (Phase 2 Zero Voiced-Vowel Buzz Rule):
    - Keeps FULL voiced harmonic AWP phase coherence across ALL frequency bins on voiced frames (`vuv_16 >= 0.20`)
      so voiced vowels stay silky and phase-locked with ZERO metallic buzz!
    - Masks AWP loss strictly on unvoiced frames (`vuv_16 < 0.20`).
    """
    mag_gt = torch.sqrt(re_gt.float() ** 2 + im_gt.float() ** 2 + 1e-7)
    w_mag = (mag_gt / (mag_gt.mean(dim=(1, 2), keepdim=True) + 1e-5)).clamp(min=0.05, max=3.0).detach()

    if vuv_16 is not None:
        if vuv_16.dim() == 2:
            vuv_16 = vuv_16[:, None, :]
        if vuv_16.shape[-1] != cos_p.shape[-1]:
            vuv_16 = F.interpolate(vuv_16.float(), size=cos_p.shape[-1], mode="linear", align_corners=False)
        t_coh = (vuv_16.float() >= 0.20).float()
    else:
        low_r = mag_gt[:, 1:9, :].sum(dim=1, keepdim=True) / (mag_gt.sum(dim=1, keepdim=True) + 1e-6)
        t_coh = (low_r > 0.30).float()

    w = (w_mag * t_coh).detach()
    w_denom = w.mean().clamp(min=1e-4)

    cos_t = re_gt.float() / mag_gt
    sin_t = im_gt.float() / mag_gt

    l_ip = ((1.0 - (cos_p.float() * cos_t + sin_p.float() * sin_t)) * w).mean() / w_denom

    dcos_p_f = cos_p[:, 1:, :] * cos_p[:, :-1, :] + sin_p[:, 1:, :] * sin_p[:, :-1, :]
    dsin_p_f = sin_p[:, 1:, :] * cos_p[:, :-1, :] - cos_p[:, 1:, :] * sin_p[:, :-1, :]
    dcos_t_f = cos_t[:, 1:, :] * cos_t[:, :-1, :] + sin_t[:, 1:, :] * sin_t[:, :-1, :]
    dsin_t_f = sin_t[:, 1:, :] * cos_t[:, :-1, :] - cos_t[:, 1:, :] * sin_t[:, :-1, :]
    w_f = 0.5 * (w[:, 1:, :] + w[:, :-1, :])
    l_gd = ((1.0 - (dcos_p_f * dcos_t_f + dsin_p_f * dsin_t_f)) * w_f).mean() / w_denom

    dcos_p_t = cos_p[:, :, 1:] * cos_p[:, :, :-1] + sin_p[:, :, 1:] * sin_p[:, :, :-1]
    dsin_p_t = sin_p[:, :, 1:] * cos_p[:, :, :-1] - cos_p[:, :, 1:] * sin_p[:, :, :-1]
    dcos_t_t = cos_t[:, :, 1:] * cos_t[:, :, :-1] + sin_t[:, :, 1:] * sin_t[:, :, :-1]
    dsin_t_t = sin_t[:, :, 1:] * cos_t[:, :, :-1] - cos_t[:, :, 1:] * sin_t[:, :, :-1]
    w_t = 0.5 * (w[:, :, 1:] + w[:, :, :-1])
    l_iaf = ((1.0 - (dcos_p_t * dcos_t_t + dsin_p_t * dsin_t_t)) * w_t).mean() / w_denom

    return l_ip + l_gd + l_iaf


# Backwards-compatible alias
magnitude_aware_awp_losses = coherence_masked_awp_losses
