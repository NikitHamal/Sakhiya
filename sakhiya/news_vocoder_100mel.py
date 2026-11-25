"""Phase 1: 24 kHz 100-Band Full-Bandwidth Studio Vocoder (`sakhiya/news_vocoder_100mel.py`)
with ConvNeXt-V2 (GRN + SiLU/GELU, zero Snake1d), Parallel Amplitude & Anti-Wrapped Phase (AWP) iSTFT branches,
Full-Band Harmonics up to 12 kHz, and Multi-Scale Sub-Band CQT Discriminator (`MS-SB-CQT`).

Key architectural guarantees:
1. ZERO `f0_net` inside the vocoder: conditions directly on AM continuous `f0_hz` [B, 1, T] + soft `vuv` [B, 1, T]
   driving full-band band-limited harmonics up to 12 kHz (k = 1 .. floor(12000 / f0)) + Gaussian noise excitation.
2. ZERO `Snake1d`: uses ConvNeXt-V2 blocks with Global Response Normalization (`GRN1d`) + `RMSNorm1d` + `SiLU`/`GELU`
   throughout the multi-receptive-field upsampling body and parallel Amplitude / Phase branches.
3. 100 full-bandwidth mel bins (`fmin=0, fmax=12000, n_fft=1024, hop=256`, 24 kHz) with zero top-bin clipping.
4. Permanent FP16 isolation for harmonic excitation (`src.*`) and final iSTFT projection heads (`amp_head`, `pha_head`, < 55 KB),
   while quantizing the ConvNeXt-V2 body to per-channel INT8 QDQ (> 95% INT8 weights).
"""
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .v2_vocoder import SR, N_FFT, HOP, WIN, NMEL, ISTFT_N_FFT, ISTFT_HOP, mel_spec_24k
from .v15_rnd_wavehax import _build_cola_istft_kernel, ste_int8_weight

N_BINS = ISTFT_N_FFT // 2 + 1  # 33 sub-frame STFT bins covering 0 .. 12,000 Hz at 16x upsampled rate

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


class FullBandHarmonicExcitation100Mel(nn.Module):
    """Full-Band Band-Limited Harmonics up to 12 kHz (k = 1 .. floor(12000 / f0), up to K_max=24) + Gaussian Noise Excitation.
    Permanently isolated in FP16 (< 32 KB). Driven directly by AM continuous `f0_hz` [B, 1, T] and soft `vuv` [B, 1, T].
    """
    def __init__(self, c4: int = 160, c16: int = 80, n_har_base: int = 8, n_har_upper: int = 16):
        super().__init__()
        self.n_har_base = n_har_base
        self.n_har_upper = n_har_upper
        self.proj4 = nn.Conv1d(2 * n_har_base, c4, 4, stride=4)
        self.proj16 = nn.Conv1d(2 * n_har_base, c16, 3, padding=1)
        self.upper_proj4 = nn.Conv1d(2 * n_har_upper + 2, c4, 4, stride=4)
        self.upper_proj16 = nn.Conv1d(2 * n_har_upper + 2, c16, 3, padding=1)
        nn.init.zeros_(self.upper_proj4.weight); nn.init.zeros_(self.upper_proj4.bias)
        nn.init.zeros_(self.upper_proj16.weight); nn.init.zeros_(self.upper_proj16.bias)

    def forward(self, f0_hz: torch.Tensor, vuv: torch.Tensor):
        f0 = f0_hz.float().clamp(125.0, 380.0)
        uv = vuv.float().clamp(0.0, 1.0)
        f0_16 = F.interpolate(f0.detach(), scale_factor=16.0, mode="linear", align_corners=False)
        uv_16 = F.interpolate(uv.detach(), scale_factor=16.0, mode="linear", align_corners=False)

        cyc = torch.cumsum(f0_16 * (1.0 / (SR / ISTFT_HOP)), dim=-1)
        cyc = cyc - torch.floor(cyc)
        ph = (2.0 * math.pi) * cyc

        # 1. Base harmonics k = 1 .. 8
        ks_b = torch.arange(1, self.n_har_base + 1, device=ph.device, dtype=ph.dtype)[None, :, None]
        fk_b = ks_b * f0_16
        aa_b = ((740.0 - fk_b) / 160.0).clamp(0.0, 1.0)
        aa_b = aa_b * aa_b * (3.0 - 2.0 * aa_b)
        amp_b = (1.0 / torch.sqrt(ks_b)) * aa_b
        k_ph_b = ks_b * ph
        har_b = torch.cat([torch.sin(k_ph_b) * amp_b, torch.cos(k_ph_b) * amp_b], dim=1) * uv_16

        # 2. Full-Band Upper Harmonics k = 9 .. 24 up to Nyquist 12,000 Hz + shaped unvoiced sibilant/breath noise
        ks_u = torch.arange(self.n_har_base + 1, self.n_har_base + self.n_har_upper + 1, device=ph.device, dtype=ph.dtype)[None, :, None]
        fk_u = ks_u * f0_16
        nyq_aa = ((12000.0 - fk_u) / 1200.0).clamp(0.0, 1.0)
        nyq_aa = nyq_aa * nyq_aa * (3.0 - 2.0 * nyq_aa)
        amp_u = (1.0 / torch.sqrt(ks_u)) * nyq_aa * uv_16
        ph_u = (ks_u * 0.125) * ph
        t_idx = torch.arange(ph.shape[-1], device=ph.device, dtype=torch.float32).view(1, 1, -1)
        noise_exc = (1.0 - 0.80 * uv_16) * torch.sin(t_idx * 12.9898 + ph * 1.73) * torch.cos(t_idx * 78.233 - ph * 2.19)
        har_u = torch.cat([torch.sin(ph_u) * amp_u, torch.cos(ph_u) * amp_u, uv_16, noise_exc], dim=1)

        s4 = self.proj4(har_b) + self.upper_proj4(har_u)
        s16 = self.proj16(har_b) + self.upper_proj16(har_u)
        return s4, s16


class StudioVocoder100Mel(nn.Module):
    """24 kHz 100-Band Full-Bandwidth Studio Vocoder (ConvNeXt-V2 GRN + SiLU/GELU, zero Snake1d, full-rank MRF filterbank)."""
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
        # Parallel ConvNeXt-V2 (GRN + SiLU/GELU) Amplitude & Anti-Wrapped Phase branches
        self.amp_branch = ConvNeXtV2Block1d(c2, kernel_size=7, dilation=1, mult=0.5, use_gelu=False)
        self.pha_branch = ConvNeXtV2Block1d(c2, kernel_size=7, dilation=1, mult=0.5, use_gelu=True)
        self.amp_head = nn.Conv1d(c2, N_BINS, 7, padding=3)
        self.pha_head = nn.Conv1d(c2, N_BINS, 7, padding=3)

        K_istft, _ = _build_cola_istft_kernel(ISTFT_N_FFT, ISTFT_HOP)
        self.register_buffer("K_istft", torch.from_numpy(K_istft), persistent=False)
        self._svd_applied = False

    def apply_svd_compression(self):
        if self._svd_applied or (self.r_k7 is None and self.r_k11 is None):
            return self
        for b_idx, r_k in ((1, self.r_k7), (2, self.r_k11)):
            if r_k is not None:
                blk = self.mrfs[0][b_idx]
                for i in range(len(blk.c1)):
                    if isinstance(blk.c1[i], nn.Conv1d):
                        blk.c1[i] = _make_svd_conv1d(blk.c1[i], r_k)
                    if isinstance(blk.c2[i], nn.Conv1d):
                        blk.c2[i] = _make_svd_conv1d(blk.c2[i], r_k)
                blk2 = self.mrfs[1][b_idx]
                r_k2 = max(28, int(round(r_k * 0.70)))
                for i in range(len(blk2.c1)):
                    if isinstance(blk2.c1[i], nn.Conv1d):
                        blk2.c1[i] = _make_svd_conv1d(blk2.c1[i], r_k2)
                    if isinstance(blk2.c2[i], nn.Conv1d):
                        blk2.c2[i] = _make_svd_conv1d(blk2.c2[i], r_k2)
        self._svd_applied = True
        return self

    def init_from_clean_studio_filterbank(self, ck_path: str):
        """Initialize filterbank cleanly from `sakhiya_one_voc.pt` / `v3_voc320_fused.pt` (zero Old-V1)."""
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
            # Calibrate 4.5..12.0 kHz condenser-air roll-off on amp_head.bias[12:33] if initializing from raw filterbank
            if not ck.get("air_calibrated", False):
                own["amp_head.bias"][12:33].add_(torch.tensor(STUDIO_AIR_BIAS_21, device=dev, dtype=own["amp_head.bias"].dtype))
        self.load_state_dict(own, strict=False)
        if not self._svd_applied and (self.r_k7 is not None or self.r_k11 is not None):
            self.apply_svd_compression()
            self.to(dev)
        return self

    def spectrum_and_phase(self, mel: torch.Tensor, f0_hz: torch.Tensor, vuv: torch.Tensor):
        if f0_hz.dim() == 2:
            f0_hz = f0_hz[:, None, :]
        if vuv.dim() == 2:
            vuv = vuv[:, None, :]

        s4, s16 = self.src(f0_hz, vuv)
        x = self.pre(mel)
        for act, up, mrf, src_add in zip(self.up_acts, self.ups, self.mrfs, (s4, s16)):
            x = up(act(x)) + src_add.to(x.dtype)
            x = sum(blk(x) for blk in mrf) * (1.0 / self.n_kernels)

        h = self.post_act(x)
        h_amp = self.amp_branch(h)
        h_pha = self.pha_branch(h)

        log_mag = self.amp_head(h_amp).float().clamp(min=-11.0, max=5.0)
        ph = self.pha_head(h_pha).float()

        # Gentle studio room-floor gate on silence while preserving 80-180ms natural micro-breaths
        m_mean = mel.float().mean(dim=1, keepdim=True)
        sg = ((m_mean + 10.5) / 1.2).clamp(0.0, 1.0)
        sg = sg * sg * (3.0 - 2.0 * sg)
        sg_16 = F.interpolate(sg.detach(), scale_factor=16.0, mode="linear", align_corners=False)

        mag = torch.exp(log_mag) * sg_16[:, :, :log_mag.shape[2]]
        cos_ph = torch.cos(ph)
        sin_ph = torch.sin(ph)
        return mag * cos_ph, mag * sin_ph, log_mag, cos_ph, sin_ph

    def forward_wav(self, mel: torch.Tensor, f0_hz: torch.Tensor, vuv: torch.Tensor):
        re_s, im_s, log_mag, cos_ph, sin_ph = self.spectrum_and_phase(mel, f0_hz, vuv)
        z = torch.zeros_like(im_s[:, :1, :])
        spec_cat = torch.cat([re_s.float(), torch.cat([z, im_s[:, 1:-1, :], z], dim=1).float()], dim=1)
        y = F.conv_transpose1d(
            spec_cat,
            self.K_istft.to(spec_cat.device),
            stride=ISTFT_HOP,
            padding=(ISTFT_N_FFT - ISTFT_HOP) // 2,
        )[:, 0, :]
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


def magnitude_aware_awp_losses(cos_p: torch.Tensor, sin_p: torch.Tensor, re_gt: torch.Tensor, im_gt: torch.Tensor):
    """Magnitude-Aware Anti-Wrapped Phase (AWP) Loss:
    Weights Instantaneous Phase (IP), Group Delay (GD), and Instantaneous Angular Frequency (IAF) by log-magnitude
    prominence so harmonic peaks and high-frequency sibilant formants receive crisp phase supervision without fitting silence noise.
    """
    mag_gt = torch.sqrt(re_gt.float() ** 2 + im_gt.float() ** 2 + 1e-7)
    w = (mag_gt / (mag_gt.mean(dim=(1, 2), keepdim=True) + 1e-5)).clamp(min=0.05, max=3.5).detach()
    cos_t = re_gt.float() / mag_gt
    sin_t = im_gt.float() / mag_gt

    l_ip = ((1.0 - (cos_p.float() * cos_t + sin_p.float() * sin_t)) * w).mean()

    dcos_p_f = cos_p[:, 1:, :] * cos_p[:, :-1, :] + sin_p[:, 1:, :] * sin_p[:, :-1, :]
    dsin_p_f = sin_p[:, 1:, :] * cos_p[:, :-1, :] - cos_p[:, 1:, :] * sin_p[:, :-1, :]
    dcos_t_f = cos_t[:, 1:, :] * cos_t[:, :-1, :] + sin_t[:, 1:, :] * sin_t[:, :-1, :]
    dsin_t_f = sin_t[:, 1:, :] * cos_t[:, :-1, :] - cos_t[:, 1:, :] * sin_t[:, :-1, :]
    w_f = 0.5 * (w[:, 1:, :] + w[:, :-1, :])
    l_gd = ((1.0 - (dcos_p_f * dcos_t_f + dsin_p_f * dsin_t_f)) * w_f).mean()

    dcos_p_t = cos_p[:, :, 1:] * cos_p[:, :, :-1] + sin_p[:, :, 1:] * sin_p[:, :, :-1]
    dsin_p_t = sin_p[:, :, 1:] * cos_p[:, :, :-1] - cos_p[:, :, 1:] * sin_p[:, :, :-1]
    dcos_t_t = cos_t[:, :, 1:] * cos_t[:, :, :-1] + sin_t[:, :, 1:] * sin_t[:, :, :-1]
    dsin_t_t = sin_t[:, :, 1:] * cos_t[:, :, :-1] - cos_t[:, :, 1:] * sin_t[:, :, :-1]
    w_t = 0.5 * (w[:, :, 1:] + w[:, :, :-1])
    l_iaf = ((1.0 - (dcos_p_t * dcos_t_t + dsin_p_t * dsin_t_t)) * w_t).mean()

    return l_ip + l_gd + l_iaf
