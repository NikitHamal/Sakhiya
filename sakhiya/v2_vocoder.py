"""Sakhiya-v2-Voc: 24 kHz 16x-upsampling Source-Filter Hybrid iSTFTNet/HiFTNet vocoder (~4.8M params).
Replaces frame-rate ConvNeXt Vocos (86.13 Hz phase-lock buzz) with 16x temporal upsampling (4x * 4x),
Snake1d periodic activations, continuous-phase harmonic source excitation, and an n_fft=64, hop=16 iSTFT head.
Zero post-hoc DSP/EQ/Griffin-Lim hacks at inference."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils import remove_weight_norm

SR, N_FFT, HOP, WIN, NMEL, FMIN, FMAX = 24000, 1024, 256, 1024, 100, 0.0, 12000.0
ISTFT_N_FFT, ISTFT_HOP = 64, 16
_mel_cache = {}


def mel_spec_24k(wav, device="cpu"):
    """wav: (B, T) or (T,) float tensor in [-1, 1] at 24 kHz -> 100-band log-mel (B, 100, T // 256)."""
    if wav.dim() == 1:
        wav = wav[None]
    if device not in _mel_cache:
        import librosa
        fb = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=NMEL, fmin=FMIN, fmax=FMAX).astype(np.float32)
        win = np.hanning(WIN + 1)[:-1].astype(np.float32)
        _mel_cache[device] = (torch.from_numpy(fb).to(device), torch.from_numpy(win).to(device))
    fb, win = _mel_cache[device]
    wav = wav.to(device)
    pad = (N_FFT - HOP) // 2
    x = F.pad(wav[:, None], (pad, pad), mode="reflect")[:, 0]
    spec = torch.stft(x, N_FFT, HOP, WIN, win, center=False, return_complex=True).abs()
    return torch.log(torch.clamp(torch.matmul(fb, spec), min=1e-5))


class Snake1d(nn.Module):
    """FP32-stabilized periodic Snake activation (prevents bf16 sin() quantization noise floor)."""
    def __init__(s, channels):
        super().__init__()
        s.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(s, x):
        xf = x.float()
        a = s.alpha.float().clamp(min=0.1)
        y = xf + (1.0 / a) * (torch.sin(a * xf) ** 2)
        return y.to(x.dtype)


class ResBlock1(nn.Module):
    def __init__(s, ch, kernel_size=3, dilations=(1, 3, 5)):
        super().__init__()
        s.a1 = nn.ModuleList([Snake1d(ch) for _ in dilations])
        s.c1 = nn.ModuleList([weight_norm(nn.Conv1d(ch, ch, kernel_size, 1, dilation=d, padding=(kernel_size * d - d) // 2)) for d in dilations])
        s.a2 = nn.ModuleList([Snake1d(ch) for _ in dilations])
        s.c2 = nn.ModuleList([weight_norm(nn.Conv1d(ch, ch, kernel_size, 1, dilation=1, padding=(kernel_size - 1) // 2)) for _ in dilations])

    def forward(s, x):
        for a1, c1, a2, c2 in zip(s.a1, s.c1, s.a2, s.c2):
            x = x + c2(a2(c1(a1(x))))
        return x


class HarmonicSource(nn.Module):
    """Sakhiya-v3 NSF Harmonic Source: 3-layer Conv pitch estimator + k=1..8 analytical harmonics + shaped unvoiced noise at 1500 Hz."""
    def __init__(s, n_mels=NMEL, c4=160, c16=80, n_har=8):
        super().__init__()
        s.n_har = n_har
        s.f0_net = nn.Sequential(
            nn.Conv1d(n_mels, 128, 5, padding=2),
            nn.SiLU(),
            nn.Conv1d(128, 128, 5, padding=2),
            nn.SiLU(),
            nn.Conv1d(128, 2, 3, padding=1),
        )
        s.proj4 = nn.Conv1d(2 * n_har, c4, 4, stride=4)
        s.proj16 = nn.Conv1d(2 * n_har, c16, 3, padding=1)

    def forward(s, mel, shift_st=None):
        with torch.amp.autocast("cuda", enabled=False):
            rf = s.f0_net(mel.float()).float()
        f0 = torch.round(60.0 + 440.0 * torch.sigmoid(rf[:, 0:1]))  # pre-smoothing 1 Hz snap locks FP32/INT8 cumsum phase; 7+5 frame filter smooths glide
        uv_raw = torch.round(torch.sigmoid(rf[:, 1:2]) * 32.0) * 0.03125
        # Voiced-weighted local log-F0 interpolation + 3-frame closing on uv (eliminates 1-frame intra-vowel voicing shutters & phase snaps!)
        lf0_raw = torch.log2(f0.clamp(130.0, 380.0))
        if f0.shape[-1] >= 7:
            uv_closed = F.max_pool1d(F.pad(uv_raw, (1, 1), mode="replicate"), 3, stride=1)
            uv_blend = 0.65 * uv_raw + 0.35 * uv_closed
            w_v = (uv_blend.detach() > 0.32).float() * uv_blend.detach() + 0.04
            num = F.avg_pool1d(F.pad(lf0_raw * w_v, (3, 3), mode="replicate"), 7, stride=1)
            den = F.avg_pool1d(F.pad(w_v, (3, 3), mode="replicate"), 7, stride=1).clamp(min=1e-3)
            lf0_loc = num / den
            lf0_sm = F.avg_pool1d(F.pad(0.50 * lf0_raw + 0.50 * lf0_loc, (2, 2), mode="replicate"), 5, stride=1)
            uv = torch.round(F.avg_pool1d(F.pad(uv_blend, (1, 1), mode="replicate"), 3, stride=1) * 1024.0) * 0.0009765625
        else:
            uv = uv_raw
            lf0_sm = lf0_raw
        sh = shift_st if shift_st is not None else getattr(s, "shift_st", None)
        if sh is not None:
            lf0_sm = lf0_sm + sh.to(f0.dtype) * (1.0 / 12.0)
        lf0_q = torch.round(lf0_sm * 256.0) * 0.00390625
        f0 = torch.round(torch.pow(2.0, lf0_q) * 16.0) * 0.0625
        f0_16 = F.interpolate(f0.detach(), scale_factor=16.0, mode="linear", align_corners=False)
        uv_16 = F.interpolate(uv.detach(), scale_factor=16.0, mode="linear", align_corners=False)
        # Wrapped cumulative cycles in [0, 1) so long utterances never suffer float32 cumsum phase drift
        cyc = torch.cumsum(f0_16 * (1.0 / (SR / ISTFT_HOP)), dim=-1)
        cyc = cyc - torch.floor(cyc)
        ph = (2.0 * math.pi) * cyc
        ks = torch.arange(1, s.n_har + 1, device=ph.device, dtype=ph.dtype)[None, :, None]
        # Smooth C1 anti-aliasing roll-off between 580 Hz and 740 Hz (eliminates abrupt k*f0=720Hz mid-syllable harmonic switching clicks)
        fk = ks * f0_16
        aa = ((740.0 - fk) / 160.0).clamp(0.0, 1.0)
        aa = aa * aa * (3.0 - 2.0 * aa)
        amp = (1.0 / torch.sqrt(ks)) * aa
        k_ph = ks * ph
        har = (torch.cat([torch.sin(k_ph) * amp, torch.cos(k_ph) * amp], dim=1) * uv_16).to(mel.dtype)
        return s.proj4(har), s.proj16(har), f0, uv


class VocoderV2(nn.Module):
    def __init__(s, n_mels=NMEL, ch=320, up_rates=(4, 4), up_kernels=(8, 8), res_kernels=(3, 7, 11)):
        super().__init__()
        s.n_kernels = len(res_kernels)
        s.pre = weight_norm(nn.Conv1d(n_mels, ch, 7, padding=3))
        c1, c2 = ch // 2, ch // 4
        s.src = HarmonicSource(n_mels, c1, c2)
        s.up_acts = nn.ModuleList([Snake1d(ch), Snake1d(c1)])
        s.ups = nn.ModuleList([
            weight_norm(nn.ConvTranspose1d(ch, c1, up_kernels[0], up_rates[0], padding=(up_kernels[0] - up_rates[0]) // 2)),
            weight_norm(nn.ConvTranspose1d(c1, c2, up_kernels[1], up_rates[1], padding=(up_kernels[1] - up_rates[1]) // 2)),
        ])
        s.mrfs = nn.ModuleList([
            nn.ModuleList([ResBlock1(c1, k) for k in res_kernels]),
            nn.ModuleList([ResBlock1(c2, k) for k in res_kernels]),
        ])
        s.post_act = Snake1d(c2)
        s.post = weight_norm(nn.Conv1d(c2, ISTFT_N_FFT + 2, 7, padding=3))
        win = np.hanning(ISTFT_N_FFT + 1)[:-1].astype(np.float32)
        s.register_buffer("win", torch.from_numpy(win), persistent=False)

    def spectrum_and_f0(s, mel, shift_st=None):
        s4, s16, f0, uv = s.src(mel, shift_st=shift_st)
        x = s.pre(mel)
        for idx, (act, up, mrf, src_add) in enumerate(zip(s.up_acts, s.ups, s.mrfs, (s4, s16))):
            x = up(act(x)) + src_add
            x = sum(blk(x) for blk in mrf) / s.n_kernels
        h = s.post(s.post_act(x)).float()
        log_mag, ph = h.chunk(2, dim=1)
        mag = torch.exp(log_mag.clamp(min=-11.0, max=5.0))
        # In-graph studio pause noise-gate & upper-bin anti-metallic roll-off
        m_mean = mel.float().mean(dim=1, keepdim=True)
        sg = ((m_mean + 10.5) / 1.2).clamp(0.0, 1.0)
        sg = sg * sg * (3.0 - 2.0 * sg)
        sg_16 = F.interpolate(sg.detach(), scale_factor=16.0, mode="linear", align_corners=False)
        k_bins = torch.arange(mag.shape[1], device=mag.device, dtype=mag.dtype)[None, :, None]
        hf_w = 1.0 - 0.22 * ((k_bins - 15.0) / 14.0).clamp(0.0, 1.0)
        mag = mag * sg_16 * hf_w
        return mag * torch.cos(ph), mag * torch.sin(ph), f0, uv

    def spectrum(s, mel, shift_st=None):
        re, im, _, _ = s.spectrum_and_f0(mel, shift_st=shift_st)
        return re, im

    def forward(s, mel):
        re, im = s.spectrum(mel)
        return ola64(torch.complex(re.float(), im.float()), s.win)

    def fuse_weight_norm(s):
        from torch.nn.utils import parametrize
        for m in s.modules():
            if parametrize.is_parametrized(m, "weight"):
                parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)
        return s

    def forward_onnx(s, mel):
        """ONNX-exportable path: iDFT as matmul (no irfft/fold) + overlap-add as ConvTranspose1d with identity kernel."""
        re, im, _, _ = s.spectrum_and_f0(mel)
        k = torch.arange(ISTFT_N_FFT // 2 + 1, dtype=torch.float32)[:, None]; n = torch.arange(ISTFT_N_FFT, dtype=torch.float32)[None]
        ang = 2.0 * math.pi * k * n / ISTFT_N_FFT
        c = torch.full((ISTFT_N_FFT // 2 + 1, 1), 2.0); c[0] = 1.0; c[-1] = 1.0
        cosb, sinb = (c * torch.cos(ang) / ISTFT_N_FFT).to(re.device), (c * torch.sin(ang) / ISTFT_N_FFT).to(re.device)
        fr = (torch.einsum("bkt,kn->bnt", re, cosb) - torch.einsum("bkt,kn->bnt", im, sinb)) * s.win[None, :, None]
        eye = torch.eye(ISTFT_N_FFT, device=re.device)[:, None, :]
        y = F.conv_transpose1d(fr, eye, stride=ISTFT_HOP)
        env = F.conv_transpose1d((s.win ** 2)[None, :, None] * torch.ones_like(re[:1, :1]), eye, stride=ISTFT_HOP)
        p = (ISTFT_N_FFT - ISTFT_HOP) // 2
        return (y / env.clamp(min=1e-8))[:, 0, p:-p]


def ola64(spec, win, n_fft=ISTFT_N_FFT, hop=ISTFT_HOP):
    T = spec.shape[-1]
    y = torch.fft.irfft(spec, n_fft, dim=1) * win[None, :, None]
    L = (T - 1) * hop + n_fft
    y = F.fold(y, (1, L), (1, n_fft), stride=(1, hop))[:, 0, 0]
    env = F.fold((win ** 2)[None, :, None].expand(1, -1, T), (1, L), (1, n_fft), stride=(1, hop))[:, 0, 0]
    p = (n_fft - hop) // 2
    return (y / env.clamp(min=1e-8))[:, p:-p]


def n_params(m):
    return sum(p.numel() for p in m.parameters())
