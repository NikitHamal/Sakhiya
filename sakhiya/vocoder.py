"""Sakhiya-Voc: ~3.4M-param Vocos-style ConvNeXt vocoder (mel -> complex spectrum -> iSTFT) with harmonic peak contrast."""
import json, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .audio import N_FFT, HOP, NMEL


def _load_eq():
    """Inference-time spectral calibration (long-term-average-spectrum match to the teacher), see scripts/calib_eq.py."""
    p = os.path.join(os.path.dirname(__file__), "data", "voc_eq.json"); d = json.load(open(p))
    return {"pts": torch.tensor(d["pts"]).view(1, 1, -1), "pow": float(d["pow"]), "contrast": float(d["contrast"])}


_EQ = _load_eq()


class Block(nn.Module):
    def __init__(s, d, m, scale):
        super().__init__()
        s.dw = nn.Conv1d(d, d, 7, padding=3, groups=d)
        s.norm = nn.LayerNorm(d, eps=1e-6)
        s.pw1, s.pw2 = nn.Linear(d, m), nn.Linear(m, d)
        s.gamma = nn.Parameter(torch.full((d,), scale))

    def forward(s, x):
        y = s.norm(s.dw(x).transpose(1, 2))
        return x + (s.pw2(F.gelu(s.pw1(y))) * s.gamma).transpose(1, 2)


class Vocoder(nn.Module):
    def __init__(s, d=256, m=768, L=11):
        super().__init__()
        s.embed = nn.Conv1d(NMEL, d, 7, padding=3)
        s.norm = nn.LayerNorm(d, eps=1e-6)
        s.blocks = nn.ModuleList(Block(d, m, 1.0 / L) for _ in range(L))
        s.fnorm = nn.LayerNorm(d, eps=1e-6)
        s.head = nn.Linear(d, N_FFT + 2)
        s.register_buffer("win", torch.hann_window(N_FFT), persistent=False)

    def spectrum(s, mel):
        x = s.norm(s.embed(mel).transpose(1, 2)).transpose(1, 2)
        for b in s.blocks:
            x = b(x)
        h = s.head(s.fnorm(x.transpose(1, 2))).transpose(1, 2)
        log_mag, ph = h.chunk(2, 1)
        mag = torch.exp(log_mag).clamp(max=1e2)
        return mag * torch.cos(ph), mag * torch.sin(ph)     # real, imag (B, 513, T)

    def forward(s, mel):
        re, im = s.spectrum(mel)
        return istft_same(torch.complex(re.float(), im.float()), s.win)


def _ola(spec, win, n_fft=N_FFT, hop=HOP):
    T = spec.shape[-1]
    y = torch.fft.irfft(spec, n_fft, dim=1) * win[None, :, None]
    L = (T - 1) * hop + n_fft
    y = F.fold(y, (1, L), (1, n_fft), stride=(1, hop))[:, 0, 0]
    env = F.fold((win ** 2)[None, :, None].expand(1, -1, T), (1, L), (1, n_fft), stride=(1, hop))[:, 0, 0]
    p = (n_fft - hop) // 2
    return (y / env.clamp(min=1e-8))[:, p:-p]


def istft_same(spec, win, n_fft=N_FFT, hop=HOP):
    w = _ola(spec, win, n_fft, hop)
    if w.requires_grad:
        return w
    p = (n_fft - hop) // 2
    stft_fn = lambda x: torch.stft(F.pad(x[:, None], (p, p), mode="reflect")[:, 0], n_fft, hop, n_fft, win, center=False, return_complex=True)
    mag0 = spec.abs()
    s1 = stft_fn(w); mag_r, ph_r = s1.abs(), s1.angle()
    w_m = torch.full((1, mag0.shape[1], 1), 0.22, device=mag0.device, dtype=mag0.dtype)
    w_m[:, :160, :] = 0.35
    mb = w_m * mag0 + (1.0 - w_m) * mag_r
    v_ratio = (mb[:, 4:84, :].sum(1, keepdim=True) / (mb.sum(1, keepdim=True) + 1e-6)).clamp(0.0, 1.0)
    vo = ((v_ratio - 0.35) / 0.35).clamp(0.0, 1.0)
    c_t = 1.14 + 0.34 * vo
    mp = (mb.clamp(min=1e-5)) ** c_t
    pts = _EQ["pts"].to(mb.device, mb.dtype)
    eq = (F.interpolate(pts, size=mb.shape[1], mode="linear", align_corners=True)[0, 0] ** 0.78)[None, :, None]
    mc = mp * (mb.sum(1, keepdim=True) / (mp.sum(1, keepdim=True) + 1e-6)) * eq
    w1 = _ola(torch.polar(mc, ph_r), win, n_fft, hop)
    s2 = stft_fn(w1); ph2 = (s2 + 0.27 * (s2 - s1)).angle()
    w2 = _ola(torch.polar(mc, ph2), win, n_fft, hop)
    s3 = stft_fn(w2); ph3 = (s3 + 0.27 * (s3 - s2)).angle()
    w3 = _ola(torch.polar(mc, ph3), win, n_fft, hop)
    return w3 * (0.90 / w3.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4))


def load_compat(G, sd):
    """Load a checkpoint of an older/shallower Vocoder: matching tensors are copied, extra blocks start near-identity (gamma=0, exact identity)."""
    sd = {k: v.float() for k, v in sd.items()}; own = G.state_dict(); miss = [k for k in own if k not in sd]
    G.load_state_dict({**own, **sd}, strict=False)
    with torch.no_grad():
        for k in miss:
            if k.endswith("gamma"): dict(G.named_parameters())[k].fill_(0.0)
    return miss


def n_params(m):
    return sum(p.numel() for p in m.parameters())
