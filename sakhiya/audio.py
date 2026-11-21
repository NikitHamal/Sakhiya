"""Shared audio front-end (training-time only; inference never needs librosa)."""
import numpy as np
import torch

SR, N_FFT, HOP, WIN, NMEL, FMIN, FMAX = 22050, 1024, 256, 1024, 80, 0.0, 11025.0
_cache = {}


def _fb(device):
    if device not in _cache:
        import librosa
        fb = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=NMEL, fmin=FMIN, fmax=FMAX)
        _cache[device] = (torch.tensor(fb, dtype=torch.float32, device=device),
                          torch.hann_window(WIN, device=device))
    return _cache[device]


def mel_spec(wav, device="cpu"):
    """wav: (B,T) or (T,) float tensor in [-1,1] -> log-mel (B, NMEL, frames)."""
    if wav.dim() == 1:
        wav = wav[None]
    fb, win = _fb(device)
    wav = wav.to(device)
    pad = (N_FFT - HOP) // 2
    x = torch.nn.functional.pad(wav[:, None], (pad, pad), mode="reflect")[:, 0]
    spec = torch.stft(x, N_FFT, HOP, WIN, win, center=False, return_complex=True).abs()
    return torch.log(torch.clamp(torch.matmul(fb, spec), min=1e-5))


def load_wav(path, trim_db=40):
    import librosa
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=SR, res_type="soxr_hq")
    x, idx = librosa.effects.trim(x, top_db=trim_db, frame_length=1024, hop_length=256)
    keep = int(0.12 * SR)
    x = np.pad(x, (keep, keep))
    return (x / (np.abs(x).max() + 1e-6) * 0.95).astype(np.float32)


def f0_track(x, n_frames):
    import pyworld as pw
    f0, t = pw.dio(x.astype(np.float64), SR, f0_floor=60, f0_ceil=450, frame_period=HOP / SR * 1000)
    f0 = pw.stonemask(x.astype(np.float64), f0, t, SR)
    out = np.zeros(n_frames, dtype=np.float32)
    m = min(n_frames, len(f0))
    out[:m] = f0[:m]
    return out
