"""Multi-Resolution Spectral Loss + Multi-Period & Multi-Resolution Spectrogram GAN Critics for Sakhiya-Voca v2."""

from __future__ import annotations

import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from .model import length_mask


def stft_magnitude(wave: torch.Tensor, fft: int, hop: int, window: torch.Tensor) -> torch.Tensor:
    spec = torch.stft(wave.float(), fft, hop, fft, window, return_complex=True)
    return spec.abs().clamp_min(1e-5)


class SpectralLoss(nn.Module):
    """Multi-resolution STFT (`(512, 128), (1024, 256), (2048, 512)`) + 80-band Slaney log-mel reconstruction loss."""
    def __init__(self, sample_rate: int = 24000, ffts: tuple[int, ...] = (512, 1024, 2048)):
        super().__init__()
        self.ffts = ffts
        for fft in ffts:
            self.register_buffer(f"win_{fft}", torch.hann_window(fft), persistent=False)
            mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=fft, n_mels=80, fmin=0.0, fmax=sample_rate / 2.0)
            self.register_buffer(f"mel_{fft}", torch.from_numpy(mel_basis).float(), persistent=False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, sample_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pred = pred.float()
        target = target.float()
        stft_total = torch.tensor(0.0, device=pred.device)
        mel_total = torch.tensor(0.0, device=pred.device)
        for fft in self.ffts:
            hop = fft // 4
            win = getattr(self, f"win_{fft}")
            mel_fb = getattr(self, f"mel_{fft}")
            p_mag = stft_magnitude(pred, fft, hop, win)
            t_mag = stft_magnitude(target, fft, hop, win)
            f_lens = (sample_lengths // hop).clamp(min=1, max=p_mag.shape[-1])
            f_mask = length_mask(f_lens, p_mag.shape[-1])[:, None, :].to(p_mag.dtype)
            denom = (f_mask.sum() * p_mag.shape[1]).clamp_min(1.0)

            diff_l1 = ((p_mag - t_mag).abs() * f_mask).sum() / denom
            diff_log = ((torch.log(p_mag) - torch.log(t_mag)).abs() * f_mask).sum() / denom
            stft_total = stft_total + (diff_l1 + diff_log)

            p_mel = torch.log(torch.matmul(mel_fb, p_mag).clamp_min(1e-5))
            t_mel = torch.log(torch.matmul(mel_fb, t_mag).clamp_min(1e-5))
            mel_denom = (f_mask.sum() * p_mel.shape[1]).clamp_min(1.0)
            mel_total = mel_total + ((p_mel - t_mel).abs() * f_mask).sum() / mel_denom

        n = len(self.ffts)
        return stft_total / n, mel_total / n


class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        ch = (1, 32, 64, 128, 256, 256)
        self.layers = nn.ModuleList([
            weight_norm(nn.Conv2d(ch[i], ch[i + 1], (5, 1), stride=(3 if i < 4 else 1, 1), padding=(2, 0)))
            for i in range(5)
        ])
        self.output = weight_norm(nn.Conv2d(256, 1, (3, 1), padding=(1, 0)))

    def forward(self, wave: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        b, t = wave.shape
        rem = t % self.period
        if rem:
            wave = F.pad(wave, (0, self.period - rem), mode="reflect")
        x = wave.view(b, 1, -1, self.period)
        feats = []
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.1)
            feats.append(x)
        return self.output(x), feats


class SpectrogramDiscriminator(nn.Module):
    def __init__(self, fft: int, hop: int, channels: int = 32):
        super().__init__()
        self.fft = fft
        self.hop = hop
        self.register_buffer("window", torch.hann_window(fft), persistent=False)
        self.layers = nn.ModuleList([
            weight_norm(nn.Conv2d(1, channels, (3, 9), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 3), padding=(1, 1))),
        ])
        self.output = weight_norm(nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)))

    def forward(self, wave: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = stft_magnitude(wave, self.fft, self.hop, self.window).transpose(1, 2)[:, None]
        feats = []
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.1)
            feats.append(x)
        return self.output(x), feats


class Discriminators(nn.Module):
    """Training-only Multi-Period + Multi-Resolution Spectrogram critics."""
    def __init__(
        self,
        periods: tuple[int, ...] = (2, 3, 5, 7, 11),
        resolutions: tuple[tuple[int, int], ...] = ((512, 128), (1024, 256), (2048, 512)),
    ):
        super().__init__()
        self.networks = nn.ModuleList(
            [PeriodDiscriminator(p) for p in periods]
            + [SpectrogramDiscriminator(fft, hop) for fft, hop in resolutions]
        )

    def forward(self, wave: torch.Tensor) -> list[tuple[torch.Tensor, list[torch.Tensor]]]:
        wave = wave.float()
        return [net(wave) for net in self.networks]


def discriminator_loss(real_outs, fake_outs) -> torch.Tensor:
    return sum((r - 1.0).square().mean() + f.square().mean() for (r, _), (f, _) in zip(real_outs, fake_outs)) / len(real_outs)


def generator_gan_loss(real_outs, fake_outs) -> tuple[torch.Tensor, torch.Tensor]:
    adv = sum((f - 1.0).square().mean() for f, _ in fake_outs) / len(fake_outs)
    fm = sum(
        F.l1_loss(f, r.detach())
        for (_, r_feats), (_, f_feats) in zip(real_outs, fake_outs)
        for r, f in zip(r_feats, f_feats)
    )
    total_layers = sum(len(f_feats) for _, f_feats in fake_outs)
    return adv, fm / max(total_layers, 1)
