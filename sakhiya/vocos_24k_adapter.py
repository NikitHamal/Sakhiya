"""Candidate B: `Sakhiya-One-Vocos` — Pretrained `charactr/vocos-mel-24khz` (13.5M, MIT) +
`Mel80To100Adapter` (`sakhiya/vocos_24k_adapter.py`).

Preserves 100% of pretrained `charactr/vocos-mel-24khz` weights by prepending a lightweight `Mel80To100Adapter`:
1. `proj_80_to_100`: 1x1 linear projection initialized via closed-form ridge regression mapping `Sakhiya-One`'s
   exact 80-band GT log-mels (plus high-band 20-bin residual) -> Vocos's exact 100-band HTK log-mels on pure Pro-v3 audio.
2. `res_blocks`: 2-layer ConvNeXt residual refinement initialized to zero so at step 0 the adapter output is
   100% the closed-form least-squares Vocos mel projection.
3. `PretrainedVocos24kHz`: pure-PyTorch ONNX-native implementation of `charactr/vocos-mel-24khz` using a
   COLA `conv_transpose1d` iSTFT kernel (`n_fft=1024, hop=256`), exporting cleanly to single-graph FP16 and QDQ INT8 ONNX.
"""
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .v2_vocoder import SR, HOP, N_FFT
from .v15_rnd_wavehax import _build_cola_istft_kernel


class AdapterConvNeXtBlock(nn.Module):
    """Lightweight ConvNeXt residual block for Mel80To100Adapter with zero-initialized output projection."""
    def __init__(self, dim: int = 100, h_dim: int = 200, kernel_size: int = 7):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw1 = nn.Linear(dim, h_dim)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(h_dim, dim)
        nn.init.zeros_(self.pw2.weight)
        nn.init.zeros_(self.pw2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dw(x)
        h = self.norm(h.transpose(1, 2))
        h = self.pw2(self.act(self.pw1(h))).transpose(1, 2)
        return x + h


class Mel80To100Adapter(nn.Module):
    """Maps `Sakhiya-One`'s 80-band log-mels (plus optional upper 20 bands) into `charactr/vocos-mel-24khz`'s
    exact 100-band HTK log-mel manifold without touching Vocos's pretrained `backbone.embed`.
    """
    def __init__(self, in_mels: int = 80, out_mels: int = 100, extra_hi_mels: int = 20):
        super().__init__()
        self.in_mels = in_mels
        self.out_mels = out_mels
        self.extra_hi_mels = extra_hi_mels
        self.proj_80_to_100 = nn.Conv1d(in_mels, out_mels, 1, bias=True)
        self.proj_hi20 = nn.Conv1d(extra_hi_mels, out_mels, 1, bias=False)
        nn.init.eye_(self.proj_80_to_100.weight[:, :, 0])
        nn.init.zeros_(self.proj_80_to_100.bias)
        nn.init.zeros_(self.proj_hi20.weight)
        self.res_blocks = nn.ModuleList([
            AdapterConvNeXtBlock(dim=out_mels, h_dim=200, kernel_size=7),
            AdapterConvNeXtBlock(dim=out_mels, h_dim=200, kernel_size=7),
        ])

    @torch.no_grad()
    def fit_closed_form_ridge(self, mel_s1_list: list, mel_vocos_list: list, ridge_lambda: float = 1e-2):
        """Initialize `proj_80_to_100` (and `proj_hi20`) via closed-form ridge regression:
        min_{W, b} || W [mel_80; mel_hi20] + b - mel_vocos_100 ||_2^2 + lambda ||W||_F^2.
        """
        X_80 = torch.cat([m[:80, :].float() for m in mel_s1_list], dim=1).T  # [N_frames, 80]
        X_hi = torch.cat([m[80:100, :].float() for m in mel_s1_list], dim=1).T  # [N_frames, 20]
        Y = torch.cat([m[:100, :].float() for m in mel_vocos_list], dim=1).T  # [N_frames, 100]

        N = X_80.shape[0]
        ones = torch.ones((N, 1), dtype=torch.float32, device=X_80.device)
        X_aug = torch.cat([X_80, X_hi, ones], dim=1)  # [N, 101]
        reg = torch.eye(101, dtype=torch.float32, device=X_80.device) * ridge_lambda
        reg[-1, -1] = 0.0
        XtX = X_aug.T @ X_aug + reg * float(N / 1000.0)
        XtY = X_aug.T @ Y
        W_full = torch.linalg.solve(XtX, XtY)  # [101, 100]

        W_80 = W_full[:80, :].T.contiguous()    # [100, 80]
        W_hi = W_full[80:100, :].T.contiguous() # [100, 20]
        b_100 = W_full[100, :].contiguous()     # [100]

        dev = self.proj_80_to_100.weight.device
        self.proj_80_to_100.weight.data.copy_(W_80[:, :, None].to(dev))
        self.proj_80_to_100.bias.data.copy_(b_100.to(dev))
        self.proj_hi20.weight.data.copy_(W_hi[:, :, None].to(dev))

        pred = X_aug @ W_full
        r2 = 1.0 - float(torch.mean((Y - pred) ** 2) / torch.var(Y))
        l1_err = float(torch.mean(torch.abs(Y - pred)))
        return dict(r2=round(r2, 5), l1_err=round(l1_err, 5), n_frames=int(N))

    def forward(self, mel_s1: torch.Tensor) -> torch.Tensor:
        m80 = mel_s1[:, :80, :]
        y = self.proj_80_to_100(m80)
        if mel_s1.shape[1] >= 100:
            y = y + self.proj_hi20(mel_s1[:, 80:100, :])
        for blk in self.res_blocks:
            y = blk(y)
        return y


class VocosConvNeXtBlock(nn.Module):
    """Exact match to `vocos.modules.ConvNeXtBlock` (`dwconv -> LayerNorm -> pwconv1 -> GELU -> pwconv2 -> gamma`)."""
    def __init__(self, dim: int = 512, intermediate_dim: int = 1536):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(torch.ones(dim) * 0.125)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.transpose(1, 2)
        return residual + x


class PretrainedVocos24kHz(nn.Module):
    """100% weight-preserving pure-PyTorch `charactr/vocos-mel-24khz` (13.5M ConvNeXt + COLA iSTFT Head).
    Zero harmonic comb source, zero `torchaudio` C++ ops, and 100% standard ONNX exportable.
    """
    def __init__(self, input_channels: int = 100, dim: int = 512, intermediate_dim: int = 1536, num_layers: int = 8):
        super().__init__()
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.convnext = nn.ModuleList([VocosConvNeXtBlock(dim, intermediate_dim) for _ in range(num_layers)])
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.head_out = nn.Linear(dim, 1026)

        self.register_buffer("mel_fb", torch.zeros(513, 100), persistent=False)
        self.register_buffer("stft_win", torch.hann_window(1024), persistent=False)
        K_1024, _ = _build_cola_istft_kernel(1024, 256)
        self.register_buffer("K_istft_1024", torch.from_numpy(K_1024), persistent=False)

    @torch.no_grad()
    def load_pretrained_hf(self, bin_path: str = None):
        if bin_path is None:
            from huggingface_hub import hf_hub_download
            bin_path = hf_hub_download("charactr/vocos-mel-24khz", "pytorch_model.bin")
        sd = torch.load(bin_path, map_location="cpu", weights_only=False)
        dev = next(self.parameters()).device

        self.embed.weight.data.copy_(sd["backbone.embed.weight"].float().to(dev))
        self.embed.bias.data.copy_(sd["backbone.embed.bias"].float().to(dev))
        self.norm.weight.data.copy_(sd["backbone.norm.weight"].float().to(dev))
        self.norm.bias.data.copy_(sd["backbone.norm.bias"].float().to(dev))
        self.final_layer_norm.weight.data.copy_(sd["backbone.final_layer_norm.weight"].float().to(dev))
        self.final_layer_norm.bias.data.copy_(sd["backbone.final_layer_norm.bias"].float().to(dev))
        self.head_out.weight.data.copy_(sd["head.out.weight"].float().to(dev))
        self.head_out.bias.data.copy_(sd["head.out.bias"].float().to(dev))

        for i, blk in enumerate(self.convnext):
            p = f"backbone.convnext.{i}."
            blk.dwconv.weight.data.copy_(sd[p + "dwconv.weight"].float().to(dev))
            blk.dwconv.bias.data.copy_(sd[p + "dwconv.bias"].float().to(dev))
            blk.norm.weight.data.copy_(sd[p + "norm.weight"].float().to(dev))
            blk.norm.bias.data.copy_(sd[p + "norm.bias"].float().to(dev))
            blk.pwconv1.weight.data.copy_(sd[p + "pwconv1.weight"].float().to(dev))
            blk.pwconv1.bias.data.copy_(sd[p + "pwconv1.bias"].float().to(dev))
            blk.pwconv2.weight.data.copy_(sd[p + "pwconv2.weight"].float().to(dev))
            blk.pwconv2.bias.data.copy_(sd[p + "pwconv2.bias"].float().to(dev))
            blk.gamma.data.copy_(sd[p + "gamma"].float().to(dev))

        if "feature_extractor.mel_spec.mel_scale.fb" in sd:
            self.mel_fb.copy_(sd["feature_extractor.mel_spec.mel_scale.fb"].float().to(dev))
        if "feature_extractor.mel_spec.spectrogram.window" in sd:
            self.stft_win.copy_(sd["feature_extractor.mel_spec.spectrogram.window"].float().to(dev))
        return self

    def extract_gt_vocos_mel(self, wav: torch.Tensor) -> torch.Tensor:
        """Exact `charactr/vocos-mel-24khz` 100-band HTK log-mel extractor (`sr=24000, n_fft=1024, hop=256, center=True, power=1`)."""
        if wav.dim() == 1:
            wav = wav[None, :]
        spec = torch.stft(
            wav.float(), n_fft=1024, hop_length=256, win_length=1024,
            window=self.stft_win.to(wav.device), center=True, pad_mode="reflect", return_complex=True
        ).abs()
        mel = torch.matmul(spec.transpose(1, 2), self.mel_fb.to(wav.device)).transpose(1, 2)
        return torch.log(torch.clamp(mel, min=1e-7))

    def forward_wav(self, mel_100: torch.Tensor):
        x = self.embed(mel_100)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        for blk in self.convnext:
            x = blk(x)
        x = self.final_layer_norm(x.transpose(1, 2))
        out = self.head_out(x).transpose(1, 2).float()
        log_mag, ph = out[:, :513, :], out[:, 513:, :]
        mag = torch.exp(log_mag).clamp(max=100.0)
        cos_ph = torch.cos(ph)
        sin_ph = torch.sin(ph)
        re_s = mag * cos_ph
        im_s = mag * sin_ph
        z = torch.zeros_like(im_s[:, :1, :])
        spec_cat = torch.cat([re_s, torch.cat([z, im_s[:, 1:-1, :], z], dim=1)], dim=1)
        y = F.conv_transpose1d(
            spec_cat,
            self.K_istft_1024.to(spec_cat.device),
            stride=256,
            padding=384,
        )[:, 0, :]
        return y, re_s, im_s, log_mag, cos_ph, sin_ph

    def forward(self, mel_100: torch.Tensor) -> torch.Tensor:
        y, _, _, _, _, _ = self.forward_wav(mel_100)
        return y
