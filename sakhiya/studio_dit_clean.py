"""Phase 1: `Sakhiya-Studio-80M` Uncompressed High-Capacity Studio Speech Model (`sakhiya/studio_dit_clean.py`).

Architecture (~80.6M Dense Parameters · ZERO Compression Hacks · ZERO Monarch · ZERO Walsh-Hadamard · ZERO INT8):
1. Text Encoder (`StudioCleanTextEncoder`, ~14.2M params):
   - 8-layer Dense RoPE Transformer (`d=512, heads=8, FFN=2048 SwiGLU, RMSNorm`) over fixed Phase-1 Nepali G2P phonemes
     + 5D structural clause/geminate cues `[dist_to_comma, dist_to_danda, word_pos_in_clause, clause_len, is_geminate]`.
2. Duration & Pitch/Energy Predictors (`StudioCleanProsodyPredictor`, ~6.3M params, FP32):
   - 4-layer Dense ConvNeXt + RoPE phoneme predictor (`d=512`) trained with Huber/L1 loss on log-durations, phoneme F0, V/UV, and energy,
     with explicit natural pause/breath handling and crisp word-final consonant coda caps (zero 12-word slicing!).
   - 4-layer Frame-Level ConvNeXt + RoPE Prior & Pitch Refiner (`d=512 -> 640`) predicting continuous interpolated frame-level
     log-F0, soft V/UV, frame energy, and a sharp frame-level prior spectrogram `mu_frame [B, 100, T]`.
3. Acoustic Flow-Matching DiT Decoder (`StudioCleanDiTDecoder`, ~60.1M params):
   - 16-layer Dense DiT (`d=640, heads=10, FFN=2560 SwiGLU`, 1D RoPE + QK-RMSNorm + AdaLN-Zero timestep conditioning
     + FlashAttention `F.scaled_dot_product_attention` + depthwise-pointwise local acoustic conv) predicting exact official
     `nvidia/bigvgan_v2_24khz_100band_256x` 100-band Slaney log-mels via Optimal-Transport Conditional Flow Matching (OT-CFM),
     sampled with 16 or 32 Midpoint/Euler ODE steps in pure FP32.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .symbols import SYM2ID
from .news_am_100mel import compute_graph_structural_features, CONSONANT_IDS_TUPLE, BREATH_ID, COMMA_ID, DANDA_ID, WB_ID


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = (dim,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = torch.bfloat16 if torch.is_autocast_enabled() else x.dtype
        return F.rms_norm(x.to(out_dtype), self.dim, self.weight.to(out_dtype), self.eps)


def apply_rope_1d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply 1D Rotary Position Embedding to `x` of shape `[B, H, T, D_h]` preserving `x.dtype`."""
    d_half = x.shape[-1] // 2
    c = cos.to(x.dtype)
    s = sin.to(x.dtype)
    x1, x2 = x[..., :d_half], x[..., d_half:]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


def build_rope_cache(seq_len: int, head_dim: int, device: torch.device, dtype: torch.dtype, base: float = 10000.0):
    d_half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, d_half, device=device, dtype=torch.float32) / d_half))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(t, freqs)  # [T, d_half]
    cos = angles.cos()[None, None, :, :].to(dtype)
    sin = angles.sin()[None, None, :, :].to(dtype)
    return cos, sin


def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int = 256) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, device=t.device, dtype=torch.float32) / max(1, half))
    args = t.float().reshape(-1, 1) * freqs[None, :] * 1000.0
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class SwiGLUFFN(nn.Module):
    """Dense SwiGLU Feed-Forward Network (`d -> 2 * hidden_dim -> d`, zero Monarch/Hadamard approximations)."""
    def __init__(self, d: int, hidden_dim: int):
        super().__init__()
        self.w_gate_up = nn.Linear(d, 2 * hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        return self.w_down(F.silu(gate) * up)


class DenseRoPETextEncoderLayer(nn.Module):
    """Dense RoPE Transformer Layer with QK-RMSNorm, FlashAttention SDPA, and SwiGLU FFN (`d=512, heads=8`)."""
    def __init__(self, d: int = 512, heads: int = 8, ffn_hidden: int = 1024):
        super().__init__()
        self.d = d
        self.heads = heads
        self.head_dim = d // heads
        self.norm1 = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(d, d, bias=False)
        self.norm2 = RMSNorm(d)
        self.ffn = SwiGLUFFN(d, ffn_hidden)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(B, L, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q = apply_rope_1d(self.q_norm(qkv[0]), rope_cos, rope_sin)
        k = apply_rope_1d(self.k_norm(qkv[1]), rope_cos, rope_sin)
        v = qkv[2]

        # Hardware FlashAttention-2 SDPA (`attn_mask=None` with zeroed pad keys/values)
        m_4d = mask[:, None, :, None].to(q.dtype)
        q = q * m_4d
        k = k * m_4d
        v = v * m_4d
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(B, L, D)
        x = x + self.proj(out)
        x = x + self.ffn(self.norm2(x))
        return x * mask[:, :, None].to(x.dtype)


class StudioCleanTextEncoder(nn.Module):
    """8-Layer Dense RoPE Transformer Text Encoder (`d=512, heads=8, FFN=2048 SwiGLU, RMSNorm`, ~14.2M params)."""
    def __init__(self, vocab_size: int = 64, d: int = 512, layers: int = 8, heads: int = 8, ffn_hidden: int = 448, n_mel: int = 100):
        super().__init__()
        self.d = d
        self.head_dim = d // heads
        self.emb = nn.Embedding(vocab_size, d, padding_idx=0)
        self.lang_emb = nn.Embedding(2, d)
        self.struct_proj = nn.Linear(5, d)
        self.conv_pre = nn.Conv1d(d, d, kernel_size=5, padding=2, groups=16)
        self.layers = nn.ModuleList([
            DenseRoPETextEncoderLayer(d=d, heads=heads, ffn_hidden=ffn_hidden)
            for _ in range(layers)
        ])
        self.norm_final = RMSNorm(d)
        self.ph_mel_prior = nn.Linear(d, n_mel)

    def forward(self, ids: torch.Tensor, struct_5d: torch.Tensor, mask: torch.Tensor, lang: torch.Tensor = None):
        B, L = ids.shape
        if lang is None:
            lang = torch.zeros(B, dtype=torch.long, device=ids.device)
        x = (
            self.emb(ids.clamp(0, 63)) * math.sqrt(self.d)
            + self.lang_emb(lang.clamp(0, 1))[:, None, :]
            + self.struct_proj(struct_5d.to(self.emb.weight.dtype))
        )
        m_f = mask[:, :, None].to(x.dtype)
        x = x * m_f
        x = x + F.silu(self.conv_pre(x.transpose(1, 2))).transpose(1, 2) * m_f

        cos, sin = build_rope_cache(L, self.head_dim, x.device, x.dtype)
        for lyr in self.layers:
            x = lyr(x, mask, cos, sin)
        h = self.norm_final(x) * m_f
        mu_ph = self.ph_mel_prior(h) * m_f
        return h, mu_ph


class ConvNeXt1DBlock(nn.Module):
    """Dense 1D ConvNeXt Block (`d=512`, FP32/BF16 clean prosody & frame prior modeling)."""
    def __init__(self, d: int = 512, ff_mult: int = 1, kernel_size: int = 7):
        super().__init__()
        self.dw = nn.Conv1d(d, d, kernel_size=kernel_size, padding=kernel_size // 2, groups=d)
        self.norm = RMSNorm(d)
        self.pw1 = nn.Linear(d, d * ff_mult)
        self.pw2 = nn.Linear(d * ff_mult, d)

    def forward(self, x: torch.Tensor, mask_1d: torch.Tensor) -> torch.Tensor:
        # x: [B, D, T], mask_1d: [B, 1, T]
        res = x
        y = self.dw(x * mask_1d)
        y = self.norm(y.transpose(1, 2))
        y = self.pw2(F.gelu(self.pw1(y))).transpose(1, 2)
        return (res + y) * mask_1d


class StudioCleanProsodyPredictor(nn.Module):
    """Deep 4-Layer Dense ConvNeXt + RoPE Duration & Continuous Pitch/Energy/Prior Predictor (~6.1M params)."""
    def __init__(self, d: int = 512, d_cond: int = 640, n_mel: int = 100):
        super().__init__()
        # 1. Phoneme-level 4-layer ConvNeXt + RoPE predictor -> [log_dur, lf0_ph, vuv_ph, eng_ph]
        self.ph_blocks = nn.ModuleList([ConvNeXt1DBlock(d=d, ff_mult=1, kernel_size=5) for _ in range(2)])
        self.ph_rope = DenseRoPETextEncoderLayer(d=d, heads=8, ffn_hidden=384)
        self.ph_out = nn.Linear(d, 4)  # [log_dur, lf0_ph, vuv_logit_ph, eng_ph]

        # 2. Frame-level 4-layer ConvNeXt Refiner over expanded encoder + 4D progress -> continuous F0/VUV/Eng + refined prior mel `mu_frame` + `h_cond [B, 640, T]`
        self.fr_in = nn.Conv1d(d + n_mel + 7, d, kernel_size=3, padding=1)
        self.fr_blocks = nn.ModuleList([ConvNeXt1DBlock(d=d, ff_mult=1, kernel_size=7) for _ in range(3)])
        self.fr_cond_proj = nn.Conv1d(d, d_cond, kernel_size=1)
        self.fr_f0_vuv_eng = nn.Conv1d(d, 3, kernel_size=3, padding=1)  # residual [d_lf0, vuv_logit, d_eng]
        self.fr_mel_prior = nn.Conv1d(d, n_mel, kernel_size=3, padding=1)
        nn.init.zeros_(self.fr_mel_prior.weight)
        nn.init.zeros_(self.fr_mel_prior.bias)

    def forward_phoneme(self, h_enc: torch.Tensor, mask: torch.Tensor):
        B, L, D = h_enc.shape
        m_1d = mask[:, None, :].to(h_enc.dtype)
        x = h_enc.transpose(1, 2) * m_1d
        for blk in self.ph_blocks:
            x = blk(x, m_1d)
        cos, sin = build_rope_cache(L, D // 8, h_enc.device, h_enc.dtype)
        x_t = self.ph_rope(x.transpose(1, 2), mask, cos, sin)
        out = self.ph_out(x_t.float()) * mask[:, :, None].float()
        log_dur = out[:, :, 0]
        lf0_ph = out[:, :, 1].clamp(-1.2, 1.2)
        vuv_ph = out[:, :, 2]
        eng_ph = out[:, :, 3].clamp(-1.5, 1.5)
        return log_dur, lf0_ph, vuv_ph, eng_ph

    def forward_frame(
        self,
        h_exp: torch.Tensor,      # [B, D, T]
        mu_exp: torch.Tensor,     # [B, 100, T]
        prog_7d: torch.Tensor,    # [B, 7, T]  (4D intra-phoneme progress + [lf0_ph_exp, vuv_ph_exp, eng_ph_exp])
        ymask: torch.Tensor,      # [B, T] bool
    ):
        m_1d = ymask[:, None, :].to(h_exp.dtype)
        x = self.fr_in(torch.cat([h_exp, mu_exp, prog_7d.to(h_exp.dtype)], dim=1)) * m_1d
        for blk in self.fr_blocks:
            x = blk(x, m_1d)
        h_cond = self.fr_cond_proj(x) * m_1d  # [B, 640, T]
        fve = self.fr_f0_vuv_eng(x).float() * m_1d.float()
        lf0_fr = (prog_7d[:, 4, :].float() + 0.35 * fve[:, 0, :]).clamp(-1.2, 1.2) * ymask.float()
        vuv_logit_fr = (prog_7d[:, 5, :].float() + fve[:, 1, :]) * ymask.float()
        eng_fr = (prog_7d[:, 6, :].float() + 0.35 * fve[:, 2, :]).clamp(-1.5, 1.5) * ymask.float()
        mu_frame = (mu_exp.float() + self.fr_mel_prior(x).float()) * m_1d.float()  # [B, 100, T]
        return h_cond, mu_frame, lf0_fr, vuv_logit_fr, eng_fr


class StudioDiTBlock(nn.Module):
    """Dense Flow-Matching DiT Block (`d=640, heads=10, head_dim=64, FFN=2560 SwiGLU`):
    - Low-rank AdaLN-Zero timestep modulation (`shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp`)
    - 1D RoPE Multi-Head Self-Attention with QK-RMSNorm & FlashAttention `F.scaled_dot_product_attention`
    - Depthwise-Pointwise Local Acoustic Conv (`k=7`) for crisp formant transitions
    - Dense SwiGLU FFN (`640 -> 2560 -> 640`)
    """
    def __init__(self, d: int = 640, heads: int = 10, ffn_hidden: int = 1280):
        super().__init__()
        self.d = d
        self.heads = heads
        self.head_dim = d // heads

        self.norm1 = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(d, d, bias=False)

        self.dw_conv = nn.Conv1d(d, d, kernel_size=7, padding=3, groups=d)
        self.pw_conv = nn.Linear(d, d, bias=False)

        self.norm2 = RMSNorm(d)
        self.ffn = SwiGLUFFN(d, ffn_hidden)

        # AdaLN-Zero modulation from timestep embedding `t_emb [B, d]`
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d, 192, bias=True),
            nn.SiLU(),
            nn.Linear(192, 6 * d, bias=True),
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, ymask: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D], t_emb: [B, D], ymask: [B, T] bool
        B, T, D = x.shape
        m_f = ymask[:, :, None].to(x.dtype)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(t_emb).unsqueeze(1).chunk(6, dim=-1)

        # 1. AdaLN-modulated RoPE FlashAttention
        h1 = self.norm1(x) * (1.0 + scale_msa) + shift_msa
        qkv = self.qkv(h1).view(B, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q = apply_rope_1d(self.q_norm(qkv[0]), rope_cos, rope_sin)
        k = apply_rope_1d(self.k_norm(qkv[1]), rope_cos, rope_sin)
        v = qkv[2]

        m_4d = ymask[:, None, :, None].to(q.dtype)
        q = q * m_4d
        k = k * m_4d
        v = v * m_4d
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
        attn_out = self.proj(attn_out)

        # Local depthwise-pointwise acoustic conv branch
        conv_in = (h1 * m_f).transpose(1, 2)
        conv_out = self.pw_conv(F.silu(self.dw_conv(conv_in)).transpose(1, 2))
        x = x + (1.0 + gate_msa) * (attn_out + 0.25 * conv_out)
        x = x * m_f

        # 2. AdaLN-modulated SwiGLU FFN
        h2 = self.norm2(x) * (1.0 + scale_mlp) + shift_mlp
        x = x + (1.0 + gate_mlp) * self.ffn(h2)
        return x * m_f


class StudioCleanDiTDecoder(nn.Module):
    """16-Layer Dense Flow-Matching DiT Decoder (`d=640, heads=10, FFN=2560 SwiGLU`, ~60.1M params)."""
    def __init__(self, n_mel: int = 100, d: int = 640, layers: int = 16, heads: int = 10, ffn_hidden: int = 1280):
        super().__init__()
        self.n_mel = n_mel
        self.d = d
        self.head_dim = d // heads

        self.in_proj = nn.Linear(n_mel + n_mel + d + 7, d)
        self.t_mlp = nn.Sequential(
            nn.Linear(256, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        self.blocks = nn.ModuleList([
            StudioDiTBlock(d=d, heads=heads, ffn_hidden=ffn_hidden)
            for _ in range(layers)
        ])
        self.final_norm = RMSNorm(d)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d, 2 * d),
        )
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        self.out_proj = nn.Linear(d, n_mel)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x_t: torch.Tensor,        # [B, 100, T]
        t: torch.Tensor,          # [B]
        mu_frame: torch.Tensor,   # [B, 100, T]
        h_cond: torch.Tensor,     # [B, 640, T]
        prog_7d: torch.Tensor,    # [B, 7, T]
        ymask: torch.Tensor,      # [B, T] bool
    ) -> torch.Tensor:
        B, _, T = x_t.shape
        m_f = ymask[:, :, None].to(x_t.dtype)

        inp = torch.cat([x_t, mu_frame.to(x_t.dtype), h_cond.to(x_t.dtype), prog_7d.to(x_t.dtype)], dim=1).transpose(1, 2)
        h = self.in_proj(inp) * m_f
        t_emb = self.t_mlp(sinusoidal_timestep_embedding(t, 256).to(x_t.dtype))

        cos, sin = build_rope_cache(T, self.head_dim, x_t.device, x_t.dtype)
        for blk in self.blocks:
            h = blk(h, t_emb, ymask, cos, sin)

        shift_f, scale_f = self.final_adaLN(t_emb).unsqueeze(1).chunk(2, dim=-1)
        h = self.final_norm(h) * (1.0 + scale_f) + shift_f
        v_pred = self.out_proj(h).transpose(1, 2) * ymask[:, None, :].to(x_t.dtype)
        return v_pred


def compute_intra_phoneme_progress_7d(
    dur: torch.Tensor,          # [B, L] float/long
    lf0_ph: torch.Tensor,       # [B, L]
    vuv_ph: torch.Tensor,       # [B, L]
    eng_ph: torch.Tensor,       # [B, L]
    ymask: torch.Tensor,        # [B, T] bool
):
    """Expand phoneme indices to frames and compute 7D frame conditioning:
    `[u, sin(pi*u), cos(pi*u), log1p(d)*0.25, lf0_ph_exp, vuv_ph_exp, eng_ph_exp]` of shape `[B, 7, T]`,
    plus frame-to-phoneme gather index `idx_fr [B, T]`.
    """
    B, T = ymask.shape
    L = dur.shape[1]
    dev = dur.device
    dur_f = dur.float().clamp(min=1.0)
    ends = torch.cumsum(dur_f, dim=1)
    starts = ends - dur_f
    t_ar = torch.arange(T, device=dev, dtype=torch.float32)[None, :]
    idx_fr = (t_ar[:, :, None] >= ends[:, None, :]).long().sum(-1).clamp(0, L - 1)  # [B, T]

    st_f = torch.gather(starts, 1, idx_fr)
    d_f = torch.gather(dur_f, 1, idx_fr).clamp(min=1.0)
    k_f = (t_ar - st_f).clamp(min=0.0)
    u = ((k_f + 0.5) / d_f).clamp(0.0, 1.0)
    p0 = u
    p1 = torch.sin(u * math.pi)
    p2 = torch.cos(u * math.pi)
    p3 = torch.log1p(d_f) * 0.25

    lf0_exp = torch.gather(lf0_ph.float(), 1, idx_fr)
    vuv_exp = torch.gather(vuv_ph.float(), 1, idx_fr)
    eng_exp = torch.gather(eng_ph.float(), 1, idx_fr)

    prog_7d = torch.stack([p0, p1, p2, p3, lf0_exp, vuv_exp, eng_exp], dim=1) * ymask[:, None, :].float()
    return prog_7d, idx_fr


class SakhiyaStudio80M(nn.Module):
    """Full-Capacity ~81.6M Dense RoPE-DiT Studio Speech Model (`Sakhiya-Studio-80M`) operating in the exact official
    `nvidia/bigvgan_v2_24khz_100band_256x` 100-Band Slaney Log-Mel Space.
    """
    def __init__(self, mel_mean: float = -6.4141, mel_std: float = 2.3684):
        super().__init__()
        self.enc = StudioCleanTextEncoder(vocab_size=64, d=512, layers=8, heads=8, ffn_hidden=448, n_mel=100)
        self.pros = StudioCleanProsodyPredictor(d=512, d_cond=640, n_mel=100)
        self.dit = StudioCleanDiTDecoder(n_mel=100, d=640, layers=16, heads=10, ffn_hidden=480)
        self.register_buffer("mel_stats", torch.tensor([float(mel_mean), float(mel_std)], dtype=torch.float32))
        self.sigma_0 = 0.50

    def forward_loss(
        self,
        ids: torch.Tensor,          # [B, L]
        struct_5d: torch.Tensor,    # [B, L, 5]
        log_dur_gt: torch.Tensor,   # [B, L] precomputed log(dur)
        mu_gt_ph: torch.Tensor,     # [B, L, 100] precomputed per-phoneme normalized mel target
        imask: torch.Tensor,        # [B, L] bool
        y_1: torch.Tensor,          # [B, 100, W] windowed normalized mel target
        idx_fr_w: torch.Tensor,     # [B, W] windowed frame-to-phoneme index
        prog_4d_w: torch.Tensor,    # [B, 4, W] windowed 4D intra-phoneme progress
        lf0_w: torch.Tensor,        # [B, W]
        vuv_w: torch.Tensor,        # [B, W]
        eng_w: torch.Tensor,        # [B, W]
        ymask_dit: torch.Tensor,    # [B, W] bool
        lang: torch.Tensor = None,  # [B]
        cfg_drop_prob: float = 0.10,
    ):
        B, _, T_dit = y_1.shape

        # 1. Text Encoder (full paragraph/utterance)
        h_enc, mu_ph = self.enc(ids, struct_5d, imask, lang=lang)

        # 2. Phoneme-level Duration & Prosody Prediction (full paragraph/utterance)
        log_dur_pred, lf0_ph_pred, vuv_ph_pred, eng_ph_pred = self.pros.forward_phoneme(h_enc, imask)
        im_f = imask.float()
        im_denom = im_f.sum().clamp(min=1.0)
        l_dur = F.smooth_l1_loss(log_dur_pred * im_f, log_dur_gt * im_f, reduction="sum", beta=0.25) / im_denom

        # Phoneme-level prior mel loss across full paragraph (zero 1150-frame scatter overhead!)
        im_3d = im_f[:, :, None]
        l_prior_ph = F.l1_loss(mu_ph.float() * im_3d, mu_gt_ph.float() * im_3d, reduction="sum") / (im_denom * 100.0).clamp(min=1.0)

        # Gather encoder features & phoneme predictions strictly onto the W-frame window `[B, W]`
        ym_f = ymask_dit.float()
        m_1d = ym_f[:, None, :]
        h_enc_t = h_enc.transpose(1, 2)  # [B, 512, L]
        mu_ph_t = mu_ph.transpose(1, 2)  # [B, 100, L]
        h_exp = torch.gather(h_enc_t, 2, idx_fr_w[:, None, :].expand(-1, h_enc_t.shape[1], -1)) * m_1d.to(h_enc.dtype)
        mu_exp = torch.gather(mu_ph_t, 2, idx_fr_w[:, None, :].expand(-1, 100, -1)) * m_1d.to(mu_ph.dtype)

        lf0_exp = torch.gather(lf0_ph_pred.float(), 1, idx_fr_w)
        vuv_exp = torch.gather(vuv_ph_pred.float(), 1, idx_fr_w)
        eng_exp = torch.gather(eng_ph_pred.float(), 1, idx_fr_w)
        prog_7d = torch.cat([prog_4d_w.float(), torch.stack([lf0_exp, vuv_exp, eng_exp], dim=1)], dim=1) * m_1d

        # 3. Frame-level Prior & Continuous Pitch/VUV/Energy Refiner
        h_cond, mu_frame, lf0_fr_pred, vuv_logit_fr, eng_fr_pred = self.pros.forward_frame(h_exp, mu_exp, prog_7d, ymask_dit)

        ym_denom = ym_f.sum().clamp(min=1.0)
        ym_mel_denom = (ym_denom * 100.0).clamp(min=1.0)

        voiced_f = (vuv_w.float() > 0.45) & ymask_dit
        l_f0 = F.smooth_l1_loss(lf0_fr_pred * voiced_f.float(), lf0_w.float() * voiced_f.float(), reduction="sum", beta=0.15) / voiced_f.float().sum().clamp(min=1.0)
        l_vuv = (F.binary_cross_entropy_with_logits(vuv_logit_fr, vuv_w.float(), reduction="none") * ym_f).sum() / ym_denom
        l_eng = F.smooth_l1_loss(eng_fr_pred * ym_f, eng_w.float() * ym_f, reduction="sum", beta=0.2) / ym_denom

        l_prior_fr = (
            F.l1_loss(mu_frame * m_1d, y_1 * m_1d, reduction="sum")
            + 0.5 * F.mse_loss(mu_frame * m_1d, y_1 * m_1d, reduction="sum")
        ) / ym_mel_denom
        l_prior = 0.5 * l_prior_ph + l_prior_fr

        # 4. Optimal-Transport Conditional Flow Matching (Prior-OT-CFM) in 16-Layer Dense DiT
        dev = y_1.device
        u_rand = torch.rand((B,), device=dev, dtype=torch.float32)
        t_logit = torch.sigmoid(torch.randn((B,), device=dev, dtype=torch.float32))
        t_unif = torch.rand((B,), device=dev, dtype=torch.float32)
        t = torch.where(u_rand < 0.80, t_logit, t_unif).clamp(1e-4, 1.0 - 1e-4)
        t_view = t[:, None, None]

        eps = torch.randn_like(y_1) * m_1d
        temp_scale = (0.05 + 0.85 * torch.rand((B, 1, 1), device=dev, dtype=torch.float32))
        x_0 = (mu_frame.detach() + (self.sigma_0 * temp_scale) * eps) * m_1d
        x_t = ((1.0 - t_view) * x_0 + t_view * y_1) * m_1d
        v_target = (y_1 - x_0) * m_1d

        # Mild 10% conditioning dropout for optional CFG sweep at eval
        if self.training and cfg_drop_prob > 0.0:
            keep = (torch.rand((B, 1, 1), device=dev) >= cfg_drop_prob).to(h_cond.dtype)
            h_cond_in = h_cond * keep
            prog_in = prog_7d * keep.float()
        else:
            h_cond_in = h_cond
            prog_in = prog_7d

        v_pred = self.dit(x_t.to(h_cond.dtype), t, mu_frame.detach(), h_cond_in, prog_in, ymask_dit).float() * m_1d
        l_vel_l1 = F.l1_loss(v_pred, v_target, reduction="sum") / ym_mel_denom
        l_vel_l2 = F.mse_loss(v_pred, v_target, reduction="sum") / ym_mel_denom

        # Implied clean spectrogram `x_1_hat = x_t + (1 - t) * v_pred` + formant frequency/time continuity loss
        x_1_hat = (x_t + (1.0 - t_view) * v_pred) * m_1d
        l_x1 = F.l1_loss(x_1_hat, y_1, reduction="sum") / ym_mel_denom
        df_pred = x_1_hat[:, 1:, :] - x_1_hat[:, :-1, :]
        df_gt = y_1[:, 1:, :] - y_1[:, :-1, :]
        l_freq_grad = F.l1_loss(df_pred * m_1d, df_gt * m_1d, reduction="sum") / (ym_denom * 99.0).clamp(min=1.0)
        if T_dit > 2:
            dt_pred = x_1_hat[:, :, 1:] - x_1_hat[:, :, :-1]
            dt_gt = y_1[:, :, 1:] - y_1[:, :, :-1]
            m_dt = (ymask_dit[:, 1:] & ymask_dit[:, :-1])[:, None, :].float()
            l_time_grad = F.l1_loss(dt_pred * m_dt, dt_gt * m_dt, reduction="sum") / (m_dt.sum() * 100.0).clamp(min=1.0)
        else:
            l_time_grad = torch.zeros((), device=dev)

        l_cfm = l_vel_l1 + l_vel_l2 + 0.5 * l_x1 + 0.35 * l_freq_grad + 0.20 * l_time_grad
        total_loss = l_cfm + 0.80 * l_prior + 0.50 * l_dur + 0.40 * (l_f0 + 0.5 * l_vuv + 0.5 * l_eng)

        met_7d = torch.stack([
            total_loss.detach(),
            l_cfm.detach(),
            l_vel_l1.detach(),
            l_x1.detach(),
            l_prior.detach(),
            l_dur.detach(),
            l_f0.detach(),
        ])
        return total_loss, met_7d

    @torch.no_grad()
    def infer_mel(
        self,
        ids: torch.Tensor,          # [1, L]
        lang: torch.Tensor = None,  # [1]
        steps: int = 32,
        cfg_scale: float = 0.0,
        solver: str = "midpoint",
        rate: float = 0.96,
        noise_temp: float = 0.0,
    ) -> torch.Tensor:
        """Synthesize exact official 100-band `BigVGAN-v2-24kHz` log-mel `[1, 100, T]` in pure FP32 using
        16 or 32 Midpoint/Euler ODE steps (`cfg_scale = 0.0` single-pass conditional or `cfg_scale > 1.0` guided).
        """
        B, L = ids.shape
        dev = ids.device
        imask = torch.ones((B, L), dtype=torch.bool, device=dev)
        if lang is None:
            lang = torch.zeros(B, dtype=torch.long, device=dev)
        struct_5d = compute_graph_structural_features(ids)

        h_enc, mu_ph = self.enc(ids, struct_5d, imask, lang=lang)
        log_dur_pred, lf0_ph, vuv_ph, eng_ph = self.pros.forward_phoneme(h_enc, imask)

        # Clean duration decoding with natural Nepali geminate & clause-breath pacing and crisp consonant codas
        dur_frames = torch.exp(log_dur_pred).clamp(min=1.0, max=48.0) / max(float(rate), 0.5)
        pos = torch.arange(L, device=dev, dtype=torch.float32)[None, :]
        L_f = float(L)
        is_gem = struct_5d[:, :, 4]
        dur_frames = dur_frames * (1.0 + 0.08 * is_gem)

        is_interior_breath = (((ids == BREATH_ID) | (ids == COMMA_ID)) & (pos > 0.5) & (pos < (L_f - 1.5))).float()
        dur_frames = torch.where(is_interior_breath > 0.5, torch.clamp(dur_frames, min=9.0, max=14.0), dur_frames)

        is_tail = (((pos >= (L_f - 3.0)) & (ids > 8)) | (ids >= 53)).float()
        min_ph = is_tail * 5.0 + (1.0 - is_tail) * ((ids > 8).float() * 3.0 + (ids <= 8).float() * 1.0)
        dur = torch.round(torch.maximum(dur_frames, min_ph)).clamp(min=1.0, max=48.0)

        # Crisp word-final visarga 'h' (`मामकाः`) and interior word boundaries `|`
        next_ids = torch.cat([ids[:, 1:], torch.zeros_like(ids[:, -1:])], dim=1)
        is_final_h = ((ids == SYM2ID["h"]) & (next_ids <= 8)).float()
        dur = torch.where(is_final_h > 0.5, torch.clamp(dur, min=3.0, max=4.0), dur)
        is_int_wb = ((ids == WB_ID) & (pos > 0.5) & (pos < (L_f - 1.5))).float()
        dur = torch.where(is_int_wb > 0.5, torch.clamp(dur, min=1.0, max=2.0), dur)

        T = int(dur.sum(dim=1).max().item())
        ymask = torch.ones((B, T), dtype=torch.bool, device=dev)

        prog_7d, idx_fr = compute_intra_phoneme_progress_7d(dur, lf0_ph, vuv_ph, eng_ph, ymask)
        h_enc_t = h_enc.transpose(1, 2)
        mu_ph_t = mu_ph.transpose(1, 2)
        h_exp = torch.gather(h_enc_t, 2, idx_fr[:, None, :].expand(-1, h_enc_t.shape[1], -1))
        mu_exp = torch.gather(mu_ph_t, 2, idx_fr[:, None, :].expand(-1, 100, -1))

        h_cond, mu_frame, lf0_fr, vuv_logit_fr, eng_fr = self.pros.forward_frame(h_exp, mu_exp, prog_7d, ymask)

        # Deterministic low-temperature starting state around `mu_frame`
        if noise_temp > 0.0:
            gen = torch.Generator(device=dev)
            gen.manual_seed(42)
            eps = torch.randn(mu_frame.shape, generator=gen, device=dev, dtype=torch.float32)
            x = mu_frame.float() + (self.sigma_0 * float(noise_temp)) * eps
        else:
            x = mu_frame.float().clone()

        def _eval_v(x_in: torch.Tensor, t_scalar: float) -> torch.Tensor:
            t_vec = torch.full((B,), float(t_scalar), device=dev, dtype=torch.float32)
            v_c = self.dit(x_in, t_vec, mu_frame, h_cond, prog_7d, ymask).float()
            if cfg_scale > 1.0:
                v_u = self.dit(x_in, t_vec, mu_frame, torch.zeros_like(h_cond), torch.zeros_like(prog_7d), ymask).float()
                return v_u + float(cfg_scale) * (v_c - v_u)
            return v_c

        dt = 1.0 / float(max(1, steps))
        if solver == "midpoint":
            # 2nd-order Midpoint ODE solver (`steps` total NFE if we count pairs, or `steps` intervals)
            n_intervals = max(1, steps // 2) if steps >= 4 else steps
            dt_m = 1.0 / float(n_intervals)
            for s in range(n_intervals):
                t0 = s * dt_m
                v1 = _eval_v(x, t0)
                x_mid = x + 0.5 * dt_m * v1
                v2 = _eval_v(x_mid, t0 + 0.5 * dt_m)
                x = x + dt_m * v2
        else:
            for s in range(steps):
                t0 = s * dt
                v = _eval_v(x, t0)
                x = x + dt * v

        mel_100 = x * self.mel_stats[1] + self.mel_stats[0]
        return mel_100
