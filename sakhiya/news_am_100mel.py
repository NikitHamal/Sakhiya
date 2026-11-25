"""Phase 2: `Sakhiya-News` Single-Style Studio Nepali News Flagship Acoustic Model (`sakhiya/news_am_100mel.py`).

Hard Architectural & Honesty Rules:
1. ONE STYLE, ONE MODEL, ZERO BLOAT:
   - Deletes all 6 style embeddings, Vedic svara/moraic heads, and the redundant 4-block Fast side-branch (`fast_blocks`/`fast_adapter`).
   - Puts 100% of our ~8.5-10.0 MB INT8 ONNX capacity into ONE deep, wide studio Nepali News flagship (4-step sampling, >= 12x real-time on 1 CPU thread).
2. Full-Sentence RoPE Encoder + Structural Clause Features + Mid-Layer Engram (Layer 3):
   - Feeds phoneme IDs + `<breath>` tokens + explicit 5D per-phoneme structural prosody cues
     (`dist_to_comma`, `dist_to_danda`, `word_pos_in_clause`, `clause_len`, `is_geminate`) + Rotary Position Embeddings (RoPE)
     + compact Layer-3 hashed phoneme-trigram Engram lookup table (~147 KB FP16) so Nepali compounds, geminates, and tech loanwords are razor sharp.
3. Stochastic Conditional Flow-Matching Duration & F0/Energy Predictors (~180 KB FP16):
   - Replaces deterministic L1 regressors with compact 4-step conditional flow-matching variance predictors trained with 15%
     condition dropout for inference-time prosody temperature / CFG control so emphatic geminates stretch naturally and news delivery sounds alive.
4. Single-Branch 10-Block 100-Mel Improved MeanFlow (`iMF`) X-Pred Decoder:
   - Uses Walsh-Hadamard Orthogonal Rotations + Monarch Block-Structured MLPs (100% standard ONNX ops) trained with Improved
     MeanFlow (`iMF`) + 15% Classifier-Free Guidance (CFG) dropout for 4-step studio decoding.
5. Hard FP16 Isolation:
   - Keeps embeddings, stochastic duration/F0/energy flow heads, and final 100-mel projection in FP16 while quantizing the
     RoPE Encoder + 10-block Monarch iMF Decoder to per-channel INT8 QDQ.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sakhiya_one import WalshHadamardMonarchBlock, _build_sylvester_hadamard
from .news_vocoder_100mel import StudioVocoder100Mel, RMSNorm1d
from .v2_vocoder import SR, HOP, N_FFT, ISTFT_N_FFT, ISTFT_HOP
from .v15_rnd_wavehax import ste_int8_weight
from .symbols import N_SYMBOLS, SYM2ID

CONSONANT_IDS_TUPLE = tuple(
    SYM2ID[p] for p in (
        "k", "kʰ", "g", "gʱ", "ŋ", "ts", "tsʰ", "dz", "dzʱ", "ɲ",
        "ʈ", "ʈʰ", "ɖ", "ɖʱ", "ɳ", "t̪", "t̪ʰ", "d̪", "d̪ʱ", "n",
        "p", "pʰ", "b", "bʱ", "m", "j", "r", "l", "w", "ʃ", "s", "h", "ɽ", "ɽʱ", "z", "f"
    ) if p in SYM2ID
)
BREATH_ID = SYM2ID[";"]
COMMA_ID = SYM2ID[","]
DANDA_ID = SYM2ID["."]
WB_ID = SYM2ID["|"]


def sinusoidal_temb(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal time embedding for flow step t in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, device=t.device, dtype=torch.float32) / max(1, half))
    args = t.float().reshape(-1, 1) * freqs[None, :] * 1000.0
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


def compute_graph_structural_features(ids: torch.Tensor) -> torch.Tensor:
    """Compute 5D per-phoneme structural prosody cues inside the ONNX graph using 100% standard tensor ops:
    [dist_to_comma, dist_to_danda, word_pos_in_clause, clause_len, is_geminate] -> shape [B, L, 5].
    """
    B, L = ids.shape
    pos = torch.arange(L, device=ids.device, dtype=torch.float32)[None, :]
    L_f = (pos[:, -1:] + 1.0).clamp(min=2.0)

    is_comma = ((ids == COMMA_ID) | (ids == BREATH_ID)).float()
    is_danda = ((ids == DANDA_ID) | (ids == SYM2ID["?"]) | (ids == SYM2ID["!"])).float()
    is_wb = (ids == WB_ID).float()

    # Cumulative clause & word counters
    clause_id = torch.cumsum(is_comma + is_danda, dim=1)
    n_clauses = (clause_id[:, -1:] + 1.0).clamp(min=1.0)
    clause_len_est = (L_f / n_clauses / 45.0).clamp(0.05, 1.0).expand(B, L)

    # Approximate normalized position within current clause & distance to next comma/danda
    word_idx = torch.cumsum(is_wb, dim=1)
    total_words = (word_idx[:, -1:] + 1.0).clamp(min=1.0)
    word_pos_in_clause = torch.remainder(word_idx, (total_words / n_clauses).clamp(min=1.0)) / (total_words / n_clauses).clamp(min=1.0)

    dist_to_danda = ((L_f - 1.0 - pos) / L_f).clamp(0.0, 1.0)
    dist_to_comma = torch.remainder(L_f - 1.0 - pos, (L_f / n_clauses).clamp(min=2.0)) / (L_f / n_clauses).clamp(min=2.0)

    next_ids = torch.cat([ids[:, 1:], torch.zeros_like(ids[:, -1:])], dim=1)
    prev_ids = torch.cat([torch.zeros_like(ids[:, :1]), ids[:, :-1]], dim=1)
    is_cons = (ids >= 9) & (ids <= 44)
    is_gem = (is_cons & ((ids == next_ids) | (ids == prev_ids))).float()

    return torch.stack([dist_to_comma, dist_to_danda, word_pos_in_clause.clamp(0.0, 1.0), clause_len_est, is_gem], dim=-1)


class RoPESelfAttention1d(nn.Module):
    """Full-Sentence Multi-Head Self-Attention with Rotary Position Embeddings (RoPE) using 100% standard ONNX ops."""
    def __init__(self, d: int = 192, n_heads: int = 4):
        super().__init__()
        assert d % n_heads == 0 and (d // n_heads) % 2 == 0
        self.d = d
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def _apply_rope(self, x: torch.Tensor, cos_p: torch.Tensor, sin_p: torch.Tensor) -> torch.Tensor:
        # x: [B, H, L, D_h]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        r1 = x1 * cos_p - x2 * sin_p
        r2 = x1 * sin_p + x2 * cos_p
        return torch.stack([r1, r2], dim=-1).reshape_as(x)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        half_d = self.head_dim // 2
        pos = torch.arange(L, device=x.device, dtype=torch.float32)[:, None]
        inv_freq = torch.exp(-math.log(10000.0) * torch.arange(0, half_d, device=x.device, dtype=torch.float32) / max(1, half_d))[None, :]
        angles = pos * inv_freq
        cos_p = torch.cos(angles)[None, None, :, :].to(x.dtype)
        sin_p = torch.sin(angles)[None, None, :, :].to(x.dtype)

        q = self._apply_rope(q, cos_p, sin_p)
        k = self._apply_rope(k, cos_p, sin_p)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        mask_bias = (1.0 - m[:, None, None, :].to(x.dtype)) * (-1e4)
        scores = scores + mask_bias
        attn = torch.softmax(scores.float(), dim=-1).to(x.dtype)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, D)
        return self.proj(out) * m[:, :, None].to(x.dtype)


class RoPEEncoderLayer(nn.Module):
    """Full-Sentence RoPE Attention + Local ConvNeXt FFN Encoder Layer."""
    def __init__(self, d: int = 192, ff: int = 384, n_heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = RoPESelfAttention1d(d, n_heads=n_heads)
        self.norm2 = nn.LayerNorm(d)
        self.c1 = nn.Conv1d(d, ff, 5, padding=2)
        self.c2 = nn.Conv1d(ff, d, 3, padding=1)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        mf = m[:, :, None].to(x.dtype)
        x = (x + self.attn(self.norm1(x), m)) * mf
        h = self.norm2(x).transpose(1, 2) * m[:, None, :].to(x.dtype)
        h = self.c2(F.gelu(self.c1(h)) * m[:, None, :].to(x.dtype)).transpose(1, 2)
        return (x + h) * mf


class NewsMidLayerEngram(nn.Module):
    """Layer-3 Hashed Phoneme-Trigram + Geminate/Breath Context Engram Lookup Table (~147 KB in FP16).
    Memorizes exact Nepali consonant conjuncts, geminate closures (`कत्ति`, `पक्कै`, `चिटिक्क`, `ठ्ठयाक्कै`),
    chandrabindu nasalized transitions, and tech loanwords.
    """
    def __init__(self, d: int = 192, n_buckets: int = 384):
        super().__init__()
        self.n_buckets = n_buckets
        self.engram_emb = nn.Embedding(n_buckets, d)
        self.engram_proj = nn.Linear(d, d)
        self.engram_gate = nn.Linear(d, d)
        nn.init.normal_(self.engram_emb.weight, std=0.02)
        nn.init.zeros_(self.engram_proj.weight)
        nn.init.zeros_(self.engram_proj.bias)
        nn.init.constant_(self.engram_gate.bias, -1.0)

    def forward(self, x: torch.Tensor, ids: torch.Tensor, is_gem: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        prev_ids = torch.cat([torch.zeros_like(ids[:, :1]), ids[:, :-1]], dim=1)
        next_ids = torch.cat([ids[:, 1:], torch.zeros_like(ids[:, -1:])], dim=1)
        gem_l = (is_gem > 0.5).to(torch.long)
        h_idx = torch.remainder(prev_ids * 73 + ids * 19 + next_ids * 3 + gem_l * 157, self.n_buckets)
        e = self.engram_emb(h_idx)
        gate = torch.sigmoid(self.engram_gate(x))
        return x + (gate * self.engram_proj(e)) * m[:, :, None].to(x.dtype)


class StochasticFlowVarianceHead(nn.Module):
    """Stochastic 4-Step Conditional Flow-Matching (CFM) Predictor for Duration & F0/Energy (~90 KB FP16 each).
    Trained with 15% condition dropout (`cfg_drop_prob = 0.15`) so inference-time CFG / temperature scaling produces
    expressive, natural Nepali geminate stretching and grounded Kathmandu pitch arcs instead of flat L1 regression.
    """
    def __init__(self, d_in: int = 192, out_dim: int = 1, h_dim: int = 96):
        super().__init__()
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.base_proj = nn.Sequential(
            nn.Conv1d(d_in, h_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(h_dim, h_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(h_dim, out_dim, 1),
        )
        # Conditional flow velocity field v(z_t, t | h_cond)
        self.cond_proj = nn.Conv1d(d_in, h_dim, 1)
        self.z_proj = nn.Conv1d(out_dim, h_dim, 3, padding=1)
        self.t_proj = nn.Linear(32, h_dim)
        self.vel_net = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(h_dim, h_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(h_dim, out_dim, 1),
        )
        nn.init.zeros_(self.vel_net[-1].weight)
        nn.init.zeros_(self.vel_net[-1].bias)

    def forward_vel(self, z_t: torch.Tensor, t: torch.Tensor, h_cond: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        # z_t: [B, out_dim, L], h_cond: [B, d_in, L], m: [B, 1, L]
        te = self.t_proj(sinusoidal_temb(t, 32).to(z_t.dtype))[:, :, None]
        hid = self.z_proj(z_t * m) + self.cond_proj(h_cond * m) + te
        return self.vel_net(hid * m) * m

    def cfm_loss(self, target: torch.Tensor, h: torch.Tensor, im: torch.Tensor, cfg_drop_prob: float = 0.15) -> torch.Tensor:
        # target: [B, out_dim, L], h: [B, L, d_in], im: [B, L]
        B, L, _ = h.shape
        m = im[:, None, :].to(h.dtype)
        h_t = h.transpose(1, 2) * m
        mu_base = self.base_proj(h_t) * m
        l_base = F.smooth_l1_loss(mu_base.float() * m.float(), target.float() * m.float(), reduction="sum") / (m.sum() * self.out_dim + 1.0)

        res_tgt = (target.float() - mu_base.detach().float()) * m.float()
        z0 = torch.randn_like(res_tgt) * 0.35 * m.float()
        t = torch.rand((B,), device=h.device, dtype=torch.float32)
        t_view = t[:, None, None]
        z_t = (1.0 - t_view) * z0 + t_view * res_tgt
        v_target = res_tgt - z0

        drop_mask = (torch.rand((B, 1, 1), device=h.device) >= cfg_drop_prob).to(h_t.dtype)
        h_cond = h_t * drop_mask
        v_pred = self.forward_vel(z_t.to(h_t.dtype), t, h_cond, m)
        l_flow = F.mse_loss(v_pred.float() * m.float(), v_target * m.float(), reduction="sum") / (m.sum() * self.out_dim + 1.0)
        return l_base + 0.5 * l_flow, mu_base

    def sample(self, h: torch.Tensor, im: torch.Tensor, steps: int = 4, cfg_scale: float = 1.25, temp: float = 0.25) -> torch.Tensor:
        # Deterministic sinusoidal pseudo-random initial latent for reproducible ONNX synthesis + 4-step Euler flow integration with CFG
        B, L, _ = h.shape
        m = im[:, None, :].to(h.dtype)
        h_t = h.transpose(1, 2) * m
        mu_base = self.base_proj(h_t) * m

        l_idx = torch.arange(L, device=h.device, dtype=torch.float32)[None, None, :]
        c_idx = torch.arange(1, self.out_dim + 1, device=h.device, dtype=torch.float32)[None, :, None]
        z = torch.sin(l_idx * 12.9898 + c_idx * 78.233) * torch.cos(l_idx * 39.346 - c_idx * 17.13) * temp * m.float()
        dt = 1.0 / float(max(1, steps))
        h_zero = torch.zeros_like(h_t)
        for s in range(steps):
            t_s = torch.full((B,), s * dt, device=h.device, dtype=torch.float32)
            v_cond = self.forward_vel(z.to(h_t.dtype), t_s, h_t, m).float()
            if cfg_scale != 1.0:
                v_uncond = self.forward_vel(z.to(h_t.dtype), t_s, h_zero, m).float()
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = v_cond
            z = (z + dt * v) * m.float()
        return (mu_base.float() + z) * m.float()


class SakhiyaNewsEncoder(nn.Module):
    """Single-Style Full-Sentence RoPE Encoder + 5D Structural Clause Cues + Layer-3 Engram."""
    def __init__(self, n_mel: int = 100, d: int = 160, L: int = 6, ff: int = 256, n_engram: int = 384):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(64, d, padding_idx=0)
        self.lang_anchor = nn.Embedding(2, d)
        nn.init.zeros_(self.lang_anchor.weight)
        self.struct_proj = nn.Linear(5, d)
        self.engram = NewsMidLayerEngram(d=d, n_buckets=n_engram)
        self.layers = nn.ModuleList([RoPEEncoderLayer(d=d, ff=ff, n_heads=4) for _ in range(L)])
        self.nf = nn.LayerNorm(d)
        self.mu = nn.Linear(d, n_mel)
        self.spk_anchor_bias = nn.Parameter(torch.zeros(1, d))

    def forward(self, ids: torch.Tensor, struct_5d: torch.Tensor, m: torch.Tensor, lang: torch.Tensor = None):
        lg = lang.clamp(0, 1) if lang is not None else torch.zeros((ids.shape[0],), dtype=torch.long, device=ids.device)
        x = (
            self.emb(ids.clamp(0, 63)) * math.sqrt(self.d)
            + self.struct_proj(struct_5d.to(self.emb.weight.dtype))
            + self.spk_anchor_bias[:, None, :]
            + self.lang_anchor(lg)[:, None, :]
        )
        is_gem = struct_5d[:, :, 4]
        for idx_l, lyr in enumerate(self.layers):
            x = lyr(x, m)
            if idx_l == 2:  # Layer 3 Mid-Layer Phoneme-Trigram Engram injection
                x = self.engram(x, ids, is_gem, m)
        h = self.nf(x) * m[:, :, None].to(x.dtype)
        return h, self.mu(h) * m[:, :, None].to(x.dtype)


class SakhiyaNewsIMFDecoder(nn.Module):
    """Single-Branch 10-Block 100-Mel Improved MeanFlow (`iMF`) X-Pred Decoder with Walsh-Hadamard Rotations & Monarch MLPs.
    Zero Fast side-branch bloat, zero style embeddings, 15% CFG condition dropout support for 4-step studio decoding.
    """
    def __init__(self, n_mel: int = 100, d: int = 192, nb: int = 10, c: int = 160, ff_mult: float = 2.5):
        super().__init__()
        self.n_mel = n_mel
        self.d = d
        self.nb = nb
        self.c = c
        self.inp = nn.Conv1d(2 * n_mel, d, 1)
        self.tm = nn.Sequential(nn.Linear(c, c), nn.SiLU(), nn.Linear(c, c))
        self.rm = nn.Sequential(nn.Linear(c, c), nn.SiLU(), nn.Linear(c, c))
        nn.init.zeros_(self.rm[2].weight); nn.init.zeros_(self.rm[2].bias)
        self.prog_proj = nn.Conv1d(5, d, 1)  # 4D intra-phoneme progress + 1D frame F0/energy conditioning
        nn.init.zeros_(self.prog_proj.weight); nn.init.zeros_(self.prog_proj.bias)
        self.blocks = nn.ModuleList([
            WalshHadamardMonarchBlock(d=d, c=c, k=7, ff_mult=ff_mult, had_size=16)
            for _ in range(nb)
        ])
        self.out = nn.Conv1d(d, n_mel, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, mu: torch.Tensor, m: torch.Tensor, prog_5d: torch.Tensor = None, r: torch.Tensor = None) -> torch.Tensor:
        c = self.tm(sinusoidal_temb(t, self.c).to(x.dtype))
        if r is not None:
            c = c + self.rm(sinusoidal_temb(t - r, self.c).to(x.dtype))
        h = self.inp(torch.cat([x, mu], dim=1))
        if prog_5d is not None:
            h = h + self.prog_proj(prog_5d.to(x.dtype))
        h = h * m
        for blk in self.blocks:
            h = blk(h, c, m)
        return self.out(h) * m


class AMF0VUVHead100Mel(nn.Module):
    """Continuous F0 (in Hz) + Soft V/UV Predictor operating on full 100-band log-mels + stochastic phoneme F0 contour (kept in FP16)."""
    def __init__(self, n_mels: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_mels, 128, 5, padding=2),
            nn.SiLU(),
            nn.Conv1d(128, 128, 5, padding=2),
            nn.SiLU(),
            nn.Conv1d(128, 2, 3, padding=1),
        )

    def forward(self, mel: torch.Tensor, f0_bias_st: torch.Tensor = None):
        rf = self.net(mel.float()).float()
        f0 = 60.0 + 440.0 * torch.sigmoid(rf[:, 0:1])
        vuv = torch.sigmoid(rf[:, 1:2]).clamp(0.0, 1.0)
        lf0 = torch.log2(f0.clamp(125.0, 380.0))
        if f0_bias_st is not None:
            lf0 = lf0 + f0_bias_st.to(lf0.dtype) * (1.0 / 12.0)
        f0_hz = torch.pow(2.0, lf0).clamp(125.0, 380.0)
        return f0_hz, vuv


def compute_progress_5d(dur: torch.Tensor, ends: torch.Tensor, idx: torch.Tensor, t_arange: torch.Tensor, f0_frame: torch.Tensor) -> torch.Tensor:
    starts = ends - dur
    st_f = torch.gather(starts, 1, idx)
    d_f = torch.gather(dur, 1, idx).clamp(min=1.0)
    k_f = (t_arange[None, :] - st_f).clamp(min=0.0)
    u = ((k_f + 0.5) / d_f).clamp(0.0, 1.0)
    p0 = u
    p1 = torch.sin(u * math.pi)
    p2 = torch.cos(u * math.pi)
    p3 = torch.log1p(d_f) * 0.25
    return torch.stack([p0, p1, p2, p3, f0_frame], dim=1)


class SakhiyaNewsFlagship(nn.Module):
    """Unified Single-Style Studio Nepali News Model (`Sakhiya-News`, ~9.3 MB INT8 ONNX, 4-Step iMF + 100-Mel MS-SB-CQT Vocoder)."""
    def __init__(self, d_enc: int = 160, L_enc: int = 6, ff_enc: int = 256, d_vel: int = 192, nb_vel: int = 10, voc_ch: int = 320, n_engram: int = 384, steps: int = 4, cfg_scale: float = 1.20):
        super().__init__()
        self.steps = steps
        self.cfg_scale = cfg_scale
        self.cfg = dict(
            tier="sakhiya_news", arch="sakhiya_news", d_enc=d_enc, L_enc=L_enc, ff_enc=ff_enc,
            d_vel=d_vel, nb_vel=nb_vel, voc_ch=voc_ch, n_engram=n_engram, steps=steps, cfg_scale=cfg_scale,
        )
        self.enc = SakhiyaNewsEncoder(n_mel=100, d=d_enc, L=L_enc, ff=ff_enc, n_engram=n_engram)
        self.dur_flow = StochasticFlowVarianceHead(d_in=d_enc, out_dim=1, h_dim=80)
        self.pros_flow = StochasticFlowVarianceHead(d_in=d_enc, out_dim=2, h_dim=80)  # [f0_offset, energy_offset]
        self.vel = SakhiyaNewsIMFDecoder(n_mel=100, d=d_vel, nb=nb_vel, c=d_enc, ff_mult=2.5)
        self.f0_head = AMF0VUVHead100Mel(n_mels=100)
        self.voc = StudioVocoder100Mel(n_mels=100, ch=voc_ch).apply_svd_compression()

        self.register_buffer("mel_stats", torch.tensor([-6.1761, 2.4932]))
        self.closure_slope = nn.Parameter(torch.tensor(-0.18))  # Grounded downward Kathmandu clause-final F0 closure
        self.geminate_stretch = nn.Parameter(torch.tensor(1.38))  # Emphatic Nepali geminate stretching multiplier

    def load_phase1_vocoder(self, voc_ck_path: str):
        dev = next(self.parameters()).device
        ck = torch.load(voc_ck_path, map_location=dev, weights_only=False)
        vsd = {k: v.float() for k, v in ck.get("G_ema", ck.get("G", ck)).items()}
        self.voc.load_state_dict(vsd, strict=False)
        if "f0_head" in ck:
            self.f0_head.load_state_dict({k: v.float() for k, v in ck["f0_head"].items()}, strict=False)
        return self

    @torch.no_grad()
    def infer(self, ids: torch.Tensor, lang: torch.Tensor = None, dur_scale: torch.Tensor = None, pitch_shift: torch.Tensor = None) -> torch.Tensor:
        """4-Step Improved MeanFlow (`iMF`) + CFG Studio Nepali News Synthesis into 100-Band ConvNeXt-V2 AWP-iSTFT Vocoder."""
        B, L = ids.shape
        dev = ids.device
        im = torch.ones((B, L), dtype=torch.bool, device=dev)
        lg_zero = (lang.to(torch.float32).reshape(B, 1) * 0.0) if lang is not None else torch.zeros((B, 1), dtype=torch.float32, device=dev)
        z_tok = torch.zeros((B, L), dtype=torch.float32, device=dev) + lg_zero
        ds_tok = (dur_scale.reshape(B, -1) + z_tok) if dur_scale is not None else torch.ones((B, L), dtype=torch.float32, device=dev)
        ps_tok = (pitch_shift.reshape(B, -1) + z_tok) if pitch_shift is not None else torch.zeros((B, L), dtype=torch.float32, device=dev)

        struct_5d = compute_graph_structural_features(ids)
        h, mu = self.enc(ids, struct_5d, im, lang=lang)

        # Stochastic 4-step CFM prosody (F0 & energy) + duration prediction
        pros_2d = self.pros_flow.sample(h, im, steps=4, cfg_scale=1.20, temp=0.20)  # [B, 2, L]
        f0_ph = pros_2d[:, 0, :].clamp(-1.5, 1.5)
        eng_ph = pros_2d[:, 1, :].clamp(-1.2, 1.2)
        mu = mu + (0.06 * eng_ph)[..., None] * 0.35

        log_dur = self.dur_flow.sample(h, im, steps=4, cfg_scale=1.20, temp=0.15)[:, 0, :]  # [B, L]
        dur_frames = (torch.exp(log_dur).clamp(min=1.0, max=55.0) * ds_tok).clamp(min=1.0, max=55.0)

        # Emphatic Nepali geminate stretching & natural 80-180ms inter-clause <breath> preservation
        is_gem = struct_5d[:, :, 4]
        dur_frames = dur_frames * (1.0 + (self.geminate_stretch.clamp(1.25, 1.65) - 1.0) * is_gem)
        is_breath = ((ids == BREATH_ID) | (ids == COMMA_ID)).float()
        dur_frames = torch.where(is_breath > 0.5, torch.clamp(dur_frames, min=9.0, max=16.0), dur_frames)

        pos = torch.arange(L, device=dev, dtype=torch.float32)[None, :]
        L_f = (pos[:, -1:] + 1.0).clamp(min=2.0)
        is_tail = (((pos >= (L_f - 3.0)) & (ids > 8)) | (ids >= 53)).float()
        min_ph = is_tail * 5.0 + (1.0 - is_tail) * ((ids > 8).float() * 3.0 + (ids <= 8).float() * 1.0)
        dur = torch.round(torch.maximum(dur_frames, min_ph)).clamp(min=1.0, max=55.0)

        # Preserve Nepali reduplication hiatus (`आआफ्नो`) and short function words (`र`, `त`, `छ`, `हो`)
        id_a, id_wb, id_r, id_h, id_schwa, id_o = SYM2ID["a"], SYM2ID["|"], SYM2ID["r"], SYM2ID["h"], SYM2ID["ʌ"], SYM2ID["o"]
        next_ids = torch.cat([ids[:, 1:], torch.zeros_like(ids[:, -1:])], dim=1)
        prev_ids = torch.cat([torch.zeros_like(ids[:, :1]), ids[:, :-1]], dim=1)
        next2_ids = torch.cat([ids[:, 2:], torch.zeros_like(ids[:, -2:])], dim=1)
        prev2_ids = torch.cat([torch.zeros_like(ids[:, :2]), ids[:, :-2]], dim=1)
        next3_ids = torch.cat([ids[:, 3:], torch.zeros_like(ids[:, -3:])], dim=1)
        prev3_ids = torch.cat([torch.zeros_like(ids[:, :3]), ids[:, :-3]], dim=1)
        is_hiatus_a1 = ((prev_ids == id_wb) & (ids == id_a) & (next_ids == id_wb) & (next2_ids == id_a)).float()
        is_hiatus_a2 = ((prev3_ids == id_wb) & (prev2_ids == id_a) & (prev_ids == id_wb) & (ids == id_a)).float()
        is_hiatus_wb = ((prev2_ids == id_wb) & (prev_ids == id_a) & (ids == id_wb) & (next_ids == id_a)).float()
        dur = torch.maximum(dur, is_hiatus_a1 * 18.0 + is_hiatus_a2 * 17.0)
        dur = torch.where(is_hiatus_wb > 0.5, torch.full_like(dur, 7.0), dur)
        is_func_c = ((prev_ids <= 6) & (next2_ids <= 6) & ((ids == id_r) | (ids == id_h)) & ((next_ids == id_schwa) | (next_ids == id_o))).float()
        is_func_v = ((prev2_ids <= 6) & (next_ids <= 6) & ((prev_ids == id_r) | (prev_ids == id_h)) & ((ids == id_schwa) | (ids == id_o))).float()
        dur = torch.maximum(dur, is_func_c * 7.0 + is_func_v * 9.0)

        ends = torch.cumsum(dur, dim=1)
        T = ends[0, -1].to(torch.int64)
        t_arange = torch.arange(0, T, device=dev, dtype=torch.float32)
        idx = (t_arange[None, :, None] >= ends[:, None, :]).to(torch.float32).sum(-1).to(torch.int64).clamp(max=L - 1)

        mu_t = mu.transpose(1, 2)
        mu_y = torch.gather(mu_t, 2, idx[:, None, :].expand(-1, mu_t.shape[1], -1))

        # Grounded Kathmandu downward clause-final F0 contour
        dist_danda_tok = struct_5d[:, :, 1]
        is_pause_tok = (ids <= 8).float()
        clause_tail = ((0.28 - dist_danda_tok).clamp(min=0.0) / 0.28) * (1.0 - is_pause_tok)
        f0_contour_tok = (f0_ph + self.closure_slope.clamp(-0.35, -0.08) * 3.5 * clause_tail + ps_tok) * (1.0 - is_pause_tok)
        f0_frame = torch.gather(f0_contour_tok, 1, idx)
        prog_5d = compute_progress_5d(dur, ends, idx, t_arange, f0_frame)
        m_t = torch.ones((B, 1, T), dtype=mu_y.dtype, device=dev)

        # 4-Step Improved MeanFlow (`iMF`) X-Pred Studio Decoding with Classifier-Free Guidance (CFG)
        xt = mu_y.clone()
        dt = 1.0 / float(max(1, self.steps))
        r_one = torch.ones((B,), device=dev, dtype=torch.float32)
        mu_zero = torch.zeros_like(mu_y)
        for s_i in range(self.steps):
            t_cur = torch.full((B,), s_i * dt, device=dev, dtype=torch.float32)
            v_cond = self.vel(xt, t_cur, mu_y, m_t, prog_5d=prog_5d, r=r_one)
            if self.cfg_scale != 1.0 and s_i == 0:
                v_uncond = self.vel(xt, t_cur, mu_zero, m_t, prog_5d=prog_5d, r=r_one)
                v_step = v_uncond + self.cfg_scale * (v_cond - v_uncond)
            else:
                v_step = v_cond
            # X-Pred MeanFlow update toward predicted clean 100-band mel target
            x1_hat = mu_y + v_step
            w_step = dt / max(dt, 1.0 - s_i * dt)
            xt = (1.0 - w_step) * xt + w_step * x1_hat

        # Full 100-band log-mel (fmin=0, fmax=12000, ZERO top-bin clipping!)
        mel_100 = xt * self.mel_stats[1] + self.mel_stats[0]
        f0_hz, vuv = self.f0_head(mel_100, f0_bias_st=f0_frame[:, None, :] * 1.8)
        y, _, _, _, _, _ = self.voc.forward_wav(mel_100, f0_hz, vuv)
        pk = y.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
        return (y * (0.88 / pk)).clamp(-0.98, 0.98)
