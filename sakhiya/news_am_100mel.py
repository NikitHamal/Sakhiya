"""Phase 2 (Leg 2): `Sakhiya-News` Single-Style Studio Nepali News Flagship Acoustic Model (`sakhiya/news_am_100mel.py`).

Leg-2 Studio Physics & Prosody Upgrades (warm-starting cleanly from Leg-1 `am_last.pt` via `strict=False`):
1. Bandwidth-Masked Studio Supervision & Algebraic Speaker/Studio FiLM:
   - Lightweight zero-initialized residual `(spk_id, is_real_unvc_24k)` FiLM tables during training, algebraically baked
     into `spk_anchor_bias` (`spk=nep_6834, is_real_unvc_24k=1`) before ONNX export for 0 extra bytes/FLOPs.
   - Masks mel reconstruction & velocity losses above 7.2 kHz (bins 75..100) on any clip with `is_real_unvc_24k == False`.
2. Coupled Utterance-Level 4-Step Prosody Flow (~195 KB FP16) + Distilled Nepali Break/Focus Head:
   - Replaces independent duration/F0/energy heads with ONE coupled bidirectional phoneme-level 4-step CFM predictor
     (`CoupledProsodyFlowHead`) jointly modeling `[log_dur, f0_mean, f0_slope, energy, creak_prob, post_word_breath_dur]`
     conditioned on the RoPE encoder + cumulative `syllables_since_last_breath` feature.
   - Auxiliary `break_focus_head` on `SakhiyaNewsEncoder` distilling Phase-0 LLM `(implicit_break, focus_prominence)`
     labels into the RoPE encoder with zero inference overhead.
3. Zero Inference CFG (`CFG = 0`) + Unrolled 2-Step / 4-Step X-Pred MeanFlow:
   - Infers strictly at `CFG = 0` (no 2-pass unconditional evaluation at inference, cutting RTF and eliminating plastic CFG over-sharpening).
   - Feeds predicted `creak_prob` and F0 slope into `StudioVocoder100Mel` (with 0-MB OLA-consistent stochastic noise branch + 0.5*F0 creak).
"""
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sakhiya_one import WalshHadamardMonarchBlock, _build_sylvester_hadamard
from .news_vocoder_100mel import StudioVocoder100Mel, RMSNorm1d, _make_svd_conv1d
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
LONG_VOWEL_IDS_SET = {SYM2ID[p] for p in ("a", "e", "o", "ã", "ĩ", "ũ", "ẽ", "õ") if p in SYM2ID}
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

    clause_id = torch.cumsum(is_comma + is_danda, dim=1)
    n_clauses = (clause_id[:, -1:] + 1.0).clamp(min=1.0)
    clause_len_est = (L_f / n_clauses / 45.0).clamp(0.05, 1.0).expand(B, L)

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


def compute_syllables_since_last_breath(ids: torch.Tensor) -> torch.Tensor:
    """Compute cumulative `syllables_since_last_breath` [B, 1, L] in [0, 1] inside the ONNX graph using standard ops."""
    is_vowel = (ids >= 45) & (ids <= 56)
    is_reset = (ids == BREATH_ID) | (ids == COMMA_ID) | (ids == DANDA_ID) | (ids == SYM2ID["?"]) | (ids == SYM2ID["!"])
    v_cum = torch.cumsum(is_vowel.float(), dim=1)
    # Approximate clause-local syllable count by modulo over average clause syllable budget (20 syllables)
    r_cnt = torch.cumsum(is_reset.float(), dim=1)
    tot_r = (r_cnt[:, -1:] + 1.0).clamp(min=1.0)
    tot_v = (v_cum[:, -1:] + 1.0).clamp(min=4.0)
    syl_per_clause = (tot_v / tot_r).clamp(min=6.0, max=24.0)
    syl_local = torch.remainder(v_cum, syl_per_clause) * (1.0 - is_reset.float())
    return (syl_local / 24.0).clamp(0.0, 1.0)[:, None, :]


def _rope_split_half(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Split-half Rotary Position Embedding (RoPE) on [B, H, T, D] using 100% standard ONNX ops."""
    T, D = x.shape[2], x.shape[3]
    h = D // 2
    f = 1.0 / (base ** (torch.arange(h, device=x.device, dtype=torch.float32) / h))
    a = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * f[None]
    c, s = a.cos()[None, None, :, :].to(x.dtype), a.sin()[None, None, :, :].to(x.dtype)
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class RoPEEncoderLayer(nn.Module):
    """Full-Sentence RoPE Self-Attention + Local ConvNeXt FFN Encoder Layer (matches `EncLayer` layout for zero-loss warm-init)."""
    def __init__(self, d: int = 144, h: int = 4, ff: int = 224, k: int = 5, r_enc: int = 104):
        super().__init__()
        self.h = h
        self.scale = 1.0 / math.sqrt(d // h)
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        if r_enc is not None:
            self.c1 = nn.Sequential(
                nn.Conv1d(d, r_enc, k, padding=k // 2, bias=False),
                nn.Conv1d(r_enc, ff, 1, bias=True),
            )
            self.c2 = nn.Sequential(
                nn.Conv1d(ff, r_enc, k, padding=k // 2, bias=False),
                nn.Conv1d(r_enc, d, 1, bias=True),
            )
        else:
            self.c1 = nn.Conv1d(d, ff, k, padding=k // 2)
            self.c2 = nn.Conv1d(ff, d, k, padding=k // 2)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        qkv = self.qkv(self.n1(x)).view(B, T, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        q, k, v = _rope_split_half(qkv[0]), _rope_split_half(qkv[1]), qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        mask_bias = (1.0 - m[:, None, None, :].to(x.dtype)) * (-1e4)
        attn = torch.softmax((scores + mask_bias).float(), dim=-1).to(x.dtype)
        a = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, d)
        x = x + self.o(a)
        y = self.n2(x).transpose(1, 2) * m[:, None].to(x.dtype)
        return x + self.c2(F.gelu(self.c1(y))).transpose(1, 2)


class NewsMidLayerEngram(nn.Module):
    """Layer-3 Hashed Phoneme-Trigram + Geminate/Breath Context Engram Lookup Table (~110 KB in FP16)."""
    def __init__(self, d: int = 144, n_buckets: int = 384):
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
        base_hash = torch.remainder(prev_ids * 73 + ids * 19 + next_ids * 3, 192)
        h_idx = base_hash + gem_l * 192
        e = self.engram_emb(h_idx.clamp(0, self.n_buckets - 1))
        gate = torch.sigmoid(self.engram_gate(x))
        return x + (gate * self.engram_proj(e)) * m[:, :, None].to(x.dtype)


class CoupledProsodyFlowHead(nn.Module):
    """Coupled Utterance-Level 4-Step Phoneme Prosody Flow (~195 KB FP16, `CFG = 0` at inference):
    Replaces independent duration/F0/energy heads with ONE coupled bidirectional phoneme-level 4-step CFM predictor
    jointly modeling the 6D vector:
      `[log_dur, f0_mean, f0_slope, energy, creak_prob, post_word_breath_dur]`
    conditioned on the RoPE encoder `h` [B, L, d_in] + `syllables_since_last_breath` [B, 1, L].
    Preserves the warm-started 4-step unrolled duration anchor (`i, t_steps, cv.0..3, o`) and base F0/energy projector
    from `am_last.pt` via `strict=False` so timing stability is 100% preserved from step 0!
    """
    def __init__(self, d_in: int = 144, out_dim: int = 6, h_dim: int = 96):
        super().__init__()
        self.d_in = d_in
        self.out_dim = out_dim
        self.h_dim = h_dim

        # Warm-started 4-step unrolled duration anchor (exact parameter names from Leg-1 `dur_flow`)
        self.i = nn.Conv1d(d_in + 1, h_dim, 1)
        self.t_steps = nn.Parameter(torch.zeros(4, h_dim, 1))
        self.cv = nn.ModuleList([nn.Conv1d(h_dim, h_dim, 3, padding=1) for _ in range(4)])
        self.o = nn.Conv1d(h_dim, 1, 1)

        # Base 5D non-duration prosody prior `[f0_mean, f0_slope, energy, creak_prob, post_word_breath_dur]`
        self.base_pros_proj = nn.Sequential(
            nn.Conv1d(d_in + 1, h_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(h_dim, h_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(h_dim, 5, 1),
        )
        nn.init.zeros_(self.base_pros_proj[-1].weight)
        nn.init.zeros_(self.base_pros_proj[-1].bias)

        # Coupled 6D bidirectional phoneme-level 4-step CFM velocity field `v(z_t, t | h, syl_since_breath)`
        self.cond_proj = nn.Conv1d(d_in + 1, h_dim, 1)
        self.z_proj = nn.Conv1d(out_dim, h_dim, 3, padding=1)
        self.t_proj = nn.Linear(32, h_dim)
        self.vel_net = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(h_dim, h_dim, 5, padding=2),
            nn.SiLU(),
            nn.Conv1d(h_dim, h_dim, 3, padding=2, dilation=2),
            nn.SiLU(),
            nn.Conv1d(h_dim, out_dim, 1),
        )
        nn.init.zeros_(self.vel_net[-1].weight)
        nn.init.zeros_(self.vel_net[-1].bias)

    def _base_dur_step(self, x: torch.Tensor, hs: torch.Tensor, m: torch.Tensor, step_idx: int) -> torch.Tensor:
        hs_in = 0.85 * hs.detach() + 0.15 * hs
        t_add = self.t_steps[step_idx : step_idx + 1].to(hs.dtype)
        z = self.i(torch.cat([hs_in.transpose(1, 2), x[:, None].to(hs.dtype)], dim=1)) + t_add
        for c in self.cv:
            z = z + F.gelu(c(z * m[:, None].to(z.dtype)))
        return self.o(z)[:, 0] * m.to(z.dtype)

    def pred_base_6d(self, ids: torch.Tensor, h: torch.Tensor, im: torch.Tensor, syl_cnt: torch.Tensor = None) -> torch.Tensor:
        B, L, _ = h.shape
        m = im[:, None, :].to(h.dtype)
        if syl_cnt is None:
            syl_cnt = compute_syllables_since_last_breath(ids)
        pos = torch.arange(L, device=ids.device, dtype=torch.float32)[None, :]
        x_d = (torch.sin(pos * 1.6180339 + ids.float() * 0.37) * 0.20).to(h.dtype)
        for idx_s in range(4):
            x_d = x_d + self._base_dur_step(x_d, h, im, idx_s) * 0.25
        mu_dur = (x_d * im.to(x_d.dtype)).float()[:, None, :]  # [B, 1, L]

        h_cat = torch.cat([h.transpose(1, 2), syl_cnt.to(h.dtype)], dim=1) * m
        mu_pros5 = (self.base_pros_proj(h_cat) * m).float()     # [B, 5, L]
        return torch.cat([mu_dur, mu_pros5], dim=1)            # [B, 6, L]

    def forward_x1(self, z_t: torch.Tensor, t: torch.Tensor, h_cat: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        te = self.t_proj(sinusoidal_temb(t, 32).to(z_t.dtype))[:, :, None]
        hid = self.z_proj(z_t * m) + self.cond_proj(h_cat * m) + te
        return self.vel_net(hid * m) * m

    def coupled_cfm_loss(self, target_6d: torch.Tensor, ids: torch.Tensor, h: torch.Tensor, im: torch.Tensor, syl_cnt: torch.Tensor = None):
        """Train Coupled 6D Prosody Flow (`[log_dur, f0_mean, f0_slope, energy, creak_prob, post_word_breath_dur]`)
        plus positive within-word Duration-F0-Energy covariance regularizer.
        """
        B, L, _ = h.shape
        m = im[:, None, :].to(h.dtype)
        if syl_cnt is None:
            syl_cnt = compute_syllables_since_last_breath(ids)
        mu_6d = self.pred_base_6d(ids, h, im, syl_cnt)  # [B, 6, L]
        l_base = F.smooth_l1_loss(mu_6d * m.float(), target_6d.float() * m.float(), reduction="sum") / (m.sum() * 6.0 + 1.0)

        res_tgt = (target_6d.float() - mu_6d.detach()).clamp(-0.5, 0.5) * m.float()
        z0 = torch.randn_like(res_tgt) * 0.12 * m.float()
        t = torch.rand((B,), device=h.device, dtype=torch.float32)
        t_view = t[:, None, None]
        z_t = (1.0 - t_view) * z0 + t_view * res_tgt
        v_target = res_tgt - z0

        h_cat = torch.cat([h.transpose(1, 2), syl_cnt.to(h.dtype)], dim=1) * m
        x1_pred = self.forward_x1(z_t.to(h.dtype), t, h_cat, m).float()
        v_pred = (x1_pred - z0) * m.float()
        l_flow = F.mse_loss(v_pred, v_target * m.float(), reduction="sum") / (m.sum() * 6.0 + 1.0)

        # Positive within-word Duration-F0-Energy prominence coupling loss
        pred_6d = mu_6d + 0.25 * x1_pred
        d_c = (pred_6d[:, 0, :] - target_6d[:, 0, :].float()) * im.float()
        f_c = (pred_6d[:, 1, :] - target_6d[:, 1, :].float()) * im.float()
        e_c = (pred_6d[:, 3, :] - target_6d[:, 3, :].float()) * im.float()
        l_cov = 0.15 * (F.l1_loss(d_c * f_c, torch.zeros_like(d_c)) + F.l1_loss(d_c * e_c, torch.zeros_like(d_c)))
        return l_base + 0.5 * l_flow + l_cov, pred_6d

    def sample(self, ids: torch.Tensor, h: torch.Tensor, im: torch.Tensor, steps: int = 4, temp: float = 0.05) -> torch.Tensor:
        """4-Step Coupled Phoneme Prosody Sampling at `CFG = 0` (single-pass per step, zero unconditional overhead!)."""
        B, L, _ = h.shape
        m = im[:, None, :].to(h.dtype)
        syl_cnt = compute_syllables_since_last_breath(ids)
        h_cat = torch.cat([h.transpose(1, 2), syl_cnt.to(h.dtype)], dim=1) * m
        mu_6d = self.pred_base_6d(ids, h, im, syl_cnt)  # [B, 6, L]

        l_idx = torch.arange(L, device=h.device, dtype=torch.float32)[None, None, :]
        c_idx = torch.arange(1, self.out_dim + 1, device=h.device, dtype=torch.float32)[None, :, None]
        z0 = torch.sin(l_idx * 12.9898 + c_idx * 78.233) * torch.cos(l_idx * 39.346 - c_idx * 17.13) * temp * m.float()
        z = z0.clone()
        dt = 1.0 / float(max(1, steps))
        for s in range(steps):
            t_s = torch.full((B,), s * dt, device=h.device, dtype=torch.float32)
            x1 = self.forward_x1(z.to(h.dtype), t_s, h_cat, m).float()
            v = x1 - z0
            z = (z + dt * v) * m.float()
        return (mu_6d + 0.25 * z.clamp(-0.15, 0.15)) * m.float()


class SakhiyaNewsEncoder(nn.Module):
    """Single-Style Full-Sentence RoPE Encoder + 5D Structural Clause Cues + Layer-3 Engram + Distilled Break/Focus Head."""
    def __init__(self, n_mel: int = 100, d: int = 144, L: int = 6, ff: int = 224, n_engram: int = 384, r_enc: int = 104):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(61, d, padding_idx=0)
        self.struct_proj = nn.Linear(5, d)
        nn.init.zeros_(self.struct_proj.weight)
        nn.init.zeros_(self.struct_proj.bias)
        self.pitch = nn.Linear(1, d)
        nn.init.zeros_(self.pitch.weight)
        nn.init.zeros_(self.pitch.bias)
        self.energy_pred = nn.Linear(d, 1)
        nn.init.zeros_(self.energy_pred.weight)
        nn.init.zeros_(self.energy_pred.bias)
        # Auxiliary training-only head distilling Phase-0 LLM `(implicit_break, focus_prominence)` into RoPE encoder (0 inference cost!)
        self.break_focus_head = nn.Linear(d, 2)
        nn.init.zeros_(self.break_focus_head.weight)
        nn.init.zeros_(self.break_focus_head.bias)

        self.engram = NewsMidLayerEngram(d=d, n_buckets=n_engram)
        self.layers = nn.ModuleList([RoPEEncoderLayer(d=d, h=4, ff=ff, k=5, r_enc=r_enc) for _ in range(L)])
        self.nf = nn.LayerNorm(d)
        self.mu = nn.Linear(d, n_mel)
        # Single-voice Kathmandu studio anchor conditioning biases
        self.spk_anchor_bias = nn.Parameter(torch.zeros(1, d))
        self.post_anchor_bias = nn.Parameter(torch.zeros(1, d))
        # Zero-initialized residual speaker/studio FiLM tables for multi-speaker/bandwidth-masked training (algebraically folded before export!)
        self.spk_film = nn.Embedding(4, d)
        self.studio_24k_film = nn.Embedding(2, d)
        nn.init.zeros_(self.spk_film.weight)
        nn.init.zeros_(self.studio_24k_film.weight)

    @torch.no_grad()
    def fold_studio_anchor_film(self):
        """Algebraically fold `spk=0 (nep_6834), is_real_unvc_24k=1` into `spk_anchor_bias` and zero `spk_film`/`studio_24k_film`."""
        add_vec = self.spk_film.weight[0:1].data + self.studio_24k_film.weight[1:2].data
        self.spk_anchor_bias.data.add_(add_vec)
        self.spk_film.weight.data.zero_()
        self.studio_24k_film.weight.data.zero_()

    def forward(self, ids: torch.Tensor, struct_5d: torch.Tensor, m: torch.Tensor, pitch_add: torch.Tensor = None, spk_ids: torch.Tensor = None, is_unvc_24k: torch.Tensor = None):
        x = (
            self.emb(ids.clamp(0, 60)) * math.sqrt(self.d)
            + self.spk_anchor_bias[:, None, :]
            + self.struct_proj(struct_5d.to(self.emb.weight.dtype))
        )
        if spk_ids is not None:
            x = x + self.spk_film(spk_ids.clamp(0, 3))[:, None, :]
        if is_unvc_24k is not None:
            x = x + self.studio_24k_film(is_unvc_24k.long().clamp(0, 1))[:, None, :]
        if pitch_add is not None:
            x = x + self.pitch(pitch_add.to(x.dtype)[..., None])
        is_gem = struct_5d[:, :, 4]
        for idx_l, lyr in enumerate(self.layers):
            x = lyr(x, m)
            if idx_l == 2:  # Layer 3 Mid-Layer Phoneme-Trigram Engram injection
                x = self.engram(x, ids, is_gem, m)
        h_content = self.nf(x)
        h = (h_content + self.post_anchor_bias[:, None, :]) * m[:, :, None].to(x.dtype)
        return h, self.mu(h) * m[:, :, None].to(x.dtype)


class SakhiyaNewsIMFDecoder(nn.Module):
    """Single-Branch 10-Block 100-Mel Improved MeanFlow (`iMF`) X-Pred Decoder with Walsh-Hadamard Rotations & Monarch MLPs.
    Trained and evaluated strictly at `CFG = 0` (zero 2-pass CFG overhead or plastic over-sharpening).
    """
    def __init__(self, n_mel: int = 100, d: int = 160, nb: int = 10, c: int = 144):
        super().__init__()
        self.n_mel = n_mel
        self.d = d
        self.nb = nb
        self.c = c
        self.inp = nn.Conv1d(2 * n_mel, d, 1)
        self.tm = nn.Sequential(nn.Linear(c, d), nn.SiLU(), nn.Linear(d, d))
        self.rm = nn.Sequential(nn.Linear(c, d), nn.SiLU(), nn.Linear(d, d))
        nn.init.zeros_(self.tm[2].weight); nn.init.zeros_(self.tm[2].bias)
        nn.init.zeros_(self.rm[2].weight); nn.init.zeros_(self.rm[2].bias)
        self.prog_proj = nn.Conv1d(5, d, 1)  # 4D intra-phoneme progress + 1D frame F0/energy conditioning
        nn.init.zeros_(self.prog_proj.weight); nn.init.zeros_(self.prog_proj.bias)

        blocks = []
        for i in range(nb):
            ff_m = 3.0
            blk = WalshHadamardMonarchBlock(d=d, c=c, k=7, ff_mult=ff_m, had_size=16)
            blk.film = nn.Identity()
            blk.film_const.requires_grad_(True)
            if i >= 7:
                nn.init.zeros_(blk.p2.weight)
                nn.init.zeros_(blk.p2.bias)
                nn.init.zeros_(blk.monarch_g2.weight)
                nn.init.zeros_(blk.monarch_g2.bias)
            blocks.append(blk)
        self.blocks = nn.ModuleList(blocks)
        self.step2_gate = nn.Parameter(torch.tensor(0.14))
        self.step3_gate = nn.Parameter(torch.tensor(0.01))
        self.step4_gate = nn.Parameter(torch.tensor(0.01))
        self.cfg_gate = nn.Parameter(torch.tensor(0.0))
        self.out = nn.Conv1d(d, n_mel, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, mu: torch.Tensor, m: torch.Tensor, prog_5d: torch.Tensor = None, r: torch.Tensor = None) -> torch.Tensor:
        t_mod = self.tm(sinusoidal_temb(t, self.c).to(x.dtype))
        if r is not None:
            t_mod = t_mod + self.rm(sinusoidal_temb(t - r, self.c).to(x.dtype))
        h = self.inp(torch.cat([x, mu], dim=1)) + t_mod[:, :, None]
        if prog_5d is not None:
            h = h + self.prog_proj(prog_5d.to(x.dtype))
        h = h * m
        c_dummy = torch.zeros((x.shape[0], self.c), dtype=x.dtype, device=x.device)
        for blk in self.blocks:
            h = blk(h, c_dummy, m)
        return self.out(h) * m


class AMF0VUVHead100Mel(nn.Module):
    """Continuous F0 (in Hz) + Soft V/UV Predictor operating on full 100-band log-mels + coupled phoneme F0 contour (kept in FP16)."""
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
        lf0 = torch.log2(f0.clamp(115.0, 380.0))
        if f0_bias_st is not None:
            lf0 = lf0 + f0_bias_st.to(lf0.dtype) * (1.0 / 12.0)
        f0_hz = torch.pow(2.0, lf0).clamp(115.0, 380.0)
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
    """Unified Single-Style Studio Nepali News Model (`Sakhiya-News` Leg 2: Coupled 6D Prosody Flow + `CFG = 0` Unrolled 2-Step/4-Step iMF + Dual-Source OLA Noise & Creak Vocoder)."""
    def __init__(self, d_enc: int = 144, L_enc: int = 6, ff_enc: int = 224, d_vel: int = 160, nb_vel: int = 10, voc_ch: int = 320, n_engram: int = 384, r_enc: int = 104, steps: int = 2, cfg_scale: float = 0.0):
        super().__init__()
        self.steps = steps
        self.cfg_scale = 0.0  # Strictly CFG = 0 at inference!
        self.cfg = dict(
            tier="sakhiya_news", arch="sakhiya_news", d_enc=d_enc, L_enc=L_enc, ff_enc=ff_enc,
            d_vel=d_vel, nb_vel=nb_vel, voc_ch=voc_ch, n_engram=n_engram, r_enc=r_enc, steps=steps, cfg_scale=0.0,
        )
        self.enc = SakhiyaNewsEncoder(n_mel=100, d=d_enc, L=L_enc, ff=ff_enc, n_engram=n_engram, r_enc=r_enc)
        # ONE Coupled 4-Step Phoneme-Level Prosody Flow jointly modeling [log_dur, f0_mean, f0_slope, energy, creak_prob, post_word_breath_dur]
        self.coupled_pros_flow = CoupledProsodyFlowHead(d_in=d_enc, out_dim=6, h_dim=96)
        self.vel = SakhiyaNewsIMFDecoder(n_mel=100, d=d_vel, nb=nb_vel, c=d_enc)
        self.f0_head = AMF0VUVHead100Mel(n_mels=100)
        self.voc = StudioVocoder100Mel(n_mels=100, ch=voc_ch, r_k7=None, r_k11=None)

        self.register_buffer("mel_stats", torch.tensor([-6.1761, 2.4932]))
        self.closure_slope = nn.Parameter(torch.tensor(-0.12))   # Grounded downward Kathmandu clause-final F0 closure
        self.dur_rate_bias = nn.Parameter(torch.tensor(0.0))
        self.geminate_stretch = nn.Parameter(torch.tensor(1.06))  # Emphatic Nepali geminate stretching multiplier

    def load_leg1_checkpoint_strict_false(self, rck_sd: dict):
        """Warm-start cleanly from Leg-1 `am_last.pt` via `strict=False`:
        Copies converged RoPE Encoder, Layer-3 Engram, 10-block Monarch iMF Decoder, F0 head, Vocoder, AND
        maps Leg-1 `dur_flow.{i,t_steps,cv,o}` + `pros_flow.base_proj` into `coupled_pros_flow`!
        """
        own = self.state_dict()
        with torch.no_grad():
            for k, v in own.items():
                if k in rck_sd and rck_sd[k].shape == v.shape:
                    v.copy_(rck_sd[k].float())
            # Map Leg-1 dur_flow.{i,t_steps,cv,o} -> coupled_pros_flow.{i,t_steps,cv,o}
            for sub_k in ("i.weight", "i.bias", "t_steps", "o.weight", "o.bias",
                          "cv.0.weight", "cv.0.bias", "cv.1.weight", "cv.1.bias",
                          "cv.2.weight", "cv.2.bias", "cv.3.weight", "cv.3.bias"):
                src_k = "dur_flow." + sub_k
                dst_k = "coupled_pros_flow." + sub_k
                if src_k in rck_sd and dst_k in own and rck_sd[src_k].shape == own[dst_k].shape:
                    own[dst_k].copy_(rck_sd[src_k].float())
            # Map Leg-1 pros_flow.base_proj.0/2 into coupled_pros_flow.base_pros_proj.0/2 (first 144 channels)
            if "pros_flow.base_proj.0.weight" in rck_sd:
                w0 = rck_sd["pros_flow.base_proj.0.weight"].float()
                b0 = rck_sd["pros_flow.base_proj.0.bias"].float()
                r_ch = min(w0.shape[0], own["coupled_pros_flow.base_pros_proj.0.weight"].shape[0])
                own["coupled_pros_flow.base_pros_proj.0.weight"][:r_ch, :144, :].copy_(w0[:r_ch])
                own["coupled_pros_flow.base_pros_proj.0.bias"][:r_ch].copy_(b0[:r_ch])
        self.load_state_dict(own, strict=False)
        return self

    def load_phase1_vocoder(self, voc_ck_path: str):
        dev = next(self.parameters()).device
        ck = torch.load(voc_ck_path, map_location=dev, weights_only=False)
        vsd = {k: v.float() for k, v in ck.get("G_ema", ck.get("G", ck)).items()}
        has_svd = any(".c1.0.0.weight" in k for k in vsd)
        if has_svd and os.path.exists("/work/tts/ck/sakhiya_one_voc.pt"):
            self.voc.init_from_clean_studio_filterbank("/work/tts/ck/sakhiya_one_voc.pt")
            own_v = self.voc.state_dict()
            for k, v in vsd.items():
                if k in own_v and own_v[k].shape == v.shape and not k.startswith("amp_head"):
                    own_v[k].copy_(v)
            self.voc.load_state_dict(own_v, strict=False)
        else:
            self.voc.init_from_clean_studio_filterbank(voc_ck_path)
        if "f0_head" in ck:
            self.f0_head.load_state_dict({k: v.float() for k, v in ck["f0_head"].items()}, strict=False)
        return self

    @torch.no_grad()
    def infer(self, ids: torch.Tensor, lang: torch.Tensor = None, dur_scale: torch.Tensor = None, pitch_shift: torch.Tensor = None) -> torch.Tensor:
        """Fast `CFG = 0` 2-Step / 4-Step X-Pred MeanFlow + Coupled 6D Prosody Flow into Dual-Source OLA Noise & Creak Vocoder."""
        B, L = ids.shape
        dev = ids.device
        im = torch.ones((B, L), dtype=torch.bool, device=dev)
        lg_zero = (lang.to(torch.float32).reshape(B, 1) * 0.0) if lang is not None else torch.zeros((B, 1), dtype=torch.float32, device=dev)
        z_tok = torch.zeros((B, L), dtype=torch.float32, device=dev) + lg_zero
        ds_tok = (dur_scale.reshape(B, -1) + z_tok) if dur_scale is not None else torch.ones((B, L), dtype=torch.float32, device=dev)
        ps_tok = (pitch_shift.reshape(B, -1) + z_tok) if pitch_shift is not None else torch.zeros((B, L), dtype=torch.float32, device=dev)

        struct_5d = compute_graph_structural_features(ids)
        pos = torch.arange(L, device=dev, dtype=torch.float32)[None, :]
        L_f = (pos[:, -1:] + 1.0).clamp(min=2.0)
        is_pause = (ids <= 8).float()
        phase = (pos / L_f).clamp(0.0, 1.0)
        tail_closure = self.closure_slope.clamp(-0.30, -0.05) * ((phase - 0.72).clamp(min=0.0) / 0.28) * (1.0 - is_pause)
        p_arch = (0.14 * torch.sin(phase * 3.14159265) + tail_closure) * (1.0 - is_pause) - 0.15 * is_pause
        p_arch = F.avg_pool1d(F.pad(p_arch[:, None, :], (1, 1), mode="replicate"), 3, stride=1)[:, 0]
        p_enc = (p_arch + 0.95 * ps_tok).clamp(-1.10, 1.10)

        h, mu = self.enc(ids, struct_5d, im, pitch_add=p_enc)

        # Coupled 4-Step Phoneme Prosody Flow: [log_dur, f0_mean, f0_slope, energy, creak_prob, post_word_breath_dur] at CFG = 0
        pros_6d = self.coupled_pros_flow.sample(ids, h, im, steps=4, temp=0.05)  # [B, 6, L]
        x_d = pros_6d[:, 0, :] + self.dur_rate_bias.clamp(-0.12, 0.12)
        f0_ph = pros_6d[:, 1, :].clamp(-0.6, 0.6)
        f0_slope_ph = pros_6d[:, 2, :].clamp(-0.5, 0.5)
        eng_ph = (self.enc.energy_pred(h).squeeze(-1) + pros_6d[:, 3, :]).clamp(-1.2, 1.2)
        creak_ph = pros_6d[:, 4, :].clamp(0.0, 1.0)

        mu = mu + (0.06 * eng_ph)[..., None] * 0.32
        dur_frames = (torch.exp(x_d).clamp(min=1.0, max=60.0) * ds_tok).clamp(min=1.0, max=60.0)

        # Emphatic Nepali geminate stretching & natural 80-180ms inter-clause <breath> preservation
        is_gem = struct_5d[:, :, 4]
        dur_frames = dur_frames * (1.0 + (self.geminate_stretch.clamp(1.03, 1.25) - 1.0) * is_gem)
        is_breath = (ids == BREATH_ID).float()
        dur_frames = torch.where(is_breath > 0.5, torch.clamp(dur_frames, min=9.0, max=16.0), dur_frames)

        is_tail = (((pos >= (L_f - 3.0)) & (ids > 8)) | (ids >= 53)).float()
        min_ph = is_tail * 5.0 + (1.0 - is_tail) * ((ids > 8).float() * 3.0 + (ids <= 8).float() * 1.0)
        dur = torch.round(torch.maximum(dur_frames, min_ph)).clamp(min=1.0, max=60.0)

        # Preserve Nepali reduplication hiatus (`आआफ्नो`) and short function words (`र`, `त`, `छ`, `हो`)
        id_a, id_wb, id_r, id_h, id_schwa, id_o = SYM2ID["a"], SYM2ID["|"], SYM2ID["r"], SYM2ID["h"], SYM2ID["ʌ"], SYM2ID["o"]
        next_ids = torch.cat([ids[:, 1:], torch.zeros_like(ids[:, -1:])], dim=1)
        prev_ids = torch.cat([torch.zeros_like(ids[:, :1]), ids[:, :-1]], dim=1)
        next2_ids = torch.cat([ids[:, 2:], torch.zeros_like(ids[:, -2:])], dim=1)
        prev2_ids = torch.cat([torch.zeros_like(ids[:, :2]), ids[:, :-2]], dim=1)
        next3_ids = torch.cat([ids[:, 3:], torch.zeros_like(ids[:, -3:])], dim=1)
        prev3_ids = torch.cat([torch.zeros_like(ids[:, :3]), ids[:, :-3]], dim=1)
        is_pre_hiatus_wb = ((ids == id_wb) & (next_ids == id_a) & (next2_ids == id_wb) & (next3_ids == id_a)).float()
        is_hiatus_wb = ((prev2_ids == id_wb) & (prev_ids == id_a) & (ids == id_wb) & (next_ids == id_a)).float()
        is_hiatus_a1 = ((prev_ids == id_wb) & (ids == id_a) & (next_ids == id_wb) & (next2_ids == id_a)).float()
        is_hiatus_a2 = ((prev3_ids == id_wb) & (prev2_ids == id_a) & (prev_ids == id_wb) & (ids == id_a)).float()
        dur = torch.where(is_pre_hiatus_wb > 0.5, torch.full_like(dur, 12.0), dur)
        dur = torch.maximum(dur, is_hiatus_a1 * 19.0 + is_hiatus_a2 * 18.0)
        dur = torch.where(is_hiatus_wb > 0.5, torch.full_like(dur, 7.0), dur)
        mu_next2 = torch.cat([mu[:, 2:, :], mu[:, -2:, :]], dim=1)
        mu_next1 = torch.cat([mu[:, 1:, :], mu[:, -1:, :]], dim=1)
        m_pre = is_pre_hiatus_wb[:, :, None]
        m_a1 = is_hiatus_a1[:, :, None]
        m_wb = is_hiatus_wb[:, :, None]
        mu = mu * (1.0 - m_pre) + mu[:, -1:, :] * m_pre
        mu = mu * (1.0 - m_a1) + mu_next2 * m_a1
        mu = mu * (1.0 - m_wb) + (0.45 * mu_next1 + 0.55 * mu[:, -1:, :]) * m_wb
        is_func_c = ((prev_ids <= 6) & (next2_ids <= 6) & ((ids == id_r) | (ids == id_h)) & ((next_ids == id_schwa) | (next_ids == id_o))).float()
        is_func_v = ((prev2_ids <= 6) & (next_ids <= 6) & ((prev_ids == id_r) | (prev_ids == id_h)) & ((ids == id_schwa) | (ids == id_o))).float()
        dur = torch.maximum(dur, is_func_c * 7.0 + is_func_v * 9.0)

        ends = torch.cumsum(dur, dim=1)
        T = ends[0, -1].to(torch.int64)
        t_arange = torch.arange(0, T, device=dev, dtype=torch.float32)
        idx = (t_arange[None, :, None] >= ends[:, None, :]).to(torch.float32).sum(-1).to(torch.int64).clamp(max=L - 1)

        mu_t = mu.transpose(1, 2)
        mu_y = torch.gather(mu_t, 2, idx[:, None, :].expand(-1, mu_t.shape[1], -1))

        # Grounded Kathmandu downward clause-final F0 contour + intra-phoneme F0 slope + phrase-final glottal creak
        dist_danda_tok = struct_5d[:, :, 1]
        clause_tail = ((0.28 - dist_danda_tok).clamp(min=0.0) / 0.28) * (1.0 - is_pause)
        f0_contour_tok = (f0_ph + 0.15 * f0_slope_ph + self.closure_slope.clamp(-0.30, -0.05) * 1.45 * clause_tail) * (1.0 - is_pause)
        f0_frame = torch.gather(f0_contour_tok, 1, idx)
        prog_5d = compute_progress_5d(dur, ends, idx, t_arange, f0_frame)
        m_t = torch.ones((B, 1, T), dtype=mu_y.dtype, device=dev)

        # Phrase-final creak envelope fed to Vocoder 0.5*F0 subharmonic excitation
        creak_tok = (creak_ph + 0.35 * clause_tail).clamp(0.0, 1.0) * (1.0 - is_pause)
        shr_frame = torch.gather(creak_tok, 1, idx)[:, None, :]

        # Strictly CFG = 0 X-Pred MeanFlow Decoding (2-step default; supports 4-step if self.steps == 4)
        t0 = torch.zeros((B,), device=dev, dtype=torch.float32)
        r_one = torch.ones((B,), device=dev, dtype=torch.float32)
        v0 = self.vel(mu_y, t0, mu_y, m_t, prog_5d=prog_5d, r=r_one)

        x_50 = mu_y + 0.50 * v0
        t_50 = torch.full((B,), 0.50, device=dev, dtype=torch.float32)
        v_50 = self.vel(x_50, t_50, mu_y, m_t, prog_5d=prog_5d, r=r_one)

        if self.steps <= 2:
            w2 = self.vel.step2_gate.clamp(0.0, 0.35)
            xt = mu_y + (1.0 - w2) * v0 + w2 * v_50
        else:
            x_25 = mu_y + 0.25 * v0
            t_25 = torch.full((B,), 0.25, device=dev, dtype=torch.float32)
            v_25 = self.vel(x_25, t_25, mu_y, m_t, prog_5d=prog_5d, r=r_one)
            x_75 = mu_y + 0.75 * (0.80 * v0 + 0.10 * v_25 + 0.10 * v_50)
            t_75 = torch.full((B,), 0.75, device=dev, dtype=torch.float32)
            v_75 = self.vel(x_75, t_75, mu_y, m_t, prog_5d=prog_5d, r=r_one)
            w2 = self.vel.step2_gate.clamp(0.0, 0.25)
            w3 = self.vel.step3_gate.clamp(0.0, 0.12)
            w4 = self.vel.step4_gate.clamp(0.0, 0.12)
            w1 = (1.0 - w2 - w3 - w4).clamp(min=0.51)
            xt = mu_y + w1 * v0 + w2 * v_50 + w3 * v_25 + w4 * v_75

        # Full 100-band log-mel (fmin=0, fmax=12000) -> Dual-Source OLA-Noise + Creak Vocoder
        mel_100 = xt * self.mel_stats[1] + self.mel_stats[0]
        f0_hz, vuv = self.f0_head(mel_100, f0_bias_st=f0_frame[:, None, :] * 0.22)
        y, _, _, _, _, _ = self.voc.forward_wav(mel_100, f0_hz, vuv, shr_creak=shr_frame)
        pk = y.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
        return (y * (0.88 / pk)).clamp(-0.98, 0.98)
