"""Sakhiya-One (~6.5-8.5 MB INT8 ONNX) Unified Breakthrough Architecture (`sakhiya/sakhiya_one.py`).

Combines Cactus-Compute-inspired systems discipline with 2026 speech-generation breakthroughs in ONE sweet-spot checkpoint:
1. Laddered Depth / Anytime Decoder (Cactus Whistle-inspired):
   - 8-block acoustic decoder (`nb_vel = 8`) where every training step supervises BOTH the full 8-block Studio exit (2-4 steps)
     AND the 4-block Fast/Eco prefix exit (1 step) sharing the exact same first 4 blocks and shared projection head (`out_head`).
2. Walsh-Hadamard Orthogonal Channel Rotations + Monarch Block-Structured MLPs:
   - Fixed orthogonal Sylvester Walsh-Hadamard block rotations (via standard ONNX Reshape + MatMul/Conv1d ops, zero custom ops)
     paired with Monarch block-structured channel mixing, spreading activation outliers evenly before INT8 quantization.
3. Mid-Layer Phoneme-Context Engram Lookup Table (~132 KB FP16):
   - Deterministic hashed phoneme-trigram + language + Vedic svara engram table (`352 x 192`) injected at Encoder Layer 3
     with gated residual fusion so hard Nepali clusters (`आआफ्नो`, `काठमाडौँ`, `गर्‍यौँ`, `पुर्‍याउन`), English loanwords,
     and Vedic sandhi/visarga articulations have crisp memorized acoustic anchors.
4. X-Pred MeanFlow Acoustic Decoding:
   - Parameterizes average velocity over interval (r, t) via clean-mel X-prediction `x_hat(x_t, r, t | mu_y, g, prog)` with
     JVP / finite-difference consistency regularization so 1-step Fast and 4-step Studio generation are both razor-sharp.
5. Structural Sanskrit / Vedic Supervision (Zero Inference Heuristics):
   - 4D Intra-Phoneme Progress Conditioning `[k/d, sin(pi*k/d), cos(pi*k/d), log(1+d)/4]`
   - Explicit discrete Vedic Svara (`none`, `udatta`, `anudatta`, `svarita`, `pracaya`) & Moraic (`laghu`, `guru`, `pluta`, `pranava_om`) embeddings.
6. Physics-Safe Mixed Precision + Learnable Step-Size Quantization (LSQ) Per-Channel QAT:
   - Continuous heads (embeddings, engram table, duration head, F0/VUV head, harmonic excitation, iSTFT `amp_head`/`pha_head`)
     stay permanently in FP16 while bulky Conv/Linear/Monarch weights use LSQ per-channel INT8 QAT (`STFT_LSD(INT8, FP32) <= 0.30 dB`).
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cfm import EncLayer, DurFlow, STYLE_PROSODY_TARGETS, temb, _dur_to_attn
from .awp_vocoder import AWPVocoder, RMSNorm1d
from .v2_vocoder import SR, HOP, N_FFT, NMEL
from .symbols import N_SYMBOLS, SYM2ID


def _build_sylvester_hadamard(n: int = 16) -> np.ndarray:
    """Build normalized orthogonal Sylvester Walsh-Hadamard matrix H_n (H_n^T @ H_n = I_n)."""
    h = np.array([[1.0]], dtype=np.float32)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return (h / math.sqrt(n)).astype(np.float32)


class WalshHadamardMonarchBlock(nn.Module):
    """Cactus-inspired ConvNeXt + Walsh-Hadamard Orthogonal Rotation + Monarch Block-Structured MLP.
    Uses 100% standard ONNX ops (Conv1d, Reshape, Transpose, Mul, Add, Gelu) -> zero custom ops, zero ConvInteger.
    """
    def __init__(self, d: int = 240, c: int = 192, k: int = 7, ff_mult: int = 3, had_size: int = 16):
        super().__init__()
        assert d % had_size == 0, f"d={d} must be divisible by had_size={had_size}"
        self.d = d
        self.had_size = had_size
        self.n_groups = d // had_size

        self.dw = nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.n = nn.GroupNorm(1, d)
        self.film = nn.Linear(c, 2 * d)
        self.film_const = nn.Parameter(torch.zeros(1, 2 * d, 1), requires_grad=False)

        # Fixed orthogonal Walsh-Hadamard block-wise channel rotation implemented as frozen grouped 1x1 Conv1d
        # (exported as a standard 16x1x1 grouped Conv1d -> tiny 256-float constant, 100% ONNX/WASM native!)
        self.had_rot = nn.Conv1d(d, d, 1, groups=self.n_groups, bias=False)
        H_mat = torch.from_numpy(_build_sylvester_hadamard(had_size))  # [16, 16]
        H_weight = H_mat.repeat(self.n_groups, 1)[:, :, None]         # [d, 16, 1]
        self.had_rot.weight = nn.Parameter(H_weight, requires_grad=False)

        # Full-capacity FFN + Walsh-Hadamard rotated Monarch block-structured residual pathway
        self.ff_mult = ff_mult
        if ff_mult > 0:
            h_ff = int(d * ff_mult)
            self.p1 = nn.Conv1d(d, h_ff, 1)
            self.p2 = nn.Conv1d(h_ff, d, 1)
        else:
            self.p1 = None
            self.p2 = None
        self.monarch_g1 = nn.Conv1d(d, d, 1, groups=self.n_groups, bias=True)
        self.monarch_g2 = nn.Conv1d(d, d, 1, groups=had_size, bias=True)
        self.monarch_scale = nn.Parameter(torch.full((1, d, 1), 0.15))
        nn.init.zeros_(self.monarch_g2.weight)
        nn.init.zeros_(self.monarch_g2.bias)

    def _monarch_forward(self, y_rot: torch.Tensor) -> torch.Tensor:
        B, C, T = y_rot.shape
        g = self.n_groups
        b = self.had_size
        z = F.silu(self.monarch_g1(y_rot))
        # Block-transpose permutation [B, g, b, T] -> [B, b, g, T] -> [B, C, T] mixes across all Hadamard blocks
        z = z.reshape(B, g, b, T).transpose(1, 2).reshape(B, C, T)
        z = self.monarch_g2(z)
        return z * self.monarch_scale.to(z.dtype)

    def forward(self, x: torch.Tensor, c: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        sc, sh = (self.film_const.to(x.dtype) if isinstance(self.film, nn.Identity) else self.film(c)[:, :, None]).chunk(2, 1)
        y = self.n(self.dw(x * m)) * (1.0 + sc) + sh
        # Apply orthogonal Walsh-Hadamard rotation to spread channel activation outliers evenly before Monarch/FFN
        y_rot = self.had_rot(y)
        ff_out = self._monarch_forward(y_rot) if self.p1 is None else (self.p2(F.gelu(self.p1(y))) + self._monarch_forward(y_rot))
        return x + ff_out * m


class MidLayerPhonemeEngram(nn.Module):
    """Mid-Layer Phoneme-Context Engram Lookup Table (~132 KB in FP16) injected at Encoder Layer 3.
    Hashes (prev_id, cur_id, next_id, is_sa, svara_id) deterministically into `n_buckets = 352` embeddings.
    """
    def __init__(self, d: int = 192, n_buckets: int = 352):
        super().__init__()
        self.n_buckets = n_buckets
        self.engram_emb = nn.Embedding(n_buckets, d)
        self.engram_proj = nn.Linear(d, d)
        self.engram_gate = nn.Linear(d, d)
        nn.init.normal_(self.engram_emb.weight, std=0.02)
        nn.init.zeros_(self.engram_proj.weight)
        nn.init.zeros_(self.engram_proj.bias)
        nn.init.constant_(self.engram_gate.bias, -1.0)

    def forward(self, x: torch.Tensor, ids: torch.Tensor, is_sa: torch.Tensor, svara_ids: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        prev_ids = torch.cat([torch.zeros_like(ids[:, :1]), ids[:, :-1]], dim=1)
        next_ids = torch.cat([ids[:, 1:], torch.zeros_like(ids[:, -1:])], dim=1)
        sa_l = (is_sa[:, None] if is_sa.dim() == 1 else is_sa).to(torch.long)
        sv_l = svara_ids.to(torch.long) if svara_ids is not None else torch.zeros_like(ids)
        h_idx = torch.remainder(prev_ids * 73 + ids * 19 + next_ids * 3 + sa_l * 131 + sv_l * 37, self.n_buckets)
        e = self.engram_emb(h_idx)
        gate = torch.sigmoid(self.engram_gate(x))
        return x + (gate * self.engram_proj(e)) * m[:, :, None].to(x.dtype)


class SakhiyaOneEncoder(nn.Module):
    """Disentangled Content + Language + Vedic Svara/Moraic + Mid-Layer Engram Encoder."""
    def __init__(self, n_mel: int = 100, d: int = 192, L: int = 6, n_spk: int = 16, n_lang: int = 8, n_sty: int = 32, ff: int = 256, n_engram: int = 352):
        super().__init__()
        self.emb = nn.Embedding(N_SYMBOLS, d, padding_idx=0)
        self.spk = nn.Embedding(n_spk, d)
        self.lang = nn.Embedding(n_lang, d)
        self.sty = nn.Embedding(n_sty, d)
        self.studio = nn.Embedding(2, d)
        self.svara_emb = nn.Embedding(5, d); nn.init.zeros_(self.svara_emb.weight)
        self.mora_emb = nn.Embedding(4, d); nn.init.zeros_(self.mora_emb.weight)
        self.engram = MidLayerPhonemeEngram(d=d, n_buckets=n_engram)
        self.pitch = nn.Linear(1, d); nn.init.zeros_(self.pitch.weight); nn.init.zeros_(self.pitch.bias)
        self.prosody = nn.Linear(4, d); nn.init.zeros_(self.prosody.weight); nn.init.zeros_(self.prosody.bias)
        self.f0_pred = nn.Linear(d, 1); nn.init.zeros_(self.f0_pred.weight); nn.init.zeros_(self.f0_pred.bias)
        self.vuv_pred = nn.Linear(d, 1); nn.init.zeros_(self.vuv_pred.weight); nn.init.constant_(self.vuv_pred.bias, 2.0)
        self.energy_pred = nn.Linear(d, 1); nn.init.zeros_(self.energy_pred.weight); nn.init.zeros_(self.energy_pred.bias)
        self.layers = nn.ModuleList(EncLayer(d, ff=ff) for _ in range(L))
        self.nf = nn.LayerNorm(d)
        self.mu = nn.Linear(d, n_mel)

    def forward(self, ids, spk, lang, sty, studio, m, pitch_add=None, prosody=None, sty_mask=None, svara_ids=None, mora_ids=None):
        lang0 = torch.zeros_like(lang)
        lang_eff = torch.where(lang == 4, lang, lang0)
        sty0 = sty * 0
        g_base = self.spk(spk) + self.lang(lang0) + self.studio(studio)
        g_sty_delta = self.sty(sty) - self.sty(sty0)
        if prosody is not None:
            g_sty_delta = g_sty_delta + self.prosody(prosody.to(g_sty_delta.dtype))
        if sty_mask is not None:
            g_sty_delta = g_sty_delta * sty_mask.to(g_sty_delta.dtype)
        g_sty_full = self.sty(sty0) + 0.24 * g_sty_delta
        g_vel = g_base + self.sty(sty0)
        g_enc = g_base + g_sty_full + 0.18 * (self.lang(lang_eff) - self.lang(lang0))
        x = self.emb(ids) * math.sqrt(self.emb.embedding_dim) + g_enc[:, None]
        if svara_ids is not None:
            x = x + self.svara_emb(svara_ids.clamp(0, 4))
        if mora_ids is not None:
            x = x + self.mora_emb(mora_ids.clamp(0, 3))
        if pitch_add is not None:
            x = x + self.pitch(pitch_add.to(x.dtype)[..., None])
        is_sa = (lang == 4) | (lang == 1)
        for idx_l, lyr in enumerate(self.layers):
            x = lyr(x, m)
            if idx_l == 2:  # Inject Mid-Layer Phoneme-Context Engram at Encoder Layer 3
                x = self.engram(x, ids, is_sa, svara_ids, m)
        h_content = self.nf(x)
        h = h_content + 0.35 * g_sty_full[:, None]
        return h, self.mu(h), g_vel


class SakhiyaOneMeanFlowDecoder(nn.Module):
    """Laddered 8-Block X-Pred MeanFlow Decoder with Walsh-Hadamard Rotations & Monarch MLPs.
    Supports nested laddered depth (`n_blocks=8` Studio mode and `n_blocks=4` Fast/Eco mode from the exact same weights).
    """
    def __init__(self, n_mel: int = 100, d: int = 240, nb: int = 8, c: int = 192):
        super().__init__()
        self.d = d
        self.nb = nb
        self.c = c
        self.inp = nn.Conv1d(2 * n_mel, d, 1)
        self.tm = nn.Sequential(nn.Linear(c, c), nn.SiLU(), nn.Linear(c, c))
        self.rm = nn.Sequential(nn.Linear(c, c), nn.SiLU(), nn.Linear(c, c))
        nn.init.zeros_(self.rm[2].weight); nn.init.zeros_(self.rm[2].bias)
        self.prog_proj = nn.Conv1d(4, d, 1)
        nn.init.zeros_(self.prog_proj.weight); nn.init.zeros_(self.prog_proj.bias)
        self.blocks = nn.ModuleList(
            WalshHadamardMonarchBlock(d=d, c=c, ff_mult=(0 if i >= 7 else 3))
            for i in range(nb)
        )
        self.fast_adapter = nn.Sequential(
            nn.Conv1d(d, d, 5, padding=2, groups=d),
            nn.GroupNorm(1, d),
            nn.GELU(),
            nn.Conv1d(d, d, 1),
        )
        nn.init.zeros_(self.fast_adapter[3].weight)
        nn.init.zeros_(self.fast_adapter[3].bias)
        self.step2_gate = nn.Parameter(torch.tensor(0.08))
        self.out = nn.Conv1d(d, n_mel, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, mu: torch.Tensor, g: torch.Tensor, m: torch.Tensor, prog: torch.Tensor = None, r: torch.Tensor = None, n_blocks: int = None) -> torch.Tensor:
        nb_run = self.nb if n_blocks is None else min(int(n_blocks), len(self.blocks))
        if isinstance(self.tm, nn.Identity):
            c = g
        else:
            c = self.tm(temb(t, self.c).to(x.dtype)) + g
            if r is not None and not isinstance(self.rm, nn.Identity):
                c = c + self.rm(temb(t - r, self.c).to(x.dtype))
        h = self.inp(torch.cat([x, mu], 1))
        if prog is not None:
            h = h + self.prog_proj(prog.to(x.dtype))
        h = h * m
        for i in range(nb_run):
            h = self.blocks[i](h, c, m)
        if nb_run <= 4:
            h = (h + self.fast_adapter(h * m)) * m
        return self.out(h) * m

    def forward_laddered(self, x: torch.Tensor, t: torch.Tensor, mu: torch.Tensor, g: torch.Tensor, m: torch.Tensor, prog: torch.Tensor = None, r: torch.Tensor = None):
        """Simultaneously compute both the 4-block Fast exit AND the 8-block Studio exit in a single forward pass."""
        if isinstance(self.tm, nn.Identity):
            c = g
        else:
            c = self.tm(temb(t, self.c).to(x.dtype)) + g
            if r is not None and not isinstance(self.rm, nn.Identity):
                c = c + self.rm(temb(t - r, self.c).to(x.dtype))
        h = self.inp(torch.cat([x, mu], 1))
        if prog is not None:
            h = h + self.prog_proj(prog.to(x.dtype))
        h = h * m
        for i in range(min(4, len(self.blocks))):
            h = self.blocks[i](h, c, m)
        h_4 = h
        h_4_bridged = (h_4 + self.fast_adapter(h_4 * m)) * m
        out_4 = self.out(h_4_bridged) * m
        for i in range(4, len(self.blocks)):
            h = self.blocks[i](h, c, m)
        h_8 = h
        out_8 = self.out(h_8) * m
        return out_8, out_4, h_8, h_4_bridged

    def xpred_from_vel(self, x_t: torch.Tensor, t: torch.Tensor, v_pred: torch.Tensor) -> torch.Tensor:
        """Convert velocity parameterization into clean-mel X-prediction x_hat = x_t + (1 - t) * v_pred."""
        t_b = t.view(-1, 1, 1).to(x_t.dtype)
        return x_t + (1.0 - t_b) * v_pred


class LSQQuantizer(torch.autograd.Function):
    """Learnable Step-Size Quantization (LSQ) per-output-channel INT8 weight quantizer with gradient scaling."""
    @staticmethod
    def forward(ctx, weight: torch.Tensor, step_size: torch.Tensor, g_scale: float):
        ctx.save_for_backward(weight, step_size)
        ctx.g_scale = g_scale
        s = step_size.abs().clamp(min=1e-7)
        w_q = torch.clamp(torch.round(weight / s), -127.0, 127.0)
        return w_q * s

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        weight, step_size = ctx.saved_tensors
        g_scale = ctx.g_scale
        s = step_size.abs().clamp(min=1e-7)
        w_div = weight / s
        w_round = torch.round(w_div)
        mask_in = (w_div >= -127.0) & (w_div <= 127.0)
        grad_w = torch.where(mask_in, grad_output, torch.zeros_like(grad_output))
        grad_s_elem = torch.where(mask_in, w_round - w_div, torch.clamp(w_round, -127.0, 127.0)) * grad_output
        red_dims = tuple(i for i in range(weight.ndim) if i != 0)
        grad_s = grad_s_elem.sum(dim=red_dims, keepdim=True) * g_scale
        return grad_w, grad_s, None


def apply_lsq_int8_weight(weight: torch.Tensor, step_size: torch.Tensor = None) -> torch.Tensor:
    wf = weight.float()
    red_dims = tuple(i for i in range(wf.ndim) if i != 0)
    if step_size is None:
        s = wf.abs().amax(dim=red_dims, keepdim=True).clamp(min=1e-8) / 127.0
        q = torch.clamp(torch.round(wf / s), -127.0, 127.0)
        return (q * s).to(weight.dtype)
    g_scale = 1.0 / math.sqrt(max(1.0, wf.numel() * 127.0))
    return LSQQuantizer.apply(wf, step_size.float(), g_scale).to(weight.dtype)


class SakhiyaOneAM(nn.Module):
    """Sakhiya-One Acoustic Model combining SakhiyaOneEncoder, DurFlow, and Laddered X-Pred MeanFlow Decoder."""
    def __init__(self, d_enc: int = 192, L_enc: int = 6, ff_enc: int = 256, d_vel: int = 240, nb_vel: int = 8, h_dur: int = 128, n_engram: int = 352):
        super().__init__()
        self.enc = SakhiyaOneEncoder(d=d_enc, L=L_enc, ff=ff_enc, n_engram=n_engram)
        self.vel = SakhiyaOneMeanFlowDecoder(d=d_vel, nb=nb_vel, c=d_enc)
        self.dur = DurFlow(d=d_enc, h=h_dur)
        self.register_buffer("stats", torch.tensor([-6.1761, 2.4932]))

    @torch.no_grad()
    def fold_time_embeddings(self, steps: int = 4):
        dev = next(self.parameters()).device
        g0 = self.enc.spk.weight[0:1] + self.enc.lang.weight[0:1] + self.enc.studio.weight[1:2] + self.enc.sty.weight[0:1]
        if not isinstance(self.vel.tm, nn.Identity):
            tb0 = temb(torch.zeros((1,), device=dev), self.vel.c)
            c0 = self.vel.tm(tb0) + g0
            for b in self.vel.blocks:
                if not isinstance(b.film, nn.Identity):
                    if b.film_const.abs().max().item() < 1e-6:
                        b.film_const.data.copy_(b.film(c0)[:, :, None])
                    b.film = nn.Identity()
                b.film_const.requires_grad_(True)
            self.vel.tm = nn.Identity()
            self.vel.rm = nn.Identity()
        else:
            for b in self.vel.blocks:
                b.film = nn.Identity()
                b.film_const.requires_grad_(True)
            self.vel.rm = nn.Identity()
        if not isinstance(self.dur.t, nn.Identity):
            ts = torch.tensor([0.0, 0.25, 0.50, 0.75], device=dev)
            if self.dur.t_steps.abs().max().item() < 1e-6:
                self.dur.t_steps.data.copy_(self.dur.t(temb(ts, self.dur.d))[:, :, None])
            self.dur.t = nn.Identity()
            self.dur.t_steps.requires_grad_(True)
        return self

    def load_pro_v3_am_weights(self, ck_path: str):
        """Initialize matching Pro-v3 weights into SakhiyaOneAM (never V1/v5/v9)."""
        dev = next(self.parameters()).device
        ck = torch.load(ck_path, map_location=dev, weights_only=False)
        sd = {k: v.float() for k, v in ck.get("ema", ck).items()}
        own = self.state_dict()
        with torch.no_grad():
            for k, v in own.items():
                src_k = k if k in sd else ("am." + k if ("am." + k) in sd else None)
                if src_k is not None:
                    src_t = sd[src_k]
                    if src_t.shape == v.shape:
                        v.copy_(src_t)
                    elif src_t.ndim == v.ndim:
                        sl = tuple(slice(0, min(a, b)) for a, b in zip(v.shape, src_t.shape))
                        v[sl].copy_(src_t[sl])
        self.load_state_dict(own, strict=False)
        return self
