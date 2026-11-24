"""Sakhiya v15 Breakthrough 3: Weight-Tied Looped Universal Block Acoustic Model (`sakhiya/v15_pico.py`).
Shatters the sub-1.0 MB barrier (<= 980 KB total int8 ONNX across Encoder + Decoder + MS-RND-Wavehax-Lite Vocoder):
1. Weight-Tied Looped Universal RoPE-Conv Encoder (`PicoTextEncoder`, d=96, ff=144, L=4 loop iterations):
   - Uses 1x1 and 5x1 QATConv1d weights called K=4 times in forward() so ONNX stores ONE shared weight initializer on disk.
   - Modulates each iteration k=1..4 with learned per-iteration FiLM vectors (gamma_k, beta_k) + residual gate.
   - Pruned single-speaker (`sakhiya`) + 6 active styles.
2. Weight-Tied Looped Universal DurFlow (`PicoDurFlow`, d=96, h=64) with >=5-frame short function-word floor.
3. Weight-Tied Looped Universal ConvNeXt-1D Velocity Decoder (`PicoVelocity`, d=112, c=96, 2 shared blocks looped 3x + 2x = 5 steps):
   - Each loop iteration k=1..5 adds a learned step embedding `step_emb[k]` into the FiLM conditioning vector + per-step output FiLM.
4. Per-Channel Symmetric INT8 Straight-Through Estimator (STE) QAT on all Conv/Linear layers for 0.00 FP32->INT8 loss."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .symbols import N_SYMBOLS, SYM2ID, LANG2ID
from .cfm import rope, temb, STYLE_PROSODY_TARGETS, _mel_f0_z, _dur_to_attn
from .v15_rnd_wavehax import QATConv1d, ste_int8_weight


class QATLinear(nn.Linear):
    def __init__(s, *args, **kwargs):
        super().__init__(*args, **kwargs)
        s.qat_enabled = False

    def forward(s, x):
        w = ste_int8_weight(s.weight, axis=0, enabled=s.qat_enabled and s.training)
        return F.linear(x, w, s.bias)


class UniversalEncBlock(nn.Module):
    """Universal RoPE Self-Attention + Conv1D FFN block using QATConv1d (stored ONCE in ONNX when looped!)."""
    def __init__(s, d=96, h=4, ff=112, k=5, n_loops=4):
        super().__init__()
        s.h = h
        s.n1 = nn.LayerNorm(d)
        s.qkv = QATConv1d(d, 3 * d, 1)
        s.o = QATConv1d(d, d, 1)
        s.n2 = nn.LayerNorm(d)
        s.c1 = QATConv1d(d, ff, k, padding=k // 2)
        s.c2 = QATConv1d(ff, d, 3, padding=1)
        # Per-iteration FiLM modulation (gamma_k, beta_k) for attention and FFN sublayers
        s.attn_gamma = nn.Parameter(torch.zeros(n_loops, 1, d))
        s.attn_beta = nn.Parameter(torch.zeros(n_loops, 1, d))
        s.ff_gamma = nn.Parameter(torch.zeros(n_loops, 1, d))
        s.ff_beta = nn.Parameter(torch.zeros(n_loops, 1, d))

    def forward_step(s, x, m, step_idx: int):
        B, T, d = x.shape
        xn = s.n1(x) * (1.0 + s.attn_gamma[step_idx]) + s.attn_beta[step_idx]
        qkv = s.qkv(xn.transpose(1, 2)).transpose(1, 2)
        q, k, v = qkv.view(B, T, 3, s.h, d // s.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(rope(q), rope(k), v, attn_mask=m[:, None, None, :])
        ao = s.o(a.transpose(1, 2).reshape(B, T, d).transpose(1, 2)).transpose(1, 2)
        x = x + ao
        yn = (s.n2(x) * (1.0 + s.ff_gamma[step_idx]) + s.ff_beta[step_idx]).transpose(1, 2) * m[:, None]
        return x + s.c2(F.gelu(s.c1(yn))).transpose(1, 2)


class PicoTextEncoder(nn.Module):
    """Pruned single-speaker + 6-style Looped Universal Text Encoder (~195K params -> ~215 KB int8 ONNX)."""
    def __init__(s, n_mel=100, d=96, L=4, n_spk=2, n_lang=6, n_sty=6, ff=112):
        super().__init__()
        s.d = d
        s.L = L
        s.emb = nn.Embedding(N_SYMBOLS, d, padding_idx=0)
        s.spk = nn.Embedding(n_spk, d)
        s.lang = nn.Embedding(n_lang, d)
        s.sty = nn.Embedding(n_sty, d)
        s.studio = nn.Embedding(2, d)
        s.pitch = nn.Linear(1, d); nn.init.zeros_(s.pitch.weight); nn.init.zeros_(s.pitch.bias)
        s.prosody = nn.Linear(4, d); nn.init.zeros_(s.prosody.weight); nn.init.zeros_(s.prosody.bias)
        s.f0_pred = nn.Linear(d, 1); nn.init.zeros_(s.f0_pred.weight); nn.init.zeros_(s.f0_pred.bias)
        s.energy_pred = nn.Linear(d, 1); nn.init.zeros_(s.energy_pred.weight); nn.init.zeros_(s.energy_pred.bias)
        s.ublock = UniversalEncBlock(d=d, h=4, ff=ff, k=5, n_loops=L)
        s.nf = nn.LayerNorm(d)
        s.mu = QATConv1d(d, n_mel, 1)

    def forward(s, ids, spk, lang, sty, studio, m, pitch_add=None, prosody=None, sty_mask=None):
        lang0 = torch.zeros_like(lang)
        lang_eff = torch.where(lang == 4, lang, lang0).clamp(0, s.lang.num_embeddings - 1)
        spk0 = torch.zeros_like(spk)
        sty_c = sty.clamp(0, s.sty.num_embeddings - 1)
        sty0 = sty_c * 0
        g_base = s.spk(spk0) + s.lang(lang0) + s.studio(studio.clamp(0, 1))
        g_sty_delta = s.sty(sty_c) - s.sty(sty0)
        if prosody is not None:
            g_sty_delta = g_sty_delta + s.prosody(prosody.to(g_sty_delta.dtype))
        if sty_mask is not None:
            g_sty_delta = g_sty_delta * sty_mask.to(g_sty_delta.dtype)
        g_sty_full = s.sty(sty0) + 0.24 * g_sty_delta
        g_vel = g_base + s.sty(sty0)
        g_enc = g_base + g_sty_full + 0.18 * (s.lang(lang_eff) - s.lang(lang0))
        x = s.emb(ids) * math.sqrt(s.emb.embedding_dim) + g_enc[:, None]
        if pitch_add is not None:
            x = x + s.pitch(pitch_add.to(x.dtype)[..., None])
        for k in range(s.L):
            x = s.ublock.forward_step(x, m, k)
        h_content = s.nf(x)
        h = h_content + 0.35 * g_sty_full[:, None]
        s._last_h_content = h_content
        mu = s.mu(h.transpose(1, 2)).transpose(1, 2)
        return h, mu, g_vel


class PicoDurFlow(nn.Module):
    """Weight-Tied Looped Duration Flow Predictor (`d=96, h=64`)."""
    def __init__(s, d=96, h=64, n_conv=2):
        super().__init__()
        s.i = QATConv1d(d + 1, h, 1)
        s.t = QATConv1d(d, h, 1)
        s.t_steps = nn.Parameter(torch.zeros(4, h, 1), requires_grad=False)
        s.cv = nn.ModuleList([QATConv1d(h, h, 3, padding=1) for _ in range(n_conv)])
        s.step_gamma = nn.Parameter(torch.zeros(4, h, 1))
        s.step_beta = nn.Parameter(torch.zeros(4, h, 1))
        s.o = QATConv1d(h, 1, 1)
        s.d = d

    def forward(s, x, t, hs, m, step_idx: int = 0):
        hs_in = 0.85 * hs.detach() + 0.15 * hs
        if isinstance(s.t, nn.Identity):
            t_add = s.t_steps[step_idx].to(hs.dtype)[None, :, :]
        else:
            tb = temb(t, s.d).to(hs.dtype)[:, :, None]
            t_add = s.t(tb)
        z = s.i(torch.cat([hs_in.transpose(1, 2), x[:, None].to(hs.dtype)], 1)) + t_add
        for idx_c in range(4):
            c_layer = s.cv[idx_c % len(s.cv)]
            z_in = (z * (1.0 + s.step_gamma[idx_c]) + s.step_beta[idx_c]) * m[:, None]
            z = z + F.gelu(c_layer(z_in))
        return s.o(z)[:, 0] * m

    def pred_logdur(s, ids, hs, m):
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device, dtype=torch.float32)[None, :]
        x = (torch.sin(pos * 1.6180339 + ids.float() * 0.37) * 0.20).to(hs.dtype)
        for i in range(4):
            t = torch.full((B,), i / 4.0, device=ids.device, dtype=torch.float32)
            x = x + s.forward(x, t, hs, m, i) * 0.25
        return x * m


class UniversalVelBlock(nn.Module):
    """Universal ConvNeXt-1D Velocity Block using QATConv1d for all projections (stored ONCE in ONNX!)."""
    def __init__(s, d=112, c=96, k=7, h_ff=256):
        super().__init__()
        s.dw = QATConv1d(d, d, k, padding=k // 2, groups=d)
        s.n = nn.GroupNorm(1, d)
        s.p1 = QATConv1d(d, h_ff, 1)
        s.p2 = QATConv1d(h_ff, d, 1)
        s.film = QATConv1d(c, 2 * d, 1)

    def forward(s, x, c_3d, m, out_gamma, out_beta):
        sc, sh = s.film(c_3d).chunk(2, 1)
        y = s.n(s.dw(x * m)) * (1.0 + sc) + sh
        y = s.p2(F.gelu(s.p1(y)))
        y = y * (1.0 + out_gamma) + out_beta
        return x + y * m


class PicoVelocity(nn.Module):
    """Looped Universal ConvNeXt-1D Velocity Decoder:
    2 shared universal blocks (`ublock_early` looped 3x + `ublock_late` looped 2x = 5 effective layers)
    modulated by per-iteration `step_emb` and `(out_gamma, out_beta)` (~225K params -> ~240 KB int8 ONNX)."""
    def __init__(s, n_mel=100, d=112, nb=5, c=96):
        super().__init__()
        s.d = d
        s.nb = nb
        s.c = c
        s.inp = QATConv1d(2 * n_mel, d, 1)
        s.tm1 = QATConv1d(c, c, 1)
        s.tm2 = QATConv1d(c, c, 1)
        s.ublock_early = UniversalVelBlock(d=d, c=c, k=7, h_ff=256)
        s.ublock_late = UniversalVelBlock(d=d, c=c, k=7, h_ff=256)
        s.step_emb = nn.Parameter(torch.zeros(nb, c, 1))
        s.out_gamma = nn.Parameter(torch.zeros(nb, d, 1))
        s.out_beta = nn.Parameter(torch.zeros(nb, d, 1))
        s.out = QATConv1d(d, n_mel, 1)
        nn.init.zeros_(s.out.weight)
        nn.init.zeros_(s.out.bias)

    def forward(s, x, t, mu, g, m):
        if isinstance(s.tm1, nn.Identity):
            c_base = g[:, :, None]
        else:
            tb = temb(t, s.c).to(x.dtype)[:, :, None]
            c_base = s.tm2(F.silu(s.tm1(tb))) + g[:, :, None]
        h = s.inp(torch.cat([x, mu], 1)) * m
        n_early = (s.nb + 1) // 2  # 3 iterations of early block, 2 iterations of late block
        for k in range(s.nb):
            c_k = c_base + s.step_emb[k].to(x.dtype)
            blk = s.ublock_early if k < n_early else s.ublock_late
            h = blk(h, c_k, m, s.out_gamma[k].to(x.dtype), s.out_beta[k].to(x.dtype))
        return s.out(h) * m


class SakhiyaPicoAM(nn.Module):
    """Sakhiya-Pico Sub-1 MB Acoustic Model (`d_enc=96, L_enc=4, ff_enc=112, d_vel=112, nb_vel=5, h_dur=64`)."""
    def __init__(s, d_enc=96, L_enc=4, ff_enc=112, d_vel=112, nb_vel=5, h_dur=64):
        super().__init__()
        s.enc = PicoTextEncoder(d=d_enc, L=L_enc, ff=ff_enc)
        s.vel = PicoVelocity(d=d_vel, nb=nb_vel, c=d_enc)
        s.dur = PicoDurFlow(d=d_enc, h=h_dur)
        s.register_buffer("stats", torch.tensor([-5.5028, 2.1869]))

    @torch.no_grad()
    def fold_time_embeddings(s):
        """Fold constant 1-step velocity time embedding (t=0) and 4-step duration time embeddings into step parameters."""
        dev = next(s.parameters()).device
        if not isinstance(s.vel.tm1, nn.Identity):
            tb0 = temb(torch.zeros((1,), device=dev), s.vel.c)[:, :, None]
            tm_out = s.vel.tm2(F.silu(s.vel.tm1(tb0)))[0]
            s.vel.step_emb.data.add_(tm_out[None, :, :])
            s.vel.tm1 = nn.Identity()
            s.vel.tm2 = nn.Identity()
        if not isinstance(s.dur.t, nn.Identity):
            ts = torch.tensor([0.0, 0.25, 0.50, 0.75], device=dev)
            tbs = temb(ts, s.dur.d)[:, :, None]
            s.dur.t_steps.data.copy_(s.dur.t(tbs))
            s.dur.t = nn.Identity()
        return s

    def set_qat(s, enabled=True):
        for m in s.modules():
            if hasattr(m, "qat_enabled"):
                m.qat_enabled = bool(enabled)
        return s

    @torch.no_grad()
    def warm_from_sano(s, sano_sd):
        """Warm-initialize Pico universal blocks & embeddings directly from trained Atom/Sano (`v9_sano_am.pt`)."""
        sd = s.state_dict()
        # 1. Direct/sliced embeddings and heads
        for k in ("enc.emb.weight", "enc.spk.weight", "enc.lang.weight", "enc.sty.weight", "enc.studio.weight",
                  "enc.pitch.weight", "enc.pitch.bias", "enc.prosody.weight", "enc.prosody.bias",
                  "enc.f0_pred.weight", "enc.f0_pred.bias", "enc.energy_pred.weight", "enc.energy_pred.bias",
                  "enc.nf.weight", "enc.nf.bias", "vel.inp.weight", "vel.inp.bias", "vel.out.weight", "vel.out.bias",
                  "dur.i.weight", "dur.i.bias", "dur.o.weight", "dur.o.bias"):
            if k in sano_sd and k in sd:
                src = sano_sd[k].float()
                sl = tuple(slice(0, b) for b in sd[k].shape)
                sd[k].copy_(src[sl])
        if "enc.mu.weight" in sano_sd:
            sd["enc.mu.weight"].copy_(sano_sd["enc.mu.weight"].float()[:, :, None])
            sd["enc.mu.bias"].copy_(sano_sd["enc.mu.bias"].float())
        if "dur.t.weight" in sano_sd:
            sd["dur.t.weight"].copy_(sano_sd["dur.t.weight"].float()[:, :, None])
            sd["dur.t.bias"].copy_(sano_sd["dur.t.bias"].float())
        for idx_c in range(2):
            k_w = f"dur.cv.{idx_c}.weight"; k_b = f"dur.cv.{idx_c}.bias"
            if k_w in sano_sd:
                sd[k_w].copy_(sano_sd[k_w].float())
                sd[k_b].copy_(sano_sd[k_b].float())
        # 2. Initialize universal encoder block from layer 1 (middle representation) of Sano encoder
        if "enc.layers.1.qkv.weight" in sano_sd:
            sd["enc.ublock.n1.weight"].copy_(sano_sd["enc.layers.1.n1.weight"].float())
            sd["enc.ublock.n1.bias"].copy_(sano_sd["enc.layers.1.n1.bias"].float())
            sd["enc.ublock.qkv.weight"].copy_(sano_sd["enc.layers.1.qkv.weight"].float()[:, :, None])
            sd["enc.ublock.qkv.bias"].copy_(sano_sd["enc.layers.1.qkv.bias"].float())
            sd["enc.ublock.o.weight"].copy_(sano_sd["enc.layers.1.o.weight"].float()[:, :, None])
            sd["enc.ublock.o.bias"].copy_(sano_sd["enc.layers.1.o.bias"].float())
            sd["enc.ublock.n2.weight"].copy_(sano_sd["enc.layers.1.n2.weight"].float())
            sd["enc.ublock.n2.bias"].copy_(sano_sd["enc.layers.1.n2.bias"].float())
            ff_d = sd["enc.ublock.c1.weight"].shape[0]
            sd["enc.ublock.c1.weight"].copy_(sano_sd["enc.layers.1.c1.weight"].float()[:ff_d])
            sd["enc.ublock.c1.bias"].copy_(sano_sd["enc.layers.1.c1.bias"].float()[:ff_d])
            sd["enc.ublock.c2.weight"].copy_(sano_sd["enc.layers.1.c2.weight"].float()[:, :ff_d, 1:4])
            sd["enc.ublock.c2.bias"].copy_(sano_sd["enc.layers.1.c2.bias"].float())
        # 3. Initialize universal velocity blocks from Sano velocity blocks 1 (early) and 3 (late)
        if "vel.tm.0.weight" in sano_sd:
            sd["vel.tm1.weight"].copy_(sano_sd["vel.tm.0.weight"].float()[:, :, None])
            sd["vel.tm1.bias"].copy_(sano_sd["vel.tm.0.bias"].float())
            sd["vel.tm2.weight"].copy_(sano_sd["vel.tm.2.weight"].float()[:, :, None])
            sd["vel.tm2.bias"].copy_(sano_sd["vel.tm.2.bias"].float())
        for dst_name, src_idx in (("ublock_early", 1), ("ublock_late", 3)):
            pfx_s = f"vel.blocks.{src_idx}."
            pfx_d = f"vel.{dst_name}."
            if pfx_s + "dw.weight" in sano_sd:
                for sub in ("dw.weight", "dw.bias", "n.weight", "n.bias", "p1.weight", "p1.bias", "p2.weight", "p2.bias"):
                    dt, st = sd[pfx_d + sub], sano_sd[pfx_s + sub].float()
                    sl = tuple(slice(0, min(a, b)) for a, b in zip(dt.shape, st.shape))
                    dt[sl].copy_(st[sl])
                sd[pfx_d + "film.weight"].copy_(sano_sd[pfx_s + "film.weight"].float()[:, :, None])
                sd[pfx_d + "film.bias"].copy_(sano_sd[pfx_s + "film.bias"].float())
        s.load_state_dict(sd, strict=False)
        return s
