"""Sakhiya-v2-AM: compact Matcha-style conditional-flow-matching acoustic model (trained from scratch, no v1/Edge/Alpa weights).
RoPE text encoder (prior mu) + flow-matching (stochastic) duration predictor + ConvNeXt-1D OT-CFM velocity decoder.
24 kHz / 100-band log-mel / hop 256.  `pitch_add` [B,L] is the Vedic-svara / prosody hook (added to the encoder states)."""
import math, torch, torch.nn as nn, torch.nn.functional as F
from .symbols import N_SYMBOLS


# v2 was trained with lang in {ne, sa} only: the Tharu rows (ids 1-3) of lang_emb are untrained noise -> alias every Tharu variety to the trained
# Nepali row (Tharu G2P phonology + speaking-rate calibration are kept by the front-end).  Wave-42 (v1) keeps symbols.LANG2ID untouched.
from .symbols import LANG2ID as _L
V2_LANG2ID = {**_L, **{k: _L["ne"] for k in ("thr", "thr_dangaura", "thr_rana", "thr_chitwania", "mai", "bho", "new", "taj")}}


def rope(x, base=10000.0):                                   # x [B,H,T,D]
    T, D = x.shape[2], x.shape[3]; h = D // 2
    f = 1.0 / (base ** (torch.arange(h, device=x.device, dtype=torch.float32) / h))
    a = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * f[None]
    c, s = a.cos().to(x.dtype), a.sin().to(x.dtype); x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], -1)


class EncLayer(nn.Module):
    def __init__(s, d=192, h=4, ff=256, k=5):
        super().__init__(); s.h = h; s.n1 = nn.LayerNorm(d); s.qkv = nn.Linear(d, 3 * d); s.o = nn.Linear(d, d)
        s.n2 = nn.LayerNorm(d); s.c1 = nn.Conv1d(d, ff, k, padding=k // 2); s.c2 = nn.Conv1d(ff, d, k, padding=k // 2)

    def forward(s, x, m):                                    # x [B,T,d], m [B,T] bool
        B, T, d = x.shape
        q, k, v = s.qkv(s.n1(x)).view(B, T, 3, s.h, d // s.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(rope(q), rope(k), v, attn_mask=m[:, None, None, :])
        x = x + s.o(a.transpose(1, 2).reshape(B, T, d))
        y = s.n2(x).transpose(1, 2) * m[:, None]
        return x + s.c2(F.gelu(s.c1(y))).transpose(1, 2)


STYLE_PROSODY_TARGETS = [
    [0.00,  0.00,  0.00,  0.00],   # 0: news / neutral (razor-clean broadcast baseline)
    [0.30,  0.80,  0.15, -0.30],   # 1: conversational (brisk, lively pitch variation)
    [-0.60, 1.20,  0.25,  0.80],   # 2: storytelling (expressive F0 arch, measured pace, narrative pauses)
    [-0.50, -0.15, -0.60, 0.40],   # 3: warm_soft (gentle, lower energy std, unhurried)
    [0.25,  1.35,  0.55, -0.20],   # 4: expressive (high pitch & energy dynamics)
    [-0.65, 0.65,  0.25,  0.85],   # 5: vedic_chant (resonant shloka/svara cadence)
]


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lam * grad_output, None


def grad_reverse(x, lam=0.20):
    return _GRL.apply(x, lam)


class TextEncoder(nn.Module):
    """Sakhiya-v4 Disentangled Content + Language + Fine-Grained Prosody Encoder.
    - Content/Language encoder is regularized with a GRL Speaker-Adversarial Classifier (`spk_adv`) so `lang` carries zero speaker timbre.
    - `g_vel` (passed to Velocity FiLM) depends ONLY on canonical speaker embedding `s.spk(spk)` (zero `g_sty` or `lang` timbre leakage!)."""
    def __init__(s, n_mel=100, d=192, L=6, n_spk=16, n_lang=8, n_sty=32, ff=256):
        super().__init__(); s.emb = nn.Embedding(N_SYMBOLS, d, padding_idx=0)
        s.spk = nn.Embedding(n_spk, d); s.lang = nn.Embedding(n_lang, d); s.sty = nn.Embedding(n_sty, d); s.studio = nn.Embedding(2, d)
        s.pitch = nn.Linear(1, d); nn.init.zeros_(s.pitch.weight); nn.init.zeros_(s.pitch.bias)
        s.prosody = nn.Linear(4, d); nn.init.zeros_(s.prosody.weight); nn.init.zeros_(s.prosody.bias)
        s.f0_pred = nn.Linear(d, 1); nn.init.zeros_(s.f0_pred.weight); nn.init.zeros_(s.f0_pred.bias)
        s.energy_pred = nn.Linear(d, 1); nn.init.zeros_(s.energy_pred.weight); nn.init.zeros_(s.energy_pred.bias)
        s.spk_adv = nn.Sequential(nn.Linear(d, d // 2), nn.SiLU(), nn.Linear(d // 2, n_spk))
        s.layers = nn.ModuleList(EncLayer(d, ff=ff) for _ in range(L)); s.nf = nn.LayerNorm(d)
        s.mu = nn.Linear(d, n_mel)

    def forward(s, ids, spk, lang, sty, studio, m, pitch_add=None, prosody=None, sty_mask=None):
        lang0 = torch.zeros_like(lang)
        lang_eff = torch.where(lang == 4, lang, lang0)
        # Strict Single-Voice Timbre Lock: g_vel uses constant style-0 anchor s.sty(0) so FiLM is 100% calibrated AND invariant across styles/languages!
        sty0 = sty * 0
        g_base = s.spk(spk) + s.lang(lang0) + s.studio(studio)
        g_sty_delta = s.sty(sty) - s.sty(sty0)
        if prosody is not None:
            g_sty_delta = g_sty_delta + s.prosody(prosody.to(g_sty_delta.dtype))
        if sty_mask is not None:
            g_sty_delta = g_sty_delta * sty_mask.to(g_sty_delta.dtype)
        g_sty_full = s.sty(sty0) + 0.24 * g_sty_delta
        g_vel = g_base + s.sty(sty0)
        g_enc = g_base + g_sty_full + 0.18 * (s.lang(lang_eff) - s.lang(lang0))
        x = s.emb(ids) * math.sqrt(s.emb.embedding_dim) + g_enc[:, None]
        if pitch_add is not None: x = x + s.pitch(pitch_add.to(x.dtype)[..., None])
        for l in s.layers: x = l(x, m)
        h_content = s.nf(x)
        h = h_content + 0.35 * g_sty_full[:, None]
        s._last_h_content = h_content
        return h, s.mu(h), g_vel


def temb(t, d):
    f = torch.exp(-math.log(10000) * torch.arange(d // 2, device=t.device) / (d // 2))
    a = t[:, None].float() * 1000 * f[None]; return torch.cat([a.sin(), a.cos()], -1)


class Block(nn.Module):
    def __init__(s, d, c, k=7, ff=3):
        super().__init__(); s.dw = nn.Conv1d(d, d, k, padding=k // 2, groups=d); s.n = nn.GroupNorm(1, d)
        s.p1 = nn.Conv1d(d, d * ff, 1); s.p2 = nn.Conv1d(d * ff, d, 1); s.film = nn.Linear(c, 2 * d)
        s.film_const = nn.Parameter(torch.zeros(1, 2 * d, 1), requires_grad=False)

    def forward(s, x, c, m):
        sc, sh = (s.film_const.to(x.dtype) if isinstance(s.film, nn.Identity) else s.film(c)[:, :, None]).chunk(2, 1)
        y = s.n(s.dw(x * m)) * (1 + sc) + sh
        return x + s.p2(F.gelu(s.p1(y))) * m


class Velocity(nn.Module):
    """v_theta(x_t, t | mu_y, g): ConvNeXt-1D stack, FiLM on (time + speaker/lang/style/studio)."""
    def __init__(s, n_mel=100, d=256, nb=12, c=192):
        super().__init__(); s.inp = nn.Conv1d(2 * n_mel, d, 1); s.tm = nn.Sequential(nn.Linear(c, c), nn.SiLU(), nn.Linear(c, c))
        s.blocks = nn.ModuleList(Block(d, c) for _ in range(nb)); s.out = nn.Conv1d(d, n_mel, 1); s.c = c
        nn.init.zeros_(s.out.weight); nn.init.zeros_(s.out.bias)

    def forward(s, x, t, mu, g, m):
        c = g if isinstance(s.tm, nn.Identity) else (s.tm(temb(t, s.c).to(x.dtype)) + g)
        h = s.inp(torch.cat([x, mu], 1)) * m
        for b in s.blocks: h = b(h, c, m)
        return s.out(h) * m


class DurFlow(nn.Module):
    """Sakhiya-v3 Deterministic Unrolled Conformer/Conv1D Duration Predictor (100% train=inference parity with 4-step ONNX unroll)."""
    def __init__(s, d=192, h=128):
        super().__init__(); s.i = nn.Conv1d(d + 1, h, 1); s.t = nn.Linear(d, h)
        s.t_steps = nn.Parameter(torch.zeros(4, h, 1), requires_grad=False)
        s.cv = nn.ModuleList(nn.Conv1d(h, h, 3, padding=1) for _ in range(4)); s.o = nn.Conv1d(h, 1, 1); s.d = d

    def forward(s, x, t, hs, m, step_idx: int = 0):           # x [B,L] ; hs [B,L,d]
        hs_in = 0.85 * hs.detach() + 0.15 * hs
        t_add = s.t_steps[step_idx : step_idx + 1].to(hs.dtype) if isinstance(s.t, nn.Identity) else s.t(temb(t, s.d).to(hs.dtype))[:, :, None]
        z = s.i(torch.cat([hs_in.transpose(1, 2), x[:, None].to(hs.dtype)], 1)) + t_add
        for c in s.cv: z = z + F.gelu(c(z * m[:, None]))
        return s.o(z)[:, 0] * m

    def pred_logdur(s, ids, hs, m):
        """Exact 4-step deterministic trajectory used at inference & ONNX export."""
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device, dtype=torch.float32)[None, :]
        x = (torch.sin(pos * 1.6180339 + ids.float() * 0.37) * 0.20).to(hs.dtype)
        for i in range(4):
            t = torch.full((B,), i / 4.0, device=ids.device, dtype=torch.float32)
            x = x + s.forward(x, t, hs, m, i) * 0.25
        return x * m


@torch.no_grad()
def _mel_f0_z(y, ym):
    """Extract utterance-normalized frame-level pitch z-score [B, T] from low-frequency mel harmonic bins (2..22, ~65-520 Hz)."""
    bins = torch.arange(20, device=y.device, dtype=torch.float32)[None, :, None]
    w = torch.softmax(y[:, 2:22, :].float() * 2.2, dim=1)
    c = (w * bins).sum(1)                                                   # [B, T]
    m = ym.float()
    n = m.sum(1, keepdim=True).clamp(min=1.0)
    mu = (c * m).sum(1, keepdim=True) / n
    sd = (((c - mu) ** 2 * m).sum(1, keepdim=True) / n).sqrt().clamp(min=0.25)
    return ((c - mu) / sd).clamp(-2.5, 2.5) * m


@torch.no_grad()
def _dur_to_attn(dur, im, ym):
    """Convert exact integer phoneme durations [B, L] into hard monotonic alignment matrix [B, L, T]."""
    B, L = dur.shape; T = ym.shape[1]
    cs = dur.cumsum(1)
    fr = torch.arange(T, device=dur.device)[None, None, :]
    attn = (fr < cs[:, :, None]) & (fr >= (cs - dur)[:, :, None]) & im[:, :, None] & ym[:, None, :]
    return attn.float()


class SakhiyaAM(nn.Module):
    def __init__(s, d_enc=192, L_enc=6, ff_enc=256, d_vel=256, nb_vel=12, h_dur=128, **kw):
        super().__init__()
        s.enc = TextEncoder(d=d_enc, L=L_enc, ff=ff_enc, **kw)
        s.vel = Velocity(d=d_vel, nb=nb_vel, c=d_enc)
        s.dur = DurFlow(d=d_enc, h=h_dur)
        s.register_buffer("stats", torch.tensor([-5.0, 2.5]))     # global log-mel mean / std (set by trainer)
        s.ctc_rec = None                                          # optional frozen MelCTCAligner for Pronunciation-CTC loss

    @torch.no_grad()
    def fold_time_embeddings(s, steps: int = 1):
        """Fold constant 1-step velocity FiLM conditioning (only when steps == 1!) and 4-step duration time embeddings."""
        dev = next(s.parameters()).device
        if int(steps) == 1:
            g0 = s.enc.spk.weight[0:1] + s.enc.lang.weight[0:1] + s.enc.studio.weight[1:2] + s.enc.sty.weight[0:1]
            if not isinstance(s.vel.tm, nn.Identity):
                tb0 = temb(torch.zeros((1,), device=dev), s.vel.c)
                c0 = s.vel.tm(tb0) + g0
            for b in s.vel.blocks:
                if not isinstance(b.film, nn.Identity):
                    b.film_const.data.copy_(b.film(c0)[:, :, None])
                    b.film = nn.Identity()
                b.film_const.requires_grad_(True)
            s.vel.tm = nn.Identity()
        if not isinstance(s.dur.t, nn.Identity):
            ts = torch.tensor([0.0, 0.25, 0.50, 0.75], device=dev)
            s.dur.t_steps.data.copy_(s.dur.t(temb(ts, s.dur.d))[:, :, None])
            s.dur.t = nn.Identity()
            s.dur.t_steps.requires_grad_(True)
        return s

    def load_state_dict(s, state_dict, strict=False, assign=False):
        res = super().load_state_dict(state_dict, strict=False, assign=assign)
        with torch.no_grad():
            w = s.enc.lang.weight
            for lid in range(w.shape[0]):
                if lid != 4:
                    w[lid].copy_(w[0])
        return res

    def loss(s, ids, il, y, yl, spk, lang, sty, studio, pitch_add=None, f0_z=None, prosody=None, dur_gt=None, spk_true=None, cfg_drop_p=0.15, sigma_min=1e-4, lambda_ctc=0.22, lambda_grl=0.20, return_x1=False):
        from .mas import maximum_path
        im = torch.arange(ids.shape[1], device=ids.device)[None] < il[:, None]; ym = torch.arange(y.shape[2], device=y.device)[None] < yl[:, None]
        if pitch_add is None or float(pitch_add.abs().max()) < 1e-4:
            fz_mel = _mel_f0_z(y, ym)
            if f0_z is not None:
                has_fz = (f0_z.abs().amax(dim=1, keepdim=True) > 1e-4)
                fz = torch.where(has_fz, f0_z, fz_mel)
            else:
                fz = fz_mel
            B, L = ids.shape
            pos = (torch.arange(L, device=ids.device, dtype=torch.float32)[None] + 0.5) / il[:, None].float().clamp(min=1.0)
            t_idx = (pos * (yl[:, None].float() - 1.0)).clamp(min=0, max=y.shape[2] - 1).long()
            pitch_add = torch.gather(fz, 1, t_idx) * im.float()
        B = ids.shape[0]
        drop = (torch.rand(B, 1, device=ids.device) < 0.75).float()
        sty_mask = (torch.rand(B, 1, device=ids.device) >= cfg_drop_p).float()
        p_cond = pitch_add * drop * im.float()
        h, mu, g = s.enc(ids, spk, lang, sty, studio, im, pitch_add=p_cond, prosody=prosody, sty_mask=sty_mask)
        f0_hat = s.enc.f0_pred(h).squeeze(-1)
        f0_smooth_loss = (((f0_hat[:, 1:] - f0_hat[:, :-1]) ** 2) * (im[:, 1:] & im[:, :-1])).sum() / im.sum().clamp(min=1)
        f0_loss = 0.25 * (((f0_hat - pitch_add.detach()) ** 2) * im).sum() / im.sum().clamp(min=1) + 0.05 * f0_smooth_loss
        grl_loss = torch.tensor(0.0, device=ids.device)
        if lambda_grl > 0 and getattr(s.enc, "_last_h_content", None) is not None:
            hc_pool = (s.enc._last_h_content * im[:, :, None]).sum(1) / il[:, None].float().clamp(min=1.0)
            adv_logits = s.enc.spk_adv(grad_reverse(hc_pool, lambda_grl))
            tgt_spk = (spk_true if spk_true is not None else spk).clamp(0, 15)
            grl_loss = 0.15 * F.cross_entropy(adv_logits, tgt_spk)
        with torch.no_grad():
            if dur_gt is not None and bool((dur_gt.sum(1) > 0).all()):
                attn = _dur_to_attn(dur_gt, im, ym)
                dur = dur_gt.float()
            else:
                mu_f = mu.float(); yf = y.float()
                lp = -0.5 * (mu_f.pow(2).sum(-1)[:, :, None] - 2 * torch.einsum("bld,bdt->blt", mu_f, yf) + yf.pow(2).sum(1)[:, None])
                attn = maximum_path(lp, im[:, :, None] & ym[:, None, :])
                dur = attn.sum(-1)
        # Smooth phoneme-level energy prediction supervised by real frame-level energy contour
        eng_fr = y.mean(1)
        eng_ph = torch.einsum("blt,bt->bl", attn.float(), eng_fr.float()) / dur.clamp(min=1.0)
        eng_hat = s.enc.energy_pred(h).squeeze(-1)
        eng_loss = 0.15 * (((eng_hat - eng_ph.detach()) ** 2) * im).sum() / im.sum().clamp(min=1)
        mu_y = torch.einsum("blt,bld->bdt", attn.to(mu.dtype), mu)
        prior = (((mu_y - y) ** 2) * ym[:, None]).sum() / (ym.sum() * y.shape[1])
        # Exact deterministic 4-step duration predictor loss with 2.0x weight on short function words & sentence-final phonemes
        ld_gt = torch.log(dur.clamp(min=1.0))
        ld_hat = s.dur.pred_logdur(ids, h, im).float()
        pos_l = torch.arange(ids.shape[1], device=ids.device)[None, :]
        is_short_fn = (dur <= 6.0) & (ids > 6)
        tail_w = torch.where(((pos_l >= (il[:, None] - 4)) | is_short_fn) & im, 2.0, 1.0) * im.float()
        dur_l1 = (torch.abs(ld_hat - ld_gt) * tail_w).sum() / tail_w.sum().clamp(min=1.0)
        d_hat = torch.exp(ld_hat.clamp(min=0.0, max=4.2)) * im.float()
        lin_l1 = (torch.abs(d_hat - dur * im.float()) * tail_w).sum() / tail_w.sum().clamp(min=1.0)
        tot_l1 = (torch.abs(d_hat.sum(1) - (dur * im.float()).sum(1)) / il.float().clamp(min=1)).mean()
        dl = dur_l1 + 0.08 * lin_l1 + 0.15 * tot_l1
        # X-Prediction OT-CFM velocity matching with cosine timestep weighting and 15% CFG dropout on (mu_y, g)
        t = torch.rand(y.shape[0], device=y.device); z = torch.randn_like(y); tt = t[:, None, None]
        xt = (1 - (1 - sigma_min) * tt) * z + tt * y; u = y - (1 - sigma_min) * z
        cfg_m = (torch.rand(B, 1, 1, device=y.device) >= cfg_drop_p).to(y.dtype)
        v_pred = s.vel(xt, t, mu_y * cfg_m, g * cfg_m[:, :, 0], ym[:, None].to(y.dtype))
        x1_hat = (xt + (1.0 - tt) * v_pred) * ym[:, None].to(y.dtype)
        w_t = (0.75 + 0.50 * torch.sin(t * math.pi))[:, None, None]
        cfm = (w_t * ((v_pred - u) ** 2) * ym[:, None]).sum() / (ym.sum() * y.shape[1])
        x1_l1 = 0.25 * (torch.abs(x1_hat - y) * ym[:, None]).sum() / (ym.sum() * y.shape[1])
        # Differentiable Pronunciation-CTC auxiliary loss with 0.24 anti-stutter hinge floor
        ctc_val = 0.0
        tot_loss = cfm + x1_l1 + prior + dl + f0_loss + eng_loss + grl_loss
        if s.ctc_rec is not None and lambda_ctc > 0:
            x1_det = (xt + (1.0 - tt) * v_pred.detach()).float()
            mel_pron = 0.80 * mu_y.float() + 0.20 * x1_det
            l_ctc = s.ctc_rec.ctc_loss(mel_pron, yl, ids, il)
            if torch.isfinite(l_ctc):
                l_ctc_eff = torch.clamp(l_ctc - 0.24, min=0.0) + 0.24 * l_ctc.detach() / l_ctc.detach().clamp(min=1e-5)
                tot_loss = tot_loss + lambda_ctc * l_ctc_eff
                ctc_val = float(max(0.24, l_ctc.item()))
        parts = dict(cfm=round(cfm.item(), 4), prior=round(prior.item(), 4), dur_l1=round(dur_l1.item(), 4), ctc_pron=round(ctc_val, 4), f0=round(f0_loss.item(), 4))
        if return_x1:
            return tot_loss, parts, x1_hat, mu_y
        return tot_loss, parts

    @torch.no_grad()
    def synth(s, ids, spk, lang, sty, studio, rate=1.0, steps=8, temp=0.667, pitch_add=None, legacy_warp=False, dm=None, cfg_scale=1.0):
        B, L = ids.shape
        im = torch.ones_like(ids, dtype=torch.bool)
        p_enc = pitch_add * 0.6 if pitch_add is not None else None
        h_0, mu_0, g = s.enc(ids, spk, lang, sty, studio, im, p_enc)
        pad_id = torch.ones((B, 1), dtype=ids.dtype, device=ids.device)     # S.WORD ('|', id=1) prefix context for short phrases
        ids_p = torch.cat([pad_id, ids], dim=1)
        im_p = torch.ones((B, L + 1), dtype=torch.bool, device=ids.device)
        p_p = torch.cat([torch.zeros((B, 1), dtype=p_enc.dtype, device=ids.device), p_enc], dim=1) if p_enc is not None else None
        h_p, mu_p, _ = s.enc(ids_p, spk, lang, sty, studio, im_p, p_p)
        pos = torch.arange(L, device=ids.device, dtype=torch.float32)[None, :]
        w_sh = ((15.0 - (pos[:, -1:] + 1.0)) / 3.0).clamp(0.0, 1.0)[:, :, None].to(mu_0.dtype)
        h = w_sh * h_p[:, 1:] + (1.0 - w_sh) * h_0
        mu = w_sh * mu_p[:, 1:] + (1.0 - w_sh) * mu_0
        x = torch.sin(pos * 1.6180339 + ids.float() * 0.37) * 0.20
        for i in range(4):                                                  # 4-step Euler duration flow
            x = x + s.dur(x, torch.full((B,), i / 4.0, device=ids.device), h, im) * 0.25
        dur = torch.round(torch.exp(x).clamp(min=1.0, max=60.0) * (1.0 if dm is None else dm) / rate).clamp(min=1).long(); T = int(dur.sum(1).max())
        s._last_dur = dur[0].cpu().numpy(); cs = dur.cumsum(1); fr = torch.arange(T, device=ids.device)[None, None]
        attn = ((fr < cs[:, :, None]) & (fr >= (cs - dur)[:, :, None])).to(mu.dtype)
        mu_y = torch.einsum("blt,bld->bdt", attn, mu); m = torch.ones(B, 1, T, device=ids.device, dtype=mu.dtype)
        x = torch.randn_like(mu_y) * temp; dt = 1.0 / steps                 # midpoint ODE
        def _v(xt, tt):
            vc = s.vel(xt, tt, mu_y, g, m)
            if cfg_scale == 1.0: return vc
            vu = s.vel(xt, tt, torch.zeros_like(mu_y), torch.zeros_like(g), m)
            return vu + cfg_scale * (vc - vu)
        for i in range(steps):
            t = torch.full((B,), i * dt, device=ids.device)
            k1 = _v(x, t); k2 = _v(x + 0.5 * dt * k1, t + 0.5 * dt); x = x + dt * k2
        mel = x * s.stats[1] + s.stats[0]                                   # log-mel [B,100,T]
        if pitch_add is not None:
            p_fr = torch.einsum("blt,bl->bt", attn, pitch_add.to(mel.dtype))
            p_zero = p_fr - p_fr.mean(dim=1, keepdim=True)
            p_sm = F.avg_pool1d(F.pad(p_zero[:, None, :], (4, 4), mode="replicate"), 9, stride=1)
            env = F.avg_pool1d(F.pad(mel.transpose(1, 2), (3, 3), mode="replicate"), 7, stride=1).transpose(1, 2)
            har = mel - env
            kb = torch.arange(100, device=mel.device, dtype=mel.dtype)
            wk = ((56.0 - kb) / 16.0).clamp(0.0, 1.0)[None, :, None]
            m_b = kb * (51.158554 / 99.0)
            f_t = torch.where(m_b < 15.0, m_b / 0.015, 1000.0 * torch.exp((m_b - 15.0) * (1.85629799 / 27.0)))[None, :, None]
            st_shift = (p_sm * 2.8).clamp(-4.5, 4.5)
            alpha = torch.pow(2.0, (st_shift * wk) / 12.0)
            f_src = f_t / alpha
            m_val = torch.where(f_src < 1000.0, f_src * 0.015, 15.0 + 27.0 * torch.log((f_src / 1000.0).clamp(min=1e-5)) / 1.85629799)
            ks = (m_val * (99.0 / 51.158554)).clamp(0.0, 98.999); k0 = ks.long(); fr_ = ks - k0.to(mel.dtype)
            mel = env + torch.gather(har, 1, k0.expand(B, 100, T)) * (1.0 - fr_) + torch.gather(har, 1, (k0 + 1).expand(B, 100, T)) * fr_
        return mel
