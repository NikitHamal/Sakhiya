"""Sakhiya distilled students: ONE single-stage text -> 24 kHz waveform ONNX graph per size tier.

phonemes (+lang) -> ConvNeXt-1D encoder -> duration predictor -> length regulator (+ in-phoneme position features)
-> frame-level pitch (log-F0 / VUV) predictor + pitch embedding -> ConvNeXt-1D decoder (decoder-heavy) ->
complex STFT head (log-magnitude + cos/sin phase, Vocos style) -> exact matrix-multiply iSTFT (ConvTranspose1d whose
cos/sin basis is generated inside the graph, so it costs 0 bytes on disk).

Parameter sharing: `n_unique < n_layers` loops a few shared ConvNeXt blocks with tiny per-iteration FiLM vectors (Pico).
QAT: `set_qat(model, True)` turns on per-channel int8 fake-quant (STE) on every QConv1d weight.
Training-only parts (MAS prior head, GAN discriminators) are never exported.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SR, N_FFT, HOP, NB, N_MEL = 24000, 1024, 256, 513, 100
N_SYM = 64          # >= sakhiya.symbols.N_SYMBOLS (61)
N_LANG = 2          # 0 = Nepali (+ English loans), 1 = Sanskrit / Vedic

# int8 MB ~= parameter count in millions (+ ~2 % fp32 norms/biases). Tuned with `python student.py` (prints counts).
TIERS = {
    't30': dict(d_e=384, L_e=6, u_e=6, ff_e=3, d_d=512, L_d=14, u_d=14, ff_d=3, hb=0, d_p=128, L_p=3),
    't15': dict(d_e=320, L_e=4, u_e=4, ff_e=3, d_d=384, L_d=12, u_d=12, ff_d=3, hb=0, d_p=96, L_p=3),
    't10': dict(d_e=256, L_e=4, u_e=4, ff_e=3, d_d=320, L_d=11, u_d=11, ff_d=3, hb=0, d_p=96, L_p=3),
    't5': dict(d_e=192, L_e=4, u_e=4, ff_e=3, d_d=256, L_d=8, u_d=8, ff_d=3, hb=0, d_p=64, L_p=3),
    't1': dict(d_e=128, L_e=4, u_e=1, ff_e=3, d_d=192, L_d=9, u_d=3, ff_d=3, hb=96, d_p=48, L_p=2),
}


def fq(w, axis=0):
    red = [i for i in range(w.dim()) if i != axis]
    s = w.detach().abs().amax(dim=red, keepdim=True).clamp(min=1e-8) / 127.0
    return w + ((w / s).round().clamp(-127, 127) * s - w).detach()


class QConv1d(nn.Conv1d):
    qat = False

    def forward(self, x):
        w = fq(self.weight) if (self.qat and self.training) else self.weight
        return self._conv_forward(x, w, self.bias)


def set_qat(model, flag):
    for m in model.modules():
        if isinstance(m, QConv1d):
            m.qat = flag


class ChanLN(nn.Module):
    """LayerNorm over channels of [B,C,T] using plain ReduceMean/Sqrt ops (identical in every ORT backend)."""

    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, c, 1)); self.b = nn.Parameter(torch.zeros(1, c, 1)); self.eps = eps

    def forward(self, x):
        m = x.mean(1, keepdim=True); v = ((x - m) ** 2).mean(1, keepdim=True)
        return (x - m) * torch.rsqrt(v + self.eps) * self.g + self.b


class Block(nn.Module):
    def __init__(self, d, ff, k=7):
        super().__init__()
        self.dw = QConv1d(d, d, k, padding=k // 2, groups=d); self.ln = ChanLN(d)
        self.p1 = QConv1d(d, d * ff, 1); self.p2 = QConv1d(d * ff, d, 1)
        self.gamma = nn.Parameter(torch.full((1, d, 1), 0.1))

    def forward(self, x, mask=None, film=None):
        h = self.dw(x if mask is None else x * mask)
        h = self.ln(h)
        if film is not None:
            h = h * (1 + film[:, :x.shape[1]]) + film[:, x.shape[1]:]
        return x + self.gamma * self.p2(F.gelu(self.p1(h)))


class Stack(nn.Module):
    """n_layers ConvNeXt blocks drawn from n_unique distinct ones; shared blocks get 2*d FiLM params per iteration."""

    def __init__(self, d, ff, n_layers, n_unique, k=7):
        super().__init__()
        self.n, self.u = n_layers, n_unique
        self.blocks = nn.ModuleList([Block(d, ff, k) for _ in range(n_unique)])
        self.film = nn.Parameter(torch.zeros(n_layers, 2 * d, 1)) if n_unique < n_layers else None

    def forward(self, x, mask=None):
        for i in range(self.n):
            f = None if self.film is None else self.film[i][None]
            x = self.blocks[i % self.u](x, mask, f)
        return x


class ISTFT(nn.Module):
    """iDFT + windowed overlap-add as ONE ConvTranspose1d; the cos/sin basis is built from Range/Cos/Sin in-graph."""

    def __init__(self):
        super().__init__()
        self.register_buffer('k', torch.arange(NB).float()); self.register_buffer('n', torch.arange(N_FFT).float())
        self.register_buffer('win', torch.hann_window(N_FFT, periodic=True))
        coef = torch.full((NB,), 2.0); coef[0] = 1.0; coef[-1] = 1.0
        self.register_buffer('coef', coef)

    def forward(self, re, im):
        with torch.autocast(device_type='cuda', enabled=False):
            return self._fwd(re.float(), im.float())

    def _fwd(self, re, im):
        ang = (2 * math.pi / N_FFT) * self.k[:, None] * self.n[None, :]
        sc = self.coef[:, None] * self.win[None, :] / (N_FFT * 1.5)
        w = torch.cat([torch.cos(ang) * sc, -torch.sin(ang) * sc], 0)[:, None, :]
        y = F.conv_transpose1d(torch.cat([re, im], 1), w, stride=HOP)
        c = (N_FFT - HOP) // 2
        return y[:, 0, c:-c]


class Student(nn.Module):
    def __init__(self, d_e, L_e, u_e, ff_e, d_d, L_d, u_d, ff_d, hb, d_p, L_p):
        super().__init__()
        self.cfg = dict(d_e=d_e, L_e=L_e, u_e=u_e, ff_e=ff_e, d_d=d_d, L_d=L_d, u_d=u_d, ff_d=ff_d, hb=hb, d_p=d_p, L_p=L_p)
        self.emb = nn.Embedding(N_SYM, d_e); self.lang = nn.Embedding(N_LANG, d_e)
        self.enc = Stack(d_e, ff_e, L_e, u_e, k=5)
        self.dur = nn.Sequential(QConv1d(d_e, d_e // 2, 3, padding=1), nn.ReLU(), QConv1d(d_e // 2, 1, 1))
        self.proj = QConv1d(d_e, d_d, 1)
        self.pos = QConv1d(2, d_d, 1)
        self.p_in = QConv1d(d_d, d_p, 1); self.p_stack = Stack(d_p, 2, L_p, L_p); self.p_out = QConv1d(d_p, 2, 1)
        self.p_emb = QConv1d(2, d_d, 1)
        self.dec = Stack(d_d, ff_d, L_d, u_d)
        self.dec_ln = ChanLN(d_d)
        self.head = nn.Sequential(QConv1d(d_d, hb, 1), nn.GELU(), QConv1d(hb, 3 * NB, 1)) if hb else QConv1d(d_d, 3 * NB, 1)
        self.istft = ISTFT()
        # training only (MAS alignment prior) -- stripped at export
        self.prior = nn.Conv1d(d_e, N_MEL, 1)
        self.register_buffer('f0_stats', torch.tensor([5.3, 0.3]))   # mean / std of log-F0 (set by training script)
        self.register_buffer('mel_stats', torch.tensor([-5.0, 2.5]))

    # ---- text side -------------------------------------------------------------------------------------------
    def encode(self, ids, lang, mask):
        x = (self.emb(ids) + self.lang(lang)[:, None, :]).transpose(1, 2) * mask
        h = self.enc(x, mask) * mask
        return h, self.dur(h * mask)[:, 0]          # h [B,d_e,L], log(dur+1) [B,L]

    # ---- frame side ------------------------------------------------------------------------------------------
    def frame_feats(self, idx, dur):
        """idx [B,T] phoneme index per frame, dur [B,L] float frames -> [B,2,T] (in-phoneme position, log duration)."""
        d = torch.gather(dur, 1, idx)
        start = torch.cumsum(dur, 1) - dur
        t = torch.arange(idx.shape[1], device=idx.device, dtype=torch.float32)[None]
        frac = (t - torch.gather(start, 1, idx) + 0.5) / d.clamp(min=1.0)
        return torch.stack([frac, torch.log(d + 1.0) / 3.0], 1)

    def decode(self, xf, feats, pitch=None):
        """xf [B,d_d,T] expanded hidden; returns wav [B,T*HOP], predicted pitch [B,2,T]."""
        x = xf + self.pos(feats)
        p = self.p_out(self.p_stack(self.p_in(x)))
        pe = torch.cat([p[:, :1], torch.sigmoid(p[:, 1:2])], 1) if pitch is None else pitch   # GT (teacher forcing) in training
        x = x + self.p_emb(pe)
        x = self.dec_ln(self.dec(x))
        o = self.head(x).float()
        mag = torch.exp(o[:, :NB].clamp(max=7.0)); c, s = o[:, NB:2 * NB], o[:, 2 * NB:]
        n = torch.sqrt(c * c + s * s + 1e-6)
        return self.istft(mag * c / n, mag * s / n), p

    @torch.no_grad()
    def infer(self, ids, lang, dur_scale, pitch_shift):
        """Single-graph inference: ids [1,L] int64, lang [1] int64, dur_scale [1] float, pitch_shift [1] float (std units)."""
        L = ids.shape[1]
        mask = torch.ones(1, 1, L, device=ids.device)
        h, ld = self.encode(ids, lang, mask)
        dur = torch.clamp(torch.round((torch.exp(ld) - 1.0) * dur_scale[:, None]), min=1.0)
        ends = torch.cumsum(dur, 1)
        T = ends[0, -1].to(torch.int64)
        t = torch.arange(0, T, device=ids.device, dtype=torch.float32)
        idx = (t[None, :, None] >= ends[:, None, :]).to(torch.float32).sum(-1).to(torch.int64).clamp(max=L - 1)
        xh = self.proj(h)
        xf = torch.gather(xh, 2, idx[:, None, :].expand(-1, xh.shape[1], -1))
        feats = self.frame_feats(idx, dur)
        x = xf + self.pos(feats)
        p = self.p_out(self.p_stack(self.p_in(x)))
        pe = torch.cat([p[:, :1] + pitch_shift[:, None, None] * (torch.sigmoid(p[:, 1:2]) > 0.5).to(p.dtype), torch.sigmoid(p[:, 1:2])], 1)
        x = x + self.p_emb(pe)
        x = self.dec_ln(self.dec(x))
        o = self.head(x).float()
        mag = torch.exp(o[:, :NB].clamp(max=7.0)); c, s = o[:, NB:2 * NB], o[:, 2 * NB:]
        n = torch.sqrt(c * c + s * s + 1e-6)
        return self.istft(mag * c / n, mag * s / n)


def build(tier):
    return Student(**TIERS[tier])


def n_params(m, deployed=True):
    skip = ('prior.',) if deployed else ()
    return sum(p.numel() for n, p in m.named_parameters() if not n.startswith(skip))


if __name__ == '__main__':
    for t in TIERS:
        m = build(t)
        print(t, 'deployed params %.2fM  (~%.2f MB int8)' % (n_params(m) / 1e6, n_params(m) * 1.02 / 1e6))
