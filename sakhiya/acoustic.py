"""Sakhiya-AM: ~9M-param non-autoregressive acoustic model (FastPitch-style, learned aligner at train time)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .symbols import N_SYMBOLS, PAD, SYM2ID
from .audio import NMEL

MEL_MU, MEL_SD = -5.5, 2.5


def pos_enc(T, d, device):
    p = torch.arange(T, device=device)[:, None].float()
    i = torch.arange(0, d, 2, device=device).float()[None]
    a = p / torch.pow(10000.0, i / d)
    return torch.cat([a.sin(), a.cos()], -1)[None]


class FFT(nn.Module):
    def __init__(s, d=256, h=4, ff=512, k=3, p=0.1):
        super().__init__()
        s.h, s.qkv, s.o = h, nn.Linear(d, 3 * d), nn.Linear(d, d)
        s.n1, s.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.c1, s.c2 = nn.Conv1d(d, ff, k, padding=k // 2), nn.Conv1d(ff, d, k, padding=k // 2)
        s.dp = nn.Dropout(p)

    def forward(s, x, mask=None):               # x (B,T,d); mask (B,T) True=valid
        B, T, d = x.shape
        q, k, v = s.qkv(x).view(B, T, 3, s.h, d // s.h).permute(2, 0, 3, 1, 4)
        am = None if mask is None else mask[:, None, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=am).transpose(1, 2).reshape(B, T, d)
        x = s.n1(x + s.dp(s.o(y)))
        if mask is not None: x = x * mask[..., None]
        y = s.c2(s.dp(F.relu(s.c1(x.transpose(1, 2))))).transpose(1, 2)
        x = s.n2(x + s.dp(y))
        return x if mask is None else x * mask[..., None]


class Pred(nn.Module):
    def __init__(s, d=256, k=3, p=0.2):
        super().__init__()
        s.c1, s.c2 = nn.Conv1d(d, d, k, padding=k // 2), nn.Conv1d(d, d, k, padding=k // 2)
        s.n1, s.n2, s.l, s.dp = nn.LayerNorm(d), nn.LayerNorm(d), nn.Linear(d, 1), nn.Dropout(p)

    def forward(s, x):
        x = s.dp(s.n1(F.relu(s.c1(x.transpose(1, 2))).transpose(1, 2)))
        x = s.dp(s.n2(F.relu(s.c2(x.transpose(1, 2))).transpose(1, 2)))
        return s.l(x)[..., 0]


class Aligner(nn.Module):   # training-only
    def __init__(s, d=256, a=128):
        super().__init__()
        s.k = nn.Sequential(nn.Conv1d(d, d, 3, padding=1), nn.ReLU(), nn.Conv1d(d, a, 1))
        s.q = nn.Sequential(nn.Conv1d(NMEL, 128, 3, padding=1), nn.ReLU(), nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(), nn.Conv1d(128, a, 1))
        s.t = nn.Parameter(torch.tensor(-3.0))

    def forward(s, emb, mel_n):        # emb (B,Tt,d); mel_n (B,80,Tm) -> logits (B,Tm,Tt)
        k, q = s.k(emb.transpose(1, 2)), s.q(mel_n)
        d = (q ** 2).sum(1)[:, :, None] + (k ** 2).sum(1)[:, None, :] - 2 * torch.einsum("bam,bat->bmt", q, k)
        return -F.softplus(s.t) * d


class Acoustic(nn.Module):
    def __init__(s, d=256, n_enc=4, n_dec=4, n_spk=32, n_lang=8):
        super().__init__()
        s.d = d
        s.emb = nn.Embedding(N_SYMBOLS, d, padding_idx=SYM2ID[PAD])
        s.sa_emb = nn.Embedding(N_SYMBOLS, d, padding_idx=SYM2ID[PAD])
        nn.init.zeros_(s.sa_emb.weight)
        s.spk, s.lang = nn.Embedding(n_spk, d), nn.Embedding(n_lang, d)
        s.enc = nn.ModuleList(FFT(d) for _ in range(n_enc))
        s.dec = nn.ModuleList(FFT(d) for _ in range(n_dec))
        s.dur, s.pit, s.eng = Pred(d), Pred(d), Pred(d)
        s.pemb, s.eemb = nn.Conv1d(1, d, 3, padding=1), nn.Conv1d(1, d, 3, padding=1)
        s.out = nn.Linear(d, NMEL)
        s.aligner = Aligner(d)

    def text_emb(s, ids, spk, lang):
        e = s.emb(ids)
        is_sa = (lang == 4)[:, None, None].to(e.dtype)
        return (e + s.sa_emb(ids) * is_sa) * math.sqrt(s.d) + s.spk(spk)[:, None] + s.lang(lang)[:, None]

    def encode(s, ids, spk, lang, mask=None):
        x = s.text_emb(ids, spk, lang)
        x = x + pos_enc(ids.shape[1], s.d, ids.device)
        for b in s.enc: x = b(x, mask)
        return x, s.dur(x), s.pit(x), s.eng(x)

    def condition(s, h, pitch, energy):
        return h + s.pemb(pitch[:, None]).transpose(1, 2) + s.eemb(energy[:, None]).transpose(1, 2)

    def decode(s, x, mask=None):
        x = x + pos_enc(x.shape[1], s.d, x.device)
        for b in s.dec: x = b(x, mask)
        return s.out(x).transpose(1, 2) * MEL_SD + MEL_MU      # (B,80,Tm) raw log-mel

    @staticmethod
    def regulate(x, dur):
        """x (B,Tt,d), dur (B,Tt) int -> (B,Tm,d) with Tm=max sum."""
        c = torch.cumsum(dur, 1)
        Tm = int(c[:, -1].max())
        idx = torch.searchsorted(c, torch.arange(Tm, device=x.device)[None].expand(x.shape[0], -1).contiguous(), right=True)
        idx = idx.clamp(max=x.shape[1] - 1)
        return torch.gather(x, 1, idx[..., None].expand(-1, -1, x.shape[2]))

    @torch.no_grad()
    def infer(s, ids, spk=0, lang=0, rate=1.0, pitch_shift=0.0, pitch_scale=1.0, energy_scale=1.0, pause_scale=1.0, pause_ids=()):
        dev = ids.device
        spk_t, lang_t = torch.tensor([spk], device=dev), torch.tensor([lang], device=dev)
        h, ld, p, e = s.encode(ids, spk_t, lang_t)
        dur = (torch.exp(ld) - 1).clamp(min=1.0)
        if pause_scale != 1.0 and pause_ids:
            pm = torch.zeros_like(dur, dtype=torch.bool)
            for i in pause_ids: pm |= ids == i
            dur = torch.where(pm, dur * pause_scale, dur)
        dur = torch.round(dur / rate).clamp(min=1).long()
        x = s.condition(h, p * pitch_scale + pitch_shift, e * energy_scale)
        return s.decode(s.regulate(x, dur)), dur
