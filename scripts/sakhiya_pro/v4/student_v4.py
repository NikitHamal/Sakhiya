"""Sakhiya v4 Unified Single-Stage Distilled Students:
  - Sakhiya-V2-Micro (`micro`): <= 4.5 MB total INT8 ONNX (~3.75 MB)
  - Sakhiya-V2-Pico  (`pico`):  1.0 to 1.4 MB total INT8 ONNX (~1.18 MB, Weight-Tied Looped Universal Blocks + FiLM)
  - Sakhiya-V2-Compact (`t10`): ~9.2 MB total INT8 ONNX

Architecture:
  phonemes (+lang) -> ConvNeXt-1D Encoder (Looped Universal + 1-KB FiLM in Pico) -> Duration Predictor -> Length Regulator
  -> Frame-Level ConvNeXt-1D Mel Decoder (Looped Universal + 1-KB FiLM in Pico) -> 100-Band 24 kHz Log-Mel ->
  Embedded 16x-Upsampling Harmonic-iSTFT Vocoder (`VocoderV2`: 128-ch `HarmonicSource` continuous-phase sines + `Snake1d`
  periodic MRF ResBlocks + 1-op COLA `n_fft=64, hop=16` iSTFT) -> 24 kHz Waveform `wav`.

Eliminates the 93.75 Hz frame-rate Vocos phase-lock buzz (`UTMOS 1.26` in v3) permanently by synthesizing waveform
harmonics at 1500 Hz (`hop=16`) with `VocoderV2` warm-sliced from `v5_c1/voc.pt` / `v14_voc144/voc.pt`.
"""
import math, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SR, N_FFT, HOP, NB, N_MEL = 24000, 1024, 256, 513, 100
ISTFT_N_FFT, ISTFT_HOP = 64, 16
N_SYM = 64
N_LANG = 2

TIERS = {
    'micro': dict(d_e=160, L_e=4, u_e=4, ff_e=2, d_d=192, L_d=6, u_d=6, ff_d=2, d_p=48, L_p=2, v_ch=192),
    'pico':  dict(d_e=112, L_e=4, u_e=1, ff_e=2, d_d=144, L_d=6, u_d=2, ff_d=2, d_p=40, L_p=2, v_ch=112),
    't10':   dict(d_e=224, L_e=4, u_e=4, ff_e=3, d_d=256, L_d=8, u_d=8, ff_d=3, d_p=64, L_p=2, v_ch=288),
    't5':    dict(d_e=160, L_e=4, u_e=4, ff_e=2, d_d=192, L_d=6, u_d=6, ff_d=2, d_p=48, L_p=2, v_ch=192),
    't1':    dict(d_e=112, L_e=4, u_e=1, ff_e=2, d_d=144, L_d=6, u_d=2, ff_d=2, d_p=40, L_p=2, v_ch=112),
}


def fq(w, axis=0):
    red = [i for i in range(w.dim()) if i != axis]
    s = w.detach().abs().amax(dim=red, keepdim=True).clamp(min=1e-7) / 127.0
    return w + ((w / s).round().clamp(-127, 127) * s - w).detach()


class QConv1d(nn.Conv1d):
    qat = False

    def forward(self, x):
        use_q = self.qat and self.training and (self.weight.numel() >= 1024) and (self.groups == 1)
        w = fq(self.weight, axis=0) if use_q else self.weight
        return self._conv_forward(x, w, self.bias)


class QConvTranspose1d(nn.ConvTranspose1d):
    qat = False

    def forward(self, x, output_size=None):
        use_q = self.qat and self.training and (self.weight.numel() >= 1024) and (self.groups == 1)
        w = fq(self.weight, axis=1) if use_q else self.weight
        return F.conv_transpose1d(x, w, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)


def set_qat(model, flag):
    for name, m in model.named_modules():
        if isinstance(m, (QConv1d, QConvTranspose1d)):
            # Keep f0_net unperturbed during QAT so cumulative harmonic phase stays locked
            m.qat = False if 'f0_net' in name else bool(flag)


class ChanLN(nn.Module):
    def __init__(self, c, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, c, 1))
        self.b = nn.Parameter(torch.zeros(1, c, 1))
        self.eps = eps

    def forward(self, x):
        m = x.mean(1, keepdim=True)
        v = ((x - m) ** 2).mean(1, keepdim=True)
        return (x - m) * torch.rsqrt(v + self.eps) * self.g + self.b


class Block(nn.Module):
    def __init__(self, d, ff, k=7):
        super().__init__()
        self.dw = QConv1d(d, d, k, padding=k // 2, groups=d)
        self.ln = ChanLN(d)
        self.p1 = QConv1d(d, d * ff, 1)
        self.p2 = QConv1d(d * ff, d, 1)
        self.gamma = nn.Parameter(torch.full((1, d, 1), 0.15))

    def forward(self, x, mask=None, film=None):
        h = self.dw(x if mask is None else x * mask)
        h = self.ln(h)
        if film is not None:
            h = h * (1.0 + film[:, :x.shape[1]]) + film[:, x.shape[1]:]
        return x + self.gamma * self.p2(F.gelu(self.p1(h)))


class Stack(nn.Module):
    """Weight-Tied Looped Universal ConvNeXt Stack with per-iteration FiLM adapters when n_unique < n_layers."""

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


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        xf = x.float()
        a = self.alpha.float().clamp(min=0.1)
        y = xf + (1.0 / a) * (torch.sin(a * xf) ** 2)
        return y.to(x.dtype)


class VocResBlock1(nn.Module):
    def __init__(self, ch, kernel_size=3, dilations=(1, 3, 5)):
        super().__init__()
        self.a1 = nn.ModuleList([Snake1d(ch) for _ in dilations])
        self.c1 = nn.ModuleList([QConv1d(ch, ch, kernel_size, 1, dilation=d, padding=(kernel_size * d - d) // 2) for d in dilations])
        self.a2 = nn.ModuleList([Snake1d(ch) for _ in dilations])
        self.c2 = nn.ModuleList([QConv1d(ch, ch, kernel_size, 1, dilation=1, padding=(kernel_size - 1) // 2) for _ in dilations])

    def forward(self, x):
        for a1, c1, a2, c2 in zip(self.a1, self.c1, self.a2, self.c2):
            x = x + c2(a2(c1(a1(x))))
        return x


class VocHarmonicSource(nn.Module):
    """16x-upsampling NSF Harmonic Source matching VocoderV2 (f0_net + 8 harmonics + shaped unvoiced sinusoids at 1500 Hz)."""

    def __init__(self, n_mels=N_MEL, c4=96, c16=48, n_har=8):
        super().__init__()
        self.n_har = n_har
        self.f0_net = nn.Sequential(
            QConv1d(n_mels, 128, 5, padding=2),
            nn.SiLU(),
            QConv1d(128, 128, 5, padding=2),
            nn.SiLU(),
            QConv1d(128, 2, 3, padding=1),
        )
        self.proj4 = QConv1d(2 * n_har, c4, 4, stride=4)
        self.proj16 = QConv1d(2 * n_har, c16, 3, padding=1)

    def forward(self, mel, shift_st=None):
        with torch.amp.autocast('cuda', enabled=False):
            rf = self.f0_net(mel.float()).float()
        f0 = 60.0 + 440.0 * torch.sigmoid(rf[:, 0:1])
        uv = torch.sigmoid(rf[:, 1:2])
        w_v = (uv.detach() > 0.35).float() * uv.detach() + 0.04
        lf0_raw = torch.log2(f0.clamp(130.0, 380.0))
        num = F.avg_pool1d(F.pad(lf0_raw * w_v, (3, 3), mode='replicate'), 7, stride=1)
        den = F.avg_pool1d(F.pad(w_v, (3, 3), mode='replicate'), 7, stride=1).clamp(min=1e-3)
        lf0_loc = num / den
        lf0_sm = F.avg_pool1d(F.pad(0.50 * lf0_raw + 0.50 * lf0_loc, (2, 2), mode='replicate'), 5, stride=1)
        if shift_st is not None:
            lf0_sm = lf0_sm + shift_st.to(f0.dtype) * (1.0 / 12.0)
        f0 = torch.pow(2.0, lf0_sm)
        uv = F.avg_pool1d(F.pad(uv, (1, 1), mode='replicate'), 3, stride=1)
        f0_16 = F.interpolate(f0.detach(), scale_factor=16.0, mode='linear', align_corners=False)
        uv_16 = F.interpolate(uv.detach(), scale_factor=16.0, mode='linear', align_corners=False)
        cyc = torch.cumsum(f0_16 * (1.0 / (SR / ISTFT_HOP)), dim=-1)
        cyc = cyc - torch.floor(cyc)
        ph = (2.0 * math.pi) * cyc
        ks = torch.arange(1, self.n_har + 1, device=ph.device, dtype=ph.dtype)[None, :, None]
        amp = (1.0 / torch.sqrt(ks)) * (ks * f0_16 < 720.0).float()
        k_ph = ks * ph
        har = torch.cat([torch.sin(k_ph) * amp, torch.cos(k_ph) * amp], dim=1) * uv_16
        t_idx = torch.arange(ph.shape[-1], device=ph.device, dtype=ph.dtype)[None, None, :]
        ap_freqs = torch.tensor([1.41421, 1.73205, 2.23607, 2.64575, 3.14159, 3.60555, 4.12311, 4.58258], device=ph.device, dtype=ph.dtype)[None, :, None]
        ap = torch.cat([torch.sin(t_idx * ap_freqs), torch.cos(t_idx * (ap_freqs * 1.17))], dim=1)
        har = (har + (1.0 - uv_16) * 0.14 * ap).to(mel.dtype)
        return self.proj4(har), self.proj16(har), f0, uv


class EmbeddedVocoderV2(nn.Module):
    """16x-upsampling Source-Filter Hybrid iSTFTNet Vocoder with 1-op COLA iSTFT64 (warm-sliced from v5_c1 or v14_voc144)."""

    def __init__(self, ch=192, res_kernels=(3, 7, 11)):
        super().__init__()
        self.ch = ch
        self.n_kernels = len(res_kernels)
        c1, c2 = ch // 2, ch // 4
        self.pre = QConv1d(N_MEL, ch, 7, padding=3)
        self.src = VocHarmonicSource(N_MEL, c1, c2)
        self.up_acts = nn.ModuleList([Snake1d(ch), Snake1d(c1)])
        self.ups = nn.ModuleList([
            QConvTranspose1d(ch, c1, 8, 4, padding=2),
            QConvTranspose1d(c1, c2, 8, 4, padding=2),
        ])
        self.mrfs = nn.ModuleList([
            nn.ModuleList([VocResBlock1(c1, k) for k in res_kernels]),
            nn.ModuleList([VocResBlock1(c2, k) for k in res_kernels]),
        ])
        self.post_act = Snake1d(c2)
        self.post = nn.Conv1d(c2, ISTFT_N_FFT + 2, 7, padding=3)

        # Precompute 1-op COLA iSTFT64 kernel (66, 1, 64)
        n_fft, hop = ISTFT_N_FFT, ISTFT_HOP
        win = np.hanning(n_fft + 1)[:-1].astype(np.float32)
        k = np.arange(n_fft // 2 + 1, dtype=np.float32)[:, None]
        n = np.arange(n_fft, dtype=np.float32)[None, :]
        ang = 2.0 * np.pi * k * n / float(n_fft)
        c = np.full((n_fft // 2 + 1, 1), 2.0, dtype=np.float32)
        c[0, 0] = 1.0
        c[-1, 0] = 1.0
        w_cola = win[None, :] / 1.5
        k_re = (c * np.cos(ang) / float(n_fft)) * w_cola
        k_im = (-c * np.sin(ang) / float(n_fft)) * w_cola
        K = np.concatenate([k_re, k_im], axis=0)[:, None, :]
        self.register_buffer('K_istft', torch.from_numpy(K.astype(np.float32)), persistent=False)

    def warm_from_pretrained(self):
        """Warm-slice exact weights from /work/tts/runs/v14_voc144/voc.pt (for ch<=144) or /work/tts/runs/v5_c1/voc.pt."""
        p144 = '/work/tts/runs/v14_voc144/voc.pt'
        p320 = '/work/tts/runs/v5_c1/voc.pt'
        ck_path = p144 if (self.ch <= 144 and os.path.exists(p144)) else (p320 if os.path.exists(p320) else None)
        if not ck_path:
            return False
        try:
            from sakhiya.v2_vocoder import VocoderV2
            ck = torch.load(ck_path, map_location='cpu', weights_only=False)
            src_ch = 144 if ck_path == p144 else 320
            ref = VocoderV2(ch=src_ch, res_kernels=(3, 7, 11)).eval()
            ref.load_state_dict(ck.get('G_ema', ck.get('G', ck)), strict=False)
            ref.fuse_weight_norm()
            rsd = ref.state_dict()
            dsd = self.state_dict()
            with torch.no_grad():
                for k, dv in dsd.items():
                    if k in rsd:
                        sv = rsd[k].float()
                        sl = tuple(slice(0, min(a, b)) for a, b in zip(dv.shape, sv.shape))
                        dv[sl].copy_(sv[sl])
            return True
        except Exception as e:
            print('vocoder warm-slice warn:', e, flush=True)
            return False

    def forward(self, mel, shift_st=None):
        s4, s16, f0, uv = self.src(mel, shift_st=shift_st)
        x = self.pre(mel)
        for act, up, mrf, src_add in zip(self.up_acts, self.ups, self.mrfs, (s4, s16)):
            x = up(act(x)) + src_add
            x = sum(blk(x) for blk in mrf) / self.n_kernels
        h = self.post(self.post_act(x)).float()
        log_mag, ph = h.chunk(2, dim=1)
        mag = torch.exp(log_mag.clamp(min=-11.0, max=5.0))
        re = mag * torch.cos(ph)
        im = mag * torch.sin(ph)
        with torch.amp.autocast('cuda', enabled=False):
            z = torch.zeros_like(im[:, :1, :])
            im_sym = torch.cat([z, im[:, 1:-1, :], z], dim=1)
            spec_cat = torch.cat([re.float(), im_sym.float()], dim=1)
            p = (ISTFT_N_FFT - ISTFT_HOP) // 2
            return F.conv_transpose1d(spec_cat, self.K_istft.to(spec_cat.device), stride=ISTFT_HOP, padding=p)[:, 0, :]


class StudentV4(nn.Module):
    def __init__(self, d_e, L_e, u_e, ff_e, d_d, L_d, u_d, ff_d, d_p, L_p, v_ch=192):
        super().__init__()
        self.cfg = dict(d_e=d_e, L_e=L_e, u_e=u_e, ff_e=ff_e, d_d=d_d, L_d=L_d, u_d=u_d, ff_d=ff_d, d_p=d_p, L_p=L_p, v_ch=v_ch)
        self.emb = nn.Embedding(N_SYM, d_e)
        self.lang = nn.Embedding(N_LANG, d_e)
        self.enc = Stack(d_e, ff_e, L_e, u_e, k=5)
        self.dur = nn.Sequential(QConv1d(d_e, d_e // 2, 3, padding=1), nn.ReLU(), QConv1d(d_e // 2, 1, 1))
        self.proj = QConv1d(d_e, d_d, 1)
        self.pos = QConv1d(2, d_d, 1)
        self.p_in = QConv1d(d_d, d_p, 1)
        self.p_stack = Stack(d_p, 2, L_p, L_p)
        self.p_out = QConv1d(d_p, 2, 1)
        self.p_emb = QConv1d(2, d_d, 1)
        self.dec = Stack(d_d, ff_d, L_d, u_d, k=7)
        self.dec_ln = ChanLN(d_d)
        self.mel_head = nn.Conv1d(d_d, N_MEL, 1)
        self.voc = EmbeddedVocoderV2(ch=v_ch, res_kernels=(3, 7, 11))
        self.prior = nn.Conv1d(d_e, N_MEL, 1)
        self.register_buffer('f0_stats', torch.tensor([5.35, 0.25]))
        self.register_buffer('mel_stats', torch.tensor([-5.5028, 2.1869]))

    def encode(self, ids, lang, mask):
        x = (self.emb(ids) + self.lang(lang)[:, None, :]).transpose(1, 2) * mask
        h = self.enc(x, mask) * mask
        return h, self.dur(h * mask)[:, 0]

    def frame_feats(self, idx, dur):
        d = torch.gather(dur, 1, idx)
        start = torch.cumsum(dur, 1) - dur
        t = torch.arange(idx.shape[1], device=idx.device, dtype=torch.float32)[None]
        frac = (t - torch.gather(start, 1, idx) + 0.5) / d.clamp(min=1.0)
        return torch.stack([frac, torch.log(d + 1.0) / 3.0], 1)

    def decode_mel(self, xf, feats, pitch=None):
        x = xf + self.pos(feats)
        p = self.p_out(self.p_stack(self.p_in(x)))
        pe = torch.cat([p[:, :1], torch.sigmoid(p[:, 1:2])], 1) if pitch is None else pitch
        x = x + self.p_emb(pe)
        h_dec = self.dec_ln(self.dec(x))
        mel_pred = self.mel_head(h_dec).float()
        return mel_pred, p

    def decode(self, xf, feats, pitch=None, return_mel=False):
        mel_pred, p = self.decode_mel(xf, feats, pitch=pitch)
        wav = self.voc(mel_pred)
        if return_mel:
            return wav, p, mel_pred
        return wav, p

    @torch.no_grad()
    def infer(self, ids, lang, dur_scale, pitch_shift):
        L = ids.shape[1]
        mask = torch.ones(1, 1, L, device=ids.device)
        h, ld = self.encode(ids, lang, mask)
        dur = torch.clamp(torch.round((torch.exp(ld) - 1.0) * dur_scale[:, None]), min=2.0)
        ends = torch.cumsum(dur, 1)
        T = ends[0, -1].to(torch.int64)
        t = torch.arange(0, T, device=ids.device, dtype=torch.float32)
        idx = (t[None, :, None] >= ends[:, None, :]).to(torch.float32).sum(-1).to(torch.int64).clamp(max=L - 1)
        xh = self.proj(h)
        xf = torch.gather(xh, 2, idx[:, None, :].expand(-1, xh.shape[1], -1))
        feats = self.frame_feats(idx, dur)
        x = xf + self.pos(feats)
        p = self.p_out(self.p_stack(self.p_in(x)))
        vuv = torch.sigmoid(p[:, 1:2])
        pe = torch.cat([p[:, :1] + pitch_shift[:, None, None] * (vuv > 0.5).to(p.dtype), vuv], 1)
        x = x + self.p_emb(pe)
        mel_pred = self.mel_head(self.dec_ln(self.dec(x))).float()
        return self.voc(mel_pred, shift_st=pitch_shift[:, None, None])


def build(tier, warm_voc=True):
    m = StudentV4(**TIERS[tier])
    if warm_voc:
        m.voc.warm_from_pretrained()
    return m


def n_params(m, deployed=True):
    skip = ('prior.',) if deployed else ()
    return sum(p.numel() for n, p in m.named_parameters() if not n.startswith(skip))


if __name__ == '__main__':
    for t in ('micro', 'pico', 't10'):
        m = build(t, warm_voc=False)
        print(t, f'{n_params(m)/1e6:.3f}M deployed params')
