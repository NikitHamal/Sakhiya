"""Sakhiya v4 Unified Single-Stage Distilled Students (`student_v4.py`):
  - Sakhiya-V2-Micro (`micro`): <= 4.5 MB total INT8 ONNX (~3.78 MB), unifying `SakhiyaAM` (`sano`) + 16x `EmbeddedVocoderV2(ch=208)`
  - Sakhiya-V2-Pico  (`pico`):  1.0 to 1.4 MB total INT8 ONNX (~1.24 MB), unifying `SakhiyaPicoAM` (Weight-Tied Looped Universal
    Encoder & Velocity Decoder with 1-KB FiLM iteration adapters) + 16x `EmbeddedVocoderV2(ch=112)`
  - Sakhiya-V2-Compact (`t10`): ~8.9 MB total INT8 ONNX, unifying `SakhiyaAM` (`voca`) + 16x `EmbeddedVocoderV2(ch=320)`

Warm-initializes from `/work/tts/ck/v9_sano_am.pt` / `/work/tts/ck/v9_voca_am.pt` and `/work/tts/runs/v5_c1/voc.pt` /
`/work/tts/runs/v14_voc144/voc.pt`, then adapts to Sakhiya-Pro v4's studio-denoised 24 kHz Nepal-native corpus in Two Phases
(Phase A clean mel/duration/F0/prior + Phase B end-to-end Vocoder-in-the-Loop + Multi-Res Complex STFT + WavLM + MPD/MRD GAN + STE INT8 QAT).
"""
import math, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sakhiya.cfm import SakhiyaAM
from sakhiya.v15_pico import SakhiyaPicoAM
from sakhiya.symbols import SYM2ID, LANG2ID

SR, N_FFT, HOP, NB, N_MEL = 24000, 1024, 256, 513, 100
ISTFT_N_FFT, ISTFT_HOP = 64, 16

TIERS = {
    'micro': dict(arch='micro', d_enc=144, L_enc=5, ff_enc=224, d_vel=160, nb_vel=7, h_dur=96, v_ch=144),
    'pico':  dict(arch='pico',  d_enc=96,  L_enc=4, ff_enc=112, d_vel=112, nb_vel=5, h_dur=64, v_ch=112),
    't10':   dict(arch='t10',   d_enc=144, L_enc=5, ff_enc=224, d_vel=160, nb_vel=7, h_dur=96, v_ch=320),
    't5':    dict(arch='micro', d_enc=144, L_enc=5, ff_enc=224, d_vel=160, nb_vel=7, h_dur=96, v_ch=144),
    't1':    dict(arch='pico',  d_enc=96,  L_enc=4, ff_enc=112, d_vel=112, nb_vel=5, h_dur=64, v_ch=112),
}


def fq(w, axis=0):
    red = [i for i in range(w.dim()) if i != axis]
    s = w.detach().abs().amax(dim=red, keepdim=True).clamp(min=1e-7) / 127.0
    return w + ((w / s).round().clamp(-127, 127) * s - w).detach()


class QConv1d(nn.Conv1d):
    qat = False

    def forward(self, x):
        use_q = self.qat and self.training and (self.weight.numel() >= 512) and (self.groups == 1)
        w = fq(self.weight, axis=0) if use_q else self.weight
        return self._conv_forward(x, w, self.bias)


class QConvTranspose1d(nn.ConvTranspose1d):
    qat = False

    def forward(self, x, output_size=None):
        use_q = self.qat and self.training and (self.weight.numel() >= 512) and (self.groups == 1)
        w = fq(self.weight, axis=1) if use_q else self.weight
        return F.conv_transpose1d(x, w, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)


def set_qat(model, flag):
    if hasattr(model, 'am') and hasattr(model.am, 'set_qat'):
        model.am.set_qat(flag)
    for name, m in model.named_modules():
        if isinstance(m, (QConv1d, QConvTranspose1d)):
            m.qat = False if 'f0_net' in name else bool(flag)


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
            t_fr = torch.arange(lf0_sm.shape[-1], device=lf0_sm.device, dtype=lf0_sm.dtype)[None, None, :]
            gamaka_st = 0.12 * torch.sin((2.0 * math.pi * 5.2 / (SR / HOP)) * t_fr) * (uv.detach() > 0.45).to(lf0_sm.dtype) * (shift_st.abs() > 0.02).to(lf0_sm.dtype)
            lf0_sm = lf0_sm + (shift_st.to(f0.dtype) + gamaka_st) * (1.0 / 12.0)
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
    def __init__(self, ch=208, res_kernels=(3, 7, 11)):
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
            w_out = F.conv_transpose1d(spec_cat, self.K_istft.to(spec_cat.device), stride=ISTFT_HOP, padding=p)[:, 0, :]
            if not self.training:
                pk = w_out.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
                w_out = (w_out * (0.85 / pk)).clamp(-0.98, 0.98)
            return w_out


def _calibrate_am(am, is_pico=False, pre_baked=True):
    with torch.no_grad():
        if is_pico or pre_baked:
            w0 = am.enc.sty.weight[0].clone()
            w2 = am.enc.sty.weight[2].clone()
            am.enc.sty.weight[1].copy_(0.45 * w0 + 0.55 * w2)
            am.enc.sty.weight[3].copy_(0.60 * w0 + 0.40 * w2)
            am.enc.sty.weight[4].copy_(0.20 * w0 + 0.80 * w2)
        else:
            w_sty = am.enc.sty.weight.clone()
            if w_sty.shape[0] > 14:
                w0 = 0.65 * w_sty[10] + 0.35 * w_sty[3]
                w2 = w0 + 1.15 * ((0.75 * w_sty[3] + 0.25 * w_sty[2]) - w0)
                w5 = 0.60 * w0 + 0.40 * (0.55 * w_sty[3] + 0.45 * w_sty[14])
                am.enc.sty.weight[0].copy_(w0)
                am.enc.sty.weight[1].copy_(0.45 * w0 + 0.55 * w2)
                am.enc.sty.weight[2].copy_(w2)
                am.enc.sty.weight[3].copy_(0.60 * w0 + 0.40 * w2)
                am.enc.sty.weight[4].copy_(0.20 * w0 + 0.80 * w2)
                am.enc.sty.weight[5].copy_(w5)
            w_e = am.enc.emb.weight
            id_l, id_r = SYM2ID['l'], SYM2ID['ɽ']
            w_e[id_r].copy_(0.60 * w_e[id_l] + 0.40 * w_e[id_r])
            nas_d = 0.5 * (w_e[SYM2ID['ũ']] - w_e[SYM2ID['u']]) + 0.5 * (w_e[SYM2ID['ã']] - w_e[SYM2ID['a']])
            w_e[SYM2ID['õ']].copy_(0.25 * w_e[SYM2ID['õ']] + 0.75 * (w_e[SYM2ID['o']] + 1.0 * nas_d + 0.2 * (w_e[SYM2ID['ŋ']] - w_e[SYM2ID['o']])))
            sa_id = LANG2ID['sa']
            am.enc.lang.weight[sa_id].copy_(0.94 * am.enc.lang.weight[0] + 0.06 * am.enc.lang.weight[sa_id])
            for lid in range(am.enc.lang.weight.shape[0]):
                if lid != sa_id:
                    am.enc.lang.weight[lid].copy_(am.enc.lang.weight[0])


class StudentV4(nn.Module):
    def __init__(self, arch='micro', d_enc=144, L_enc=5, ff_enc=224, d_vel=160, nb_vel=7, h_dur=96, v_ch=144, **kw):
        super().__init__()
        self.cfg = dict(arch=arch, d_enc=d_enc, L_enc=L_enc, ff_enc=ff_enc, d_vel=d_vel, nb_vel=nb_vel, h_dur=h_dur, v_ch=v_ch)
        self.is_pico = (arch == 'pico')
        if self.is_pico:
            self.am = SakhiyaPicoAM(d_enc=d_enc, L_enc=L_enc, ff_enc=ff_enc, d_vel=d_vel, nb_vel=nb_vel, h_dur=h_dur)
        else:
            # Prune unused embedding rows on micro/t10 so only active speaker 0, langs 0..5, styles 0..5 are stored in ONNX
            self.am = SakhiyaAM(d_enc=d_enc, L_enc=L_enc, ff_enc=ff_enc, d_vel=d_vel, nb_vel=nb_vel, h_dur=h_dur, n_spk=2, n_lang=6, n_sty=6)
            if hasattr(self.am.enc, 'spk_adv'):
                delattr(self.am.enc, 'spk_adv')
        self.voc = EmbeddedVocoderV2(ch=v_ch, res_kernels=(3, 7, 11))
        self.register_buffer('f0_stats', torch.tensor([5.3706, 0.2200]))
        self.register_buffer('mel_stats', torch.tensor([-6.1761, 2.4932]))

    def warm_from_pretrained(self):
        sano_p = '/work/tts/ck/v9_sano_am.pt'
        voca_p = '/work/tts/ck/v9_voca_am.pt'
        ck_p = voca_p if (self.cfg['d_enc'] == 144 and os.path.exists(voca_p)) else (sano_p if os.path.exists(sano_p) else None)
        if ck_p:
            try:
                ck = torch.load(ck_p, map_location='cpu', weights_only=False)
                pre_baked = bool(ck.get('pre_baked_v25', True))
                raw_sd = {k: v.float() for k, v in ck.get('ema', ck.get('net', ck)).items() if not k.startswith(('ctc_rec.', 'enc.spk_adv.'))}
                full_am = SakhiyaAM(
                    d_enc=self.cfg['d_enc'], L_enc=self.cfg['L_enc'],
                    ff_enc=144 if self.is_pico else self.cfg['ff_enc'],
                    d_vel=self.cfg['d_vel'], nb_vel=self.cfg['nb_vel'], h_dur=self.cfg['h_dur'],
                ).eval()
                if 'mean' in ck and 'std' in ck:
                    full_am.stats.copy_(torch.tensor([float(ck['mean']), float(ck['std'])]))
                full_am.load_state_dict(raw_sd, strict=False)
                _calibrate_am(full_am, is_pico=False, pre_baked=pre_baked)
                cal_sd = full_am.state_dict()
                if self.is_pico:
                    self.am.warm_from_sano(cal_sd)
                    self.am.stats.copy_(full_am.stats)
                    _calibrate_am(self.am, is_pico=True, pre_baked=True)
                else:
                    dsd = self.am.state_dict()
                    with torch.no_grad():
                        for k, dv in dsd.items():
                            if k in cal_sd:
                                sv = cal_sd[k].float()
                                sl = tuple(slice(0, min(a, b)) for a, b in zip(dv.shape, sv.shape))
                                dv[sl].copy_(sv[sl])
                        self.am.stats.copy_(full_am.stats)
                self.mel_stats.copy_(self.am.stats)
            except Exception as e:
                print('AM warm-start warn:', e, flush=True)
        self.voc.warm_from_pretrained()
        return self

    def forward_am_train(self, ids, lang, im, path):
        """Training forward pass: returns mu [B,100,L], ld [B,L], mel_pred [B,100,T]."""
        B, L = ids.shape
        T = path.shape[2]
        spk0 = torch.zeros((B,), dtype=torch.long, device=ids.device)
        sty0 = torch.zeros((B,), dtype=torch.long, device=ids.device)
        studio1 = torch.ones((B,), dtype=torch.long, device=ids.device)
        lang_eff = torch.where(lang == 1, torch.full_like(lang, 4), torch.zeros_like(lang))
        im_bool = im[:, 0] > 0.5
        h, mu, g = self.am.enc(ids, spk0, lang_eff, sty0, studio1, im_bool)
        ld = self.am.dur.pred_logdur(ids, h, im_bool)
        mu_t = mu.transpose(1, 2).float()
        mu_y = torch.matmul(mu_t, path)
        t0 = torch.zeros((B,), device=ids.device, dtype=torch.float32)
        m_frame = (path.sum(1, keepdim=True) > 0).to(mu_y.dtype)
        xt = mu_y + self.am.vel(mu_y, t0, mu_y, g, m_frame)
        mel_pred = xt * self.am.stats[1] + self.am.stats[0]
        return mu_t, ld, mel_pred

    @torch.no_grad()
    def infer(self, ids, lang, dur_scale, pitch_shift):
        B, L = ids.shape
        im = torch.ones((B, L), dtype=torch.bool, device=ids.device)
        spk0 = torch.zeros((B,), dtype=torch.long, device=ids.device)
        sty0 = torch.where(lang == 1, torch.full_like(lang, 5), torch.zeros_like(lang))
        studio1 = torch.ones((B,), dtype=torch.long, device=ids.device)
        lang_eff = torch.where(lang == 1, torch.full_like(lang, 4), torch.zeros_like(lang))
        h, mu, g = self.am.enc(ids, spk0, lang_eff, sty0, studio1, im)
        x_d = self.am.dur.pred_logdur(ids, h, im)
        z_tok = torch.zeros((B, L), dtype=torch.float32, device=ids.device)
        ds_tok = dur_scale.reshape(B, -1) + z_tok
        dur = torch.clamp(torch.round(torch.exp(x_d) * ds_tok), min=3.0, max=52.0)
        # Protect sentence-final tokens from frame collapse
        pos = torch.arange(L, device=ids.device, dtype=torch.float32)[None, :]
        is_tail = ((pos >= (float(L) - 3.0)) & (ids > 6)).float()
        dur = torch.maximum(dur, is_tail * 5.0 + (1.0 - is_tail) * 2.0)
        # Sanskrit/Vedic Chandas safeguards: resonant Pranava Om (o + m >= 36 + 18 frames) & light breathy visarga 'h' (<= 5 frames)
        is_sa = (lang == 1).float()[:, None]
        id_o, id_m, id_h = SYM2ID['o'], SYM2ID['m'], SYM2ID['h']
        next_ids = torch.cat([ids[:, 1:], ids[:, -1:]], dim=1)
        prev_ids = torch.cat([ids[:, :1], ids[:, :-1]], dim=1)
        is_om_o = ((ids == id_o) & (next_ids == id_m)).float() * is_sa
        is_om_m = ((ids == id_m) & (prev_ids == id_o)).float() * is_sa
        dur = torch.maximum(dur, is_om_o * 36.0 + is_om_m * 18.0)
        is_final_h = ((ids == id_h) & (next_ids <= 6)).float() * is_sa
        dur = torch.where(is_final_h > 0.5, torch.clamp(dur, min=3.0, max=5.0), dur)
        ends = torch.cumsum(dur, 1)
        T = ends[0, -1].to(torch.int64)
        t = torch.arange(0, T, device=ids.device, dtype=torch.float32)
        idx = (t[None, :, None] >= ends[:, None, :]).to(torch.float32).sum(-1).to(torch.int64).clamp(max=L - 1)
        mu_t = mu.transpose(1, 2)
        mu_y = torch.gather(mu_t, 2, idx[:, None, :].expand(-1, N_MEL, -1))
        m_frame = torch.ones((B, 1, T), dtype=mu_y.dtype, device=ids.device)
        t0 = torch.zeros((B,), device=ids.device, dtype=torch.float32)
        xt = mu_y + self.am.vel(mu_y, t0, mu_y, g, m_frame)
        mel = xt * self.am.stats[1] + self.am.stats[0]
        ps_tok = pitch_shift.reshape(B, -1) + z_tok
        ps_fr = torch.gather(ps_tok, 1, idx)[:, None, :]
        ps_sm = F.avg_pool1d(F.pad(ps_fr, (3, 3), mode='replicate'), 7, stride=1)
        return self.voc(mel, shift_st=ps_sm)


def build(tier, warm_voc=True):
    m = StudentV4(**TIERS[tier])
    if warm_voc:
        m.warm_from_pretrained()
    return m


def n_params(m, deployed=True):
    return sum(p.numel() for p in m.parameters())


if __name__ == '__main__':
    for t in ('micro', 'pico', 't10'):
        m = build(t, warm_voc=False)
        print(t, f'{n_params(m)/1e6:.3f}M deployed params')
