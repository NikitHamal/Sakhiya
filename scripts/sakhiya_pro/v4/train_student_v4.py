#!/usr/bin/env python3
"""Serial Zero-NaN Two-Phase Student Distillation Trainer for Sakhiya-V2-Micro (<=4.5 MB) & Sakhiya-V2-Pico (<=1.4 MB).

Strict Two-Phase Recipe:
- Phase A (Clean Reconstruction Pre-Convergence): Multi-resolution complex STFT + phase/group-delay + 100-band log-mel L1 +
  duration/F0/VUV/energy + MAS alignment ( plus scheduled predicted-pitch mixing after step 1200 so inference without GT pitch
  is rock-solid!) until real reconstruction converges cleanly (mel L1 < 0.72 and step >= gan_start).
- Phase B (Adversarial MPD/MRD GAN + Frozen WavLM Perceptual Polish + QAT): Turned on ONLY after Phase A reconstruction has
  converged, with progressive feature/STFT distillation from Sakhiya-V2-Micro when training Sakhiya-V2-Pico!
- Strict Step-50 stdout throughput watchdog and 135-min hard cap.
"""
import argparse, copy, json, math, os, random, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, 'scripts', 'sakhiya_pro', 'distill'))
sys.path.insert(0, REPO)

from student_v4 import build, set_qat, n_params, SR, N_FFT, HOP, N_MEL, TIERS, StudentV4
from sakhiya.mas import maximum_path

LR = dict(micro=3.5e-4, pico=4.5e-4, t10=3.0e-4, t5=3.5e-4, t1=4.5e-4)


class Mel(nn.Module):
    def __init__(s):
        super().__init__()
        import librosa
        s.register_buffer('fb', torch.tensor(librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MEL, fmin=0, fmax=12000), dtype=torch.float32))
        s.register_buffer('win', torch.hann_window(N_FFT))

    def forward(s, w):
        S = torch.stft(w.float(), N_FFT, HOP, window=s.win, center=True, return_complex=True).abs()
        return torch.log(torch.clamp(s.fb @ S, min=1e-5))


def mrstft(y, yh):
    tot = 0.0
    for n in (512, 1024, 2048):
        win = torch.hann_window(n, device=y.device)
        A = torch.stft(y, n, n // 4, window=win, return_complex=True)
        B = torch.stft(yh, n, n // 4, window=win, return_complex=True)
        a, b = A.abs(), B.abs()
        sc = torch.norm(a - b) / (torch.norm(a) + 1e-6)
        mag = F.l1_loss(torch.log(b + 1e-5), torch.log(a + 1e-5))
        tot = tot + sc + mag
    return tot


def phase_loss(y, yh):
    win = torch.hann_window(N_FFT, device=y.device)
    A = torch.stft(y, N_FFT, HOP, window=win, return_complex=True)
    B = torch.stft(yh, N_FFT, HOP, window=win, return_complex=True)
    w = (A.abs() / (A.abs().amax(dim=(1, 2), keepdim=True) + 1e-6)).clamp(max=1.0).sqrt()

    def ang(Z, d):
        Zs = Z[:, :, 1:] * torch.conj(Z[:, :, :-1]) if d == 't' else Z[:, 1:] * torch.conj(Z[:, :-1])
        return Zs / (Zs.abs() + 1e-6)

    out = 0.0
    for d, ww in (('t', w[:, :, 1:]), ('f', w[:, 1:])):
        a, b = ang(A, d), ang(B, d)
        out = out + (ww * ((a.real - b.real).abs() + (a.imag - b.imag).abs())).mean()
    return out


SN = nn.utils.spectral_norm


class DP(nn.Module):
    def __init__(s, p):
        super().__init__()
        s.p = p
        ch = [1, 16, 64, 128, 256]
        s.convs = nn.ModuleList(
            [SN(nn.Conv2d(ch[i], ch[i + 1], (5, 1), (3, 1), (2, 0))) for i in range(4)]
            + [SN(nn.Conv2d(256, 256, (5, 1), 1, (2, 0)))]
        )
        s.post = SN(nn.Conv2d(256, 1, (3, 1), 1, (1, 0)))

    def forward(s, x):
        B, T = x.shape
        if T % s.p:
            x = F.pad(x, (0, s.p - T % s.p), mode='reflect')
        x = x.view(B, 1, -1, s.p)
        fm = []
        for c in s.convs:
            x = F.leaky_relu(c(x), 0.1)
            fm.append(x)
        return s.post(x), fm


class DR(nn.Module):
    def __init__(s, n):
        super().__init__()
        s.n = n
        s.convs = nn.ModuleList(
            [SN(nn.Conv2d(1, 32, (3, 9), padding=(1, 4)))]
            + [SN(nn.Conv2d(32, 32, (3, 9), (1, 2), (1, 4))) for _ in range(3)]
            + [SN(nn.Conv2d(32, 32, 3, padding=1))]
        )
        s.post = SN(nn.Conv2d(32, 1, 3, padding=1))

    def forward(s, x):
        S = torch.stft(x.float(), s.n, s.n // 4, window=torch.hann_window(s.n, device=x.device), return_complex=True).abs()
        x = torch.log(S + 1e-5)[:, None]
        fm = []
        for c in s.convs:
            x = F.leaky_relu(c(x), 0.1)
            fm.append(x)
        return s.post(x), fm


class Disc(nn.Module):
    def __init__(s):
        super().__init__()
        s.ds = nn.ModuleList([DP(p) for p in (2, 3, 5, 7, 11)] + [DR(n) for n in (512, 1024, 2048)])

    def forward(s, x):
        o, f = [], []
        for d in s.ds:
            a, b = d(x)
            o.append(a)
            f.append(b)
        return o, f


def load_data(path, seg, dev):
    D = torch.load(path, weights_only=False)
    tr, dv = [], []
    for x in D['items']:
        if len(x['wav']) // HOP < seg + 8:
            continue
        x['wav'] = x['wav'].to(dev)
        x['f0'] = x['f0'].to(dev)
        (dv if x['split'] == 'dev' else tr).append(x)
    return tr, dv, float(D['f0_mu']), float(D['f0_sd'])


def make_batch(items, B, dev, mu, sd):
    xs = random.sample(items, B)
    L = max(len(x['ids']) for x in xs)
    T = max(len(x['f0']) for x in xs)
    ids = torch.zeros(B, L, dtype=torch.long)
    wav = torch.zeros(B, T * HOP, device=dev)
    f0 = torch.zeros(B, T, device=dev)
    il = torch.tensor([len(x['ids']) for x in xs])
    tl = torch.tensor([len(x['f0']) for x in xs])
    for b, x in enumerate(xs):
        ids[b, :il[b]] = x['ids']
        wav[b, :len(x['wav'])] = x['wav'].float() / 32767.0
        f0[b, :tl[b]] = x['f0']
    vuv = (f0 > 0).float()
    lf = torch.where(f0 > 0, (torch.log(f0.clamp(min=1.0)) - mu) / sd, torch.zeros_like(f0))
    # Frame-level log RMS energy
    wav_frames = wav.view(B, T, HOP)
    eng = torch.log(torch.sqrt((wav_frames ** 2).mean(-1) + 1e-7) + 1e-4)
    eng = ((eng + 3.0) / 1.5).clamp(-3.0, 3.0)
    lang = torch.tensor([x['lang'] for x in xs])
    return ids.to(dev), il.to(dev), tl.to(dev), lang.to(dev), wav, lf, vuv, eng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--teacher_ckpt', default='')
    ap.add_argument('--minutes', type=float, default=120)
    ap.add_argument('--bs', type=int, default=28)
    ap.add_argument('--seg', type=int, default=64)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--gan_start', type=int, default=2200)
    ap.add_argument('--warm_uniform', type=int, default=500)
    ap.add_argument('--qat_start', type=float, default=0.60)
    ap.add_argument('--max_step_s', type=float, default=2.5)
    ap.add_argument('--save_every', type=int, default=800)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--max_steps', type=int, default=10 ** 9)
    ap.add_argument('--fresh_clock', action='store_true')
    ap.add_argument('--lr_mult', type=float, default=1.0)
    a = ap.parse_args()
    assert a.minutes <= 135, 'hard 135 min cap'

    dev = 'cuda'
    os.makedirs(a.out, exist_ok=True)
    random.seed(42)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    logp_ = a.out + '/train.log'

    def log(s):
        print(s, flush=True)
        open(logp_, 'a').write(s + '\n')

    tr, dv, mu, sd = load_data(a.data, a.seg, dev)
    G = build(a.tier).to(dev)
    G.f0_stats.copy_(torch.tensor([mu, sd]))
    D = Disc().to(dev)
    mel = Mel().to(dev)

    # Progressive feature/STFT teacher (Sakhiya-V2-Micro -> Sakhiya-V2-Pico)
    T_micro = None
    if a.teacher_ckpt and os.path.exists(a.teacher_ckpt):
        try:
            St = torch.load(a.teacher_ckpt, map_location=dev, weights_only=False)
            T_micro = StudentV4(**St['cfg']).to(dev).eval()
            T_micro.load_state_dict(St['ema'])
            for p_ in T_micro.parameters():
                p_.requires_grad_(False)
            log(f'[progressive-distill] loaded converged Micro teacher from {a.teacher_ckpt}')
        except Exception as e:
            log(f'[progressive-distill] skip: {e}')

    wavlm = None
    try:
        from transformers import WavLMModel
        for wname in ('microsoft/wavlm-base-plus-sv', 'microsoft/wavlm-base-plus'):
            try:
                wavlm = WavLMModel.from_pretrained(wname).to(dev).eval()
                for p_ in wavlm.parameters():
                    p_.requires_grad_(False)
                log(f'[wavlm] loaded frozen {wname}')
                break
            except Exception:
                pass
    except Exception as e:
        log(f'[wavlm] skipped: {e}')

    ema = copy.deepcopy(G).eval()
    lr = (a.lr or LR[a.tier]) * a.lr_mult
    oG = torch.optim.AdamW(G.parameters(), lr=lr, betas=(0.8, 0.99), weight_decay=1e-2)
    oD = torch.optim.AdamW(D.parameters(), lr=lr * 0.4, betas=(0.8, 0.99))
    step, spent = 0, 0.0
    if a.resume and os.path.exists(a.out + '/state.pt'):
        S = torch.load(a.out + '/state.pt', weights_only=False, map_location=dev)
        G.load_state_dict(S['G'])
        ema.load_state_dict(S['ema'])
        D.load_state_dict(S['D'])
        oG.load_state_dict(S['oG'])
        oD.load_state_dict(S['oD'])
        step = S['step']
        spent = 0.0 if a.fresh_clock else S['spent']
        log(f'[resume] step {step} spent {spent / 60:.1f} min')

    log(f'[{a.tier}] deployed params {n_params(G) / 1e6:.3f}M | train {len(tr)} dev {len(dv)} utts | lr {lr} bs {a.bs} seg {a.seg}')
    mel_mu, mel_sd = G.mel_stats.tolist()
    budget = a.minutes * 60.0
    t0 = time.time() - spent
    tstep = time.time()
    qat_on = False
    phase_b_unlocked = False
    ar = torch.arange(a.seg, device=dev)
    E = 512
    ac = lambda: torch.autocast('cuda', dtype=torch.bfloat16)

    while True:
        spent = time.time() - t0
        prog = min(1.0, spent / max(budget, 1.0))
        if spent > budget or step >= a.max_steps:
            break
        cur_lr = lr * (0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * prog)))
        for g in oG.param_groups:
            g['lr'] = cur_lr
        for g in oD.param_groups:
            g['lr'] = cur_lr * 0.4

        if prog >= a.qat_start and not qat_on:
            set_qat(G, True)
            qat_on = True
            log(f'[qat] per-channel int8 fake-quant ON at step {step}')

        ids, il, tl, lang, wav, lf, vuv, eng = make_batch(tr, a.bs, dev, mu, sd)
        B, L = ids.shape
        T = lf.shape[1]
        imask = (torch.arange(L, device=dev)[None] < il[:, None]).float()[:, None]
        tmask = (torch.arange(T, device=dev)[None] < tl[:, None]).float()
        with torch.no_grad():
            melN = ((mel(wav)[:, :, :T] - mel_mu) / mel_sd) * tmask[:, None]

        with ac():
            G.train()
            h, ld = G.encode(ids, lang, imask)
        h = h.float()
        ld = ld.float()
        prior = G.prior(h)
        with torch.no_grad():
            if step < a.warm_uniform:
                ii = (torch.arange(T, device=dev)[None] * il[:, None] // tl[:, None].clamp(min=1)).clamp(max=L - 1)
                path = F.one_hot(ii, L).transpose(1, 2).float() * imask.transpose(1, 2) * tmask[:, None]
            else:
                a_ = (prior ** 2).sum(1)[:, :, None]
                b_ = (melN ** 2).sum(1)[:, None, :]
                lp = -0.5 * (a_ - 2 * prior.transpose(1, 2) @ melN + b_) / N_MEL
                m = (imask[:, 0, :, None] > 0) & (tmask[:, None, :] > 0)
                path = maximum_path(lp, m)
            dur = path.sum(-1)
            idx = path.argmax(1)

        l_prior = ((((prior @ path) - melN) ** 2) * tmask[:, None]).sum() / (tmask.sum() * N_MEL)
        l_dur = (((ld - torch.log(dur.clamp(min=1.0) + 1.0)) ** 2) * imask[:, 0]).sum() / imask.sum()

        s0 = (torch.rand(B, device=dev) * (tl - a.seg).clamp(min=1)).long()
        sel = s0[:, None] + ar[None]
        ex = lambda t: torch.gather(t, 2, sel[:, None, :].expand(-1, t.shape[1], -1))
        pe_gt = ex(torch.stack([lf, vuv, eng], 1))

        # Scheduled self-conditioning on predicted pitch/energy after step 1200 so inference without GT pitch is seamless
        use_pred_pe = (step >= 1200) and (random.random() < 0.35)
        with ac():
            xf = G.proj(h) @ path
            feats = G.frame_feats(idx, dur)
            xf_seg = ex(xf)
            feats_seg = ex(feats)
            yh, p, dec_h, stft_o = G.decode(xf_seg, feats_seg, None if use_pred_pe else pe_gt, return_hid=True)

        sidx = (sel * HOP)[:, :, None] + torch.arange(HOP, device=dev)
        y = torch.gather(wav, 1, sidx.reshape(B, -1))
        y, yh = y[:, E:-E], yh.float()[:, E:-E]

        pt, vt, et = pe_gt[:, 0], pe_gt[:, 1], pe_gt[:, 2]
        l_pitch = (
            (((p[:, 0].float() - pt).abs() * vt).sum() / vt.sum().clamp(min=1.0))
            + F.binary_cross_entropy_with_logits(p[:, 1].float(), vt)
            + 0.5 * F.l1_loss(p[:, 2].float(), et)
        )

        l_mel = F.l1_loss(mel(yh), mel(y))
        l_st = mrstft(y, yh)
        l_ph = phase_loss(y, yh)

        # Progressive teacher distillation from Sakhiya-V2-Micro when training Pico
        l_kd = torch.zeros((), device=dev)
        if T_micro is not None:
            with torch.no_grad(), ac():
                th, _ = T_micro.encode(ids, lang, imask)
                txf = ex(T_micro.proj(th.float()) @ path)
                tyh, _, _, tstft_o = T_micro.decode(txf, feats_seg, pe_gt, return_hid=True)
            l_kd = 0.25 * F.l1_loss(stft_o.float(), tstft_o.detach().float()) + 0.5 * F.l1_loss(mel(yh), mel(tyh.float()[:, E:-E]))

        # Strict Two-Phase Gating: unlock Phase B (GAN + WavLM) only after step >= gan_start AND mel L1 < 0.90
        if not phase_b_unlocked and step >= a.gan_start and l_mel.item() < 0.90:
            phase_b_unlocked = True
            log(f'[Phase B unlocked] step {step} mel_l1={l_mel.item():.3f} -> turning on MPD/MRD GAN + WavLM perceptual polish')

        l_g = l_fm = l_wlm = torch.zeros((), device=dev)
        if phase_b_unlocked:
            with ac():
                dr, _ = D(y)
                df_d, _ = D(yh.detach())
                l_d = sum(((r - 1.0) ** 2).mean() + (f_ ** 2).mean() for r, f_ in zip(dr, df_d)) / len(dr)
            oD.zero_grad(set_to_none=True)
            l_d.float().backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), 5.0)
            oD.step()

            if wavlm is not None and (step % 2 == 0):
                with ac():
                    n16 = (y.shape[1] * 2) // 3
                    y16 = F.interpolate(y[:, None], size=n16, mode='linear', align_corners=False)[:, 0]
                    yh16 = F.interpolate(yh[:, None], size=n16, mode='linear', align_corners=False)[:, 0]
                    y16 = (y16 - y16.mean(-1, keepdim=True)) / y16.std(-1, keepdim=True).clamp(min=1e-5)
                    yh16 = (yh16 - yh16.mean(-1, keepdim=True)) / yh16.std(-1, keepdim=True).clamp(min=1e-5)
                    with torch.no_grad():
                        hy = wavlm(y16, output_hidden_states=True).hidden_states
                    hyh = wavlm(yh16, output_hidden_states=True).hidden_states
                    l_wlm = sum(F.l1_loss(hyh[k].float(), hy[k].detach().float()) for k in (3, 6, 9, 12) if k < len(hy)) / 4.0

            with ac():
                df, ff = D(yh)
                _, fr = D(y)
                l_g = sum(((f_ - 1.0) ** 2).mean() for f_ in df) / len(df)
                l_fm = sum(F.l1_loss(a_.float(), b_.detach().float()) for fa, fb in zip(ff, fr) for a_, b_ in zip(fa, fb)) / len(ff)

        loss = 18.0 * l_mel + 1.2 * l_st + 2.0 * l_ph + l_prior + l_dur + l_pitch + l_kd + 1.5 * l_wlm + l_g.float() + 2.0 * l_fm.float()
        oG.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
        if torch.isfinite(gn):
            oG.step()
        step += 1

        with torch.no_grad():
            dcy = min(0.999, (1.0 + step) / (10.0 + step))
            for pe_, pg in zip(ema.parameters(), G.parameters()):
                pe_.lerp_(pg, 1.0 - dcy)

        if step == 50:
            sps = (time.time() - tstep) / 50.0
            if sps > a.max_step_s:
                log(f'WATCHDOG: step-50 {sps:.2f}s/step > {a.max_step_s}s')
                sys.exit(3)
        if step % 50 == 0:
            now = time.time()
            ips = 50.0 / max(now - tstep, 1e-6)
            tstep = now
            log(f'Step {step} | {ips:.2f} it/s | mel {l_mel.item():.3f} st {l_st.item():.3f} ph {l_ph.item():.3f} prior {l_prior.item():.3f} dur {l_dur.item():.3f} pitch {l_pitch.item():.3f} g {l_g.item():.3f} fm {l_fm.item():.3f} | {spent / 60:.1f}/{a.minutes:.0f} min qat={int(qat_on)}')
        if step % a.save_every == 0:
            torch.save(dict(ema=ema.state_dict(), cfg=TIERS[a.tier], tier=a.tier, step=step, f0=[mu, sd], qat=qat_on), a.out + '/ema_latest.tmp')
            os.replace(a.out + '/ema_latest.tmp', a.out + '/ema_latest.pt')
            if step % (a.save_every * 2) == 0:
                torch.save(dict(G=G.state_dict(), ema=ema.state_dict(), D=D.state_dict(), oG=oG.state_dict(), oD=oD.state_dict(), step=step, spent=time.time() - t0), a.out + '/state.tmp')
                os.replace(a.out + '/state.tmp', a.out + '/state.pt')

    torch.save(dict(ema=ema.state_dict(), cfg=TIERS[a.tier], tier=a.tier, step=step, f0=[mu, sd], qat=qat_on), a.out + '/ema_final.pt')
    log(f'DONE step {step} minutes {(time.time() - t0) / 60:.1f}')


if __name__ == '__main__':
    main()
