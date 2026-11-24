#!/usr/bin/env python3
"""Phase-Locked Teacher-Guided Complex-iSTFT + MR-STFT + MPD/MRD GAN Distillation for 144-ch Vocoders
(`scripts/sakhiya_pro/distill2/distill_voc144.py`).

Solves the root cause of robotic/shuttering audio in Micro (`t5`) and Pico (`t1`):
1. Copies `src.f0_net` directly from the converged 320-ch Pro-v3 vocoder (`/work/tts/ck/v3_voc320.pt`) and snaps/freezes it,
   so `voc144` and `voc144_t1` share 100% bit-exact fundamental frequency `f0(t)`, `uv(t)`, and continuous-phase harmonic
   excitation `har(t)` with `v3_voc320` at every sample.
2. For `--svd_t1`, applies the exact `r_voc_k7=50, r_voc_k11=31` SVD factorization BEFORE training so all SVD conv layers are
   jointly optimized end-to-end (eliminating post-hoc SVD truncation degradation and clicks).
3. Trains on a 50/50 mix of ground-truth Pro-v3 mels (`xm`) and predicted student/teacher mels (from `v3_rectified_cache.pt` +
   smoothed mels) using:
   - Sample-synchronous complex iSTFT spectrum & instantaneous-frequency matching against `v3_voc320(mel_in)`
   - Direct phase-locked waveform L1 + Multi-Resolution Complex STFT loss against both `v3_voc320` and Pro-v3 GT audio
   - 100-band log-mel L1 loss + `DiscV2` MPD/MRD hinge GAN adversarial & feature-matching loss
   - STE Per-Channel INT8 QAT in the second half of training so INT8 ONNX export is lossless.
"""
import argparse, copy, math, os, random, sys, time
import torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(HERE, '..', 'distill')
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
for p in (DIST, REPO, '/work/sp'):
    if p not in sys.path:
        sys.path.insert(0, p)

from sakhiya.v2_vocoder import VocoderV2, mel_spec_24k, ola64, HOP  # noqa: E402
from scripts.train_v2_voc import DiscV2, mr_complex_stft_loss  # noqa: E402
from student import _make_svd_seq, snap_f0_int8, ste_int8_weight  # noqa: E402
from train_student import spec_phase_gd_loss  # noqa: E402


def vin(mel, comb_gain=0.10):
    env = F.avg_pool1d(F.pad(mel.transpose(1, 2), (3, 3), mode='replicate'), 7, stride=1).transpose(1, 2)
    kb = torch.arange(100, device=mel.device, dtype=mel.dtype)
    wk = ((64.0 - kb) / 18.0).clamp(0.0, 1.0)[None, :, None]
    mel_w = env + (mel - env) * (1.0 + comb_gain * wk)
    return torch.cat([mel_w[:, :98], mel_w[:, 97:98], mel_w[:, 97:98]], dim=1)


def install_voc_qat_hooks(voc):
    hooks = []
    for name, m in voc.named_modules():
        if name.startswith('src.f0_net') or name.endswith('post'):
            continue
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)) and m.weight.numel() >= 128:
            ax = 1 if isinstance(m, nn.ConvTranspose1d) else 0
            def _pre(mod, inp, _ax=ax):
                if mod.training and getattr(voc, '_qat_on', False):
                    mod._orig_w = mod.weight.data.clone()
                    mod.weight.data.copy_(ste_int8_weight(mod.weight, axis=_ax, enabled=True).data)
            def _post(mod, inp, out):
                if hasattr(mod, '_orig_w'):
                    mod.weight.data.copy_(mod._orig_w)
                    delattr(mod, '_orig_w')
            hooks.append(m.register_forward_pre_hook(_pre))
            hooks.append(m.register_forward_hook(_post))
    return hooks


def build_voc144(init_p, teacher_p, svd_t1=False, dev='cuda'):
    G = VocoderV2(ch=144, res_kernels=(3, 7, 11)).to(dev)
    if os.path.exists(init_p):
        ck = torch.load(init_p, map_location=dev, weights_only=False)
        sd = {k: v.float() for k, v in ck.get('G_ema', ck.get('G', ck)).items()}
        if any('parametrizations' in k or 'weight_g' in k for k in sd):
            G.load_state_dict(sd, strict=False)
            G.fuse_weight_norm()
        else:
            G.fuse_weight_norm()
            G.load_state_dict(sd, strict=False)
    else:
        G.fuse_weight_norm()

    # Copy phase-locked f0_net from converged v3_voc320.pt
    T = VocoderV2(ch=320, res_kernels=(3, 7, 11)).to(dev)
    T.fuse_weight_norm()
    tck = torch.load(teacher_p, map_location=dev, weights_only=False)
    tsd = {k: v.float() for k, v in tck.get('G_ema', tck.get('G', tck)).items()}
    T.load_state_dict(tsd, strict=False)
    snap_f0_int8(T)
    T.eval()
    for p in T.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        G.src.f0_net.load_state_dict(T.src.f0_net.state_dict(), strict=True)
    snap_f0_int8(G)

    if svd_t1:
        for b_idx, r_k in ((1, 50), (2, 31)):
            blk = G.mrfs[0][b_idx]
            for i in range(len(blk.c1)):
                if isinstance(blk.c1[i], nn.Conv1d):
                    blk.c1[i] = _make_svd_seq(blk.c1[i], r_k, use_qat=False)
                if isinstance(blk.c2[i], nn.Conv1d):
                    blk.c2[i] = _make_svd_seq(blk.c2[i], r_k, use_qat=False)
        G.to(dev)
    return G, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', default='/work/tts/ck/v14_voc144.pt')
    ap.add_argument('--teacher', default='/work/tts/ck/v3_voc320.pt')
    ap.add_argument('--data', default='/work/sp/distill/gt_feats.pt')
    ap.add_argument('--rect', default='/work/sp/distill/v3_rectified_cache.pt')
    ap.add_argument('--save', required=True)
    ap.add_argument('--svd_t1', action='store_true')
    ap.add_argument('--minutes', type=float, default=14.0)
    ap.add_argument('--bs', type=int, default=40)
    ap.add_argument('--seg', type=int, default=44)
    ap.add_argument('--lr', type=float, default=2.2e-4)
    ap.add_argument('--warm', type=int, default=200)
    ap.add_argument('--qat_start', type=float, default=0.50)
    a = ap.parse_args()

    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    tag = 'voc144_t1' if a.svd_t1 else 'voc144'
    logf = f'/work/tts/logs/{tag}.log'

    def log(s):
        print(s, flush=True)
        open(logf, 'a').write(s + '\n')

    D0 = torch.load(a.data, weights_only=False)
    SF = a.seg
    tr = [x for x in D0['items'] if x.get('split') != 'dev' and len(x['wav']) // HOP > SF + 4]
    WAV = torch.cat([x['wav'] for x in tr]).to(dev)
    lens = torch.tensor([len(x['wav']) // HOP for x in tr], device=dev)
    starts = torch.cumsum(torch.tensor([len(x['wav']) for x in tr]), 0) - torch.tensor([len(x['wav']) for x in tr])
    starts = starts.to(dev)
    ar_w = torch.arange(SF * HOP, device=dev)

    rect_mels = []
    if os.path.exists(a.rect):
        R_d = torch.load(a.rect, map_location='cpu', weights_only=False)
        r_mean, r_std = float(R_d.get('mean', -6.1760635)), float(R_d.get('std', 2.4932315))
        for item in R_d['cache']:
            yc = item['y_clean']  # [T, 100]
            if yc.shape[0] > SF + 4:
                rect_mels.append((yc.T.float() * r_std + r_mean).half().to(dev))
    log(f'[{tag}] loaded {len(tr)} GT wavs + {len(rect_mels)} Rectified teacher mels | minutes={a.minutes}')

    G, T_voc = build_voc144(a.init, a.teacher, svd_t1=a.svd_t1, dev=dev)
    G._qat_on = False
    install_voc_qat_hooks(G)
    G_ema = copy.deepcopy(G).eval()
    D = DiscV2().to(dev).train()

    oG = torch.optim.AdamW([p for p in G.parameters() if p.requires_grad], lr=a.lr, betas=(0.8, 0.99), weight_decay=1e-4)
    oD = torch.optim.AdamW(D.parameters(), lr=a.lr * 0.8, betas=(0.8, 0.99), weight_decay=1e-4)

    t0 = time.time()
    tstep = time.time()
    step = 0
    budget = a.minutes * 60.0

    def save_ckpt(st_):
        sd_out = {k: v.detach().float().cpu() for k, v in G_ema.state_dict().items()}
        torch.save(dict(G_ema=sd_out, step=st_, ch=144, svd_t1=a.svd_t1), a.save + '.tmp')
        os.replace(a.save + '.tmp', a.save)

    while True:
        spent = time.time() - t0
        if spent > budget:
            break
        step += 1
        prog = min(1.0, spent / max(budget, 1.0))
        lr_cur = a.lr * (0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * prog)))
        for g_ in oG.param_groups:
            g_['lr'] = lr_cur
        for g_ in oD.param_groups:
            g_['lr'] = lr_cur * 0.8

        if prog >= a.qat_start and not G._qat_on:
            G._qat_on = True
            log(f'[{tag}] STE INT8 QAT enabled at step {step}')

        # Half batch from GT waveforms (+ optional smoothing to mimic 1-step AM), half from rectified teacher mels
        B1 = a.bs // 2 if rect_mels else a.bs
        B2 = a.bs - B1
        i = torch.randint(0, len(tr), (B1,), device=dev)
        f = (torch.rand(B1, device=dev) * (lens[i] - SF - 1).float()).long()
        x_gt = WAV[(starts[i] + f * HOP)[:, None] + ar_w[None]].float() / 32767.0
        xm1 = mel_spec_24k(x_gt, dev)
        sm_mask = (torch.rand(B1, device=dev) < 0.45)[:, None, None]
        xm1_sm = 0.65 * xm1 + 0.35 * F.avg_pool1d(F.pad(xm1, (1, 1), mode='replicate'), 3, stride=1)
        mel_1 = vin(torch.where(sm_mask, xm1_sm, xm1), comb_gain=0.10)

        if B2 > 0:
            m2_list = []
            for _ in range(B2):
                rm = rect_mels[random.randrange(len(rect_mels))]
                s_idx = random.randint(0, rm.shape[1] - SF - 1)
                m2_list.append(rm[:, s_idx:s_idx + SF])
            xm2 = torch.stack(m2_list, dim=0).float()
            mel_2 = vin(xm2, comb_gain=0.10)
            mel_in = torch.cat([mel_1, mel_2], dim=0)
        else:
            mel_in = mel_1

        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            re_t, im_t, _, _ = T_voc.spectrum_and_f0(mel_in)
        y_t = ola64(torch.complex(re_t.float(), im_t.float()), T_voc.win)

        G.train()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            re_s, im_s, _, _ = G.spectrum_and_f0(mel_in)
        y_s = ola64(torch.complex(re_s.float(), im_s.float()), G.win)

        T_len = min(y_s.shape[1], y_t.shape[1], x_gt.shape[1])
        y_s, y_t, x_gt = y_s[:, :T_len], y_t[:, :T_len], x_gt[:, :T_len]
        y_real = torch.cat([x_gt, y_t[B1:].detach()], dim=0) if B2 > 0 else x_gt

        use_gan = (step > a.warm)
        ld_v = 0.0
        fr_d = None
        if use_gan:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                dr, fr_d = D(y_real[:, None])
                df, _ = D(y_s.detach()[:, None])
                ld = sum(F.relu(1.0 - r.float()).mean() + F.relu(1.0 + q.float()).mean() for r, q in zip(dr, df)) / len(dr)
            oD.zero_grad(set_to_none=True)
            ld.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), 10.0)
            oD.step()
            ld_v = float(ld.item())

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # 1. Sample-synchronous phase-locked complex iSTFT spectrum + instantaneous frequency loss vs v3_voc320
            l_spec = spec_phase_gd_loss(re_s, im_s, re_t, im_t) + 0.5 * (F.l1_loss(re_s.float(), re_t.float()) + F.l1_loss(im_s.float(), im_t.float()))
            # 2. Direct phase-locked waveform L1 + Multi-Resolution Complex STFT loss
            l_wav = F.l1_loss(y_s, y_t)
            l_st = 0.65 * mr_complex_stft_loss(y_s, y_t, dev) + 0.35 * mr_complex_stft_loss(y_s[:B1], x_gt, dev)
            # 3. 100-band log-mel L1 loss
            ym_s = mel_spec_24k(y_s, dev)
            ym_t = mel_spec_24k(y_t, dev)
            l_mel = 0.65 * F.l1_loss(ym_s, ym_t) + 0.35 * F.l1_loss(ym_s[:B1], xm1[:, :, :ym_s.shape[2]])

            lg = 45.0 * l_mel + 2.5 * l_st + 2.5 * l_spec + 8.0 * l_wav
            if use_gan and fr_d is not None:
                fr = [[t.detach() for t in fs] for fs in fr_d]
                df, ff = D(y_s[:, None])
                l_adv = sum(F.relu(1.0 - q.float()).mean() for q in df) / len(df)
                n_fm = sum(len(A) for A in fr)
                l_fm = sum(F.l1_loss(b.float(), a_.float()) for A, B in zip(fr, ff) for a_, b in zip(A, B)) / n_fm
                lg = lg + l_adv + 4.0 * l_fm

        oG.zero_grad(set_to_none=True)
        lg.backward()
        gn = torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
        if torch.isfinite(gn):
            oG.step()

        with torch.no_grad():
            dcy = min(0.995, (1 + step) / (12 + step))
            for pe, p in zip(G_ema.parameters(), G.parameters()):
                pe.lerp_(p, 1.0 - dcy)

        if step == 50:
            sps = 50.0 / max(1e-6, time.time() - t0)
            log(f'[GATE] {tag} step-50 throughput {sps:.2f} steps/s')
            if sps < 0.8:
                log('[GATE] too slow')
                sys.exit(3)

        if step % 100 == 0:
            now = time.time()
            ips = 100.0 / max(1e-6, now - tstep)
            tstep = now
            log(f'Step {step} mel {l_mel.item():.4f} stft {l_st.item():.3f} f0 {l_spec.item():.3f} D {ld_v:.3f} | {(now - t0) / 60:.1f}/{a.minutes:.0f} min ({ips:.1f} it/s)')

        if step in (400, 800, 1500) or step % 500 == 0:
            save_ckpt(step)

    save_ckpt(step)
    log(f'VOC_GT_DONE tag={tag} steps={step} -> {a.save}')


if __name__ == '__main__':
    main()
