#!/usr/bin/env python3
"""Stage V — Pro-v3 vocoder GAN (HiFi-GAN style MPD+MRD hinge GAN) on REAL Pro-v3 24 kHz audio, per tier width.

The ladder used to render every student through the *V1 speaker's* vocoder (the reason all tiers sounded like old Sakhiya). This
stage re-trains the Harmonic-iSTFT vocoder (ch=320 or 144) on ground-truth Pro-v3 mel -> waveform pairs, warm-started from V1 for speed,
and writes /work/tts/ck/v3_voc{ch}.pt (fused, picked up by student.warm_init) + the discriminator for the joint student stage.
usage: train_voc_gt.py --ch 320 --init /work/tts/ck/v5_c1_voc.pt --minutes 40"""
import argparse, copy, math, os, sys, time
import torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, REPO)
from sakhiya.v2_vocoder import VocoderV2, mel_spec_24k, ola64, HOP  # noqa: E402
from scripts.train_v2_voc import DiscV2, mr_complex_stft_loss, gpu_f0_uv  # noqa: E402


def vin(mel, sharpen):
    """Exactly what the shipped student graph feeds the vocoder: comb sharpening (optional) then bins 98/99 <- 97."""
    env = F.avg_pool1d(F.pad(mel.transpose(1, 2), (3, 3), mode='replicate'), 7, stride=1).transpose(1, 2)
    kb = torch.arange(100, device=mel.device, dtype=mel.dtype); wk = ((64.0 - kb) / 18.0).clamp(0.0, 1.0)[None, :, None]
    sh = env + (mel - env) * (1.0 + 0.10 * wk)
    mel = torch.where(sharpen[:, None, None], sh, mel)
    return torch.cat([mel[:, :98], mel[:, 97:98], mel[:, 97:98]], dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ch', type=int, default=320); ap.add_argument('--init', required=True); ap.add_argument('--data', default='/work/sp/distill/gt_feats.pt')
    ap.add_argument('--out', default=None); ap.add_argument('--minutes', type=float, default=40); ap.add_argument('--bs', type=int, default=48)
    ap.add_argument('--seg', type=int, default=40); ap.add_argument('--lr', type=float, default=1.2e-4); ap.add_argument('--warm', type=int, default=600)
    ap.add_argument('--ema', type=float, default=0.995); ap.add_argument('--save', default=None)
    a = ap.parse_args()
    dev = 'cuda'; out = a.out or f'/work/sp/distill/voc{a.ch}'; os.makedirs(out, exist_ok=True)
    save = a.save or f'/work/tts/ck/v3_voc{a.ch}.pt'
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    logf = out + '/train.log'

    def log(s):
        print(s, flush=True); open(logf, 'a').write(s + '\n')
    D0 = torch.load(a.data, weights_only=False)
    tr = [x for x in D0['items'] if x['split'] != 'dev' and len(x['wav']) // HOP > a.seg + 2]
    va = [x for x in D0['items'] if x['split'] == 'dev'][:40]
    SF = a.seg
    WAV = torch.cat([x['wav'] for x in tr]).to(dev); lens = torch.tensor([len(x['wav']) // HOP for x in tr], device=dev)
    starts = torch.cumsum(torch.tensor([len(x['wav']) for x in tr]), 0) - torch.tensor([len(x['wav']) for x in tr]); starts = starts.to(dev)
    ar_w = torch.arange(SF * HOP, device=dev)
    log(f'[voc{a.ch}] train {len(tr)} utts {WAV.numel() / 24000 / 3600:.2f} h | dev {len(va)}')
    G = VocoderV2(ch=a.ch).to(dev)
    ck = torch.load(a.init, map_location=dev, weights_only=False)
    sd = {k: v.float() for k, v in ck.get('G_ema', ck.get('G', ck)).items()}
    print('load', G.load_state_dict(sd, strict=False))
    G_ema = copy.deepcopy(G).eval(); D = DiscV2().to(dev)
    dp = '/work/tts/ck/voc_disc_state.pt'
    if os.path.exists(dp):
        try:
            s = torch.load(dp, map_location=dev, weights_only=False); print('disc init', D.load_state_dict(s.get('D', s), strict=False))
        except Exception as e: print('disc init skip', e)
    oG = torch.optim.AdamW(G.parameters(), a.lr, betas=(0.8, 0.99), weight_decay=1e-5); oD = torch.optim.AdamW(D.parameters(), a.lr, betas=(0.8, 0.99))
    step, t0 = 0, time.time(); G.train(); D.train()

    def save_fused(path_, step_):
        # NOTE: never deepcopy a parametrized module and fuse it: the copy shares its dynamic ParametrizedConv1d class with G,
        # and removing the parametrization strips `.weight` from the live generator (crash at step 1000). Build a fresh module.
        g = VocoderV2(ch=a.ch); g.load_state_dict({k: v.detach().cpu() for k, v in G_ema.state_dict().items()}); g = g.fuse_weight_norm()
        torch.save(dict(G_ema={k: v.float() for k, v in g.state_dict().items()}, step=step_, ch=a.ch), path_ + '.tmp'); os.replace(path_ + '.tmp', path_)

    @torch.no_grad()
    def val():
        ls = []
        for x in va:
            w = (x['wav'].float() / 32767.0)[None].to(dev); m = mel_spec_24k(w, dev)
            y = G_ema(vin(m, torch.zeros(1, dtype=torch.bool, device=dev))); T = min(y.shape[1], w.shape[1])
            ls.append(F.l1_loss(mel_spec_24k(y[:, :T], dev), m[:, :, :T // HOP]).item())
        return sum(ls) / max(1, len(ls))
    log(f'[voc{a.ch}] baseline val mel-L1 (init) {val():.4f}')
    while time.time() - t0 < a.minutes * 60:
        step += 1
        prog = min(1.0, (time.time() - t0) / (a.minutes * 60)); lr = a.lr * (0.06 + 0.94 * 0.5 * (1 + math.cos(math.pi * prog)))
        for g_ in oG.param_groups + oD.param_groups: g_['lr'] = lr
        i = torch.randint(0, len(tr), (a.bs,), device=dev)
        f = (torch.rand(a.bs, device=dev) * (lens[i] - SF - 1).float()).long()
        x = WAV[(starts[i] + f * HOP)[:, None] + ar_w[None]].float() / 32767.0
        xm = mel_spec_24k(x, dev)
        sharpen = torch.rand(a.bs, device=dev) < 0.5
        mel_in = xm
        sm = torch.rand(a.bs, device=dev) < 0.35   # mimic acoustic-model over-smoothing so the vocoder stays robust to predicted mels
        if bool(sm.any()):
            ms = F.avg_pool1d(F.pad(xm, (1, 1), mode='replicate'), 3, stride=1)
            mel_in = torch.where(sm[:, None, None], 0.6 * xm + 0.4 * ms + torch.randn_like(xm) * 0.05, xm)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            re_v, im_v, f0_p, uv_p = G.spectrum_and_f0(vin(mel_in, sharpen))
        y = ola64(torch.complex(re_v.float(), im_v.float()), G.win)
        T = min(y.shape[1], x.shape[1]); y, x = y[:, :T], x[:, :T]
        use_gan = step > a.warm; ld_v = 0.0; fr_d = None
        if use_gan:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                dr, fr_d = D(x[:, None]); df, _ = D(y.detach()[:, None])
                ld = sum(F.relu(1.0 - r.float()).mean() + F.relu(1.0 + q.float()).mean() for r, q in zip(dr, df)) / len(dr)
            oD.zero_grad(set_to_none=True); ld.backward(); torch.nn.utils.clip_grad_norm_(D.parameters(), 10.0); oD.step(); ld_v = ld.item()
        f0_gt, uv_gt = gpu_f0_uv(x, SF)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            ym = mel_spec_24k(y, dev); l_mel = F.l1_loss(ym, xm[:, :, :ym.shape[2]])
            l_st = mr_complex_stft_loss(y, x, dev)
            l_f0 = F.l1_loss(torch.log(f0_p[:, :, :SF] + 1e-5) * uv_gt, torch.log(f0_gt + 1e-5) * uv_gt) + 0.25 * (-(uv_gt.float() * torch.log(uv_p[:, :, :SF].float() + 1e-6) + (1 - uv_gt.float()) * torch.log(1 - uv_p[:, :, :SF].float() + 1e-6)).mean())
            lg = 45.0 * l_mel + 2.0 * l_st + 2.0 * l_f0
            if use_gan:
                fr = [[t.detach() for t in fs] for fs in fr_d]; df, ff = D(y[:, None])
                l_adv = sum(F.relu(1.0 - q.float()).mean() for q in df) / len(df)
                n_fm = sum(len(A) for A in fr)
                l_fm = sum(F.l1_loss(b.float(), a_.float()) for A, B in zip(fr, ff) for a_, b in zip(A, B)) / n_fm
                lg = lg + l_adv + 4.0 * l_fm
        oG.zero_grad(set_to_none=True); lg.backward(); gn = torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
        if torch.isfinite(gn): oG.step()
        with torch.no_grad():
            for pe, p in zip(G_ema.parameters(), G.parameters()): pe.lerp_(p, 1.0 - a.ema)
        if step == 50:
            sps = 50.0 / (time.time() - t0); log(f'[GATE] step-50 throughput {sps:.2f} steps/s')
            if sps < 0.8: log('[GATE] too slow'); sys.exit(3)
        if step % 100 == 0:
            log(f'Step {step} mel {l_mel.item():.4f} stft {l_stft_v(l_st):.3f} f0 {l_f0.item():.3f} D {ld_v:.3f} | {(time.time() - t0) / 60:.1f}/{a.minutes:.0f} min')
        if step % 1000 == 0:
            log(f'VAL step {step} ema mel-L1 {val():.4f}'); save_fused(save, step)
            torch.save(dict(D=D.state_dict(), step=step), out + '/D.pt')
    log(f'VAL final ema mel-L1 {val():.4f}'); save_fused(save, step); torch.save(dict(D=D.state_dict(), step=step), out + '/D.pt')
    log(f'VOC_GT_DONE ch={a.ch} steps={step} -> {save}')


def l_stft_v(x): return float(x.item())


if __name__ == '__main__':
    main()
