#!/usr/bin/env python3
"""Stage S — ground-truth Pro-v3 student training (replaces the V1-teacher distillation that made every tier sound like old Sakhiya).

Every target is REAL Pro-v3 audio: forced-aligned phoneme durations, the real 100-band log-mel, and the real waveform. The student
single graph (encoder -> duration -> CFM velocity decoder -> Harmonic-iSTFT vocoder, exactly `student.Student.infer`) is trained end-to-end:
  mel L1 + mel-prior + ODE end-point + log-duration + gradient/cepstral sharpness (anti-blur) + multi-res complex STFT + WavLM perceptual
  (every 2nd step) + MPD/MRD hinge GAN with feature matching on the final waveform, then STE per-channel INT8 QAT.
Writes /work/sp/distill/runs/<tier>/ema_latest.pt (+ema_final.pt) in the format export_gate.py / ladder_daemon.py already consume.
usage: train_student_gt.py --tier t15 --minutes 120"""
import argparse, copy, math, os, random, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); DIST = os.path.join(HERE, '..', 'distill'); REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
for p in (DIST, REPO):
    if p not in sys.path: sys.path.insert(0, p)
from student import build, set_qat, n_params, TIERS  # noqa: E402
from train_student import install_ste_qat_hooks, mrstft, _ode_solve  # noqa: E402
from sakhiya.v2_vocoder import mel_spec_24k, HOP  # noqa: E402
from scripts.train_v2_voc import DiscV2, mr_complex_stft_loss  # noqa: E402

LR = dict(t30=1.2e-4, t15=1.4e-4, t10=1.6e-4, t5=1.8e-4, t1=2.2e-4)


def post_mel(G, mel):
    """Replicates Student.infer post-processing between the CFM output and the vocoder."""
    env = F.avg_pool1d(F.pad(mel.transpose(1, 2), (3, 3), mode='replicate'), 7, stride=1).transpose(1, 2)
    kb = torch.arange(100, device=mel.device, dtype=mel.dtype); wk = ((64.0 - kb) / 18.0).clamp(0.0, 1.0)[None, :, None]
    mel_w = env + (mel - env) * (1.0 + (0.10 if G.steps <= 2 else 0.06) * wk)
    if G.arch == 'sakhiya_pico':
        mu_ = mel_w.mean(dim=(1, 2), keepdim=True); mel_w = (mel_w - mu_) * 1.04 + mu_ - 0.04
        env_f = F.avg_pool1d(F.pad(mel_w.transpose(1, 2), (2, 2), mode='replicate'), 5, stride=1).transpose(1, 2)
        wf = ((72.0 - kb) / 24.0).clamp(0.0, 1.0)[None, :, None]; mel_w = mel_w + 0.14 * wf * (mel_w - env_f)
    return torch.cat([mel_w[:, :98], mel_w[:, 97:98], mel_w[:, 97:98]], dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', required=True); ap.add_argument('--data', default='/work/sp/distill/gt_feats.pt')
    ap.add_argument('--out', default=None); ap.add_argument('--minutes', type=float, default=120); ap.add_argument('--bs', type=int, default=24)
    ap.add_argument('--seg', type=int, default=64); ap.add_argument('--lr', type=float, default=None); ap.add_argument('--gan_start', type=int, default=1500)
    ap.add_argument('--qat_start', type=float, default=0.55); ap.add_argument('--max_step_s', type=float, default=4.0); ap.add_argument('--save_every', type=int, default=500)
    ap.add_argument('--resume', action='store_true'); ap.add_argument('--no_wavlm', action='store_true'); ap.add_argument('--lr_mult', type=float, default=1.0)
    a = ap.parse_args()
    dev = 'cuda'; out = a.out or f'/work/sp/distill/runs/{a.tier}'; os.makedirs(out, exist_ok=True)
    random.seed(1); torch.manual_seed(1); torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

    def log(s):
        print(s, flush=True); open(out + '/train.log', 'a').write(s + '\n')
    Dd = torch.load(a.data, weights_only=False)
    tr, dv = [], []
    for x in Dd['items']:
        if x['mel'].shape[1] < a.seg + 2 or len(x['ids']) < 4: continue
        x['mel'] = x['mel'].to(dev); x['wav'] = x['wav'].to(dev); x['ids'] = x['ids'].to(dev); dur = x['dur'].to(dev).long(); x['dur'] = dur
        x['fidx'] = torch.repeat_interleave(torch.arange(len(dur), device=dev), dur)[: x['mel'].shape[1]]
        (dv if x['split'] == 'dev' else tr).append(x)
    tr = [x for x in tr if len(x['fidx']) == x['mel'].shape[1]]
    log(f'[{a.tier}] GT data: train {len(tr)} utts {sum(len(x["wav"]) for x in tr) / 24000 / 3600:.2f} h | dev {len(dv)}')

    G = build(a.tier).to(dev); G.warm_init('/work/tts/ck'); install_ste_qat_hooks(G)
    stats = G.am.stats.clone()
    D = DiscV2().to(dev)
    vch = TIERS[a.tier]['voc_ch']; dpath = f'/work/sp/distill/voc{vch}/D.pt'
    if os.path.exists(dpath):
        try: D.load_state_dict(torch.load(dpath, map_location=dev, weights_only=False)['D']); log(f'[D] init from {dpath}')
        except Exception as e: log(f'[D] init skip {e}')
    wavlm = None
    if not a.no_wavlm:
        try:
            from transformers import WavLMModel
            wavlm = WavLMModel.from_pretrained('microsoft/wavlm-base-plus').to(dev).eval()
            for p in wavlm.parameters(): p.requires_grad_(False)
            log('[wavlm] loaded')
        except Exception as e: log(f'[wavlm] skipped {e}')
    ema = copy.deepcopy(G).eval()
    lr = (a.lr or LR[a.tier]) * a.lr_mult; vr = 0.25
    am_p = [p for n, p in G.named_parameters() if not n.startswith('voc.') and p.requires_grad]; vo_p = [p for n, p in G.named_parameters() if n.startswith('voc.') and p.requires_grad]
    oG = torch.optim.AdamW([dict(params=am_p, lr=lr), dict(params=vo_p, lr=lr * vr)], betas=(0.8, 0.99), weight_decay=1e-4)
    oD = torch.optim.AdamW(D.parameters(), lr=1e-4, betas=(0.8, 0.99))
    step, spent = 0, 0.0
    if a.resume and os.path.exists(out + '/state.pt'):
        try:
            S = torch.load(out + '/state.pt', map_location=dev, weights_only=False)
            if S.get('cfg') == G.cfg:
                G.load_state_dict(S['G']); ema.load_state_dict(S['ema']); oG.load_state_dict(S['oG']); D.load_state_dict(S['D']); step, spent = S['step'], S['spent']
                log(f'[resume] step {step} spent {spent / 60:.1f} min')
        except Exception as e: log(f'[resume] failed {e}')
    log(f'[{a.tier}] params {n_params(G) / 1e6:.3f}M lr {lr} bs {a.bs} seg {a.seg} steps/ode {G.steps}')
    budget = a.minutes * 60; t0 = time.time() - spent; tstep = time.time(); qat_on = bool(G._qat_flag)
    ac = lambda: torch.autocast('cuda', dtype=torch.bfloat16)
    ar = torch.arange(a.seg, device=dev); last = {}

    while True:
        spent = time.time() - t0; prog = min(1.0, spent / budget)
        if spent > budget: break
        mult = 0.04 + 0.96 * 0.5 * (1.0 + math.cos(math.pi * prog)) * min(1.0, step / 300 + 0.05)
        oG.param_groups[0]['lr'] = lr * mult; oG.param_groups[1]['lr'] = lr * vr * mult
        for g_ in oD.param_groups: g_['lr'] = 1e-4 * (0.1 + 0.9 * mult)
        if prog >= a.qat_start and not qat_on:
            set_qat(G, True); qat_on = True; log(f'[qat] int8 fake-quant ON at step {step}')
        xs = random.sample(tr, a.bs); B = a.bs; L = max(len(x['ids']) for x in xs)
        ids = torch.zeros(B, L, dtype=torch.long, device=dev); dur = torch.zeros(B, L, device=dev); il = torch.zeros(B, dtype=torch.long, device=dev)
        fi = torch.zeros(B, a.seg, dtype=torch.long, device=dev); mel_gt = torch.zeros(B, 100, a.seg, device=dev); wav_gt = torch.zeros(B, a.seg * HOP, device=dev)
        lang = torch.zeros(B, dtype=torch.long, device=dev)
        for b, x in enumerate(xs):
            n = len(x['ids']); ids[b, :n] = x['ids']; dur[b, :n] = x['dur'].float(); il[b] = n; lang[b] = int(x['lang'])
            T = x['mel'].shape[1]; s0 = random.randint(0, T - a.seg)
            fi[b] = x['fidx'][s0:s0 + a.seg]; mel_gt[b] = x['mel'][:, s0:s0 + a.seg].float(); wav_gt[b] = x['wav'][s0 * HOP:(s0 + a.seg) * HOP].float() / 32767.0
        im = torch.arange(L, device=dev)[None] < il[:, None]
        spk0 = torch.zeros(B, dtype=torch.long, device=dev); studio = torch.ones(B, dtype=torch.long, device=dev)
        lang_eff = torch.where(lang == 1, torch.full_like(lang, 4), torch.zeros_like(lang)); sty = torch.where(lang == 1, torch.full_like(lang, 5), torch.zeros_like(lang))
        pros = G.pros_tbl[sty]; ones_m = torch.ones((B, 1), device=dev)
        p_add = (torch.randn(B, L, device=dev) * 0.14).clamp(-0.42, 0.42) * im.float()
        G.train()
        with ac():
            hs, mus, gs = G.am.enc(ids, spk0, lang_eff, sty, studio, im, pitch_add=p_add, prosody=pros, sty_mask=ones_m)
            if hasattr(G.am.enc, 'energy_pred'):
                eng_hat = G.am.enc.energy_pred(hs).squeeze(-1).clamp(-1.2, 1.2)
                mus = mus + (0.10 * pros[:, 2:3] + 0.06 * eng_hat)[..., None] * 0.32
            lds = G.am.dur.pred_logdur(ids, hs, im) + G.dur_rate_bias.clamp(-0.12, 0.12)
            mus_t = mus.transpose(1, 2)
            mu_raw = torch.gather(mus_t, 2, fi[:, None, :].expand(-1, 100, -1))
            mu_sm = F.avg_pool1d(F.pad(mu_raw, (1, 1), mode='replicate'), 3, stride=1)
            mu_seg = 0.72 * mu_raw + 0.28 * mu_sm
            m_seg = torch.ones((B, 1, a.seg), device=dev)
            xt = _ode_solve(G.am.vel, mu_seg, gs, m_seg, G.steps)
            mel_s = xt * stats[1] + stats[0]
            y_s, re_s, im_s, f0_s, _ = G.voc_wav(post_mel(G, mel_s))
        y_s = y_s.float(); n_ = min(y_s.shape[1], wav_gt.shape[1]); y_s = y_s[:, :n_]; y_g = wav_gt[:, :n_]
        y_norm = (mel_gt - stats[0]) / stats[1]
        l_dur = F.l1_loss(lds[im].float(), torch.log(dur.clamp(min=1.0))[im])
        l_prior = F.l1_loss(mu_seg.float(), y_norm); l_ode = F.l1_loss(xt.float(), y_norm)
        ms = mel_s.float()
        l_df = F.l1_loss(ms[:, 1:] - ms[:, :-1], mel_gt[:, 1:] - mel_gt[:, :-1]); l_dt = F.l1_loss(ms[:, :, 1:] - ms[:, :, :-1], mel_gt[:, :, 1:] - mel_gt[:, :, :-1])
        l_cep = F.l1_loss(torch.fft.rfft(ms, dim=1).abs()[:, 4:48], torch.fft.rfft(mel_gt, dim=1).abs()[:, 4:48])
        ym = mel_spec_24k(y_s, dev)
        l_mel = F.l1_loss(ms, mel_gt) + 0.8 * F.l1_loss(ym[:, :, :a.seg], mel_gt[:, :, :ym.shape[2]]) + 0.6 * l_df + 0.3 * l_dt + 0.02 * l_cep
        l_st = mr_complex_stft_loss(y_s, y_g, dev)
        step += 1
        l_wlm = torch.zeros((), device=dev)
        if wavlm is not None and step % 2 == 0:
            with ac():
                n16 = (y_s.shape[1] * 2) // 3
                a16 = F.interpolate(y_g[:, None], size=n16, mode='linear', align_corners=False)[:, 0]; b16 = F.interpolate(y_s[:, None], size=n16, mode='linear', align_corners=False)[:, 0]
                a16 = (a16 - a16.mean(-1, keepdim=True)) / a16.std(-1, keepdim=True).clamp(min=1e-5); b16 = (b16 - b16.mean(-1, keepdim=True)) / b16.std(-1, keepdim=True).clamp(min=1e-5)
                with torch.no_grad(): ha = wavlm(a16, output_hidden_states=True).hidden_states
                hb = wavlm(b16, output_hidden_states=True).hidden_states
                l_wlm = sum(F.l1_loss(hb[k].float(), ha[k].detach().float()) for k in (3, 6, 9, 12) if k < len(ha)) / 4.0
        use_gan = step > a.gan_start
        l_adv = torch.zeros((), device=dev); l_fm = torch.zeros((), device=dev); ld_v = 0.0
        if use_gan:
            with ac():
                dr, fr = D(y_g[:, None]); df, _ = D(y_s.detach()[:, None])
                ld = sum(F.relu(1.0 - r.float()).mean() + F.relu(1.0 + q.float()).mean() for r, q in zip(dr, df)) / len(dr)
            oD.zero_grad(set_to_none=True); ld.backward(); torch.nn.utils.clip_grad_norm_(D.parameters(), 10.0); oD.step(); ld_v = ld.item()
            with ac():
                fr = [[t.detach() for t in fs] for fs in fr]; df, ff = D(y_s[:, None])
                l_adv = sum(F.relu(1.0 - q.float()).mean() for q in df) / len(df)
                l_fm = sum(F.l1_loss(b_.float(), a_.float()) for A, Bf in zip(fr, ff) for a_, b_ in zip(A, Bf)) / sum(len(A) for A in fr)
        gw = min(1.0, (step - a.gan_start) / 500.0) if use_gan else 0.0
        loss = 6.0 * l_mel + 2.5 * l_prior + 1.5 * l_ode + 1.5 * l_dur + 1.5 * l_st + 1.2 * l_wlm + gw * (1.0 * l_adv + 3.0 * l_fm)
        oG.zero_grad(set_to_none=True); loss.backward(); gn = torch.nn.utils.clip_grad_norm_(G.parameters(), 3.0)
        if torch.isfinite(gn): oG.step()
        with torch.no_grad():
            dcy = min(0.998, (1 + step) / (10 + step))
            for pe, pg in zip(ema.parameters(), G.parameters()): pe.lerp_(pg, 1 - dcy)
        if step == 50:
            sps = (time.time() - tstep) / 50
            if sps > a.max_step_s: log(f'WATCHDOG: step-50 {sps:.2f}s/step > {a.max_step_s}s'); sys.exit(3)
        if step % 50 == 0:
            now = time.time(); ips = 50 / max(1e-6, now - tstep); tstep = now
            log(f'Step {step} | {ips:.2f} it/s | mel {l_mel.item():.3f} prior {l_prior.item():.3f} ode {l_ode.item():.3f} dur {l_dur.item():.3f} st {l_st.item():.3f} wlm {l_wlm.item():.3f} adv {l_adv.item():.3f} fm {l_fm.item():.3f} D {ld_v:.2f} | {spent / 60:.1f}/{a.minutes:.0f} min qat={int(qat_on)}')
        if step in (50, 150) or step % a.save_every == 0:
            torch.save(dict(ema=ema.state_dict(), cfg=G.cfg, tier=a.tier, arch=G.arch, step=step, f0=[5.418, 0.203], qat=qat_on, gt=True), out + '/ema_latest.tmp'); os.replace(out + '/ema_latest.tmp', out + '/ema_latest.pt')
            if step % (a.save_every * 2) == 0:
                torch.save(dict(G=G.state_dict(), ema=ema.state_dict(), oG=oG.state_dict(), D=D.state_dict(), cfg=G.cfg, step=step, spent=time.time() - t0), out + '/state.tmp'); os.replace(out + '/state.tmp', out + '/state.pt')
    torch.save(dict(ema=ema.state_dict(), cfg=G.cfg, tier=a.tier, arch=G.arch, step=step, f0=[5.418, 0.203], qat=qat_on, gt=True), out + '/ema_final.pt')
    torch.save(dict(ema=ema.state_dict(), cfg=G.cfg, tier=a.tier, arch=G.arch, step=step, f0=[5.418, 0.203], qat=qat_on, gt=True), out + '/ema_latest.pt')
    log(f'DONE step {step} minutes {(time.time() - t0) / 60:.1f}')


if __name__ == '__main__':
    main()
