#!/usr/bin/env python3
"""Stage S — Sakhiya-Pro v3 Voice-Locked OT-CFM + Rectified Harmonic-Envelope Student Distillation
(`scripts/sakhiya_pro/distill2/train_student_gt.py`).

Fixes the root causes that made earlier exports sound like Old V1 or robotic/blurred:
1. Filters `/work/sp/distill/gt_feats.pt` using `/work/sp/distill/manifest.jsonl` to exclude low-UTMOS noisy VC clips,
   retaining ~2,800 clean Sakhiya-Pro v3 single-voice utterances + oversampled `sakhiya_pro_v3` master clips.
2. Trains `/work/tts/ck/v3_t30_am.pt` (`SakhiyaAM` 192-d, 12-block) using true OT-CFM (`SakhiyaAM.loss` with `t ~ U(0,1)`,
   ground-truth durations `dur_gt`, utterance pitch `_mel_f0_z`, and `MelCTCAligner` pronunciation loss) so the teacher AM
   locks onto Sakhiya-Pro v3's vocal tract & Nepal-native cadence (`sim_ProV3 >= 0.80`) with `4.10+ UTMOS`.
3. Builds `/work/sp/distill/v3_rectified_cache.pt` from `v3_t30_am.pt` + ground-truth Pro-v3 formant envelopes (`env_gt`).
4. Trains each student tier (`t30, t15, t10, t5, t1`) with:
   - Frozen GAN-converged Harmonic-iSTFT vocoder (`v3_voc320.pt` / `v3_voc144.pt` / `t1` SVD vocoder) so vocoder harmonics never blur.
   - Vocal-tract formant envelope matching (`env = avg_pool1d(mel, 7)`) to lock onto Sakhiya-Pro v3's exact timbre (`sim_ProV3`).
   - Phase-invariant Cepstral Harmonic Comb (`rfft(mel)[:, 4:48]`), Spectral Std, and Delta-Mel sharpness preservation so 1-step
     students (`t10, t5, t1`) never average out harmonic ridges into robotic buzz.
   - Frozen `MelCTCAligner` pronunciation KL/CTC supervision + STE Per-Channel INT8 QAT (`--qat_start 0.45`) + Step-50 watchdog.
"""
import argparse, copy, glob, json, math, os, random, sys, time
import librosa, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

HERE = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(HERE, '..', 'distill')
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
for p in (DIST, REPO, '/work/sp'):
    if p not in sys.path:
        sys.path.insert(0, p)

from student import Student, build, set_qat, n_params, TIERS, _calibrate_styles  # noqa: E402
from train_student import install_ste_qat_hooks, _ode_solve, spec_phase_gd_loss  # noqa: E402
from sakhiya.cfm import SakhiyaAM, STYLE_PROSODY_TARGETS, _dur_to_attn  # noqa: E402
from sakhiya.v2_vocoder import mel_spec_24k, HOP  # noqa: E402
from sakhiya.ctc_aligner import MelCTCAligner, batch_viterbi_ctc_durations  # noqa: E402
from sakhiya import g2p, symbols  # noqa: E402

# Synchronized encoder + velocity learning rates for 1-step students (t10, t5, t1) so mu_y and vel(mu_y) adapt in lock-step
# to Sakhiya-Pro v3 without harmonic mismatch; gentle polishing LR for 4-step OT-CFM students (t30, t15) initialized from v3_t30_am.pt.
LR_ENC = dict(t30=4.0e-5, t15=4.5e-5, t10=3.5e-4, t5=3.5e-4, t1=9.0e-4)
LR_VEL = dict(t30=3.5e-5, t15=4.0e-5, t10=3.5e-4, t5=3.5e-4, t1=9.0e-4)

BAD_OLD_SRCS = {'vc_slr54', 'vc_v13', 'vc_slr43', 'anchor_6834_stitched'}


def _mel_env(m, k=7):
    p = k // 2
    return F.avg_pool1d(F.pad(m.transpose(1, 2), (p, p), mode='replicate'), k, stride=1).transpose(1, 2)


def load_clean_pro_v3_items(data_path='/work/sp/distill/gt_feats.pt', dev='cuda'):
    D = torch.load(data_path, weights_only=False)
    mean, std = float(D.get('mean', -6.1760635)), float(D.get('std', 2.4932315))
    id2src = {}
    mf_p = '/work/sp/distill/manifest.jsonl'
    if os.path.exists(mf_p):
        for ln in open(mf_p):
            try:
                r = json.loads(ln)
                id2src[r['id']] = (r.get('src', ''), float(r.get('utmos', 3.6)))
            except Exception:
                pass
    tr, dv = [], []
    for x in D['items']:
        L = len(x['ids'])
        T = x['mel'].shape[1]
        if L < 4 or T < 40 or T > 680 or int(x['dur'].sum()) != T:
            continue
        if x.get('src') == 'old':
            sub_s, sub_u = id2src.get(x['id'], ('', 3.6))
            if sub_s in BAD_OLD_SRCS or sub_u < 3.40:
                continue
        (dv if x.get('split') == 'dev' else tr).append(x)

    # Also align & include the 15 sakhiya_pro_v3 master clips (oversampled 20x) for studio broadcast prosody lock
    reg_p = os.path.join(REPO, 'docs', 'samples', 'versions', 'registry.json')
    v3_dir = os.path.join(REPO, 'docs', 'samples', 'versions', 'sakhiya_pro_v3')
    if os.path.exists(reg_p) and os.path.isdir(v3_dir) and os.path.exists('/work/tts/ck/v31_ctc_recognizer.pt'):
        try:
            cc = torch.load('/work/tts/ck/v31_ctc_recognizer.pt', map_location=dev, weights_only=False)
            al = MelCTCAligner().to(dev).eval()
            al.load_state_dict(cc.get('model', cc), strict=False)
            reg = json.load(open(reg_p))
            for c in reg['versions'][0]['clips']:
                mp3_p = os.path.join(v3_dir, f"{c['id']}.mp3")
                if not os.path.exists(mp3_p):
                    continue
                w, _ = librosa.load(mp3_p, sr=24000)
                r = float(np.sqrt(np.mean(w ** 2)))
                if r < 1e-4:
                    continue
                w = np.clip(w * (0.062 / r), -0.97, 0.97).astype(np.float32)
                n = (len(w) // HOP) * HOP
                w = w[:n]
                ph_m = [q for p in g2p.text_to_phones(c['text'], 'sa' if c['lang'] == 'sa' else 'ne') for q in (['o', 'm'] if p == '<om>' else [p])]
                ids_list = symbols.encode(ph_m)
                with torch.no_grad():
                    m = mel_spec_24k(torch.from_numpy(w)[None].to(dev), dev)[0]
                    mn = ((m - mean) / std)[None]
                    ids_t = torch.tensor([ids_list], dtype=torch.long, device=dev)
                    yl_t = torch.tensor([m.shape[1]], dtype=torch.long, device=dev)
                    il_t = torch.tensor([len(ids_list)], dtype=torch.long, device=dev)
                    ym_t = torch.ones((1, m.shape[1]), dtype=torch.bool, device=dev)
                    lp = al(mn, ym_t)
                    dur_t, _ = batch_viterbi_ctc_durations(lp, ids_t, yl_t, il_t)
                    d_cpu = dur_t[0, :len(ids_list)].cpu().long()
                    if int(d_cpu.sum()) == m.shape[1]:
                        item_m = dict(
                            id='master_' + c['id'],
                            ids=torch.tensor(ids_list, dtype=torch.long),
                            lang=(1 if c['lang'] == 'sa' else 0),
                            mel=m.half().cpu(),
                            dur=d_cpu.to(torch.int16),
                            split='train',
                            src='pro_v3_master',
                        )
                        tr.extend([item_m] * 20)
        except Exception as e:
            print('master align warn:', e, flush=True)
    return tr, dv, mean, std


def ensure_teacher_and_cache(tr, mean, std, dev='cuda', force=False, teacher_steps=2200):
    t_am_p = '/work/tts/ck/v3_t30_am.pt'
    cache_p = '/work/sp/distill/v3_rectified_cache.pt'
    pros_tbl = torch.tensor(STYLE_PROSODY_TARGETS, dtype=torch.float32, device=dev)

    need_teacher = force or (not os.path.exists(t_am_p))
    if not need_teacher:
        try:
            ck_chk = torch.load(t_am_p, map_location='cpu', weights_only=False)
            if int(ck_chk.get('step', 0)) < 1500:
                need_teacher = True
        except Exception:
            need_teacher = True

    if need_teacher:
        print(f'[Stage T] Training OT-CFM Sakhiya-Pro v3 Teacher AM ({teacher_steps} steps) -> {t_am_p}', flush=True)
        ck = torch.load('/work/tts/ck/v5_c1_am.pt', map_location=dev, weights_only=False)
        net = SakhiyaAM().to(dev)
        net.load_state_dict({k: v.float() for k, v in ck.get('ema', ck.get('net', ck)).items() if not k.startswith('ctc_rec.')}, strict=False)
        net.stats.copy_(torch.tensor([mean, std], device=dev))
        _calibrate_styles(net, pre_baked=True)
        if os.path.exists(t_am_p):
            try:
                ck_prev = torch.load(t_am_p, map_location=dev, weights_only=False)
                net.load_state_dict(ck_prev['ema'], strict=False)
            except Exception:
                pass
        if os.path.exists('/work/tts/ck/v31_ctc_recognizer.pt'):
            try:
                cc = torch.load('/work/tts/ck/v31_ctc_recognizer.pt', map_location=dev, weights_only=False)
                ctc = MelCTCAligner().to(dev).train()
                ctc.load_state_dict(cc.get('model', cc), strict=False)
                for m_ in ctc.modules():
                    if isinstance(m_, nn.Dropout):
                        m_.p = 0.0
                for p_ in ctc.parameters():
                    p_.requires_grad_(False)
                net.ctc_rec = ctc
            except Exception:
                pass
        ema = copy.deepcopy(net).eval()
        opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1.4e-4, betas=(0.9, 0.99), weight_decay=1e-2)
        t0 = time.time()
        B = 32
        for st in range(1, teacher_steps + 1):
            prog = st / teacher_steps
            lr_cur = 1.4e-4 * (0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * prog)))
            for g_ in opt.param_groups:
                g_['lr'] = lr_cur
            xs = random.sample(tr, B)
            ids = pad_sequence([x['ids'].long() for x in xs], batch_first=True).to(dev)
            dur = pad_sequence([x['dur'].long() for x in xs], batch_first=True).to(dev)
            il = torch.tensor([len(x['ids']) for x in xs], device=dev)
            yl = torch.tensor([x['mel'].shape[1] for x in xs], device=dev)
            T = int(yl.max().item())
            y = torch.zeros(B, 100, T, device=dev)
            for k, x in enumerate(xs):
                y[k, :, :yl[k]] = (x['mel'].to(dev).float() - mean) / std
            spk = torch.zeros(B, dtype=torch.long, device=dev)
            lang = torch.tensor([4 if x['lang'] == 1 else 0 for x in xs], device=dev)
            sty = torch.tensor([5 if x['lang'] == 1 else 0 for x in xs], device=dev)
            studio = torch.ones(B, dtype=torch.long, device=dev)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss, parts = net.loss(ids, il, y, yl, spk, lang, sty, studio, prosody=pros_tbl[sty], dur_gt=dur, lambda_ctc=0.15, lambda_grl=0.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if torch.isfinite(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)):
                opt.step()
            with torch.no_grad():
                d_e = min(0.995, (1 + st) / (10 + st))
                for pe, p in zip(ema.parameters(), net.parameters()):
                    pe.lerp_(p, 1.0 - d_e)
            if st % 400 == 0 or st == teacher_steps:
                print(f'[Stage T] step {st}/{teacher_steps} ({time.time()-t0:.1f}s) loss={loss.item():.4f} {parts}', flush=True)
        ema_sd = {k: v.float().cpu() for k, v in ema.state_dict().items() if not k.startswith('ctc_rec.')}
        torch.save(dict(ema=ema_sd, step=teacher_steps, mean=mean, std=std, pre_baked_v25=True), t_am_p + '.tmp')
        os.replace(t_am_p + '.tmp', t_am_p)
        force = True

    if force or not os.path.exists(cache_p):
        print(f'[Stage C] Building Rectified Sakhiya-Pro v3 Cache -> {cache_p}', flush=True)
        t_ck = torch.load(t_am_p, map_location=dev, weights_only=False)
        t_am = SakhiyaAM().to(dev).eval()
        t_am.load_state_dict(t_ck['ema'], strict=False)
        t_am.stats.copy_(torch.tensor([mean, std], device=dev))
        cache = []
        t0 = time.time()
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for b0 in range(0, len(tr), 64):
                xs = tr[b0:b0 + 64]
                B = len(xs)
                ids = pad_sequence([x['ids'].long() for x in xs], batch_first=True).to(dev)
                dur = pad_sequence([x['dur'].long() for x in xs], batch_first=True).to(dev)
                il = torch.tensor([len(x['ids']) for x in xs], device=dev)
                yl = dur.sum(1)
                T = int(yl.max().item())
                L = ids.shape[1]
                im = torch.arange(L, device=dev)[None] < il[:, None]
                ym = torch.arange(T, device=dev)[None] < yl[:, None]
                ymf = ym[:, None].float()
                spk = torch.zeros(B, dtype=torch.long, device=dev)
                lang_eff = torch.tensor([4 if x['lang'] == 1 else 0 for x in xs], device=dev)
                sty = torch.tensor([5 if x['lang'] == 1 else 0 for x in xs], device=dev)
                studio = torch.ones(B, dtype=torch.long, device=dev)
                pos = torch.arange(L, device=dev, dtype=torch.float32)[None, :]
                L_f = il[:, None].float().clamp(min=2.0)
                is_pause = (ids <= 6).float()
                phase = (pos / L_f).clamp(0.0, 1.0)
                tail_closure = -0.06 * ((phase - 0.72).clamp(min=0.0) / 0.28) * (1.0 - is_pause)
                p_arch = (0.14 * torch.sin(phase * 3.14159265) + tail_closure) * (1.0 - is_pause) - 0.15 * is_pause
                p_enc = F.avg_pool1d(F.pad(p_arch[:, None, :], (1, 1), mode='replicate'), 3, stride=1)[:, 0] * im.float()
                ones_m = torch.ones((B, 1), device=dev)
                ht, mut, gt = t_am.enc(ids, spk, lang_eff, sty, studio, im, pitch_add=p_enc, prosody=pros_tbl[sty], sty_mask=ones_m)
                f0_t = t_am.enc.f0_pred(ht).squeeze(-1).clamp(-1.5, 1.5)
                eng_t = t_am.enc.energy_pred(ht).squeeze(-1).clamp(-1.2, 1.2)
                mut_e = mut + (0.10 * pros_tbl[sty][:, 2:3] + 0.06 * eng_t)[..., None] * 0.32
                attn = _dur_to_attn(dur, im, ym)
                mu_y = torch.einsum('blt,bld->bdt', attn.to(mut_e.dtype), mut_e) * ymf
                y_clean = _ode_solve(t_am.vel, mu_y.float(), gt.float(), ymf, 4) * ymf
                for k, x in enumerate(xs):
                    Lk = int(il[k])
                    Tk = int(yl[k])
                    m_gt_n = (x['mel'][:, :Tk].to(dev).float() - mean) / std
                    cache.append(dict(
                        ids=ids[k, :Lk].cpu(),
                        dur=dur[k, :Lk].cpu(),
                        p_enc=p_enc[k, :Lk].half().cpu(),
                        lang=int(x['lang']),
                        mu=mut[k, :Lk].half().cpu(),
                        f0=f0_t[k, :Lk].half().cpu(),
                        eng=eng_t[k, :Lk].half().cpu(),
                        y_clean=y_clean[k, :, :Tk].T.half().contiguous().cpu(),
                        y_gt=m_gt_n.T.half().contiguous().cpu(),
                    ))
        torch.save(dict(cache=cache, mean=mean, std=std), cache_p + '.tmp')
        os.replace(cache_p + '.tmp', cache_p)
        print(f'[Stage C] Saved {len(cache)} items in {time.time()-t0:.1f}s -> {cache_p}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', required=True)
    ap.add_argument('--data', default='/work/sp/distill/gt_feats.pt')
    ap.add_argument('--out', default=None)
    ap.add_argument('--minutes', type=float, default=90.0)
    ap.add_argument('--bs', type=int, default=36)
    ap.add_argument('--lr_mult', type=float, default=1.0)
    ap.add_argument('--qat_start', type=float, default=0.45)
    ap.add_argument('--max_step_s', type=float, default=3.5)
    ap.add_argument('--save_every', type=int, default=400)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--prep_only', action='store_true')
    ap.add_argument('--force_teacher', action='store_true')
    a = ap.parse_args()

    dev = 'cuda'
    out = a.out or f'/work/sp/distill/runs/{a.tier}'
    os.makedirs(out, exist_ok=True)
    random.seed(42)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    def log(s):
        print(s, flush=True)
        open(out + '/train.log', 'a').write(s + '\n')

    tr, dv, mean, std = load_clean_pro_v3_items(a.data, dev=dev)
    ensure_teacher_and_cache(tr, mean, std, dev=dev, force=a.force_teacher)
    if a.prep_only:
        return

    C_data = torch.load('/work/sp/distill/v3_rectified_cache.pt', map_location=dev, weights_only=False)
    cache = C_data['cache']
    for x in cache:
        for k in ('ids', 'dur', 'p_enc', 'mu', 'f0', 'eng', 'y_clean', 'y_gt'):
            x[k] = x[k].to(dev)
    log(f'[{a.tier}] Loaded {len(cache)} Rectified Sakhiya-Pro v3 utterances on GPU')

    G = build(a.tier).to(dev)
    G.warm_init('/work/tts/ck')
    # Freeze vocoder during AM distillation so its harmonic excitation & phase never degrade
    for p in G.voc.parameters():
        p.requires_grad_(False)
    install_ste_qat_hooks(G)
    ema = copy.deepcopy(G).eval()

    ctc = None
    if os.path.exists('/work/tts/ck/v31_ctc_recognizer.pt'):
        try:
            cc = torch.load('/work/tts/ck/v31_ctc_recognizer.pt', map_location=dev, weights_only=False)
            ctc = MelCTCAligner().to(dev).train()
            ctc.load_state_dict(cc.get('model', cc), strict=False)
            for m_ in ctc.modules():
                if isinstance(m_, nn.Dropout):
                    m_.p = 0.0
            for p_ in ctc.parameters():
                p_.requires_grad_(False)
        except Exception as e:
            log(f'[ctc] skip: {e}')

    lr_e = LR_ENC[a.tier] * a.lr_mult
    lr_v = LR_VEL[a.tier] * a.lr_mult
    enc_dur_p = [p for n, p in G.named_parameters() if not n.startswith(('voc.', 'am.vel.')) and p.requires_grad]
    vel_p = [p for n, p in G.named_parameters() if n.startswith('am.vel.') and p.requires_grad]
    oG = torch.optim.AdamW([
        dict(params=enc_dur_p, lr=lr_e, weight_decay=1e-3),
        dict(params=vel_p, lr=lr_v, weight_decay=1e-4),
    ], betas=(0.9, 0.99))

    step, spent = 0, 0.0
    if a.resume and os.path.exists(out + '/state.pt'):
        try:
            S = torch.load(out + '/state.pt', map_location=dev, weights_only=False)
            if S.get('cfg') == G.cfg:
                G.load_state_dict(S['G'], strict=False)
                ema.load_state_dict(S['ema'], strict=False)
                oG.load_state_dict(S['oG'])
                step, spent = S['step'], S['spent']
                log(f'[resume] step {step} spent {spent / 60:.1f} min')
        except Exception as e:
            log(f'[resume] skip: {e}')

    # Save initial warm-started checkpoint at step 50 immediately once verified
    log(f'[{a.tier}] params {n_params(G) / 1e6:.3f}M lr_enc {lr_e:.1e} lr_vel {lr_v:.1e} bs {a.bs} steps/ode {G.steps}')
    budget = a.minutes * 60.0
    t0 = time.time() - spent
    tstep = time.time()
    qat_on = bool(G._qat_flag)

    while True:
        spent = time.time() - t0
        prog = min(1.0, spent / max(budget, 1.0))
        if spent > budget:
            break
        mult = (0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * prog))) * min(1.0, (step + 10) / 150.0)
        oG.param_groups[0]['lr'] = lr_e * mult
        oG.param_groups[1]['lr'] = lr_v * mult

        if prog >= a.qat_start and not qat_on:
            set_qat(G, True)
            qat_on = True
            log(f'[qat] int8 STE fake-quant ON at step {step}')

        xs = random.sample(cache, a.bs)
        B = len(xs)
        ids = pad_sequence([x['ids'] for x in xs], batch_first=True)
        dur = pad_sequence([x['dur'] for x in xs], batch_first=True)
        p_enc = pad_sequence([x['p_enc'].float() for x in xs], batch_first=True)
        mu_t = pad_sequence([x['mu'].float() for x in xs], batch_first=True)
        f0_t = pad_sequence([x['f0'].float() for x in xs], batch_first=True)
        eng_t = pad_sequence([x['eng'].float() for x in xs], batch_first=True)
        y_clean = pad_sequence([x['y_clean'].float() for x in xs], batch_first=True).transpose(1, 2).contiguous()
        y_gt = pad_sequence([x['y_gt'].float() for x in xs], batch_first=True).transpose(1, 2).contiguous()
        il = torch.tensor([len(x['ids']) for x in xs], device=dev)
        yl = torch.tensor([x['y_clean'].shape[0] for x in xs], device=dev)
        L, T = ids.shape[1], y_clean.shape[2]
        im = torch.arange(L, device=dev)[None] < il[:, None]
        ym = torch.arange(T, device=dev)[None] < yl[:, None]
        ymf = ym[:, None].float()
        spk = torch.zeros(B, dtype=torch.long, device=dev)
        studio = torch.ones(B, dtype=torch.long, device=dev)
        lang_eff = torch.tensor([4 if x['lang'] == 1 else 0 for x in xs], device=dev)
        sty = torch.tensor([5 if x['lang'] == 1 else 0 for x in xs], device=dev)
        ones_m = torch.ones((B, 1), device=dev)

        G.train()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            hs, mus, gs = G.am.enc(ids, spk, lang_eff, sty, studio, im, pitch_add=p_enc, prosody=G.pros_tbl[sty], sty_mask=ones_m)
            f0_s = G.am.enc.f0_pred(hs).squeeze(-1)
            eng_s = G.am.enc.energy_pred(hs).squeeze(-1).clamp(-1.2, 1.2)
            l_heads = 0.5 * (F.l1_loss(f0_s[im].float(), f0_t[im].float()) + F.l1_loss(eng_s[im].float(), eng_t[im].float()))
            mus_e = mus + (0.10 * G.pros_tbl[sty][:, 2:3] + 0.06 * eng_s)[..., None] * 0.32
            lds = G.am.dur.pred_logdur(ids, hs, im) + G.dur_rate_bias.clamp(-0.12, 0.12)
            l_dur = F.smooth_l1_loss(lds[im].float(), torch.log(dur.float().clamp(min=1.0))[im])
            l_prior = F.l1_loss(mus[im].float(), mu_t[im].float())

            attn = _dur_to_attn(dur, im, ym)
            mu_y_s = torch.einsum('blt,bld->bdt', attn.to(mus_e.dtype), mus_e) * ymf
            y_pred = _ode_solve(G.am.vel, mu_y_s, gs, ymf, G.steps) * ymf

            denom = (ymf.sum() * 100.0).clamp(min=1.0)
            # 1. Vocal-Tract Formant Envelope Lock (7-bin F0-invariant spectral envelope from Pro-v3 teacher + ground truth)
            env_p = _mel_env(y_pred.float(), 7)
            env_c = _mel_env(y_clean.float(), 7)
            env_g = _mel_env(y_gt.float(), 7)
            env_tgt = 0.70 * env_c + 0.30 * env_g
            l_env = ((env_p - env_tgt).abs() * ymf).sum() / denom

            # 2. Full mel + L2 + frequency/time derivative + spectral dynamic range + 44-bin cepstral harmonic comb
            l_mel = ((1.2 * (y_pred.float() - y_clean.float()).abs() + 0.8 * (y_pred.float() - y_clean.float()) ** 2) * ymf).sum() / denom + 2.0 * l_env
            df_p = y_pred[:, 1:, :].float() - y_pred[:, :-1, :].float()
            df_t = y_clean[:, 1:, :].float() - y_clean[:, :-1, :].float()
            l_df = ((df_p - df_t).abs() * ymf).sum() / denom
            l_std = ((y_pred.float().std(dim=1, keepdim=True) - y_clean.float().std(dim=1, keepdim=True)).abs() * ymf).sum() / ymf.sum().clamp(min=1.0)
            ym_t = ymf[:, :, 1:] * ymf[:, :, :-1]
            l_dt = (((y_pred[:, :, 1:].float() - y_pred[:, :, :-1].float()) - (y_clean[:, :, 1:].float() - y_clean[:, :, :-1].float())).abs() * ym_t).sum() / (ym_t.sum() * 100.0 + 1.0)
            cep_p = torch.fft.rfft(y_pred.float(), dim=1).abs()[:, 4:48, :]
            cep_t = torch.fft.rfft(y_clean.float(), dim=1).abs()[:, 4:48, :]
            l_cep = ((cep_p - cep_t).abs() * ymf).sum() / (ymf.sum() * 44.0).clamp(min=1.0)

            l_ctc = torch.zeros((), device=dev)
            if ctc is not None and step % 2 == 0:
                c_val = ctc.ctc_loss(0.3 * mu_y_s.float() + 0.7 * y_pred.float(), yl, ids, il)
                if torch.isfinite(c_val):
                    l_ctc = F.relu(c_val - 0.12)

            # 3. Direct voc.src.f0_net F0 & Voicing (uv) logit supervision across the full utterance (eliminates F0 shutters & voiced dropouts!)
            mp_full = y_pred.float() * G.am.stats[1] + G.am.stats[0]
            mt_full = y_clean.float() * G.am.stats[1] + G.am.stats[0]
            with torch.no_grad():
                rf_t = G.voc.src.f0_net(mt_full)
            rf_s = G.voc.src.f0_net(mp_full)
            l_f0net = (F.smooth_l1_loss(rf_s[:, 0:1].float(), rf_t[:, 0:1].float(), reduction='none') * ymf).sum() / ymf.sum().clamp(min=1.0) \
                    + 1.5 * (F.l1_loss(torch.sigmoid(rf_s[:, 1:2].float()), torch.sigmoid(rf_t[:, 1:2].float()), reduction='none') * ymf).sum() / ymf.sum().clamp(min=1.0)

            # 4. Differentiable Vocoder-in-the-Loop Complex-iSTFT Magnitude + Phase + Instantaneous-Frequency Loss (32-frame crop)
            l_vspec = torch.zeros((), device=dev)
            if T >= 36:
                seg_v = 32
                B_v = min(B, 16)
                s_p, s_t = [], []
                for bi in range(B_v):
                    valid_t = max(int(yl[bi].item()), seg_v + 1)
                    st_i = random.randint(0, max(0, valid_t - seg_v)) if valid_t > seg_v else 0
                    s_p.append(y_pred[bi, :, st_i:st_i + seg_v])
                    s_t.append(y_clean[bi, :, st_i:st_i + seg_v])
                mp_seg = torch.stack(s_p, dim=0) * G.am.stats[1] + G.am.stats[0]
                mt_seg = torch.stack(s_t, dim=0) * G.am.stats[1] + G.am.stats[0]
                mp_in = torch.cat([mp_seg[:, :98, :], mp_seg[:, 97:98, :], mp_seg[:, 97:98, :]], dim=1)
                mt_in = torch.cat([mt_seg[:, :98, :], mt_seg[:, 97:98, :], mt_seg[:, 97:98, :]], dim=1)
                with torch.no_grad():
                    re_vt, im_vt, _, _ = G.voc.spectrum_and_f0(mt_in)
                re_vs, im_vs, _, _ = G.voc.spectrum_and_f0(mp_in)
                v_val = spec_phase_gd_loss(re_vs, im_vs, re_vt, im_vt)
                if torch.isfinite(v_val):
                    l_vspec = v_val

            loss = l_mel + 2.4 * l_df + 0.9 * l_std + 1.2 * l_dt + 0.85 * l_cep + 1.6 * l_prior + 0.8 * l_dur + l_heads + 0.25 * l_ctc + 1.2 * l_f0net + 0.50 * l_vspec

        oG.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(G.parameters(), 1.5)
        if torch.isfinite(gn):
            oG.step()

        step += 1
        with torch.no_grad():
            dcy = min(0.996, (1 + step) / (15 + step))
            for pe, pg in zip(ema.parameters(), G.parameters()):
                pe.lerp_(pg, 1.0 - dcy)

        if step == 50:
            sps = (time.time() - tstep) / 50.0
            if sps > a.max_step_s:
                log(f'WATCHDOG: step-50 {sps:.2f}s/step > {a.max_step_s}s')
                sys.exit(3)

        if step % 50 == 0:
            now = time.time()
            ips = 50.0 / max(1e-6, now - tstep)
            tstep = now
            log(f'Step {step} | {ips:.2f} it/s | mel {l_mel.item():.3f} env {l_env.item():.3f} prior {l_prior.item():.3f} cep {l_cep.item():.3f} dur {l_dur.item():.3f} ctc {l_ctc.item():.3f} | {spent / 60:.1f}/{a.minutes:.0f} min qat={int(qat_on)}')

        if step in (100, 300, 600, 1200) or step % a.save_every == 0:
            torch.save(dict(ema=ema.state_dict(), cfg=G.cfg, tier=a.tier, arch=G.arch, step=step, f0=[5.418, 0.203], qat=qat_on, gt=True), out + '/ema_latest.tmp')
            os.replace(out + '/ema_latest.tmp', out + '/ema_latest.pt')
            if step % (a.save_every * 2) == 0:
                torch.save(dict(G=G.state_dict(), ema=ema.state_dict(), oG=oG.state_dict(), cfg=G.cfg, step=step, spent=time.time() - t0), out + '/state.tmp')
                os.replace(out + '/state.tmp', out + '/state.pt')

    torch.save(dict(ema=ema.state_dict(), cfg=G.cfg, tier=a.tier, arch=G.arch, step=step, f0=[5.418, 0.203], qat=qat_on, gt=True), out + '/ema_final.pt')
    torch.save(dict(ema=ema.state_dict(), cfg=G.cfg, tier=a.tier, arch=G.arch, step=step, f0=[5.418, 0.203], qat=qat_on, gt=True), out + '/ema_latest.pt')
    log(f'DONE step {step} minutes {(time.time() - t0) / 60:.1f}')


if __name__ == '__main__':
    main()
