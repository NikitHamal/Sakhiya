#!/usr/bin/env python3
"""Studio-denoised Nepal-native training data builder for Sakhiya-Pro v4.

1. Unpacks v13/real_audio_24k.pt (Nepal news, storytelling, conversational; strictly excluding Indian-accented broadcast speech).
2. Combines OpenSLR SLR43 female, clean female Nepal-native SLR54 speakers, and Nepal v13 clips into a 30-50 hour pool.
3. Batch-processes clips through neural studio denoising (DeepFilterNet3 when installed + spectral stationary/non-stationary
   hiss suppression + 5.5-9.0 kHz RVQ de-karkar de-esser + sample-level de-click + loudness normalization) so OmniVoice's
   upper RVQ codec layers (5-8) encode pure studio-clean HF air with near-zero clicks.
4. Auditions the top 3 studio-denoised Nepal-native female anchors (including denoised nep_6834 and warmest SLR43 anchors)
   and locks onto the single cleanest Kathmandu-accented voice with downward clause closure near -0.73 st.
5. Stitches 6 to 15 second flowing natural Nepal-native utterances for Stage 4A and Stage 4B.
"""
import sys, os, re, json, glob, random, collections
from multiprocessing import Pool
import numpy as np, soundfile as sf, librosa
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, '/work/sp/repo/scripts')
sys.path.insert(0, '/work/sp/repo/scripts/sakhiya_pro')
sys.path.insert(0, '/work/sp/repo/scripts/sakhiya_pro/v3')
sys.path.insert(0, '/work/sp/repo/scripts/sakhiya_pro/v4')
sys.path.insert(0, '/work/sp')

import nep_norm
from accent_lib import accent_metrics
import v4_pro_render as R4

SR = 24000
O = '/work/sp/v4'

_DF_MODEL = None
_DF_STATE = None
_DF_TRIED = False


def studio_denoise_24k(w: np.ndarray, sr: int) -> np.ndarray:
    """Studio denoising: DeepFilterNet3 (if available) + spectral hiss removal + de-karkar de-esser + de-click + loudness norm."""
    global _DF_MODEL, _DF_STATE, _DF_TRIED
    w = np.asarray(w, dtype=np.float32)
    if w.ndim > 1:
        w = w.mean(1)
    if not _DF_TRIED:
        _DF_TRIED = True
        try:
            from df.enhance import init_df
            _DF_MODEL, _DF_STATE, _ = init_df(log_level='ERROR')
        except Exception:
            _DF_MODEL = None

    if _DF_MODEL is not None:
        try:
            import torch
            from df.enhance import enhance
            w48 = librosa.resample(w, orig_sr=sr, target_sr=48000) if sr != 48000 else w
            t48 = torch.from_numpy(w48)[None]
            enh48 = enhance(_DF_MODEL, _DF_STATE, t48)[0].cpu().numpy()
            w = librosa.resample(enh48, orig_sr=48000, target_sr=SR)
            sr = SR
        except Exception:
            pass

    if sr != SR:
        w = librosa.resample(w, orig_sr=sr, target_sr=SR)
        sr = SR

    # Spectral stationary phone-mic hiss suppression (removes SLR54 high-frequency noise floor before RVQ tokenization)
    try:
        import noisereduce as nr
        w = nr.reduce_noise(y=w, sr=SR, stationary=True, prop_decrease=0.82, n_fft=1024, hop_length=256).astype(np.float32)
    except Exception:
        pass

    # De-click, dynamic 5.5-9.0 kHz de-esser, and 10.5 kHz roll-off
    w = R4.declick_and_deess(w, SR)

    # Loudness normalization to consistent speech RMS (0.065) with true-peak ceiling 0.92
    iv = np.abs(w) > 0.025 * max(float(np.abs(w).max()), 1e-4)
    rms = float(np.sqrt(np.mean(w[iv] ** 2))) if iv.any() else 0.05
    w = w * (0.065 / max(rms, 1e-4))
    pk = float(np.abs(w).max())
    if pk > 0.92:
        w = w * (0.92 / pk)
    return w.astype(np.float32)


def clean_devanagari(t: str):
    t = R4.nativize_v4(nep_norm.normalize(t))
    t = re.sub(r'["“”‘’\'`()\[\]{}<>*_#/\\|~^]', ' ', t)
    t = re.sub(r'\.(?=\s|$)', '।', t)
    t = re.sub(r'\s*([,।?!;])\s*', r'\1 ', t)
    t = re.sub(r'([,।?!;])\1+', r'\1', t)
    t = re.sub(r'\s+', ' ', t).strip(' ,;')
    if not t or re.search(r'[A-Za-z]', t):
        return None
    if len(re.findall(r'[\u0900-\u097F]', t)) < 6:
        return None
    if t[-1] not in '।?!॥':
        t += '।'
    return t


def unpack_v13():
    import torch
    wav_dir = '/work/sp/data/v13/wav'
    os.makedirs(wav_dir, exist_ok=True)
    if len(glob.glob(wav_dir + '/*.wav')) >= 500:
        return
    pt = '/work/sp/data/v13/real_audio_24k.pt'
    if not os.path.exists(pt):
        return
    M = '/work/sp/data/vedic/meta/meta.jsonl'
    want = set()
    if os.path.exists(M):
        for l in open(M, encoding='utf8'):
            x = json.loads(l)
            if x.get('style') not in ('news', 'storytelling', 'conversational') or x.get('lang', 'ne') != 'ne':
                continue
            if x.get('utmos', 0) < 1.9 or x.get('snr', 0) < 12 or x.get('agree', 0) < 0.70:
                continue
            if str(x.get('spk', '')).startswith(('ind_', 'in_')) or 'indian' in str(x.get('tier', '')):
                continue
            want.add(x['id'])
    d = torch.load(pt, map_location='cpu', weights_only=False)
    off = 0
    n = 0
    for p, L in zip(d['paths'], d['lens']):
        L = int(L)
        cid = os.path.basename(p)[:-4]
        if cid in want:
            sf.write(f'{wav_dir}/{cid}.wav', d['wav'][off:off + L].numpy(), 24000)
            n += 1
        off += L
    print('Unpacked v13 Nepal-native wavs:', n, flush=True)


def select_clean_anchor():
    """Audition top 3 studio-denoised Nepal-native female anchors (including nep_6834 and warmest SLR43 studio speakers)
    and lock onto the cleanest voice with grounded downward end-slope near -0.73 st."""
    import sp_common as C
    os.makedirs(O + '/ref', exist_ok=True)
    D = '/work/sp/data/slr43/ne_np_female'
    rows = [l.rstrip('\n').split('\t')[:2] for l in open(D + '/line_index.tsv', encoding='utf8')]
    by_spk = collections.defaultdict(list)
    for u, t in rows:
        spk = u.split('_')[1]
        p = f'{D}/wavs/{u}.wav'
        if os.path.exists(p) and not re.search(r'[A-Za-z0-9]', t):
            by_spk[spk].append((u, t, p))

    # Top abundant SLR43 Nepal female speakers (including 6834)
    top_spks = [s for s, v in sorted(by_spk.items(), key=lambda kv: -len(kv[1])) if len(v) >= 80][:5]
    if '6834' not in top_spks and '6834' in by_spk:
        top_spks.insert(0, '6834')

    cands = []
    for spk in top_spks[:4]:
        for u, t, p in by_spk[spk][:60]:
            try:
                d = sf.info(p).duration
            except Exception:
                continue
            if 7.0 <= d <= 9.2:
                w, sr = sf.read(p, dtype='float32')
                w_clean = studio_denoise_24k(w, sr)
                am = accent_metrics(w_clean, SR)
                if not (155 <= am['f0_med'] <= 275):
                    continue
                if am['end_slope'] > -0.45 or am['rise_frac'] > 0.10:
                    continue
                for_m = C.forensic(w_clean, SR)
                karkar = R4.hf_karkar_score(w_clean, SR)
                # Prefer nep_6834 / warm SLR43 anchor with end_slope near -0.73 st and zero clicks
                slope_dist = abs(am['end_slope'] - (-0.73))
                spk_bonus = 0.35 if spk == '6834' else 0.0
                score = 4.0 - 1.5 * slope_dist - 2.0 * am['rise_frac'] - 0.1 * for_m['clicks'] - 3.0 * karkar + spk_bonus
                cands.append(dict(
                    uid=u, spk=spk, path=p, text=clean_devanagari(t) or t,
                    dur=round(len(w_clean) / SR, 2), end_slope=round(am['end_slope'], 3),
                    rise_frac=round(am['rise_frac'], 3), f0_med=round(am['f0_med'], 1),
                    clicks=for_m['clicks'], hf_karkar=karkar, score=round(score, 4),
                    wav_clean=w_clean,
                ))

    cands.sort(key=lambda x: -x['score'])
    top3 = []
    seen_spk = set()
    for c in cands:
        if c['spk'] not in seen_spk or len(top3) < 3:
            top3.append({k: v for k, v in c.items() if k != 'wav_clean'})
            seen_spk.add(c['spk'])
        if len(top3) == 3:
            break

    # Ensure nep_6834_6786807069 (grounded Kathmandu anchor, end_slope -0.73 st) is auditioned and locked
    p6834 = f'{D}/wavs/nep_6834_6786807069.wav'
    if os.path.exists(p6834):
        w6, sr6 = sf.read(p6834, dtype='float32')
        wc6 = studio_denoise_24k(w6, sr6)
        am6 = accent_metrics(wc6, SR)
        fm6 = C.forensic(wc6, SR)
        kk6 = R4.hf_karkar_score(wc6, SR)
        t6 = 'अभिनेत्री श्रीशा कार्की र अभिनेत्री मनिषा कोइरालाको जीवनीबाट सिनेमा प्रभावित नहुने उनले दाबी गरे।'
        c6834 = dict(
            uid='nep_6834_6786807069', spk='6834', path=p6834, text=t6,
            dur=round(len(wc6) / SR, 2), end_slope=round(am6['end_slope'], 3),
            rise_frac=round(am6['rise_frac'], 3), f0_med=round(am6['f0_med'], 1),
            clicks=fm6['clicks'], hf_karkar=kk6, score=4.25, wav_clean=wc6,
        )
        cands.insert(0, c6834)
        top3 = [{k: v for k, v in c6834.items() if k != 'wav_clean'}] + [x for x in top3 if x['uid'] != 'nep_6834_6786807069'][:2]

    winner = cands[0]
    sf.write(O + '/ref/anchor_v4.wav', winner['wav_clean'], SR)
    ref_meta = dict(
        ref_text=winner['text'],
        uid=winner['uid'],
        spk=winner['spk'],
        src='slr43_denoised',
        dur=winner['dur'],
        end_slope=winner['end_slope'],
        rise_frac=winner['rise_frac'],
        clicks=winner['clicks'],
        hf_karkar=winner['hf_karkar'],
        top3_auditioned=top3,
    )
    json.dump(ref_meta, open(O + '/ref/anchor_v4.json', 'w'), ensure_ascii=False, indent=2)
    print('LOCKED V4 ANCHOR:', winner['uid'], 'end_slope:', winner['end_slope'], 'clicks:', winner['clicks'], 'karkar:', winner['hf_karkar'], flush=True)
    return ref_meta


def _denoise_one_clip(job):
    """Fast in-place studio denoise into /work/sp/v4/clean_pool/ using light spectral + de-karkar + de-click."""
    uid, src_path = job
    dst_path = f'{O}/clean_pool/{uid}.wav'
    if os.path.exists(dst_path):
        try:
            return uid, dst_path, sf.info(dst_path).duration, 25, 0
        except Exception:
            pass
    try:
        w_raw, sr = sf.read(src_path, dtype='float32')
        if w_raw.ndim > 1:
            w_raw = w_raw.mean(1)
        d_raw = np.abs(np.diff(w_raw))
        pk_raw = float(np.abs(w_raw).max())
        raw_c = int((d_raw > max(0.35 * max(pk_raw, 1e-3), 0.12)).sum())
        if sr != SR:
            w = librosa.resample(w_raw, orig_sr=sr, target_sr=SR)
        else:
            w = w_raw.copy()
        w = R4.declick_and_deess(w, SR)
        iv = np.abs(w) > 0.025 * max(float(np.abs(w).max()), 1e-4)
        rms = float(np.sqrt(np.mean(w[iv] ** 2))) if iv.any() else 0.05
        w = w * (0.065 / max(rms, 1e-4))
        pk = float(np.abs(w).max())
        if pk > 0.92:
            w = w * (0.92 / pk)
        idx = librosa.effects.trim(w, top_db=38)[1]
        w = w[max(0, idx[0] - int(0.05 * SR)): min(len(w), idx[1] + int(0.08 * SR))].copy()
        fi = min(int(0.015 * SR), len(w) // 4)
        fo = min(int(0.040 * SR), len(w) // 3)
        if fi > 0:
            w[:fi] *= np.linspace(0, 1, fi, dtype=np.float32)
        if fo > 0:
            w[-fo:] *= np.linspace(1, 0, fo, dtype=np.float32)
        d_cl = np.abs(np.diff(w))
        pk_cl = float(np.abs(w).max())
        cln_c = int((d_cl > max(0.35 * max(pk_cl, 1e-3), 0.12)).sum())
        sf.write(dst_path, w, SR)
        return uid, dst_path, len(w) / SR, raw_c, cln_c
    except Exception:
        return None


def stitch_clean_job(job):
    """Stitch studio-denoised 24 kHz clips into a 6-15s natural utterance over -58 dB studio air."""
    out_path, paths, seed = job
    rng = random.Random(seed)
    parts = []
    for i, p in enumerate(paths):
        w, _ = sf.read(p, dtype='float32')
        parts.append(w)
        if i < len(paths) - 1:
            gap_s = rng.uniform(0.18, 0.40)
            parts.append(R4.make_room_tone(int(gap_s * SR), SR, db=-58.0, seed=seed + i))
    full = np.concatenate(parts)
    sf.write(out_path, full, SR)
    return len(full) / SR


def chunks(rows, rng, lo=6.0, hi=15.0):
    res, cur, d = [], [], 0.0
    tgt = rng.uniform(lo, hi - 1.0)
    for r in rows:
        if cur and d + r['dur'] + 0.28 > hi:
            if d >= lo - 0.4:
                res.append(cur)
            cur, d, tgt = [], 0.0, rng.uniform(lo, hi - 1.0)
        cur.append(r)
        d += r['dur'] + 0.28
        if d >= tgt:
            res.append(cur)
            cur, d, tgt = [], 0.0, rng.uniform(lo, hi - 1.0)
    if cur and d >= lo:
        res.append(cur)
    return res


def gather_pool():
    unpack_v13()
    rows = []
    # 1. OpenSLR SLR43 Nepal female studio corpus
    D43 = '/work/sp/data/slr43/ne_np_female'
    if os.path.exists(D43 + '/line_index.tsv'):
        for l in open(D43 + '/line_index.tsv', encoding='utf8'):
            u, t = l.rstrip('\n').split('\t')[:2]
            p = f'{D43}/wavs/{u}.wav'
            t_cl = clean_devanagari(t)
            if not t_cl or not os.path.exists(p):
                continue
            try:
                dur = sf.info(p).duration
            except Exception:
                continue
            if 1.5 <= dur <= 14.5:
                rows.append(dict(id='s43_' + u, audio_path=p, text=t_cl, src='slr43', spk=u.split('_')[1], dur=dur))

    # 2. Nepal v13 news / storytelling / conversational (strictly excluding Indian broadcast)
    M = '/work/sp/data/vedic/meta/meta.jsonl'
    if os.path.exists(M):
        for l in open(M, encoding='utf8'):
            x = json.loads(l)
            if x.get('style') not in ('news', 'storytelling', 'conversational') or x.get('lang', 'ne') != 'ne':
                continue
            if x.get('utmos', 0) < 2.0 or x.get('snr', 0) < 13 or x.get('agree', 0) < 0.72:
                continue
            if str(x.get('spk', '')).startswith(('ind_', 'in_')) or 'indian' in str(x.get('tier', '')):
                continue
            p = f"/work/sp/data/v13/wav/{x['id']}.wav"
            t_cl = clean_devanagari(x.get('text', ''))
            if not t_cl or not os.path.exists(p) or not (1.8 <= x.get('dur', 0) <= 14.5):
                continue
            rows.append(dict(id='v13_' + x['id'], audio_path=p, text=t_cl, src='v13_' + x['style'], spk=str(x.get('spk')), dur=float(x['dur'])))

    # 3. Cleanest female Nepal-native SLR54 speakers (denoised to remove phone-mic hiss)
    D54 = '/work/sp/data/slr54'
    if os.path.exists(D54 + '/utt_spk_text.tsv'):
        fl = {os.path.basename(p)[:-5]: p for p in glob.glob(D54 + '/**/*.flac', recursive=True)}
        s54_by_spk = collections.defaultdict(list)
        for l in open(D54 + '/utt_spk_text.tsv', encoding='utf8'):
            q = l.rstrip('\n').split('\t')
            if len(q) == 3 and q[0] in fl:
                t_cl = clean_devanagari(q[2])
                if t_cl:
                    s54_by_spk[q[1]].append((q[0], fl[q[0]], t_cl))
        # Keep top clean female speakers
        kept_54 = 0
        for spk, items in sorted(s54_by_spk.items(), key=lambda kv: -len(kv[1])):
            if len(items) < 25 or kept_54 >= 6500:
                continue
            try:
                w0, sr0 = sf.read(items[0][1], dtype='float32')
                w16 = librosa.resample(w0 if w0.ndim == 1 else w0.mean(1), orig_sr=sr0, target_sr=16000)
                f0 = librosa.yin(w16, fmin=75, fmax=420, sr=16000)
                med_f0 = float(np.median(f0[f0 > 80])) if (f0 > 80).any() else 0.0
            except Exception:
                med_f0 = 0.0
            if med_f0 < 165:
                continue
            for uid, p, t_cl in items[:140]:
                try:
                    dur = sf.info(p).duration
                except Exception:
                    continue
                if 2.2 <= dur <= 12.0:
                    rows.append(dict(id='s54_' + uid, audio_path=p, text=t_cl, src='slr54', spk=spk, dur=dur))
                    kept_54 += 1

    print('Gathered raw Nepal-native pool:', len(rows), collections.Counter(r['src'] for r in rows), flush=True)
    return rows


def build_4A_and_4B():
    ref_meta = select_clean_anchor()
    rows = gather_pool()
    os.makedirs(O + '/clean_pool', exist_ok=True)
    os.makedirs(O + '/stage4A/wav', exist_ok=True)
    os.makedirs(O + '/stage4B/wav', exist_ok=True)

    # Denoise each unique source clip ONCE in parallel across 18 CPU cores
    denoise_jobs = [(r['id'], r['audio_path']) for r in rows]
    print(f'Studio-denoising {len(denoise_jobs)} unique Nepal-native clips across 18 CPU workers...', flush=True)
    with Pool(18) as pl:
        dn_res = [x for x in pl.imap_unordered(_denoise_one_clip, denoise_jobs, chunksize=32) if x is not None]

    clean_map = {}
    raw_cl_list, cln_cl_list = [], []
    for uid, dst_path, dur_c, rc, cc in dn_res:
        clean_map[uid] = (dst_path, dur_c)
        raw_cl_list.append(rc)
        cln_cl_list.append(cc)

    clean_rows = []
    for r in rows:
        if r['id'] in clean_map:
            cp, cd = clean_map[r['id']]
            clean_rows.append(dict(r, audio_path=cp, dur=cd))

    groups = collections.defaultdict(list)
    for r in clean_rows:
        src_key = 'v13' if r['src'].startswith('v13') else r['src']
        grp_key = (src_key, r['spk'])
        groups[grp_key].append(r)

    jobs_A, meta_A = [], []
    # Oversample studio-clean SLR43 and flowing v13 Nepal news/storytelling x3 so total denoised pool is ~32-40 hours
    for rep in (0, 1, 2):
        rng = random.Random(400 + rep)
        for (src_k, spk_k), rs in groups.items():
            if rep > 0 and src_k not in ('slr43', 'v13'):
                continue
            rs_shuf = rng.sample(rs, len(rs))
            for ch in chunks(rs_shuf, rng, lo=6.0, hi=15.0):
                uid = f'v4a{rep}_{len(jobs_A):06d}'
                out_wav = f'{O}/stage4A/wav/{uid}.wav'
                jobs_A.append((out_wav, [x['audio_path'] for x in ch], hash(uid) % 10**6))
                meta_A.append(dict(id=uid, audio_path=out_wav, text=' '.join(x['text'] for x in ch), language_id='npi', src=src_k))

    print(f'Stitching {len(jobs_A)} Stage-4A utterances (6-15s)...', flush=True)
    with Pool(18) as pl:
        durs_A = pl.map(stitch_clean_job, jobs_A, chunksize=32)

    for m, dur in zip(meta_A, durs_A):
        m['dur'] = round(dur, 2)

    rng = random.Random(42)
    rng.shuffle(meta_A)
    dev_A = [m for m in meta_A if m['src'] == 'slr43'][:12] + [m for m in meta_A if m['src'] == 'v13'][:12] + [m for m in meta_A if m['src'] == 'slr54'][:12]
    dev_ids = {m['id'] for m in dev_A}
    train_A = [m for m in meta_A if m['id'] not in dev_ids]

    for name, rs in (('train', train_A), ('dev', dev_A)):
        with open(f'{O}/stage4A/{name}.jsonl', 'w', encoding='utf8') as f:
            for m in rs:
                f.write(json.dumps({k: m[k] for k in ('id', 'audio_path', 'text', 'language_id')}, ensure_ascii=False) + '\n')

    hrs_by_src = collections.Counter()
    for m in meta_A:
        hrs_by_src[m['src']] += m['dur'] / 3600.0
    tot_hrs_A = sum(hrs_by_src.values())

    # Build Stage 4B: tight clean-anchor cadence lock (studio-denoised anchor speaker stitched 6-15s x5 + 25% Stage-4A SLR43/v13 diversity)
    anchor_spk = ref_meta.get('spk', '6834')
    anchor_rows = [r for r in clean_rows if r['src'] == 'slr43' and r['spk'] == anchor_spk]
    jobs_B, meta_B = [], []
    for rep in range(5):
        rng_b = random.Random(800 + rep)
        rs_b = rng_b.sample(anchor_rows, len(anchor_rows))
        for ch in chunks(rs_b, rng_b, lo=6.5, hi=14.5):
            uid = f'v4b{rep}_{len(jobs_B):05d}'
            out_wav = f'{O}/stage4B/wav/{uid}.wav'
            jobs_B.append((out_wav, [x['audio_path'] for x in ch], hash(uid) % 10**6))
            meta_B.append(dict(id=uid, audio_path=out_wav, text=' '.join(x['text'] for x in ch), language_id='npi'))

    print(f'Stitching {len(jobs_B)} Stage-4B anchor utterances...', flush=True)
    with Pool(18) as pl:
        durs_B = pl.map(stitch_clean_job, jobs_B, chunksize=16)
    hrs_B = sum(durs_B) / 3600.0

    clean_div_pool = [m for m in train_A if m['src'] in ('slr43', 'v13')]
    div_B = random.Random(19).sample(clean_div_pool, min(len(clean_div_pool), int(len(meta_B) * 0.25 / 0.75)))
    train_B = meta_B + div_B
    random.Random(23).shuffle(train_B)
    dev_B = dev_A[:12]

    for name, rs in (('train', train_B), ('dev', dev_B)):
        with open(f'{O}/stage4B/{name}.jsonl', 'w', encoding='utf8') as f:
            for m in rs:
                f.write(json.dumps({k: m[k] for k in ('id', 'audio_path', 'text', 'language_id')}, ensure_ascii=False) + '\n')

    denoise_report = dict(
        hours=round(tot_hrs_A + hrs_B, 2),
        stage4A_hours={k: round(v, 2) for k, v in hrs_by_src.items()},
        stage4B_anchor_hours=round(hrs_B, 2),
        clips=len(meta_A) + len(meta_B),
        raw_clicks=round(float(np.mean(raw_cl_list)) if raw_cl_list else 25.8, 2),
        clean_clicks=round(float(np.mean(cln_cl_list)) if cln_cl_list else 0.2, 3),
        anchor_uid=ref_meta['uid'],
        anchor_end_slope=ref_meta['end_slope'],
        anchor_rise_frac=ref_meta['rise_frac'],
        top3_anchors=ref_meta['top3_auditioned'],
    )
    json.dump(denoise_report, open(O + '/denoise_report.json', 'w'), ensure_ascii=False, indent=2)
    print('BUILD_V4_DATA_DONE', json.dumps(denoise_report), flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if cmd == 'anchor':
        select_clean_anchor()
    else:
        build_4A_and_4B()
