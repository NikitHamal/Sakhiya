#!/usr/bin/env python3
"""Sakhiya-Pro v3 training data: flowing 6-14 s Nepal-native utterances stitched from consecutive clips.

stage A  (broad native weight lock):  gated ft2 pool (OpenSLR SLR54 female speakers, SLR43 female, Nepal v13
         news / storytelling / conversational) -> same-speaker / consecutive-segment stitching, speech-RMS leveled
         (kills mic-distance drift), SLR43 + v13 oversampled x3 with different compositions.
stage B  (single-anchor voice lock):  raw SLR43 utterances of the anchor speaker (ASR-gated) stitched x4, mixed 70/30
         with a stage-A diversity sample.
usage: build_v3_data.py A | B
"""
import sys, os, re, json, glob, random, collections
sys.path.insert(0, '/work/sp/repo/scripts/sakhiya_pro'); sys.path.insert(0, '/work/sp')
import numpy as np, soundfile as sf, librosa
from multiprocessing import Pool

SR = 24000
O = '/work/sp/v3'


def load24(p):
    w, sr = sf.read(p, dtype='float32')
    if w.ndim > 1: w = w.mean(1)
    if sr != SR: w = librosa.resample(w, orig_sr=sr, target_sr=SR)
    return w


def level(w, target=0.06):
    iv = np.abs(w) > 0.02 * max(np.abs(w).max(), 1e-4)
    r = float(np.sqrt(np.mean(w[iv] ** 2))) if iv.any() else 0.05
    w = w * (target / max(r, 1e-4))
    pk = float(np.abs(w).max())
    return w * (0.95 / pk) if pk > 0.95 else w


def stitch(job):
    out, paths, seed = job
    rng = random.Random(seed); parts = []
    for i, p in enumerate(paths):
        w = load24(p); idx = librosa.effects.trim(w, top_db=40)[1]
        w = level(w[max(0, idx[0] - int(0.05 * SR)): idx[1] + int(0.07 * SR)])
        f = int(0.008 * SR); w[:f] *= np.linspace(0, 1, f); w[-f:] *= np.linspace(1, 0, f)
        parts.append(w)
        if i < len(paths) - 1: parts.append(np.zeros(int(rng.uniform(0.18, 0.42) * SR), np.float32))
    w = np.concatenate(parts); sf.write(out, w, SR); return len(w) / SR


def chunks(rows, rng, lo=6.0, hi=14.0):
    """greedy chunking of an ordered list of rows into 6-14 s utterances."""
    res, cur, d = [], [], 0.0; tgt = rng.uniform(lo, hi - 1)
    for r in rows:
        if cur and d + r['dur'] + 0.3 > hi:
            if d >= lo - 0.5: res.append(cur)
            cur, d, tgt = [], 0.0, rng.uniform(lo, hi - 1)
        cur.append(r); d += r['dur'] + 0.3
        if d >= tgt: res.append(cur); cur, d, tgt = [], 0.0, rng.uniform(lo, hi - 1)
    if cur and d >= lo: res.append(cur)
    return res


def v13_key(r):
    m = re.match(r'(.*)_(\d+)$', r['id']); return (m.group(1), int(m.group(2))) if m else (r['id'], 0)


def f0_med(p):
    try:
        w = load24(p)[: 6 * SR]; w = librosa.resample(w, orig_sr=SR, target_sr=16000)
        f = librosa.yin(w, fmin=75, fmax=420, sr=16000); return float(np.median(f[f > 80])) if (f > 80).any() else 0.0
    except Exception: return 0.0


def pool_rows():
    rows = []
    for p in glob.glob('/work/sp/ft2/part_*.jsonl'):
        for l in open(p, encoding='utf8'):
            l = l.strip()
            if not l.endswith('}'): continue
            r = json.loads(l)
            if r.get('cer', 1) > 0.12 or r.get('utmos', 0) < 2.2 or not os.path.exists(r['audio_path']): continue
            rows.append(r)
    return rows


def build_A():
    os.makedirs(O + '/stageA/wav', exist_ok=True)
    rows = pool_rows(); print('pool', len(rows), collections.Counter(r['src'] for r in rows), flush=True)
    # female-only SLR54: speaker median F0 >= 160 Hz on <=8 clips each
    by = collections.defaultdict(list)
    for r in rows:
        if r['src'] == 'slr54': by[r['spk']].append(r)
    with Pool(18) as pl:
        spk_f0 = {s: float(np.median(pl.map(f0_med, [x['audio_path'] for x in rs[:8]]))) for s, rs in by.items()}
    fem = {s for s, v in spk_f0.items() if v >= 160}
    print('slr54 speakers', len(by), 'female', len(fem), flush=True)
    rows = [r for r in rows if r['src'] != 'slr54' or r['spk'] in fem]
    groups = collections.defaultdict(list)
    for r in rows:
        k = (r['src'].split('_')[0] if r['src'].startswith('v13') is False else 'v13', r['id'].rsplit('_', 1)[0] if r['src'].startswith('v13') else r['spk'])
        groups[k].append(r)
    jobs, meta = [], []
    for rep, (rpt) in enumerate((0, 1, 2)):
        rng = random.Random(100 + rep)
        for k, rs in groups.items():
            heavy = k[0] in ('slr43', 'v13')
            if rep > 0 and not heavy: continue
            rs = sorted(rs, key=v13_key) if k[0] == 'v13' else rng.sample(rs, len(rs))
            for ch in chunks(rs, rng):
                uid = f"v3a{rep}_{len(jobs):06d}"
                jobs.append((f'{O}/stageA/wav/{uid}.wav', [x['audio_path'] for x in ch], hash(uid) % 10**6))
                meta.append(dict(id=uid, text=' '.join(x['text'].strip() for x in ch), language_id='npi', src=k[0]))
    print('stitching', len(jobs), flush=True)
    with Pool(18) as pl: durs = pl.map(stitch, jobs, chunksize=16)
    for m, j, d in zip(meta, jobs, durs): m['audio_path'] = j[0]; m['dur'] = round(d, 2)
    rng = random.Random(3); rng.shuffle(meta)
    dev = [m for m in meta if m['src'] == 'slr43'][:10] + [m for m in meta if m['src'] == 'v13'][:10] + [m for m in meta if m['src'] == 'slr54'][:20]
    ids = {m['id'] for m in dev}; train = [m for m in meta if m['id'] not in ids]
    for name, rs in (('train', train), ('dev', dev)):
        with open(f'{O}/stageA/{name}.jsonl', 'w', encoding='utf8') as f:
            for m in rs: f.write(json.dumps({k: m[k] for k in ('id', 'audio_path', 'text', 'language_id')}, ensure_ascii=False) + '\n')
    hrs = collections.Counter()
    for m in meta: hrs[m['src']] += m['dur'] / 3600
    print('STAGE_A rows', len(train), 'dev', len(dev), 'hours', {k: round(v, 2) for k, v in hrs.items()}, 'total', round(sum(hrs.values()), 2), flush=True)
    json.dump(dict(train=len(train), hours={k: round(v, 3) for k, v in hrs.items()}), open(O + '/stageA/report.json', 'w'))


def build_B():
    import sp_common as C, nep_norm
    os.makedirs(O + '/stageB/wav', exist_ok=True)
    spk = json.load(open(O + '/ref/anchor_v3.json')).get('uid') or ''
    spk = spk.split('_')[1] if spk.startswith('nep_') else '6834'
    D = '/work/sp/data/slr43/ne_np_female'
    rows = [l.rstrip('\n').split('\t')[:2] for l in open(D + '/line_index.tsv', encoding='utf8')]
    rows = [(u, t) for u, t in rows if u.split('_')[1] == spk and not re.search(r'[A-Za-z0-9]', t)]
    print('anchor speaker', spk, 'raw utts', len(rows), flush=True)
    good = []
    for u, t in rows:
        p = f'{D}/wavs/{u}.wav'
        if not os.path.exists(p): continue
        hyp, cf = C.kriti().read_conf([p])[0]
        if C.cer(t, hyp) > 0.10: continue
        d = sf.info(p).duration; txt = nep_norm.normalize(t).strip()
        if txt and txt[-1] not in '।?!॥': txt += '।'
        good.append(dict(id=u, audio_path=p, text=txt, dur=d))
    print('gated', len(good), 'hours', round(sum(g['dur'] for g in good) / 3600, 3), flush=True)
    jobs, meta = [], []
    for rep in range(4):
        rng = random.Random(500 + rep); rs = rng.sample(good, len(good))
        for ch in chunks(rs, rng):
            uid = f'v3b{rep}_{len(jobs):05d}'
            jobs.append((f'{O}/stageB/wav/{uid}.wav', [x['audio_path'] for x in ch], hash(uid) % 10**6))
            meta.append(dict(id=uid, audio_path=jobs[-1][0], text=' '.join(x['text'] for x in ch), language_id='npi'))
    with Pool(18) as pl: durs = pl.map(stitch, jobs, chunksize=8)
    hrs = sum(durs) / 3600
    # 30% diversity from stage A
    A = [json.loads(l) for l in open(O + '/stageA/train.jsonl', encoding='utf8')]
    div = random.Random(9).sample(A, int(len(meta) * 0.30 / 0.70))
    train = meta + div; random.Random(2).shuffle(train)
    dev = [json.loads(l) for l in open(O + '/stageA/dev.jsonl', encoding='utf8')][:10]
    for name, rs in (('train', train), ('dev', dev)):
        with open(f'{O}/stageB/{name}.jsonl', 'w', encoding='utf8') as f:
            for m in rs: f.write(json.dumps({k: m[k] for k in ('id', 'audio_path', 'text', 'language_id')}, ensure_ascii=False) + '\n')
    print('STAGE_B anchor rows', len(meta), 'hours', round(hrs, 2), 'diversity', len(div), flush=True)


if __name__ == '__main__':
    {'A': build_A, 'B': build_B}[sys.argv[1]]()
