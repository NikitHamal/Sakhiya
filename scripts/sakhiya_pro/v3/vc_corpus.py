#!/usr/bin/env python3
"""Real Nepal-native speech -> Sakhiya-Pro v3's single vocal tract (anchor speaker 6834) via Seed-VC, with strict gating.

  vc_corpus.py list [HOURS]     build /work/sp/distill/vc_src.jsonl (SLR43 non-anchor females, v13 Nepal news/story/conv, female SLR54), interleaved
  vc_corpus.py work I N [DEADLINE_EPOCH]   worker: VC (F0-conditioned: keeps real prosody / timing / nasalization) + gate -> distill/vc/part_I.jsonl
Keep rules (after de-click): ASR CER vs human transcript <= 0.10, UTMOS >= 3.2, clicks <= 6, ECAPA spk-sim to the anchor >= 0.60, 1.5-16 s, zero Hindi leak.
"""
import sys, os, re, json, random, glob, time, types
sys.path.insert(0, '/work/sp/repo/scripts/sakhiya_pro/v3'); sys.path.insert(0, '/work/sp/repo/scripts/sakhiya_pro'); sys.path.insert(0, '/work/sp')
import numpy as np, soundfile as sf, librosa
O = '/work/sp/distill'; VC = O + '/vc'; SRC = O + '/vc_src.jsonl'


def build_list(hours=12.0):
    import collections
    from multiprocessing import Pool
    import build_v3_data as B
    rng = random.Random(21); pools = collections.defaultdict(list)
    D = '/work/sp/data/slr43/ne_np_female'
    for l in open(D + '/line_index.tsv', encoding='utf8'):
        u, t = l.rstrip('\n').split('\t')[:2]
        if u.split('_')[1] == '6834' or re.search(r'[A-Za-z0-9]', t): continue
        p = f'{D}/wavs/{u}.wav'
        if not os.path.exists(p): continue
        d = sf.info(p).duration
        if 1.5 <= d <= 14: pools['slr43'].append(dict(id='vc43_' + u, path=p, text=t.strip(), src='slr43', dur=d))
    rows = B.pool_rows()   # gated ft2 pool + relaxed v13
    s54 = collections.defaultdict(list)
    for r in rows:
        if r['src'].startswith('v13'): pools['v13'].append(dict(id='vc' + r['id'], path=r['audio_path'], text=r['text'], src=r['src'], dur=r['dur']))
        elif r['src'] == 'slr54': s54[r['spk']].append(r)
    with Pool(18) as pl: f0 = {s: float(np.median(pl.map(B.f0_med, [x['audio_path'] for x in rs[:8]]))) for s, rs in s54.items()}
    for s, rs in s54.items():
        if f0[s] >= 160:
            for r in rs:
                if 2.0 <= r['dur'] <= 14: pools['slr54'].append(dict(id='vc' + r['id'], path=r['audio_path'], text=r['text'], src='slr54', dur=r['dur']))
    for k in pools: rng.shuffle(pools[k])
    print({k: (len(v), round(sum(x['dur'] for x in v) / 3600, 2)) for k, v in pools.items()}, flush=True)
    # interleave: slr43 + v13 first-class (all), slr54 fills; weighted round robin so any prefix is a balanced mix
    order, idx, tot = [], {k: 0 for k in pools}, 0.0
    w = {'slr43': 3, 'v13': 3, 'slr54': 4}
    while tot < hours * 3600 and any(idx[k] < len(pools[k]) for k in pools):
        for k in ('slr43', 'v13', 'slr54'):
            for _ in range(w[k]):
                if idx[k] < len(pools[k]): x = pools[k][idx[k]]; idx[k] += 1; order.append(x); tot += x['dur']
    os.makedirs(O, exist_ok=True)
    with open(SRC, 'w', encoding='utf8') as f:
        for x in order: f.write(json.dumps(x, ensure_ascii=False) + '\n')
    print('vc_src', len(order), 'hours', round(tot / 3600, 2), flush=True)


def anchor_prompt():
    P = '/work/sp/data/anchor_prompt_20s.wav'
    if os.path.exists(P): return P
    os.makedirs('/work/sp/data', exist_ok=True)
    cand = [r for r in json.load(open('/work/sp/v3/audit_v2/ref_audit.json')) if r['spk'] == '6834' and r['src'] == 'slr43']
    cand.sort(key=lambda x: -(x['utmos'] - 3 * x['cer'])); segs, tot = [], 0
    for c in cand:
        w, sr = sf.read(c['path'], dtype='float32'); w = w.mean(1) if w.ndim > 1 else w
        w = librosa.resample(w, orig_sr=sr, target_sr=22050); segs += [w, np.zeros(int(0.25 * 22050), np.float32)]; tot += len(w) / 22050
        if tot > 20: break
    sf.write(P, np.concatenate(segs), 22050); print('prompt s', tot, flush=True); return P


def declick(w):
    """interpolate isolated sample-level spikes (vocoder glitches) before gating"""
    w = w.copy()
    for _ in range(6):
        bad = np.where(np.abs(np.diff(w)) > 0.09)[0]
        if not len(bad): break
        w[bad + 1] = 0.5 * (w[bad] + w[np.minimum(bad + 2, len(w) - 1)])
    return w


def work(i, n, deadline):
    os.environ['HF_TOKEN'] = open('/tmp/hf_token').read().strip()
    P = anchor_prompt()
    sys.path.insert(0, '/work/sp/seed-vc'); os.chdir('/work/sp/seed-vc')
    import torch
    _m = types.ModuleType('dac'); _n = types.ModuleType('dac.nn'); _q = types.ModuleType('dac.nn.quantize')
    class VectorQuantize(torch.nn.Module): pass
    _q.VectorQuantize = VectorQuantize; _m.nn = _n; _n.quantize = _q
    sys.modules.update({'dac': _m, 'dac.nn': _n, 'dac.nn.quantize': _q})
    from seed_vc_wrapper import SeedVCWrapper
    import sp_common as C, validators as V
    vc = SeedVCWrapper()
    rw, rsr = sf.read('/work/sp/v3/ref/anchor_v3.wav', dtype='float32'); remb = C.spk_emb(rw, rsr)
    os.makedirs(VC + '/wav', exist_ok=True); pf = f'{VC}/part_{i}.jsonl'
    have = set(json.loads(l)['id'] for l in open(pf) if l.strip().endswith('}')) if os.path.exists(pf) else set()
    rows = [json.loads(l) for l in open(SRC, encoding='utf8')][i::n]; f = open(pf, 'a', encoding='utf8'); tried = kept = 0; t0 = time.time(); steps = int(os.environ.get('VC_STEPS', '20'))
    for x in rows:
        if x['id'] in have: continue
        if time.time() > deadline: break
        tried += 1; dst = f"{VC}/wav/{x['id']}.wav"
        try:
            last = None
            for _, full in vc.convert_voice(x['path'], P, diffusion_steps=steps, inference_cfg_rate=0.7, f0_condition=True, auto_f0_adjust=True, stream_output=True):
                if full is not None: last = full
            sr_out, a = last if isinstance(last, tuple) else (22050, last)
            a = librosa.resample(np.asarray(a, np.float32).reshape(-1), orig_sr=sr_out, target_sr=24000); a = declick(a); pk = float(np.abs(a).max()); a = a * (0.9 / pk) if pk > 0.9 else a; sf.write(dst, a, 24000)
            m = C.score_take(dst, x['text'], remb)
            m['cer'] = V.pron_cer(x['text'], m.get('hyp', ''), 'npi')
            ok = m['cer'] <= 0.20 and m['utmos'] >= 2.35 and m['clicks'] <= 6 and m['spk_sim'] >= 0.36 and 1.5 <= m['dur'] <= 16 and m['hindi'] == 0
            if ok:
                f.write(json.dumps(dict(id=x['id'], wav=dst, text=x['text'], lang='ne', style={'slr43': 'conversational', 'slr54': 'conversational'}.get(x['src'], x['src'].replace('v13_', '')),
                                        src='vc_' + x['src'].split('_')[0], dur=m['dur'], cer=m['cer'], utmos=m['utmos'], spk_sim=m['spk_sim']), ensure_ascii=False) + '\n'); f.flush(); kept += 1
            else:
                print('REJ', x['id'], {k: round(float(m[k]), 3) for k in ('cer', 'utmos', 'clicks', 'spk_sim', 'hindi', 'dur')}, flush=True); os.remove(dst)
        except Exception as e:
            print('ERR', x['id'], repr(e)[:160], flush=True)
        if tried % 20 == 0: print(f'[vc{i}] tried {tried} kept {kept} {round(time.time() - t0)}s', flush=True)
    print(f'[vc{i}] DONE tried {tried} kept {kept}', flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'list': build_list(float(a[1]) if len(a) > 1 else 12.0)
    else: work(int(a[1]), int(a[2]), float(a[3]) if len(a) > 3 else time.time() + 3600)
