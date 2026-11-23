import sys, os, json, numpy as np, soundfile as sf, librosa, collections
sys.path.insert(0, '/work/sp'); import sp_common as C
D = '/work/sp/data/slr43/ne_np_female'
rows = [l.rstrip('\n').split('\t') for l in open(D + '/line_index.tsv')]
print('rows', len(rows), 'wavs', len(os.listdir(D + '/wavs')), flush=True)
out = []
by = collections.defaultdict(list)
for r in rows: by[r[0].split('_')[1]].append(r)
# 14 utts per speaker with 5-12 s, to rank speakers
for sp, rs in by.items():
    if len(rs) < 40: continue
    import random; random.Random(0).shuffle(rs)
    sc = []
    for uid, txt in rs[:60]:
        p = f'{D}/wavs/{uid}.wav'
        if not os.path.exists(p): continue
        w, sr = sf.read(p, dtype='float32')
        if w.ndim > 1: w = w.mean(1)
        d = len(w) / sr
        if d < 4 or d > 14: continue
        u = C.utmos_score(w, sr); hyp, cf = C.kriti().read_conf([p])[0]
        f0 = librosa.yin(librosa.resample(w, orig_sr=sr, target_sr=16000), fmin=70, fmax=400, sr=16000)
        sc.append(dict(uid=uid, txt=txt, dur=d, sr=sr, utmos=u, cer=C.cer(txt, hyp), conf=cf, f0=float(np.median(f0))))
    if not sc: continue
    s = dict(spk=sp, n=len(rs), utmos=np.mean([x['utmos'] for x in sc]), cer=np.mean([x['cer'] for x in sc]), f0=np.mean([x['f0'] for x in sc]), sr=sc[0]['sr'], cands=sc)
    out.append(s); print(sp, s['n'], round(s['utmos'], 3), round(s['cer'], 3), round(s['f0']), s['sr'], flush=True)
json.dump(out, open('/work/sp/ref_candidates.json', 'w'), ensure_ascii=False)
out.sort(key=lambda s: -(s['utmos'] - 4 * s['cer']))
best = out[0]; print('BEST SPEAKER', best['spk'])
c = [x for x in best['cands'] if 6 <= x['dur'] <= 11 and x['cer'] < 0.08]
c.sort(key=lambda x: -(x['utmos'] - 3 * x['cer']))
ref = c[0]; print('REF', ref)
os.makedirs('/work/sp/ref', exist_ok=True)
w, sr = sf.read(f"{D}/wavs/{ref['uid']}.wav", dtype='float32')
if w.ndim > 1: w = w.mean(1)
w = librosa.resample(w, orig_sr=sr, target_sr=24000) if sr != 24000 else w
sf.write('/work/sp/ref/anchor_ref.wav', w, 24000)
json.dump(dict(ref, spk=best['spk'], ref_text=ref['txt']), open('/work/sp/ref/anchor_ref.json', 'w'), ensure_ascii=False)
