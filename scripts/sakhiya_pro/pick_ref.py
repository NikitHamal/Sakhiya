import json, sys, os, soundfile as sf, librosa
out = json.load(open('/work/sp/ref_candidates.json'))
D = '/work/sp/data/slr43/ne_np_female'
best = None
for s in out:
    if s['spk'] not in ('6834', '7957', '0258'): continue
    for x in s['cands']:
        if 5 <= x['dur'] <= 12 and x['cer'] < 0.12 and x['utmos'] > 3.0:
            sc = x['utmos'] - 3 * x['cer']
            if not best or sc > best[0]: best = (sc, s['spk'], x)
print(best)
sc, spk, x = best
w, sr = sf.read(f"{D}/wavs/{x['uid']}.wav", dtype='float32')
if w.ndim > 1: w = w.mean(1)
w = librosa.resample(w, orig_sr=sr, target_sr=24000)
os.makedirs('/work/sp/ref', exist_ok=True); sf.write('/work/sp/ref/anchor_ref.wav', w, 24000)
json.dump(dict(x, spk=spk, ref_text=x['txt']), open('/work/sp/ref/anchor_ref.json', 'w'), ensure_ascii=False)
# per speaker durations
import collections
rows = [l.rstrip('\n').split('\t') for l in open(D + '/line_index.tsv')]
c = collections.Counter(r[0].split('_')[1] for r in rows)
tot = collections.defaultdict(float)
for r in rows:
    i = sf.info(f"{D}/wavs/{r[0]}.wav"); tot[r[0].split('_')[1]] += i.duration
print({k: (c[k], round(tot[k] / 60, 1)) for k in c})
