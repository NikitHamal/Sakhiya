import sys, os, json, glob, numpy as np, soundfile as sf
sys.path.insert(0, '/work/sp'); import sp_common as C; from bake_sents import SENTS
ref = json.load(open('/work/sp/ref/anchor_ref.json'))
w, sr = sf.read('/work/sp/ref/anchor_ref.wav', dtype='float32'); remb = C.spk_emb(w, sr)
res = {}
for bd in sorted(glob.glob('/work/sp/bakeoff/*/')):
    b = os.path.basename(bd.rstrip('/')); res[b] = {}
    for p in sorted(glob.glob(bd + '*.wav')):
        k = os.path.basename(p)[:-4]; sid, var = k.rsplit('_', 1)
        s = next(x for x in SENTS if x['id'] == sid)
        try: res[b][k] = C.score_take(p, s['dev'], remb)
        except Exception as e: res[b][k] = dict(error=str(e)[:200])
summ = {}
for b, d in res.items():
    ok = [v for v in d.values() if 'cer' in v]
    if not ok: continue
    g = lambda key: float(np.mean([v[key] for v in ok]))
    summ[b] = dict(n=len(ok), cer=g('cer'), wer=g('wer'), utmos=g('utmos'), spk_sim=g('spk_sim'), clicks=g('clicks'), pauses=g('pauses'), mean_pause=g('mean_pause'), hindi=sum(v['hindi'] for v in ok), cps=g('cps'),
        cer_tech=float(np.mean([v['cer'] for k, v in d.items() if 'cer' in v and k.split('_')[0] in ('s1', 's2', 's3', 's4', 's5')])),
        cer_vedic=float(np.mean([v['cer'] for k, v in d.items() if 'cer' in v and k.startswith('s8')] or [1])),
        cer_lat=float(np.mean([v['cer'] for k, v in d.items() if 'cer' in v and k.endswith('_lat')] or [np.nan])))
    summ[b]['composite'] = 1 - min(1, summ[b]['cer']) * 1.5 + summ[b]['utmos'] / 5 * 0.6 + summ[b]['spk_sim'] * 0.4 - 0.02 * summ[b]['clicks'] - 0.3 * min(1, summ[b]['cer_vedic']) * 0.5
json.dump(dict(summary=summ, takes=res), open('/work/sp/bakeoff/scores.json', 'w'), ensure_ascii=False, indent=1)
for b, v in sorted(summ.items(), key=lambda kv: -kv[1]['composite']): print(b, {k: round(x, 3) if isinstance(x, float) else x for k, x in v.items()})
