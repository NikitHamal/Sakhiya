import json, os, collections, numpy as np, torch
M = {}
for l in open('/work/sp/data/mm/asr/asr.jsonl', encoding='utf8'):
    r = json.loads(l); M[os.path.basename(r['path'])[:-4]] = r
G = json.load(open('/work/sp/data/mm/gold_text.json'))
d = torch.load('/work/sp/data/v13/real_audio_24k.pt', weights_only=False)
off = np.concatenate([[0], np.cumsum(d['lens'].numpy())])
keep = []
for i, p in enumerate(d['paths']):
    k = os.path.basename(p)[:-4]; cat = p.split('/')[-2]
    if cat == 'vedic_chant' or k not in G or k not in M: continue
    m = M[k]; t = G[k]; dur = d['lens'][i].item() / 24000
    keep.append(dict(i=i, id=k, cat=cat, dur=dur, text=t, utmos=m.get('utmos', 0), snr=m.get('snr', 0), clicks=m.get('clicks', 9), clip=m.get('clip', 1), hf=m.get('hf', 1), spk=m.get('spk'), smin=m.get('spk_min_sim', 0), nw=len(t.split())))
print('joined', len(keep), sum(x['dur'] for x in keep) / 3600, 'h')
f = [x for x in keep if 3 <= x['dur'] <= 14 and x['utmos'] >= 3.15 and x['snr'] >= 22 and x['clicks'] == 0 and x['clip'] <= 0.0005 and x['smin'] >= 0.92 and x['nw'] >= 4]
print('filtered', len(f), sum(x['dur'] for x in f) / 3600, collections.Counter(x['cat'] for x in f))
print(collections.Counter(x['spk'] for x in f).most_common(12))
json.dump(f, open('/work/sp/data/ia_filtered.json', 'w'), ensure_ascii=False)
