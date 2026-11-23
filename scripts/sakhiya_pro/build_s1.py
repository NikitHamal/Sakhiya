#!/usr/bin/env python3
"""Stage-1 broad Nepal-native phonology corpus for Sakhiya-Pro v2.
Sources: OpenSLR SLR54 (multi-speaker Nepal, 16k->24k), SLR43 non-anchor native speakers (originals, no VC),
relaxed-filter v13 news/storytelling/conversational clips (Gemini gold text, Nepal speakers only: no Indian-broadcast).
worker mode:  build_s1.py work i N   (CER/UTMOS gate with Kriti ASR + UTMOS22)   -> /work/sp/ft2/part_i.jsonl
merge mode:   build_s1.py merge TARGET_HOURS
"""
import sys, os, json, re, random, glob
sys.path.insert(0, '/work/sp'); sys.path.insert(0, '/work/sp/repo/scripts/sakhiya_pro')
import numpy as np, soundfile as sf, librosa
import nep_norm
O = '/work/sp/ft2'; os.makedirs(O + '/wav', exist_ok=True)


def clean_text(t):
    t = nep_norm.normalize(t)
    t = re.sub(r'["“”‘’\'`()\[\]{}<>*_#/\\|~^]', ' ', t)
    t = re.sub(r'\.(?=\s|$)', '।', t)
    t = re.sub(r'\s*([,।?!;])\s*', r'\1 ', t)
    t = re.sub(r'([,।?!;])\1+', r'\1', t)
    t = re.sub(r'\s+', ' ', t).strip(' ,;')
    if not t or re.search(r'[A-Za-z0-9]', t): return None
    if len(re.findall(r'[\u0900-\u097F]', t)) < 6: return None
    if t[-1] not in '।?!॥': t += '।'
    return t


def candidates():
    rng = random.Random(11); C = []
    # SLR54
    D = '/work/sp/data/slr54'
    fl = {os.path.basename(p)[:-5]: p for p in glob.glob(D + '/**/*.flac', recursive=True)}
    for l in open(D + '/utt_spk_text.tsv', encoding='utf8'):
        p = l.rstrip('\n').split('\t')
        if len(p) == 3 and p[0] in fl: C.append(dict(id='s54_' + p[0], path=fl[p[0]], text=p[2], src='slr54', spk=p[1]))
    # SLR43 non-anchor natives (originals)
    D = '/work/sp/data/slr43/ne_np_female'
    for l in open(D + '/line_index.tsv', encoding='utf8'):
        u, t = l.rstrip('\n').split('\t')[:2]
        C.append(dict(id='s43_' + u, path=f'{D}/wavs/{u}.wav', text=t, src='slr43_6834' if u.split('_')[1] == '6834' else 'slr43', spk=u.split('_')[1]))
    # relaxed v13 (24k) news / storytelling / conversational with Gemini gold text
    # (Session 67 fresh box: Gemini gold text / per-clip ASR table are gone; the private dataset's meta.jsonl carries
    #  Whisper/CTC-agreed text + utmos/snr/spk, which we use with a relaxed filter. Indian-broadcast and vedic stay out.)
    M = '/work/sp/data/vedic/meta/meta.jsonl'
    have = {}
    if os.path.exists(M):
        for l in open(M, encoding='utf8'):
            x = json.loads(l)
            if x.get('style') not in ('news', 'storytelling', 'conversational') or x.get('lang', 'ne') != 'ne': continue
            if x.get('utmos', 0) < 1.9 or x.get('snr', 0) < 12 or x.get('agree', 0) < 0.70: continue
            if str(x.get('spk', '')).startswith(('ind_', 'in_')) or 'indian' in str(x.get('tier', '')): continue
            have[x['id']] = x
    for cid, x in have.items():
        C.append(dict(id='v13_' + cid, path=f'/work/sp/data/v13/wav/{cid}.wav', text=x['text'], src='v13_' + x['style'], spk=x.get('spk', 'v13'), v13=True))
    return C


def v13wav():
    import torch
    os.makedirs('/work/sp/data/v13/wav', exist_ok=True)
    want = {c['id'][4:] for c in candidates() if c.get('v13')}
    d = torch.load('/work/sp/data/v13/real_audio_24k.pt', map_location='cpu', weights_only=False)
    off = 0; n = 0
    for p, L in zip(d['paths'], d['lens']):
        L = int(L); cid = os.path.basename(p)[:-4]
        if cid in want: sf.write(f'/work/sp/data/v13/wav/{cid}.wav', d['wav'][off:off + L].numpy(), 24000); n += 1
        off += L
    print('v13 wavs', n, 'of', len(want))


def work(i, n):
    import sp_common as C
    cands = candidates(); rng = random.Random(5); rng.shuffle(cands)
    # per-source budgets of candidates (slr54 capped at ~34k utterances)
    by = {}
    out = []
    for c in cands:
        by[c['src']] = by.get(c['src'], 0) + 1
        if c['src'] == 'slr54' and by[c['src']] > 34000: continue
        out.append(c)
    mine = out[i::n]; f = open(f'{O}/part_{i}.jsonl', 'w', encoding='utf8'); kept = 0
    v13 = None
    for j, c in enumerate(mine):
        try:
            t = clean_text(c['text'])
            if not t: continue
            if c.get('v13'):
                w, sr = sf.read(c['path'], dtype='float32') if os.path.exists(c['path']) else (None, None)
                if w is None: continue
            else:
                w, sr = sf.read(c['path'], dtype='float32')
            if w.ndim > 1: w = w.mean(1)
            d = len(w) / sr
            if d < 2.5 or d > 16: continue
            if sr != 24000: w = librosa.resample(w, orig_sr=sr, target_sr=24000)
            iv = librosa.effects.split(w, top_db=40, frame_length=1024, hop_length=256)
            if not len(iv): continue
            a = max(0, iv[0][0] - int(.12 * 24000)); b = min(len(w), iv[-1][1] + int(.25 * 24000)); w = w[a:b]
            pk = float(np.abs(w).max()) or 1.0
            if pk >= 0.999 and (np.abs(w) > 0.995).mean() > 0.002: continue
            w = w * (0.89 / pk)
            q = f"{O}/wav/{c['id']}.wav"; sf.write(q, w, 24000)
            hyp, cf = C.kriti().read_conf([q])[0]; cer = C.cer(t, hyp)
            if cer > (0.22 if c['src'] == 'slr54' else 0.30): os.remove(q); continue
            u = C.utmos_score(w, 24000)
            if u < (2.0 if c['src'] == 'slr54' else 1.9): os.remove(q); continue
            f.write(json.dumps(dict(id=re.sub(r'[^A-Za-z0-9_\-]', '-', c['id']), audio_path=q, text=t, language_id='npi', src=c['src'], spk=c['spk'], dur=len(w) / 24000, cer=cer, utmos=u), ensure_ascii=False) + '\n'); f.flush(); kept += 1
        except Exception as e:
            print('ERR', c['id'], str(e)[:100], flush=True)
        if j % 200 == 0: print(f'w{i} {j}/{len(mine)} kept {kept}', flush=True)
    print(f'w{i} DONE kept {kept}', flush=True)


def merge(target_h):
    rows = []
    for p in glob.glob(O + '/part_*.jsonl'): rows += [json.loads(l) for l in open(p, encoding='utf8')]
    rng = random.Random(2); rng.shuffle(rows)
    # cap speaker dominance (<= 1.2 h per speaker), prefer cleaner clips
    rows.sort(key=lambda r: -r['utmos']); per = {}; sel = []; tot = 0
    for r in rows:
        k = (r['src'], r['spk'])
        if per.get(k, 0) > 1.2 * 3600 and r['src'] == 'slr54': continue
        sel.append(r); per[k] = per.get(k, 0) + r['dur']; tot += r['dur']
        if tot > target_h * 3600: break
    rng.shuffle(sel)
    dev = sel[:60]; tr = sel[60:]
    # extra: previous v1 pool (anchor + VC'd + vedic) so Sanskrit/anchor stay in the mix
    old = [json.loads(l) for l in open('/work/sp/ft/train.jsonl', encoding='utf8')] if os.path.exists('/work/sp/ft/train.jsonl') else []
    om = {r['id']: r for r in json.load(open('/work/sp/ft/train_meta.json'))} if old else {}
    old = [r for r in old if os.path.exists(r['audio_path'])]
    for r in old: tr.append(dict(id=r['id'], audio_path=r['audio_path'], text=r['text'], language_id=r['language_id'], src='v1_' + om.get(r['id'].replace('_r1', '').replace('_r2', ''), {}).get('src', 'old'), dur=om.get(r['id'], {}).get('dur', 5)))
    rng.shuffle(tr)
    bysrc = {}
    for r in tr: bysrc[r['src']] = bysrc.get(r['src'], 0) + r.get('dur', 0) / 3600
    json.dump(tr, open(O + '/selected_rows.json', 'w'), ensure_ascii=False)
    for name, rs in (('train', tr), ('dev', dev)):
        with open(f'{O}/{name}.jsonl', 'w', encoding='utf8') as f:
            for r in rs: f.write(json.dumps({k: r[k] for k in ('id', 'audio_path', 'text', 'language_id')}, ensure_ascii=False) + '\n')
    rep = dict(train_rows=len(tr), dev_rows=len(dev), hours_by_src=bysrc, total_hours=sum(bysrc.values()), speakers=len(per))
    json.dump(rep, open(O + '/dataset_report.json', 'w'), indent=1); print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    if sys.argv[1] == 'v13wav': v13wav()
    elif sys.argv[1] == 'work': work(int(sys.argv[2]), int(sys.argv[3]))
    else: merge(float(sys.argv[2]) if len(sys.argv) > 2 else 40)
