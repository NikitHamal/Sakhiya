import sys, os, json, re, random, unicodedata, numpy as np, soundfile as sf, librosa
sys.path.insert(0, '/work/sp'); import sp_common as C
W = '/work/sp/data'; O = '/work/sp/ft'; os.makedirs(O + '/wav', exist_ok=True)
anchor = json.load(open('/work/sp/ref/anchor_ref.json'))
rw, rsr = sf.read('/work/sp/ref/anchor_ref.wav', dtype='float32'); remb = C.spk_emb(rw, rsr)

def clean_text(t):
    t = unicodedata.normalize('NFC', t).replace('\u200b', '').replace('\ufeff', '')
    t = re.sub(r'["“”‘’\'`()\[\]{}<>*_#/\\|~^]', ' ', t)
    t = t.replace('।।', '।').replace('॥', '।') if False else t
    t = re.sub(r'\.(?=\s|$)', '।', t)             # sentence-final full stop -> danda
    t = re.sub(r'\s*([,।?!;])\s*', r'\1 ', t)
    t = re.sub(r'([,।?!;])\1+', r'\1', t)
    t = re.sub(r'\s+', ' ', t).strip(' ,;')
    if not t: return None
    if re.search(r'[A-Za-z0-9]', t): return None   # mined text with stray Latin/ASCII digits -> drop (keeps G2P/tokenizer clean)
    if t[-1] not in '।?!॥': t += '।'
    return t

def prep(w, sr, target_peak=0.89, lead=0.12, tail=0.2):
    w = w.astype(np.float32)
    if w.ndim > 1: w = w.mean(1)
    if sr != 24000: w = librosa.resample(w, orig_sr=sr, target_sr=24000)
    iv = librosa.effects.split(w, top_db=42, frame_length=1024, hop_length=256)
    if len(iv):
        a = max(0, iv[0][0] - int(lead * 24000)); b = min(len(w), iv[-1][1] + int(tail * 24000)); w = w[a:b]
    pk = float(np.abs(w).max()) or 1.0
    return w * (target_peak / pk)

rows, dev, stats = [], [], {}
# A) SLR43 anchor speaker 6834
D = '/work/sp/data/slr43/ne_np_female'
A = [l.rstrip('\n').split('\t') for l in open(D + '/line_index.tsv', encoding='utf8')]
A = [(u, t) for u, t in A if u.split('_')[1] == '6834']
random.Random(7).shuffle(A)
for j, (u, t) in enumerate(A):
    t2 = clean_text(t)
    if not t2: continue
    w, sr = sf.read(f'{D}/wavs/{u}.wav', dtype='float32'); w = prep(w, sr)
    p = f'{O}/wav/anchor_{u}.wav'; sf.write(p, w, 24000)
    r = dict(id=f'anchor_{u}', audio_path=p, text=t2, language_id='npi', src='slr43_6834', dur=len(w) / 24000)
    (dev if j < 10 else rows).append(r)
stats['anchor_train'] = len(rows)
# B) VC'd IA news/story/conversational
F = json.load(open(W + '/ia_filtered.json')); keep = 0; drop = dict(text=0, missing=0, cer=0, spk=0, mos=0)
vc_rows = []
for x in F:
    p = f"{W}/ia_vc/{x['id']}.wav"
    if not os.path.exists(p): drop['missing'] += 1; continue
    t = clean_text(x['text'])
    if not t: drop['text'] += 1; continue
    w, sr = sf.read(p, dtype='float32'); w = prep(w, sr)
    if len(w) / 24000 < 2.0: drop['missing'] += 1; continue
    q = f"{O}/wav/ia_{x['id']}.wav"; sf.write(q, w, 24000)
    hyp, cf = C.kriti().read_conf([q])[0]; cer = C.cer(t, hyp)
    # mined radio transcripts are noisy: only trust the text if the ORIGINAL audio agrees with it, and VC must not damage content
    ho, _ = C.kriti().read_conf([f"{W}/ia_wav/{x['id']}.wav"])[0]; cer_o = C.cer(t, ho)
    if cer_o > 0.30 or cer > min(0.40, cer_o + 0.20): drop['cer'] += 1; continue
    sim = float(np.dot(C.spk_emb(w, 24000), remb))
    if sim < 0.5: drop['spk'] += 1; continue
    u = C.utmos_score(w, 24000)
    if u < 2.6: drop['mos'] += 1; continue
    vc_rows.append(dict(id=f"ia_{x['id']}", audio_path=q, text=t, language_id='npi', src='ia_vc_' + x['cat'], dur=len(w) / 24000, cer=cer, sim=sim, utmos=u, cat=x['cat']))
stats['ia_vc_kept'] = len(vc_rows); stats['ia_drop'] = drop
random.Random(3).shuffle(vc_rows); dev += vc_rows[:6]; vc_rows = vc_rows[6:]; rows += vc_rows
# B2) other SLR43 Nepal-native studio speakers unified to the anchor vocal tract by Seed-VC (clean, reliable transcripts)
SD = W + '/slr_vc'; slr_rows = []; sdrop = dict(text=0, cer=0, spk=0, mos=0)
if os.path.exists(W + '/slr_sel.json'):
    for x in json.load(open(W + '/slr_sel.json')):
        p = f"{SD}/{x['id']}.wav"
        if not os.path.exists(p): continue
        t = clean_text(x['text'])
        if not t: sdrop['text'] += 1; continue
        w, sr = sf.read(p, dtype='float32'); w = prep(w, sr)
        q = f"{O}/wav/slr_{x['id']}.wav"; sf.write(q, w, 24000)
        hyp, cf = C.kriti().read_conf([q])[0]; cer = C.cer(t, hyp)
        if cer > 0.25: sdrop['cer'] += 1; continue
        sim = float(np.dot(C.spk_emb(w, 24000), remb))
        if sim < 0.5: sdrop['spk'] += 1; continue
        u = C.utmos_score(w, 24000)
        if u < 2.8: sdrop['mos'] += 1; continue
        slr_rows.append(dict(id=f"slr_{x['id']}", audio_path=q, text=t, language_id='npi', src='slr43_vc', dur=len(w) / 24000, cer=cer, sim=sim, utmos=u))
    stats['slr_vc_kept'] = len(slr_rows); stats['slr_drop'] = sdrop
    random.Random(4).shuffle(slr_rows); dev += slr_rows[:4]; rows += slr_rows[4:]
# C) Vedic (VC'd, f0-conditioned)
if os.path.exists(W + '/ved_sel.json'):
    V = json.load(open(W + '/ved_sel.json')); vr = []
    for x in V:
        p = f"{W}/ved_vc/{x['id']}.wav"
        if not os.path.exists(p): continue
        w, sr = sf.read(p, dtype='float32'); w = prep(w, sr)
        q = f"{O}/wav/ved_{x['id']}.wav"; sf.write(q, w, 24000)
        hyp, cf = C.kriti().read_conf([q])[0]
        sim = float(np.dot(C.spk_emb(w, 24000), remb)); u = C.utmos_score(w, 24000)
        if u < 2.0 or sim < 0.35: continue
        vr.append(dict(id=f"ved_{x['id']}", audio_path=q, text=x['text'], language_id='sa', src='vedic_vc', dur=len(w) / 24000, sim=sim, utmos=u))
    stats['vedic_kept'] = len(vr); dev += vr[:3]; rows += vr[3:]
# anchor up-weighting: 3x
anch = [r for r in rows if r['src'] == 'slr43_6834']
for k in (1, 2): rows += [dict(r, id=r['id'] + f'_r{k}') for r in anch]
random.Random(1).shuffle(rows)
for name, rs in (('train', rows), ('dev', dev)):
    with open(f'{O}/{name}.jsonl', 'w', encoding='utf8') as f:
        for r in rs: f.write(json.dumps({k: r[k] for k in ('id', 'audio_path', 'text', 'language_id')}, ensure_ascii=False) + '\n')
    json.dump(rs, open(f'{O}/{name}_meta.json', 'w'), ensure_ascii=False)
bysrc = {}
for r in rows: bysrc[r['src']] = bysrc.get(r['src'], 0) + r['dur'] / 3600
stats.update(train_rows=len(rows), dev_rows=len(dev), hours_by_src=bysrc, total_hours=sum(bysrc.values()))
json.dump(stats, open(O + '/dataset_report.json', 'w'), indent=1); print(json.dumps(stats, indent=1))
