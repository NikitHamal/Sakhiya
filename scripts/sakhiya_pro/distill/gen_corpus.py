#!/usr/bin/env python3
"""Sakhiya-Pro v2 -> single-voice distillation corpus (24 kHz) with strict auto-filtering.
  gen_corpus.py texts                 build /work/sp/distill/texts.jsonl (clean Nepali Wikipedia, SLR54/43 transcripts, mined news/story/
                                      conversation text, English-loanword + number templates, real Sanskrit verses); eval lines are excluded
  gen_corpus.py work I N [DEADLINE]   teacher worker: Best-of-N Pro v2 takes + forensic gate + strict keep rules -> corpus/part_I.jsonl
  gen_corpus.py merge                 manifest.jsonl (train/dev split) + per-category hours
Keep rules (ALL must hold): gate_pass, ASR CER <= 0.06 (Nepali) / sound-folded <= 0.25 (Sanskrit; Nepali ASR is not a Sanskrit model),
clicks <= 3, UTMOS >= 3.5, ECAPA sim to the anchor >= 0.55, pace 4-9.5 syl/s, zero Hindi leaks, 1.5-16 s."""
import sys, os, re, json, random, glob, time, unicodedata
HERE = os.path.dirname(os.path.abspath(__file__)); SPD = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, SPD); sys.path.insert(0, os.path.join(SPD, '..')); sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..'))); sys.path.insert(0, '/work/sp')
O = '/work/sp/distill'; W = O + '/corpus'
OKCH = re.compile(r'^[\u0900-\u097F\s,.?!।॥\-–;:\'"()0-9A-Za-z]+$')
TECH = ['Anthropic', 'Gemini', 'Jensen Huang', 'GPT-6', 'Liquid Glass Dock', 'Claude', 'OpenAI', 'NVIDIA', 'Google', 'Android', 'iPhone', 'YouTube', 'Facebook', 'WhatsApp', 'Wi-Fi', 'Bluetooth', 'GitHub', 'Linux', 'Python', 'Tesla', 'Microsoft', 'Amazon', 'Samsung', 'TikTok', 'ChatGPT', 'Windows 11', 'USB-C', '5G', 'AI', 'GPU', 'CPU']
TECH_T = ['{} ले आज नयाँ सुविधा सार्वजनिक गरेको छ।', 'काठमाडौँका धेरै विद्यार्थीले {} प्रयोग गर्न थालेका छन्।', 'के तपाईंले {} को बारेमा सुन्नुभएको छ?', '{} का कारण काम गर्ने तरिका नै परिवर्तन भएको छ।',
          'नेपालमा पनि {} को माग बढ्दै गएको विज्ञहरूले बताएका छन्।', 'आजको कार्यक्रममा हामी {} र यसको प्रभावबारे कुरा गर्नेछौँ।', 'उनले {} बारे विस्तारमा जानकारी दिए।']
NUM_T = ['यस वर्ष ०५ प्रतिशत बढी मानिसले सेवा लिएका छन्।', 'बैठक २०८२ साल मंसिर ८ गते बस्ने भएको छ।', 'कुल बजेट १२.५ अर्ब रुपैयाँ पुगेको छ।', 'तापक्रम २८ डिग्री सेल्सियस रेकर्ड गरिएको थियो।',
         'उनले ३ घण्टा १५ मिनेटमा दौड पूरा गरे।', 'काठमाडौँ उपत्यकामा १२०० भन्दा बढी विद्यालय छन्।', 'सडकको लम्बाइ ४५.७ किलोमिटर रहेको छ।', 'मतदानमा ६८.४ प्रतिशत मतदाताले भाग लिए।']


def nz(t): return re.sub(r'\s+', '', re.sub(r'[।॥,?!.\-–—:;"\'()]', '', unicodedata.normalize('NFC', t)))


def clean(t):
    import nep_norm
    t = re.sub(r'\s+', ' ', t).strip()
    t = re.sub(r'\.(?=\s|$)', '।', t)
    if not (28 <= len(t) <= 190) or not OKCH.match(t): return None
    if len(re.findall(r'[\u0900-\u097F]', t)) < 14: return None
    if t[-1] not in '।?!॥': t += '।'
    return t


def eval_lines():
    ex = set()
    try:
        reg = json.load(open(os.path.join(SPD, '..', '..', 'docs', 'samples', 'versions', 'registry.json')))
        for c in reg['versions'][0]['clips']: ex.add(nz(c['text']))
    except Exception as e: print('registry skip', e)
    try:
        from bake_sents import SENTS
        for s in SENTS:
            for k in ('dev', 'lat'):
                if s.get(k): ex.add(nz(s[k]))
    except Exception as e: print('bake skip', e)
    return ex


def texts():
    rng = random.Random(7); ex = eval_lines(); out = []; seen = set()
    def add(t, lang, style, src):
        t = clean(t) if lang == 'ne' else re.sub(r'\s+', ' ', t).strip()
        if not t: return
        k = nz(t)
        if k in seen or k in ex: return
        seen.add(k); out.append(dict(text=t, lang=lang, style=style, src=src))
    try:
        from datasets import load_dataset
        wk = load_dataset('wikimedia/wikipedia', '20231101.ne', split='train'); n = 0
        for r in wk:
            for s in re.split(r'(?<=[।?!])\s+|\n+', r['text']):
                if re.search(r'[A-Za-z]{3,}', s) and rng.random() < 0.9: continue
                add(s, 'ne', 'news', 'wiki'); n += 1
        print('wiki cand', n)
    except Exception as e: print('wiki skip', e)
    for p in ('/work/sp/data/slr54/utt_spk_text.tsv',):
        if os.path.exists(p):
            for l in open(p, encoding='utf8'):
                q = l.rstrip('\n').split('\t')
                if len(q) == 3: add(q[2], 'ne', 'conversational', 'slr54')
    p = '/work/sp/data/slr43/ne_np_female/line_index.tsv'
    if os.path.exists(p):
        for l in open(p, encoding='utf8'): add(l.split('\t')[1], 'ne', 'conversational', 'slr43')
    sa = []
    M = '/work/sp/data/vedic/meta/meta.jsonl'
    if os.path.exists(M):
        for l in open(M, encoding='utf8'):
            x = json.loads(l); t = x.get('text', '')
            if x.get('lang') == 'sa' and x.get('agree', 0) >= 0.9 and 14 <= len(t) <= 220 and re.match(r'^[\u0900-\u097F\s,।॥ॐ]+$', t): sa.append(t)
            elif x.get('lang', 'ne') == 'ne' and x.get('style') in ('news', 'storytelling', 'conversational') and x.get('agree', 0) >= 0.8 and x.get('utmos', 0) >= 1.9 and not str(x.get('spk', '')).startswith(('ind_', 'in_')):
                add(t, 'ne', x['style'], 'mined')
    SAOK = re.compile(r'^[\u0900-\u097F\s,।॥ॐऽ]+$')
    def addsa(t, src):
        t = re.sub(r'\s+', ' ', t.replace('\n', ' ')).strip()
        if not (16 <= len(t) <= 170) or not SAOK.match(t): return
        k = nz(t)
        if k in seen or k in ex: return
        seen.add(k); out.append(dict(text=t, lang='sa', style='vedic', src=src))
    try:
        from datasets import load_dataset
        g = load_dataset('yashnbx/gita_supersite_sanskrit_tts', split='train').select_columns(['text'])
        for r in g:
            t = re.sub(r'\s+', ' ', r['text']).strip().replace('\n', ' ')
            if len(t) <= 120: addsa(t if t.endswith(('।', '॥')) else t + '॥', 'gita')
            for h in re.split(r'(?<=[।॥])\s*', t):
                if len(h) >= 20: addsa(h if h.endswith(('।', '॥')) else h + '।', 'gita')
        print('gita ok')
    except Exception as e: print('gita skip', e)
    try:
        from datasets import load_dataset
        sw = load_dataset('wikimedia/wikipedia', '20231101.sa', split='train')
        for r in sw:
            for q in re.split(r'(?<=[।॥])\s+|\n+', r['text']): addsa(q if q.endswith(('।', '॥')) else q + '।', 'sa_wiki')
    except Exception as e: print('sa wiki skip', e)
    for t in sa:
        k = nz(t)
        if k not in seen: seen.add(k); out.append(dict(text=re.sub(r'\s+', ' ', t).strip(), lang='sa', style='vedic', src='mined_sa'))
    for t in TECH:
        for tp in rng.sample(TECH_T, 4): out.append(dict(text=tp.format(t), lang='ne', style='tech', src='tech'))
    for t in NUM_T: out.append(dict(text=t, lang='ne', style='numbers', src='num'))
    by = {}
    for r in out: by.setdefault((r['src'] if r['src'] != 'mined' else 'mined_' + r['style']), []).append(r)
    cap = dict(wiki=7000, slr54=2500, slr43=800, mined_news=1200, mined_storytelling=1800, mined_conversational=1800, mined_sa=60, gita=1700, sa_wiki=1500, tech=200, num=50)
    sel = []
    for k, v in by.items():
        rng.shuffle(v); sel += v[:cap.get(k, 1500)]
    rng.shuffle(sel)
    os.makedirs(O, exist_ok=True)
    with open(O + '/texts.jsonl', 'w', encoding='utf8') as f:
        for i, r in enumerate(sel): f.write(json.dumps(dict(id=f'd{i:05d}', **r), ensure_ascii=False) + '\n')
    print('texts', len(sel), {k: min(len(v), cap.get(k, 1500)) for k, v in by.items()})


def work(i, n, deadline):
    import soundfile as sf, numpy as np
    import sakhiya_pro_render as SR
    import validators as V
    os.makedirs(W + '/wav', exist_ok=True)
    R = SR.Renderer(os.environ.get('TEACHER', '/work/sp/final/sakhiya_pro_v2'), n_take=2, num_step=32)
    rows = [json.loads(l) for l in open(O + '/texts.jsonl')][i::n]
    pf = f'{W}/part_{i}.jsonl'; have = set(json.loads(l)['id'] for l in open(pf)) if os.path.exists(pf) else set()
    f = open(pf, 'a', encoding='utf8'); kept = tried = 0
    for r in rows:
        if r['id'] in have: continue
        if time.time() > deadline: break
        tried += 1
        try:
            w, sr, info = R.take(r['text'], lang='sa' if r['lang'] == 'sa' else 'npi', max_extra=2)
            c = max(info['cands'], key=lambda x: x['sel'])
            ok = info['gate_pass'] and c['clicks'] <= 3 and c['utmos'] >= 3.5 and c['spk_sim'] >= 0.55 and 1.5 <= c['dur'] <= 16 and c['hindi'] == 0 and c['cer'] <= (0.25 if r['lang'] == 'sa' else 0.06)
            if ok:
                p = f"{W}/wav/{r['id']}.wav"; sf.write(p, w, sr)
                import nep_norm
                spoken = r['text'] if r['lang'] == 'sa' else nep_norm.normalize(SR.SVARA.sub('', r['text']))
                f.write(json.dumps(dict(id=r['id'], wav=p, text=spoken, lang=r['lang'], style=r['style'], src=r['src'], dur=c['dur'], cer=c['cer'], utmos=c['utmos'], spk_sim=c['spk_sim']), ensure_ascii=False) + '\n'); f.flush(); kept += 1
        except Exception as e: print('ERR', r['id'], str(e)[:120], flush=True)
        if tried % 25 == 0: print(f'[w{i}] tried {tried} kept {kept} {time.strftime("%T")}', flush=True)
    print(f'[w{i}] DONE tried {tried} kept {kept}', flush=True)


def merge():
    rows = [json.loads(l) for p in glob.glob(W + '/part_*.jsonl') for l in open(p) if l.strip().endswith('}')]
    rng = random.Random(3); rng.shuffle(rows)
    dev_ids = set(r['id'] for r in rows if r['lang'] == 'sa')
    dev = set(list(dev_ids)[:30]) | set(r['id'] for r in [x for x in rows if x['lang'] != 'sa'][:90])
    with open(O + '/manifest.jsonl', 'w', encoding='utf8') as f:
        for r in rows: f.write(json.dumps(dict(r, split='dev' if r['id'] in dev else 'train'), ensure_ascii=False) + '\n')
    hrs = {}
    for r in rows: hrs[r['src']] = hrs.get(r['src'], 0) + r['dur'] / 3600
    rep = dict(utts=len(rows), hours=round(sum(hrs.values()), 2), by_src={k: round(v, 2) for k, v in hrs.items()}, mean_utmos=round(sum(r['utmos'] for r in rows) / max(1, len(rows)), 3))
    json.dump(rep, open(O + '/corpus_report.json', 'w'), indent=1); print(json.dumps(rep))


if __name__ == '__main__':
    m = sys.argv[1]
    if m == 'texts': texts()
    elif m == 'work': work(int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]) if len(sys.argv) > 4 else 9e12)
    elif m == 'merge': merge()
