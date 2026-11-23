#!/usr/bin/env python3
"""Sakhiya-Pro v3 zero-training ACCENT AUDIT (runs on the Blackwell box, ~10-15 min).

1. Config audit   : language id must be the explicit Nepali tag 'npi' (never auto); instruct is always None.
2. Anchor audit   : rank 7-9 s pure-Devanagari native clips (OpenSLR SLR43 female + Nepal v13 news/story/conv) by
                    ASR CER, UTMOS, F0 range and *downward* closure (clause-final F0 slope) -> top refs.
3. Sweep stage A  : model x ref (guidance 2.0, original spelling) on the Episode-01 failing lines + tricky words.
4. Sweep stage B  : best ref x guidance {1.5, 2.0, 2.5} x spelling {orig, native-nepal} (2 seeds).
Objective accent proxies (no human in the loop): end_slope (st; + = floating English-style rise), rise_frac
(share of clause ends that rise > 1 st), declination (st/s), f0_std, plus CER / UTMOS / spk_sim / Hindi leak.
usage: accent_audit.py MODEL_DIR [OUT_DIR]
"""
import sys, os, json, re, random, time, glob
sys.path.insert(0, '/work/sp/repo/scripts/sakhiya_pro/v3'); sys.path.insert(0, '/work/sp/repo/scripts/sakhiya_pro'); sys.path.insert(0, '/work/sp/repo/scripts'); sys.path.insert(0, '/work/sp')
import numpy as np, soundfile as sf, librosa, torch
import sp_common as C, validators as V, nep_norm
from nepal_loan import nativize

MODEL = sys.argv[1] if len(sys.argv) > 1 else '/work/sp/final/sakhiya_pro_v2'
OUT = sys.argv[2] if len(sys.argv) > 2 else '/work/sp/v3/audit'
os.makedirs(OUT, exist_ok=True)
LANG = 'npi'   # explicit Nepali tag (OmniVoice lang_map: "nepali" -> "npi"); NEVER None / auto
assert LANG == 'npi'


def accent_metrics(w, sr):
    """Pitch-contour accent proxies. Positive end_slope = clause ends float UP (English-like); native Nepali closes DOWN."""
    w16 = librosa.resample(np.asarray(w, np.float32), orig_sr=sr, target_sr=16000)
    hop = 160
    f0 = librosa.yin(w16, fmin=75, fmax=420, sr=16000, frame_length=1024, hop_length=hop)
    rms = librosa.feature.rms(y=w16, frame_length=1024, hop_length=hop)[0][:len(f0)]
    f0 = f0[:len(rms)]
    voiced = (rms > 0.12 * np.percentile(rms, 95)) & (f0 > 80) & (f0 < 400)
    if voiced.sum() < 30: return dict(end_slope=0.0, rise_frac=0.0, decl=0.0, f0_std=0.0, f0_med=0.0)
    st = 12 * np.log2(np.where(voiced, f0, np.nan) / 100.0)
    t = np.arange(len(st)) * hop / 16000.0
    # median-smooth the voiced contour to kill octave glitches
    sm = st.copy(); idx = np.where(voiced)[0]
    for k in idx: sm[k] = np.nanmedian(st[max(0, k - 3):k + 4])
    ok = ~np.isnan(sm)
    decl = float(np.polyfit(t[ok], sm[ok], 1)[0])
    # clause ends = voiced segments followed by >=150 ms of unvoiced/pause (or the final segment)
    segs, start = [], None
    for i, v in enumerate(ok):
        if v and start is None: start = i
        if (not v) and start is not None:
            segs.append([start, i]); start = None
    if start is not None: segs.append([start, len(ok)])
    merged = []
    for s in segs:
        if merged and (s[0] - merged[-1][1]) * hop / 16000 < 0.15: merged[-1][1] = s[1]
        else: merged.append(s)
    slopes = []
    for a, b in merged:
        if (b - a) * hop / 16000 < 0.8: continue
        tail = sm[max(a, b - 30):b]; pre = sm[max(a, b - 90):max(a, b - 30)]
        tail, pre = tail[~np.isnan(tail)], pre[~np.isnan(pre)]
        if len(tail) < 8 or len(pre) < 8: continue
        slopes.append(float(np.mean(tail) - np.mean(pre)))
    end_slope = float(slopes[-1]) if slopes else 0.0
    return dict(end_slope=end_slope, rise_frac=float(np.mean([s > 1.0 for s in slopes])) if slopes else 0.0,
                clause_slope=float(np.mean(slopes)) if slopes else 0.0, decl=decl, f0_std=float(np.nanstd(sm[ok])), f0_med=float(100 * 2 ** (np.nanmedian(sm[ok]) / 12)))


# ------------------------------------------------------------------ test lines
S = json.load(open('/work/sp/repo/video/script.json'))
EP = {ln['id']: ln['ne'] for sc in S['scenes'] for ln in sc['lines']}
LINES = [EP[k] for k in ('c3', 'n1a', 'n2a', 'n4a', 'f1', 'o1')] + [
    'नमस्कार, म तपाईंको समाचार वाचक। आजका मुख्य समाचार सुरु गरौँ।',
    'शेयर बजारमा आज गिरावट आयो, तर अपवर्तन सूचकाङ्क बढ्यो।',
    'स्क्रिनमा औँलाले तान्न मिल्छ, र प्रम्प्ट चुहिएको त्रुटि देखियो।',
    'इन्भिडियाका प्रमुख जेन्सेन हुआङले पाँचौँ पुस्ताको चिप ल्याए।',
    'हामी भेटौँला, आशा छ, फेरि नयाँ कुरा ल्याउँछौँ।',
    'टेक्स्ट टु स्पीच मोडेल, च्याटजिपिटी वालेट, लिक्विड ग्लास डक, स्प्रिङ मोसन र पाथवेज फिल्टर।',
]
json.dump(LINES, open(OUT + '/lines.json', 'w'), ensure_ascii=False, indent=1)

# ------------------------------------------------------------------ 2. anchor audit
def candidates():
    rng = random.Random(5); out = []
    D = '/work/sp/data/slr43/ne_np_female'
    rows = [l.rstrip('\n').split('\t')[:2] for l in open(D + '/line_index.tsv', encoding='utf8')]
    rng.shuffle(rows)
    for u, t in rows:
        p = f'{D}/wavs/{u}.wav'
        if not os.path.exists(p) or re.search(r'[A-Za-z0-9]', t): continue
        try: d = sf.info(p).duration
        except Exception: continue
        if 7.0 <= d <= 9.0: out.append(dict(uid=u, path=p, txt=t, src='slr43', spk=u.split('_')[1], dur=d))
        if len(out) >= 260: break
    M = '/work/sp/data/vedic/meta/meta.jsonl'; n13 = 0
    for l in open(M, encoding='utf8'):
        x = json.loads(l)
        if x.get('style') not in ('news', 'storytelling', 'conversational') or x.get('lang', 'ne') != 'ne': continue
        if str(x.get('spk', '')).startswith(('ind_', 'in_')) or 'indian' in str(x.get('tier', '')): continue
        if x.get('agree', 0) < 0.8 or x.get('snr', 0) < 15 or re.search(r'[A-Za-z0-9]', x['text']): continue
        p = f"/work/sp/data/v13/wav/{x['id']}.wav"
        if not os.path.exists(p) or not (7.0 <= x['dur'] <= 9.0): continue
        out.append(dict(uid=x['id'], path=p, txt=x['text'], src='v13_' + x['style'], spk=str(x.get('spk')), dur=x['dur'])); n13 += 1
    print('anchor candidates', len(out), 'v13', n13, flush=True)
    return out


def audit_refs():
    P = OUT + '/ref_audit.json'
    if os.path.exists(P): return json.load(open(P))
    res = []
    for c in candidates():
        try:
            w, sr = sf.read(c['path'], dtype='float32')
            if w.ndim > 1: w = w.mean(1)
            hyp, cf = C.kriti().read_conf([c['path']])[0]
            m = accent_metrics(w, sr)
            u = C.utmos_score(w, sr)
            cer = C.cer(c['txt'], hyp)
            lead = float(librosa.effects.trim(w, top_db=35)[1][0] / sr)
            fem = 150 <= m['f0_med'] <= 290
            score = u - 4 * cer + 0.4 * cf - 0.20 * max(0, m['end_slope']) + 0.10 * min(0, -m['end_slope'] if m['end_slope'] < 0 else 0) \
                - 0.15 * max(0, m['decl'] + 0.5) - (3 if not fem else 0) - 0.5 * (lead > 0.6)
            res.append(dict(c, utmos=u, cer=cer, conf=cf, sr=sr, score=float(score), female=bool(fem), **m))
        except Exception as e:
            print('skip', c['uid'], e, flush=True)
    res.sort(key=lambda r: -r['score'])
    json.dump(res, open(P, 'w'), ensure_ascii=False, indent=1)
    return res


refs = audit_refs()
print('--- top anchor candidates (7-9 s, pure Devanagari)', flush=True)
for r in refs[:10]:
    print(f"{r['src']:<20}{r['uid'][:34]:<36}d={r['dur']:.1f} utmos={r['utmos']:.2f} cer={r['cer']:.3f} f0={r['f0_med']:.0f} end={r['end_slope']:+.2f} decl={r['decl']:+.2f} score={r['score']:.2f}", flush=True)
real_end = [r['end_slope'] for r in refs if r['cer'] < 0.1]
print('REAL native end_slope median %.2f st, rise_frac mean %.2f' % (np.median(real_end), np.mean([r['rise_frac'] for r in refs if r['cer'] < 0.1])), flush=True)
os.makedirs('/work/sp/v3/ref', exist_ok=True)
top = []
for r in refs:
    if r['female'] and r['cer'] < 0.08 and r['utmos'] > 3.0:
        top.append(r)
    if len(top) == 3: break
cur = json.load(open('/work/sp/ref/anchor_ref.json'))
rw, rsr = sf.read('/work/sp/ref/anchor_ref.wav', dtype='float32')
cur_m = accent_metrics(rw, rsr)
print('CURRENT anchor', cur['uid'], 'dur', round(cur['dur'], 1), 'end_slope %.2f decl %.2f' % (cur_m['end_slope'], cur_m['decl']), flush=True)
REFS = {'cur': dict(path='/work/sp/ref/anchor_ref.wav', text=cur['ref_text'])}
for i, r in enumerate(top):
    w, sr = sf.read(r['path'], dtype='float32')
    if w.ndim > 1: w = w.mean(1)
    if sr != 24000: w = librosa.resample(w, orig_sr=sr, target_sr=24000)
    p = f'/work/sp/v3/ref/cand{i}.wav'; sf.write(p, w, 24000)
    REFS[f'cand{i}'] = dict(path=p, text=r['txt'], uid=r['uid'], src=r['src'])

# ------------------------------------------------------------------ 3/4. synth sweeps
from omnivoice.models.omnivoice import OmniVoice
m = OmniVoice.from_pretrained(MODEL, device_map='cuda', dtype=torch.bfloat16)
print('OmniVoice loaded; language tag =', LANG, '; instruct = None', flush=True)
EMB = {k: C.spk_emb(*sf.read(v['path'], dtype='float32')) for k, v in REFS.items()}
tmp = OUT + '/tmp'; os.makedirs(tmp, exist_ok=True)


def run(ref_k, gs, spell, seed, line_i, steps=32, keep=None):
    txt0 = LINES[line_i]
    txt = nep_norm.normalize(nativize(txt0) if spell == 'native' else txt0)
    torch.manual_seed(seed)
    a = m.generate(text=txt, language=LANG, instruct=None, ref_audio=REFS[ref_k]['path'], ref_text=REFS[ref_k]['text'], num_step=steps, guidance_scale=gs)
    w = np.asarray(a[0], np.float32); p = keep or f'{tmp}/x.wav'; sf.write(p, w, m.sampling_rate)
    f = C.score_take(p, txt0, EMB[ref_k]); f['cer'] = V.pron_cer(txt0, f.get('hyp', ''), LANG)
    f.update(accent_metrics(w, m.sampling_rate)); return f


def agg(rows):
    k = ['cer', 'utmos', 'spk_sim', 'hindi', 'clicks', 'end_slope', 'rise_frac', 'decl', 'f0_std', 'cps']
    d = {x: float(np.mean([r[x] for r in rows])) for x in k}
    d['score'] = -3 * d['cer'] + 0.5 * d['utmos'] + 1.5 * d['spk_sim'] - 0.5 * d['hindi'] - 0.02 * d['clicks'] - 0.25 * max(0, d['end_slope']) - 0.5 * d['rise_frac']
    return d


A = {}
print('--- Stage A: ref sweep (gs 2.0, original spelling)', flush=True)
for rk in REFS:
    A[rk] = agg([run(rk, 2.0, 'orig', 11, i) for i in range(len(LINES))])
    print('A', rk, {k: round(v, 3) for k, v in A[rk].items()}, flush=True)
best_ref = max(A, key=lambda k: A[k]['score'])
print('best ref:', best_ref, REFS[best_ref].get('uid'), flush=True)
B = {}
print('--- Stage B: guidance x spelling on best ref', flush=True)
for gs in (1.5, 2.0, 2.5):
    for sp in ('orig', 'native'):
        B[f'{gs}_{sp}'] = agg([run(best_ref, gs, sp, sd, i) for i in range(len(LINES)) for sd in (11, 22)])
        print('B', gs, sp, {k: round(v, 3) for k, v in B[f'{gs}_{sp}'].items()}, flush=True)
best = max(B, key=lambda k: B[k]['score'])
gs_b, sp_b = best.split('_')
print('BEST CONFIG', best, flush=True)
os.makedirs(OUT + '/best', exist_ok=True)
for i in range(len(LINES)):
    run(best_ref, float(gs_b), sp_b, 11, i, keep=f'{OUT}/best/line{i:02d}.wav')
json.dump(dict(model=MODEL, language=LANG, instruct=None, real_end_slope_median=float(np.median(real_end)), refs={k: v for k, v in REFS.items()},
               stageA=A, stageB=B, best_ref=best_ref, best_gs=float(gs_b), best_spelling=sp_b, top_refs=top[:3]),
          open(OUT + '/audit.json', 'w'), ensure_ascii=False, indent=1)
if best_ref != 'cur':
    os.system(f"cp {REFS[best_ref]['path']} /work/sp/v3/ref/anchor_v3.wav")
    json.dump(dict(ref_text=REFS[best_ref]['text'], uid=REFS[best_ref].get('uid'), src=REFS[best_ref].get('src')), open('/work/sp/v3/ref/anchor_v3.json', 'w'), ensure_ascii=False)
else:
    os.system('cp /work/sp/ref/anchor_ref.wav /work/sp/v3/ref/anchor_v3.wav; cp /work/sp/ref/anchor_ref.json /work/sp/v3/ref/anchor_v3.json')
print('AUDIT_DONE', flush=True)
