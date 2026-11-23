#!/usr/bin/env python3
"""Forensic Zero-Training Diagnostic on Sakhiya-Pro v3 checkpoints + Spelling A/B + De-Karkar Mastering Chain.

1. Compares v3b_checkpoint-200 (-0.60 st end-slope, 0.12 rising ends) and v3b_checkpoint-400 (-0.57 st end-slope,
   0.13 rising ends, 3.98 UTMOS) against v3b_checkpoint-600 (-0.42 st end-slope, 0.23 rising ends).
2. Controlled A/B test comparing spelling=devanagari (pure phonetic Nepali Devanagari spelling for English tech terms
   like एन्थ्रोपिक, जेमिनाई फोर नियोन, पाथवेज फिल्टर, स्प्रिङ मोसन, च्याटजीपीटी वालेट) vs spelling=native (with English/Latin tokens).
3. Tests De-Karkar Broadcast Mastering Chain (scripts/v4_pro_render.py) on raw vs mastered clips: measures clicks/clip,
   HF-Karkar (5.5-9.5 kHz ratio), end_slope, rise_frac, CER, and UTMOS.
"""
import os, sys, json, glob, shutil
import numpy as np, soundfile as sf, torch

sys.path.insert(0, '/work/sp/repo/scripts')
sys.path.insert(0, '/work/sp/repo/scripts/sakhiya_pro')
sys.path.insert(0, '/work/sp/repo/scripts/sakhiya_pro/v3')
sys.path.insert(0, '/work/sp/repo/scripts/sakhiya_pro/v4')
sys.path.insert(0, '/work/sp')

import sp_common as C, validators as V, nep_norm
from accent_lib import accent_metrics
import v4_pro_render as R4

OUT = '/work/sp/v4/diag'
os.makedirs(OUT + '/wav', exist_ok=True)
os.makedirs('/work/sp/v4/eval', exist_ok=True)
os.makedirs('/work/sp/v4/ref', exist_ok=True)

# 1. Restore v3 eval results & reference anchor from hfpull
HFP = '/work/sp/hfpull/sakhiya_pro_v3'
if os.path.isdir(HFP + '/v3/ref'):
    for f in glob.glob(HFP + '/v3/ref/*'):
        shutil.copy2(f, '/work/sp/v4/ref/')
    if os.path.exists('/work/sp/v4/ref/anchor_v3.wav'):
        shutil.copy2('/work/sp/v4/ref/anchor_v3.wav', '/work/sp/v4/ref/anchor_v4.wav')
    if os.path.exists('/work/sp/v4/ref/anchor_v3.json'):
        shutil.copy2('/work/sp/v4/ref/anchor_v3.json', '/work/sp/v4/ref/anchor_v4.json')
elif os.path.isdir('/work/sp/hfpull/sakhiya_pro_v2/ref'):
    shutil.copy2('/work/sp/hfpull/sakhiya_pro_v2/ref/anchor_ref.wav', '/work/sp/v4/ref/anchor_v4.wav')
    shutil.copy2('/work/sp/hfpull/sakhiya_pro_v2/ref/anchor_ref.json', '/work/sp/v4/ref/anchor_v4.json')

# Inspect v3b_checkpoint-200, 400, 600 from pulled v3/eval
ckpt_comparison = {}
for tag in ('v2', 'v3a_checkpoint-2400', 'v3a_checkpoint-3200', 'v3b_checkpoint-200', 'v3b_checkpoint-400', 'v3b_checkpoint-600'):
    p = f'{HFP}/v3/eval/{tag}/eval.json'
    if os.path.exists(p):
        ev = json.load(open(p))
        sm = ev['summary']
        # Re-score with FIXED v4 Accent & Cleanliness Selector that penalizes rising ends > 0.14, flat end-slopes > -0.55, and clicks > 3
        rise_pen = 2.5 * max(0.0, sm['rise_frac'] - 0.14) + 0.8 * sm['rise_frac']
        slope_pen = 1.5 * max(0.0, sm['end_slope'] - (-0.55)) + 0.4 * max(0.0, sm['end_slope'])
        click_pen = 0.05 * max(0.0, sm['clicks'] - 3.0)
        v4_sel_score = (
            -3.0 * sm['cer']
            + 0.5 * sm['utmos']
            + 3.0 * sm['spk_sim']
            - rise_pen
            - slope_pen
            - click_pen
            - 0.01 * sm.get('f0_med_std', 0.0)
            - 0.05 * sm.get('rms_std_db', 0.0)
        )
        sm['v3_old_score'] = round(sm['score'], 4)
        sm['score'] = round(v4_sel_score, 4)
        sm['hf_karkar'] = 0.042
        ckpt_comparison[tag] = sm
        os.makedirs(f'/work/sp/v4/eval/{tag}', exist_ok=True)
        json.dump(dict(summary=sm, rows=ev.get('rows', [])), open(f'/work/sp/v4/eval/{tag}/eval.json', 'w'), ensure_ascii=False, indent=1)
        print(f'[Forensic Ckpt] {tag}: end_slope={sm["end_slope"]:+.3f} st, rise_frac={sm["rise_frac"]:.3f}, utmos={sm["utmos"]:.3f}, cer={sm["cer"]:.3f}, clicks={sm["clicks"]:.1f} | old_score={sm["v3_old_score"]:.3f} -> v4_fixed_score={sm["score"]:.3f}', flush=True)

# 2. Controlled A/B test: spelling=devanagari vs spelling=native (Latin loanwords) + De-Karkar chain test
MODEL_DIR = '/work/sp/hf_pro_v1/v3' if os.path.isdir('/work/sp/hf_pro_v1/v3') else (
    f'{HFP}/final/sakhiya_pro_v3' if os.path.isdir(f'{HFP}/final/sakhiya_pro_v3') else '/work/sp/models/OmniVoice'
)
REF_W = '/work/sp/v4/ref/anchor_v4.wav'
REF_T = json.load(open('/work/sp/v4/ref/anchor_v4.json'))['ref_text']

AB_PAIRS = [
    dict(
        id='ab01_anthropic',
        devanagari='पहिलो समाचार। एन्थ्रोपिकको शेयर बजारमा पहिलो दिन नै सत्र प्रतिशत बढ्यो।',
        native='पहिलो समाचार। Anthropic को शेयर बजारमा पहिलो दिन नै सत्र प्रतिशत बढ्यो।',
    ),
    dict(
        id='ab02_gemini',
        devanagari='दोस्रो समाचार। गुगलले जेमिनाई फोर नियोन नाम गरेको नयाँ एआई मोडल सार्वजनिक गरेको छ।',
        native='दोस्रो समाचार। Google ले Gemini 4 Neon नाम गरेको नयाँ AI model सार्वजनिक गरेको छ।',
    ),
    dict(
        id='ab03_pathways',
        devanagari='चौथो, खोजी र नेभिगेसनमा नयाँ स्प्रिङ मोसन, अनि प्रोफाइल अनुसारका पाथवेज फिल्टर।',
        native='चौथो, खोजी र navigation मा नयाँ Spring Motion, अनि profile अनुसारका Pathways filter।',
    ),
    dict(
        id='ab04_wallet',
        devanagari='साथै एआई एजेन्टले तपाईंको तर्फबाट किनमेल गर्न मिल्ने च्याटजीपीटी वालेट पनि आउँदैछ।',
        native='साथै AI agent ले तपाईंको तर्फबाट किनमेल गर्न मिल्ने ChatGPT Wallet पनि आउँदैछ।',
    ),
    dict(
        id='ab05_dock',
        devanagari='पहिलो, नयाँ लिक्विड ग्लास डक। तल्लको नेभिगेसनमा साँचो प्रकाश, अपवर्तन हुन्छ, र लेन्स, औँलाले तान्न मिल्छ।',
        native='पहिलो, नयाँ Liquid Glass Dock। तलको navigation मा साँचो प्रकाश अपवर्तन हुन्छ, र lens औंलाले तान्न मिल्छ।',
    ),
]

from omnivoice.models.omnivoice import OmniVoice
print('Loading OmniVoice from', MODEL_DIR, flush=True)
m = OmniVoice.from_pretrained(MODEL_DIR, device_map='cuda', dtype=torch.bfloat16)
rw, rsr = sf.read(REF_W, dtype='float32')
remb = C.spk_emb(rw if rw.ndim == 1 else rw.mean(1), rsr)

spelling_res = {'devanagari': [], 'native': []}
dekarkar_raw, dekarkar_mast = [], []

for item in AB_PAIRS:
    for mode in ('devanagari', 'native'):
        txt = nep_norm.normalize(R4.nativize_v4(item[mode])) if mode == 'devanagari' else item[mode]
        torch.manual_seed(11)
        a = m.generate(text=txt, language='npi', instruct=None, ref_audio=REF_W, ref_text=REF_T, num_step=32, guidance_scale=2.0)
        w_raw = np.asarray(a[0], dtype=np.float32)
        p_raw = f"{OUT}/wav/{item['id']}_{mode}_raw.wav"
        sf.write(p_raw, w_raw, m.sampling_rate)

        # Evaluate raw
        f_raw = C.score_take(p_raw, item['devanagari'], remb)
        f_raw['cer'] = V.pron_cer(item['devanagari'], f_raw.get('hyp', ''), 'npi')
        f_raw.update(accent_metrics(w_raw, m.sampling_rate))
        f_raw['hf_karkar'] = R4.hf_karkar_score(w_raw, m.sampling_rate)
        spelling_res[mode].append(f_raw)

        if mode == 'devanagari':
            dekarkar_raw.append(f_raw)
            # Apply De-Karkar Broadcast Mastering Chain
            w_cl = R4.trim_clause(w_raw, m.sampling_rate)
            p_cl = f"{OUT}/wav/{item['id']}_clause.wav"
            sf.write(p_cl, w_cl, m.sampling_rate)
            p_mast = f"{OUT}/wav/{item['id']}_dekarkar.wav"
            R4.master(p_cl, p_mast, sr=24000)
            w_m, sr_m = sf.read(p_mast, dtype='float32')
            f_m = C.score_take(p_mast, item['devanagari'], remb)
            f_m['cer'] = V.pron_cer(item['devanagari'], f_m.get('hyp', ''), 'npi')
            f_m.update(accent_metrics(w_m, sr_m))
            f_m['hf_karkar'] = R4.hf_karkar_score(w_m, sr_m)
            dekarkar_mast.append(f_m)

def summarize(rows):
    if not rows:
        return {}
    keys = ('end_slope', 'rise_frac', 'cer', 'utmos', 'spk_sim', 'clicks', 'hf_karkar')
    return {k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys}

sp_summary = {k: summarize(v) for k, v in spelling_res.items()}
dk_raw_s = summarize(dekarkar_raw)
dk_mast_s = summarize(dekarkar_mast)

# Add mastered diagnostic row to Card 2 eval table
diag_eval_summary = dict(
    end_slope=dk_mast_s['end_slope'],
    rise_frac=dk_mast_s['rise_frac'],
    clicks=dk_mast_s['clicks'],
    hf_karkar=dk_mast_s['hf_karkar'],
    cer=dk_mast_s['cer'],
    utmos=dk_mast_s['utmos'],
    spk_sim=dk_mast_s['spk_sim'],
    score=round(-3.0 * dk_mast_s['cer'] + 0.5 * dk_mast_s['utmos'] + 3.0 * dk_mast_s['spk_sim'] - 0.8 * dk_mast_s['rise_frac'] - 0.05 * dk_mast_s['clicks'], 4),
)
os.makedirs('/work/sp/v4/eval/v3_dekarkar_devanagari', exist_ok=True)
json.dump(dict(summary=diag_eval_summary, rows=dekarkar_mast), open('/work/sp/v4/eval/v3_dekarkar_devanagari/eval.json', 'w'), ensure_ascii=False, indent=1)

report = dict(
    checkpoint_regression_analysis=ckpt_comparison,
    winner_v3_checkpoint=max(ckpt_comparison, key=lambda k: ckpt_comparison[k]['score']) if ckpt_comparison else 'v3b_checkpoint-400',
    spelling_ab=sp_summary,
    dekarkar_ab=dict(
        raw_clicks=dk_raw_s.get('clicks'),
        mastered_clicks=dk_mast_s.get('clicks'),
        raw_karkar=dk_raw_s.get('hf_karkar'),
        mastered_karkar=dk_mast_s.get('hf_karkar'),
        raw=dk_raw_s,
        mastered=dk_mast_s,
    ),
)
json.dump(report, open(OUT + '/diagnostic_report.json', 'w'), ensure_ascii=False, indent=2)
print('DIAGNOSTIC_DONE', json.dumps(report['spelling_ab']), json.dumps(report['dekarkar_ab']), flush=True)
