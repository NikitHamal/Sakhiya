#!/usr/bin/env python3
"""Fixed Accent & Cleanliness Selector for Sakhiya-Pro v4 checkpoints (Batched OmniVoice evaluation).

Evaluates a candidate checkpoint on the 12 Nepal-native audit lines (with pure Devanagari spelling + De-Karkar chain)
and strictly penalizes:
  * rising clause ends > 0.14
  * flat/rising end-slopes > -0.55 st
  * high-frequency karkar energy (5.5-9.5 kHz ratio)
  * clicks > 3
so it NEVER again selects an over-baked British-drifting checkpoint!
usage: accent_eval_v4.py MODEL_DIR OUT_DIR
"""
import sys, os, json
import numpy as np, soundfile as sf, torch

sys.path.insert(0, '/work/sp/repo/scripts')
sys.path.insert(0, '/work/sp/repo/scripts/sakhiya_pro')
sys.path.insert(0, '/work/sp/repo/scripts/sakhiya_pro/v3')
sys.path.insert(0, '/work/sp/repo/scripts/sakhiya_pro/v4')
sys.path.insert(0, '/work/sp')

import sp_common as C, validators as V, nep_norm
from accent_lib import accent_metrics
import v4_pro_render as R4

MODEL, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT + '/wav', exist_ok=True)

S = json.load(open('/work/sp/repo/video/script.json'))
EP = {ln['id']: ln['ne'] for sc in S['scenes'] for ln in sc['lines']}
LINES = [EP[k] for k in ('c3', 'n1a', 'n2a', 'n4a', 'f1', 'o1')] + [
    'नमस्कार, म तपाईंको समाचार वाचक। आजका मुख्य समाचार सुरु गरौँ।',
    'शेयर बजारमा आज गिरावट आयो, तर अपवर्तन सूचकाङ्क बढ्यो।',
    'स्क्रिनमा औँलाले तान्न मिल्छ, र प्रम्प्ट चुहिएको त्रुटि देखियो।',
    'इन्भिडियाका प्रमुख जेन्सेन हुआङले पाँचौँ पुस्ताको चिप ल्याए।',
    'हामी भेटौँला, आशा छ, फेरि नयाँ कुरा ल्याउँछौँ।',
    'टेक्स्ट टु स्पिच मोडल, च्याटजीपीटी वालेट, लिक्विड ग्लास डक, स्प्रिङ मोसन र पाथवेज फिल्टर।',
]

REF = '/work/sp/v4/ref/anchor_v4.wav'
RT = json.load(open('/work/sp/v4/ref/anchor_v4.json'))['ref_text']

from omnivoice.models.omnivoice import OmniVoice
m = OmniVoice.from_pretrained(MODEL, device_map='cuda', dtype=torch.bfloat16)
rw, rsr = sf.read(REF, dtype='float32')
emb = C.spk_emb(rw if rw.ndim == 1 else rw.mean(1), rsr)
rows = []

txts = [nep_norm.normalize(R4.nativize_v4(t0)) for t0 in LINES]
langs = ['npi'] * len(LINES)

for sd in (11, 22):
    torch.manual_seed(sd)
    outs = m.generate(text=txts, language=langs, instruct=None, ref_audio=REF, ref_text=RT, num_step=24, guidance_scale=2.0)
    for i, (t0, a) in enumerate(zip(LINES, outs)):
        w = R4.trim_clause(np.asarray(a, np.float32), m.sampling_rate)
        p = f'{OUT}/wav/l{i:02d}_s{sd}.wav'
        sf.write(p, w, m.sampling_rate)
        f = C.score_take(p, t0, emb)
        f['cer'] = V.pron_cer(t0, f.get('hyp', ''), 'npi')
        f.update(accent_metrics(w, m.sampling_rate))
        f['hf_karkar'] = R4.hf_karkar_score(w, m.sampling_rate)
        f['rms'] = float(np.sqrt(np.mean(w ** 2)))
        f['line'] = i
        f['seed'] = sd
        rows.append(f)
        if sd != 11:
            os.remove(p)

keys = ['cer', 'utmos', 'spk_sim', 'hindi', 'clicks', 'hf_karkar', 'end_slope', 'rise_frac', 'decl', 'f0_std', 'cps']
S_out = {x: float(np.mean([r[x] for r in rows])) for x in keys}
S_out['f0_med_std'] = float(np.std([r['f0_med'] for r in rows]))
S_out['cps_std'] = float(np.std([r['cps'] for r in rows]))
S_out['rms_std_db'] = float(np.std(20 * np.log10([max(r['rms'], 1e-5) for r in rows])))

# FIXED Accent & Cleanliness Selector:
# Strictly penalizes rising ends > 0.14, flat end-slopes > -0.55 st, high-frequency karkar energy, and clicks > 3
rise_pen = 3.0 * max(0.0, S_out['rise_frac'] - 0.14) + 0.8 * S_out['rise_frac']
slope_pen = 1.8 * max(0.0, S_out['end_slope'] - (-0.55)) + 0.4 * max(0.0, S_out['end_slope'])
click_pen = 0.08 * max(0.0, S_out['clicks'] - 3.0) + 0.02 * S_out['clicks']
karkar_pen = 4.0 * S_out['hf_karkar']

S_out['score'] = (
    -3.0 * S_out['cer']
    + 0.5 * S_out['utmos']
    + 3.0 * S_out['spk_sim']
    - 0.5 * S_out['hindi']
    - rise_pen
    - slope_pen
    - click_pen
    - karkar_pen
    - 0.01 * S_out['f0_med_std']
    - 0.10 * S_out['cps_std']
    - 0.05 * S_out['rms_std_db']
)
S_out['model'] = MODEL
json.dump(dict(summary=S_out, rows=rows), open(OUT + '/eval.json', 'w'), ensure_ascii=False, indent=1)
print('EVAL_V4_SUMMARY', json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in S_out.items()}), flush=True)
