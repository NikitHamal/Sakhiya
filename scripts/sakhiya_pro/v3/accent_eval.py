#!/usr/bin/env python3
"""Fast v3 accent + consistency eval of one model dir on the 12 audit lines (fixed config from the audit).
usage: accent_eval.py MODEL_DIR OUT_DIR   -> OUT_DIR/eval.json (+ seed-11 wavs); prints one JSON summary line."""
import sys, os, json, numpy as np, soundfile as sf, torch
sys.path.insert(0, '/work/sp/repo/scripts/sakhiya_pro/v3'); sys.path.insert(0, '/work/sp/repo/scripts/sakhiya_pro'); sys.path.insert(0, '/work/sp')
import sp_common as C, validators as V, nep_norm
from nepal_loan import nativize
from accent_lib import accent_metrics
MODEL, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT + '/wav', exist_ok=True)
LINES = json.load(open('/work/sp/v3/audit_v2/lines.json'))
REF = '/work/sp/v3/ref/anchor_v3.wav'; RT = json.load(open('/work/sp/v3/ref/anchor_v3.json'))['ref_text']
from omnivoice.models.omnivoice import OmniVoice
m = OmniVoice.from_pretrained(MODEL, device_map='cuda', dtype=torch.bfloat16)
rw, rsr = sf.read(REF, dtype='float32'); emb = C.spk_emb(rw, rsr); rows = []
for i, t0 in enumerate(LINES):
    for sd in (11, 22):
        torch.manual_seed(sd)
        a = m.generate(text=nep_norm.normalize(nativize(t0)), language='npi', instruct=None, ref_audio=REF, ref_text=RT, num_step=32, guidance_scale=2.0)
        w = np.asarray(a[0], np.float32); p = f'{OUT}/wav/l{i:02d}_s{sd}.wav'; sf.write(p, w, m.sampling_rate)
        f = C.score_take(p, t0, emb); f['cer'] = V.pron_cer(t0, f.get('hyp', ''), 'npi'); f.update(accent_metrics(w, m.sampling_rate))
        f['rms'] = float(np.sqrt(np.mean(w ** 2))); f['line'] = i; f['seed'] = sd; f['hyp'] = f.get('hyp', ''); rows.append(f)
        if sd != 11: os.remove(p)
k = ['cer', 'utmos', 'spk_sim', 'hindi', 'clicks', 'end_slope', 'rise_frac', 'decl', 'f0_std', 'cps']
S = {x: float(np.mean([r[x] for r in rows])) for x in k}
S['f0_med_std'] = float(np.std([r['f0_med'] for r in rows])); S['cps_std'] = float(np.std([r['cps'] for r in rows]))
S['rms_std_db'] = float(np.std(20 * np.log10([max(r['rms'], 1e-5) for r in rows])))
S['score'] = -3 * S['cer'] + 0.5 * S['utmos'] + 3.0 * S['spk_sim'] - 0.5 * S['hindi'] - 0.02 * S['clicks'] - 0.25 * max(0, S['end_slope']) - 0.5 * S['rise_frac'] - 0.01 * S['f0_med_std'] - 0.1 * S['cps_std'] - 0.05 * S['rms_std_db']
S['model'] = MODEL
json.dump(dict(summary=S, rows=rows), open(OUT + '/eval.json', 'w'), ensure_ascii=False, indent=1)
print('EVAL_SUMMARY', json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in S.items()}), flush=True)
