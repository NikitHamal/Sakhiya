#!/usr/bin/env python3
"""Synthesize the 8 hard sentences (+ fixed dev lines) from a checkpoint and score the *audio*
(Kriti CTC ASR CER, Hindi leakage, UTMOS, ECAPA similarity to the anchor, click/pause forensics).
usage: eval_ckpt.py <ckpt_dir> <tag> [num_step=32] [guidance=2.0]"""
import sys, os, json, numpy as np, torch, soundfile as sf
sys.path.insert(0, '/work/sp'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sp_common as C
from bake_sents import SENTS
from omnivoice.models.omnivoice import OmniVoice
ck, tag = sys.argv[1], sys.argv[2]
ns = int(sys.argv[3]) if len(sys.argv) > 3 else 32; gs = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
ref = json.load(open('/work/sp/ref/anchor_ref.json'))
m = OmniVoice.from_pretrained(ck, device_map='cuda', dtype=torch.bfloat16)
out = f'/work/sp/evals/{tag}'; os.makedirs(out, exist_ok=True)
rw, rsr = sf.read('/work/sp/ref/anchor_ref.wav', dtype='float32'); remb = C.spk_emb(rw, rsr)
res = {}
for s in SENTS:
    for var in ('dev', 'lat'):
        if not s[var]: continue
        torch.manual_seed(1234)
        a = m.generate(text=s[var], language='npi', ref_audio='/work/sp/ref/anchor_ref.wav', ref_text=ref['ref_text'], num_step=ns, guidance_scale=gs)
        p = f"{out}/{s['id']}_{var}.wav"; sf.write(p, a[0], m.sampling_rate)
        res[f"{s['id']}_{var}"] = C.score_take(p, s['dev'], remb)
ok = list(res.values()); g = lambda k: float(np.mean([v[k] for v in ok]))
summ = dict(tag=tag, n=len(ok), cer=g('cer'), wer=g('wer'), utmos=g('utmos'), spk_sim=g('spk_sim'), clicks=g('clicks'), pauses=g('pauses'),
            hindi=sum(v['hindi'] for v in ok), cer_vedic=float(np.mean([v['cer'] for k, v in res.items() if k.startswith('s8')])),
            cer_tech=float(np.mean([v['cer'] for k, v in res.items() if k.split('_')[0] in ('s1', 's2', 's3', 's4', 's5')])))
json.dump(dict(summary=summ, takes=res), open(f'{out}/scores.json', 'w'), ensure_ascii=False, indent=1)
print('EVAL', json.dumps({k: round(x, 3) if isinstance(x, float) else x for k, x in summ.items()}), flush=True)
