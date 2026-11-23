"""Seed-VC unify other SLR43 Nepal-native studio speakers -> anchor vocal tract (6834). Clean, reliable transcripts.
env: VC_SHARD=i/n for parallel workers. Output data/slr_vc/<uid>.wav (24 kHz) + data/slr_sel.json"""
import sys, os, json, random, types, time, numpy as np, torch, soundfile as sf, librosa
sys.path.insert(0, '/work/sp/seed-vc'); os.chdir('/work/sp/seed-vc')
os.environ['HF_TOKEN'] = open('/tmp/hf_token').read().strip()
W = '/work/sp/data'; D = W + '/slr43/ne_np_female'; os.makedirs(W + '/slr_vc', exist_ok=True)
SEL = W + '/slr_sel.json'
if not os.path.exists(SEL):
    rows = [l.rstrip('\n').split('\t') for l in open(D + '/line_index.tsv', encoding='utf8')]
    rows = [(u, t) for u, t in rows if u.split('_')[1] != '6834']
    random.Random(5).shuffle(rows); sel = []
    for u, t in rows:
        if any(c.isascii() and c.isalnum() for c in t): continue
        n = sf.info(f'{D}/wavs/{u}.wav'); d = n.frames / n.samplerate
        if 2.5 <= d <= 12: sel.append(dict(id=u, text=t, dur=d))
        if len(sel) >= 1000: break
    json.dump(sel, open(SEL, 'w'), ensure_ascii=False)
sel = json.load(open(SEL)); print('selected', len(sel), flush=True)
sh = os.environ.get('VC_SHARD')
if sh: i, n = map(int, sh.split('/')); sel = sel[i::n]
_m = types.ModuleType('dac'); _n = types.ModuleType('dac.nn'); _q = types.ModuleType('dac.nn.quantize')
class VectorQuantize(torch.nn.Module): pass
_q.VectorQuantize = VectorQuantize; _m.nn = _n; _n.quantize = _q
sys.modules.update({'dac': _m, 'dac.nn': _n, 'dac.nn.quantize': _q})
from seed_vc_wrapper import SeedVCWrapper
vc = SeedVCWrapper(); ok = 0; t0 = time.time()
for k, x in enumerate(sel):
    dst = f"{W}/slr_vc/{x['id']}.wav"
    if os.path.exists(dst): continue
    try:
        last = None
        for _, full in vc.convert_voice(f"{D}/wavs/{x['id']}.wav", W + '/anchor_prompt_20s.wav', diffusion_steps=30, inference_cfg_rate=0.7, f0_condition=True, auto_f0_adjust=True, stream_output=True):
            if full is not None: last = full
        sr_out, a = last if isinstance(last, tuple) else (22050, last)
        a = librosa.resample(np.asarray(a, np.float32).reshape(-1), orig_sr=sr_out, target_sr=24000); sf.write(dst, a, 24000); ok += 1
    except Exception as e: print('ERR', x['id'], repr(e)[:200], flush=True)
    if k % 20 == 0: print(k, len(sel), ok, round(time.time() - t0), flush=True)
print('SLR_VC_DONE', ok, flush=True)
