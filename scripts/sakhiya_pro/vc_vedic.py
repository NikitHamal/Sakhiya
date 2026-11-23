import sys, os, json, time, random, numpy as np, torch, soundfile as sf, librosa, types
sys.path.insert(0, '/work/sp/seed-vc'); os.chdir('/work/sp/seed-vc')
os.environ['HF_TOKEN'] = open('/tmp/hf_token').read().strip()
W = '/work/sp/data'; os.makedirs(W + '/ved_wav', exist_ok=True); os.makedirs(W + '/ved_vc', exist_ok=True)
gold = json.load(open(W + '/mm/vedic/gold.json')); ref = json.load(open(W + '/mm/vedic/ref_text.json'))
d = torch.load(W + '/v13/real_audio_24k.pt', weights_only=False)
off = np.concatenate([[0], np.cumsum(d['lens'].numpy())]); idx = {p: i for i, p in enumerate(d['paths'])}
c = [(p, g) for p, g in gold.items() if g.get('tier') == 'gold' and g.get('melodic') and 4 <= g['dur'] <= 14 and g['match'] >= 85 and p in idx and p in ref]
random.Random(0).shuffle(c); seen = set(); sel = []
for p, g in c:
    t = ref[p]
    if t in seen: continue
    seen.add(t); sel.append((p, t))
    if len(sel) >= 140: break
print('vedic selected', len(sel), flush=True)
json.dump([dict(id=os.path.basename(p)[:-4], text=t, i=idx[p]) for p, t in sel], open(W + '/ved_sel.json', 'w'), ensure_ascii=False)
_m = types.ModuleType('dac'); _n = types.ModuleType('dac.nn'); _q = types.ModuleType('dac.nn.quantize')
class VectorQuantize(torch.nn.Module): pass
_q.VectorQuantize = VectorQuantize; _m.nn = _n; _n.quantize = _q
sys.modules.update({'dac': _m, 'dac.nn': _n, 'dac.nn.quantize': _q})
from seed_vc_wrapper import SeedVCWrapper
vc = SeedVCWrapper(); ok = 0
for k, (p, t) in enumerate(sel):
    i = idx[p]; idn = os.path.basename(p)[:-4]; src = f'{W}/ved_wav/{idn}.wav'; dst = f'{W}/ved_vc/{idn}.wav'
    if os.path.exists(dst): continue
    sf.write(src, d['wav'][off[i]:off[i + 1]].float().numpy(), 24000)
    try:
        last = None
        for _, full in vc.convert_voice(src, W + '/anchor_prompt_20s.wav', diffusion_steps=30, inference_cfg_rate=0.7, f0_condition=True, auto_f0_adjust=True, stream_output=True):
            if full is not None: last = full
        sr_out, a = last if isinstance(last, tuple) else (22050, last)
        a = librosa.resample(np.asarray(a, np.float32).reshape(-1), orig_sr=sr_out, target_sr=24000); sf.write(dst, a, 24000); ok += 1
    except Exception as e: print('ERR', idn, repr(e)[:200], flush=True)
    if k % 20 == 0: print(k, len(sel), ok, flush=True)
print('VED_VC_DONE', ok, flush=True)
