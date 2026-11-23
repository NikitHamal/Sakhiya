import sys, os, json, time, numpy as np, torch, soundfile as sf, librosa
sys.path.insert(0, '/work/sp/seed-vc'); os.chdir('/work/sp/seed-vc')
os.environ['HF_TOKEN'] = open('/tmp/hf_token').read().strip()
W = '/work/sp/data'; os.makedirs(W + '/ia_wav', exist_ok=True); os.makedirs(W + '/ia_vc', exist_ok=True)
# anchor target prompt: best 6834 clips concatenated (~20 s), cleanly cut
cand = json.load(open('/work/sp/ref_candidates.json'))
s = next(x for x in cand if x['spk'] == '6834')
cs = sorted(s['cands'], key=lambda x: -(x['utmos'] - 3 * x['cer']))
segs, tot = [], 0
for c in cs:
    w, sr = sf.read(f"{W}/slr43/ne_np_female/wavs/{c['uid']}.wav", dtype='float32'); w = w.mean(1) if w.ndim > 1 else w
    w = librosa.resample(w, orig_sr=sr, target_sr=22050); segs += [w, np.zeros(int(0.25 * 22050), np.float32)]; tot += len(w) / 22050
    if tot > 20: break
sf.write(W + '/anchor_prompt_20s.wav', np.concatenate(segs), 22050); print('prompt s', tot, flush=True)
d = torch.load(W + '/v13/real_audio_24k.pt', weights_only=False)
off = np.concatenate([[0], np.cumsum(d['lens'].numpy())])
F = json.load(open(W + '/ia_filtered.json'))
_sh = os.environ.get('VC_SHARD')
if _sh: _i, _n = map(int, _sh.split('/')); F = F[::-1][_i::_n]
import types
_m = types.ModuleType('dac'); _n = types.ModuleType('dac.nn'); _q = types.ModuleType('dac.nn.quantize')
class VectorQuantize(torch.nn.Module): pass
_q.VectorQuantize = VectorQuantize; _m.nn = _n; _n.quantize = _q
sys.modules.update({'dac': _m, 'dac.nn': _n, 'dac.nn.quantize': _q})
from seed_vc_wrapper import SeedVCWrapper
vc = SeedVCWrapper()
n_ok = 0; t0 = time.time()
for k, x in enumerate(F):
    src = f"{W}/ia_wav/{x['id']}.wav"; dst = f"{W}/ia_vc/{x['id']}.wav"
    if os.path.exists(dst): n_ok += 1; continue
    w = d['wav'][off[x['i']]:off[x['i'] + 1]].float().numpy(); sf.write(src, w, 24000)
    try:
        last = None
        for _, full in vc.convert_voice(src, W + '/anchor_prompt_20s.wav', diffusion_steps=30, inference_cfg_rate=0.7, f0_condition=True, auto_f0_adjust=True, stream_output=True):
            if full is not None: last = full
        if last is None: continue
        sr_out, a = last if isinstance(last, tuple) else (vc.sr if hasattr(vc, 'sr') else 22050, last)
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        a = librosa.resample(a, orig_sr=sr_out, target_sr=24000)
        sf.write(dst, a, 24000); n_ok += 1
    except Exception as e: print('ERR', x['id'], repr(e)[:200], flush=True)
    if k % 20 == 0: print(k, len(F), n_ok, round(time.time() - t0), flush=True)
print('VC_DONE', n_ok, flush=True)
