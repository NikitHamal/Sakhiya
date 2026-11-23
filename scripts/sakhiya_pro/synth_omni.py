import sys, os, json, time, torch, soundfile as sf
sys.path.insert(0, '/work/sp'); from bake_sents import SENTS
from omnivoice.models.omnivoice import OmniVoice
ref = json.load(open('/work/sp/ref/anchor_ref.json'))
m = OmniVoice.from_pretrained('/work/sp/models/OmniVoice', device_map='cuda', dtype=torch.bfloat16)
lang = os.environ.get('OMNI_LANG', 'npi'); out = '/work/sp/bakeoff/omnivoice' + ('' if lang == 'npi' else '_' + lang); os.makedirs(out, exist_ok=True)
lang = os.environ.get('OMNI_LANG', 'npi')
for s in SENTS:
    for var in ('dev', 'lat'):
        if not s[var]: continue
        t0 = time.time()
        a = m.generate(text=s[var], language=lang, ref_audio='/work/sp/ref/anchor_ref.wav', ref_text=ref['ref_text'], num_step=32, guidance_scale=2.0)
        sf.write(f"{out}/{s['id']}_{var}.wav", a[0], m.sampling_rate); print(s['id'], var, round(time.time() - t0, 1), len(a[0]) / m.sampling_rate, flush=True)
