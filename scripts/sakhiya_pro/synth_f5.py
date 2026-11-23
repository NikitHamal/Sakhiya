"""usage: synth_f5.py <ckpt name: swar_v6_ema|podvoices_ema> [outdir]"""
import sys, os, json, time, torch, soundfile as sf
name = sys.argv[1]
os.environ['ENGLISH_LATIN'] = '1' if name == 'swar_v6_ema' else '0'
sys.path.insert(0, '/work/sp'); sys.path.insert(0, '/work/sp/models/swar/app/pipeline'); sys.path.insert(0, '/work/sp/models/swar/app')
from bake_sents import SENTS
import clone as K
from f5_tts.api import F5TTS
ref = json.load(open('/work/sp/ref/anchor_ref.json'))
rt = K.to_model_text(ref['ref_text'], 'chars')
K.patch_f5_chunker(); K.set_pace(rt, '/work/sp/ref/anchor_ref.wav')
tts = F5TTS(model='F5TTS_Base', ckpt_file=f'/work/sp/models/swar/{name}.pt', vocab_file='/work/sp/models/swar/vocab.txt')
tts.ema_model.float()
out = f'/work/sp/bakeoff/{name}'; os.makedirs(out, exist_ok=True)
for s in SENTS:
    for var in ('dev', 'lat'):
        if not s[var]: continue
        if var == 'lat' and name != 'swar_v6_ema': continue
        g = K.to_model_text(s[var], 'chars'); t0 = time.time()
        tts.infer(ref_file='/work/sp/ref/anchor_ref.wav', ref_text=rt, gen_text=g, file_wave=f"{out}/{s['id']}_{var}.wav", remove_silence=False, speed=K.speed_for(g, 1.0), seed=1234)
        print(s['id'], var, round(time.time() - t0, 1), flush=True)
