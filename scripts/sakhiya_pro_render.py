#!/usr/bin/env python3
"""Sakhiya-Pro broadcast render + mastering pipeline.

 * Best-of-N (default 3) candidate takes per line, auto-selected by ASR CER (Kriti CTC), duration/pace sanity,
   click forensics, Hindi-leak count and UTMOS.
 * ffmpeg broadcast vocal chain: 80 Hz high-pass, subtle warmth (+1.5 dB @ 180 Hz) and presence (+2 dB @ 3.5 kHz) EQ,
   gentle 2:1 compression, two-pass loudnorm to -14 LUFS (TP -1.5 dBTP).

usage (on the GPU box):
  python scripts/sakhiya_pro_render.py --model /work/sp/final/sakhiya_pro_v1 --ep01 --out /work/sp/render/ep01
  python scripts/sakhiya_pro_render.py --model ... --pack --out /work/sp/render/pack
  python scripts/sakhiya_pro_render.py --model ... --text "नमस्ते।" --out /tmp/x.wav
"""
import argparse, json, os, re, subprocess, sys, tempfile
import numpy as np, soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, 'sakhiya_pro')); sys.path.insert(0, '/work/sp')
REF_WAV = os.environ.get('SP_REF_WAV', '/work/sp/ref/anchor_ref.wav'); REF_JSON = os.environ.get('SP_REF_JSON', '/work/sp/ref/anchor_ref.json')
SVARA = re.compile('[\u0951\u0952\u0953\u0954]')
CHAIN = ('highpass=f=80,equalizer=f=180:t=q:w=1.0:g=1.5,equalizer=f=3500:t=q:w=1.2:g=2.0,'
         'acompressor=threshold=-20dB:ratio=2:attack=15:release=160:makeup=2')

def master(src, dst, lufs=-14.0, tp=-1.5, sr=48000):
    """ffmpeg broadcast chain with two-pass loudnorm. Writes 48 kHz wav (or mp3 by extension)."""
    f1 = f'{CHAIN},loudnorm=I={lufs}:TP={tp}:LRA=9:print_format=json'
    r = subprocess.run(['ffmpeg', '-hide_banner', '-nostats', '-i', src, '-af', f1, '-f', 'null', '-'], capture_output=True, text=True)
    m = re.search(r'\{[^{}]*"input_i"[^{}]*\}', r.stderr, re.S)
    ln = ''
    if m:
        j = json.loads(m.group(0))
        ln = (f':measured_I={j["input_i"]}:measured_TP={j["input_tp"]}:measured_LRA={j["input_lra"]}:measured_thresh={j["input_thresh"]}'
              f':offset={j["target_offset"]}:linear=true')
    f2 = f'{CHAIN},loudnorm=I={lufs}:TP={tp}:LRA=9{ln}'
    codec = ['-codec:a', 'libmp3lame', '-qscale:a', '2'] if dst.endswith('.mp3') else ['-c:a', 'pcm_s16le']
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', src, '-af', f2, '-ar', str(sr), '-ac', '1'] + codec + [dst], check=True)

def trim(w, sr, lead=0.08, tail=0.18):
    a = np.abs(w); thr = max(0.008, 0.02 * float(a.max())); idx = np.where(a > thr)[0]
    if not idx.size: return w
    w = w[max(0, idx[0] - int(lead * sr)): min(len(w), idx[-1] + int(tail * sr))].copy()
    f = int(0.012 * sr); w[:f] *= np.linspace(0, 1, f); w[-f:] *= np.linspace(1, 0, f)
    return w

class Renderer:
    def __init__(self, model_dir, n_take=3, num_step=32, guidance=2.0):
        import torch
        from omnivoice.models.omnivoice import OmniVoice
        import sp_common as C
        self.C, self.torch, self.n, self.ns, self.gs = C, torch, n_take, num_step, guidance
        self.m = OmniVoice.from_pretrained(model_dir, device_map='cuda', dtype=torch.bfloat16)
        self.ref_text = json.load(open(REF_JSON))['ref_text']
        rw, rsr = sf.read(REF_WAV, dtype='float32'); self.remb = C.spk_emb(rw, rsr)

    def score(self, path, text):
        f = self.C.score_take(path, text, self.remb)
        n = max(1, len(re.findall(r'[क-ह]', text)))
        cps_pen = 0.0 if 4.0 <= f['cps'] <= 9.5 else 0.5   # pace sanity: very rushed or dragging = bad
        f['sel'] = (-3.0 * f['cer'] - 0.25 * f['clicks'] - 0.5 * f['hindi'] - cps_pen - 0.5 * max(0, f['long_gaps'])
                    + 0.3 * f['utmos'] + 0.5 * f['spk_sim'])
        return f

    def take(self, text, lang='npi', seeds=None):
        text = SVARA.sub('', text)
        seeds = seeds or [11, 22, 33][:self.n]
        tmp = tempfile.mkdtemp(); best = None; cands = []
        for k, sd in enumerate(seeds):
            self.torch.manual_seed(sd)
            a = self.m.generate(text=text, language=lang, ref_audio=REF_WAV, ref_text=self.ref_text, num_step=self.ns, guidance_scale=self.gs)
            w = trim(np.asarray(a[0], dtype=np.float32), self.m.sampling_rate)
            p = f'{tmp}/t{k}.wav'; sf.write(p, w, self.m.sampling_rate)
            f = self.score(p, text); cands.append(dict(seed=sd, **{x: f[x] for x in ('cer', 'clicks', 'hindi', 'utmos', 'spk_sim', 'dur', 'sel')}))
            if best is None or f['sel'] > best[0]: best = (f['sel'], p, f, w)
        return best[3], self.m.sampling_rate, dict(chosen=max(cands, key=lambda c: c['sel'])['seed'], cands=cands)

def render_ep01(R, out, final_mp3=True):
    os.makedirs(out + '/lines', exist_ok=True)
    S = json.load(open(os.path.join(ROOT, 'video', 'script.json'))); sr = 48000
    tl, track, t = [], [], 0.0
    def add_sil(sec):
        nonlocal t
        track.append(np.zeros(int(sec * sr), np.float32)); t += sec
    report = {}
    for sc in S['scenes']:
        add_sil(sc.get('pre', 0.5))
        for ln in sc['lines']:
            w, wsr, info = R.take(ln['ne']); raw = f"{out}/lines/{ln['id']}_raw.wav"; sf.write(raw, w, wsr)
            fin = f"{out}/lines/{ln['id']}.wav"; master(raw, fin)
            x, _ = sf.read(fin, dtype='float32'); tl.append(dict(id=ln['id'], scene=sc['id'], start=round(t, 3), end=round(t + len(x) / sr, 3), text=ln['ne']))
            track.append(x); t += len(x) / sr; report[ln['id']] = info; add_sil(0.28)
        add_sil(sc.get('post', 0.5))
    full = np.concatenate(track); sf.write(out + '/voiceover_raw.wav', full, sr)
    master(out + '/voiceover_raw.wav', out + '/voiceover_ep01.wav'); master(out + '/voiceover_raw.wav', out + '/voiceover_ep01.mp3')
    json.dump(tl, open(out + '/timeline.json', 'w'), ensure_ascii=False, indent=1); json.dump(report, open(out + '/take_report.json', 'w'), ensure_ascii=False, indent=1)
    print('EP01 done', round(t, 1), 's', flush=True)

def render_pack(R, out):
    sys.path.insert(0, HERE)
    reg = json.load(open(os.path.join(ROOT, 'docs', 'samples', 'versions', 'registry.json')))
    clips = reg['versions'][0]['clips']; os.makedirs(out, exist_ok=True); meta = []
    for c in clips:
        lang = 'sa' if c['lang'] == 'sa' else 'npi'
        w, wsr, info = R.take(c['text'], lang=lang); raw = f"{out}/{c['id']}_raw.wav"; sf.write(raw, w, wsr)
        mp3 = f"{out}/{c['id']}.mp3"; master(raw, mp3); os.remove(raw)
        x, sr = sf.read(mp3, dtype='float32') if False else (w, wsr)
        meta.append(dict(id=c['id'], lang=c['lang'], style=c['style'], text=c['text'], mp3=f"{c['id']}.mp3", duration_s=round(len(w) / wsr, 2), take=info))
        print(c['id'], meta[-1]['duration_s'], info['chosen'], flush=True)
    json.dump(meta, open(out + '/pack_meta.json', 'w'), ensure_ascii=False, indent=1)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--ep01', action='store_true'); ap.add_argument('--pack', action='store_true'); ap.add_argument('--text')
    ap.add_argument('--lang', default='npi'); ap.add_argument('--takes', type=int, default=3); ap.add_argument('--steps', type=int, default=32)
    a = ap.parse_args()
    R = Renderer(a.model, a.takes, a.steps)
    if a.ep01: render_ep01(R, a.out)
    elif a.pack: render_pack(R, a.out)
    elif a.text:
        w, sr, info = R.take(a.text, a.lang); raw = a.out + '.raw.wav'; sf.write(raw, w, sr); master(raw, a.out); os.remove(raw); print(json.dumps(info))
