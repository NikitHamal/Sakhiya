#!/usr/bin/env python3
"""Sakhiya-Pro Studio: any text prompt -> broadcast-mastered MP3 (optionally + captioned MP4).

Worker for the MoLab "Sakhiya-Pro Studio" card (monitor/cells_studio.py). One job = one directory:
  <job>/job.json     input  {text, model, lang, takes, steps, pace, bitrate, master, video, fmt, title, variation}
  <job>/status.json  live progress (stage, done/total, per-chunk log) - polled by the card
  <job>/voice.mp3 / voice.wav / [video.mp4 captions.ass] / meta.json   outputs

Modes:
  python3 scripts/sakhiya_pro_studio.py --serve [/work/sp/studio]   warm daemon: model stays loaded, picks up queued jobs
                                                                    (heartbeat in <root>/server.json, exits after 60 min idle)
  python3 scripts/sakhiya_pro_studio.py <job_dir>                   one-shot (loads the model, renders one job)
  python3 scripts/sakhiya_pro_studio.py --text "नमस्ते।" --out /tmp/x.mp3     CLI: text -> mp3

Pipeline: sentence/clause chunking -> Best-of-N takes per chunk (Kriti ASR CER, clicks, Hindi leak, pace, UTMOS,
speaker similarity) -> natural pauses -> optional pace -> ffmpeg broadcast chain (80 Hz HPF, warmth/presence EQ,
gentle compression, -14 LUFS) -> 1280x720 or 720x1280 MP4 (waveform + libass-shaped Devanagari captions).

usage: python3 scripts/sakhiya_pro_studio.py /work/sp/studio/jobs/<id>
"""
import glob, json, os, re, subprocess, sys, tempfile, time, traceback
import numpy as np, soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
FONTS = '/usr/share/fonts/truetype/noto'
VEDIC = re.compile('[\u0951-\u0954ॐ॥]')


ROOT_DIR = '/work/sp/studio'
ABBR = re.compile(r'(?:\b[A-Z]|\b(?:Dr|Mr|Mrs|Ms|Prof|St|Rs|No|vs|etc|Inc|Ltd))\.$')


def normalize_text(text):
    """Make pasted text safe for the model: strip markdown/URLs/zero-width junk, Latin '.'/'|' -> '।'/'॥' after Devanagari,
    tidy spacing before punctuation and guarantee a closing mark per paragraph (gives the downward sentence closure)."""
    t = text.replace('\r', '')
    t = re.sub(r'https?://\S+|www\.\S+', '', t)
    t = re.sub(r'[`*_#>~]+', '', t)
    t = re.sub('[\u200b\u2060\ufeff]', '', t)
    t = t.replace('\u2026', ',').replace('...', ',').replace('\u201c', '"').replace('\u201d', '"').replace('\u2018', "'").replace('\u2019', "'")
    t = t.replace('||', '॥').replace('|', '।')
    t = re.sub(r'(?<=[\u0900-\u097f])\s*\.(?=\s|$)', '।', t)
    t = re.sub(r'(?<=\d)\s*%', ' प्रतिशत' if re.search(r'[\u0900-\u097f]', t) else '%', t)
    t = re.sub(r'[ \t]+', ' ', t)
    t = re.sub(r'\s+([।॥?!,;:])', r'\1', t)
    paras = []
    for p in re.split(r'\n\s*\n|\n', t):
        p = p.strip(' -•\t')
        if not p: continue
        if not re.search(r'[।॥?!.:;]["\')]*$', p):
            p += '।' if re.search(r'[\u0900-\u097f]', p) else '.'
        paras.append(p)
    return '\n\n'.join(paras)


def split_text(text, max_len=190):
    """-> list of dict(text, pause). Splits on । ? ! ॥ and '. '; long sentences split at commas/spaces. Abbreviations (Dr. Rs.) are kept whole."""
    out = []
    paras = [p.strip() for p in re.split(r'\n\s*\n|\n', text) if p.strip()]
    for pi, para in enumerate(paras):
        raw = [s.strip() for s in re.split(r'(?<=[।॥?!])\s*|(?<=[.])\s+', para) if s.strip()]
        sents = []
        for s in raw:   # glue pieces that were split after an abbreviation
            if sents and ABBR.search(sents[-1]): sents[-1] += ' ' + s
            else: sents.append(s)
        for si, s in enumerate(sents):
            parts = [s]
            while len(parts[-1]) > max_len:
                t = parts.pop(); cut = -1
                for m in re.finditer(r'[,;:—–]\s+', t[:max_len]): cut = m.end()
                if cut < max_len * 0.4:
                    cut = t.rfind(' ', 0, max_len)
                if cut <= 0: cut = max_len
                parts += [t[:cut].strip(), t[cut:].strip()]
            for k, p in enumerate(parts):
                last_part = k == len(parts) - 1
                pause = 0.20 if not last_part else (0.38 if re.search(r'[।॥?!.]["\')]*$', p) else 0.28)
                if last_part and si == len(sents) - 1: pause = 0.75 if pi < len(paras) - 1 else 0.0
                out.append(dict(text=p, pause=pause))
    return out


def resolve_model(m):
    if m and os.path.isdir(m) and os.path.exists(os.path.join(m, 'model.safetensors')): return m
    return '/work/sp/final/sakhiya_pro_v1'


def ass_time(t):
    h, r = divmod(t, 3600); mi, s = divmod(r, 60)
    return f'{int(h)}:{int(mi):02d}:{s:05.2f}'


def write_ass(path, caps, W, H, title, total):
    big = int(H * (0.058 if W > H else 0.040)); small = int(H * 0.030)
    hdr = (f'[Script Info]\nScriptType: v4.00+\nPlayResX: {W}\nPlayResY: {H}\nWrapStyle: 0\nScaledBorderAndShadow: yes\n\n'
           '[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,'
           'ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n'
           f'Style: Cap,Noto Sans Devanagari,{big},&H00FFFFFF,&H000000FF,&H00301A0B,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,5,{int(W*0.07)},{int(W*0.07)},{int(H*0.10)},1\n'
           f'Style: Title,Noto Sans Devanagari,{small},&H00D5EA5E,&H000000FF,&H00301A0B,&H00000000,-1,0,0,0,100,100,2,0,1,2,0,8,40,40,{int(H*0.04)},1\n\n'
           '[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n')
    ev = [f'Dialogue: 0,{ass_time(0)},{ass_time(total + 0.5)},Title,,0,0,0,,{clean(title)}']
    for c in caps:
        ev.append(f'Dialogue: 1,{ass_time(c["t0"])},{ass_time(c["t1"] + 0.12)},Cap,,0,0,0,,{{\\fad(120,120)}}{clean(c["text"])}')
    open(path, 'w', encoding='utf8').write(hdr + '\n'.join(ev) + '\n')


def clean(t):
    return re.sub(r'[{}\\]', '', t).replace('\n', '\\N')


def make_video(job, wav48, caps, total, fmt, title):
    W, H = (1280, 720) if fmt != 'vert' else (720, 1280)
    write_ass(f'{job}/captions.ass', caps, W, H, title, total)
    wh = int(H * (0.26 if W > H else 0.14))
    fc = (f'[1:a]asplit[a1][a2];[a2]showwaves=s={W}x{wh}:mode=cline:colors=0x5eead4:rate=30[w];'
          f'[0:v][w]overlay=0:H-h-{int(H*0.03)}[v1];[v1]subtitles=captions.ass:fontsdir={FONTS}[v]')
    bg = f'gradients=s={W}x{H}:c0=0x0b1a30:c1=0x1b1140:x0=0:y0=0:x1={W}:y1={H}:d={total + 1:.2f}:r=30:speed=0.0'
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi', '-i', bg, '-i', 'voice.wav', '-filter_complex', fc, '-map', '[v]', '-map', '[a1]',
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'veryfast', '-crf', '20', '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', 'video.mp4']
    r = subprocess.run(cmd, cwd=job, capture_output=True, text=True)
    if r.returncode != 0:  # fallback: plain colour background if the gradients source is unavailable
        cmd[7] = f'color=c=0x0b1a30:s={W}x{H}:r=30:d={total + 1:.2f}'
        r = subprocess.run(cmd, cwd=job, capture_output=True, text=True)
        if r.returncode != 0: raise RuntimeError('ffmpeg video failed: ' + r.stderr[-600:])


def lufs_of(path):
    r = subprocess.run(['ffmpeg', '-hide_banner', '-nostats', '-i', path, '-af', 'loudnorm=I=-14:print_format=json', '-f', 'null', '-'], capture_output=True, text=True)
    m = re.search(r'"input_i"\s*:\s*"(-?[0-9.]+)"', r.stderr)
    return float(m.group(1)) if m else None


def fix_loudness(path, target=-14.0):
    """loudnorm under-shoots very short clips (< ~4 s): measure and apply a corrective gain + true-peak limiter."""
    for _ in range(2):
        l = lufs_of(path)
        if l is None or abs(l - target) <= 0.7: return l
        tmp = path + '.fix.wav'
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', path, '-af', f'volume={target - l:.2f}dB,alimiter=limit=0.84:level=disabled', '-ar', '48000', '-ac', '1', '-c:a', 'pcm_s16le', tmp], check=True)
        os.replace(tmp, path)
    return lufs_of(path)


def encode_mp3(src_wav, dst, bitrate='256k', title='Sakhiya-Pro'):
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', src_wav, '-codec:a', 'libmp3lame', '-b:a', bitrate, '-ar', '48000', '-ac', '1',
                    '-id3v2_version', '3', '-metadata', f'title={title}', '-metadata', 'artist=Sakhiya-Pro', '-metadata', 'comment=Generated by Sakhiya-Pro (NEBians)', dst], check=True)


def plain_master(src, dst):
    """Mastering OFF: just peak-safe 48 kHz mono 16-bit (no EQ / compression / loudness change beyond -1 dBFS peak)."""
    x, sr = sf.read(src, dtype='float32'); pk = float(np.abs(x).max()) or 1.0
    sf.write(dst, (x * min(1.0, 0.89 / pk)).astype(np.float32), sr, subtype='PCM_16')


class Engine:
    """Holds the loaded Renderer so repeated jobs skip the ~20 s model load."""
    def __init__(self): self.R = None; self.model = None; self.SR = None
    def get(self, model, takes, steps):
        if self.SR is None:
            import sakhiya_pro_render as SR; self.SR = SR
        if self.R is None or self.model != model:
            if self.R is not None:
                import gc, torch; self.R = None; gc.collect(); torch.cuda.empty_cache()
            self.R = self.SR.Renderer(model, n_take=takes, num_step=steps); self.model = model
        self.R.n = takes; self.R.ns = steps
        return self.R


def run_job(job, eng):
    J = json.load(open(f'{job}/job.json')); st = dict(stage='starting', done=0, total=0, log=[], started=time.time(), error=None)
    cancel = f'{job}/cancel'
    def save(**k):
        st.update(k); st['updated'] = time.time(); json.dump(st, open(f'{job}/status.json.tmp', 'w'), ensure_ascii=False); os.replace(f'{job}/status.json.tmp', f'{job}/status.json')
    try:
        text = normalize_text(J['text']) if J.get('normalize', True) else J['text']
        chunks = split_text(text)
        if not chunks: raise ValueError('empty text')
        save(stage='loading model', total=len(chunks), text_used=text)
        model = resolve_model(J.get('model')); takes = max(1, min(5, int(J.get('takes', 3)))); steps = int(J.get('steps', 32))
        R = eng.get(model, takes, steps); SR = eng.SR
        var = int(J.get('variation', 0))
        seeds_all = [s + 100 * var for s in (11, 22, 33, 44, 55)][:takes]
        sr = None; track = []; t = 0.0; caps = []; meta = []; pace = float(J.get('pace', 1.0)); t_syn = time.time()
        for i, c in enumerate(chunks):
            if os.path.exists(cancel): raise RuntimeError('cancelled by user')
            lang = J.get('lang', 'auto')
            if lang == 'auto': lang = 'sa' if VEDIC.search(c['text']) else 'npi'
            eta = (time.time() - t_syn) / i * (len(chunks) - i) if i else None
            save(stage=f'synthesizing {i + 1}/{len(chunks)}', done=i, eta=eta)
            w, sr, info = R.take(c['text'], lang=lang, seeds=seeds_all)
            ch = [x for x in info['cands'] if x['seed'] == info['chosen']][0]
            if 'gate_pass' not in info and (ch['cer'] > 0.45 or ch['hindi'] > 1):   # safety net for older renderers (new one gates itself): dropped words / Hindi leak -> try a fresh set of seeds
                w2, sr2, info2 = R.take(c['text'], lang=lang, seeds=[s + 1000 for s in seeds_all])
                ch2 = [x for x in info2['cands'] if x['seed'] == info2['chosen']][0]
                if ch2['sel'] > ch['sel']: w, sr, info, ch = w2, sr2, info2, ch2; st['log'].append(f'   ↻ line {i + 1} re-rolled (better take found)')
            c0 = t; track.append(w); t += len(w) / sr
            caps.append(dict(text=c['text'], t0=c0 / pace, t1=t / pace))
            if c['pause'] > 0: track.append(np.zeros(int(c['pause'] * sr), np.float32)); t += c['pause']
            meta.append(dict(text=c['text'], lang=lang, chosen_seed=info['chosen'], cer=round(ch['cer'], 3), utmos=round(ch['utmos'], 2), clicks=ch['clicks'], hindi=ch['hindi'], dur=round(len(w) / sr, 2)))
            st['log'].append(f"{i + 1}. [{lang}] take {info['chosen']} · CER {ch['cer']:.2f} · UTMOS {ch['utmos']:.2f} · clicks {ch['clicks']:.0f} · {len(w) / sr:.1f}s")
            save(done=i + 1)
        save(stage='mastering', eta=None)
        raw = f'{job}/raw.wav'; x = np.concatenate(track).astype(np.float32); sf.write(raw, x, sr)
        src = raw
        if abs(pace - 1.0) > 0.005:
            src = f'{job}/paced.wav'; subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', raw, '-af', f'atempo={pace:.3f}', src], check=True)
        if J.get('master', True): SR.master(src, f'{job}/voice.wav')
        else: plain_master(src, f'{job}/voice.wav')
        if J.get('master', True): fix_loudness(f'{job}/voice.wav')
        save(stage='encoding MP3')
        encode_mp3(f'{job}/voice.wav', f'{job}/voice.mp3', J.get('bitrate', '256k'), J.get('title') or 'Sakhiya-Pro')
        total = len(x) / sr / pace
        lufs = lufs_of(f'{job}/voice.wav')
        if J.get('video'):
            save(stage='rendering MP4')
            make_video(job, f'{job}/voice.wav', caps, total, J.get('fmt', 'land'), J.get('title') or 'Sakhiya-Pro')
        for f in ('raw.wav', 'paced.wav'):
            if os.path.exists(f'{job}/{f}'): os.remove(f'{job}/{f}')
        json.dump(dict(job=J, text_used=text, model=model, duration_s=round(total, 2), lufs=lufs, chunks=meta, secs=round(time.time() - st['started'], 1)), open(f'{job}/meta.json', 'w'), ensure_ascii=False, indent=1)
        save(stage='done', done=len(chunks), finished=time.time(), duration_s=round(total, 2), lufs=lufs, eta=None)
        return True
    except Exception as e:
        traceback.print_exc(); save(stage='error', error=f'{type(e).__name__}: {e}', finished=time.time())
        return False


def serve(root=ROOT_DIR, idle_exit=3600):
    os.makedirs(f'{root}/jobs', exist_ok=True); eng = Engine(); last = time.time()
    def beat(state):
        json.dump(dict(pid=os.getpid(), ts=time.time(), state=state, model=eng.model), open(f'{root}/server.json.tmp', 'w')); os.replace(f'{root}/server.json.tmp', f'{root}/server.json')
    beat('idle'); print('studio server up, pid', os.getpid(), flush=True)
    while time.time() - last < idle_exit:
        queued = []
        for d in sorted(glob.glob(f'{root}/jobs/*')):
            try:
                if json.load(open(f'{d}/status.json')).get('stage') == 'queued' and os.path.exists(f'{d}/job.json'): queued.append(d)
            except Exception: pass
        if not queued: beat('idle'); time.sleep(1.0); continue
        d = queued[0]; beat('busy ' + os.path.basename(d)); print('job', d, flush=True)
        run_job(d, eng); last = time.time()
        import torch; torch.cuda.empty_cache()
    beat('exited (idle)'); os.remove(f'{root}/server.json')


def main(job):
    sys.exit(0 if run_job(job, Engine()) else 1)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a and a[0] == '--serve': serve(a[1] if len(a) > 1 else ROOT_DIR)
    elif a and a[0] == '--text':
        import argparse; ap = argparse.ArgumentParser(); ap.add_argument('--text'); ap.add_argument('--out', default='/tmp/sakhiya_pro.mp3'); ap.add_argument('--model', default=''); ap.add_argument('--takes', type=int, default=3); ap.add_argument('--pace', type=float, default=1.0)
        o = ap.parse_args(a); jd = tempfile.mkdtemp(prefix='studio_cli_')
        json.dump(dict(text=o.text, model=o.model, takes=o.takes, pace=o.pace), open(f'{jd}/job.json', 'w'), ensure_ascii=False)
        ok = run_job(jd, Engine()); print('ok' if ok else 'FAILED', jd)
        if ok: import shutil; shutil.copy(f'{jd}/voice.mp3', o.out); print('wrote', o.out)
    else: main(os.path.abspath(a[0]))
