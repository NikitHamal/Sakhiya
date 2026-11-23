#!/usr/bin/env python3
"""Sakhiya-Pro Studio: any text prompt -> broadcast-mastered voiceover -> captioned MP4.

Worker for the MoLab "Sakhiya-Pro Studio" card (monitor/cells_studio.py). One job = one directory:
  <job>/job.json     input  {text, model, lang, takes, steps, pace, fmt, title}
  <job>/status.json  live progress (stage, done/total, per-chunk log) - polled by the card
  <job>/voice.wav / voice.mp3 / video.mp4 / captions.ass / meta.json   outputs

Pipeline: sentence/clause chunking -> Best-of-N takes per chunk (Kriti ASR CER, clicks, Hindi leak, pace, UTMOS,
speaker similarity) -> natural pauses -> optional pace -> ffmpeg broadcast chain (80 Hz HPF, warmth/presence EQ,
gentle compression, -14 LUFS) -> 1280x720 or 720x1280 MP4 (waveform + libass-shaped Devanagari captions).

usage: python3 scripts/sakhiya_pro_studio.py /work/sp/studio/jobs/<id>
"""
import json, os, re, subprocess, sys, time, traceback
import numpy as np, soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
FONTS = '/usr/share/fonts/truetype/noto'
VEDIC = re.compile('[\u0951-\u0954ॐ॥]')


def split_text(text, max_len=190):
    """-> list of dict(text, pause_after). Splits on । ? ! ॥ and '. ' ; long sentences split at commas/spaces."""
    out = []
    paras = [p.strip() for p in re.split(r'\n\s*\n|\n', text.replace('\r', '')) if p.strip()]
    for pi, para in enumerate(paras):
        sents = [s.strip() for s in re.split(r'(?<=[।॥?!])\s*|(?<=[.])\s+', para) if s.strip()]
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
                pause = 0.20 if not last_part else (0.38 if re.search(r'[।॥?!.]$', p) else 0.28)
                if last_part and si == len(sents) - 1: pause = 0.75 if pi < len(paras) - 1 else 0.0
                out.append(dict(text=p, pause=pause))
    return out


def resolve_model(m):
    if m and os.path.isdir(m): return m
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


def main(job):
    J = json.load(open(f'{job}/job.json')); st = dict(stage='starting', done=0, total=0, log=[], started=time.time(), error=None)
    def save(**k):
        st.update(k); st['updated'] = time.time(); json.dump(st, open(f'{job}/status.json.tmp', 'w'), ensure_ascii=False); os.replace(f'{job}/status.json.tmp', f'{job}/status.json')
    try:
        chunks = split_text(J['text'])
        if not chunks: raise ValueError('empty text')
        save(stage='loading model', total=len(chunks))
        import sakhiya_pro_render as SR
        model = resolve_model(J.get('model'))
        R = SR.Renderer(model, n_take=int(J.get('takes', 3)), num_step=int(J.get('steps', 32)))
        seeds_all = [11, 22, 33, 44, 55][:R.n]
        sr = None; track = []; t = 0.0; caps = []; meta = []; pace = float(J.get('pace', 1.0))
        for i, c in enumerate(chunks):
            lang = J.get('lang', 'auto')
            if lang == 'auto': lang = 'sa' if VEDIC.search(c['text']) else 'npi'
            save(stage=f'synthesizing {i + 1}/{len(chunks)}', done=i)
            w, sr, info = R.take(c['text'], lang=lang, seeds=seeds_all)
            ch = [x for x in info['cands'] if x['seed'] == info['chosen']][0]
            c0 = t; track.append(w); t += len(w) / sr
            caps.append(dict(text=c['text'], t0=c0 / pace, t1=t / pace))
            if c['pause'] > 0: track.append(np.zeros(int(c['pause'] * sr), np.float32)); t += c['pause']
            meta.append(dict(text=c['text'], lang=lang, chosen_seed=info['chosen'], cer=round(ch['cer'], 3), utmos=round(ch['utmos'], 2), clicks=ch['clicks'], hindi=ch['hindi'], dur=round(len(w) / sr, 2)))
            st['log'].append(f"{i + 1}. [{lang}] take {info['chosen']} · CER {ch['cer']:.2f} · UTMOS {ch['utmos']:.2f} · clicks {ch['clicks']:.0f} · {len(w) / sr:.1f}s")
            save(done=i + 1)
        save(stage='mastering (-14 LUFS)')
        raw = f'{job}/raw.wav'; x = np.concatenate(track).astype(np.float32); sf.write(raw, x, sr)
        src = raw
        if abs(pace - 1.0) > 0.005:
            src = f'{job}/paced.wav'; subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', raw, '-af', f'atempo={pace:.3f}', src], check=True)
        SR.master(src, f'{job}/voice.wav'); SR.master(src, f'{job}/voice.mp3')
        total = len(x) / sr / pace
        lufs = lufs_of(f'{job}/voice.wav')
        save(stage='rendering MP4')
        make_video(job, f'{job}/voice.wav', caps, total, J.get('fmt', 'land'), J.get('title') or 'Sakhiya-Pro')
        for f in ('raw.wav', 'paced.wav'):
            if os.path.exists(f'{job}/{f}'): os.remove(f'{job}/{f}')
        json.dump(dict(job=J, model=model, duration_s=round(total, 2), lufs=lufs, chunks=meta, secs=round(time.time() - st['started'], 1)), open(f'{job}/meta.json', 'w'), ensure_ascii=False, indent=1)
        save(stage='done', done=len(chunks), finished=time.time(), duration_s=round(total, 2), lufs=lufs)
    except Exception as e:
        traceback.print_exc(); save(stage='error', error=f'{type(e).__name__}: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main(os.path.abspath(sys.argv[1]))
