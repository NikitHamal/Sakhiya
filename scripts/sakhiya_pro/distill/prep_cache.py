#!/usr/bin/env python3
"""Pre-extract distillation shards: exact 24 kHz int16 waveform + phoneme ids + frame-level F0 (WORLD) into ONE torch file.
usage: prep_cache.py MANIFEST.jsonl OUT.pt   (rows: id, wav, text, lang 'ne'|'sa'; text = exactly what the teacher spoke)
Log-mels are computed on the GPU inside the trainer (faster than any disk read); alignments come from MAS during training."""
import sys, os, json, numpy as np
from multiprocessing import Pool
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')); sys.path.insert(0, REPO)
HOP = 256; SR = 24000


def one(r):
    import soundfile as sf, librosa, pyworld as pw
    from sakhiya import g2p, symbols
    w, sr = sf.read(r['wav'], dtype='float32')
    if w.ndim > 1: w = w.mean(1)
    if sr != SR: w = librosa.resample(w, orig_sr=sr, target_sr=SR)
    n = len(w) // HOP * HOP; w = w[:n]
    try: ids = symbols.encode(g2p.text_to_phones(r['text'], 'sa' if r['lang'] == 'sa' else 'ne'))
    except Exception: return None
    T = n // HOP
    if T < 80 or len(ids) < 3 or len(ids) > T * 0.6: return None
    w64 = w.astype(np.float64)
    f0, t = pw.dio(w64, SR, f0_floor=60, f0_ceil=450, frame_period=HOP / SR * 1000)
    f0 = pw.stonemask(w64, f0, t, SR)[:T]
    f0 = np.pad(f0, (0, T - len(f0))).astype(np.float32)
    return dict(id=r['id'], ids=np.array(ids, np.int16), lang=1 if r['lang'] == 'sa' else 0, wav=(np.clip(w, -1, 1) * 32767).astype(np.int16), f0=f0, split=r.get('split', 'train'))


if __name__ == '__main__':
    import torch
    rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    with Pool(int(os.environ.get('NPROC', 16))) as p: out = [x for x in p.imap(one, rows, chunksize=8) if x]
    lf = np.concatenate([np.log(x['f0'][x['f0'] > 0]) for x in out]) if out else np.array([5.3])
    hrs = sum(len(x['wav']) for x in out) / SR / 3600
    items = [dict(id=x['id'], ids=torch.from_numpy(x['ids'].astype(np.int64)), lang=x['lang'], wav=torch.from_numpy(x['wav']), f0=torch.from_numpy(x['f0']), split=x['split']) for x in out]
    torch.save(dict(items=items, f0_mu=float(lf.mean()), f0_sd=float(lf.std()), hours=hrs), sys.argv[2])
    print(json.dumps(dict(n_in=len(rows), n_out=len(out), hours=round(hrs, 2), f0_mu=float(lf.mean()), f0_sd=float(lf.std()))))
