#!/usr/bin/env python3
"""Ground-truth Pro-v3 feature shards for the student ladder (NO V1 teacher anywhere).

Merges (a) the surviving Pro-v3 cache (/work/sp/distill/cache_train.pt) with (b) every new Pro-v3 teacher take under
/work/sp/distill/corpus/part_*.jsonl, level-normalises each 24 kHz waveform, computes the exact 100-band log-mel the vocoder was
built on, and forced-aligns phonemes to frames with the frozen MelCTCAligner (Viterbi) -> exact integer durations.
Output: /work/sp/distill/gt_feats.pt  (items: ids, lang, wav int16, mel fp16 [100,T], dur int16 [L], conf, split, src)
usage: prep_gt.py [--min_utmos 3.45] [--max_cer 0.08] [--out PATH]"""
import argparse, glob, json, os, sys, hashlib
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, REPO)
SR, HOP = 24000, 256


def level(w):
    r = float(np.sqrt(np.mean(w ** 2)))
    if r < 1e-4: return None
    return np.clip(w * (0.062 / r), -0.97, 0.97).astype(np.float32)


def one(r):
    import soundfile as sf, librosa
    from sakhiya import g2p, symbols
    try:
        w, sr = sf.read(r['wav'], dtype='float32')
        if w.ndim > 1: w = w.mean(1)
        if sr != SR: w = librosa.resample(w, orig_sr=sr, target_sr=SR)
        n = len(w) // HOP * HOP; w = level(w[:n])
        if w is None: return None
        ids = symbols.encode(g2p.text_to_phones(r['text'], 'sa' if r['lang'] == 'sa' else 'ne'))
    except Exception:
        return None
    T = n // HOP
    if T < 80 or len(ids) < 3 or len(ids) > T * 0.6: return None
    return dict(id=r['id'], ids=np.array(ids, np.int16), lang=1 if r['lang'] == 'sa' else 0, wav=(w * 32767).astype(np.int16),
                split='dev' if int(hashlib.md5(r['id'].encode()).hexdigest(), 16) % 70 == 0 else 'train', src=r.get('src', ''))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min_utmos', type=float, default=3.45); ap.add_argument('--max_cer', type=float, default=0.08)
    ap.add_argument('--out', default='/work/sp/distill/gt_feats.pt'); ap.add_argument('--nproc', type=int, default=16)
    ap.add_argument('--old', default='/work/sp/distill/cache_train.pt'); ap.add_argument('--conf', type=float, default=-14.0)
    a = ap.parse_args()
    from multiprocessing import Pool
    items = []
    if os.path.exists(a.old):
        D = torch.load(a.old, weights_only=False)
        for x in D['items']:
            w = level(x['wav'].float().numpy() / 32767.0)
            if w is None: continue
            items.append(dict(id=x['id'], ids=x['ids'].long(), lang=int(x['lang']), wav=torch.from_numpy((w * 32767).astype(np.int16)),
                              split=x.get('split', 'train') if x.get('split') in ('dev', 'train') else 'train', src='old'))
        print('old cache items', len(items), flush=True)
    have = {x['id'] for x in items}
    rows = []
    for p in sorted(glob.glob('/work/sp/distill/corpus/part_*.jsonl')):
        for l in open(p):
            try: r = json.loads(l)
            except Exception: continue
            if r['id'] in have or not os.path.exists(r.get('wav', '')): continue
            if r.get('utmos', 0) < a.min_utmos: continue
            if r['lang'] != 'sa' and r.get('cer', 1) > a.max_cer: continue
            rows.append(r); have.add(r['id'])
    print('new rows', len(rows), flush=True)
    with Pool(a.nproc) as pool:
        new = [x for x in pool.imap(one, rows, chunksize=8) if x]
    for x in new:
        items.append(dict(id=x['id'], ids=torch.from_numpy(x['ids'].astype(np.int64)), lang=x['lang'], wav=torch.from_numpy(x['wav']), split=x['split'], src=x['src']))
    print('total items', len(items), 'hours', round(sum(len(x['wav']) for x in items) / SR / 3600, 2), flush=True)

    # ---- mel + CTC forced alignment on GPU ----
    from sakhiya.v2_vocoder import mel_spec_24k
    from sakhiya.ctc_aligner import MelCTCAligner, batch_viterbi_ctc_durations
    dev = 'cuda'
    cc = torch.load('/work/tts/ck/v31_ctc_recognizer.pt', map_location=dev, weights_only=False)
    mean, std = float(cc.get('mean', -6.1761)), float(cc.get('std', 2.4932))
    al = MelCTCAligner().to(dev); al.load_state_dict(cc.get('model', cc)); al.eval()
    order = sorted(range(len(items)), key=lambda i: len(items[i]['wav']))
    kept = []; confs = []
    with torch.no_grad():
        for b0 in range(0, len(order), 48):
            idx = order[b0:b0 + 48]; xs = [items[i] for i in idx]
            T = max(len(x['wav']) // HOP for x in xs); L = max(len(x['ids']) for x in xs)
            mel = torch.zeros(len(xs), 100, T, device=dev); ids = torch.zeros(len(xs), L, dtype=torch.long, device=dev)
            yl = torch.tensor([len(x['wav']) // HOP for x in xs], device=dev); il = torch.tensor([len(x['ids']) for x in xs], device=dev)
            raw = []
            for k, x in enumerate(xs):
                m = mel_spec_24k(x['wav'].to(dev).float() / 32767.0, dev)[0]
                mel[k, :, :m.shape[1]] = (m - mean) / std; raw.append(m.half().cpu()); ids[k, :len(x['ids'])] = x['ids'].to(dev)
            ym = torch.arange(T, device=dev)[None] < yl[:, None]
            lp = al(mel, ym)
            dur, conf = batch_viterbi_ctc_durations(lp, ids, yl, il)
            for k, x in enumerate(xs):
                cf = float(conf[k].item()); confs.append(cf)
                Tk = raw[k].shape[1]; d = dur[k, :len(x['ids'])].cpu().long()
                if cf > a.conf and int(d.sum()) == Tk:
                    x['mel'] = raw[k]; x['dur'] = d.to(torch.int16); x['conf'] = round(cf, 3); kept.append(x)
            if (b0 // 48) % 40 == 0: print(f'aligned {b0}/{len(order)} kept {len(kept)}', flush=True)
    c = np.array(confs)
    print('conf pct 5/25/50/95', np.percentile(c, [5, 25, 50, 95]).round(2).tolist(), flush=True)
    by = {}
    for x in kept: by[x['src']] = by.get(x['src'], 0) + len(x['wav']) / SR / 3600
    rep = dict(items=len(kept), hours=round(sum(by.values()), 2), by_src={k: round(v, 2) for k, v in by.items()}, mean=mean, std=std)
    print(json.dumps(rep), flush=True)
    torch.save(dict(items=kept, mean=mean, std=std, hours=rep['hours']), a.out + '.tmp'); os.replace(a.out + '.tmp', a.out)
    json.dump(rep, open(os.path.dirname(a.out) + '/gt_report.json', 'w'))
    print('PREP_GT_DONE', flush=True)


if __name__ == '__main__':
    main()
