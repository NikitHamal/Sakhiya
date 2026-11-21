#!/usr/bin/env python3
"""Stage 4b/5a: acoustic punctuation restoration + style tags + g2p verification + Parquet/WebDataset packaging.
Punctuation comes from the audio, not from the ASR text:  pauses between CTC-aligned words -> ',' / '।' ; final F0 rise -> '?' ;
F0+energy peak on an interjection (वाह ओहो अरे ...) -> '!' ; sustained harmonic F0 on the first word 'ओम' -> 'ॐ' ; chant pauses -> '।', end '॥'.
in: asr/asr.jsonl  out: dataset/raw/*.parquet (+ .tar webdataset) + dataset/meta.jsonl"""
import argparse, io, json, os, re, sys, tarfile, time, glob
from multiprocessing import Pool
import numpy as np, soundfile as sf, pyarrow as pa, pyarrow.parquet as pq

ROOT = os.environ.get("TTS_ROOT", "/work/tts"); sys.path.insert(0, os.environ.get("REPO", f"{ROOT}/repo"))
INTERJ = {"वाह", "ओहो", "अरे", "हाय", "उफ", "ऐ", "ए", "छि", "ओह", "बाबा", "अहा", "आहा", "हे", "शाबास", "धन्य", "हत्तेरी", "ल", "अहो", "हैन"}
QWORDS = {"के", "किन", "कहाँ", "कहां", "कसरी", "कति", "को", "कुन", "कहिले", "कसले", "कसको", "कसलाई", "कता", "कस्तो"}
VEND = re.compile(r"(छ|छन्|छौं|छु|थियो|थिए|थिन्|हो|हुन्|भयो|भए|गर्यो|गरे|गरिन्|हुन्छ|गर्छ|दिए|लिए|आयो|गयो)$")
STYLE = {"storytelling": "storytelling", "news": "news", "conversational": "conversational", "vedic_chant": "vedic_chant"}
def st(a, b): return 12 * np.log2(max(a, 1.0) / max(b, 1.0))

def f0_energy(w, sr=24000):
    import pyworld as pw
    x = w.astype(np.float64); f0, t = pw.dio(x, sr, f0_floor=60, f0_ceil=520, frame_period=10.0); f0 = pw.stonemask(x, f0, t, sr)
    hop = sr // 100; n = len(f0); fr = np.array([x[i * hop:(i + 1) * hop + hop] for i in range(n - 1)] + [x[(n - 1) * hop:]], dtype=object)
    e = np.array([20 * np.log10(np.sqrt(np.mean(c ** 2)) + 1e-6) if len(c) else -120 for c in fr]); return f0, e

def punctuate(words, f0, e, cat):
    """words: [{w,t0,t1}] ; returns text, per-word tags"""
    v = f0 > 0; med = np.median(f0[v]) if v.any() else 150.0; out = []; fl = lambda t: min(int(t * 100), len(f0) - 1)
    for i, wd in enumerate(words):
        w = wd["w"]; a, b = fl(wd["t0"]), max(fl(wd["t1"]), fl(wd["t0"]) + 1); seg = f0[a:b + 1]; sv = seg[seg > 0]; last = i == len(words) - 1
        if i == 0 and cat == "vedic_chant" and w in ("ओम", "ओं") and len(sv) > 25 and np.std(12 * np.log2(sv / np.median(sv))) < 1.3: w = "ॐ"
        mark = ""; base = w.rstrip("।")
        if base in INTERJ and len(sv) > 3 and st(sv.max(), med) > 5 and e[a:b + 1].max() > np.mean(e[e > -80]) + 4: mark = "!"
        if not last:
            gap = words[i + 1]["t0"] - wd["t1"]
            if cat == "vedic_chant": mark = mark or ("।" if gap >= 0.5 else "")
            elif not mark and gap >= 0.45 and VEND.search(base): mark = "।"
            elif not mark and gap >= 0.22: mark = ","
        else:
            tail = f0[fl(max(wd["t1"] - 0.35, wd["t0"])):fl(wd["t1"]) + 1]; tail = tail[tail > 0]; pre = f0[fl(max(wd["t0"] - 0.6, 0)):fl(max(wd["t1"] - 0.35, wd["t0"]))]; pre = pre[pre > 0]
            rise = st(np.median(tail[-max(3, len(tail) // 3):]), np.median(pre)) if len(tail) > 5 and len(pre) > 5 else 0.0
            if cat == "vedic_chant": mark = "॥"
            elif rise >= 3.5 or (rise >= 2.0 and any(q in {x["w"] for x in words} for q in QWORDS)): mark = "?"
            elif base in INTERJ or (len(sv) > 3 and st(sv.max(), med) > 6.5 and e[a:b + 1].max() > np.mean(e[e > -80]) + 6): mark = "!"
            elif not mark: mark = "।"
        out.append(w + mark)
    return " ".join(out)

def proc(r):
    try:
        from sakhiya import g2p
        from sakhiya.symbols import SYM2ID
        w, sr = sf.read(r["path"], dtype="float32"); f0, e = f0_energy(w); words = r["words"]; cat = r["cat"]; lang = "sa" if cat == "vedic_chant" else "ne"
        text = punctuate(words, f0, e, cat); ph = g2p.text_to_phones(text, lang)
        ids = [SYM2ID[p] for p in ph if p in SYM2ID]
        if not ids or len(ids) < 0.6 * len(re.sub(r"[^\u0900-\u097F]", "", text)): return None
        # phoneme durations (10 ms frames): word spans exact from CTC, split inside a word by per-phone weight; gaps become pause frames
        durs = []; p_words = [g2p.text_to_phones(x["w"], lang, add_end=False) for x in words]
        for x, pw_ in zip(words, p_words):
            n = max(1, len([p for p in pw_ if p in SYM2ID])); tot = max(n, round((x["t1"] - x["t0"]) * 100)); durs += [tot // n + (1 if k < tot % n else 0) for k in range(n)]
        v = f0 > 0; lf = np.where(v, np.log(np.maximum(f0, 1)), np.nan); idx = np.arange(len(lf))
        if v.sum() > 3: lf = np.interp(idx, idx[v], lf[v[:]]) 
        else: return None
        sv = f0[v]; rng = float(np.percentile(12 * np.log2(sv / np.median(sv)), 95) - np.percentile(12 * np.log2(sv / np.median(sv)), 5))
        return {"id": os.path.basename(r["path"])[:-4], "text": text, "text_raw": r["text_w"], "style": STYLE[cat], "lang": lang, "phone_ids": ids, "phone_durs_10ms": durs,
                "f0_logf0_10ms": lf.astype(np.float16), "voiced": v.astype(np.uint8), "utmos": r["utmos"], "snr": r["snr"], "spk": r["spk"], "src": r["src"], "dur": r["dur"],
                "agree": r["agree"], "conf_w": r["conf_w"], "f0_range_st": round(rng, 2), "tier": r["tier"], "wav": open(r["path"], "rb").read()}
    except Exception as ex: return None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--agree", type=float, default=0.94); ap.add_argument("--agree-silver", type=float, default=0.85)
    ap.add_argument("--conf", type=float, default=0.55); ap.add_argument("--utmos-gold", type=float, default=3.4); ap.add_argument("--shard-size", type=int, default=1500)
    ap.add_argument("--out", default=f"{ROOT}/dataset/raw"); ap.add_argument("--workers", type=int, default=18); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); sp = json.load(open(f"{ROOT}/gate/speakers.json")); rows = []
    for l in open(f"{ROOT}/asr/asr.jsonl"):
        r = json.loads(l)
        tw = r["text_w"]
        if re.search(r"[^\u0900-\u097F\s.,!?|।॥\-–\"\'0-9ॐ]", tw) or re.search(r"(\S+)( \1){2,}", tw) or len(r["words"]) < 3: continue   # foreign-script / repetition-loop hallucinations
        if not r["words"] or r["min_word_frames"] < 2 or (r["conf_w"] < a.conf and not r.get("ref")) or r["align_lp"] < (-3.5 if r.get("ref") else -2.2): continue
        gold = (r["agree"] >= a.agree or r.get("ref")) and r["utmos"] >= a.utmos_gold; silver = r["agree"] >= a.agree_silver or bool(r.get("ref"))
        if not silver: continue
        r["spk"] = sp.get(r["path"], "unk"); r["tier"] = "gold" if gold else "silver"; rows.append(r)
    mp = f"{ROOT}/dataset/meta.jsonl"; os.makedirs(os.path.dirname(mp), exist_ok=True)
    seen = {json.loads(l)["id"] for l in open(mp)} if os.path.exists(mp) else set(); rows = [r for r in rows if os.path.basename(r["path"])[:-4] not in seen]
    print(len(rows), "new clips pass ASR/alignment gates", flush=True); n = 0; k = int(time.time()); buf = []; meta = open(mp, "a")
    def flush():
        nonlocal buf, k
        if not buf: return
        tb = pa.Table.from_pylist([{**b, "f0_logf0_10ms": b["f0_logf0_10ms"].astype(np.float32).tolist()} for b in buf]); pq.write_table(tb, f"{a.out}/train-{k:05d}.parquet", compression="zstd")
        with tarfile.open(f"{a.out}/train-{k:05d}.tar", "w") as tf:
            for b in buf:
                for ext, data in (("wav", b["wav"]), ("json", json.dumps({x: (y.tolist() if hasattr(y, "tolist") else y) for x, y in b.items() if x != "wav"}, ensure_ascii=False).encode())):
                    ti = tarfile.TarInfo(f"{b['id']}.{ext}"); ti.size = len(data); tf.addfile(ti, io.BytesIO(data))
        k += 1; buf = []
    with Pool(a.workers) as p:
        for o in p.imap(proc, rows, chunksize=8):
            if o is None: continue
            n += 1; buf.append(o); meta.write(json.dumps({x: y for x, y in o.items() if x not in ("wav", "f0_logf0_10ms", "voiced", "phone_ids", "phone_durs_10ms")}, ensure_ascii=False) + "\n")
            if len(buf) >= a.shard_size: flush()
    flush(); meta.close(); print(f"packed {n} clips into {k} shards -> {a.out}", flush=True)

if __name__ == "__main__": main()
