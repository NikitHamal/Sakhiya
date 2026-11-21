#!/usr/bin/env python3
"""Ingest prathoshap/vagdhenu-data (CC-BY-4.0 studio Vedic/Sanskrit chanting, reference Devanagari text) as the TRUSTED vedic source.
Writes 24 kHz mono clips into segments/vedic_chant/vag_*.wav (so the normal gate -> ASR -> pack chain picks them up),
a segments.vag.jsonl manifest, and vedic/ref_text.json (path -> reference text) used by yt_asr_ensemble.py as the transcript.
usage: python scripts/ingest_vagdhenu_gold.py [--min 2.0] [--max 16]"""
import argparse, csv, glob, json, os
import numpy as np, soundfile as sf, librosa
ROOT = os.environ.get("TTS_ROOT", "/work/tts"); D = f"{ROOT}/data/vag"; SEG = f"{ROOT}/segments/vedic_chant"; V = f"{ROOT}/vedic"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--min", type=float, default=2.0); ap.add_argument("--max", type=float, default=16.0); a = ap.parse_args()
    os.makedirs(SEG, exist_ok=True); os.makedirs(V, exist_ok=True); ref, rows, n = {}, [], 0
    for cfg in sorted(glob.glob(f"{D}/style_*")):
        mp = f"{cfg}/metadata.csv"
        if not os.path.exists(mp): continue
        for r in csv.DictReader(open(mp, encoding="utf8")):
            t = " ".join((r.get("text_devanagari") or "").split()); fn = r.get("file_name", "")
            src = fn if os.path.exists(fn) else f"{cfg}/{fn}"
            if len(t) < 6 or not os.path.exists(src): continue
            try: x, sr = sf.read(src, dtype="float32")
            except Exception: continue
            x = x.mean(1) if x.ndim > 1 else x; d = len(x) / sr
            if not (a.min <= d <= a.max): continue
            if sr != 24000: x = librosa.resample(x, orig_sr=sr, target_sr=24000)
            x = x / max(1e-3, float(np.abs(x).max())) * 0.9; p = f"{SEG}/vag_{os.path.basename(cfg)}_{n:05d}.wav"; sf.write(p, x, 24000)
            ref[p] = t; rows.append({"path": p, "cat": "vedic_chant", "src": os.path.basename(p).rsplit("_", 1)[0] + f"_{n // 400}", "t0": 0.0, "dur": round(d, 2), "speech": round(d * 0.9, 2)}); n += 1
    with open(f"{ROOT}/segments/segments.vag.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    old = json.load(open(f"{V}/ref_text.json")) if os.path.exists(f"{V}/ref_text.json") else {}
    old.update(ref); json.dump(old, open(f"{V}/ref_text.json", "w"), ensure_ascii=False)
    print("VAG_INGEST clips", n, "hours", round(sum(r["dur"] for r in rows) / 3600, 2), flush=True)

if __name__ == "__main__": main()
