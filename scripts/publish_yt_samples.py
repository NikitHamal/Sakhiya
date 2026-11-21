#!/usr/bin/env python3
"""Publish 20 real mined (+ their sakhiya-VC renderings when available) previews to docs/samples/youtube_mined/ (mp3 + index.json) and push.
Run on the GPU box (has git credentials via /tmp/gitenv.sh).  usage: publish_yt_samples.py [--n 20] [--no-push]"""
import argparse, glob, io, json, os, subprocess, sys
import pyarrow.parquet as pq
ROOT = os.environ.get("TTS_ROOT", "/work/tts"); REPO = f"{ROOT}/repo"; OUT = f"{REPO}/docs/samples/youtube_mined"

def mp3(wav_bytes, path):
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", "pipe:0", "-ac", "1", "-b:a", "72k", path], input=wav_bytes, check=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=20); ap.add_argument("--no-push", action="store_true"); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); raw = {}
    for p in glob.glob(f"{ROOT}/dataset/raw/*.parquet"):
        for r in pq.read_table(p).to_pylist(): raw[r["id"]] = r
    vc = {}
    for p in glob.glob(f"{ROOT}/dataset/vc/*.parquet"):
        for r in pq.read_table(p).to_pylist(): vc[r["id"]] = r
    pool = [raw[i] for i in vc if i in raw] or list(raw.values())
    per = max(1, a.n // 4); pick = []; styles = ["storytelling", "vedic_chant", "news", "conversational"]
    for s in styles:   # prefer clips with expressive punctuation (? !) and high UTMOS
        c = sorted([r for r in pool if r["style"] == s], key=lambda r: (-(("?" in r["text"]) + ("!" in r["text"])), -r["utmos"]))[:per]; pick += c
    pick = (pick + sorted(pool, key=lambda r: -r["utmos"]))[:a.n] if len(pick) < a.n else pick[:a.n]; idx = []
    for k, r in enumerate(pick):
        e = {"id": r["id"], "style": r["style"], "text": r["text"], "utmos": r["utmos"], "tier": r["tier"], "dur": r["dur"], "f0_range_st": r["f0_range_st"], "mined": f"{r['id']}_mined.mp3"}
        mp3(r["wav"], f"{OUT}/{r['id']}_mined.mp3")
        if r["id"] in vc: mp3(vc[r["id"]]["wav"], f"{OUT}/{r['id']}_sakhiya.mp3"); e.update(sakhiya=f"{r['id']}_sakhiya.mp3", f0_corr=vc[r["id"]]["f0_corr_vs_src"], sim=vc[r["id"]]["sim_sakhiya"], octave_shift=vc[r["id"]]["octave_shift"])
        idx.append(e)
    json.dump({"note": "Real mined expressive Nepali/Vedic clips (Internet Archive sources, vocals isolated) and their sakhiya voice-converted renderings.", "items": idx}, open(f"{OUT}/index.json", "w"), ensure_ascii=False, indent=1)
    print("published", len(idx), "clips;", sum(1 for e in idx if "sakhiya" in e), "with VC")
    if not a.no_push:
        env = dict(os.environ, GIT_ASKPASS="/tmp/gitenv.sh", GIT_TERMINAL_PROMPT="0")
        for c in ("git add docs/samples/youtube_mined", "git commit -qm 'docs: 20 mined + voice-converted YouTube-corpus previews'", "git pull -q --rebase origin main", "git push -q origin HEAD:main"):
            subprocess.run(c, shell=True, cwd=REPO, env=env)

if __name__ == "__main__": main()
