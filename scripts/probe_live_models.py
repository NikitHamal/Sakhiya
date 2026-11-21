#!/usr/bin/env python3
"""Probe every Live-API model the owner's key can open for Nepali / Sanskrit transcription (extra models = extra tokens-per-minute budget).
usage: python scripts/probe_live_models.py [model ...]   (key from GEMINI_API_KEY or /tmp/gemini_key, never committed)
Prints per model: ok/err, seconds, and the text it heard for 2 Nepali + 1 Vedic clip already transcribed by the main model, with similarity to that text."""
import asyncio, difflib, json, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import gemini_transcribe as G

CAND = ["gemini-2.5-flash-native-audio-latest", "gemini-2.5-flash-native-audio-preview-09-2025", "gemini-2.5-flash-native-audio-preview-12-2025",
        "gemini-3.8-live", "gemini-3.8-live-extended-thinking", "gemini-3.5-live-translate-preview"]

def pick():
    rows = G.jl(G.OUT); ne = [r for r in rows if r["style"] != "vedic_chant" and r["lang"] == "nepali" and len(r["text"]) > 20 and r["model"].endswith("transcribe-live")][:2]
    ve = [r for r in rows if r["style"] == "vedic_chant" and r["lang"] == "devanagari" and len(r["text"]) > 20][:1]
    return ne + ve

async def main():
    models = sys.argv[1:] or CAND; clips = pick()
    for m in models:
        for r in clips:
            t0 = time.time()
            try:
                txt = await asyncio.wait_for(G.live(G.pcm16(r["path"]), m, [] if r["style"] == "vedic_chant" else ["ne-NP"]), 90)
                sim = difflib.SequenceMatcher(None, txt, r["text"]).ratio()
                print(f"{m:55s} OK  {time.time()-t0:5.1f}s sim={sim:.2f} | {txt[:90]}", flush=True)
            except Exception as e:
                print(f"{m:55s} ERR {time.time()-t0:5.1f}s {repr(e)[:110]}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
