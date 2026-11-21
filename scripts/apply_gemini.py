#!/usr/bin/env python3
"""Merge Gemini transcripts (asr/gemini.jsonl) into the training inputs.  Run AFTER apply_corrections.py (owner fixes always win).
 gold_text.json            : id -> best words  (owner fix > Gemini > nothing).  Written in place, owner-fix entries untouched.
 asr/gemini_vedic.json     : {"ok": [paths Gemini heard as real Sanskrit, Devanagari], "bad": {path: language heard}}  -> Vedic purity gate
 blocklist_gemini.json     : Vedic ids Gemini heard as Tamil/Telugu/Malayalam/Kannada/English/none (never train on them)
usage: python scripts/apply_gemini.py [--root /work/tts]"""
import argparse, collections, json, os, time
BADLANG = {"tamil", "telugu", "malayalam", "kannada", "english", "other", "none", ""}
def deva(t):
    a = [c for c in t if c.isalpha()]; return sum("\u0900" <= c <= "\u097f" for c in a) / max(1, len(a))
def load(p, d):
    try: return json.load(open(p))
    except Exception: return d
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=os.environ.get("TTS_ROOT", "/work/tts")); R = ap.parse_args().root
    rows = {}
    for l in open(f"{R}/asr/gemini.jsonl"):
        try: r = json.loads(l); rows[r["id"]] = r
        except Exception: pass
    fx = load(f"{R}/corrections.json", {}); owner = {k for k, v in fx.items() if v.get("text")}
    gold = load(f"{R}/gold_text.json", {}); ok, bad, n = [], {}, 0
    for cid, r in rows.items():
        t = r["text"]
        if r["style"] == "vedic_chant":
            if r["lang"] in BADLANG or (t and deva(t) < 0.8) or not t: bad[r["path"]] = r["lang"] or "none"
            else: ok.append(r["path"])
            if r["path"] in bad: continue
        if t and cid not in owner and deva(t) > 0.8: gold[cid] = t; n += 1
    json.dump(gold, open(f"{R}/gold_text.json", "w"), ensure_ascii=False, indent=1)
    json.dump({"ok": ok, "bad": bad, "ts": time.time()}, open(f"{R}/asr/gemini_vedic.json", "w"), ensure_ascii=False)
    json.dump(sorted(os.path.basename(p)[:-4] for p in bad), open(f"{R}/blocklist_gemini.json", "w"))
    print(f"gemini rows {len(rows)} · gold words +{n} (owner fixes kept: {len(owner)}) · Vedic real Sanskrit {len(ok)} · Vedic rejected {len(bad)} {dict(collections.Counter(bad.values()).most_common(6))}", flush=True)
if __name__ == "__main__": main()
