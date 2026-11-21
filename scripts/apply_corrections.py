#!/usr/bin/env python3
"""Turn the owner's audit fixes (/work/tts/corrections.json) into training inputs.
 - gold_text.json : clip id -> corrected words (verdict ok/fix).  Trainers and the dataset packer must prefer these over any ASR guess.
 - blocklist.json : clip ids marked garbage / wrong language.  They are excluded from training, the Clip Library and the dataset.
 - ref_text.json  : Vedic reference text is patched in place (backup kept) so the verified-Sanskrit card shows the owner's words.
usage: python scripts/apply_corrections.py [--root /work/tts]"""
import argparse, json, os, shutil, time
def load(p, d):
    try: return json.load(open(p))
    except Exception: return d
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=os.environ.get("TTS_ROOT", "/work/tts")); a = ap.parse_args(); R = a.root
    fx = load(f"{R}/corrections.json", {}); gold = {k: v["text"] for k, v in fx.items() if v.get("verdict") in ("ok", "fix") and v.get("text")}
    bad = sorted(k for k, v in fx.items() if v.get("verdict") == "bad")
    json.dump(gold, open(f"{R}/gold_text.json", "w"), ensure_ascii=False, indent=1); json.dump(bad, open(f"{R}/blocklist.json", "w"), indent=1)
    rp = f"{R}/vedic/ref_text.json"; ref = load(rp, {}); n = 0
    if ref:
        if not os.path.exists(rp + ".owner.bak"): shutil.copy(rp, rp + ".owner.bak")
        for p in list(ref):
            cid = os.path.basename(p).rsplit(".", 1)[0]
            if cid in gold and ref[p] != gold[cid]: ref[p] = gold[cid]; n += 1
            if cid in bad: ref.pop(p, None)
        json.dump(ref, open(rp, "w"), ensure_ascii=False)
    print(f"gold {len(gold)} · garbage {len(bad)} · vedic ref patched {n} · {time.strftime('%F %T')}", flush=True)
if __name__ == "__main__": main()
