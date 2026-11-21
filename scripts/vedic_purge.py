#!/usr/bin/env python3
"""Remove every vedic_chant row that did not pass the purity gate from gate/passed.jsonl, asr/asr.jsonl and dataset/meta.jsonl
(backups kept as *.prepurge.bak). Rows from trusted reference sources (vedic/ref_text.json) are always kept.
usage: python scripts/vedic_purge.py   (run after scripts/vedic_purity_gate.py)"""
import json, os, shutil
ROOT = os.environ.get("TTS_ROOT", "/work/tts"); V = f"{ROOT}/vedic"
ok = json.load(open(f"{V}/ok_paths.json")); good = set(ok["ok_paths"]); gsrc = set(ok["good_sources"])
ref = set(json.load(open(f"{V}/ref_text.json"))) if os.path.exists(f"{V}/ref_text.json") else set()

def keep(path, cat):
    if cat != "vedic_chant": return True
    return path in ref or path in good or os.path.basename(path).rsplit("_", 1)[0] in gsrc   # unjudged clips of a verified-pure source are kept

def purge(p, pathkey, catkey, idmode=False):
    if not os.path.exists(p): return 0, 0
    rows = [json.loads(l) for l in open(p)]; out = []
    for r in rows:
        path = r.get(pathkey) or ""
        if idmode:   # dataset meta rows only carry an id; find the cleaned wav under segments/vedic_chant
            path = f"{ROOT}/segments/vedic_chant/{r.get('id', '')}.wav"
            if r.get(catkey) != "vedic_chant": out.append(r); continue
            cat = "vedic_chant"
        else: cat = r.get(catkey)
        if keep(path, cat): out.append(r)
    if len(out) != len(rows):
        if not os.path.exists(p + ".prepurge.bak"): shutil.copy(p, p + ".prepurge.bak")
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, p)
    return len(rows), len(out)

if __name__ == "__main__":
    for name, args in (("gate/passed", (f"{ROOT}/gate/passed.jsonl", "path", "cat")), ("asr", (f"{ROOT}/asr/asr.jsonl", "path", "cat")), ("meta", (f"{ROOT}/dataset/meta.jsonl", "id", "style", True))):
        a, b = purge(*args); print(f"{name}: {a} -> {b} rows", flush=True)
