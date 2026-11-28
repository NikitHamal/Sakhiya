#!/usr/bin/env python3
"""Aggregate EXP6 jsonl -> first-consonant label distribution per (corpus, word class) + bias-corrected b-rate."""
import json, glob, sys, collections
FAM = {"b": "b", "β": "b", "bʱ": "b", "bʰ": "b", "v": "v/w", "ʋ": "v/w", "w": "v/w", "ʊ": "v/w", "ɸ": "v/w", "f": "v/w",
       "ts": "ts", "ʦ": "ts", "tʃ": "tʃ/c", "c": "tʃ/c", "tɕ": "tʃ/c", "ʃ": "ʃ/s", "s": "ʃ/s", "dz": "dz", "dʒ": "dʒ/ɟ", "ɟ": "dʒ/ɟ", "g": "g", "ɡ": "g", "j": "j"}
out = {}; D = sys.argv[1] if len(sys.argv) > 1 else "/work/sonnet/s2/out"
for f in sorted(glob.glob(f"{D}/exp6_*.jsonl")):
    for l in open(f, encoding="utf8"):
        r = json.loads(l); k = (r["corp"], r["cls"]); fam = FAM.get(r["first"], "other:" + str(r["first"]))
        out.setdefault(k, collections.Counter())[fam] += 1
res = {}
for (c, k), cnt in sorted(out.items()):
    n = sum(cnt.values()); res[f"{c}|{k}"] = {"n": n, **{a: round(b / n, 3) for a, b in cnt.most_common(6)}}
json.dump(res, open(f"{D}/exp6_summary.json", "w"), ensure_ascii=False, indent=1)
for k, v in res.items(): print(k, v)
# GY: does the decoded word contain g (Nepali gj) or dʒ/ɲ (Sanskritic dzɲ)?
gy = collections.defaultdict(lambda: [0, 0, 0])
for f in glob.glob(f"{D}/exp6_*.jsonl"):
    for l in open(f, encoding="utf8"):
        r = json.loads(l)
        if r["cls"] != "GY": continue
        s = "".join(r["ph"]); g = gy[r["corp"]]; g[2] += 1
        if "g" in s or "ɡ" in s: g[0] += 1
        if "dʒ" in s or "ɟ" in s or "ɲ" in s or "ɳ" in s: g[1] += 1
print("GY (n_with_g, n_with_dʒ/ɲ, n):", dict(gy))
