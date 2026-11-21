#!/usr/bin/env python3
"""Keep the website selector to exactly three entries: Sakhiya-V1, Sakhiya-V1-Nano, Sakhiya-V1-Atom (latest shipped of each).
Older sample packs stay on disk in the repo; they are just not listed. Also appends a measured STATUS entry."""
import json, os
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
p = f"{REPO}/docs/samples/versions/registry.json"; r = json.load(open(p)); by = {v["tag"]: v for v in r["versions"]}
keep = [by["v5_final"]]
for new, old in (("v9_nano", "v8_nano"), ("v9_atom", "v8_atom")):
    t = by.get(new) or by.get(old)
    if t: keep.append(t)
for v in keep:
    if v["tag"] == "v9_nano": v["label"] = "Sakhiya-V1-Nano"
    if v["tag"] == "v9_atom": v["label"] = "Sakhiya-V1-Atom"
r["versions"] = keep; json.dump(r, open(p, "w"), ensure_ascii=False, indent=1)
lines = ["## Session 64 — v9: cached-teacher distillation of Nano/Atom (measured, auto-gated)",
         "- Teacher targets are cached in GPU memory (`scripts/v9_am_cached.py`): ~7 steps/s at batch 96 (v8: 2.5 steps/s at batch 48), hard 100-minute cap, step-50 gate.",
         "- Shipped only if the real forensic gate beat the shipped v8 (`scripts/v9_ship.py`); selector lists V1 / Nano / Atom only (`scripts/v9_registry.py`)."]
for tag, nm in (("v9_nano", "Nano"), ("v9_atom", "Atom"), ("v8_nano", "Nano v8"), ("v8_atom", "Atom v8")):
    f = f"{REPO}/eval/forensic_{tag}.json"
    if os.path.exists(f):
        g = json.load(open(f)); lines.append(f"- {nm} ({tag}): UTMOS {g.get('utmos_15')}, word drops {g.get('word_drop_count')}, stutters {g.get('stutter_count')}, clicks {g.get('clicks')}")
s = open(f"{REPO}/STATUS.md", encoding="utf8").read()
if "## Session 64" not in s:
    i = s.index("## Session 63"); open(f"{REPO}/STATUS.md", "w", encoding="utf8").write(s[:i] + "\n".join(lines) + "\n\n" + s[i:])
