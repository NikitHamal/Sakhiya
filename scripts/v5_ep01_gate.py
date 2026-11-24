#!/usr/bin/env python3
"""Gate all 33 mastered EP01 lines (+ integrated LUFS of the full voiceover). usage: v5_ep01_gate.py <ep01_dir> <anchor_json> <tag>"""
import json, os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import leak_loop_gate as G
d, aj, tag = sys.argv[1:4]
OUT = os.environ.get("V5_STATE", "/work/tts/v5_recovery/state"); os.makedirs(OUT, exist_ok=True)
tl = json.load(open(f"{d}/timeline.json")); ref = json.load(open(aj))["ref_text"]
paths = [f"{d}/lines/{t['id']}.wav" for t in tl]
hyps = G._asr_batch(paths)
rows = []
for t, p in zip(tl, paths):
    w, sr = G.load_wav(p)
    r = G.check_clip(w, sr, t["text"], ref, hyps.get(p), "nepali")
    rows.append(dict(id=t["id"], ok=r["ok"], rejects=r["rejects"], clicks=r["clicks"]))
    if not r["ok"]: print("REJECT", t["id"], r["rejects"])
ff = subprocess.run(["ffmpeg", "-nostats", "-i", f"{d}/voiceover_ep01.wav", "-af", "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True).stderr
lufs = re.findall(r"I:\s+(-?[\d.]+) LUFS", ff)[-1] if "LUFS" in ff else None
rep = json.load(open(f"{d}/take_report.json"))
res = dict(tag=tag, lines=len(rows), gate_pass=sum(r["ok"] for r in rows), renderer_gate=sum(v["gate_pass"] for v in rep.values()),
           total_clicks=sum(r["clicks"] for r in rows), lufs=lufs, rejected=[r for r in rows if not r["ok"]])
json.dump(res, open(f"{OUT}/ep01_{tag}.json", "w"), ensure_ascii=False, indent=1)
print("EP01", json.dumps(res, ensure_ascii=False)[:500])
