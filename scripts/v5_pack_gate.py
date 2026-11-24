#!/usr/bin/env python3
"""Run the Leak/Loop/Splice gate + accent selector thresholds on a rendered Nepali pack (pack_meta.json + mp3s).
usage: v5_pack_gate.py <pack_dir> <anchor_json> <tag>   -> state/nepali_<tag>.json
Promotion thresholds (owner spec): end_slope <= -0.55 st, rise_frac <= 0.14, clicks <= 2, CER <= 0.06, UTMOS >= 3.90, zero gate rejects."""
import json, os, statistics, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import leak_loop_gate as G

pack, anchor_json, tag = sys.argv[1:4]
OUT = os.environ.get("V5_STATE", "/work/tts/v5_recovery/state"); os.makedirs(OUT, exist_ok=True)
meta = json.load(open(f"{pack}/pack_meta.json"))
ref_text = json.load(open(anchor_json))["ref_text"]
paths = [f"{pack}/{m['mp3']}" for m in meta]
try:
    hyps = G._asr_batch(paths)
except Exception as e:
    print("ASR unavailable:", repr(e)[:160]); hyps = {}
rows = []
for m, p in zip(meta, paths):
    w, sr = G.load_wav(p)
    g = G.check_clip(w, sr, m["text"], ref_text, hyps.get(p), "nepali")
    t = m["take"]; c = next((x for x in t["cands"] if x["seed"] == t["chosen"]), t["cands"][0])
    hard = []
    if c["end_slope"] > -0.55: hard.append("end_slope")
    if c["rise_frac"] > 0.14: hard.append("rise")
    if c["cer"] > 0.06: hard.append("cer")
    if c["utmos"] < 3.90: hard.append("utmos")
    rows.append(dict(id=m["id"], gate_ok=g["ok"], rejects=g["rejects"], metrics_fail=hard, hyp=g["hyp"], **{k: round(c[k], 3) for k in ("end_slope", "rise_frac", "cer", "utmos", "hf_karkar", "spk_sim")}, clicks=c["clicks"]))
    print(("PASS " if g["ok"] else "REJECT "), m["id"], g["rejects"], "metric-fails:", hard)
agg = {k: round(statistics.mean(r[k] for r in rows), 3) for k in ("end_slope", "rise_frac", "cer", "utmos", "hf_karkar", "spk_sim")}
agg.update(clips=len(rows), gate_pass=sum(r["gate_ok"] for r in rows), metric_pass=sum(not r["metrics_fail"] for r in rows),
           all_pass=sum(r["gate_ok"] and not r["metrics_fail"] for r in rows))
json.dump(dict(tag=tag, agg=agg, rows=rows), open(f"{OUT}/nepali_{tag}.json", "w"), ensure_ascii=False, indent=1)
print("AGG", json.dumps(agg))
