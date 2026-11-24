#!/usr/bin/env python3
"""Aggregate Vedic v5 results into /work/tts/v5_recovery/state/vedic.json (rendered by the Marimo 'Isolated Vedic' card)."""
import glob, json, os
V = "/work/tts/v5_vedic"; S = "/work/tts/v5_recovery/state"
def J(p):
    try: return json.load(open(p))
    except Exception: return None
out = {"status": "Vedic isolation in progress - Nepali weights (v3b step-600) FROZEN; adapter loaded only for style=vedic_chant",
       "banned": ["ओऽऽऽम् stretch spelling", "hard-sliced anchors with full-verse ref_text", "Seed-VC converted anchors", "shared anchor for Shanti+Gayatri"]}
pr = J(f"{V}/prep_report.json")
if pr: out["lora_dataset"] = {"kept_clips": pr["kept"], "hours": pr["hours"], "by_source": pr["by_src"], "asr_match_min": pr["min_match"], "dropped": pr["dropped"]}
tr = J(f"{V}/exp/vl1/train_report.json")
if tr: out["lora_train"] = tr; out["nepali_untouched"] = bool(tr.get("base_frozen"))
else:
    ck = sorted(glob.glob(f"{V}/exp/vl1/adapter-*"))
    out["lora_train"] = {"status": "training or not finished", "latest_adapter": os.path.basename(ck[-1]) if ck else None}
ev = {}
for f in sorted(glob.glob(f"{V}/eval/*/summary.json")):
    s = J(f)
    if s: ev[s["tag"]] = {"gate_passed": f'{s["passed"]}/{s["n"]}', "per_chant_pass": s["per_chant_pass"], "chants_with_clean_take": s["chants_with_a_clean_take"]}
out["leak_loop_gate_evals"] = ev
json.dump(out, open(f"{S}/vedic.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(out, ensure_ascii=False)[:1500])
