#!/usr/bin/env python3
"""Merge measured GGUF sizes + CPU RTFs (eval/ggml_proof/tier_report.json) into docs/tiers.json (ONNX fields untouched)."""
import json, os
R = {t["tier"]: t for t in json.load(open("eval/ggml_proof/tier_report.json"))["tiers"]}
p = "docs/tiers.json"; T = json.load(open(p))
for t in T["tiers"]:
    r = R.get(t["id"])
    if not r: continue
    t["onnx_mb"] = t["mb"]
    t["gguf_file"] = f"sakhiya_{t['id']}_q8_0.gguf"
    q = "q8_0_tiny" if (t["id"] == "atom_lite" and "q8_0_tiny" in r["sizes_mb"]) else "q8_0"   # Atom-Lite ships the tiny profile as sakhiya_atom_lite_q8_0.gguf
    t["gguf_mb"] = r["sizes_mb"][q]
    t["gguf_exact_mb"] = r["sizes_mb"]["q8_0_exact"] if t["id"] != "v1" else r["sizes_mb"]["q8_0"]
    t["gguf_f16_mb"] = r["sizes_mb"]["f16"]
    t["rtf_onnx"] = t["rtf"]
    t["rtf_ggml_1t"] = r["rtf"][q + "_t1"]
    t["rtf_ggml_mt"] = r["rtf"][q + "_t4"]
    t["ggml_min_cosine_exact"] = r["min_cosine"]["q8_0_exact" if t["id"] != "v1" else "q8_0"]
T["ggml_note"] = ("gguf_* / rtf_ggml_*: measured by tests/test_ggml_tiers.py + sakhiya-cli on a 4-vCPU sandbox (1 and 4 threads); "
                  "rtf / rtf_onnx come from the original ONNX forensic gate on different hardware, so compare within an engine.")
json.dump(T, open(p, "w"), indent=1, ensure_ascii=False)
print([(t["id"], t.get("gguf_mb"), t.get("rtf_ggml_1t"), t.get("rtf_ggml_mt")) for t in T["tiers"]])
