#!/usr/bin/env python3
"""EXP9 (Sonnet S4): objective probe of Gemini's in-training Sakhiya-Voca sample renders vs REAL Rasa-F numbers
(sonnet/results/exp1_real_vs_shipped.json).  Reuses exp1's `acoustics()`.  CPU only.
usage: python exp9_s4_ckpt_probe.py SAMPLES_DIR OUT_JSON [REPO]"""
import sys, os, re, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, librosa
SD, OUT = sys.argv[1], sys.argv[2]
REPO = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(__file__), "..", "..")
SR = 24000
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp1_real_vs_shipped.py")).read()
a, b = src.index("import pyworld as pw"), src.index("def agg(")
exec(src[a:b])           # defines acoustics()
real = json.load(open(os.path.join(REPO, "sonnet/results/exp1_real_vs_shipped.json")))["REAL_F"]["acoustics"]
import glob
# SD = checkpoints dir (e.g. /work/tts/s4/checkpoints); we scan per-step dirs because the Progress.sample index overwrote F with M (same-second filename)
rows = []
for f in sorted(glob.glob(os.path.join(SD, "*", "samples", "step_*", "*_step*_*.wav"))):
    m = re.search(r"/(\w+)/samples/step_(\d+)/.*_(F_sakhiya|M_sagar)\.wav$", f)
    if m: rows.append({"job": m.group(1), "file": f, "label": f"step {int(m.group(2)):,} · {m.group(3)} · ''"})
res = {}
for r in rows:
    w, sr = sf.read(r["file"], dtype="float32")
    if w.ndim > 1: w = w.mean(1)
    if sr != SR: w = librosa.resample(w, orig_sr=sr, target_sr=SR)
    m = re.search(r"step ([\d,]+) · (\w+)", r["label"]); step = int(m.group(1).replace(",", "")); spk = m.group(2)
    A = acoustics(w, r["label"].split("'")[1] if "'" in r["label"] else "")
    A.pop("graphemes_per_s", None)
    res.setdefault(f'{r["job"]}|{spk}', {})[step] = A
try:
    import torch
    ut = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    for r in rows:
        w, sr = sf.read(r["file"], dtype="float32")
        if w.ndim > 1: w = w.mean(1)
        w16 = librosa.resample(w, orig_sr=sr, target_sr=16000)
        m = re.search(r"step ([\d,]+) · (\w+)", r["label"]); step = int(m.group(1).replace(",", ""))
        res[f'{r["job"]}|{m.group(2)}'][step]["utmos"] = float(ut(torch.from_numpy(w16)[None], 16000).item())
except Exception as e:
    print("utmos skipped:", repr(e)[:150])
json.dump({"real_F_ref": {k: real[k]["mean"] for k in real}, "probes": res}, open(OUT, "w"), indent=1, default=float)
keys = ["f0_std_st", "f0_phrase_std_st", "voiced_frac", "n_pauses", "hnr_db", "jitter_local", "tilt_4_8k_minus_0_2k_db", "utmos"]
print("REAL_F", {k: round(real[k]["mean"], 3) for k in keys if k in real})
for k, v in res.items():
    s = max(v); print(k, "last step", s, {kk: round(v[s][kk], 3) for kk in keys if kk in v[s] and v[s][kk] is not None})
