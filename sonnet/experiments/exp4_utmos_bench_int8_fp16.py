#!/usr/bin/env python3
"""EXP4 (Sonnet lane): robustness re-check of the exp3 UTMOS finding on the repo's OWN held-out bench sentences
(eval/bench_general_200.json, ne) and with the INT8 graph the browser actually serves.   usage: exp4 REPO OUT_JSON"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, librosa, torch
REPO, OUT = sys.argv[1], sys.argv[2]; sys.path.insert(0, REPO)
from sakhiya.onnx_tts import SakhiyaONNX
mos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).eval()
S = [x["text"] for x in json.load(open(os.path.join(REPO, "eval/bench_general_200.json")))["sentences"] if x["lang"] == "ne"][:40]
res = {}
for name, d, tiny, st in [("VOCOS_INT8_news", "docs/models_sakhiya_one_vocos", True, "news"), ("VOCOS_FP16_news", "docs/models_sakhiya_one_vocos", False, "news"),
                          ("VOCOS_FP16_neutral", "docs/models_sakhiya_one_vocos", False, "neutral"),
                          ("V1_INT8_news", "docs/models_v2", True, "news"), ("ONE_COMPACT_news", "docs/models_sakhiya_one", True, "news"),
                          ("NANO_news", "docs/models_voca", True, "news")]:
    try:
        m = SakhiyaONNX(os.path.join(REPO, d), tiny=tiny, threads=4); sc = []; dur = []
        for t in S:
            w, sr = m.say(t, speaker="sakhiya", lang="ne", style=st); w = np.asarray(w, np.float32).reshape(-1)
            x = librosa.resample(w, orig_sr=sr, target_sr=16000); dur.append(len(w) / sr)
            with torch.no_grad(): sc.append(float(mos(torch.from_numpy(x)[None], 16000)))
        res[name] = {"utmos_mean": float(np.mean(sc)), "utmos_std": float(np.std(sc)), "utmos_min": float(np.min(sc)), "mean_dur_s": float(np.mean(dur)), "n": len(sc)}
    except Exception as e:
        res[name] = {"error": repr(e)[:200]}
    print(name, res[name], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
