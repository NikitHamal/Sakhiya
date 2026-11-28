#!/usr/bin/env python3
"""EXP3 (Sonnet lane): (a) inventory of the whole ai4bharat/Rasa Nepali release by gender/style/duration, WITHOUT
downloading audio (parquet column projection over HfFileSystem); (b) UTMOS22 on real Rasa female vs shipped models
on the same sentences - to see what the proxy rewards.   usage: exp3 REPO OUT_JSON"""
import os, sys, json, io, time, warnings, collections
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, librosa, torch
REPO, OUT = sys.argv[1], sys.argv[2]; sys.path.insert(0, REPO); SR = 24000; T0 = time.time()
def log(*a): print(f"[{time.time()-T0:5.0f}s]", *a, flush=True)
tok = open("/tmp/hf_token").read().strip()
from huggingface_hub import HfFileSystem, hf_hub_download
import pyarrow.parquet as pq
fs = HfFileSystem(token=tok)
files = sorted(f for f in fs.ls("datasets/ai4bharat/Rasa/Nepali", detail=False) if f.endswith(".parquet"))
rows = []
for f in files:
    try:
        t = pq.ParquetFile(fs.open(f)).read(columns=["gender", "style", "duration", "text"]).to_pandas()
        t["split"] = "train" if "/train-" in f else "test"; rows.append(t)
    except Exception as e: log("fail", f, repr(e)[:100])
import pandas as pd
df = pd.concat(rows); df["dur"] = pd.to_numeric(df["duration"], errors="coerce")
log("total utts", len(df), "hours", df.dur.sum() / 3600)
inv = {}
for (g, s), x in df.groupby([df.gender.str.lower(), "style"]):
    inv.setdefault(g, {})[s] = {"utts": int(len(x)), "hours": round(float(x.dur.sum() / 3600), 2), "mean_s": round(float(x.dur.mean()), 2)}
tot = {g: {"utts": int(len(x)), "hours": round(float(x.dur.sum() / 3600), 2)} for g, x in df.groupby(df.gender.str.lower())}
uniq_text = df.groupby(df.gender.str.lower()).text.nunique().to_dict()
res = {"totals": tot, "inventory": inv, "unique_texts": {k: int(v) for k, v in uniq_text.items()}, "text_len_chars_mean": float(df.text.str.len().mean())}
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
# ---- UTMOS
try:
    mos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).eval()
except Exception as e:
    res["utmos_error"] = repr(e)[:300]; json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1); sys.exit()
tb = pq.read_table(hf_hub_download("ai4bharat/Rasa", "Nepali/test-00000-of-00007.parquet", repo_type="dataset", token=tok)).to_pandas()
d = tb[tb["style"].isin(["WIKI", "NEWS", "BOOK", "CONV"]) & tb.text.str.len().between(35, 140)].sample(n=40, random_state=0)
from sakhiya.onnx_tts import SakhiyaONNX
sets = collections.OrderedDict()
def dec(a):
    w, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32"); w = w.mean(1) if w.ndim > 1 else w
    return librosa.resample(w, orig_sr=sr, target_sr=16000)
sets["REAL_RASA_F"] = [dec(r.audio) for r in d.itertuples()]
for name, dd, tiny in [("S_VOCOS_FP16", "docs/models_sakhiya_one_vocos", False), ("S_V1_INT8", "docs/models_v2", True), ("S_ONE_COMPACT_INT8", "docs/models_sakhiya_one", True)]:
    m = SakhiyaONNX(os.path.join(REPO, dd), tiny=tiny, threads=4); L = []
    for r in d.itertuples():
        w, sr = m.say(r.text, speaker="sakhiya", lang="ne", style="news"); w = np.asarray(w, np.float32).reshape(-1)
        L.append(librosa.resample(w, orig_sr=sr, target_sr=16000))
    sets[name] = L
res["utmos22"] = {}
for k, L in sets.items():
    sc = []
    for w in L:
        with torch.no_grad(): sc.append(float(mos(torch.from_numpy(w)[None], 16000)))
    res["utmos22"][k] = {"mean": float(np.mean(sc)), "std": float(np.std(sc))}; log("utmos", k, np.mean(sc))
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1); log("DONE")
