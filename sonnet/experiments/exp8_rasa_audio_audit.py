#!/usr/bin/env python3
"""EXP8 (Sonnet lane, session 2): AUDIO-level curation audit of Rasa Nepali (random clips of 2 female + 2 male train shards).
Per clip: native sample-rate, edge silence, peak/clipping, RMS dBFS, noise floor (p5 frame dBFS), dynamic range proxy (p95-p5),
99 % spectral roll-off (true bandwidth: is the 48 kHz real?), median F0 (yin), net speaking rate (chars/s of non-silent time).
Output = per gender x style-group quantiles -> concrete curation thresholds for the GPU-session data build.
usage: exp8_rasa_audio_audit.py OUT_JSON [N_PER_SHARD]
"""
import sys, io, json, os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, librosa, pyarrow.parquet as pq, pandas as pd
os.environ.setdefault("HF_HOME", "/work/sonnet/hf")
from huggingface_hub import hf_hub_download
tok = open("/tmp/hf_token").read().strip(); N = int(sys.argv[2]) if len(sys.argv) > 2 else 80
SH = ["train-00003-of-00057", "train-00015-of-00057", "train-00034-of-00057", "train-00047-of-00057"]
GRP = {"WIKI": "neutral", "BOOK": "neutral", "NEWS": "neutral", "CONV": "conv", "PROPER NOUN": "proper"}
rows = []; T0 = time.time()
for sh in SH:
    df = pq.read_table(hf_hub_download("ai4bharat/Rasa", f"Nepali/{sh}.parquet", repo_type="dataset", token=tok)).to_pandas()
    for r in df.sample(n=min(N, len(df)), random_state=1).itertuples():
        w, sr = sf.read(io.BytesIO(r.audio["bytes"]), dtype="float32")
        if w.ndim > 1: w = w.mean(1)
        hop = int(sr * 0.01); fr = librosa.feature.rms(y=w, frame_length=hop * 2, hop_length=hop)[0]; db = 20 * np.log10(fr + 1e-7)
        act = db > (np.percentile(db, 95) - 35)
        idx = np.where(act)[0]; lead = idx[0] * 0.01 if len(idx) else 0; trail = (len(db) - 1 - idx[-1]) * 0.01 if len(idx) else 0
        S = np.abs(np.fft.rfft(w[: len(w) // 2048 * 2048].reshape(-1, 2048) * np.hanning(2048), axis=1)) ** 2; ps = S.mean(0); cs = np.cumsum(ps) / ps.sum()
        roll99 = float(np.fft.rfftfreq(2048, 1 / sr)[np.searchsorted(cs, 0.99)])
        w24 = librosa.resample(w, orig_sr=sr, target_sr=24000)
        f0 = librosa.yin(w24, fmin=70, fmax=420, sr=24000, frame_length=1024, hop_length=240); fv = f0[(f0 > 72) & (f0 < 410)]
        dur = len(w) / sr; net = max(0.3, dur - lead - trail)
        rows.append(dict(shard=sh, gender=str(r.gender).lower(), style=r.style, grp=GRP.get(r.style, "emotion/command"), sr=sr, dur=dur, lead=lead, trail=trail,
                         peak=float(np.abs(w).max()), clipfrac=float((np.abs(w) > 0.999).mean()), rms_db=float(np.percentile(db, 75)), floor_db=float(np.percentile(db, 5)),
                         dr_db=float(np.percentile(db, 95) - np.percentile(db, 5)), roll99=roll99, f0=float(np.median(fv)) if len(fv) else 0.0,
                         f0_iqr_st=float(12 * np.log2(np.percentile(fv, 75) / np.percentile(fv, 25))) if len(fv) > 20 else 0.0, cps=len(r.text) / net,
                         gap_frac=float((~act[idx[0]:idx[-1] + 1]).mean()) if len(idx) > 1 else 0.0))
    print(sh, len(rows), round(time.time() - T0), "s", flush=True)
d = pd.DataFrame(rows); out = {"n": len(d), "sample_rates": d.sr.value_counts().to_dict()}
cols = ["dur", "lead", "trail", "peak", "clipfrac", "rms_db", "floor_db", "dr_db", "roll99", "f0", "f0_iqr_st", "cps", "gap_frac"]
out["by_gender_group"] = {f"{g}/{k}": {c: {q: round(float(x[c].quantile(v)), 3) for q, v in (("p05", .05), ("p50", .5), ("p95", .95))} for c in cols} | {"n": int(len(x))}
                          for (g, k), x in d.groupby(["gender", "grp"])}
out["by_gender"] = {g: {c: {q: round(float(x[c].quantile(v)), 3) for q, v in (("p05", .05), ("p50", .5), ("p95", .95))} for c in cols} for g, x in d.groupby("gender")}
out["clipped_clips_frac"] = {g: round(float((x.clipfrac > 1e-4).mean()), 3) for g, x in d.groupby("gender")}
out["f0_by_style"] = {f"{g}/{s}": round(float(x.f0.median()), 1) for (g, s), x in d.groupby(["gender", "style"])}
json.dump(out, open(sys.argv[1], "w"), indent=1); d.to_csv(sys.argv[1].replace(".json", ".csv"), index=False)
print(json.dumps(out["by_gender"], indent=1)[:3500])
