#!/usr/bin/env python3
"""EXP1 (Sonnet lane): how far is the shipped voice from REAL studio Nepali speech, measured with
strict, un-gamed metrics?  CPU only.  Compares, on the SAME sentences:
  REAL_F / REAL_M : ai4bharat/Rasa Nepali test clips (female / male, studio 48k->24k)
  S_VOCOS         : shipped default  docs/models_sakhiya_one_vocos (cand_b_v2, FP16 graph)
  S_V1            : shipped Sakhiya-V1 docs/models_v2 (INT8)
Metrics: prosody (F0 range / dynamics, speaking rate, pauses, energy dynamics), voice-quality (HNR, jitter,
shimmer), STRICT ASR CER/WER with facebook/mms-1b-all (npi, no phoneme fuzzing, no loan-word offsets),
UTMOS22 (to show what that proxy rewards).
usage: python exp1_real_vs_shipped.py REPO_DIR OUT_JSON [N]
"""
import os, sys, json, io, time, re, warnings
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, librosa
REPO, OUT = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 40
sys.path.insert(0, REPO)
SR = 24000
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)

# ---------------------------------------------------------------- real data
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
tok = open("/tmp/hf_token").read().strip()
pqf = hf_hub_download("ai4bharat/Rasa", "Nepali/test-00000-of-00007.parquet", repo_type="dataset", token=tok)
tb = pq.read_table(pqf).to_pandas()
log("rasa shard rows", len(tb), "cols", list(tb.columns))
log(tb.groupby(["gender", "style"]).size().to_string())
def dec(a):
    w, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
    if w.ndim > 1: w = w.mean(1)
    return librosa.resample(w, orig_sr=sr, target_sr=SR) if sr != SR else w
rng = np.random.RandomState(0)
def pick(g, n):
    d = tb[(tb.gender.str.lower().str.startswith(g)) & (tb.text.str.len().between(35, 140))]
    # neutral reading styles only so we compare like with like
    dn = d[~d["style"].str.lower().isin(["happy", "sad", "angry", "fear", "disgust", "surprise"])]
    dn = dn if len(dn) >= n else d
    return dn.sample(n=min(n, len(dn)), random_state=0)
rf, rm = pick("f", N), pick("m", N)
log("styles used F:", rf["style"].value_counts().to_dict(), "M:", rm["style"].value_counts().to_dict())
sets = {"REAL_F": [(r.text, dec(r.audio)) for r in rf.itertuples()],
        "REAL_M": [(r.text, dec(r.audio)) for r in rm.itertuples()]}
texts = [t for t, _ in sets["REAL_F"]]

# ---------------------------------------------------------------- shipped models on the REAL female texts
from sakhiya.onnx_tts import SakhiyaONNX
def run_model(name, d, tiny, **kw):
    m = SakhiyaONNX(os.path.join(REPO, d), tiny=tiny, threads=4)
    out = []
    for t in texts:
        try:
            w, sr = m.say(t, speaker="sakhiya", lang="ne", style="news", **kw)
            w = np.asarray(w, dtype=np.float32).reshape(-1)
            if sr != SR: w = librosa.resample(w, orig_sr=sr, target_sr=SR)
            out.append((t, w))
        except Exception as e:
            log("synth fail", name, repr(e)[:120]); out.append((t, np.zeros(SR, np.float32)))
    sets[name] = out; log("synth done", name)
run_model("S_VOCOS_FP16", "docs/models_sakhiya_one_vocos", False)
run_model("S_V1_INT8", "docs/models_v2", True)

# ---------------------------------------------------------------- acoustics
import pyworld as pw, parselmouth
from parselmouth.praat import call
def acoustics(w, text):
    w = w.astype(np.float64); w = w / (np.abs(w).max() + 1e-9) * 0.7
    f0, t = pw.harvest(w, SR, f0_floor=70, f0_ceil=450, frame_period=10.0)
    v = f0 > 0
    r = {}
    if v.sum() > 20:
        st = 12 * np.log2(f0[v] / np.median(f0[v]))
        r["f0_med_hz"] = float(np.median(f0[v])); r["f0_std_st"] = float(st.std())
        r["f0_p5_p95_st"] = float(np.percentile(st, 95) - np.percentile(st, 5))
        # dynamics inside voiced runs only
        d = np.abs(np.diff(12 * np.log2(np.where(v, f0, np.nan)))); d = d[~np.isnan(d)]
        r["f0_abs_delta_st_per_10ms"] = float(d.mean()) if len(d) else None
        # slow (phrase-level) vs fast: std of 300ms smoothed contour
        ff = np.where(v, 12 * np.log2(np.maximum(f0, 1) / np.median(f0[v])), np.nan)
        s = np.array([np.nanmean(ff[i:i + 30]) if np.isfinite(ff[i:i + 30]).any() else np.nan for i in range(0, len(ff) - 30, 15)])
        r["f0_phrase_std_st"] = float(np.nanstd(s))
    r["voiced_frac"] = float(v.mean())
    rms = librosa.feature.rms(y=w.astype(np.float32), frame_length=1024, hop_length=240)[0]
    db = 20 * np.log10(rms + 1e-6); act = db > (db.max() - 40)
    sil = ~act; # pauses >=150ms inside utterance (strip leading/trailing)
    idx = np.where(act)[0]
    if len(idx) > 5:
        a0, a1 = idx[0], idx[-1]; core = sil[a0:a1 + 1]
        runs, c = [], 0
        for x in core:
            if x: c += 1
            else:
                if c: runs.append(c); c = 0
        pa = [x * 0.01 for x in runs if x * 0.01 >= 0.15]
        r["n_pauses"] = len(pa); r["pause_mean_s"] = float(np.mean(pa)) if pa else 0.0
        r["speech_dur_s"] = float((a1 - a0 + 1) * 0.01 - sum(pa))
        gr = len([c for c in text if '\u0900' <= c <= '\u097f' and c not in '\u093e\u093f\u0940\u0941\u0942\u0947\u0948\u094b\u094c\u094d\u0902\u0901\u0903'])
        r["graphemes_per_s"] = gr / max(r["speech_dur_s"], 0.2)
        r["energy_std_db"] = float(db[a0:a1 + 1][act[a0:a1 + 1]].std())
    r["dur_s"] = len(w) / SR
    try:
        sn = parselmouth.Sound(w, SR)
        h = call(sn, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        r["hnr_db"] = float(call(h, "Get mean", 0, 0))
        pp = call(sn, "To PointProcess (periodic, cc)", 75, 450)
        r["jitter_local"] = float(call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
        r["shimmer_local"] = float(call([sn, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
    except Exception as e:
        pass
    S = np.abs(librosa.stft(w.astype(np.float32), n_fft=1024, hop_length=256)) ** 2
    fr = librosa.fft_frequencies(sr=SR, n_fft=1024); m = S.mean(1) + 1e-12
    def band(a, b): return 10 * np.log10(m[(fr >= a) & (fr < b)].mean())
    r["tilt_8_12k_minus_4_8k_db"] = float(band(8000, 12000) - band(4000, 8000))
    r["tilt_4_8k_minus_0_2k_db"] = float(band(4000, 8000) - band(0, 2000))
    return r
def agg(lst):
    keys = sorted({k for d in lst for k in d if d[k] is not None})
    return {k: {"mean": float(np.nanmean([d[k] for d in lst if k in d and d[k] is not None])),
                "std": float(np.nanstd([d[k] for d in lst if k in d and d[k] is not None]))} for k in keys}
res = {"N": N}
per_clip = {}
for name, lst in sets.items():
    A = [acoustics(w, t) for t, w in lst]
    per_clip[name] = A; res[name] = {"acoustics": agg(A)}
    log("acoustics", name)

# between-utterance prosodic diversity: how much do per-utterance f0_std/rate vary (re-use of one template = low)
for name in sets:
    fs = [a.get("f0_std_st") for a in per_clip[name] if a.get("f0_std_st") is not None]
    gs = [a.get("graphemes_per_s") for a in per_clip[name] if a.get("graphemes_per_s") is not None]
    res[name]["diversity"] = {"f0_std_st_across_utts_std": float(np.std(fs)), "rate_across_utts_std": float(np.std(gs))}

# ---------------------------------------------------------------- strict ASR
import torch, jiwer
torch.set_num_threads(16)
from transformers import AutoProcessor, Wav2Vec2ForCTC
proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="npi", token=tok)
asr = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="npi", ignore_mismatched_sizes=True, token=tok).eval()
def norm(s):
    s = re.sub(r"[।॥,.?!\-–—\"'“”‘’():;|]", " ", s); s = s.replace("ँ", "ं")
    return re.sub(r"\s+", " ", s).strip()
def transcribe(w):
    x = librosa.resample(w.astype(np.float32), orig_sr=SR, target_sr=16000)
    with torch.no_grad():
        lg = asr(proc(x, sampling_rate=16000, return_tensors="pt").input_values).logits
    return proc.batch_decode(lg.argmax(-1))[0]
for name, lst in sets.items():
    refs, hyps, cers, wers = [], [], [], []
    for t, w in lst:
        h = transcribe(w); r_, h_ = norm(t), norm(h)
        refs.append(r_); hyps.append(h_)
        cers.append(jiwer.cer(r_, h_) if r_ else 0); wers.append(jiwer.wer(r_, h_) if r_ else 0)
    res[name]["asr_strict"] = {"cer_mean": float(np.mean(cers)), "wer_mean": float(np.mean(wers)),
        "cer_nospace": float(jiwer.cer(re.sub(" ", "", " ".join(refs)), re.sub(" ", "", " ".join(hyps)))),
        "examples": [{"ref": a, "hyp": b} for a, b in list(zip(refs, hyps))[:6]]}
    log("asr", name, res[name]["asr_strict"]["cer_mean"], res[name]["asr_strict"]["wer_mean"])
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)

# ---------------------------------------------------------------- UTMOS22 (what does the proxy reward?)
try:
    mos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).eval()
    for name, lst in sets.items():
        sc = []
        for t, w in lst:
            x = librosa.resample(w.astype(np.float32), orig_sr=SR, target_sr=16000)
            with torch.no_grad(): sc.append(float(mos(torch.from_numpy(x)[None], 16000)))
        res[name]["utmos22"] = {"mean": float(np.mean(sc)), "std": float(np.std(sc))}
        log("utmos", name, np.mean(sc))
except Exception as e:
    res["utmos_error"] = repr(e)[:300]
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
# keep a few wavs for later listening/ASR re-analysis
os.makedirs(OUT + ".wavs", exist_ok=True)
for name, lst in sets.items():
    for i, (t, w) in enumerate(lst[:6]): sf.write(f"{OUT}.wavs/{name}_{i}.wav", w, SR)
log("DONE")
