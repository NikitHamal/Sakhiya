#!/usr/bin/env python3
"""EXP5 (Sonnet lane): what is "the bar"?  Microsoft Edge neural Nepali voices (ne-NP-HemkalaNeural F, ne-NP-SagarNeural M) are the
best widely-available Nepali TTS.  Score them with the SAME battery as exp1/exp3 on the SAME 40 Rasa sentences.
REFERENCE ONLY (not a training source: Edge TTS output is not licensed for model training).   usage: exp5 REPO OUT_JSON"""
import os, sys, json, io, asyncio, warnings, re, subprocess, tempfile
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, librosa, torch, jiwer
REPO, OUT = sys.argv[1], sys.argv[2]; SR = 24000
import edge_tts
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
tok = open("/tmp/hf_token").read().strip()
tb = pq.read_table(hf_hub_download("ai4bharat/Rasa", "Nepali/test-00000-of-00007.parquet", repo_type="dataset", token=tok)).to_pandas()
d = tb[tb["style"].isin(["WIKI", "NEWS", "BOOK", "CONV"]) & tb.text.str.len().between(35, 140)].sample(n=40, random_state=0)
texts = list(d.text)
async def one(t, v, p):
    await edge_tts.Communicate(t, v).save(p)
def synth(voice):
    out = []
    for i, t in enumerate(texts):
        p = f"/tmp/edge_{voice}_{i}.mp3"
        try:
            asyncio.run(one(t, voice, p)); w, _ = librosa.load(p, sr=SR, mono=True); out.append(w)
        except Exception as e:
            print("fail", voice, i, repr(e)[:100]); out.append(np.zeros(SR, np.float32))
    return out
sets = {"EDGE_HEMKALA_F": synth("ne-NP-HemkalaNeural"), "EDGE_SAGAR_M": synth("ne-NP-SagarNeural")}
mos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).eval()
from transformers import AutoProcessor, Wav2Vec2ForCTC
proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="npi", token=tok)
asr = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="npi", ignore_mismatched_sizes=True, token=tok).eval()
torch.set_num_threads(16)
def norm(s):
    s = re.sub(r"[।॥,.?!\-–—\"'“”‘’():;|]", " ", s).replace("ँ", "ं"); return re.sub(r"\s+", " ", s).strip()
import pyworld as pw
res = {}
for name, L in sets.items():
    sc, cers, wers, pa, rate, dd, df0 = [], [], [], [], [], [], []
    for t, w in zip(texts, L):
        x = librosa.resample(w, orig_sr=SR, target_sr=16000)
        with torch.no_grad():
            sc.append(float(mos(torch.from_numpy(x)[None], 16000)))
            h = proc.batch_decode(asr(proc(x, sampling_rate=16000, return_tensors="pt").input_values).logits.argmax(-1))[0]
        r_, h_ = norm(t), norm(h); cers.append(jiwer.cer(r_, h_)); wers.append(jiwer.wer(r_, h_))
        db = 20 * np.log10(librosa.feature.rms(y=w, frame_length=1024, hop_length=240)[0] + 1e-6); act = db > db.max() - 40
        idx = np.where(act)[0]; core = ~act[idx[0]:idx[-1] + 1]; runs, c = [], 0
        for z in core:
            if z: c += 1
            else:
                if c: runs.append(c); c = 0
        pa.append(len([q for q in runs if q * 0.01 >= 0.15]))
        gr = len([ch for ch in t if '\u0900' <= ch <= '\u097f' and ch not in '\u093e\u093f\u0940\u0941\u0942\u0947\u0948\u094b\u094c\u094d\u0902\u0901\u0903'])
        rate.append(gr / max(0.2, (idx[-1] - idx[0] + 1) * 0.01)); dd.append(len(w) / SR)
        f0, _ = pw.harvest(w.astype(np.float64), SR, f0_floor=70, f0_ceil=450, frame_period=10.0); v = f0 > 0
        if v.sum() > 20:
            ff = 12 * np.log2(np.where(v, f0, np.nan) / np.median(f0[v])); dlt = np.abs(np.diff(ff)); df0.append(float(np.nanmean(dlt)))
    res[name] = dict(utmos22=float(np.mean(sc)), cer=float(np.mean(cers)), wer=float(np.mean(wers)), pauses_per_utt=float(np.mean(pa)),
                     graphemes_per_s=float(np.mean(rate)), mean_dur_s=float(np.mean(dd)), abs_dF0_st_per_10ms=float(np.mean(df0)))
    print(name, res[name], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
