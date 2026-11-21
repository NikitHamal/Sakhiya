#!/usr/bin/env python3
"""Deep acoustic diagnostic for `काठमाडौँ` across all human clips + 15-clip v3.1 glitch/ASR audit."""
import json, sys
sys.path.insert(0, "/work/tts/repo")
import librosa, numpy as np, torch
from transformers import AutoProcessor, Wav2Vec2ForCTC
from sakhiya.ctc_aligner import MelCTCAligner
from sakhiya.v2_vocoder import mel_spec_24k
from sakhiya.symbols import SYMBOLS

dev = "cuda" if torch.cuda.is_available() else "cpu"
D = torch.load("/work/tts/feats/v3_24k/v31_studio_aligned_24k.pt", map_location="cpu", weights_only=False)
mean, std = float(D["mean"]), float(D["std"])
rec = MelCTCAligner().to(dev).eval()
rec.load_state_dict(torch.load("/work/tts/feats/v31_ctc_recognizer.pt", map_location=dev, weights_only=False)["model"])

print("=== 1. ALL HUMAN CLIPS OF काठमाडौँ / काठमाडौं ===")
hits = [x for x in D["items"] if "काठमाड" in x["text"] or "काठमाण्ड" in x["text"]]
print(f"Found {len(hits)} human clips containing काठमाड/काठमाण्ड:")
for idx, x in enumerate(hits[:12]):
    mel = ((x["mel"].float().to(dev)[None] - mean) / std)
    with torch.no_grad():
        lp = rec(mel)[:, 0].argmax(-1).cpu().tolist()
    ids_l = x["ids"].tolist()
    durs_l = x["dur_ctc"].tolist()
    cs = np.cumsum([0] + durs_l)
    words = x["text"].split()
    wi = next((i for i, tok in enumerate(words) if "काठमाड" in tok or "काठमाण्ड" in tok), 0)
    w_bounds = [0] + [i + 1 for i, pid in enumerate(ids_l) if pid == 1] + [len(ids_l)]
    b0 = w_bounds[min(wi, len(w_bounds) - 2)]
    b1 = w_bounds[min(wi + 1, len(w_bounds) - 1)]
    f0_i, f1_i = max(0, int(cs[b0]) - 2), min(len(lp), int(cs[b1]) + 2)
    raw_frames = [SYMBOLS[i] if i > 0 else "_" for i in lp[f0_i:f1_i]]
    ded = [SYMBOLS[i] for k, i in enumerate(lp[f0_i:f1_i]) if i != 0 and (k == 0 or lp[f0_i + k - 1] != i)]
    al_slice = [f"{SYMBOLS[ids_l[k]]}:{durs_l[k]}" for k in range(b0, b1)]
    print(f"  #{idx+1} [spk={x['spk']} src={x.get('src')} word={words[wi]}]")
    print(f"     ALIGN_DUR: {' '.join(al_slice)}")
    print(f"     CTC_DEDUP: {' '.join(ded)}")
    print(f"     CTC_FRAME: {' '.join(raw_frames)}")

print("\n=== 2. 15-CLIP v3.1 F0 GLITCH & MMS-1B-ALL AUDIT ===")
proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="npi")
asr = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="npi", ignore_mismatched_sizes=True).to(dev).eval()
man = json.load(open("/work/tts/repo/docs/samples/versions/v3_1/manifest.json"))
tot_glitches = 0
for c in man["clips"]:
    w24, _ = librosa.load("/work/tts/repo/docs/" + c["mp3"], sr=24000)
    w16 = librosa.resample(w24, orig_sr=24000, target_sr=16000)
    with torch.no_grad():
        inp = proc(w16, sampling_rate=16000, return_tensors="pt").input_values.to(dev)
        hyp = proc.batch_decode(asr(inp).logits.argmax(-1))[0]
    f0 = librosa.yin(w24.astype(np.float32), fmin=80, fmax=420, sr=24000, hop_length=256)
    rms = librosa.feature.rms(y=w24, frame_length=1024, hop_length=256)[0][:len(f0)]
    voiced = rms > (np.max(rms) * 0.08)
    st_diff = np.abs(12.0 * np.log2(np.maximum(f0[1:], 1.0) / np.maximum(f0[:-1], 1.0)))
    v_pair = voiced[1:] & voiced[:-1]
    glitches = int(np.sum(st_diff[v_pair] > 3.0))
    tot_glitches += glitches
    max_jump = float(np.max(st_diff[v_pair])) if np.any(v_pair) else 0.0
    print(f"[{c['id']:19s}] ({c['style']:14s}) glitch={glitches:2d} max_jump={max_jump:5.2f}st | HYP={hyp[:65]}")
print(f"TOTAL GLITCH FRAMES ACROSS 15 v3.1 CLIPS: {tot_glitches}")
