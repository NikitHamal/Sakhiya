#!/usr/bin/env python3
"""Acoustic-Verified Nepali & Sanskrit G2P Truth Audit (Phase 1):
1. Extracts exact word-level human CTC phone slices for `काठमाडौँ`, `गाउँ`, `आउँछ`, `जाऔँ`, `छौँ`, `बाट`, `तपाईँ`, `गर्नुभएको`, `हिँड्दै`, `साथी`, `विद्यालय`, `अद्भुत`, `ऐतिहासिक`, `माछापुच्छ्रे`.
2. Transcribes the 15 frozen `v3_1` baseline clips with MMS-1b-all + MelCTCAligner + F0 glitch detector."""
import json, os, sys
sys.path.insert(0, "/work/tts/repo")
import librosa, numpy as np, torch
from transformers import AutoProcessor, Wav2Vec2ForCTC
from sakhiya.ctc_aligner import MelCTCAligner
from sakhiya.v2_vocoder import mel_spec_24k
from sakhiya.symbols import SYMBOLS
from sakhiya import g2p

dev = "cuda" if torch.cuda.is_available() else "cpu"
D = torch.load("/work/tts/feats/v3_24k/v31_studio_aligned_24k.pt", map_location="cpu", weights_only=False)
mean, std = float(D["mean"]), float(D["std"])
rec = MelCTCAligner().to(dev).eval()
rec.load_state_dict(torch.load("/work/tts/feats/v31_ctc_recognizer.pt", map_location=dev, weights_only=False)["model"])

targets = ["काठमाडौँ", "काठमाडौं", "गाउँ", "आउँछ", "जाऔँ", "छौँ", "बाट", "तपाईँ", "गर्नुभएको", "हिँड्दै", "साथी", "विद्यालय", "अद्भुत", "ऐतिहासिक", "माछापुच्छ्रे"]
print("=== 1. WORD-LEVEL HUMAN ACOUSTIC CTC AUDIT ===")
for w in targets:
    hits = [x for x in D["items"] if w in x["text"]]
    print(f"WORD {w} | cur_g2p={' '.join(g2p.word_to_phones(w))} | clips={len(hits)}")
    for x in hits[:3]:
        mel = ((x["mel"].float().to(dev)[None] - mean) / std)
        with torch.no_grad():
            lp = rec(mel)[:, 0].argmax(-1).cpu().tolist()
        # Find where the word appears in x["ids"] and slice the exact frame range using x["dur_ctc"]
        ids_l = x["ids"].tolist()
        durs_l = x["dur_ctc"].tolist()
        cs = np.cumsum([0] + durs_l)
        w_ph = g2p.word_to_phones(w)
        # locate subsequence or approximate word index
        words = x["text"].split()
        wi = next((i for i, tok in enumerate(words) if w in tok), 0)
        # slice frames by word boundaries (id == 1 is '|')
        w_bounds = [0] + [idx + 1 for idx, pid in enumerate(ids_l) if pid == 1] + [len(ids_l)]
        b0 = w_bounds[min(wi, len(w_bounds) - 2)]
        b1 = w_bounds[min(wi + 1, len(w_bounds) - 1)]
        f0_idx, f1_idx = max(0, int(cs[b0]) - 4), min(len(lp), int(cs[b1]) + 4)
        sub_lp = lp[f0_idx:f1_idx]
        ded = [SYMBOLS[i] for k, i in enumerate(sub_lp) if i != 0 and (k == 0 or sub_lp[k - 1] != i)]
        al_slice = [f"{SYMBOLS[ids_l[k]]}:{durs_l[k]}" for k in range(b0, b1)]
        print(f"  [spk={x['spk']} {x.get('src')}] ALIGN={' '.join(al_slice)} | CTC_HEARD={' '.join(ded)}")

print("\n=== 2. MMS-1B-ALL + F0 GLITCH AUDIT ON FROZEN v3.1 15 CLIPS ===")
proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="npi")
asr = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="npi", ignore_mismatched_sizes=True).to(dev).eval()
man = json.load(open("/work/tts/repo/docs/samples/versions/v3_1/manifest.json"))
for c in man["clips"]:
    w24, _ = librosa.load("/work/tts/repo/docs/" + c["mp3"], sr=24000)
    w16 = librosa.resample(w24, orig_sr=24000, target_sr=16000)
    with torch.no_grad():
        inp = proc(w16, sampling_rate=16000, return_tensors="pt").input_values.to(dev)
        hyp = proc.batch_decode(asr(inp).logits.argmax(-1))[0]
    f0 = librosa.pyin(w24.astype(np.float32), fmin=80, fmax=420, sr=24000, hop_length=256)[0]
    valid = ~np.isnan(f0)
    st_diff = np.abs(12.0 * np.log2(f0[1:] / np.maximum(f0[:-1], 1e-3)))
    glitches = int(np.sum(st_diff[valid[1:] & valid[:-1]] > 3.5))
    print(f"[{c['id']}] glitches={glitches} | REF={c['text'][:45]} | HYP={hyp[:55]}")
