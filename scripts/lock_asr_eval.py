"""Round-trip ASR (MMS-1b-all) CER on Sanskrit / Nepali / Tharu clips; run from either checkout to A/B front-end changes.
python3 scripts/lock_asr_eval.py --export DIR --out JSON   (lock_voice False/True are both measured; old code ignores the flag)"""
import argparse, json, os, sys, re, unicodedata
import numpy as np, torch, jiwer, soxr
sys.path.insert(0, os.getcwd())
from sakhiya.onnx_tts import SakhiyaONNX
from sakhiya import g2p
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scripts.voice_unify_eval import NE, SA, TH
norm = lambda t: re.sub(r"\s+", " ", re.sub(r"[^\u0900-\u0963\u0966-\u097F\s]", " ", unicodedata.normalize("NFC", t))).strip()
ph = lambda t: "".join(g2p.text_to_phones(t, add_end=False)).replace("|", "")
ap = argparse.ArgumentParser(); ap.add_argument("--export", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
from transformers import AutoProcessor, Wav2Vec2ForCTC
dev = "cuda"; rid = "facebook/mms-1b-all"
proc = AutoProcessor.from_pretrained(rid, target_lang="npi"); asr = Wav2Vec2ForCTC.from_pretrained(rid, target_lang="npi", ignore_mismatched_sizes=True).to(dev).eval()
@torch.no_grad()
def tr(w, sr):
    x = proc(soxr.resample(w.astype(np.float32), sr, 16000), sampling_rate=16000, return_tensors="pt").input_values.to(dev)
    return proc.batch_decode(asr(x).logits.argmax(-1))[0]
tts = SakhiyaONNX(a.export, threads=4); res = {}
for lock in (False, True):
    for lang, texts in (("ne", NE), ("sa", SA), ("thr_dangaura", TH)):
        pc, cc = [], []
        for t in texts:
            w, sr = tts.say(t, lang=lang, lock_voice=lock); h = tr(w, sr)
            pc.append(jiwer.cer(ph(norm(t)), ph(norm(h)))); cc.append(jiwer.cer(norm(t), norm(h)))
        res[f"{lang}|lock={lock}"] = {"pCER": float(np.mean(pc)), "CER": float(np.mean(cc))}
        print(lang, lock, res[f"{lang}|lock={lock}"], flush=True)
json.dump(res, open(a.out, "w"), indent=1, allow_nan=False)
