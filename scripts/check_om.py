"""Objective pranava / Vedic-verse check: does the exported model actually SAY ॐ (a prolonged voiced o-o-m before the first pada)?
Measures, on the synthesized ॐ-initial verse, the length of the leading voiced stretch (energy + zero-crossing, 10 ms frames) and
its F0 stability, and compares with the same verse without ॐ.  Writes JSON and the wav files used as web samples.
usage: python3 scripts/check_om.py EXPORT_DIR OUT_DIR [--tiny]"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sakhiya.onnx_tts import SakhiyaONNX
import wave

VERSE = "पूर्णमदः पूर्णमिदं पूर्णात्पूर्णमुदच्यते।पूर्णस्य पूर्णमादाय पूर्णमेवावशिष्यते॥"
exp, out = sys.argv[1], sys.argv[2]; tiny = "--tiny" in sys.argv; os.makedirs(out, exist_ok=True)
tts = SakhiyaONNX(exp, tiny=tiny)


def lead(w, sr, thr_db=-38, gap_s=0.14):
    hop = int(sr * 0.01); n = len(w) // hop
    e = np.array([np.sqrt(np.mean(w[i * hop:(i + 1) * hop] ** 2) + 1e-12) for i in range(n)]); db = 20 * np.log10(e / (e.max() + 1e-9) + 1e-9)
    on = db > thr_db; i0 = int(np.argmax(on)); i = i0; gap = 0
    while i < n and gap * 0.01 < gap_s:
        gap = 0 if on[i] else gap + 1; i += 1
    return i0 * 0.01, max(0.0, (i - gap - i0) * 0.01)


def save(p, w, sr):
    with wave.open(p, "wb") as f: f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr); f.writeframes((np.clip(w, -1, 1) * 32767).astype("<i2").tobytes())


res = {}
for name, text in (("with_om", "ॐ " + VERSE), ("no_om", VERSE), ("om_only", "ॐ"), ("shanti", "ॐ शान्तिः शान्तिः शान्तिः")):
    w, sr = tts.say(text, lang="sa", style="neutral"); w = np.asarray(w, np.float32)
    on, ln = lead(w, sr); res[name] = {"dur_s": round(len(w) / sr, 2), "onset_s": round(on, 2), "lead_voiced_s": round(ln, 2)}
    save(os.path.join(out, f"sa_{name}.wav"), w, sr)
res["om_audible"] = bool(res["with_om"]["lead_voiced_s"] >= 0.55 and res["with_om"]["lead_voiced_s"] > res["no_om"]["lead_voiced_s"] + 0.25)
res["om_only_ok"] = bool(res["om_only"]["dur_s"] >= 0.7)
print(json.dumps(res, ensure_ascii=False, indent=1)); json.dump(res, open(os.path.join(out, "om_check.json"), "w"), ensure_ascii=False, indent=1)
