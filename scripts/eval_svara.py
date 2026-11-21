"""Standalone Sanskrit svara measurement (same block as scripts/evaluate.py): F0 in semitones (re. median) on svarita/udatta (+) vs anudatta (-) vowels of a
Rigveda-marked line, for svara depth 0 / 1 on fp16 and tiny-int8, merged into the given metrics.json files.  usage: eval_svara.py <models_dir> <metrics.json> [...]"""
import json, sys, numpy as np, librosa
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
from sakhiya.onnx_tts import SakhiyaONNX
from sakhiya import g2p
VT = "अ॒ग्निमी॑ळे पु॒रोहि॑तं य॒ज्ञस्य॑ दे॒वमृ॒त्विज॑म् । होता॑रं रत्न॒धात॑मम् ॥"
from sakhiya import symbols as SY
_ph, _off = g2p.text_to_phones_acc(VT, "sa"); base = [o for p, o in zip(_ph, _off) if p in SY.SYM2ID]; out = {}
for tiny in (False, True):
    m = SakhiyaONNX(sys.argv[1], tiny=tiny, threads=4)
    for dep in (1.0, 0.0):
        w, sr = m.say(VT, speaker="female", lang="sa", style="chant", svara=dep); dur = np.concatenate([t[1] for t in m.trace])
        if dep == 1.0: base = np.concatenate([t[0] for t in m.trace]) / 0.9      # per-phone offsets of the chunks actually synthesised
        f0 = librosa.pyin(w.astype(np.float32), fmin=70, fmax=400, sr=sr, hop_length=256, frame_length=2048)[0]; st = 12 * np.log2(f0 / np.nanmedian(f0))
        fr = np.repeat(np.array(base), dur)[: len(st)]; s = st[: len(fr)]; hi, lo = float(np.nanmean(s[fr >= 0.9])), float(np.nanmean(s[fr <= -0.9]))
        out[("int8_" if tiny else "") + f"svara_depth_{dep:g}"] = dict(svarita_st=round(hi, 2), anudatta_st=round(lo, 2), contrast_st=round(hi - lo, 2), f0_std_st=round(float(np.nanstd(st)), 2))
print(json.dumps(out, indent=1))
for p in sys.argv[2:]:
    d = json.load(open(p)); d["svara"] = out; json.dump(d, open(p, "w"), indent=1, ensure_ascii=False)
