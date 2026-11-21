"""Owner 5-sentence proof pack: render with an export dir, round-trip through MMS-npi, check the reported bug words.
python3 scripts/v25_proof_asr.py --export DIR --out eval/x.json [--wav-dir DIR]   (coarse: MMS has no Sanskrit/ळ model; not a substitute for listening)"""
import argparse, json, os, re, sys, unicodedata
import numpy as np, torch, soxr, soundfile as sf
sys.path.insert(0, os.getcwd())
from sakhiya.onnx_tts import SakhiyaONNX
ap = argparse.ArgumentParser(); ap.add_argument("--export", required=True); ap.add_argument("--out", required=True); ap.add_argument("--wav-dir", default=None); a = ap.parse_args()
P = [("kakhma", "ne", {}, "नेपालको राजधानी काठमाडौँ हिमालको काखमा रहेको एउटा ऐतिहासिक सहर हो।", ["काखमा"], ["काख्वा"]),
     ("om", "sa", {"svara": 1.0}, "ॐ पूर्णमदः पूर्णमिदं पूर्णात्पूर्णमुदच्यते। पूर्णस्य पूर्णमादाय पूर्णमेवावशिष्यते॥", ["ओम", "उदच्य"], ["कोहम", "उधच्य"]),
     ("rv", "sa", {"svara": 1.0}, "अग्निमीळे पुरोहितं यज्ञस्य देवमृत्विजम्। होतारं रत्नधातमम्॥", ["अग्निमी"], ["अग्निमीडे"]),
     ("gita", "sa", {}, "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः। मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय॥", ["युयुत्सव", "पाण्डवाश्चैव", "मामका"], ["युयुच्वह", "मामकहा", "पाण्डवाससैव"]),
     ("story", "ne", {}, "एक समयको कुरा हो, एउटा सानो गाउँमा एउटी बुढी आमा बस्थिन्। उनी बिहानै उठेर चरालाई दाना हाल्थिन्।", ["उठेर"], ["उठ्या"])]
ALTS = {
    "उदच्य": ("उदच्य", "मुदच्य"),
    "अग्निमी": ("अग्निमी", "अग्निमिले"),
    "युयुत्सव": ("युयुत्सव", "युयुतसव", "युयुत्छव"),
    "पाण्डवाश्चैव": ("पाण्डवाश्चैव", "पाण्डवासचैव", "पान्डवासचैव", "पाण्डवासचेव"),
}
norm = lambda t: re.sub(r"\s+", " ", re.sub(r"[^\u0900-\u0963\u0966-\u097F\s]", " ", unicodedata.normalize("NFC", t))).strip()
from transformers import AutoProcessor, Wav2Vec2ForCTC
dev = "cuda"; rid = "facebook/mms-1b-all"
proc = AutoProcessor.from_pretrained(rid, target_lang="npi"); asr = Wav2Vec2ForCTC.from_pretrained(rid, target_lang="npi", ignore_mismatched_sizes=True).to(dev).eval()
tts = SakhiyaONNX(a.export, threads=8); res = {}
for k, lang, kw, t, good, bad in P:
    w, sr = tts.say(t, lang=lang, **kw); w = np.asarray(w, np.float32)
    if a.wav_dir: os.makedirs(a.wav_dir, exist_ok=True); sf.write(f"{a.wav_dir}/{k}.wav", w, sr)
    with torch.no_grad():
        x = proc(soxr.resample(w, sr, 16000), sampling_rate=16000, return_tensors="pt").input_values.to(dev)
        h = norm(proc.batch_decode(asr(x).logits.argmax(-1))[0])
    sq = lambda s: s.replace(" ", "")
    match_g = lambda g: any(sq(v) in sq(h) for v in ALTS.get(g, (g,)))
    res[k] = dict(asr=h, hit=[g for g in good if match_g(g)], miss=[g for g in good if not match_g(g)], bad_seen=[b for b in bad if sq(b) in sq(h)])
    print(k, res[k], flush=True)
json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
