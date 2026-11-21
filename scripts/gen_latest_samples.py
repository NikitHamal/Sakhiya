"""Render the newest Sakhiya v2 checkpoint and Wave 42 side-by-side as small MP3s for the web page.

The page plays these static files directly, so visitors never have to download/compile a 30 MB ONNX model
just to hear the latest checkpoint.   usage (on the GPU box):
  python3 scripts/gen_latest_samples.py [--v2 /work/tts/export_v2_pub]
Writes docs/samples/latest/*.mp3 + manifest.json.   Idempotent; safe to run from the publish loop.
"""
import os, sys, json, time, argparse, shutil
import numpy as np, soundfile as sf

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from sakhiya.onnx_tts import SakhiyaONNX

ap = argparse.ArgumentParser()
ap.add_argument("--v2", default="/work/tts/export_v2_pub")
ap.add_argument("--out", default=os.path.join(ROOT, "docs", "samples", "latest"))
a = ap.parse_args()

OM = "ॐ पूर्णमदः पूर्णमिदं पूर्णात्पूर्णमुदच्यते। पूर्णस्य पूर्णमादाय पूर्णमेवावशिष्यते॥"
RV = "अ॒ग्निमी॑ळे पु॒रोहि॑तं य॒ज्ञस्य॑ दे॒वमृ॒त्विज॑म् । होता॑रं रत्न॒धात॑मम् ॥"
# id, group, label, lang, text, extra kwargs
OWNER_MIXED = ("हेलो साथीहरू। म सखिया हुँ, निकित हमालज्यूले बनाउनुभएको सानो टेक्स्ट टु स्पीच मोडेल। म संस्कृत पनि बोल्न सक्छु। "
               "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥ र थारू भाषामा पनि बोल्न सक्छु। "
               "हेलो संघरियाहुक्रे। मै सखिया हों, निकित हमाल जी बनाइल छोट टेक्स्ट टु स्पीच मोडेल।")
ITEMS = [
    ("owner_mixed", "Mixed (Nepali · Sanskrit · Tharu)", "Owner Mixed-Language Test (Auto)", "auto", OWNER_MIXED, {}),
    ("ne_hello", "Nepali", "Greeting (Conversational)", "ne", "नमस्ते! सखिया एउटा सानो तर स्पष्ट नेपाली आवाज हो। आजको मौसम धेरै राम्रो छ।", dict(style="conversational")),
    ("ne_news", "Nepali", "News-style sentence", "ne", "नेपालको राजधानी काठमाडौं हिमालको काखमा रहेको एउटा ऐतिहासिक सहर हो।", {}),
    ("ne_story", "Nepali", "Short story (Storytelling)", "ne", "एक समयको कुरा हो, एउटा सानो गाउँमा एउटी बूढी आमा बस्थिन्। उनी बिहानै उठेर चरालाई दाना हाल्थिन्।", dict(style="storytelling")),
    ("ne_roman", "Nepali", "Romanized Nepali (ma gaunma baschu)", "ne", "ma gaunma baschu", {}),
    ("sa_om", "Sanskrit", "ॐ पूर्णमदः (Isha Upanishad)", "sa", OM, dict(style="chant", svara=1.0)),
    ("sa_rv", "Sanskrit", "Rigveda 1.1.1 (Vedic accents)", "sa", RV, dict(style="chant", svara=1.0)),
    ("sa_gita", "Sanskrit", "Bhagavad Gita 1.1", "sa", "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥", {}),
    ("thr_chitwania", "Tharu", "Chitwania", "thr_chitwania", "जात जातक भासा अलग बा", {}),
    ("thr_rana", "Tharu", "Rana", "thr_rana", "हमार नाउ का हो", {}),
    ("thr_dangaura", "Tharu", "Dangaura", "thr_dangaura", "हमार गाउँ बहुत सुन्दर बा", {}),
]


def save(path, w, sr):
    w = np.asarray(w, dtype=np.float32)
    w = w / max(1.0, float(np.abs(w).max()) / 0.95)
    sf.write(path, w, sr, format="MP3")


os.makedirs(a.out, exist_ok=True)
for f in os.listdir(a.out):
    if f.endswith(".mp3"): os.remove(os.path.join(a.out, f))
w42_dir = "/work/tts/wave42_bak" if os.path.exists("/work/tts/wave42_bak/config.json") else os.path.join(ROOT, "docs", "models")
w42 = SakhiyaONNX(w42_dir, tiny=False, threads=4)
v2 = SakhiyaONNX(a.v2, tiny=False, threads=4)
man_v2 = {}
for p in (os.path.join(ROOT, "docs", "models_v2", "manifest.json"),):
    if os.path.exists(p): man_v2 = json.load(open(p))
rows = []
for sid, grp, label, lang, text, kw in ITEMS:
    row = dict(id=sid, group=grp, label=label, lang=lang, text=text)
    for tag, eng in (("wave42", w42), ("v2", v2)):
        try:
            t0 = time.time(); w, sr = eng.say(text, lang=lang, **kw); dt = time.time() - t0
            save(os.path.join(a.out, f"{sid}_{tag}.mp3"), w, sr)
            row[tag] = f"{sid}_{tag}.mp3"; row[tag + "_sec"] = round(len(w) / sr, 2)
            print(sid, tag, "ok", row[tag + "_sec"], "s", flush=True)
        except Exception as e:
            print(sid, tag, "FAILED", repr(e)[:200], flush=True)
    rows.append(row)
man = dict(generated_at=time.time(), generated_utc=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
           v2=dict(am_run=man_v2.get("am_run"), am_step=man_v2.get("am_step"), voc_step=man_v2.get("voc_step"),
                   voc_val_l1=man_v2.get("voc_val_l1"), published_utc=man_v2.get("published_utc")),
           note="Rendered server-side from the newest published checkpoints; v2 is a PREVIEW in training and is not yet promoted over Wave 42.",
           items=rows)
json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), ensure_ascii=False, indent=1)
print("done", len(rows))
