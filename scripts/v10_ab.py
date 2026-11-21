#!/usr/bin/env python3
"""Render the same sentences with the shipped tiers and the v10 candidates (int8 graphs, as the site serves them) into
/work/tts/v10/samples for the MoLab Listen card. Also prints a quick UTMOS per model on these sentences."""
import os, sys
import numpy as np, soundfile as sf, torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, REPO)
from sakhiya.onnx_tts import SakhiyaONNX  # noqa: E402
from scripts.v2_full_eval import load_utmos  # noqa: E402
from scripts.train_v3_voc import resamp16  # noqa: E402
R = "/work/tts"; OUT = f"{R}/v10/samples"; os.makedirs(OUT, exist_ok=True)
SENT = [("1_news", "ne", "news", "सरकारले आज नयाँ बजेट सार्वजनिक गर्दै शिक्षा र स्वास्थ्य क्षेत्रमा थप लगानी गर्ने घोषणा गरेको छ।"),
        ("2_story", "ne", "storytelling", "ओहो! पहाडको टुप्पोबाट झरेको झरना कस्तो सुन्दर देखिएको छ, हगि?"),
        ("3_vedic", "sa", "vedic_chant", "ॐ असतो मा सद्गमय। तमसो मा ज्योतिर्गमय। मृत्योर्मा अमृतं गमय॥")]
MODELS = [("V1", f"{REPO}/docs/models_v2"), ("Nano-shipped", f"{REPO}/docs/models_voca"), ("Nano-v10", f"{R}/v10/stage_nano"),
          ("Atom-shipped", f"{REPO}/docs/models_sano"), ("Atom-v10", f"{R}/v10/stage_atom")]
dev = "cuda" if torch.cuda.is_available() else "cpu"; ut = load_utmos(dev)
for f in os.listdir(OUT): os.remove(os.path.join(OUT, f))
for name, d in MODELS:
    if not os.path.isdir(d): continue
    eng = SakhiyaONNX(d, tiny=True, threads=4); sc = []
    for sid, lang, style, text in SENT:
        w, sr = eng.say(text, speaker="sakhiya", lang=lang, style=style); w = np.asarray(w, dtype=np.float32)
        sf.write(f"{OUT}/{sid}__{name}.wav", w, sr, subtype="PCM_16")
        x = torch.from_numpy(w)[None].to(dev)
        with torch.no_grad(): sc.append(float(ut(resamp16(x), torch.full((1,), 16000, device=dev)).mean()))
    print(f"AB {name:14s} utmos(3 sents)={np.mean(sc):.3f} {['%.2f' % s for s in sc]}", flush=True)
