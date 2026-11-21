#!/usr/bin/env python3
"""Render the 5 fixed v13 listening sentences with one model dir (int8 graphs, as the site serves them)."""
import argparse, os, sys
import numpy as np, soundfile as sf
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, REPO)
from sakhiya.onnx_tts import SakhiyaONNX  # noqa: E402
SENT = [("1_news", "ne", "news", "सरकारले आज नयाँ बजेट सार्वजनिक गर्दै शिक्षा र स्वास्थ्य क्षेत्रमा थप लगानी गर्ने घोषणा गरेको छ।"),
        ("2_story", "ne", "storytelling", "धेरै पहिले एउटा सानो गाउँमा एउटी बुढी आमा बस्नुहुन्थ्यो, उहाँको एउटा प्यारो बाख्रो थियो।"),
        ("3_expressive", "ne", "expressive", "वाह! के अद्भुत दृश्य हो यो! माछापुच्छ्रे हिमाल बिहानको घाममा चम्किरहेको छ!"),
        ("4_conv", "ne", "conversational", "अनि तिमी भोलि कति बजे आउँछौ? म चाहिँ बिहानै तयार भएर बस्छु है।"),
        ("5_vedic", "sa", "vedic_chant", "ॐ असतो मा सद्गमय। तमसो मा ज्योतिर्गमय। मृत्योर्मा अमृतं गमय॥")]
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--name", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); eng = SakhiyaONNX(a.model, tiny=True, threads=4)
for sid, lang, style, text in SENT:
    w, sr = eng.say(text, speaker="sakhiya", lang=lang, style=style)
    sf.write(f"{a.out}/{sid}__{a.name}.wav", np.asarray(w, dtype=np.float32), sr, subtype="PCM_16")
print("rendered", a.name)
