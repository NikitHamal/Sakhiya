#!/usr/bin/env python3
"""Build docs/cartridges.json (< 20 KB): hand-designed Expression Cartridges.
Each cartridge = Studio parameters (apply today to every shipped tier) + an 8-point, 3-channel phrase contour
(f0_delta in semitones, energy_delta in dB-like units, dur_ratio) for the v6 contour hook (50 %-dropout conditioning).
These are crafted shapes, NOT extracted from recordings (expressive data mining is a separate owner session)."""
import json, math, os

def shape(kind, n=8):
    t = [i / (n - 1) for i in range(n)]
    f = {
        "flat": lambda x: 0.0, "fall": lambda x: -2.5 * x, "rise": lambda x: 3.0 * x ** 2,
        "arch": lambda x: 2.2 * math.sin(math.pi * x) - 0.8 * x, "wow": lambda x: 4.0 * math.sin(math.pi * x) ** 0.8,
        "calm": lambda x: -0.6 * x, "chant": lambda x: 0.4 * math.sin(2 * math.pi * x), "story": lambda x: 1.2 * math.sin(math.pi * x) - 1.0 * x,
    }[kind]
    return [round(f(x), 2) for x in t]

def cart(id, label, style, desc, f0, en, dur, **params):
    return dict(id=id, label=label, style=style, desc=desc, params={k: round(v, 3) for k, v in params.items()},
                contour=dict(f0=shape(f0), en=[round(en * (1 - 0.5 * abs(2 * i / 7 - 1)), 2) for i in range(8)], dur=[round(dur[0] + (dur[1] - dur[0]) * i / 7, 2) for i in range(8)]))

C = [
    cart("news_crisp", "News Crisp", "news", "Clear broadcast reading, short even pauses.", "fall", 0.6, (-0.05, 0.12), rate=1.04, pause_scale=0.9, pitch_scale=0.95, energy_scale=1.05),
    cart("news_headline", "News Headline", "news", "Slightly higher, faster, firm phrase endings.", "arch", 0.8, (-0.08, 0.15), rate=1.10, pause_scale=0.8, pitch_scale=1.05, energy_scale=1.12),
    cart("folk_story", "Warm Folk Story", "storytelling", "Slow, warm grandmother-style narration.", "story", 0.4, (0.0, 0.2), rate=0.90, pause_scale=1.25, pitch_scale=1.20, energy_scale=1.0),
    cart("gentle_pause_story", "Gentle Pause Storytelling", "storytelling", "Longer breaths at commas, soft endings.", "calm", 0.2, (0.05, 0.3), rate=0.88, pause_scale=1.6, pitch_scale=1.10, energy_scale=0.92),
    cart("excited_wow", "Excited वाह!", "storytelling", "Bright rise on exclamations, lively pace.", "wow", 1.2, (-0.12, 0.1), rate=1.08, pause_scale=0.85, pitch_scale=1.50, energy_scale=1.25),
    cart("curious_question", "Curious Question", "storytelling", "Rising final contour for questions.", "rise", 0.7, (0.0, 0.25), rate=0.98, pause_scale=1.0, pitch_scale=1.35, energy_scale=1.05),
    cart("calm_meditation", "Calm Meditation", "storytelling", "Low, slow, spacious.", "calm", -0.3, (0.1, 0.35), rate=0.80, pause_scale=1.9, pitch_scale=0.90, energy_scale=0.85),
    cart("friendly_chat", "Friendly Chat", "storytelling", "Casual conversational lift.", "arch", 0.5, (-0.05, 0.1), rate=1.05, pause_scale=0.9, pitch_scale=1.25, energy_scale=1.0),
    cart("soft_bedtime", "Soft Bedtime", "storytelling", "Very soft, drifting endings.", "calm", -0.5, (0.1, 0.4), rate=0.78, pause_scale=2.0, pitch_scale=0.95, energy_scale=0.80),
    cart("serious_report", "Serious Report", "news", "Low, steady, weighty.", "fall", 0.3, (0.0, 0.1), rate=0.96, pause_scale=1.1, pitch_scale=0.85, energy_scale=1.0),
    cart("gayatri_72", "Gayatri Laya 72 BPM", "vedic_chant", "Three 8-syllable pādas on the tonic with a Re lift.", "chant", 0.5, (0.0, 0.3), rate=1.0, pause_scale=1.2, pitch_scale=1.0, svara=1.0, bpm=72),
    cart("rigveda_svara", "Rigveda Svara", "vedic_chant", "Udātta / anudātta / svarita contour, 66 BPM.", "chant", 0.4, (0.0, 0.25), rate=0.95, pause_scale=1.3, pitch_scale=1.0, svara=1.5, bpm=66),
    cart("gita_anushtubh", "Gita Anushtubh", "vedic_chant", "Classical shloka cadence, 4 × 8, 80 BPM.", "chant", 0.3, (0.0, 0.2), rate=1.0, pause_scale=1.1, pitch_scale=1.0, svara=1.0, bpm=80),
    cart("om_resonant", "Resonant ॐ", "vedic_chant", "Long sustained ॐ with a slow 5 Hz gamaka, 56 BPM.", "chant", 0.8, (0.2, 0.5), rate=0.90, pause_scale=1.4, pitch_scale=1.0, svara=1.0, bpm=56),
    cart("stotra_fast", "Stotra Brisk", "vedic_chant", "Brisk devotional recitation, 100 BPM.", "chant", 0.4, (-0.1, 0.1), rate=1.05, pause_scale=0.9, pitch_scale=1.0, svara=0.8, bpm=100),
    cart("mantra_slow_deep", "Mantra Slow & Deep", "vedic_chant", "Deep, slow japa-style, 48 BPM.", "calm", 0.5, (0.15, 0.45), rate=0.85, pause_scale=1.6, pitch_scale=0.95, svara=1.2, bpm=48),
]
out = dict(version=1, note="Hand-designed contours (8 points each: f0 in semitones, energy, duration ratio). Params drive the Studio sliders today; contours feed the v6 contour hook.", cartridges=C)
p = os.path.join(os.path.dirname(__file__), "..", "docs", "cartridges.json")
json.dump(out, open(p, "w"), ensure_ascii=False, separators=(",", ":"))
print(len(C), "cartridges", os.path.getsize(p), "bytes")
