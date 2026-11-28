#!/usr/bin/env python3
"""NEBians News Ep.1 audio: Sakhiya TTS (docs/models_sakhiya_one_vocos), `news` style. Voice only, no music.
Outputs (video/build/): clips/*.wav, timeline.json, env.json, voice.wav, mix.wav (-16 LUFS), audio_report.json"""
import json, os, sys, wave
from multiprocessing import Pool
import numpy as np
from scipy.signal import resample_poly, sosfilt, butter, fftconvolve
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
B = os.path.join(ROOT, "video", "build"); os.makedirs(B + "/clips", exist_ok=True)
SR, FPS = 48000, 30
S = json.load(open(os.path.join(ROOT, "video", "script.json")))
_m = None

def _init():
    global _m
    sys.path.insert(0, ROOT)
    from sakhiya.onnx_tts import SakhiyaONNX
    model_dir = os.environ.get("SAKHIYA_MODEL_DIR", os.path.join(ROOT, "docs", "models_sakhiya_one_vocos"))
    _m = SakhiyaONNX(model_dir, tiny=bool(int(os.environ.get("SAKHIYA_TINY", "0"))), threads=2)

PRON = S.get("pron", {})
def tts_text(t):
    for a, b in PRON.items(): t = t.replace(a, b)
    return t

def _peaking_sos(f0, gain_db, q, sr):
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2.0 * q)
    cw = np.cos(w0)
    b0, b1, b2 = 1.0 + alpha * A, -2.0 * cw, 1.0 - alpha * A
    a0, a1, a2 = 1.0 + alpha / A, -2.0 * cw, 1.0 - alpha / A
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)

def _highshelf_sos(f0, gain_db, q, sr):
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / sr
    cw, sw = np.cos(w0), np.sin(w0)
    sqA = 2.0 * np.sqrt(A) * (sw / (2.0 * q))
    b0 = A * ((A + 1.0) + (A - 1.0) * cw + sqA)
    b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cw)
    b2 = A * ((A + 1.0) + (A - 1.0) * cw - sqA)
    a0 = (A + 1.0) - (A - 1.0) * cw + sqA
    a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cw)
    a2 = (A + 1.0) - (A - 1.0) * cw - sqA
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)

def clean(w, sr=48000):
    """Phase-clean broadcast studio chain: HPF, time-domain crossover de-esser/burst tamer, zero-phase EQ, subtle tube warmth & studio early reflection."""
    from scipy.signal import sosfiltfilt
    from scipy.ndimage import uniform_filter1d
    w = sosfiltfilt(butter(2, 75, "hp", fs=sr, output="sos"), w).astype(np.float32)

    # 1. Time-domain 4th-order split-band de-esser & dental burst tamer (zero STFT random-phase fuzz)
    hi = sosfiltfilt(butter(4, 4100, "hp", fs=sr, output="sos"), w).astype(np.float32)
    ref = sosfiltfilt(butter(2, [140, 1800], "bp", fs=sr, output="sos"), w).astype(np.float32)
    e_hi = np.sqrt(uniform_filter1d(hi * hi, max(1, int(0.004 * sr))) + 1e-10)
    e_ref = np.sqrt(uniform_filter1d(ref * ref, max(1, int(0.012 * sr))) + 1e-10)
    voiced = e_ref > 0.04 * float(np.max(e_ref))
    ratio_base = float(np.median(e_hi[voiced] / np.maximum(e_ref[voiced], 1e-6))) if voiced.any() else 0.18
    excess = uniform_filter1d(np.maximum(1.0, e_hi / np.maximum(e_ref * max(ratio_base * 1.45, 0.16), 0.012)), max(1, int(0.008 * sr)))
    g_hi = np.clip(1.0 / (1.0 + 0.68 * np.maximum(0.0, excess - 1.0) ** 1.15), 0.46, 1.0).astype(np.float32)
    y = w - hi * (1.0 - g_hi)

    # 2. Zero-phase parametric broadcast EQ (removes boxy 1k & metallic 3.45k/5.6k glare, preserves warm body & natural air)
    eq_sos = np.vstack([
        _peaking_sos(195.0, +1.8, 0.85, sr),
        _peaking_sos(1050.0, -1.6, 1.25, sr),
        _peaking_sos(3450.0, -3.1, 1.15, sr),
        _peaking_sos(5600.0, -2.2, 1.30, sr),
        _highshelf_sos(6800.0, -3.5, 0.70, sr),
    ])
    y = sosfiltfilt(eq_sos, y).astype(np.float32)

    # 3. Subtle analog soft saturation + decorrelated studio-booth early-reflection tail (2-22 ms, -33 dB, 220-4800 Hz)
    y = (0.82 * y + 0.18 * (np.tanh(y * 1.35) / 1.35)).astype(np.float32)
    rng = np.random.default_rng(7)
    ir_len = int(0.022 * sr)
    ir = rng.standard_normal(ir_len).astype(np.float32) * np.exp(-(np.arange(ir_len, dtype=np.float32) / sr) / 0.0055)
    ir[:int(0.002 * sr)] = 0.0
    ir = sosfiltfilt(butter(2, [220, 4800], "bp", fs=sr, output="sos"), ir).astype(np.float32)
    ir *= (0.022 / max(float(np.sqrt((ir * ir).sum())), 1e-6))
    return (y + fftconvolve(y, ir, mode="full")[:len(y)].astype(np.float32)).astype(np.float32)

def trim_sent(w, sr):
    """Onset-safe & tail-safe trimmer: preserves quiet final syllables (तर्फ, नमस्ते) with a 28 ms raised-cosine fade-out and 65 ms silence tail."""
    w = np.asarray(w, dtype=np.float32).copy()
    a = np.abs(w)
    if not a.size or a.max() < 1e-5:
        return w
    pk = float(a.max())
    idx_lead = np.where(a > max(0.010, 0.018 * pk))[0]
    idx_tail = np.where(a > max(0.003, 0.006 * pk))[0]
    if not idx_lead.size or not idx_tail.size:
        return w
    i0 = max(0, int(idx_lead[0]) - int(0.024 * sr))
    i1 = min(len(w), int(idx_tail[-1]) + int(0.085 * sr))
    w = w[i0:i1]
    fi = min(len(w) // 3, int(0.014 * sr))
    fo = min(len(w) // 3, int(0.028 * sr))
    if fi > 1:
        w[:fi] *= (0.5 - 0.5 * np.cos(np.pi * np.arange(fi, dtype=np.float32) / fi))
    if fo > 1:
        w[-fo:] *= (0.5 + 0.5 * np.cos(np.pi * np.arange(fo, dtype=np.float32) / fo))
    return np.concatenate([np.zeros(int(0.020 * sr), dtype=np.float32), w, np.zeros(int(0.065 * sr), dtype=np.float32)])

def _synth(job):
    import re
    idx_line, lid, text = job
    text = tts_text(text)
    sents = [s.strip() for s in re.split(r"(?<=[।॥?!])\s+", text.strip()) if s.strip()] or [text.strip()]
    # Subtle sentence-to-sentence pitch/rate variation so consecutive news lines never sound looped
    ps_cycle = [-0.03, -0.01, -0.04, -0.015]
    rt_cycle = [0.96, 0.97, 0.95, 0.965]
    parts = []
    sr_out = 24000
    s_cuts = []
    cur_samples = 0
    for s_i, sent in enumerate(sents):
        ps_val = ps_cycle[(idx_line + s_i) % 4] + (0.015 if (s_i == 0 and len(sents) > 1) else 0.0)
        rt_val = rt_cycle[(idx_line + s_i) % 4]
        w_s, sr_out = _m.say(sent, lang="ne", style="news", rate=rt_val, pitch_shift=ps_val)
        w_s = trim_sent(w_s, int(sr_out))
        parts.append(w_s)
        cur_samples += len(w_s)
        if s_i + 1 < len(sents):
            # Enforce ~425 ms total inter-sentence pause on '।' (65 ms tail + 360 ms explicit silence)
            p_len = int(0.360 * sr_out)
            s_cuts.append(cur_samples + int(0.220 * sr_out))
            parts.append(np.zeros(p_len, dtype=np.float32))
            cur_samples += p_len
    w_all = np.concatenate(parts) if len(parts) > 1 else parts[0]
    s_fracs = [round(c / max(len(w_all), 1), 4) for c in s_cuts] + [1.0]
    return lid, w_all.astype(np.float32), int(sr_out), s_fracs

def check(w, sr, text):
    a = np.abs(w); fr = int(0.02 * sr); n = len(a) // fr
    rms = np.sqrt((a[:n * fr].reshape(n, fr) ** 2).mean(1)); sil = rms < 0.006
    run = best = 0
    for s in sil[2:-2]:
        run = run + 1 if s else 0; best = max(best, run)
    words = len(text.split()); dur = len(w) / sr
    return dict(dur=round(dur, 2), words_per_s=round(words / dur, 2), clip_frac=float((a > 0.985).mean()),
                max_pause_s=round(best * 0.02, 2), peak=round(float(a.max()), 3))

def main():
    lines = [(idx, l["id"], l["ne"]) for idx, (sc, l) in enumerate((sc, l) for sc in S["scenes"] for l in sc["lines"])]
    with Pool(min(4, len(lines)), initializer=_init) as p: res = p.map(_synth, lines)
    clips, s_cuts_map, rep = {}, {}, {}
    line_map = {lid: txt for _, lid, txt in lines}
    for lid, w, sr, s_fracs in res:
        w = resample_poly(w, SR, sr).astype(np.float32) if sr != SR else w.copy()
        w = clean(w, SR)
        # Gentle edge guard without re-clipping the 65 ms tail
        fi, fo = int(0.012 * SR), int(0.025 * SR)
        w[:fi] *= np.linspace(0, 1, fi, dtype=np.float32); w[-fo:] *= np.linspace(1, 0, fo, dtype=np.float32)
        active = w[np.abs(w) > 0.008]
        r = float(np.sqrt((active ** 2).mean()) if active.size else np.sqrt((w ** 2).mean())) + 1e-9
        w = w * (0.105 / r)
        pk = float(np.abs(w).max()); w = w * (0.89 / pk) if pk > 0.89 else w
        clips[lid] = w; s_cuts_map[lid] = s_fracs; txt = line_map[lid]; rep[lid] = check(w, SR, txt)
        wav(f"{B}/clips/{lid}.wav", w)
    json.dump(rep, open(B + "/audio_report.json", "w"), indent=1)
    t, scenes, tl = 0.0, [], []
    for sc in S["scenes"]:
        t0 = t; t += sc["pre"]; ls = []
        for l in sc["lines"]:
            d = len(clips[l["id"]]) / SR; ls.append(dict(l, t=round(t, 3), d=round(d, 3), s_cuts=s_cuts_map[l["id"]]))
            t += d + (0.44 if l["ne"].count("।") > 1 else 0.36)
        t = t - 0.36 + sc["post"]; scenes.append(dict({k: v for k, v in sc.items() if k != "lines"}, t0=round(t0, 3), t1=round(t, 3), lines=ls))
    total = t; tl = dict(total=round(total, 3), fps=FPS, scenes=scenes, title=S["title"], date=S["date"])
    json.dump(tl, open(B + "/timeline.json", "w"), ensure_ascii=False, indent=1)
    voice = np.zeros(int(total * SR) + SR, np.float32)
    for sc in scenes:
        for l in sc["lines"]:
            i = int(l["t"] * SR); c = clips[l["id"]]; voice[i:i + len(c)] += c
    wav(B + "/voice.wav", voice)
    env(voice, total); master(B)
    print("TOTAL", round(total, 1), "s", "flags:", {k: v for k, v in rep.items() if v["max_pause_s"] > 0.9 or v["clip_frac"] > 0.001 or v["words_per_s"] < 1.2 or v["words_per_s"] > 4.2})

def master(B):
    import subprocess
    subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-i", B + "/voice.wav", "-af", "acompressor=threshold=-20dB:ratio=2.2:attack=8:release=140:makeup=1,loudnorm=I=-16:TP=-1.5:LRA=7,aresample=48000", "-ac", "2", B + "/mix.wav"])

def env(v, total):
    n = int(total * FPS) + 2; hop = SR // FPS; win = 2048; out = []
    for i in range(n):
        s = v[max(0, i * hop - win // 2): i * hop + win // 2]
        if len(s) < win: s = np.pad(s, (0, win - len(s)))
        sp = np.abs(np.fft.rfft(s * np.hanning(win))); e = np.edges if False else None
        edges = np.geomspace(2, 360, 33).astype(int)
        bands = [float(sp[edges[k]:max(edges[k] + 1, edges[k + 1])].mean()) for k in range(32)]
        out.append([round(float(np.sqrt((s ** 2).mean())), 4)] + [round(b, 3) for b in bands])
    a = np.array(out); a[:, 1:] /= max(1e-6, np.percentile(a[:, 1:], 99)); a[:, 0] /= max(1e-6, np.percentile(a[:, 0], 99))
    json.dump(np.clip(a, 0, 1.2).round(3).tolist(), open(B + "/env.json", "w"))

def wav(path, w):
    x = np.clip(w, -1, 1); w_ = wave.open(path, "wb"); w_.setnchannels(1); w_.setsampwidth(2); w_.setframerate(SR)
    w_.writeframes((x * 32767).astype(np.int16).tobytes()); w_.close()

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__)); main()
