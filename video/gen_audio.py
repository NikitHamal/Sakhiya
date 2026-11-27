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
    _m = SakhiyaONNX(model_dir, tiny=True, threads=2)

PRON = S.get("pron", {})
def tts_text(t):
    for a, b in PRON.items(): t = t.replace(a, b)
    return t

def clean(w, sr=48000):
    """Studio voice clean-up (code only): high-pass, narrowband whistle-spike suppressor, split-band de-esser & burst tamer,
    unvoiced sibilant phase dispersion above 4.5 kHz, smooth 7-9 kHz shelf/low-pass, warm chest body, zero comb reverb."""
    from scipy.signal import stft, istft, sosfiltfilt
    from scipy.ndimage import uniform_filter1d, median_filter
    w = sosfiltfilt(butter(2, 80, "hp", fs=sr, output="sos"), w).astype(np.float32)
    f, _, Z = stft(w, sr, nperseg=1024, noverlap=768)
    mag = np.abs(Z).astype(np.float32)
    ph = np.angle(Z).astype(np.float32)
    P = mag ** 2

    # 1. Frequency-axis narrowband spike suppressor in 2.4k-10.5k (removes metallic whistle bins on स/श/छ/त)
    log_mag = 20.0 * np.log10(np.maximum(mag, 1e-7))
    smooth_log = uniform_filter1d(median_filter(log_mag, size=(9, 1)), size=5, axis=0)
    spike_db = np.maximum(0.0, log_mag - smooth_log - 1.0)
    bw_spike = np.clip((f - 2300.0) / 700.0, 0.0, 1.0) * np.clip((11000.0 - f) / 1000.0, 0.0, 1.0)
    G_spike = 10.0 ** (-(bw_spike[:, None] * spike_db * 0.82) / 20.0)

    # 2. Dual-band dynamic de-esser & dental-burst glare tamer referenced to low-vowel band (150-1800 Hz)
    ref_band = (f >= 150) & (f <= 1800)
    glare_band = (f >= 2600) & (f <= 5200)   # त/थ/छ/च burst & affricate glare
    sib_band = (f >= 5000) & (f <= 10200)    # स/श/छ high sibilant whistle/hiss
    Er = P[ref_band].mean(0) + 1e-9
    Eg = P[glare_band].mean(0)
    Es = P[sib_band].mean(0)
    voiced = Er > 0.015 * Er.max()

    rg = Eg / Er
    thr_g = 1.12 * np.median(rg[voiced]) if voiced.any() else 0.5
    red_g = np.clip(10.0 * np.log10(np.maximum(rg, 1e-9) / max(thr_g, 1e-6)), 0.0, 6.8)
    red_g = uniform_filter1d(np.maximum(red_g, uniform_filter1d(red_g, 3)), 3)

    rs = Es / Er
    thr_s = 1.08 * np.median(rs[voiced]) if voiced.any() else 0.3
    red_s = np.clip(10.0 * np.log10(np.maximum(rs, 1e-9) / max(thr_s, 1e-6)), 0.0, 9.5)
    red_s = uniform_filter1d(np.maximum(red_s, uniform_filter1d(red_s, 3)), 3)

    bw_g = np.clip((f - 2200.0) / 700.0, 0.0, 1.0) * np.clip((5600.0 - f) / 700.0, 0.0, 1.0)
    bw_s = np.clip((f - 4300.0) / 900.0, 0.0, 1.0) * np.clip((11500.0 - f) / 1000.0, 0.0, 1.0)
    G_dyn = 10.0 ** (-(bw_g[:, None] * red_g[None, :] + bw_s[:, None] * red_s[None, :]) / 20.0)

    # 3. Static broadcast EQ: dip metallic 3.2-4.8k (-3.2 dB), smooth shelf above 6.2k (-5.5 dB), 8th-order lowpass at 9.2k
    G_eq = 10.0 ** ((-3.2 * np.exp(-((f - 3850.0) / 750.0) ** 2) - 5.5 * np.clip((f - 6200.0) / 2600.0, 0.0, 1.0))[:, None] / 20.0)
    G_eq *= (1.0 / (1.0 + (f / 9200.0) ** 8))[:, None]

    # 4. Unvoiced sibilant phase dispersion above 4.5 kHz (replaces periodic vocoder buzz on स/श/छ with silky air)
    unvoiced_sib = np.clip((rs / max(thr_s * 1.2, 1e-6) - 0.8) / 1.5, 0.0, 1.0) * np.clip(1.0 - Er / (0.12 * Er.max() + 1e-9), 0.0, 1.0)
    rng = np.random.default_rng(42)
    rand_ph = rng.uniform(-np.pi, np.pi, size=ph.shape).astype(np.float32)
    f_mask = np.clip((f - 4500.0) / 1500.0, 0.0, 1.0)[:, None]
    alpha_ph = 0.35 * f_mask * unvoiced_sib[None, :]
    Z_out = (mag * G_spike * G_dyn * G_eq) * np.exp(1j * (ph + alpha_ph * rand_ph))

    _, y = istft(Z_out, sr, nperseg=1024, noverlap=768)
    y = y[:len(w)].astype(np.float32)
    if len(y) < len(w): y = np.pad(y, (0, len(w) - len(y)))
    # Gentle warm body around 130-260 Hz (+1.4 dB); no artificial noise-reverb comb filtering
    y = y + 0.16 * sosfiltfilt(butter(2, [130, 260], "bp", fs=sr, output="sos"), y).astype(np.float32)
    return y.astype(np.float32)

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
    for s_i, sent in enumerate(sents):
        ps_val = ps_cycle[(idx_line + s_i) % 4] + (0.015 if (s_i == 0 and len(sents) > 1) else 0.0)
        rt_val = rt_cycle[(idx_line + s_i) % 4]
        w_s, sr_out = _m.say(sent, lang="ne", style="news", rate=rt_val, pitch_shift=ps_val)
        w_s = trim_sent(w_s, int(sr_out))
        parts.append(w_s)
        if s_i + 1 < len(sents):
            # Enforce ~425 ms total inter-sentence pause on '।' (65 ms tail + 360 ms explicit silence)
            parts.append(np.zeros(int(0.360 * sr_out), dtype=np.float32))
    w_all = np.concatenate(parts) if len(parts) > 1 else parts[0]
    return lid, w_all.astype(np.float32), int(sr_out)

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
    clips, rep = {}, {}
    line_map = {lid: txt for _, lid, txt in lines}
    for lid, w, sr in res:
        w = resample_poly(w, SR, sr).astype(np.float32) if sr != SR else w.copy()
        w = clean(w, SR)
        # Gentle edge guard without re-clipping the 65 ms tail
        fi, fo = int(0.012 * SR), int(0.025 * SR)
        w[:fi] *= np.linspace(0, 1, fi, dtype=np.float32); w[-fo:] *= np.linspace(1, 0, fo, dtype=np.float32)
        active = w[np.abs(w) > 0.008]
        r = float(np.sqrt((active ** 2).mean()) if active.size else np.sqrt((w ** 2).mean())) + 1e-9
        w = w * (0.105 / r)
        pk = float(np.abs(w).max()); w = w * (0.89 / pk) if pk > 0.89 else w
        clips[lid] = w; txt = line_map[lid]; rep[lid] = check(w, SR, txt)
        wav(f"{B}/clips/{lid}.wav", w)
    json.dump(rep, open(B + "/audio_report.json", "w"), indent=1)
    t, scenes, tl = 0.0, [], []
    for sc in S["scenes"]:
        t0 = t; t += sc["pre"]; ls = []
        for l in sc["lines"]:
            d = len(clips[l["id"]]) / SR; ls.append(dict(l, t=round(t, 3), d=round(d, 3)))
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
