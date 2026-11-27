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
    """Voice clean-up (code only): high-pass, split-band de-esser, tame the 3.5-4.5k 'metal' band, roll off hiss/air, tiny room."""
    from scipy.signal import stft, istft, sosfiltfilt
    w = sosfiltfilt(butter(2, 75, "hp", fs=sr, output="sos"), w).astype(np.float32)
    f, _, Z = stft(w, sr, nperseg=1024, noverlap=768)
    P = np.abs(Z) ** 2
    sib = (f >= 4800) & (f <= 11500); ref = (f >= 200) & (f <= 3500)
    Es = P[sib].mean(0); Er = P[ref].mean(0) + 1e-9; r = Es / Er
    voiced = Er > 0.02 * Er.max()
    thr = 1.5 * np.median(r[voiced]) if voiced.any() else 1.0
    red = np.clip(10 * np.log10(np.maximum(r, 1e-9) / thr), 0, 9)      # dB to pull down per frame
    red = np.maximum(red, np.convolve(red, np.ones(3) / 3, "same"))
    red = np.convolve(red, np.ones(3) / 3, "same")
    G = np.ones_like(P)
    bw = np.clip((f - 4000) / 1500, 0, 1) * np.clip((13500 - f) / 1500, 0, 1)   # soft band edges
    G = 10 ** (-(bw[:, None] * red[None, :]) / 20)
    # static tone shaping: dip metallic 3.4-4.6k (-2.2 dB), air shelf above 7k (-4 dB), steep lowpass ~10.5k
    G *= 10 ** ((-2.2 * np.exp(-((f - 3900) / 650) ** 2) - 2.5 * np.clip((f - 7000) / 2500, 0, 1))[:, None] / 20)
    G *= (1 / (1 + (f / 10500) ** 8))[:, None]
    _, y = istft(Z * G, sr, nperseg=1024, noverlap=768); y = y[:len(w)].astype(np.float32)
    if len(y) < len(w): y = np.pad(y, (0, len(w) - len(y)))
    # gentle body: +1.2 dB around 180 Hz so the voice sounds less thin/robotic
    y = y + 0.14 * sosfiltfilt(butter(2, [130, 260], "bp", fs=sr, output="sos"), y).astype(np.float32)
    rng = np.random.default_rng(7); n = int(0.16 * sr); t = np.arange(n) / sr
    ir = rng.standard_normal(n) * np.exp(-t / 0.045); ir[0] = 0; ir = sosfilt(butter(2, 5500, "lp", fs=sr, output="sos"), ir)
    ir /= np.abs(ir).sum() + 1e-9
    wet = fftconvolve(y, ir * 3.0)[:len(y)]
    return (y + 0.07 * wet).astype(np.float32)

def _synth(job):
    lid, text = job
    text = tts_text(text)
    w, sr = _m.say(text, lang="ne", style="news")
    return lid, np.asarray(w, dtype=np.float32), int(sr)

def trim(w, sr):
    a = np.abs(w); thr = max(0.012, 0.02 * a.max()); idx = np.where(a > thr)[0]
    if not idx.size: return w
    p = int(0.05 * sr); w = w[max(0, idx[0] - p): idx[-1] + p]
    f = int(0.012 * sr); w[:f] *= np.linspace(0, 1, f); w[-f:] *= np.linspace(1, 0, f); return w

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
    lines = [(l["id"], l["ne"]) for sc in S["scenes"] for l in sc["lines"]]
    with Pool(min(10, len(lines)), initializer=_init) as p: res = p.map(_synth, lines)
    clips, rep = {}, {}
    for lid, w, sr in res:
        w = trim(w.copy(), sr); w = resample_poly(w, SR, sr).astype(np.float32) if sr != SR else w
        w = trim(clean(w, SR), SR)
        r = float(np.sqrt((w ** 2).mean()) + 1e-9); w = w * (0.095 / r)
        pk = float(np.abs(w).max()); w = w * (0.89 / pk) if pk > 0.89 else w
        clips[lid] = w; txt = dict(lines)[lid]; rep[lid] = check(w, SR, txt)
        wav(f"{B}/clips/{lid}.wav", w)
    json.dump(rep, open(B + "/audio_report.json", "w"), indent=1)
    t, scenes, tl = 0.0, [], []
    for sc in S["scenes"]:
        t0 = t; t += sc["pre"]; ls = []
        for l in sc["lines"]:
            d = len(clips[l["id"]]) / SR; ls.append(dict(l, t=round(t, 3), d=round(d, 3)))
            t += d + (0.42 if l["ne"].count("।") > 1 else 0.3)
        t = t - 0.3 + sc["post"]; scenes.append(dict({k: v for k, v in sc.items() if k != "lines"}, t0=round(t0, 3), t1=round(t, 3), lines=ls))
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
