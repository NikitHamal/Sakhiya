#!/usr/bin/env python3
"""Pre-render the Ampixa NepTTS-Bench prompts (docs/data/ampixa_prompts.json) with every promoted Sakhiya tier on the
GPU box (CPU INT8 ONNX, 1 thread/worker) and publish them as static, cleaned MP3s for GitHub Pages.

Clean-up chain applied to every clip (see `clean()`):  DC/rumble high-pass -> stationary denoise (min-statistics
Wiener gate, smoothed in time+freq) -> de-shutter (fills sub-25 ms envelope drop-outs) -> STFT de-click -> loudness
norm (-19 LUFS, -1 dBFS soft ceiling) -> 6 ms fades.   Per-clip before/after metrics land in docs/samples/ampixa/index.json.

usage:  python3 scripts/ampixa_render.py --tiers v1,t30,t15,micro,pico [--limit N] [--workers 18]
Models: v1/micro/pico come from docs/models_*; t30/t15 are downloaded once from HF (NikitHamal/sakhiya-pro-v1/students/*).
Nothing is submitted anywhere; no secrets are written to disk by this script (HF token is only read from env/HF_TOKEN file)."""
import argparse, json, os, shutil, subprocess, sys, time
from multiprocessing import Pool
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

TIERS = {
    "v1":    dict(label="Sakhiya-V1 (flagship)", dir="docs/models_v2", tiny=True, mb=15.981, utmos=3.957),
    "t30":   dict(label="Pro-v3 T30", hf="students/t30", mb=15.853, utmos=4.081),
    "t15":   dict(label="Pro-v3 T15", hf="students/t15", mb=14.36, utmos=3.882),
    "micro": dict(label="V2-Micro", dir="docs/models_micro", tiny=True, mb=3.975, utmos=3.297),
    "pico":  dict(label="V2-Pico", dir="docs/models_pico", tiny=True, mb=1.369, utmos=2.708),
}
WORK = os.environ.get("AMPIXA_WORK", "/work/tts/ampixa")


def style_of(p):
    c = p.get("category")
    return "expressive" if c == "emotion" else ("storytelling" if c == "natural_speech" else "news")


# ---------------------------------------------------------------- cleaning
def _stft(w, n=1024, hop=256):
    from scipy.signal import stft
    return stft(w, nperseg=n, noverlap=n - hop, window="hann", boundary="even", padded=True)[2]


def _istft(S, n=1024, hop=256, length=None):
    from scipy.signal import istft
    y = istft(S, nperseg=n, noverlap=n - hop, window="hann", boundary=True)[1]
    y = y.astype(np.float32)
    if length is not None:
        y = y[:length] if len(y) >= length else np.pad(y, (0, length - len(y)))
    return y


def noise_floor_db(w, sr):
    fr = int(0.02 * sr)
    n = len(w) // fr
    if n < 4:
        return -90.0
    rms = np.sqrt(np.mean(w[:n * fr].reshape(n, fr) ** 2, axis=1) + 1e-12)
    return float(20 * np.log10(np.percentile(rms, 10) + 1e-9))


def click_count(w, n_fft=512, hop=256):
    if len(w) < n_fft * 2:
        return 0
    win = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    fr = np.lib.stride_tricks.sliding_window_view(np.pad(w, (n_fft // 2, n_fft // 2), mode="reflect"), n_fft)[::hop] * win
    spec = np.abs(np.fft.rfft(fr, axis=1)).T
    flux = np.sqrt(np.mean(np.maximum(0.0, np.diff(np.log(spec + 1e-4), axis=1)) ** 2, axis=0))
    z = (flux - np.median(flux)) / (np.std(flux) + 1e-5)
    return int(np.sum(z > 7.5))


def _env(w, sr, ms=5):
    h = max(1, int(sr * ms / 1000))
    n = len(w) // h
    return np.sqrt(np.mean(w[:n * h].reshape(n, h) ** 2, axis=1) + 1e-12), h


def shutter_dips(w, sr):
    """Count short (<=25 ms) envelope drop-outs of >9 dB inside voiced speech = 'shutter' / chatter artefacts."""
    from scipy.ndimage import median_filter, maximum_filter1d
    e, h = _env(w, sr)
    if len(e) < 30:
        return 0, e, h, np.zeros(0, bool)
    loc = median_filter(e, size=17)                      # ~85 ms context
    ctx = maximum_filter1d(loc, size=3)
    dip = (e < 0.35 * np.maximum(loc, 1e-6)) & (loc > 0.02 * e.max()) & (ctx > 0.04 * e.max())
    # group into runs, keep only short ones (<= 5 frames = 25 ms) -> real shutter, not phrase pauses
    out = np.zeros_like(dip)
    i, cnt = 0, 0
    while i < len(dip):
        if dip[i]:
            j = i
            while j < len(dip) and dip[j]:
                j += 1
            if j - i <= 5:
                out[i:j] = True
                cnt += 1
            i = j
        else:
            i += 1
    return cnt, e, h, out


def deshutter(w, sr):
    cnt, e, h, mask = shutter_dips(w, sr)
    if cnt == 0:
        return w, 0
    from scipy.ndimage import median_filter, gaussian_filter1d
    loc = median_filter(e, size=17)
    gain = np.ones(len(e), np.float32)
    gain[mask] = np.clip(0.7 * loc[mask] / np.maximum(e[mask], 1e-6), 1.0, 2.0)   # <= +6 dB, only inside the dip
    gain = gaussian_filter1d(gain, 1.2)
    g = np.repeat(gain, h)
    g = np.pad(g, (0, max(0, len(w) - len(g))), constant_values=1.0)[:len(w)]
    return (w * g).astype(np.float32), cnt


def denoise(w, sr, strength=1.25, floor_db=-16.0):
    from scipy.ndimage import uniform_filter
    S = _stft(w)
    mag = np.abs(S)
    pw = mag ** 2
    n_pw = np.percentile(pw, 12, axis=1, keepdims=True) * 1.4          # min-statistics stationary noise estimate
    snr = pw / (n_pw * strength + 1e-12)
    g = np.maximum(1.0 - 1.0 / np.maximum(snr, 1e-6), 0.0)
    g = np.sqrt(np.maximum(g, 10 ** (floor_db / 10)))
    g = uniform_filter(g, size=(3, 3))                                  # kills musical-noise islands
    return _istft(S * g, length=len(w))


def clean(w, sr):
    from scipy.signal import butter, sosfiltfilt
    from sakhiya.onnx_tts import _declick_np
    w = np.asarray(w, np.float32)
    m0 = dict(noise_db=noise_floor_db(w, sr), clicks=click_count(w), shutter=shutter_dips(w, sr)[0])
    w = w - float(np.mean(w))
    w = sosfiltfilt(butter(2, 35, "highpass", fs=sr, output="sos"), w).astype(np.float32)
    w = denoise(w, sr)
    w, _ = deshutter(w, sr)
    w = _declick_np(w)
    try:
        import pyloudnorm as pyln
        if len(w) > int(0.45 * sr):
            lu = pyln.Meter(sr).integrated_loudness(w.astype(np.float64))
            if np.isfinite(lu):
                w = (w * 10 ** ((-19.0 - lu) / 20)).astype(np.float32)
    except Exception:
        pass
    pk = float(np.max(np.abs(w))) + 1e-9
    if pk > 0.89:                                                       # soft ceiling, no hard clipping
        w = (0.89 * np.tanh(w / 0.89 * 1.0)).astype(np.float32) if pk > 1.0 else (w * (0.89 / pk)).astype(np.float32)
    f = int(0.006 * sr)
    if len(w) > 2 * f:
        r = (0.5 - 0.5 * np.cos(np.pi * np.arange(f) / f)).astype(np.float32)
        w[:f] *= r
        w[-f:] *= r[::-1]
    m1 = dict(noise_db=noise_floor_db(w, sr), clicks=click_count(w), shutter=shutter_dips(w, sr)[0])
    if m1["clicks"] > m0["clicks"]:                                     # never leave a clip clickier than the raw render
        w = _declick_np(w)
        m1["clicks"] = click_count(w)
    return w, m0, m1


# ---------------------------------------------------------------- rendering
_ENG = {}


def _engine(tier, mdir):
    if tier not in _ENG:
        from sakhiya.onnx_tts import SakhiyaONNX
        _ENG[tier] = SakhiyaONNX(mdir, tiny=True, threads=1)
    return _ENG[tier]


def render_one(job):
    tier, mdir, p, out_dir = job
    import soundfile as sf
    sid = p["sent_id"]
    mp3 = os.path.join(out_dir, sid + ".mp3")
    if os.path.exists(mp3) and os.path.getsize(mp3) > 800:
        return tier, sid, None
    try:
        eng = _engine(tier, mdir)
        t0 = time.time()
        w, sr = eng.say(p["text_devanagari"], speaker="sakhiya", lang="ne", style=style_of(p))
        rt = time.time() - t0
        w = np.asarray(w, np.float32)
        if not np.all(np.isfinite(w)) or len(w) < sr // 5:
            return tier, sid, dict(error="empty/nan")
        c, m0, m1 = clean(w, sr)
        tmp = mp3 + ".tmp.wav"
        sf.write(tmp, c, sr, subtype="PCM_16")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-ac", "1", "-codec:a", "libmp3lame", "-b:a", "40k", mp3], check=True)
        os.remove(tmp)
        dur = len(c) / sr
        return tier, sid, dict(dur=round(dur, 2), rtf=round(rt / dur, 4), before=m0, after=m1, kb=round(os.path.getsize(mp3) / 1024, 1))
    except Exception as e:  # noqa
        return tier, sid, dict(error=str(e)[:160])


def ensure_models(tiers):
    dirs = {}
    for t in tiers:
        spec = TIERS[t]
        if "dir" in spec:
            dirs[t] = os.path.join(ROOT, spec["dir"])
            continue
        d = os.path.join(WORK, "models", t)
        os.makedirs(d, exist_ok=True)
        onnx_p = os.path.join(d, "model.int8.onnx")
        if not os.path.exists(onnx_p):
            from huggingface_hub import hf_hub_download
            tok = os.environ.get("HF_TOKEN") or (open("/tmp/hf_token").read().strip() if os.path.exists("/tmp/hf_token") else None)
            fp = hf_hub_download("NikitHamal/sakhiya-pro-v1", spec["hf"] + "/model.int8.onnx", token=tok)
            shutil.copy(fp, onnx_p)
        shutil.copy(os.path.join(ROOT, "docs", "models_micro", "config.json"), os.path.join(d, "config.json"))
        dirs[t] = d
    return dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default="v1,t30,t15,micro,pico")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "samples", "ampixa"))
    a = ap.parse_args()
    tiers = a.tiers.split(",")
    prompts = json.load(open(os.path.join(ROOT, "docs", "data", "ampixa_prompts.json")))
    if a.limit:
        prompts = prompts[:a.limit]
    dirs = ensure_models(tiers)
    jobs = []
    for t in tiers:
        od = os.path.join(a.out, t)
        os.makedirs(od, exist_ok=True)
        jobs += [(t, dirs[t], p, od) for p in prompts]
    t0 = time.time()
    res = {t: {} for t in tiers}
    with Pool(a.workers) as pool:
        for i, (t, sid, r) in enumerate(pool.imap_unordered(render_one, jobs, chunksize=4)):
            if r is not None:
                res[t][sid] = r
            if i % 100 == 0:
                print(f"[{i}/{len(jobs)}] {time.time() - t0:.0f}s", flush=True)
    idx_p = os.path.join(a.out, "index.json")
    old = json.load(open(idx_p)) if os.path.exists(idx_p) else {}
    clips = old.get("clips", {})
    for t in tiers:
        clips.setdefault(t, {}).update(res[t])
    summ = {}
    for t in tiers:
        ok = [v for v in clips[t].values() if "error" not in v]
        bad = [k for k, v in clips[t].items() if "error" in v]
        if ok:
            summ[t] = dict(n=len(ok), errors=bad, rtf_mean=round(float(np.mean([v["rtf"] for v in ok])), 4),
                           noise_db_before=round(float(np.mean([v["before"]["noise_db"] for v in ok])), 1),
                           noise_db_after=round(float(np.mean([v["after"]["noise_db"] for v in ok])), 1),
                           clicks_before=int(sum(v["before"]["clicks"] for v in ok)), clicks_after=int(sum(v["after"]["clicks"] for v in ok)),
                           shutter_before=int(sum(v["before"]["shutter"] for v in ok)), shutter_after=int(sum(v["after"]["shutter"] for v in ok)),
                           mb_audio=round(sum(v["kb"] for v in ok) / 1024, 2))
    meta = {t: {k: v for k, v in TIERS[t].items() if k in ("label", "mb", "utmos")} for t in TIERS}
    json.dump(dict(generated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), prompts=len(prompts), bitrate="mp3 40k mono",
                   cleanup="hp35 + min-stats Wiener denoise + de-shutter + STFT de-click + -19 LUFS", tiers=meta, summary=summ, clips=clips),
              open(idx_p, "w"), separators=(",", ":"))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
