"""Mine the Vagdhenu Sanskrit chant corpus (prathoshap/vagdhenu-data, CC-BY-4.0, studio-clean, ~5.3 h, one male reciter)
into feats/vag.pt as extra CLEAN Sanskrit pronunciation/laya data for the single female `sakhiya` voice.

The reciter is male, so every clip is converted to a female timbre with WORLD (f0 x F0R, spectral-envelope warp x FMT)
before feature extraction.  This is a disclosed augmentation of real, licensed speech (see PROVENANCE.md); it contains no
synthetic-teacher audio.  Clips whose conversion leaves unstable pitch are dropped by the same gates as curate.py.
usage: python scripts/mine_vagdhenu.py [--f0r 1.85] [--fmt 1.17]"""
import os, sys, csv, glob, random, argparse, collections
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import curate
from sakhiya import sanskrit as SA, symbols as S

D = "/work/tts/data/vag"; OUT = "/work/tts/feats"


def convert(it):
    import pyworld as pw, soundfile as sf, librosa
    if os.path.exists(it["path"]): return it
    x, sr = sf.read(it["raw"], dtype="float64"); x = x.mean(1) if x.ndim > 1 else x
    if sr != 24000: x = librosa.resample(x, orig_sr=sr, target_sr=24000)
    f0, t = pw.dio(x, 24000, f0_floor=60, f0_ceil=400, frame_period=5.0, speed=2); f0 = pw.stonemask(x, f0, t, 24000); sp = pw.cheaptrick(x, f0, t, 24000); ap = pw.d4c(x, f0, t, 24000)
    n = sp.shape[1]; src = np.arange(n) / it["fmt"]; sp2 = np.stack([np.interp(src, np.arange(n), s) for s in sp]); sp2 = np.maximum(sp2, 1e-16)
    y = pw.synthesize(f0 * it["f0r"], sp2, ap, 24000, 5.0).astype(np.float32); y = y / max(1e-3, np.abs(y).max()) * 0.9
    sf.write(it["path"], y, 24000); return it


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--f0r", type=float, default=1.85); ap.add_argument("--fmt", type=float, default=1.17)
    ap.add_argument("--procs", type=int, default=18); a = ap.parse_args()
    from huggingface_hub import snapshot_download
    snapshot_download("prathoshap/vagdhenu-data", repo_type="dataset", local_dir=D, allow_patterns=["style_a/*", "style_a/audio/*", "style_b/*", "style_b/audio/*"])
    os.makedirs(f"{D}/fem", exist_ok=True); its = []
    for cfg in ("style_a", "style_b"):
        for r in csv.DictReader(open(f"{D}/{cfg}/metadata.csv", encoding="utf8")):
            t = " ".join(r["text_devanagari"].split())
            if len(t) < 6: continue
            i = f"vag_{cfg}_{len(its):05d}"
            its.append(dict(id=i, spk="sakhiya", text=t, raw=f"{D}/{cfg}/{r['file_name']}", path=f"{D}/fem/{i}.wav", src="vagdhenu", lic="cc-by-4.0", lang="sa", maxdur=26, f0r=a.f0r, fmt=a.fmt))
    print("clips", len(its), flush=True)
    with ProcessPoolExecutor(a.procs) as ex: its = list(ex.map(convert, its, chunksize=4))
    print("converted", len(its), flush=True)
    with ProcessPoolExecutor(a.procs) as ex: res = [r for r in ex.map(curate.proc, its, chunksize=4) if r]
    res = [r for r in res if r["clip"] < 2e-3 and r["jump"] < 0.16]; random.seed(11)
    for r in res:
        r["split"] = "val" if random.random() < 0.03 else "train"; r.pop("mmel", None); r.pop("raw", None); r["wide"] = True
        ph = SA.text_to_phones(r["text"], add_end=True); r["ph"] = S.encode(ph); r["phs"] = " ".join(ph)
    h = sum(len(r["wav"]) for r in res) / 22050 / 3600
    print("VAG kept", len(res), "hours", round(h, 2), "f0 median Hz", round(float(np.exp(np.median([r["f0m"] for r in res]))), 1), collections.Counter(r["split"] for r in res), flush=True)
    import torch; torch.save(res, f"{OUT}/vag.pt"); print("VAG_DONE", flush=True)
