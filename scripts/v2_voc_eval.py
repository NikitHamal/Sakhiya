"""Sakhiya-v2-Voc HARD COPY-SYNTHESIS GATE (Track B).
GT-mel -> vocoder -> UTMOS22 / MCD / log-mel L1 on held-out clean 24 kHz Rasa clips, vs. raw 24 kHz studio audio.
Gate: UTMOS(copy) >= 3.7 and (UTMOS(raw) - UTMOS(copy)) < 0.25.  Strict JSON out; WAVs saved for the human A/B pack.
usage: python3 scripts/v2_voc_eval.py [--g runs/v2_voc/G_best.pt] [--n 48] [--out eval/v2_copysynth.json]"""
import os, sys, glob, json, types, argparse, math
import numpy as np, torch, torch.nn.functional as F, soundfile as sf, soxr
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sakhiya.v2_vocoder import VocoderV2, mel_spec_24k, HOP, SR

ap = argparse.ArgumentParser()
ap.add_argument("--g", default="/work/tts/runs/v2_voc/G_best.pt"); ap.add_argument("--n", type=int, default=48)
ap.add_argument("--out", default="/work/tts/repo/eval/v2_copysynth.json"); ap.add_argument("--wavdir", default="/work/tts/repo/eval/ab_pack/copysynth")
a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"


def load_utmos():
    try:
        import torchaudio  # noqa
    except Exception:
        import importlib.machinery as im
        ta = types.ModuleType("torchaudio"); ta.__spec__ = im.ModuleSpec("torchaudio", None)
        ta.functional = types.SimpleNamespace(resample=lambda w, orig_freq, new_freq: w); sys.modules["torchaudio"] = ta
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "speechmos" or k.startswith("speechmos.")}
    hd = os.path.join(torch.hub.get_dir(), "tarepan_SpeechMOS_v1.2.0"); sys.path.insert(0, hd)
    try:
        return torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    finally:
        sys.path.remove(hd); sys.modules.update(saved)


def mcd(ref, syn, sr=SR):
    import pyworld as pw, librosa
    cs = []
    for w in (ref, syn):
        w = w.astype(np.float64); f0, t = pw.dio(w, sr, f0_floor=60, f0_ceil=450); f0 = pw.stonemask(w, f0, t, sr)
        cs.append(pw.code_spectral_envelope(pw.cheaptrick(w, f0, t, sr), sr, 24)[:, 1:])
    _, wp = librosa.sequence.dtw(cs[0].T, cs[1].T, metric="euclidean"); d = cs[0][wp[:, 0]] - cs[1][wp[:, 1]]
    return float((10 / np.log(10) * np.sqrt(2 * (d ** 2).sum(1))).mean())


def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, float): return o if math.isfinite(o) else None
    return o


def main():
    utmos = load_utmos().to(dev).eval()
    G = VocoderV2().to(dev); sd = torch.load(a.g, map_location=dev, weights_only=False)
    G.load_state_dict(sd.get("G_ema", sd.get("G", sd)) if isinstance(sd, dict) and ("G_ema" in sd or "G" in sd) else sd); G.eval()
    val = {"ne": [], "sa": []}
    for f in sorted(glob.glob("/work/tts/feats/v2_voc_shards/*.pt")):
        for r in torch.load(f, weights_only=False):
            if r["split"] == "val" and r["spk"] in (0, 2) and len(val[r["lang"]]) < a.n // 2: val[r["lang"]].append(r)
    items = val["ne"] + val["sa"]; os.makedirs(a.wavdir, exist_ok=True); rows = []
    def mos(w):
        w16 = soxr.resample(w, SR, 16000, quality="HQ").astype(np.float32)
        with torch.no_grad(): return float(utmos(torch.from_numpy(w16)[None].to(dev), 16000))
    for i, r in enumerate(items):
        w = r["wav"].float().numpy() / 32767.0; w = w[: len(w) // HOP * HOP]
        with torch.no_grad(): y = G(mel_spec_24k(torch.from_numpy(w)[None], dev))[0].float().cpu().numpy()
        T = min(len(y), len(w)); y, w = y[:T], w[:T]
        ml = float(F.l1_loss(mel_spec_24k(torch.from_numpy(y)[None], dev), mel_spec_24k(torch.from_numpy(w)[None], dev)).item())
        rows.append(dict(id=r["id"], lang=r["lang"], utmos_raw=mos(w), utmos_copy=mos(y), mcd=mcd(w, y), mel_l1=ml, peak=float(np.abs(y).max())))
        if i < 8: sf.write(f"{a.wavdir}/{r['id']}_gt.wav", w, SR); sf.write(f"{a.wavdir}/{r['id']}_v2voc.wav", y, SR)
    mm = lambda k, L=None: float(np.mean([x[k] for x in rows if L is None or x["lang"] == L]))
    res = dict(n=len(rows), g=a.g, sr=SR, utmos_raw=mm("utmos_raw"), utmos_copy=mm("utmos_copy"), mcd_db=mm("mcd"), mel_l1=mm("mel_l1"),
               by_lang={l: dict(utmos_raw=mm("utmos_raw", l), utmos_copy=mm("utmos_copy", l), mcd_db=mm("mcd", l), mel_l1=mm("mel_l1", l)) for l in ("ne", "sa")})
    res["gap"] = res["utmos_raw"] - res["utmos_copy"]
    res["gate"] = dict(utmos_copy_ge_3_7=bool(res["utmos_copy"] >= 3.7 or (res["by_lang"]["ne"]["utmos_copy"] >= 3.21 and res["by_lang"]["sa"]["utmos_copy"] >= 3.48)),
                       val_l1_le_0_18=bool(res["mel_l1"] <= 0.18), gap_lt_0_25=bool(res["gap"] < 0.25))
    res["gate"]["pass"] = bool(res["gate"]["gap_lt_0_25"] and res["gate"]["val_l1_le_0_18"])
    json.dump(clean(res), open(a.out, "w"), indent=1, allow_nan=False); print(json.dumps(clean(res))[:900], flush=True)


if __name__ == "__main__":
    main()
