#!/usr/bin/env python3
"""EXP10 (Sonnet S4): text-matched HELD-OUT evaluation of a Sakhiya-Voca checkpoint against the REAL Rasa recordings.

Why: the training loss is computed on seen sentences (memorisation is invisible), and exp9 compared different sentences.
Here every test sentence is *unseen* (sentence-hash `test` split, 0 % overlap) and we have the real recording of that exact
sentence, so every number is a paired comparison.  CPU-only, ~1 min for 100 clips.

Per clip (speaker-matched, NEUTRAL styles WIKI/NEWS/CONV/BOOK only):
  mel_dtw   : mean |log-mel(synth) - log-mel(real)| along a DTW path   (lower = closer spectral content, timing-invariant)
  dur_ratio : synth duration / real duration                            (1.0 ideal; <0.9 or >1.1 = tempo wrong)
  f0_med_r  : median f0 synth / real                                    (speaker identity, 1.0 ideal)
  f0_std_r  : voiced-f0 std (semitones) synth / real                    (<0.8 = monotone / flat prosody)
  mcd_like  : DTW distance on 25-band cepstra (dB-ish)
Aggregated per speaker (F / M).  No metric is a promotion criterion on its own; ears decide (Listen page).

usage (on the GPU box, CPU torch):
  python3 sonnet/experiments/exp10_heldout_textmatched.py CKPT.pt OUT.json [--data DIR] [--n 60] [--vmode rasa|kathmandu]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import librosa  # noqa: E402
import torch  # noqa: E402

from sakhiya.voca.frontend import phonemize_utterance  # noqa: E402
from sakhiya.voca.model import SakhiyaVoca, SakhiyaVocaConfig  # noqa: E402

SR, HOP = 24000, 256
NEUTRAL = {"WIKI", "NEWS", "CONV", "BOOK"}
STYLE_ID = {"WIKI": 0, "NEWS": 1, "CONV": 2, "BOOK": 3}


def logmel(w: np.ndarray) -> np.ndarray:
    S = np.abs(librosa.stft(w, n_fft=1024, hop_length=HOP, win_length=1024)) ** 2
    M = librosa.feature.melspectrogram(S=S, sr=SR, n_mels=80, fmax=12000)
    return np.log(np.maximum(M, 1e-8)).T.astype(np.float32)


def f0_track(w: np.ndarray):
    f0, vf, _ = librosa.pyin(w, fmin=60, fmax=500, sr=SR, frame_length=2048, hop_length=HOP * 2)
    v = f0[vf & np.isfinite(f0)]
    return v


def dtw_cost(a: np.ndarray, b: np.ndarray) -> float:
    D, wp = librosa.sequence.dtw(X=a.T, Y=b.T, metric="cityblock")
    return float(D[-1, -1] / len(wp) / a.shape[1])


def load_model(ckpt: Path) -> tuple[SakhiyaVoca, SakhiyaVocaConfig]:
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = SakhiyaVocaConfig.from_dict(st["config"])
    m = SakhiyaVoca(cfg)
    sd = st["model"] if "model" in st else st["ema_model"]
    m.load_state_dict(sd)
    return m.eval(), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--data", type=Path, default=Path("/work/tts/s4/data/rasa_ne_full"))
    ap.add_argument("--n", type=int, default=60, help="clips per speaker")
    ap.add_argument("--vmode", default="rasa")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()

    model, cfg = load_model(a.ckpt)
    rows = [json.loads(l) for l in (a.data / "manifest.jsonl").open() if l.strip()]
    rows = [r for r in rows if r["split"] == a.split and r["style"].upper() in NEUTRAL and 2.0 <= r["frames"] * HOP / SR <= 9.0]
    out = {"ckpt": str(a.ckpt), "step": None, "params": sum(p.numel() for p in model.parameters()), "split": a.split, "per_speaker": {}}
    try:
        out["step"] = int(torch.load(a.ckpt, map_location="cpu", weights_only=False).get("step"))
    except Exception:
        pass
    torch.set_num_threads(8)
    shard_cache: dict[str, list] = {}
    for spk, label in ((0, "F"), (1, "M")):
        sel = [r for r in rows if r["speaker_id"] == spk][: a.n]
        M = {k: [] for k in ("mel_dtw", "dur_ratio", "f0_med_r", "f0_std_r")}
        for r in sel:
            sh = shard_cache.setdefault(r["shard"], torch.load(a.data / "shards" / r["shard"], map_location="cpu", weights_only=True))
            real = sh[r["shard_index"]]["wave"].float().numpy()
            syms, ids, stress = phonemize_utterance(r["text"], lang="ne", v_mode=a.vmode)
            with torch.no_grad():
                wave, _ = model.synthesize(torch.tensor([ids]), torch.tensor([stress]), speaker=spk,
                                           style=STYLE_ID[r["style"].upper()] if cfg.num_styles > 0 else None)
            syn = wave[0].numpy().astype(np.float32)
            syn = syn / max(np.abs(syn).max(), 1e-4) * 0.92
            real = real / max(np.abs(real).max(), 1e-4) * 0.92
            M["dur_ratio"].append(len(syn) / len(real))
            M["mel_dtw"].append(dtw_cost(logmel(syn), logmel(real)))
            fs, fr = f0_track(syn), f0_track(real)
            if len(fs) > 20 and len(fr) > 20:
                M["f0_med_r"].append(float(np.median(fs) / np.median(fr)))
                M["f0_std_r"].append(float(np.std(12 * np.log2(fs / np.median(fs))) / max(np.std(12 * np.log2(fr / np.median(fr))), 1e-3)))
        out["per_speaker"][label] = {"n": len(sel), **{k: {"mean": float(np.mean(v)), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))} for k, v in M.items() if v}}
        print(label, {k: round(v["mean"], 3) for k, v in out["per_speaker"][label].items() if isinstance(v, dict)}, flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
