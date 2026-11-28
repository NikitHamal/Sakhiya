#!/usr/bin/env python3
"""EXP11 (Sonnet S4): objective INTELLIGIBILITY probe - ASR round-trip CER on HELD-OUT, text-matched Rasa sentences.

EXP10 measures spectral/prosodic closeness; neither it nor the training loss says whether a listener (or an ASR) can
*understand* the sounds.  This is the first intelligibility number for Sakhiya-Voca checkpoints:

  CER(real recording of sentence s)   -> the ASR floor (accent/dialect/ASR error, not TTS)
  CER(synth(s))                       -> floor + TTS error
  gap = CER(synth) - CER(real)        -> what the TTS adds     (the number to drive to ~0)

ASR = facebook/mms-1b-all [npi] (the repo's calibrated scorer, sonnet/05_evaluation_protocol.md).  Strict: Unicode NFC,
punctuation and case stripped, whitespace collapsed; NO schwa / geminate / व->ब normalisation.
Also dumps a character-level substitution/deletion/insertion confusion table (synth vs reference), so we can see WHICH
Nepali sounds are unclear (e.g. aspirated vs unaspirated, retroflex, nasalisation, vowel length), plus the worst sentences.

usage (GPU box):  python3 sonnet/experiments/exp11_asr_roundtrip.py CKPT.pt OUT.json [--n 30] [--vmode rasa]
CPU torch renders the TTS (small model); MMS runs on cuda if available.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import unicodedata
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


def norm(t: str) -> str:
    t = unicodedata.normalize("NFC", t)
    t = re.sub(r"[^\w\s\u0900-\u097F]", " ", t)          # punctuation -> space
    t = t.replace("।", " ").replace("॥", " ")
    t = re.sub(r"[\u0966-\u096F]", lambda m: str(ord(m.group()) - 0x0966), t)  # Devanagari digits -> ascii
    return re.sub(r"\s+", " ", t).strip().lower()


def align(ref: str, hyp: str):
    """Levenshtein with backtrace -> list of (op, ref_char, hyp_char)."""
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("ok" if ref[i - 1] == hyp[j - 1] else "sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            ops.append(("del", ref[i - 1], ""))
            i -= 1
        else:
            ops.append(("ins", "", hyp[j - 1]))
            j -= 1
    return ops[::-1]


def load_model(ckpt: Path):
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = SakhiyaVocaConfig.from_dict(st["config"])
    m = SakhiyaVoca(cfg)
    m.load_state_dict(st["model"] if "model" in st else st["ema_model"])
    return m.eval(), cfg, st.get("step")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--data", type=Path, default=Path("/work/tts/s4/data/rasa_ne_full"))
    ap.add_argument("--n", type=int, default=30, help="clips per speaker")
    ap.add_argument("--vmode", default="rasa")
    ap.add_argument("--split", default="test")
    ap.add_argument("--wav-dir", type=Path, default=None, help="optionally save synth wavs here")
    a = ap.parse_args()

    from transformers import AutoProcessor, Wav2Vec2ForCTC
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rid = "facebook/mms-1b-all"
    proc = AutoProcessor.from_pretrained(rid, target_lang="npi")
    asr = Wav2Vec2ForCTC.from_pretrained(rid, target_lang="npi", ignore_mismatched_sizes=True).to(dev).eval()

    def transcribe(w24: np.ndarray) -> str:
        w = librosa.resample(w24.astype(np.float32), orig_sr=SR, target_sr=16000)
        w = (w - w.mean()) / max(w.std(), 1e-5)
        inp = proc(w, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            lg = asr(inp.input_values.to(dev)).logits
        return proc.decode(lg.argmax(-1)[0])

    model, cfg, step = load_model(a.ckpt)
    torch.set_num_threads(4)
    rows = [json.loads(l) for l in (a.data / "manifest.jsonl").open() if l.strip()]
    rows = [r for r in rows if r["split"] == a.split and r["style"].upper() in NEUTRAL and 2.0 <= r["frames"] * HOP / SR <= 9.0]
    out = {"ckpt": str(a.ckpt), "step": step, "vmode": a.vmode, "asr": rid + "[npi]", "per_speaker": {}, "worst": []}
    conf = collections.Counter()
    ref_char_counts = collections.Counter()
    shard_cache: dict[str, list] = {}
    all_rows = []
    for spk, label in ((0, "F"), (1, "M")):
        sel = [r for r in rows if r["speaker_id"] == spk][: a.n]
        E = {"real": [0, 0], "synth": [0, 0]}  # edits, ref chars
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
            if a.wav_dir:
                import soundfile as sf
                a.wav_dir.mkdir(parents=True, exist_ok=True)
                sf.write(a.wav_dir / f"{label}_{r['shard']}_{r['shard_index']}.wav", syn, SR)
            ref = norm(r["text"])
            h_real, h_syn = norm(transcribe(real)), norm(transcribe(syn))
            row = {"spk": label, "ref": ref, "real": h_real, "synth": h_syn}
            for key, h in (("real", h_real), ("synth", h_syn)):
                ops = align(ref, h)
                ed = sum(o[0] != "ok" for o in ops)
                E[key][0] += ed
                E[key][1] += len(ref)
                row["cer_" + key] = ed / max(len(ref), 1)
                if key == "synth":
                    for op, rc, hc in ops:
                        if op != "ok":
                            conf[(op, rc, hc)] += 1
                    ref_char_counts.update(ref)
            all_rows.append(row)
        cer = {k: v[0] / max(v[1], 1) for k, v in E.items()}
        out["per_speaker"][label] = {"n": len(sel), "cer_real": cer["real"], "cer_synth": cer["synth"], "gap": cer["synth"] - cer["real"]}
        print(label, {k: round(v, 4) for k, v in out["per_speaker"][label].items() if k != "n"}, flush=True)
    out["worst"] = sorted(all_rows, key=lambda r: r["cer_synth"] - r["cer_real"], reverse=True)[:10]
    out["confusions_top"] = [
        {"op": op, "ref": rc, "hyp": hc, "count": c, "ref_freq": ref_char_counts.get(rc, 0)}
        for (op, rc, hc), c in conf.most_common(40)
    ]
    # per reference char: error rate (how often is this char wrong in synth transcription)
    per_char = collections.Counter()
    for (op, rc, hc), c in conf.items():
        if op in ("sub", "del") and rc:
            per_char[rc] += c
    out["char_error_rate_top"] = sorted(
        [{"char": ch, "err": e, "n": ref_char_counts[ch], "rate": e / ref_char_counts[ch]} for ch, e in per_char.items() if ref_char_counts[ch] >= 15],
        key=lambda d: -d["rate"])[:25]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("worst-char-rate:", [(d["char"], round(d["rate"], 2)) for d in out["char_error_rate_top"][:12]])


if __name__ == "__main__":
    main()
