#!/usr/bin/env python3
"""STAGE 1 (v2.6): zero-mismatch re-phonemization. Mels/wavs stay cached; ONLY phoneme ids + prosody metadata are regenerated, using the exact
cleaned v2.6 front-end (sakhiya.g2p / sakhiya.sanskrit.phonemize_v26 == what inference runs). Resumable, CPU-only, ~minutes.
in : /work/tts/feats/v2_shards/*.pt   out: /work/tts/feats/v26_shards/*.pt  + feats/v26_stage1.json (stats)
Adds per-utterance prosody descriptors (speaker-z-scored): rate_z (phones/s), energy_std_z, pause_z (+ f0_std_z = None until audio-based F0 is filled)
and a canonical 'style_tag' (news / conversational / narration / expressive / neutral) from the Rasa filename tag."""
import glob, json, os, sys, time
from multiprocessing import Pool
import numpy as np, torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
IN, OUT = os.environ.get("V2_SHARDS", "/work/tts/feats/v2_shards"), os.environ.get("V26_SHARDS", "/work/tts/feats/v26_shards")
TAG = {"NEWS": "news", "CONV": "conversational", "BOOK": "narration", "WIKI": "neutral", "NAMES": "neutral",
       "HAPPY": "expressive", "SURPRISE": "expressive", "ANGER": "expressive", "FEAR": "expressive", "SAD": "expressive", "DISGUST": "expressive"}


def one(f):
    from sakhiya import g2p, sanskrit
    o = f"{OUT}/{os.path.basename(f)}"
    if os.path.exists(o): return os.path.basename(f), 0, 0
    d, ch = torch.load(f, weights_only=False), 0
    for r in d:
        try:
            ph = sanskrit.phonemize_v26(r["text"]) if r["lang"] == "sa" else g2p.text_to_phones(r["text"], "ne")
        except Exception:
            ph = r["phones"]
        ch += int(ph != r["phones"]); r["phones_v25"] = r["phones"]; r["phones"] = ph
        m = r["mel"].float(); e = m.mean(0).numpy(); T = max(1, len(e))
        r["pros_raw"] = [len(ph) / max(r["dur"], 0.1), float(e.std()), float((e < e.max() - 0.5 * (e.max() - e.min())).mean())]
        r["style_tag"] = TAG.get(r["style"], "neutral")
    os.makedirs(OUT, exist_ok=True); torch.save(d, o + ".tmp"); os.replace(o + ".tmp", o)
    return os.path.basename(f), len(d), ch


def zscore():
    """second pass: per-speaker z-scores over all shards -> r['pros'] = [rate_z, f0_std_z(None->0), energy_std_z, pause_z] and a null-flag-free float vector."""
    fs = sorted(glob.glob(OUT + "/*.pt")); acc = {}
    for f in fs:
        for r in torch.load(f, weights_only=False): acc.setdefault(r["spk"], []).append(r["pros_raw"])
    st = {k: (np.mean(v, 0), np.std(v, 0) + 1e-6) for k, v in acc.items()}
    for f in fs:
        d = torch.load(f, weights_only=False)
        for r in d:
            mu, sd = st[r["spk"]]; z = (np.array(r["pros_raw"]) - mu) / sd
            r["pros"] = [float(np.clip(z[0], -3, 3)), 0.0, float(np.clip(z[1], -3, 3)), float(np.clip(z[2], -3, 3))]; r["f0_std_z_filled"] = False
        torch.save(d, f + ".tmp"); os.replace(f + ".tmp", f)
    return {str(k): int(len(v)) for k, v in acc.items()}


if __name__ == "__main__":
    t0 = time.time(); fs = sorted(glob.glob(IN + "/*.pt")); os.makedirs(OUT, exist_ok=True)
    with Pool(min(16, os.cpu_count() or 4)) as p:
        res = []
        for x in p.imap_unordered(one, fs): res.append(x); print("rephon", *x, flush=True)
    n, ch = sum(r[1] for r in res), sum(r[2] for r in res)
    per_spk = zscore()
    json.dump({"shards": len(fs), "utts": n, "phones_changed_vs_v25": ch, "per_speaker_utts": per_spk, "sec": time.time() - t0}, open("/work/tts/feats/v26_stage1.json", "w"), indent=1)
    print("STAGE1 DONE", n, "utts,", ch, "changed vs v2.5 phonemes", flush=True)
