"""Wave 43 data pass: own_ms.pt -> own_ms43.pt (Sanskrit-first, real human speech only).

1. Re-phonemise every Sanskrit row with the fixed front-end (ॐ was silently dropped before, so the 8 pranava-initial Vagdhenu
   recitations were trained with an audio/phoneme mismatch at the start of the clip).  ॐ-initial rows are up-weighted so the
   model learns a prolonged, chanted o-o-m.
2. Curation: drop Sanskrit/Nepali train clips that are noisy, clipped, reverberant, jumpy in F0 or badly mis-transcribed
   (thresholds below are per-source percentiles of the stored quality columns snr / clip / hf / jump / cer).
usage: python3 scripts/build_own43.py [--om_rep 8] [--out /work/tts/feats/own_ms43.pt]"""
import os, sys, argparse, collections
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sakhiya import audio, sanskrit as SA, symbols as S

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/work/tts/feats/own_ms.pt"); ap.add_argument("--out", default="/work/tts/feats/own_ms43.pt")
ap.add_argument("--om_rep", type=int, default=8); ap.add_argument("--drop_snr", type=float, default=8.0)   # drop lowest-SNR 8 %
ap.add_argument("--drop_cer", type=float, default=12.0)                                                       # drop worst-transcribed 12 %
a = ap.parse_args()
rows = torch.load(a.src, weights_only=False)
n0 = collections.Counter((r["split"], r["lang"]) for r in rows)


def pct(lang, k, q):
    v = np.array([r.get(k) if r.get(k) is not None else np.nan for r in rows if r["split"] == "train" and r["lang"] == lang and r.get("orig_spk") != "vag"], float)
    return float(np.nanpercentile(v, q))


# per-language percentile gates (the stored ASR `cer` has a different scale for ne and sa, so absolute thresholds would wipe out Nepali)
TH = {l: dict(snr=pct(l, "snr", a.drop_snr), cer=pct(l, "cer", 100 - a.drop_cer), jump=pct(l, "jump", 97)) for l in ("ne", "sa")}
print("gates", TH, flush=True)


def good(r):
    if r["split"] != "train": return True
    if r.get("orig_spk") == "vag": return True            # studio-clean CC-BY chant: always kept
    t = TH[r["lang"]]
    if (r.get("snr") or 99) < t["snr"] or (r.get("cer") or 0.0) > t["cer"] or (r.get("jump") or 0.0) > t["jump"]: return False
    return not (r.get("clip") and r["clip"] > 0.002)       # clipped-sample fraction


out, om, changed = [], 0, 0
for r in rows:
    if not good(r): continue
    if r["lang"] == "sa":
        ph = SA.text_to_phones(r["text"], add_end=True, verse=False)
        ph = [q for k in ph for q in (SA.OM_PH if k == SA.OM else (k,))]
        ids = S.encode(ph)
        if list(ids) != list(r["ph"]): changed += 1
        r["ph"] = ids; r["phs"] = " ".join(ph)
        if "ॐ" in r["text"] and r["split"] == "train": om += 1; out.extend(dict(r) for _ in range(a.om_rep - 1))
    out.append(r)
n1 = collections.Counter((r["split"], r["lang"]) for r in out)
hrs = collections.defaultdict(float)
for r in out:
    if r["split"] == "train": hrs[r["lang"]] += len(r["wav"]) / audio.SR / 3600
print("before", dict(n0), "\nafter ", dict(n1), "\nrephonemised-changed", changed, "om-rows", om, "train hours", {k: round(v, 2) for k, v in hrs.items()}, flush=True)
torch.save(out, a.out); print("OWN43_DONE", flush=True)
