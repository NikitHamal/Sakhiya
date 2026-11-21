"""Build own.pt: a REAL-human-speech-only, single-female-voice training set (no Edge-TTS, no Alpa audio).

All kept clips are re-labelled with one speaker id `sakhiya`, so the acoustic model learns a single voice that
speaks Nepali (lang=ne, incl. Tharu front-end) and Sanskrit (lang=sa). Validation / test clips keep their
original split and original text so eval stays comparable. Sanskrit is up-weighted because it is the scarce tier.
usage: python3 scripts/build_own.py [--sa_rep 3] [--min_snr 14] [--max_cer 0.25]"""
import os, sys, argparse, collections
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sakhiya import audio, g2p, sanskrit as SA, symbols as S

ap = argparse.ArgumentParser(); ap.add_argument("--sa_rep", type=int, default=3); ap.add_argument("--min_snr", type=float, default=14.0)
ap.add_argument("--max_cer", type=float, default=0.25); ap.add_argument("--out", default="/work/tts/feats/own.pt"); ap.add_argument("--vag_rep", type=int, default=2); ap.add_argument("--ne_rep", type=int, default=1); ap.add_argument("--raw_spk", action="store_true"); ap.add_argument("--keep_spk", action="store_true")
a = ap.parse_args()
F = "/work/tts/feats"
TEACHER = ("edge", "rasa_alpa", "alpa")   # never train the own voice on synthetic teacher audio
FEMALE_NE = lambda r: r["spk"].startswith("s") or r["spk"].startswith("fl_f")   # SLR43 (female) + FLEURS female


def keep(r):
    src = str(r.get("src", "")); spk = str(r["spk"])
    if any(t in src for t in TEACHER) or spk.startswith("edge") or "alpa" in src: return False
    if r["split"] != "train": return r.get("lang") == "sa" or src in ("slr43", "fleurs", "cv", "shruti")
    if r.get("lang") == "sa" or src == "shruti":
        return spk.startswith("sa_f") and r.get("snr", 99) >= a.min_snr and r.get("cer", 0.0) <= a.max_cer
    return FEMALE_NE(r)


mix = torch.load(f"{F}/mix.pt", weights_only=False)
print("mix rows", len(mix), "src", collections.Counter(str(r.get("src")) for r in mix).most_common(12), flush=True)
rows = []
for r in mix:
    if not keep(r): continue
    sa = r.get("lang") == "sa" or r.get("src") == "shruti"
    ph = SA.text_to_phones(r["text"], add_end=True, verse=False) if sa else g2p.text_to_phones(r["text"])
    r["ph"] = S.encode(ph); r["phs"] = " ".join(ph); r["lang"] = "sa" if sa else "ne"
    if r["split"] == "train": r["orig_spk"] = r["spk"]; r["spk"] = "sakhiya"
    elif r["split"] == "val": r["orig_spk"] = r["spk"]; r["spk"] = "sakhiya"
    else: r["orig_spk"] = r["spk"]; r["spk"] = "sakhiya"
    rows.append(r)
del mix
if os.path.exists(f"{F}/vag.pt"):   # Vagdhenu CC-BY studio chant (female-converted, vag.pt): clean Sanskrit pronunciation + laya
    vag = torch.load(f"{F}/vag.pt", weights_only=False)
    for r in vag: r["orig_spk"] = "vag"; r["spk"] = "sakhiya"; r["lang"] = "sa"
    rows += [r for r in vag if r["split"] != "train"] + [dict(r) for r in vag if r["split"] == "train" for _ in range(a.vag_rep)]
    print("vagdhenu rows added", len(vag), "x", a.vag_rep, flush=True)
if a.keep_spk:   # Gemini's diagnosis (issue #1): one id for ~36 mics blurs formants. Keep real speaker ids as timbre/channel labels in training;
    # the main Nepali female + Vagdhenu keep `sakhiya`, which is the ONE voice exported and used at inference.
    h = collections.defaultdict(float)
    for r in rows:
        if r["split"] == "train" and r["lang"] == "ne": h[r["orig_spk"]] += len(r["wav"])
    main = max(h, key=h.get); print("keep_spk main Nepali female", main, round(h[main] / audio.SR / 3600, 2), "h", flush=True)
    for r in rows:
        if r["split"] != "train" or r.get("orig_spk") in (main, "vag"): continue
        r["spk"] = r["orig_spk"]
if a.raw_spk:   # Wave 41: every clip keeps its REAL speaker id (incl. `vag`); the exported single voice is the cleanest real female (s3614)
    for r in rows: r["spk"] = r["orig_spk"]
extra = [dict(x) for x in rows if x["split"] == "train" and x["lang"] == "sa" and x.get("orig_spk") != "vag" for _ in range(a.sa_rep - 1)]
rows += extra
rows += [dict(x) for x in rows if x["split"] == "train" and x["lang"] == "ne" for _ in range(a.ne_rep - 1)]   # optional Nepali up-weighting
hrs = collections.defaultdict(float)
for r in rows:
    if r["split"] == "train": hrs[r["lang"]] += len(r["wav"]) / audio.SR / 3600
print("own.pt rows", len(rows), "train hours (with Sanskrit x%d):" % a.sa_rep, {k: round(v, 2) for k, v in hrs.items()},
      "splits", collections.Counter((r["split"], r["lang"]) for r in rows), flush=True)
torch.save(rows, a.out); print("OWN_DONE", flush=True)
