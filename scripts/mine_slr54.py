"""Mine extra clean Nepali speech from OpenSLR-54 (CC-BY-SA 4.0, commercially usable) into feats/slr54.pt.
Same per-clip gates as curate.py (SNR / clipping / pitch stability / ASR phoneme-CER) plus a stricter
speaker-level selection: only speakers with >= MINN surviving clips, best-scored first.
Records are tagged wide=False when the 16 kHz source has no energy above 8 kHz (never used by the vocoder).
Then merge with v2:  python scripts/mine_slr54.py --merge   ->  feats/v3.pt"""
import os, sys, glob, random, collections, argparse, json
from concurrent.futures import ProcessPoolExecutor
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import curate

R = "/work/tts/data/slr54"; OUT = "/work/tts/feats"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=9000); ap.add_argument("--minn", type=int, default=50)
    ap.add_argument("--cer", type=float, default=0.22); ap.add_argument("--snr", type=float, default=24); ap.add_argument("--merge", action="store_true"); ap.add_argument("--mms", action="store_true")
    ap.add_argument("--maxspk", type=int, default=30); ap.add_argument("--topspk", type=int, default=60); ap.add_argument("--percap", type=int, default=220); ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    if a.merge:
        v2 = torch.load(f"{OUT}/v2.pt", weights_only=False); ex = torch.load(f"{OUT}/slr54.pt", weights_only=False)
        torch.save(v2 + ex, f"{OUT}/v3.pt"); print("v3", len(v2), "+", len(ex)); sys.exit()
    idx = {os.path.basename(p)[:-5]: p for p in glob.glob(R + "/**/data/*/*.flac", recursive=True)}
    its = []
    for ln in open(glob.glob(R + "/**/utt_spk_text.tsv", recursive=True)[0], encoding="utf8"):
        p = ln.rstrip("\n").split("\t")
        if len(p) >= 3 and p[0] in idx and len(p[2].split()) >= 3:
            its.append(dict(id="o54_" + p[0], spk="o54_" + p[1], text=p[2].strip(), path=idx[p[0]], src="slr54", lic="cc-by-sa-4.0"))
    random.seed(1); random.shuffle(its)
    cnt = collections.Counter(i["spk"] for i in its); top = {s for s, _ in cnt.most_common(a.topspk)}   # speaker-first: dense speakers only
    taken = collections.Counter(); sel = []
    for i in its:
        if i["spk"] in top and taken[i["spk"]] < a.percap: taken[i["spk"]] += 1; sel.append(i)
    its = sel[: a.n]; print("per-speaker utts (top):", cnt.most_common(5), flush=True)
    print("candidates", len(its), "speakers", len({i["spk"] for i in its}), flush=True)
    with ProcessPoolExecutor(a.procs) as ex: res = [r for r in ex.map(curate.proc, its, chunksize=8) if r]
    res = [r for r in res if r["clip"] < 5e-4 and r["vr"] > 0.25 and r["jump"] < 0.06 and r["snr"] >= a.snr]
    print("after snr/pitch", len(res), flush=True)
    if a.mms:
        import feats_edge; feats_edge.asr_cer_mms(res)    # calibrated MMS-1b-all scorer (same one used for the benchmark)
    else: curate.asr_cer(res)
    print("cer quantiles", np.percentile([r["cer"] for r in res], [10, 25, 50, 75, 90]).round(3).tolist(), flush=True)
    res = [r for r in res if r["cer"] <= a.cer]
    print("after asr", len(res), flush=True)
    by = collections.defaultdict(list)
    for r in res: by[r["spk"]].append(r)
    sc = {s: float(np.median([r["snr"] for r in rs])) - 40 * float(np.median([r["cer"] for r in rs])) for s, rs in by.items() if len(rs) >= a.minn}
    keep = sorted(sc, key=lambda s: -sc[s])[: a.maxspk]
    out = []
    for s in keep:
        for r in by[s]: r.pop("mmel", None); r["split"] = "train"; out.append(r)
    info = {s: dict(n=len(by[s]), score=sc[s], snr=float(np.median([r["snr"] for r in by[s]])), f0m=float(np.exp(np.median([r["f0m"] for r in by[s]]))), cer=float(np.median([r["cer"] for r in by[s]])), wide=float(np.mean([r["wide"] for r in by[s]]))) for s in keep}
    print("kept", len(out), "clips", len(keep), "speakers; hours", round(sum(len(r["wav"]) for r in out) / 22050 / 3600, 2), flush=True)
    for s in keep[:12]: print(s, info[s], flush=True)
    for r in out: r["gender"] = "f" if info[r["spk"]]["f0m"] > 165 else "m"
    print("hours by gender", {g: round(sum(len(r["wav"]) for r in out if r["gender"] == g) / 22050 / 3600, 2) for g in "fm"}, flush=True)
    torch.save(out, f"{OUT}/slr54.pt"); json.dump(info, open(f"{OUT}/slr54_speakers.json", "w"), indent=1); print("MINE_DONE", flush=True)
