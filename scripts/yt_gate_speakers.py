#!/usr/bin/env python3
"""Stage 3: GPU acoustic quality gate + speaker diarization-by-embedding.
For every segment: UTMOS22-strong, SNR, clipping, click/glitch (spectral-flux z-spikes as in forensic_audio_gate.py),
spectral-hole / metallic score, WavLM-Base-Plus-SV embedding (whole clip + 1.5 s windows -> mid-clip speaker-switch / overlap score).
Writes /work/tts/gate/gate.jsonl (all metrics) and gate_emb.npy; then `--cluster` assigns speaker ids per category and `--filter` applies thresholds.
usage: yt_gate_speakers.py [--score] [--cluster] [--filter --utmos 3.4]"""
import argparse, glob, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, soundfile as sf, torch, torch.nn.functional as F, torchaudio.functional as AF

ROOT = os.environ.get("TTS_ROOT", "/work/tts"); G = f"{ROOT}/gate"; SR = 24000
os.environ.setdefault("HF_HOME", f"{ROOT}/hf")
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def load_utmos(dev):
    return torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(dev).eval()

def acoustics(w):
    """w: cuda float tensor @24k. returns snr_db, clip_frac, clicks, hf_hole."""
    fr = w.unfold(0, 960, 480); rms = fr.pow(2).mean(1).clamp_min(1e-10)
    lo, hi = torch.quantile(rms, 0.10), torch.quantile(rms, 0.90)
    snr = float(10 * torch.log10(hi / lo))
    clip = float(((w.abs() > 0.94).float().diff().clamp_min(0)).sum() / len(w) * 100)
    spec = torch.stft(w, 512, 256, window=torch.hann_window(512, device=w.device), return_complex=True).abs()
    ls = torch.log(spec + 1e-4); flux = ls.diff(dim=1).clamp_min(0).pow(2).mean(0).sqrt()
    z = (flux - flux.median()) / (flux.std() + 1e-5); clicks = int((z > 8.0).sum())
    band = spec.pow(2).mean(1); hf = float(band[170:].sum() / band.sum().clamp_min(1e-9))   # >~8 kHz energy share (metallic/hole check)
    return snr, clip, clicks, hf

@torch.no_grad()
def score(a):
    si, sn = map(int, a.shard.split("/"))
    from transformers import AutoFeatureExtractor, WavLMForXVector
    dev = "cuda"; ut = load_utmos(dev)
    sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(dev).eval().half()
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    segs = [json.loads(l) for p in sorted(glob.glob(f"{ROOT}/segments/segments.*.jsonl")) for l in open(p)]
    os.makedirs(G, exist_ok=True); out = f"{G}/gate_{si}.jsonl"
    done = {json.loads(l)["path"] for p in glob.glob(f"{G}/gate_*.jsonl") for l in open(p)}
    todo = [s for s in segs if s["path"] not in done][si::sn]; log(f"{len(todo)} to score ({len(done)} done)")
    embs = []; ex = ThreadPoolExecutor(8); B = 32
    rd = lambda s: sf.read(s["path"], dtype="float32")[0]
    for i in range(0, len(todo), B):
        bt = todo[i:i + B]; wavs = list(ex.map(rd, bt)); rows = []
        for s, w in zip(bt, wavs):
            wt = torch.from_numpy(w).to(dev); snr, clip, clicks, hf = acoustics(wt)
            w16 = AF.resample(wt, SR, 16000); m = float(ut(w16[None], 16000)[0])
            # whole-clip + windowed speaker embeddings
            win, hop = 24000, 12000; ws = [w16[j:j + win] for j in range(0, max(1, len(w16) - win + 1), hop)] or [w16]
            inp = fe([x.cpu().numpy() for x in [w16] + ws], sampling_rate=16000, return_tensors="pt", padding=True, return_attention_mask=True)
            e = sv(input_values=inp["input_values"].to(dev).half(), attention_mask=inp["attention_mask"].to(dev)).embeddings.float()
            e = F.normalize(e, dim=-1); sims = (e[1:] @ e[0]).cpu().numpy()
            embs.append(e[0].cpu().numpy())
            rows.append({**s, "utmos": round(m, 3), "snr": round(snr, 1), "clip": round(clip, 4), "clicks": clicks, "hf": round(hf, 4),
                         "spk_min_sim": round(float(sims.min()), 3), "spk_mean_sim": round(float(sims.mean()), 3)})
        with open(out, "a") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        np.save(f"{G}/emb_{si}_{int(time.time()*1000)}.npy", np.stack(embs[-len(rows):]))
        with open(f"{G}/order_{si}_{i}.txt", "w") as f: f.write("\n".join(r["path"] for r in rows))
        if (i // B) % 20 == 0: log(f"scored {i+len(bt)}/{len(todo)}")

def load_embs():
    d = {}
    for f in glob.glob(f"{G}/emb_*.npy"):    # emb_<shard>_<ms>.npy pairs with order via sorted timestamps within the shard
        pass
    for si in sorted({os.path.basename(f).split("_")[1] for f in glob.glob(f"{G}/emb_*.npy")}):
        npys = sorted(glob.glob(f"{G}/emb_{si}_*.npy")); ords = sorted(glob.glob(f"{G}/order_{si}_*.txt"), key=lambda x: int(x.rsplit("_", 1)[1][:-4]))
        for n, o in zip(npys, ords):
            E = np.load(n); P = [x for x in open(o).read().split("\n") if x]
            for p, e in zip(P, E): d[p] = e
    return d

def cluster():
    from sklearn.cluster import AgglomerativeClustering
    rows = [json.loads(l) for p in glob.glob(f"{G}/gate_*.jsonl") for l in open(p)]; emb = load_embs(); spk = {}
    for cat in sorted({r["cat"] for r in rows}):
        rs = [r for r in rows if r["cat"] == cat and r["path"] in emb]
        # one centroid per source file first (cheap, robust), then cluster centroids across files
        by = {}
        for r in rs: by.setdefault(r["src"], []).append(r)
        for src, lst in by.items():
            X = np.stack([emb[r["path"]] for r in lst])
            if len(lst) < 4: lab = np.zeros(len(lst), int)
            else: lab = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=0.22).fit_predict(X)
            for r, l in zip(lst, lab): spk[r["path"]] = f"{cat[:3]}_{src[:28]}_s{l}"
        log(cat, "speakers", len({v for k, v in spk.items() if f'/{cat}/' in k}))
    json.dump(spk, open(f"{G}/speakers.json", "w"), ensure_ascii=False)

def filt(a):
    rows = [json.loads(l) for p in glob.glob(f"{G}/gate_*.jsonl") for l in open(p)]; spk = json.load(open(f"{G}/speakers.json"))
    thr = {"vedic_chant": a.utmos_chant}; keep, why = [], {}
    for r in rows:
        r["spk"] = spk.get(r["path"], "unk"); t = thr.get(r["cat"], a.utmos); bad = None
        if r["utmos"] < t: bad = "utmos"
        elif r["spk_min_sim"] < a.spk_sim: bad = "speaker_switch"
        elif r["clip"] > 0.05: bad = "clipping"
        elif r["clicks"] > a.clicks: bad = "clicks"
        elif r["snr"] < a.snr: bad = "snr"
        elif r["hf"] > 0.35: bad = "metallic"
        if bad: why[bad] = why.get(bad, 0) + 1
        else: keep.append(r)
    with open(f"{G}/passed.jsonl", "w") as f:
        for r in keep: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    hrs = sum(r["dur"] for r in keep) / 3600; log(f"kept {len(keep)}/{len(rows)} = {hrs:.1f} h; rejected: {why}")
    for c in sorted({r['cat'] for r in rows}): log(c, round(sum(r['dur'] for r in keep if r['cat'] == c) / 3600, 2), "h")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true"); ap.add_argument("--shard", default="0/1"); ap.add_argument("--cluster", action="store_true"); ap.add_argument("--filter", action="store_true")
    ap.add_argument("--utmos", type=float, default=3.4); ap.add_argument("--utmos-chant", type=float, default=3.4)
    ap.add_argument("--spk-sim", type=float, default=0.55); ap.add_argument("--clicks", type=int, default=3); ap.add_argument("--snr", type=float, default=18.0)
    a = ap.parse_args()
    if a.score: score(a)
    if a.cluster: cluster()
    if a.filter: filt(a)
