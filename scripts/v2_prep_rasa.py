"""Sakhiya v2 Stage 1 (Track A): ai4bharat/Rasa (Nepali + Sanskrit, F/M studio, 48 kHz) -> 24 kHz feats/v2_studio_24k.pt.
soxr HQ resample -> -23 LUFS -> gentle silence trim (keep 100 ms margins) -> 100-band log-mel (n_fft/win 1024, hop 256, fmax 12k, floor 1e-5).
Each real speaker keeps its own spk_id (never merged); is_studio=1; no Hindi, no WORLD, no radio.
usage: python3 scripts/v2_prep_rasa.py [rasa_dir] [out.pt]"""
import glob, io, os, re, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np, soundfile as sf, soxr, librosa, pyloudnorm as pyln
import torch
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SR, NF, HOP, NM = 24000, 1024, 256, 100
SPK = {("Nepali", "Female"): 0, ("Nepali", "Male"): 1, ("Sanskrit", "Female"): 2, ("Sanskrit", "Male"): 3}
LANG = {"Nepali": "ne", "Sanskrit": "sa"}
STYLES = {}
MEL = librosa.filters.mel(sr=SR, n_fft=NF, n_mels=NM, fmin=0, fmax=12000).astype(np.float32)
WIN = np.hanning(NF + 1)[:-1].astype(np.float32)


def logmel(y):
    p = (NF - HOP) // 2
    y = np.pad(y, (p, p), mode="reflect")
    fr = np.lib.stride_tricks.sliding_window_view(y, NF)[::HOP] * WIN
    s = np.abs(np.fft.rfft(fr, axis=-1)).astype(np.float32)
    return np.log(np.maximum(s @ MEL.T, 1e-5)).T.astype(np.float16)  # [100, T]


SH = "/work/tts/feats/v2_shards"


def process(f):
    sp = f"{SH}/{f.split('/')[-2]}_{os.path.basename(f)}.pt"                 # resumable per-parquet shards
    if os.path.exists(sp): return torch.load(sp, weights_only=False)
    import pyarrow.parquet as pq
    from sakhiya import g2p, sanskrit
    meter = pyln.Meter(SR)
    out, pf = [], pq.ParquetFile(f)
    for g in range(pf.num_row_groups):
        for r in pf.read_row_group(g).to_pylist():
            try:
                lang = LANG[r["language"]]; spk = SPK[(r["language"], r["gender"])]
                y, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                if y.ndim > 1: y = y.mean(1)
                y = y[::2].copy() if sr == 48000 else soxr.resample(y, sr, SR, quality="HQ")
                _, (a, b) = librosa.effects.trim(y, top_db=45, frame_length=1024, hop_length=256)
                m = int(0.10 * SR); y = y[max(0, a - m): min(len(y), b + m)]
                if len(y) < SR * 0.6 or len(y) > SR * 24: continue
                L = meter.integrated_loudness(y); y = pyln.normalize.loudness(y, L, -23.0)
                pk = np.abs(y).max()
                if pk > 0.97: y = y * (0.97 / pk)
                txt = r["text"].strip()
                ph = sanskrit.phonemize_v26(txt) if lang == "sa" else g2p.text_to_phones(txt, "ne")   # v2.6: SAME front-end as inference (no stale G2P)
                st = re.sub(r"^[A-Z]+_[FM]_|_\d+$", "", r["filename"]).strip("_")
                out.append(dict(id=r["filename"], text=txt, phones=ph, mel=torch.from_numpy(logmel(y)), spk=spk,
                                lang=lang, style=st, is_studio=1, dur=len(y) / SR, src="rasa"))
            except Exception as e:
                print("skip", r.get("filename"), repr(e)[:90], flush=True)
    os.makedirs(SH, exist_ok=True); torch.save(out, sp + ".tmp"); os.replace(sp + ".tmp", sp)
    print("done", os.path.basename(f), len(out), flush=True)
    return out


def process_path(f):
    sp = f"{SH}/{f.split('/')[-2]}_{os.path.basename(f)}.pt"
    if not os.path.exists(sp): process(f)
    return sp


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "/work/tts/data/rasa"
    dst = sys.argv[2] if len(sys.argv) > 2 else "/work/tts/feats/v2_studio_24k.pt"
    ms = int(os.environ.get("MAX_SHARDS_PER_LANG", "0"))
    ne_f = sorted(glob.glob(f"{root}/Nepali/*.parquet"))
    sa_f = sorted(glob.glob(f"{root}/Sanskrit/*.parquet"))
    files = (ne_f[::max(1, len(ne_f) // ms)][:ms] + sa_f[::max(1, len(sa_f) // ms)][:ms]) if ms > 0 else (ne_f + sa_f)
    print(len(files), "parquet files", flush=True); t = time.time()
    # session 43: workers only write their shard and return the path (returning ~490 mel tensors per shard through the pipe exhausted fds and hung the pool)
    with Pool(min(int(os.environ.get("NW", 12)), len(files)), maxtasksperchild=1) as p: shards = list(p.imap_unordered(process_path, files))
    res = [x for sp in shards for x in torch.load(sp, weights_only=False)]
    styles = sorted({x["style"] for x in res}); sid = {s: i for i, s in enumerate(styles)}
    rng = np.random.RandomState(0); hold = set()
    for k in set(x["spk"] for x in res):                       # 1.5 % per-speaker held-out (never trained on)
        ids = [x["id"] for x in res if x["spk"] == k]; hold |= set(rng.choice(ids, max(8, len(ids) // 66), replace=False))
    for x in res: x["style_id"] = sid[x["style"]]; x["split"] = "val" if x["id"] in hold else "train"
    hrs = {}
    for x in res: hrs[f'{x["lang"]}/spk{x["spk"]}'] = hrs.get(f'{x["lang"]}/spk{x["spk"]}', 0) + x["dur"] / 3600
    torch.save(dict(items=res, styles=styles, spk=dict((f"{a}_{b}", v) for (a, b), v in SPK.items()),
                    spec=dict(sr=SR, n_fft=NF, hop=HOP, n_mels=NM, fmax=12000, floor=1e-5)), dst + ".tmp")
    os.replace(dst + ".tmp", dst)
    json.dump(dict(hours=hrs, n=len(res), styles=styles), open(dst + ".json", "w"), indent=1)
    print("V2_STUDIO_DONE", len(res), {k: round(v, 2) for k, v in hrs.items()}, f"{time.time()-t:.0f}s", flush=True)
