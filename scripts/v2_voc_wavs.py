"""Sakhiya-v2-Voc data (Track B): clean 24 kHz studio waveforms from ai4bharat/Rasa, int16, per-parquet resumable shards.
Audio chain is IDENTICAL to scripts/v2_prep_rasa.py (soxr VHQ -> trim top_db 45 w/ 100 ms margins -> -23 LUFS -> peak 0.97),
so mel(wav) matches the AM shard mel and GTA mels line up frame-for-frame by `id`.
No radio, no WORLD, no Hindi, no Edge-TTS.  usage: python3 scripts/v2_voc_wavs.py [per_file_cap=120] [procs=8]"""
import glob, io, os, sys, zlib
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np, soundfile as sf, soxr, librosa, pyloudnorm as pyln, torch
from multiprocessing import Pool

SR = 24000
SPK = {("Nepali", "Female"): 0, ("Nepali", "Male"): 1, ("Sanskrit", "Female"): 2, ("Sanskrit", "Male"): 3}
OUT = "/work/tts/feats/v2_voc_shards"
CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 120


def process(f):
    sp = f"{OUT}/{f.split('/')[-2]}_{os.path.basename(f)}.pt"
    if os.path.exists(sp):
        return sp
    import pyarrow.parquet as pq
    meter = pyln.Meter(SR)
    out, pf, nm = [], pq.ParquetFile(f), 0
    for g in range(pf.num_row_groups):
        for r in pf.read_row_group(g).to_pylist():
            try:
                y, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                if y.ndim > 1:
                    y = y.mean(1)
                y = y[::2].copy() if sr == 48000 else soxr.resample(y, sr, SR, quality="HQ")
                _, (a, b) = librosa.effects.trim(y, top_db=45, frame_length=1024, hop_length=256)
                m = int(0.10 * SR)
                y = y[max(0, a - m): min(len(y), b + m)]
                if len(y) < SR * 0.7 or len(y) > SR * 18:
                    continue
                L = meter.integrated_loudness(y)
                y = pyln.normalize.loudness(y, L, -23.0)
                pk = np.abs(y).max()
                if pk > 0.97:
                    y = y * (0.97 / pk)
                fid = r["filename"]
                out.append(dict(id=fid, wav=torch.from_numpy((y * 32767.0).astype(np.int16)),
                                spk=SPK[(r["language"], r["gender"])], lang=r["language"][:2].lower(),
                                split="val" if zlib.crc32(fid.encode()) % 60 == 0 else "train"))
            except Exception as e:
                print("skip", r.get("filename"), repr(e)[:80], flush=True)
            if len(out) >= CAP:
                break
        if len(out) >= CAP:
            break
    os.makedirs(OUT, exist_ok=True)
    torch.save(out, sp + ".tmp")
    os.replace(sp + ".tmp", sp)
    print("done", os.path.basename(f), len(out), flush=True)
    return sp


if __name__ == "__main__":
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    max_shards = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    ne = sorted(glob.glob("/work/tts/data/rasa/Nepali/*.parquet"))
    sa = sorted(glob.glob("/work/tts/data/rasa/Sanskrit/*.parquet"))
    # Interleave first-half (Female) and second-half (Male) parquets across Nepali and Sanskrit for strict 50/50 F/M balance
    ne_a, ne_b = ne[:len(ne)//2], ne[len(ne)//2:][::-1]
    sa_a, sa_b = sa[:len(sa)//2], sa[len(sa)//2:][::-1]
    order = []
    for k in range(max(len(ne_a), len(ne_b), len(sa_a), len(sa_b))):
        for lst in (ne_a, ne_b, sa_a, sa_b):
            if k < len(lst): order.append(lst[k])
    order = order[:max_shards]
    with Pool(procs) as p:
        for _ in p.imap_unordered(process, order):
            pass
    print("ALL_DONE", len(order), flush=True)
