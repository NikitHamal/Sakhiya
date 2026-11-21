#!/usr/bin/env python3
"""Stage 5b: render the mined expressive corpus in the canonical `sakhiya` female voice at 24 kHz.
1. matching set  : Sakhiya's own ONNX TTS (docs/models) renders ~400 transcripts of the corpus -> WavLM-Large layer-6 frames (kNN-VC target pool)
2. octave-aware  : WORLD (formant-preserving) shifts the source by WHOLE octaves toward Sakhiya's median F0 (male chanters +12, female 0) so melody intervals/laya survive
3. kNN-VC        : bshall/knn-vc (WavLM + HiFi-GAN, topk=4) frame-synchronous -> 16 kHz -> 24 kHz ; metrics: log-F0 Pearson vs source, WavLM-SV cos to Sakhiya, UTMOS22
out: dataset/vc/vc-<ts>.parquet|tar (same schema as raw, spk='sakhiya', tier carried) ; dataset/vc_done.txt
usage: yt_vc_sakhiya.py [--limit 4000] [--min-corr 0.55]"""
import argparse, glob, io, json, os, subprocess, sys, tarfile, time
import numpy as np, soundfile as sf, torch, torch.nn.functional as F, torchaudio.functional as AF, pyarrow.parquet as pq, pyarrow as pa
import torchaudio as _ta   # torchaudio 2.11 needs TorchCodec for load(); the kNN-VC hub code calls torchaudio.load -> use soundfile instead (fixes vc.log ImportError)
def _sf_load(path, normalize=True, **kw):
    x, sr = sf.read(str(path), dtype="float32", always_2d=True); return torch.from_numpy(x.T.copy()), sr
_ta.load = _sf_load
ROOT = os.environ.get("TTS_ROOT", "/work/tts"); REPO = f"{ROOT}/repo"; sys.path.insert(0, REPO); dev = "cuda"
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def make_matching_set(texts_meta, n=400):
    out = f"{ROOT}/vc_ref"; os.makedirs(out, exist_ok=True)
    if len(glob.glob(f"{out}/*.wav")) >= n * 0.8: return sorted(glob.glob(f"{out}/*.wav"))
    try: import onnxruntime  # noqa
    except Exception: subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime"])
    from sakhiya.onnx_tts import SakhiyaONNX
    tts = SakhiyaONNX(f"{REPO}/docs/models/", tiny=False); rng = np.random.RandomState(0); rng.shuffle(texts_meta); ok = 0
    for m in texts_meta:
        if ok >= n: break
        t = m["text"]
        if not (30 < len(t) < 220): continue
        try:
            w, sr = tts.say(t, speaker="sakhiya", lang=m["lang"], style=m["style"]); sf.write(f"{out}/{ok:04d}.wav", np.asarray(w, dtype=np.float32), sr); ok += 1
        except Exception: continue
    log("matching set rendered:", ok); return sorted(glob.glob(f"{out}/*.wav"))

def world_octave_shift(w, sr, target_med):
    import pyworld as pw
    x = w.astype(np.float64); f0, t = pw.harvest(x, sr, f0_floor=60, f0_ceil=520, frame_period=10.0); v = f0 > 0
    if v.sum() < 20: return w, 0
    n_oct = int(round(np.log2(target_med / np.median(f0[v])))); n_oct = max(-1, min(2, n_oct))
    if n_oct == 0: return w, 0
    sp = pw.cheaptrick(x, f0, t, sr); ap = pw.d4c(x, f0, t, sr); y = pw.synthesize(f0 * (2.0 ** n_oct), sp, ap, sr, 10.0)
    return y.astype(np.float32), n_oct

def f0_logs(w, sr=24000):
    import pyworld as pw
    f0, t = pw.dio(w.astype(np.float64), sr, f0_floor=60, f0_ceil=600, frame_period=10.0); f0 = pw.stonemask(w.astype(np.float64), f0, t, sr); return f0

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=4000); ap.add_argument("--min-corr", type=float, default=0.55); a = ap.parse_args()
    meta = [json.loads(l) for l in open(f"{ROOT}/dataset/meta.jsonl")]; done = set(open(f"{ROOT}/dataset/vc_done.txt").read().split()) if os.path.exists(f"{ROOT}/dataset/vc_done.txt") else set()
    if not meta: return log("no meta yet")
    ref = make_matching_set([m for m in meta if m["tier"] == "gold" or m["utmos"] > 2.5][:3000] or meta[:3000])
    knn = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True, device=dev)
    ms = knn.get_matching_set(ref); tgt_f0 = np.median(np.concatenate([f0_logs(sf.read(r, dtype="float32")[0], 24000)[f0_logs(sf.read(r, dtype="float32")[0], 24000) > 0] for r in ref[:40]]))
    log(f"matching frames {ms.shape}, sakhiya median F0 {tgt_f0:.0f} Hz")
    from transformers import AutoFeatureExtractor, WavLMForXVector
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv"); sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(dev).eval()
    ut = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(dev).eval()
    emb = lambda w16: F.normalize(sv(**{k: v.to(dev) for k, v in fe(w16, sampling_rate=16000, return_tensors="pt").items()}).embeddings, dim=-1)[0]
    ref_e = F.normalize(torch.stack([emb(AF.resample(torch.from_numpy(sf.read(r, dtype="float32")[0]), 24000, 16000).numpy()) for r in ref[:30]]).mean(0), dim=0)
    rows, order = [], {"vedic_chant": 0, "storytelling": 1, "news": 2, "conversational": 3}
    for p in sorted(glob.glob(f"{ROOT}/dataset/raw/*.parquet")):
        t = pq.read_table(p).to_pylist(); rows += [r for r in t if r["id"] not in done]
    rows.sort(key=lambda r: (order.get(r["style"], 9), -sum(c in "?!" for c in r["text"]))); rows = rows[:a.limit]; log(len(rows), "clips to convert"); outb, nk = [], 0
    with torch.no_grad():
        for i, r in enumerate(rows):
            try:
                w, sr = sf.read(io.BytesIO(r["wav"]), dtype="float32"); w2, n_oct = world_octave_shift(w, sr, tgt_f0)
                w16 = AF.resample(torch.from_numpy(w2), 24000, 16000).to(dev); q = knn.get_features(w16[None].cpu() if False else w16)
                y16 = knn.match(q, ms, topk=4).to(dev); y24 = AF.resample(y16.float(), 16000, 24000).cpu().numpy(); y24 = y24 / (np.abs(y24).max() + 1e-6) * 0.95
                fs, fc = f0_logs(w2), f0_logs(y24); n = min(len(fs), len(fc)); m = (fs[:n] > 0) & (fc[:n] > 0)
                corr = float(np.corrcoef(np.log(fs[:n][m]), np.log(fc[:n][m]))[0, 1]) if m.sum() > 30 else 0.0
                sim = float(emb(AF.resample(torch.from_numpy(y24), 24000, 16000).numpy()) @ ref_e); mos = float(ut(AF.resample(torch.from_numpy(y24).to(dev), 24000, 16000)[None], 16000)[0])
                done.add(r["id"])
                if corr < a.min_corr or sim < 0.6: continue
                lf = np.where(fc > 0, np.log(np.maximum(fc, 1)), np.nan); idx = np.arange(len(lf)); lf = np.interp(idx, idx[fc > 0], lf[fc > 0])
                b = io.BytesIO(); sf.write(b, y24, 24000, format="WAV", subtype="PCM_16")
                outb.append({**{k: v for k, v in r.items() if k not in ("wav", "f0_logf0_10ms", "voiced", "spk")}, "spk": "sakhiya", "wav": b.getvalue(), "f0_logf0_10ms": lf.astype(np.float32).tolist(),
                             "voiced": (fc > 0).astype(np.uint8).tolist(), "src_spk": r["spk"], "octave_shift": n_oct, "f0_corr_vs_src": round(corr, 3), "sim_sakhiya": round(sim, 3), "utmos_vc": round(mos, 3)})
            except Exception as ex: log("vc fail", r["id"], repr(ex)[:120])
            if len(outb) >= 500 or (i == len(rows) - 1 and outb):
                os.makedirs(f"{ROOT}/dataset/vc", exist_ok=True); k = int(time.time()); pq.write_table(pa.Table.from_pylist(outb), f"{ROOT}/dataset/vc/vc-{k}.parquet", compression="zstd")
                with tarfile.open(f"{ROOT}/dataset/vc/vc-{k}.tar", "w") as tf:
                    for o in outb:
                        for ext, d in (("wav", o["wav"]), ("json", json.dumps({x: y for x, y in o.items() if x != "wav"}, ensure_ascii=False).encode())):
                            ti = tarfile.TarInfo(f"{o['id']}.{ext}"); ti.size = len(d); tf.addfile(ti, io.BytesIO(d))
                nk += len(outb); outb = []; open(f"{ROOT}/dataset/vc_done.txt", "w").write("\n".join(done)); log(f"vc {i+1}/{len(rows)} kept {nk}")
    open(f"{ROOT}/dataset/vc_done.txt", "w").write("\n".join(done)); log("VC_DONE kept", nk)

if __name__ == "__main__": main()
