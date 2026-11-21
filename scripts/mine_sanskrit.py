"""Mine real native Sanskrit speech (owner: hours of real data, proper svara/laya) into feats/sa.pt.
Source: surajp/shrutilipi_sanskrit (AI4Bharat Shrutilipi, All-India-Radio Sanskrit news/talks, CC-BY-4.0, public HF parquet).
Gates (same family as curate.py): duration, SNR, clipping, pitch stability, wideband flag, then ASR phoneme-CER with MMS-1b-all (san, else hin fallback)
against the transcript's Sanskrit phonemes.  Pseudo-speakers: gender by median F0, then 2 timbre clusters (k-means on mean-mel) -> sa_f0/sa_f1/sa_m0/sa_m1.
usage: python scripts/mine_sanskrit.py --shards 6 --cer 0.35 --snr 18 --hours 14"""
import os, io, sys, glob, json, random, argparse, collections
from concurrent.futures import ProcessPoolExecutor
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import curate

D = "/work/tts/data/sa"; OUT = "/work/tts/feats"


def dump(shards):
    import pyarrow.parquet as pq, soundfile as sf
    os.makedirs(f"{D}/wav", exist_ok=True); its = []
    for p in sorted(glob.glob(f"{D}/hf/data/*.parquet"))[:shards]:
        for b in pq.ParquetFile(p).iter_batches(batch_size=64):
            for r in b.to_pylist():
                t = " ".join((r["transcriptions"] or "").split()); i = f"sa_{len(its):06d}"
                if len(t) < 12 or not curate.OK.match(t): continue
                try:
                    x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                    if x.ndim > 1: x = x.mean(1)
                    sf.write(f"{D}/wav/{i}.wav", x, sr)
                except Exception: continue
                its.append(dict(id=i, spk="sa_x", text=t if t[-1] in "।॥?!" else t + "।", path=f"{D}/wav/{i}.wav", src="shruti", lic="cc-by-4.0", lang="sa", maxdur=17))
        print("dumped", p, len(its), flush=True)
    return its


def asr_cer_sa(res, bs=32):
    import librosa, jiwer
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    from sakhiya import audio, sanskrit
    rid = "facebook/mms-1b-all"; pr = AutoProcessor.from_pretrained(rid); lang = "san"
    try: pr.tokenizer.set_target_lang(lang)
    except Exception: lang = "hin"; pr.tokenizer.set_target_lang(lang)
    m = Wav2Vec2ForCTC.from_pretrained(rid, target_lang=lang, ignore_mismatched_sizes=True).cuda().eval().half(); print("ASR adapter", lang, flush=True)
    order = sorted(range(len(res)), key=lambda i: len(res[i]["wav"]))
    for b in range(0, len(order), bs):
        rs = [res[i] for i in order[b:b + bs]]
        ws = [librosa.resample(r["wav"].astype(np.float32) / 32767, orig_sr=audio.SR, target_sr=16000) for r in rs]
        inp = pr(ws, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad(): lg = m(inp.input_values.cuda().half()).logits.argmax(-1)
        for r, h in zip(rs, pr.batch_decode(lg)):
            hyp = sanskrit.text_to_phones(h, verse=False)
            r["cer"] = jiwer.cer(r["phs"].replace(" ", "").replace("|", ""), " ".join(hyp).replace(" ", "").replace("|", "")) if h.strip() else 1.0
        if b % 960 == 0: print("asr", b, len(res), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--shards", type=int, default=6); ap.add_argument("--cer", type=float, default=0.35)
    ap.add_argument("--snr", type=float, default=18); ap.add_argument("--hours", type=float, default=14); ap.add_argument("--procs", type=int, default=18)
    a = ap.parse_args()
    its = dump(a.shards); print("candidates", len(its), flush=True)
    with ProcessPoolExecutor(a.procs) as ex: res = [r for r in ex.map(curate.proc, its, chunksize=8) if r]
    print("decoded", len(res), "hours", sum(len(r["wav"]) for r in res) / 22050 / 3600, flush=True)
    res = [r for r in res if r["clip"] < 2e-3 and r["jump"] < 0.12 and r["snr"] >= a.snr]
    print("after snr/clip/pitch", len(res), flush=True)
    asr_cer_sa(res)
    print("cer quantiles", np.percentile([r["cer"] for r in res], [10, 25, 50, 75, 90]).round(3).tolist(), flush=True)
    res = sorted([r for r in res if r["cer"] <= a.cer], key=lambda r: r["cer"] - r["snr"] / 200); keep, tot = [], 0.0
    for r in res:
        if tot > a.hours: break
        keep.append(r); tot += len(r["wav"]) / 22050 / 3600
    res = keep
    f0 = np.array([r["f0m"] for r in res]); thr = np.log(165.0)
    from sklearn.cluster import KMeans
    for g, sel in (("f", [i for i, r in enumerate(res) if r["f0m"] >= thr]), ("m", [i for i, r in enumerate(res) if r["f0m"] < thr])):
        if len(sel) < 40: continue
        lab = KMeans(2, n_init=5, random_state=0).fit_predict(np.stack([res[i]["mmel"] for i in sel]))
        for i, l in zip(sel, lab): res[i]["spk"] = f"sa_{g}{l}"
    res = [r for r in res if r["spk"] != "sa_x"]; random.seed(5)
    for r in res: r.pop("mmel", None); r["wide"] = bool(r.get("wide", True)); r["split"] = "val" if random.random() < 0.02 else "train"
    per = collections.defaultdict(float)
    for r in res: per[r["spk"]] += len(r["wav"]) / 22050 / 3600
    print("FINAL hours", {k: round(v, 2) for k, v in per.items()}, "clips", len(res), flush=True)
    torch.save(res, f"{OUT}/sa.pt"); json.dump({k: round(v, 2) for k, v in per.items()}, open(f"{OUT}/sa_hours.json", "w")); print("SA_DONE", flush=True)
