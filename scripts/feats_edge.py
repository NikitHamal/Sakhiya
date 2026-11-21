"""Turn the distilled two-voice corpus (data/edge) into curated training features: feats/edge.pt.
Gates: decode/duration, clipping, pitch stability, ASR phoneme-CER vs the *input text* (drops teacher misreads:
numerals, abbreviations, foreign names), then 1.5% val split.  python scripts/feats_edge.py [--cer 0.30]"""
import os, sys, json, random, argparse, collections
from concurrent.futures import ProcessPoolExecutor
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import curate

E = "/work/tts/data/edge"; OUT = "/work/tts/feats"


def asr_cer_mms(res, bs=48):
    """Calibrated scorer (MMS-1b-all[npi], 0.063 CER on human test clips): phoneme-CER of the ASR hypothesis vs the input text."""
    import librosa, jiwer
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    from sakhiya import audio, g2p
    rid = "facebook/mms-1b-all"; pr = AutoProcessor.from_pretrained(rid, target_lang="npi")
    m = Wav2Vec2ForCTC.from_pretrained(rid, target_lang="npi", ignore_mismatched_sizes=True).cuda().eval().half()
    order = sorted(range(len(res)), key=lambda i: len(res[i]["wav"]))
    for b in range(0, len(order), bs):
        rs = [res[i] for i in order[b:b + bs]]
        ws = [librosa.resample(r["wav"].astype(np.float32) / 32767, orig_sr=audio.SR, target_sr=16000) for r in rs]
        inp = pr(ws, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad(): lg = m(inp.input_values.cuda().half()).logits.argmax(-1)
        for r, h in zip(rs, pr.batch_decode(lg)):
            import re as _re; hyp = g2p.text_to_phones(_re.sub(r"[A-Za-z]+", " ", h))   # MMS hypotheses are Devanagari; stray Latin must not spawn the node Roman bridge per clip
            r["cer"] = jiwer.cer(r["phs"].replace(" ", ""), " ".join(hyp).replace(" ", "").replace("|", "")) if h.strip() else 1.0
        if b % 1920 == 0: print("asr", b, len(res), flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cer", type=float, default=0.30); ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--asr", default="mms"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--out", default="edge")
    a = ap.parse_args()
    seen, its = set(), []
    for ln in open(f"{E}/manifest.jsonl", encoding="utf8"):
        m = json.loads(ln); p = f"{E}/{m['spk']}/{m['id']}.mp3"
        if m["id"] in seen or not os.path.exists(p): continue
        seen.add(m["id"]); its.append(dict(id=m["id"], spk="edge_" + m["spk"], text=m["text"], path=p, src="edge", lic="research-only"))
    if a.limit: its = its[: a.limit]
    print("candidates", collections.Counter(i["spk"] for i in its), flush=True)
    with ProcessPoolExecutor(a.procs) as ex: res = [r for r in ex.map(curate.proc, its, chunksize=8) if r]
    print("decoded", len(res), collections.Counter(r["spk"] for r in res), flush=True)
    res = [r for r in res if r["clip"] < 1e-3 and r["jump"] < 0.09]
    print("after clip/pitch", len(res), flush=True)
    (asr_cer_mms if a.asr == "mms" else curate.asr_cer)(res)
    print("cer quantiles", np.percentile([r["cer"] for r in res], [10, 50, 75, 90, 95]).round(3).tolist(), flush=True)
    res = [r for r in res if r["cer"] <= a.cer]
    random.seed(3)
    for r in res:
        r.pop("mmel", None); r["wide"] = True; r["split"] = "val" if random.random() < 0.015 else "train"
    per = collections.defaultdict(float)
    for r in res:
        if r["split"] == "train": per[r["spk"]] += len(r["wav"]) / 22050 / 3600
    print("FINAL hours", dict(per), "clips", len(res), flush=True)
    torch.save(res, f"{OUT}/{a.out}.pt"); print("FEATS_DONE", flush=True)
