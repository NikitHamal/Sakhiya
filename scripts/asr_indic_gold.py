#!/usr/bin/env python3
"""Most accurate Nepali words we can get today: ai4bharat IndicConformer-600M (lang 'ne', CTC + RNNT decoders), run on CPU in parallel workers
(its ONNX graph has no CUDA provider here).  Measured on the first clips it writes clean Devanagari where Whisper-large-v3 writes phonetic garbage
('मेरो कविताको शीर्षक छ मरेको कवि' vs 'मेरो कविता को सिर्षक्षा मरे को कभी').
Transcribes every non-Vedic clip we know (asr.jsonl + dataset/meta.jsonl + gate/passed.jsonl) -> /work/tts/asr/indic.jsonl  (resumable)
  {id, path, cat, text_ic (RNNT, falls back to CTC), ic_ctc, ic_rnnt, text_w (whisper), agree_ic (CTC vs RNNT %), agree_w (IC vs whisper %)}
Final ? ! । from the earlier acoustic punctuation pass is carried over by scripts/apply_corrections.py / publish_clip_library.py.
usage: python scripts/asr_indic_gold.py [workers=8]"""
import glob, json, os, re, shutil, sys, time
from multiprocessing import Pool
ROOT = os.environ.get("TTS_ROOT", "/work/tts"); OUT = f"{ROOT}/asr/indic.jsonl"; REPO = "ai4bharat/indic-conformer-600m-multilingual"
def jl(p):
    try: return [json.loads(l) for l in open(p)]
    except Exception: return []
_m = None
def _deref():
    """HF cache symlinks make onnxruntime refuse external weights ('path escapes model directory') -> replace them by real files once."""
    import huggingface_hub.constants as hc
    for snap in glob.glob(f"{hc.HF_HUB_CACHE}/models--{REPO.replace('/', '--')}/snapshots/*"):
        for d, _, fs in os.walk(snap):
            for f in fs:
                p = os.path.join(d, f)
                if os.path.islink(p): t = os.path.realpath(p); os.remove(p); shutil.copy2(t, p)
def _init():
    global _m
    import torch; torch.set_num_threads(2); os.environ["OMP_NUM_THREADS"] = "2"
    from transformers import AutoModel
    _m = AutoModel.from_pretrained(REPO, trust_remote_code=True, token=open("/tmp/hf_token").read().strip())
def _one(it):
    import torch, soundfile as sf, torchaudio.functional as AF
    from rapidfuzz import fuzz
    try:
        x, sr = sf.read(it["path"], dtype="float32"); x = x if x.ndim == 1 else x.mean(1)
        w = AF.resample(torch.from_numpy(x), sr, 16000).unsqueeze(0)
        c = str(_m(w, "ne", "ctc")).strip()
        try: r = str(_m(w, "ne", "rnnt")).strip()
        except Exception: r = ""
        it = dict(it, ic_ctc=c, ic_rnnt=r, text_ic=r or c, agree_ic=round(fuzz.ratio(c, r), 1) if r else None)
        it["agree_w"] = round(fuzz.ratio(it["text_ic"], re.sub(r"[,?!।.]", "", it.get("text_w", ""))), 1) if it.get("text_w") else None
        return it
    except Exception as e: return {"id": it["id"], "err": str(e)[:100]}
def items():
    out = {}
    for r in jl(f"{ROOT}/asr/asr.jsonl"):
        if r.get("cat") != "vedic_chant" and os.path.exists(r["path"]): out[os.path.basename(r["path"])[:-4]] = {"path": r["path"], "cat": r["cat"], "text_w": r.get("text_w", "")}
    for m in jl(f"{ROOT}/dataset/meta.jsonl"):
        p = next(iter(glob.glob(f"{ROOT}/segments/*/{m['id']}.wav")), "")
        if p and m.get("style") != "vedic_chant" and m["id"] not in out: out[m["id"]] = {"path": p, "cat": m.get("style", ""), "text_w": m.get("text", "")}
    for r in jl(f"{ROOT}/gate/passed.jsonl"):
        i = os.path.basename(r["path"])[:-4]
        if r.get("cat") != "vedic_chant" and i not in out and os.path.exists(r["path"]): out[i] = {"path": r["path"], "cat": r["cat"], "text_w": ""}
    return [dict(id=i, **v) for i, v in out.items()]
if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8; _deref(); done = {r["id"] for r in jl(OUT) if "text_ic" in r}
    todo = [x for x in items() if x["id"] not in done]; print(time.strftime("%H:%M:%S"), len(todo), "to do,", len(done), "done", flush=True)
    todo.sort(key=lambda x: (x["cat"] == "news", x["id"]))   # storytelling/talk first
    with Pool(n, initializer=_init) as P, open(OUT, "a") as fo:
        for k, r in enumerate(P.imap_unordered(_one, todo, chunksize=4)):
            if "text_ic" in r: fo.write(json.dumps(r, ensure_ascii=False) + "\n"); fo.flush()
            if k % 100 == 0: print(time.strftime("%H:%M:%S"), k, len(todo), flush=True)
    print("DONE", flush=True)
