#!/usr/bin/env python3
"""More accurate Nepali words: transcribe every kept / heard Nepali clip with up to THREE independent models and pick the consensus.
  A  openai/whisper-large-v3 (full, not turbo), language=ne, beam 5
  B  facebook/mms-1b-all CTC, adapter npi
  C  ai4bharat/indic-conformer-600m-multilingual (lang ne, CTC+RNNT) - gated: used only if the HF token has been granted access
Output /work/tts/asr/candidates.jsonl: {id, path, style, cands:{A,B,C}, best, best_model, consensus}.  The audit card shows all candidates so the owner
can fix a clip with one copy-paste.  Resumable.  run: python scripts/gpucap.py 0.15 scripts/asr_candidates.py [--limit N]"""
import argparse, glob, json, os, sys, time
import numpy as np, soundfile as sf, torch, torchaudio.functional as AF
ROOT = os.environ.get("TTS_ROOT", "/work/tts"); dev = "cuda" if torch.cuda.is_available() else "cpu"
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def sim(a, b):
    from rapidfuzz import fuzz
    return fuzz.ratio(a, b) / 100 if a and b else 0.0
def jl(p):
    try: return [json.loads(l) for l in open(p)]
    except Exception: return []
def clips(limit):
    out = {}
    for m in jl(f"{ROOT}/dataset/meta.jsonl"):
        p = next(iter(glob.glob(f"{ROOT}/segments/*/{m['id']}.wav")), "")
        if p and m.get("style") != "vedic_chant": out[m["id"]] = (p, m.get("style", ""))
    for r in jl(f"{ROOT}/asr/asr.jsonl"):
        if r.get("cat") != "vedic_chant" and os.path.exists(r["path"]): out.setdefault(os.path.basename(r["path"])[:-4], (r["path"], r["cat"]))
    items = sorted(out.items()); return items[:limit] if limit else items
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    from transformers import WhisperForConditionalGeneration, WhisperProcessor, Wav2Vec2ForCTC, AutoProcessor, AutoModel
    tok = open("/tmp/hf_token").read().strip() if os.path.exists("/tmp/hf_token") else None
    wp = WhisperProcessor.from_pretrained("openai/whisper-large-v3"); wm = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3", torch_dtype=torch.float16).to(dev).eval()
    mp = AutoProcessor.from_pretrained("facebook/mms-1b-all"); mp.tokenizer.set_target_lang("npi")
    mm = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="npi", ignore_mismatched_sizes=True, torch_dtype=torch.float16).to(dev).eval()
    ic = None
    try: ic = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True, token=tok).to(dev).eval(); log("IndicConformer ON")
    except Exception as e: log("IndicConformer OFF (gated - request access on its Hugging Face page):", type(e).__name__)
    out = f"{ROOT}/asr/candidates.jsonl"; done = {r["id"] for r in jl(out)}
    todo = [(i, v) for i, v in clips(a.limit) if i not in done]; log(len(todo), "clips to transcribe"); fo = open(out, "a"); B = 6
    for s in range(0, len(todo), B):
        bt = todo[s:s + B]; ws = [AF.resample(torch.from_numpy(sf.read(p, dtype="float32")[0]), 24000, 16000) for _, (p, _) in bt]
        with torch.no_grad():
            f = wp([w.numpy() for w in ws], sampling_rate=16000, return_tensors="pt").input_features.to(dev).half()
            A = wp.batch_decode(wm.generate(f, language="ne", task="transcribe", num_beams=3, max_new_tokens=225), skip_special_tokens=True)
            Bt = []
            for w in ws:
                x = mp(w.numpy(), sampling_rate=16000, return_tensors="pt").input_values.to(dev).half()
                Bt.append(mp.decode(mm(x).logits.argmax(-1)[0]))
            Ct = []
            for w in ws:
                try: Ct.append(str(ic(w.unsqueeze(0).to(dev), "ne", "ctc")).strip() if ic else "")
                except Exception: Ct.append("")
        for (cid, (p, st)), x, y, z in zip(bt, A, Bt, Ct):
            c = {k: v.strip() for k, v in (("A", x), ("B", y), ("C", z)) if v and v.strip()}
            sc = {k: np.mean([sim(v, c[j]) for j in c if j != k] or [0]) for k, v in c.items()}
            bk = max(sc, key=sc.get) if sc else ""
            fo.write(json.dumps({"id": cid, "path": p, "style": st, "cands": c, "best": c.get(bk, ""), "best_model": bk, "consensus": round(float(sc.get(bk, 0)), 3)}, ensure_ascii=False) + "\n")
        fo.flush(); log(s + len(bt), "/", len(todo))
if __name__ == "__main__": main()
