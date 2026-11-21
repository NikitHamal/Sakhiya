#!/usr/bin/env python3
"""Vedic GOLD audit (owner: 'kept ones still contain garbage, genuine mantras removed').
Every Vedic clip that has a reference verse text is re-listened to by an independent Devanagari CTC model (MMS-hin) and compared letter-by-letter
with the reference text.  A clip is GOLD only if (a) the sound really matches the written verse, (b) the speaking rate is physically plausible for
Sanskrit, (c) it is mostly voiced (chanting, not noise/talk-over) and (d) it has a sensible length.  Tiers: gold / ok / doubtful.
Also tags 'melodic' (sustained, stepwise pitch = swara) vs 'recited'.   out: /work/tts/vedic/gold.json  {path: {...}}   (resumable)
usage: python scripts/gpucap.py 0.15 scripts/vedic_gold_audit.py"""
import json, os, sys, time
import numpy as np, soundfile as sf, torch, torchaudio.functional as AF
from rapidfuzz import fuzz
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yt_asr_ensemble import dev_only
from vedic_purity_gate import chant_feats
ROOT = os.environ.get("TTS_ROOT", "/work/tts"); os.environ.setdefault("HF_HOME", f"{ROOT}/hf"); OUT = f"{ROOT}/vedic/gold.json"
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def tier(r):
    why = []
    if r["match"] < 55: why.append(f"sound does not match the verse ({r['match']:.0f}%)")
    if not 2.5 <= r["dur"] <= 18: why.append("length outside 2.5-18 s")
    if not 3.0 <= r["lps"] <= 14.0: why.append(f"speed {r['lps']:.1f} letters/s is not natural chanting")
    if r["voiced"] < 0.40: why.append("too little voiced sound")
    if why: return "doubtful", why
    return ("gold", []) if r["match"] >= 72 and r["voiced"] >= 0.5 else ("ok", [])

def main():
    from transformers import Wav2Vec2ForCTC, AutoProcessor
    ref = json.load(open(f"{ROOT}/vedic/ref_text.json")); res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    ver = set()
    try: ver = {json.loads(l)["path"] for l in open(f"{ROOT}/vedic/verified.jsonl") if json.loads(l).get("ok")}
    except Exception: pass
    todo = [p for p in ref if p not in res and os.path.exists(p) and (os.path.basename(p)[:4] in ("vag_", "gss_") or p in ver)]
    log(len(todo), "clips to audit,", len(res), "done")
    P = AutoProcessor.from_pretrained("facebook/mms-1b-all"); P.tokenizer.set_target_lang("hin")
    M = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="hin", ignore_mismatched_sizes=True, torch_dtype=torch.float16).cuda().eval()
    voc = P.tokenizer.get_vocab(); inv = {v: c for c, v in voc.items()}; blank = voc["<pad>"]
    for i, p in enumerate(todo):
        try:
            x, sr = sf.read(p, dtype="float32"); x = x if x.ndim == 1 else x.mean(1); dur = len(x) / sr
            w = AF.resample(torch.from_numpy(x), sr, 16000)[: 16000 * 20]; w = (w - w.mean()) / (w.std() + 1e-7)
            with torch.no_grad(): lg = M(w[None].half().cuda()).logits[0].argmax(-1).tolist()
            hyp, prev = [], -1
            for t in lg:
                if t != prev and t != blank: hyp.append(inv.get(t, ""))
                prev = t
            hyp = dev_only("".join(hyp)); rf = dev_only(ref[p])
            vr, steady, rng = chant_feats(x, sr)
            r = {"match": float(fuzz.ratio(rf, hyp)), "partial": float(fuzz.partial_ratio(rf, hyp)), "dur": dur, "lps": len(rf) / max(dur, 0.1), "voiced": vr, "steady": steady, "range_st": rng,
                 "hyp": hyp[:160], "melodic": bool(steady >= 0.55 and vr >= 0.5 and rng >= 1.5)}
            r["tier"], r["why"] = tier(r); res[p] = r
        except Exception as e: log("ERR", p, str(e)[:100])
        if i % 100 == 0 or i == len(todo) - 1:
            json.dump(res, open(OUT + ".tmp", "w"), ensure_ascii=False); os.replace(OUT + ".tmp", OUT); log(i, len(todo), {t: sum(1 for v in res.values() if v["tier"] == t) for t in ("gold", "ok", "doubtful")})
    log("DONE")

if __name__ == "__main__": main()
