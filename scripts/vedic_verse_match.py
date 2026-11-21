#!/usr/bin/env python3
"""Recover the GENUINE chanted verses that the language-ID purge threw away (owner: 'real mantras were removed, kept ones had garbage').
Whisper language-ID cannot recognise chanted Sanskrit (it says English/Gujarati/Tamil), so it removed real chanting.  Here every candidate clip from the
Gita class / discourse recordings is transcribed by MMS-hin (Devanagari CTC) and searched against the 701 exact Gita verses (+ verse pairs).
A clip is accepted ONLY if its sound matches a real verse (partial ratio >= 72, >= 25 letters); its text is then the REFERENCE verse text, not an ASR guess.
Talk / commentary (Malayalam, Tamil, English) does not match any verse and stays rejected.   out: /work/tts/vedic/verified.jsonl  (+ ref_text.json)"""
import glob, json, os, re, sys, time, collections
import numpy as np, pandas as pd, soundfile as sf, torch, torchaudio.functional as AF
from rapidfuzz import fuzz, process
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yt_asr_ensemble import dev_only
from vedic_gita_ingest import clean
ROOT = os.environ.get("TTS_ROOT", "/work/tts"); os.environ.setdefault("HF_HOME", f"{ROOT}/hf")
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def main():
    from transformers import Wav2Vec2ForCTC, AutoProcessor
    ddir = f"{ROOT}/data/gita_ss/data"; df = pd.concat([pd.read_parquet(f"{ddir}/{f}") for f in os.listdir(ddir) if f.endswith(".parquet")])
    verses = [clean(t).replace("।", "").split() for t in df["text"]]; verses = [v for v in verses if v]
    cand = []   # (flat letters, word list, per-letter word index)
    for i in range(len(verses)):
        for w in (verses[i], verses[i] + (verses[i + 1] if i + 1 < len(verses) else [])):
            flat = ""; idx = []
            for k, x in enumerate(w): d = dev_only(x); flat += d; idx += [k] * len(d)
            cand.append((flat, w, idx))
    keys = [c[0] for c in cand]; log(len(cand), "reference verse windows")
    P = AutoProcessor.from_pretrained("facebook/mms-1b-all"); P.tokenizer.set_target_lang("hin")
    M = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="hin", ignore_mismatched_sizes=True, torch_dtype=torch.float16).cuda().eval()
    voc = P.tokenizer.get_vocab(); inv = {v: c for c, v in voc.items()}; blank = voc["<pad>"]
    files = sorted(f for pat in ("ia_gita-nochur__*", "ia_BhagavadGitaSanskrit__*") for f in glob.glob(f"{ROOT}/segments/vedic_chant/{pat}.wav"))
    out = f"{ROOT}/vedic/verified.jsonl"; done = {json.loads(l)["path"] for l in open(out)} if os.path.exists(out) else set(); files = [f for f in files if f not in done]
    log(len(files), "candidate clips"); refp = f"{ROOT}/vedic/ref_text.json"; REF = json.load(open(refp)); n_ok = 0; B = 24
    for i in range(0, len(files), B):
        bt = files[i:i + B]; ws = []
        for f in bt:
            x, sr = sf.read(f, dtype="float32"); x = x.mean(1) if x.ndim > 1 else x; ws.append(AF.resample(torch.from_numpy(x), sr, 16000).numpy())
        with torch.no_grad():
            inp = P(ws, sampling_rate=16000, return_tensors="pt", padding=True, return_attention_mask=True)
            ids = M(inp.input_values.cuda().half(), attention_mask=inp.attention_mask.cuda()).logits.argmax(-1).cpu()
        with open(out, "a") as fo:
            for j, f in enumerate(bt):
                T = min(ids.shape[1], int(np.ceil(len(ws[j]) / 320))); s = ids[j, :T].tolist(); seq = [a for a, b in zip(s, [-1] + s[:-1]) if a != b and a != blank]
                q = dev_only("".join(inv[a] for a in seq).replace("|", ""))
                r = {"path": f, "src": os.path.basename(f).rsplit("_", 1)[0], "dur": round(len(ws[j]) / 16000, 2), "text_mms": q, "ok": False, "score": 0.0}
                if len(q) >= 25:
                    m = process.extractOne(q, keys, scorer=fuzz.partial_ratio, processor=None, score_cutoff=60)
                    if m is not None:
                        ci = m[2]; al = fuzz.partial_ratio_alignment(q, keys[ci]); sc = al.score; flat, w, idx = cand[ci]; r["score"] = round(sc, 1)
                        if sc >= 72:
                            a, b = idx[min(al.dest_start, len(idx) - 1)], idx[min(max(al.dest_end - 1, 0), len(idx) - 1)]
                            r.update(ok=True, text_ref=" ".join(w[a:b + 1])); REF[f] = r["text_ref"]; n_ok += 1
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        if (i // B) % 20 == 0: log(f"{i + len(bt)}/{len(files)} accepted {n_ok}")
    json.dump(REF, open(refp, "w"), ensure_ascii=False)
    rows = [json.loads(l) for l in open(out)]; bysrc = collections.Counter(r["src"] for r in rows if r["ok"])
    rep = {"clips_checked": len(rows), "genuine_verses": sum(r["ok"] for r in rows), "hours": round(sum(r["dur"] for r in rows if r["ok"]) / 3600, 2), "by_source": dict(bysrc)}
    json.dump(rep, open(f"{ROOT}/vedic/verified_report.json", "w")); log("VERIFIED_DONE", rep)

if __name__ == "__main__": main()
