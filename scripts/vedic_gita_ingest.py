#!/usr/bin/env python3
"""Genuine Sanskrit source #2: yashnbx/gita_supersite_sanskrit_tts (701 Bhagavad-Gita shlokas, 3.3 h, EXACT verse text).
Each verse is CTC force-aligned (MMS 'hin' Devanagari adapter) to its reference text; verses > 14 s are cut at the longest breath pause
between words, so every clip is 2.5-14 s with a 150 ms tail and a *correct* text.  The text is the reference, never an ASR guess.
out: /work/tts/segments/vedic_chant/gss_*.wav, segments/segments.gss.jsonl, vedic/ref_text.json (+= path->text), vedic/gss_report.json"""
import io, json, os, re, sys, time
import numpy as np, pandas as pd, soundfile as sf, torch, torchaudio.functional as AF
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yt_asr_ensemble import ctc_align, dev_only
ROOT = os.environ.get("TTS_ROOT", "/work/tts"); os.environ.setdefault("HF_HOME", f"{ROOT}/hf"); OUT = f"{ROOT}/segments/vedic_chant"; os.makedirs(OUT, exist_ok=True)
MAXD, MIND, TAIL = 14.0, 2.5, 0.15
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def clean(t):
    t = re.sub(r"[।॥]+\s*[\d.\s]*[।॥]+", " । ", t); t = re.sub(r"[0-9०-९.]+", " ", t); t = re.sub(r"[^\u0900-\u097F\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def main():
    from transformers import Wav2Vec2ForCTC, AutoProcessor
    P = AutoProcessor.from_pretrained("facebook/mms-1b-all"); P.tokenizer.set_target_lang("hin")
    M = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="hin", ignore_mismatched_sizes=True, torch_dtype=torch.float16).cuda().eval()
    voc = P.tokenizer.get_vocab(); blank = voc["<pad>"]; sp = voc["|"]
    ddir = f"{ROOT}/data/gita_ss/data"; df = pd.concat([pd.read_parquet(f"{ddir}/{f}") for f in os.listdir(ddir) if f.endswith(".parquet")])
    refp = f"{ROOT}/vedic/ref_text.json"; REF = json.load(open(refp)); segs = []; stat = {"verses": 0, "clips": 0, "bad_align": 0, "hours": 0.0}
    for _, r in df.iterrows():
        txt = clean(r["text"]); words = txt.split()
        x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32"); x = x.mean(1) if x.ndim > 1 else x
        w16 = AF.resample(torch.from_numpy(x), sr, 16000) if sr != 16000 else torch.from_numpy(x); stat["verses"] += 1
        wi = [i for i, w in enumerate(words) if dev_only(w)]; toks = []; own = []
        for k, i in enumerate(wi):
            ids = [voc[c] for c in dev_only(words[i]) if c in voc]
            if not ids: continue
            if toks: toks.append(sp); own.append(-1)
            toks += ids; own += [i] * len(ids)
        with torch.no_grad():
            inp = P(w16.numpy(), sampling_rate=16000, return_tensors="pt"); lp = torch.log_softmax(M(inp.input_values.cuda().half()).logits.float(), -1)[0].cpu().numpy()
        T = min(lp.shape[0], int(np.ceil(len(w16) / 320))); al = ctc_align(lp[:T], toks, blank) if toks else None
        if al is None or al[1] < -2.0: stat["bad_align"] += 1; continue
        spans, mlp = al; wsp = {}
        for (a, b), o in zip(spans, own):
            if o >= 0: wsp[o] = [min(a, wsp.get(o, [a, b])[0]), max(b, wsp.get(o, [a, b])[1])]
        order = sorted(wsp); fr = 0.02; x24 = AF.resample(torch.from_numpy(x), sr, 24000).numpy(); dur = len(x24) / 24000
        cut = [0]; s0 = wsp[order[0]][0] * fr
        while True:   # greedy: next cut = largest inter-word gap whose segment length is in [6, MAXD] s
            st = wsp[order[cut[-1]]][0] * fr; best = None
            for j in range(cut[-1] + 1, len(order)):
                L = wsp[order[j - 1]][1] * fr - st
                if L > MAXD: break
                gap = wsp[order[j]][0] * fr - wsp[order[j - 1]][1] * fr
                if L >= 6.0 and (best is None or gap > best[0]): best = (gap, j)
            if best is None and wsp[order[-1]][1] * fr - st > MAXD:   # no 6 s+ piece fits: accept shorter pieces (>= 3 s)
                for j in range(cut[-1] + 1, len(order)):
                    L = wsp[order[j - 1]][1] * fr - st
                    if L > MAXD: break
                    gap = wsp[order[j]][0] * fr - wsp[order[j - 1]][1] * fr
                    if L >= 3.0 and (best is None or gap > best[0]): best = (gap, j)
            if wsp[order[-1]][1] * fr - st <= MAXD or best is None: break
            cut.append(best[1])
        cut.append(len(order))
        for k in range(len(cut) - 1):
            a, b = order[cut[k]], order[cut[k + 1] - 1]; t0 = max(0.0, wsp[a][0] * fr - 0.12) if k else max(0.0, wsp[a][0] * fr - 0.2)
            nxt = (wsp[order[cut[k + 1]]][0] * fr - 0.05) if cut[k + 1] < len(order) else dur; e = wsp[b][1] * fr
            t1 = min(max(min(nxt, e + 0.45), e + TAIL), dur); seg = x24[int(t0 * 24000):int(t1 * 24000)]
            if len(seg) / 24000 < MIND or len(seg) / 24000 > 14.6: continue
            tail = int((TAIL - (t1 - wsp[b][1] * fr)) * 24000)
            if tail > 0: seg = np.concatenate([seg, np.zeros(tail, "float32")])
            p = f"{OUT}/gss_{r['shloka_id']}_{k}.wav".replace(" ", "_"); sf.write(p, seg, 24000); text = " ".join(words[a:b + 1] if True else [])
            REF[p] = text; segs.append({"path": p, "cat": "vedic_chant", "src": "gss_gita", "t0": round(t0, 2), "dur": round(len(seg) / 24000, 2), "speech": round(len(seg) / 24000 - TAIL, 2), "align_lp": round(mlp, 3)})
            stat["clips"] += 1; stat["hours"] += len(seg) / 24000 / 3600
        if stat["verses"] % 100 == 0: log(stat)
    json.dump(REF, open(refp, "w"), ensure_ascii=False)
    with open(f"{ROOT}/segments/segments.gss.jsonl", "w") as f:
        for s in segs: f.write(json.dumps(s, ensure_ascii=False) + "\n")
    stat["hours"] = round(stat["hours"], 2); json.dump(stat, open(f"{ROOT}/vedic/gss_report.json", "w")); log("GSS_DONE", stat)

if __name__ == "__main__": main()
