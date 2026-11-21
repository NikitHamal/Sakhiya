#!/usr/bin/env python3
"""Stage 4a: GPU ASR ensemble (Whisper-large-v3-turbo + facebook/mms-1b-all CTC [npi / hin adapters]) with cross-model Devanagari
agreement and exact CTC forced alignment (own numpy Viterbi over MMS emissions, 20 ms frames) of the Whisper transcript.
in : /work/tts/gate/passed.jsonl   out: /work/tts/asr/asr.jsonl  (text_w, text_m, conf_w, conf_m, agree, align_lp, words[{w,t0,t1,lp}], min_word_frames)"""
import json, os, re, sys, time, unicodedata
from concurrent.futures import ThreadPoolExecutor
import numpy as np, soundfile as sf, torch, torchaudio.functional as AF, jiwer

ROOT = os.environ.get("TTS_ROOT", "/work/tts"); os.environ.setdefault("HF_HOME", f"{ROOT}/hf"); dev = "cuda"
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
DEV_RE = re.compile(r"[\u0900-\u097F]")
def dev_only(t):
    t = unicodedata.normalize("NFC", t).replace("ॐ", "ओम").replace("ँ", "ं").replace("\u200c", "").replace("\u200d", "").replace("़", "")
    return "".join(c for c in t if DEV_RE.match(c) and c not in "।॥")

def ctc_align(lp, toks, blank):
    """lp (T,C) log-probs, toks list[int] -> per-token (start,end) frames + mean token logprob, or None."""
    T, L = lp.shape[0], len(toks); S = 2 * L + 1
    if L == 0 or T < L + sum(1 for i in range(1, L) if toks[i] == toks[i - 1]): return None
    ext = np.full(S, blank); ext[1::2] = toks
    skip = np.zeros(S, bool); skip[3::2] = np.array(toks[1:]) != np.array(toks[:-1]) if L > 1 else False; skip[1] = False
    dp = np.full(S, -1e30); dp[0] = lp[0, blank]; dp[1] = lp[0, toks[0]]; bp = np.zeros((T, S), np.int8)
    for t in range(1, T):
        c1 = np.concatenate([[-1e30], dp[:-1]]); c2 = np.concatenate([[-1e30, -1e30], dp[:-2]]); c2 = np.where(skip, c2, -1e30)
        st = np.stack([dp, c1, c2]); k = st.argmax(0); bp[t] = k; dp = st[k, np.arange(S)] + lp[t, ext]
    s = S - 1 if dp[S - 1] >= dp[S - 2] else S - 2; path = np.zeros(T, int)
    for t in range(T - 1, -1, -1): path[t] = s; s -= int(bp[t, s])
    spans = [[None, None] for _ in range(L)]; tot = 0.0
    for t, s in enumerate(path):
        if s % 2 == 1:
            i = s // 2; spans[i][0] = t if spans[i][0] is None else spans[i][0]; spans[i][1] = t + 1; tot += lp[t, ext[s]]
    if any(a is None for a, _ in spans): return None
    return spans, float(tot / max(1, sum(b - a for a, b in spans)))

def main():
    from transformers import WhisperForConditionalGeneration, WhisperProcessor, Wav2Vec2ForCTC, AutoProcessor
    wp = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
    wm = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo", torch_dtype=torch.float16).to(dev).eval()
    mp = {}; mm = {}
    for key, ad in (("ne", "npi"), ("sa", "hin")):
        mp[key] = AutoProcessor.from_pretrained("facebook/mms-1b-all"); mp[key].tokenizer.set_target_lang(ad)
        mm[key] = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang=ad, ignore_mismatched_sizes=True, torch_dtype=torch.float16).to(dev).eval()
    rows = [json.loads(l) for l in open(f"{ROOT}/gate/passed.jsonl")]; os.makedirs(f"{ROOT}/asr", exist_ok=True); out = f"{ROOT}/asr/asr.jsonl"
    done = {json.loads(l)["path"] for l in open(out)} if os.path.exists(out) else set()
    rows = [r for r in rows if r["path"] not in done]
    # v6.1 owner fix: vedic clips must have passed scripts/vedic_purity_gate.py (real Sanskrit chanting, not Dravidian/English talk)
    vok = f"{ROOT}/vedic/ok_paths.json"; REF = json.load(open(f"{ROOT}/vedic/ref_text.json")) if os.path.exists(f"{ROOT}/vedic/ref_text.json") else {}
    okp = set(json.load(open(vok))["ok_paths"]) if os.path.exists(vok) else None
    rows = [r for r in rows if r["cat"] != "vedic_chant" or r["path"] in REF or (okp is not None and r["path"] in okp)]
    rows.sort(key=lambda r: (r["cat"] == "vedic_chant", r["dur"])); log(len(rows), "clips")
    ex = ThreadPoolExecutor(12); B = 24
    for i in range(0, len(rows), B):
        bt = rows[i:i + B]; k = "sa" if bt[0]["cat"] == "vedic_chant" else "ne"; bt = [r for r in bt if (r["cat"] == "vedic_chant") == (k == "sa")]
        if not bt: continue
        ws = list(ex.map(lambda r: AF.resample(torch.from_numpy(sf.read(r["path"], dtype="float32")[0]), 24000, 16000).numpy(), bt))
        with torch.no_grad():
            f = wp(ws, sampling_rate=16000, return_tensors="pt").input_features.to(dev).half()
            g = wm.generate(f, language=k, task="transcribe", num_beams=1, max_new_tokens=225, return_dict_in_generate=True, output_scores=True)
            ts = wm.compute_transition_scores(g.sequences, g.scores, normalize_logits=True).float()
            txt = wp.batch_decode(g.sequences, skip_special_tokens=True)
            gen = g.sequences[:, -ts.shape[1]:]; valid = (gen != wm.config.pad_token_id) & (gen < 50257)
            cw = ((ts.exp() * valid).sum(1) / valid.sum(1).clamp(min=1)).cpu().tolist()
            inp = mp[k](ws, sampling_rate=16000, return_tensors="pt", padding=True, return_attention_mask=True)
            lg = mm[k](inp.input_values.to(dev).half(), attention_mask=inp.attention_mask.to(dev)).logits.float()
            lpb = torch.log_softmax(lg, -1).cpu().numpy(); tm = lg.argmax(-1).cpu()
        voc = mp[k].tokenizer.get_vocab(); inv = {v: c for c, v in voc.items()}; blank = voc["<pad>"]; sp = voc["|"]
        res = []
        for j, r in enumerate(bt):
            T = int(np.ceil(len(ws[j]) / 320)); T = min(T, lpb.shape[1]); lp = lpb[j, :T]
            ids = [x for x, y in zip(tm[j, :T].tolist(), [-1] + tm[j, :T].tolist()[:-1]) if x != y and x != blank]
            tm_txt = "".join(inv[x] for x in ids).replace("|", " ").strip(); pm = float(np.exp(lp.max(-1)[lp.argmax(-1) != blank]).mean()) if (lp.argmax(-1) != blank).any() else 0.0
            tw = txt[j].strip(); refd = r["path"] in REF
            if refd: tw = REF[r["path"]]   # studio clip: the dataset's own Devanagari text is the transcript (Whisper/MMS only cross-check)
            words = [w for w in re.split(r"\s+", re.sub(r"[^\u0900-\u097F\s]", " ", tw.replace("ॐ", "ओम"))) if dev_only(w)]
            agree = 1 - min(1.0, jiwer.cer(dev_only(tm_txt) or "x", dev_only(tw) or "y")) if dev_only(tw) else 0.0
            wd, wt = [], []
            for wi, w in enumerate(words):
                cs = [c for c in unicodedata.normalize("NFC", w).replace("ँ", "ं") if c in voc]
                if wi: wt.append(sp)
                wt += [voc[c] for c in cs]; wd.append((w, len(cs)))
            al = ctc_align(lp, wt, blank) if wt else None; wl, mn, alp = [], 0, -9.0
            if al:
                sps, alp = al; p = 0
                for wi, (w, n) in enumerate(wd):
                    if wi: p += 1
                    a, b = sps[p][0], sps[p + n - 1][1]; wl.append({"w": w, "t0": round(a * .02, 3), "t1": round(b * .02, 3)}); p += n
                mn = min(w["t1"] - w["t0"] for w in wl) / .02 if wl else 0
            res.append({**r, "text_w": tw, "text_m": tm_txt, "conf_w": round(cw[j], 3), "conf_m": round(pm, 3), "agree": round(agree, 3),
                        "align_lp": round(alp, 3), "words": wl, "min_word_frames": mn, "lang": k, "ref": refd})
        with open(out, "a") as fo:
            for r in res: fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        if (i // B) % 10 == 0: log(f"asr {i+len(bt)}/{len(rows)}  last agree={res[-1]['agree']} {res[-1]['text_w'][:50]}")

if __name__ == "__main__": main()
