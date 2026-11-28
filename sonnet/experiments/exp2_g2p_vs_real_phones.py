#!/usr/bin/env python3
"""EXP2 (Sonnet lane): is our G2P target what real Nepali speakers actually say?
Differential test with a language-agnostic phone recogniser (facebook/wav2vec2-lv-60-espeak-cv-ft):
  PER(real audio phones , G2P phones)   = G2P-vs-human mismatch  (+ recogniser noise)
  PER(synth audio phones, G2P phones)   = recogniser noise + how faithfully our AM follows its own G2P
If real >> synth, the G2P target itself deviates from human speech.  Everything is collapsed to coarse classes
(vowel identity + consonant manner/place-agnostic) so that IPA-inventory differences do not dominate.
Also reports vowel-count ratio (schwa deletion bias): vowels(G2P)/vowels(recogniser) on real audio.
usage: python exp2_g2p_vs_real_phones.py REPO OUT_JSON [N]
"""
import os, sys, json, io, re, time, warnings, collections
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, librosa, torch
REPO, OUT = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 60
sys.path.insert(0, REPO); SR = 24000; T0 = time.time()
def log(*a): print(f"[{time.time()-T0:5.0f}s]", *a, flush=True)
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
tok = open("/tmp/hf_token").read().strip()
tb = pq.read_table(hf_hub_download("ai4bharat/Rasa", "Nepali/test-00000-of-00007.parquet", repo_type="dataset", token=tok)).to_pandas()
d = tb[tb["style"].isin(["WIKI", "NEWS", "BOOK", "CONV"]) & tb.text.str.len().between(35, 120)].sample(n=N, random_state=1)
from sakhiya import g2p
from sakhiya.onnx_tts import SakhiyaONNX
m = SakhiyaONNX(os.path.join(REPO, "docs/models_sakhiya_one_vocos"), tiny=False, threads=4)
from transformers import Wav2Vec2ForCTC
# the stock tokenizer needs the espeak library even for decoding; decode CTC ids by hand from vocab.json instead
VOCAB = json.load(open(hf_hub_download("facebook/wav2vec2-lv-60-espeak-cv-ft", "vocab.json", token=tok)))
ID2P = {v: k for k, v in VOCAB.items()}
class P:
    @staticmethod
    def batch_decode(ids):
        out = []
        for row in ids:
            prev, toks = None, []
            for i in row.tolist():
                if i != prev and ID2P.get(i, "<pad>") not in ("<pad>", "<s>", "</s>", "<unk>", "|"): toks.append(ID2P[i])
                prev = i
            out.append(" ".join(toks))
        return out
R = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft", token=tok).eval()
torch.set_num_threads(16)
def recog(w, sr):
    x = librosa.resample(w.astype(np.float32), orig_sr=sr, target_sr=16000)
    x = (x - x.mean()) / (x.std() + 1e-7)
    with torch.no_grad(): lg = R(torch.from_numpy(x)[None]).logits
    return P.batch_decode(lg.argmax(-1))[0].split()
VOW = {"a": "a", "ɑ": "a", "æ": "a", "ɐ": "A", "ʌ": "A", "ə": "A", "ɜ": "A", "ɘ": "A", "i": "i", "ɪ": "i", "ɨ": "i", "y": "i", "u": "u", "ʊ": "u", "ʉ": "u",
       "e": "e", "ɛ": "e", "ɜ": "A", "o": "o", "ɔ": "o", "ø": "o", "ɤ": "o"}
CONS = {"p": "p", "b": "b", "t": "t", "d": "d", "ʈ": "t", "ɖ": "d", "k": "k", "g": "g", "ɡ": "g", "q": "k", "m": "m", "n": "n", "ɳ": "n", "ŋ": "n", "ɲ": "n", "ɱ": "m",
        "s": "s", "ʃ": "s", "ʂ": "s", "ɕ": "s", "z": "z", "ʒ": "z", "h": "h", "ɦ": "h", "f": "f", "v": "w", "w": "w", "ʋ": "w", "j": "j", "l": "l", "ɭ": "l", "ɫ": "l",
        "r": "r", "ɾ": "r", "ɽ": "r", "ɹ": "r", "ʁ": "r", "c": "c", "ɟ": "c", "x": "k", "ɣ": "g", "θ": "t", "ð": "d", "β": "b", "ʔ": ""}
def cls(ph):
    ph = ph.replace("ː", "").replace("̃", "").replace("̪", "").replace("̥", "").replace("ʰ", "").replace("ʱ", "").replace("ʲ", "").replace("ˑ", "")
    if ph in ("tʃ", "ts", "dʒ", "dz", "t͡ʃ", "d͡ʒ", "t͡s", "d͡z", "tʂ", "dʐ"): return ["c"]
    out = []
    for ch in ph:
        if ch in VOW: out.append(VOW[ch])
        elif ch in CONS and CONS[ch]: out.append(CONS[ch])
    # diphthongs in recogniser output like "ai" would already be split by chars -> 2 vowels (kept)
    return out
def seq(phs): return [c for p in phs for c in cls(p) if c]
def edit(a, b):
    dp = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(b) + 1):
            cur = dp[j]; dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1])); prev = cur
    return dp[-1]
isv = lambda c: c in "aAiueo"
rows = []; sub_AI = collections.Counter()
for r in d.itertuples():
    ref_g2p = seq([p for p in g2p.text_to_phones(r.text, "ne") if p not in ("|", ".", ",", "?", "!", ";")])
    wr, sr = sf.read(io.BytesIO(r.audio["bytes"]), dtype="float32"); wr = wr.mean(1) if wr.ndim > 1 else wr
    hr = seq(recog(wr, sr))
    ws, ssr = m.say(r.text, speaker="sakhiya", lang="ne", style="news"); ws = np.asarray(ws, np.float32).reshape(-1)
    hs = seq(recog(ws, ssr))
    ev = lambda s: [c for c in s if isv(c)]
    rows.append(dict(txt=r.text, n_g2p=len(ref_g2p),
        per_real=edit(ref_g2p, hr) / max(1, len(ref_g2p)), per_synth=edit(ref_g2p, hs) / max(1, len(ref_g2p)),
        per_real_vs_synth=edit(hs, hr) / max(1, len(hr)),
        vper_real=edit(ev(ref_g2p), ev(hr)) / max(1, len(ev(ref_g2p))), vper_synth=edit(ev(ref_g2p), ev(hs)) / max(1, len(ev(ref_g2p))),
        nv_g2p=len(ev(ref_g2p)), nv_real=len(ev(hr)), nv_synth=len(ev(hs)),
        nA_g2p=ref_g2p.count("A"), nA_real=hr.count("A"), nA_synth=hs.count("A"),
        dur_real=len(wr) / sr, dur_synth=len(ws) / ssr, nchars=len(r.text)))
agg = {k: float(np.mean([x[k] for x in rows])) for k in rows[0] if k != "txt"}
agg["vowel_count_ratio_g2p_over_real"] = float(sum(x["nv_g2p"] for x in rows) / sum(x["nv_real"] for x in rows))
agg["schwa_count_ratio_g2p_over_real"] = float(sum(x["nA_g2p"] for x in rows) / max(1, sum(x["nA_real"] for x in rows)))
agg["dur_ratio_synth_over_real"] = float(sum(x["dur_synth"] for x in rows) / sum(x["dur_real"] for x in rows))
agg["secs_per_char_real"] = float(sum(x["dur_real"] for x in rows) / sum(x["nchars"] for x in rows))
agg["secs_per_char_synth"] = float(sum(x["dur_synth"] for x in rows) / sum(x["nchars"] for x in rows))
json.dump(dict(N=N, agg=agg, rows=rows[:20]), open(OUT, "w"), ensure_ascii=False, indent=1)
log(json.dumps(agg, indent=1))
