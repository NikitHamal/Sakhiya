#!/usr/bin/env python3
"""EXP6 (Sonnet lane, session 2): word-LEVEL accent probe WITH a control, to settle the Rasa-accent dispute.

Gemini EXP5 counted every [b] vs [v/w] emitted before /i,j/ ANYWHERE in 20 whole clips (19 events, 1 female speaker,
no control corpus) and concluded "Rasa = 89.5 % Indian [v]".  That measures neither the word nor the recogniser bias.

Here: (1) CTC-force-align each transcript with MMS-1b-all (npi, grapheme CTC) to get word boundaries;
(2) crop each target word (-60 ms / +40 ms); (3) run facebook/wav2vec2-lv-60-espeak-cv-ft on the crop only and read the
FIRST CONSONANT label; (4) do it identically for Rasa F, Rasa M and Nepal-recorded controls (SLR43 female, SLR54).
Word classes: VI (वि-, Kathmandu [bi]), BI (ब-, recogniser positive control for [b]), SW (स्व-, [sw] control),
CHA (च-), CHH (छ-), GY (ज्ञ-containing words: Kathmandu [gj]).
usage: exp6_accent_control.py CORPUS NTHREADS MAXCLIPS   CORPUS in rasa_F rasa_M slr43 slr54
"""
import os, sys, io, re, json, glob, time, warnings, zipfile
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, librosa, torch
import torchaudio.functional as AF
CORP, NT, MAXC = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
torch.set_num_threads(NT)
os.environ.setdefault("HF_HOME", "/work/sonnet/hf")
tok = open("/tmp/hf_token").read().strip()
OUT = f"/work/sonnet/s2/out/exp6_{CORP}.jsonl"
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)

CLASSES = [("VI", r"^वि"), ("BI", r"^ब"), ("SW", r"^स्व"), ("CHA", r"^च"), ("CHH", r"^छ"), ("GY", r"ज्ञ")]
def classify(w):
    for n, p in CLASSES:
        if re.search(p, w): return n
    return None

# ---------------- corpus iterators: yield (id, wave16k, text)
def load16(b_or_path):
    w, sr = sf.read(io.BytesIO(b_or_path) if isinstance(b_or_path, bytes) else b_or_path, dtype="float32")
    if w.ndim > 1: w = w.mean(1)
    return librosa.resample(w, orig_sr=sr, target_sr=16000) if sr != 16000 else w
def clean_words(t):
    t = re.sub(r"[।॥,.?!\-–—\"'“”‘’():;|/\d०-९%]", " ", t)
    return [w for w in t.split() if w]
def iter_rows():
    if CORP.startswith("rasa"):
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        gen = "female" if CORP.endswith("F") else "male"
        files = [hf_hub_download("ai4bharat/Rasa", f"Nepali/test-0000{i}-of-00007.parquet", repo_type="dataset", token=tok) for i in range(int(os.environ.get("RASA_SHARDS", "1")))]
        for f in files:
            df = pq.read_table(f).to_pandas()
            df = df[df.gender.str.lower() == gen]
            for r in df.itertuples():
                yield r.Index, r.text, (lambda b=r.audio["bytes"]: load16(b))
    elif CORP == "slr43":
        D = "/work/sonnet/s2/data/slr43"
        idx = [l.rstrip("\n").split("\t") for l in open(f"{D}/line_index_female.tsv", encoding="utf8")]
        for p in idx:
            if len(p) < 3: continue
            yield p[0], p[-1], (lambda f=f"{D}/{p[0]}.wav": load16(f))
    elif CORP == "slr54":
        D = "/work/sonnet/s2/data/slr54"
        for l in open(f"{D}/utt_spk_text.tsv", encoding="utf8"):
            p = l.rstrip("\n").split("\t")
            if len(p) < 3: continue
            f = f"{D}/asr_nepali/data/{p[0][:2]}/{p[0]}.flac"
            if os.path.exists(f): yield p[0], p[2], (lambda f=f: load16(f))

# ---------------- models
from transformers import AutoProcessor, Wav2Vec2ForCTC
proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="npi", token=tok)
mms = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="npi", ignore_mismatched_sizes=True, token=tok).eval()
vocab = proc.tokenizer.get_vocab(); blank = vocab.get("<pad>", 0)
rec = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft", token=tok).eval()
rv = json.load(open(__import__("huggingface_hub").hf_hub_download("facebook/wav2vec2-lv-60-espeak-cv-ft", "vocab.json", token=tok)))
id2p = {v: k for k, v in rv.items()}
SPECIAL = {"<pad>", "<s>", "</s>", "<unk>", "|", ""}
VOW = set("aeiouɑɐəɔɛɪʊɨʌæɒɜɞøyœɯɤɵʉ")
def phones(wav):
    x = (wav - wav.mean()) / (wav.std() + 1e-7)
    with torch.no_grad(): lg = rec(torch.from_numpy(x)[None]).logits[0]
    ids = lg.argmax(-1).tolist(); out = []; prev = None
    for i in ids:
        p = id2p.get(i, "")
        if i != prev and p not in SPECIAL: out.append(p)
        prev = i
    return out
def first_cons(ph):
    for p in ph:
        if p and p[0] not in VOW and p not in ("ː", "ˑ", "ʰ", "ʱ", "̃"): return p
    return None

def align_words(wav, words):
    """grapheme CTC forced alignment -> per-word (t_start, t_end, mean_score) seconds, or None"""
    chars, widx = [], []
    for wi, w in enumerate(words):
        for ch in w:
            if ch in vocab: chars.append(vocab[ch]); widx.append(wi)
        if wi < len(words) - 1 and "|" in vocab: chars.append(vocab["|"]); widx.append(-1)
    if not chars: return None
    x = proc(wav, sampling_rate=16000, return_tensors="pt").input_values
    with torch.no_grad(): lp = torch.log_softmax(mms(x).logits, -1)
    T = lp.shape[1]
    if T < len(chars) + 2: return None
    try:
        al, sc = AF.forced_align(lp, torch.tensor([chars], dtype=torch.int32), blank=blank)
    except Exception: return None
    spans = AF.merge_tokens(al[0], sc[0].exp(), blank=blank)
    if len(spans) != len(chars): return None
    fr = len(wav) / 16000 / T
    res = {}
    for s, wi in zip(spans, widx):
        if wi < 0: continue
        a = res.setdefault(wi, [s.start * fr, s.end * fr, []]); a[1] = s.end * fr; a[2].append(s.score)
    return {k: (v[0], v[1], float(np.mean(v[2]))) for k, v in res.items()}

done = 0; per = {k: 0 for k, _ in CLASSES}; CAP = {"VI": 60, "BI": 40, "SW": 25, "CHA": 40, "CHH": 40, "GY": 25}
fo = open(OUT, "w", encoding="utf8"); nclip = 0
for rid, text, getw in iter_rows():
    words = clean_words(text)
    tg = [(i, w, classify(w)) for i, w in enumerate(words)]
    tg = [(i, w, c) for i, w, c in tg if c and per[c] < CAP[c]]
    if not tg: continue
    wav = getw()
    if not (0.8 < len(wav) / 16000 < 25): continue
    al = align_words(wav, words)
    nclip += 1
    if al is None: continue
    for i, w, c in tg:
        if i not in al: continue
        a, b, sc = al[i]
        seg = wav[max(0, int((a - 0.06) * 16000)): int((b + 0.04) * 16000)]
        if len(seg) < 4800: continue
        ph = phones(seg); per[c] += 1
        fo.write(json.dumps({"corp": CORP, "id": str(rid), "cls": c, "word": w, "dur": round(b - a, 3), "ali_score": round(sc, 3), "ph": ph, "first": first_cons(ph)}, ensure_ascii=False) + "\n"); fo.flush()
    if nclip % 10 == 0: log("clips", nclip, per)
    if nclip >= MAXC or all(per[k] >= CAP[k] for k in per): break
log("DONE", nclip, per)
