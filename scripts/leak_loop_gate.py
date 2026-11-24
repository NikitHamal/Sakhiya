#!/usr/bin/env python3
"""Prompt-Leak, Loop & Splice Gate (v5 recovery).

Every rendered WAV (Nepali, Vedic, Episode, student-corpus utterance) must pass before it is shipped or enters a corpus.
A clip is REJECTED when ANY of these is true:
  * LEAK    - ASR hypothesis contains a word/phrase of ref_text that is NOT in target_text (prompt-tail leakage)
  * LOOP    - a word / akshara n-gram repeats >= 2 consecutive times in the hypothesis more than the target allows
              (e.g. "नम् नम्", "अग्नि अग्नि")
  * SPLICE  - hard mid-word discontinuity: >25 dB envelope step inside 6 ms between voiced levels, or a hard-cut tail
  * GASP    - noise-like breath burst before the first voiced onset (leading gasp)
  * CLICKS  - more than --max-clicks (default 2) sample-level clicks
  * OM      - Vedic mode, target starts with ॐ and the first voiced run is shorter than 0.45 s (collapsed "ओ")
ASR is pluggable: --hyp-json {wav: hypothesis}, or sp_common.kriti() on the GPU box, or none (acoustic-only).

Manifest (JSON list): [{"wav": ..., "target_text": ..., "ref_text": ..., "mode": "nepali"|"vedic"}]
Exit code 0 only when every clip passes.  Pure functions are unit-tested in tests/test_leak_loop_gate.py.
"""
import argparse, difflib, json, os, re, sys
import numpy as np

PUNCT = re.compile(r'[।॥,?!.\-–—:;"\'()॰ऽ\u0951-\u0954\u200c\u200d\u2018\u2019\u201c\u201d]')
OM_MIN_S = 0.45


def norm_tokens(t):
    t = (t or "").replace("ॐ", " ॐ ")
    t = PUNCT.sub(" ", t)
    return [w for w in re.sub(r"\s+", " ", t).strip().split(" ") if w]


# akshara = consonant cluster (C + virama)* C + optional vowel sign / nasal / visarga, or independent vowel / ॐ
_AKS = re.compile(r"(?:[क-ह\u0958-\u095f](?:\u094d(?=[क-ह\u0958-\u095f]))?)+[\u093e-\u094c\u0901-\u0903\u093c\u094d]*"
                  r"|[अ-औॠॐ][\u0901-\u0903]*")


def aksharas(t):
    return _AKS.findall("".join(norm_tokens(t)))


def _sim(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def leaked_words(ref_text, target_text, hyp, fuzzy=0.84):
    """Words that appear in hyp, belong to ref_text, and do not belong to target_text."""
    ref, tgt, hw = set(norm_tokens(ref_text)), norm_tokens(target_text), norm_tokens(hyp)
    only_ref = [r for r in ref if r not in tgt and not any(_sim(r, t) >= fuzzy for t in tgt)]
    out = []
    for h in hw:
        if h in tgt or any(_sim(h, t) >= fuzzy for t in tgt):
            continue
        if any(h == r or _sim(h, r) >= fuzzy for r in only_ref):
            out.append(h)
    return out


def max_repeat(seq, max_n=4):
    """Longest run of consecutive repeats of any n-gram (n<=max_n). Returns (repeats, ngram)."""
    best, arg = 1, None
    L = len(seq)
    for n in range(1, max_n + 1):
        i = 0
        while i + n <= L:
            r = 1
            while i + (r + 1) * n <= L and seq[i + r * n:i + (r + 1) * n] == seq[i:i + n]:
                r += 1
            if r > best:
                best, arg = r, " ".join(seq[i:i + n])
            i += max(1, n * r if r > 1 else 1)
    return best, arg


def loop_report(target_text, hyp, min_rep=2):
    """Loops in hyp not allowed by the target. Checks words and aksharas."""
    out = []
    for name, th, hh in (("word", norm_tokens(target_text), norm_tokens(hyp)), ("akshara", aksharas(target_text), aksharas(hyp))):
        tr, _ = max_repeat(th)
        hr, ng = max_repeat(hh)
        # akshara-level repeats of >=3 are required (natural reduplications like "तत् त" are common); words need >=2
        need = min_rep if name == "word" else max(min_rep + 1, 3)
        if hr >= need and hr > tr:
            out.append(dict(kind=name, ngram=ng, repeats=hr))
    return out


# ------------------------------------------------------------------ acoustic checks
def _frames_db(w, sr, ms=3.0):
    n = max(8, int(sr * ms / 1000))
    k = len(w) // n
    if k < 4:
        return np.full(4, -120.0), n
    x = w[:k * n].reshape(k, n)
    return 20 * np.log10(np.sqrt((x ** 2).mean(1)) + 1e-9), n


def count_clicks(w, sr):
    w = w.astype(np.float32)
    pk = float(np.abs(w).max()) if len(w) else 0.0
    if pk < 1e-4:
        return 0
    d = np.abs(np.diff(w))
    mask = d > max(0.35 * pk, 0.12)
    # count isolated events (merge neighbours within 2 ms)
    idx = np.flatnonzero(mask)
    if not len(idx):
        return 0
    return int(1 + (np.diff(idx) > int(0.002 * sr)).sum())


def find_splices(w, sr, step_db=25.0, voiced_db=-24.0):
    """Hard envelope steps between voiced levels (a cut that lands mid-word) within ~6 ms. Returns times (s)."""
    w = w.astype(np.float32)
    db, n = _frames_db(w, sr, 3.0)
    out = []
    for i in range(2, len(db) - 2):
        a, b = db[i - 1], db[i + 1]          # level just before vs just after (one 3 ms frame skipped to allow a partial frame)
        if a - b >= step_db and a > voiced_db:
            # real word-final decay never drops 25 dB in 6 ms; require the loud side to be sustained 30 ms prior
            lo = max(0, i - 10)
            if np.median(db[lo:i]) > voiced_db - 6 and db[max(0, i - 6):i - 2].min() > a - 12:   # not a gradual fade
                out.append(round(i * n / sr, 3))
        elif b - a >= step_db and b > voiced_db and np.median(db[i:i + 10]) > voiced_db - 6 and a > -75:
            # abrupt onset out of NON-silent floor (spliced into the middle of other audio)
            out.append(round(i * n / sr, 3))
    merged = []
    for t in out:
        if not merged or t - merged[-1] > 0.05:
            merged.append(t)
    # a hard cut at the very end is covered by the tail check below
    end_db = db[-3:].max()
    if len(db) > 10 and end_db > voiced_db + 8 and w.size and abs(w[-1]) > 0.08:
        merged.append(round(len(w) / sr, 3))
    return merged


def leading_gasp(w, sr):
    """Noise-like breath burst (high spectral flatness, high ZCR) ahead of the first voiced onset."""
    import librosa
    w = w.astype(np.float32)
    db, n = _frames_db(w, sr, 10.0)
    thr = max(db.max() - 38.0, -62.0)
    on = np.flatnonzero(db > thr)
    if not len(on):
        return False
    first = on[0] * n / sr
    seg = w[int(max(0.0, first) * sr):int((first + 0.35) * sr)]
    if len(seg) < int(0.06 * sr):
        return False
    flat = float(librosa.feature.spectral_flatness(y=seg, n_fft=512, hop_length=128).mean())
    zcr = float(librosa.feature.zero_crossing_rate(seg, frame_length=512, hop_length=128).mean())
    peak_ratio = float(db[on[0]:on[0] + 35].max() - db.max())
    # a gasp is noisy and quiet-ish relative to the body of the clip, and noisy means flat spectrum + high ZCR
    return bool(flat > 0.22 and zcr > 0.20 and peak_ratio < -6.0)


def first_voiced_run_s(w, sr, bridge_ms=80, voiced_rel_db=-30.0):
    db, n = _frames_db(w.astype(np.float32), sr, 10.0)
    thr = db.max() + voiced_rel_db
    v = db > thr
    idx = np.flatnonzero(v)
    if not len(idx):
        return 0.0
    gap = max(1, int(bridge_ms / 10))
    start = idx[0]; end = start
    for i in idx[1:]:
        if i - end <= gap:
            end = i
        else:
            break
    return float((end - start + 1) * n / sr)


def load_wav(p):
    import soundfile as sf
    w, sr = sf.read(p, dtype="float32")
    return (w if w.ndim == 1 else w.mean(1)), sr


def check_clip(w, sr, target_text, ref_text, hyp=None, mode="nepali", max_clicks=2):
    rej = []
    clicks = count_clicks(w, sr)
    if clicks > max_clicks:
        rej.append(f"CLICKS:{clicks}")
    sp = find_splices(w, sr)
    if sp:
        rej.append(f"SPLICE@{sp[:3]}")
    if leading_gasp(w, sr):
        rej.append("GASP")
    om = None
    if mode == "vedic" and (target_text or "").strip().startswith("ॐ"):
        om = first_voiced_run_s(w, sr)
        if om < OM_MIN_S:
            rej.append(f"OM_COLLAPSED:{om:.2f}s")
    leaks, loops = [], []
    if hyp is not None:
        leaks = leaked_words(ref_text or "", target_text, hyp)
        loops = loop_report(target_text, hyp)
        if leaks:
            rej.append(f"LEAK:{leaks[:4]}")
        if loops:
            rej.append(f"LOOP:{loops[:2]}")
    return dict(ok=not rej, rejects=rej, clicks=clicks, splices=sp, om_s=om, leaks=leaks, loops=loops, hyp=hyp,
                dur=round(len(w) / sr, 2))


def _asr_batch(paths):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sakhiya_pro"))
    import sp_common as C
    return {p: C.kriti().read_conf([p])[0][0] for p in paths}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--hyp-json")
    ap.add_argument("--no-asr", action="store_true")
    ap.add_argument("--max-clicks", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    man = json.load(open(a.manifest, encoding="utf8"))
    hyps = json.load(open(a.hyp_json, encoding="utf8")) if a.hyp_json else {}
    if not hyps and not a.no_asr:
        try:
            hyps = _asr_batch([m["wav"] for m in man])
        except Exception as e:  # acoustic-only fallback must be explicit, never silent
            print("ASR unavailable, acoustic-only gate:", repr(e)[:120], file=sys.stderr)
    res, bad = [], 0
    for m in man:
        w, sr = load_wav(m["wav"])
        r = check_clip(w, sr, m["target_text"], m.get("ref_text", ""), hyps.get(m["wav"], m.get("hyp")), m.get("mode", "nepali"), a.max_clicks)
        r["wav"] = m["wav"]; res.append(r); bad += (not r["ok"])
        print(("PASS " if r["ok"] else "REJECT ") + os.path.basename(m["wav"]), "" if r["ok"] else r["rejects"])
    rep = dict(n=len(res), rejected=bad, passed=len(res) - bad, clips=res)
    if a.out:
        json.dump(rep, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"GATE {len(res) - bad}/{len(res)} passed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
