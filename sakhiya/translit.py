"""Romanized Nepali -> Devanagari: typist-proof skeleton lexicon snap + rule fallback."""
import os
import re

HERE = os.path.dirname(__file__)
RC = {"chh": "छ", "ch": "च", "kh": "ख", "gh": "घ", "jh": "झ", "th": "थ", "dh": "ध", "ph": "फ", "bh": "भ",
      "sh": "श", "k": "क", "g": "ग", "c": "क", "j": "ज", "t": "त", "d": "द", "n": "न", "p": "प", "b": "ब",
      "m": "म", "y": "य", "r": "र", "l": "ल", "w": "व", "v": "व", "s": "स", "h": "ह", "f": "फ", "z": "ज",
      "q": "क", "x": "छ"}
RV = ["aa", "ai", "au", "ee", "ii", "oo", "uu", "a", "i", "u", "e", "o"]
IND = {"a": "अ", "aa": "आ", "i": "इ", "ii": "ई", "ee": "ई", "u": "उ", "uu": "ऊ", "oo": "ऊ", "e": "ए",
       "ai": "ऐ", "o": "ओ", "au": "औ"}
MAT = {"a": "", "aa": "ा", "i": "ि", "ii": "ी", "ee": "ी", "u": "ु", "uu": "ू", "oo": "ू", "e": "े",
       "ai": "ै", "o": "ो", "au": "ौ"}
KEYMAP = {"ʈ": "t̪", "ʈʰ": "t̪ʰ", "ɖ": "d̪", "ɖʱ": "d̪ʱ", "ɳ": "n", "ŋ": "n", "ɲ": "n", "ʃ": "s", "w": "b",
          "ɽ": "r", "ɽʱ": "r", "ɔ": "o", "ɛ": "e", "z": "dz", "f": "pʰ"}
_lex = None


def _tok(r):
    r = r.lower(); i = 0; out = []
    while i < len(r):
        for L in (3, 2, 1):
            seg = r[i:i + L]
            if L == 3 and seg == "chh" or L == 2 and (seg in RC or seg in RV) or \
               L == 1 and (seg in RC or seg in RV):
                out.append(seg); i += L; break
        else:
            i += 1
    return out


def roman_key(r):
    toks = _tok(r.replace("~", "")); k = []
    for idx, t in enumerate(toks):
        if t in RC:
            from .g2p import C
            ph = C[RC[t]]
            k.append(KEYMAP.get(ph, ph))
        elif t in ("i", "ii", "ee"):
            k.append("i")
        elif t in ("u", "uu", "oo"):
            k.append("u")
        elif t == "au":
            k.append("au")
        elif t == "e":
            k.append("e")
        elif t == "ai":
            k.append("ai")
        elif t == "o":
            k.append("o")
        elif t in ("a", "aa") and (idx == 0 or idx == len(toks) - 1):
            k.append("a")
    return " ".join(k)


def dev_key(w):
    from .g2p import word_to_phones
    ps = [p.replace("̃", "") for p in word_to_phones(w)]
    k = []; i = 0
    while i < len(ps):
        p = ps[i]
        if p == "ʌ" and i + 1 < len(ps) and ps[i + 1] in ("i", "u"):
            k.append("a" + ps[i + 1]); i += 2; continue
        p = KEYMAP.get(p, p)
        if p in ("ʌ", "a"):
            if i == 0 or i == len(ps) - 1:
                k.append("a")
        else:
            k.append(p)
        i += 1
    return " ".join(k)


def _read(name):
    path = os.path.join(HERE, "data", name)
    return open(path, encoding="utf8").read().splitlines() if os.path.exists(path) else []


def build_keys(words):
    """words: iterable of (devanagari, freq) -> {skeleton key: devanagari} (most frequent wins)."""
    best = {}
    for w, f in words:
        k = dev_key(w)
        if k and (k not in best or f > best[k][1]):
            best[k] = (w, f)
    return {k: v[0] for k, v in best.items()}


def _load(lang="ne"):
    """Assets (adapted from NikitHamal/transliterationlm): 110k-word Nepali frequency lexicon, 750 exact chat-roman
    overrides and the Tharu (Dangaura / Rana / Chitwania) roman->Devanagari lexicons with chat variants."""
    global _lex
    if _lex is None:
        keyed = _read("lexkey_ne.tsv")
        if keyed:
            ne = dict(ln.split("\t")[:2] for ln in keyed)
        else:
            ne = build_keys((l.split("\t")[0], int(l.split("\t")[1])) for l in _read("lex_ne.tsv") if l.count("\t"))
        direct = {a: b for a, b in (l.split("\t")[:2] for l in _read("direct_ne.tsv") if l.count("\t"))}
        direct.update({"gaun": "गाउँ", "gau": "गाउँ", "gaoon": "गाउँ", "gaaun": "गाउँ", "gaon": "गाउँ",
                       "gaunma": "गाउँमा", "gauma": "गाउँमा", "gaoonma": "गाउँमा", "gaaunma": "गाउँमा", "gaonma": "गाउँमा",
                       "gaunko": "गाउँको", "gaunle": "गाउँले", "gaupalika": "गाउँपालिका", "gaaupaalika": "गाउँपालिका",
                       "thau": "ठाउँ", "thaun": "ठाउँ", "thaoon": "ठाउँ", "thauma": "ठाउँमा", "thaunma": "ठाउँमा", "nau": "नाउँ",
                       "baschu": "बस्छु", "baschhu": "बस्छु", "auchu": "आउँछु", "aauchu": "आउँछु", "janchu": "जान्छु",
                       "khanchu": "खान्छु", "garchu": "गर्छु", "bhanchu": "भन्छु", "huncha": "हुन्छ", "hunchha": "हुन्छ",
                       "hudaina": "हुँदैन", "chaina": "छैन", "chhaina": "छैन", "chau": "छौ", "chhau": "छौ", "chan": "छन्"})
        _lex = dict(ne=ne, direct=direct)
    t = {"thr": "thr_dangaura", "thr_dangaura": "thr_dangaura", "thr_rana": "thr_rana", "thr_chitwania": "thr_chitwania", "thr_chit": "thr_chitwania"}.get(lang)
    if t and t not in _lex:
        ex = {}
        for l in _read(t + ".tsv"):
            if l.count("\t"):
                a, b = l.split("\t")[:2]
                ex.setdefault(a, b)
        km = {}
        for a, b in sorted(ex.items(), key=lambda kv: abs(len(kv[0]) - len(kv[1]))):
            rk = roman_key(a)
            if rk: km.setdefault(rk, b)
        _lex[t] = dict(exact=ex, key=km)
    return _lex, t


def rules(r):
    toks = _tok(r); out = []; prev_c = False
    for j, t in enumerate(toks):
        if t in RC:
            if prev_c:
                out.append("्")
            out.append(RC[t]); prev_c = True
        else:
            out.append(MAT[t] if prev_c else IND[t]); prev_c = False
    return "".join(out)


SFX = {"haru": "हरू", "bata": "बाट", "sanga": "सँग", "lai": "लाई", "ma": "मा", "me": "मे", "ko": "को", "ka": "का", "ki": "की",
       "le": "ले", "ke": "के", "ker": "केर", "kar": "कर", "hen": "हेन", "se": "से", "sana": "सना", "tak": "तक", "dekhi": "देखि"}


def _lookup(w, k, lex, t):
    if t:
        hit = lex[t]["exact"].get(w) or lex[t]["key"].get(k)
        if hit: return hit
    if w in lex["direct"]: return lex["direct"][w]
    return lex["ne"].get(k) if k else None


def roman_to_deva(word, lang="ne"):
    if re.search(r"[\u0900-\u097F]", word):
        return word
    has_tilde = "~" in word
    lex, t = _load(lang); w = re.sub(r"[^a-z]", "", word.lower()); k = roman_key(w)
    def split():
        for sf in sorted(SFX, key=len, reverse=True):
            if w.endswith(sf) and len(w) - len(sf) >= 2:
                st = w[: -len(sf)]; h = _lookup(st, roman_key(st), lex, t)
                if h:
                    if has_tilde and "~" in word[:-len(sf)] and not h.endswith("ँ"): h += "ँ"
                    return h + SFX[sf]
    if t:                                   # Tharu: dialect lexicon (exact), then stem+clitic, then skeletons
        hit = lex[t]["exact"].get(w) or lex[t]["exact"].get(word.lower()) or split() or _lookup(w, k, lex, t)
    else:
        hit = _lookup(w, k, lex, t) or split()
    out = hit or rules(w)
    if has_tilde and "ँ" not in out and "ं" not in out:
        out += "ँ"
    return out
