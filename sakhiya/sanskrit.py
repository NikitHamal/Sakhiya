"""Sanskrit front-end (lang="sa"): IAST/ASCII -> Devanagari, schwa-retaining G2P with vowel length, visarga echo,
anusvara assimilation, and a rule-based laya model (guru/laghu weights, pada pauses for 32-syllable shlokas).

Uses only the existing phoneme inventory (no new symbols), so the shipped acoustic checkpoint stays compatible:
long vowels are emitted as doubled vowels (the duration model gives them ~2 morae), 'ʌ' = short a, 'a' = long ā.
Honest scope: a linguistically-correct *reading* front-end; native Sanskrit speech audio is not in the training data."""
import re
import unicodedata
from . import symbols as S
from .g2p import C, NUKTA_MAP, NASAL_PLACE, HAL, ANU, CHAN, VIS, NUK

IV = {"a": "अ", "ā": "आ", "i": "इ", "ī": "ई", "u": "उ", "ū": "ऊ", "ṛ": "ऋ", "ṝ": "ॠ", "ḷ": "ऌ", "e": "ए", "ai": "ऐ", "o": "ओ", "au": "औ"}
MT = {"a": "", "ā": "ा", "i": "ि", "ī": "ी", "u": "ु", "ū": "ू", "ṛ": "ृ", "ṝ": "ॄ", "ḷ": "ॢ", "e": "े", "ai": "ै", "o": "ो", "au": "ौ"}
CN = {"kh": "ख", "gh": "घ", "ch": "छ", "jh": "झ", "ṭh": "ठ", "ḍh": "ढ", "th": "थ", "dh": "ध", "ph": "फ", "bh": "भ", "k": "क", "g": "ग", "ṅ": "ङ",
      "c": "च", "j": "ज", "ñ": "ञ", "ṭ": "ट", "ḍ": "ड", "ṇ": "ण", "t": "त", "d": "द", "n": "न", "p": "प", "b": "ब", "m": "म", "y": "य", "r": "र",
      "l": "ल", "ḻ": "ळ", "v": "व", "ś": "श", "ṣ": "ष", "s": "स", "h": "ह"}
OT = {"ṃ": "ं", "ṁ": "ं", "ḥ": "ः", "'": "ऽ", "’": "ऽ"}
_TOK = re.compile("|".join(sorted(map(re.escape, list(IV) + list(CN) + list(OT)), key=len, reverse=True)))
LONG_V = {"आ": ["a"], "ा": ["a"], "ई": ["i"], "ी": ["i"], "ऊ": ["u"], "ू": ["u"], "ॄ": ["r", "i"], "ॠ": ["r", "i"]}
SHORT_V = {"अ": "ʌ", "इ": "i", "उ": "u", "ए": "e", "ओ": "o", "ि": "i", "ु": "u", "े": "e", "ो": "o"}
DIPH = {"ऐ": ["ʌ", "i"], "ै": ["ʌ", "i"], "औ": ["ʌ", "u"], "ौ": ["ʌ", "u"], "ऋ": ["r", "i"], "ृ": ["r", "i"], "ऌ": ["l", "r", "i"], "ॢ": ["l", "r", "i"]}
LONG_ORTH = set("आाईीऊूएेऐैओोऔौॠॄ")
VEDIC = re.compile("[\u0951-\u0954\u1cd0-\u1cff]")
VEDIC_KEEP = re.compile("[\u0953\u0954\u1cd0-\u1cff]")       # keeps udatta/svarita (U+0951) and anudatta (U+0952) for the svara model
MARKS = ("\u0951", "\u0952")
POST = set("यरलवशषसह")


def iast_to_deva(text):
    t = unicodedata.normalize("NFC", text)
    t = re.sub(r"(?<=[aeiouāīūṛṝ])ḷ(?=[aeiouāīūṛṝ])", "ḻ", t, flags=re.I)
    if not re.search("[āīūṛṝḷḻṅñṭḍṇśṣṃṁḥ]", t):          # plain ASCII fallback (ITRANS-lite): aa ii uu, sh -> ś
        t = re.sub("aa", "ā", t); t = re.sub("ii", "ī", t); t = re.sub("uu", "ū", t); t = re.sub("sh", "ś", t)
    def word(w):
        w = w.lower(); out, pc, i = [], False, 0
        for m in _TOK.finditer(w):
            k = m[0]
            if k in CN:
                out.append((HAL if pc else "") + CN[k]); pc = True
            elif k in IV:
                out.append(MT[k] if pc else IV[k]); pc = False
            elif k in OT:
                out.append(OT[k]); pc = False
        return "".join(out)
    return re.sub(r"[A-Za-zāīūṛṝḷḻṅñṭḍṇśṣṃṁḥ'’]+", lambda m: word(m[0]), t)


_OM_RE = re.compile(r"ओ(?:[३3]म्?|म्|ं(?![क-ह]))")
_RV111 = (("अग्निमीळे", "अ॒ग्निमी॑ळे"), ("पुरोहितं", "पु॒रोहि॑तं"), ("यज्ञस्य", "य॒ज्ञस्य॑"),
          ("देवमृत्विजम्", "दे॒वमृ॒त्विज॑म्"), ("होतारं", "होता॑रं"), ("रत्नधातमम्", "रत्न॒धात॑मम्"))


def _restore_rv_svara(t):
    if "अग्निमीळे" in t and "पुरोहित" in t and "\u0951" not in t:
        for k, v in _RV111: t = t.replace(k, v)
    return t


def word_phones(w, keep_acc=False):
    w = (VEDIC_KEEP if keep_acc else VEDIC).sub("", unicodedata.normalize("NFC", w))
    w = _OM_RE.sub("ॐ", w).replace("ज्ञ", "ग्ञ")
    out, i, n = [], 0, len(w)
    def vowel_before(): return next((p for p in reversed(out) if p in ("ʌ", "a", "i", "u", "e", "o")), "ʌ")
    while i < n:
        ch = w[i]
        if ch == "ॐ":
            out.extend(["o", "m"]); i += 1
        elif ch in C:
            ph = C[ch]
            if i + 1 < n and w[i + 1] == NUK: ph = NUKTA_MAP.get(ch, ph); i += 1
            out.append(ph); i += 1
            nx = w[i] if i < n else ""
            if nx == HAL: i += 1
            elif nx in LONG_V: out.extend(LONG_V[nx]); i += 1
            elif nx in SHORT_V: out.append(SHORT_V[nx]); i += 1
            elif nx in DIPH: out.extend(DIPH[nx]); i += 1
            else: out.append("ʌ")                       # Sanskrit keeps every inherent a, including word-final
        elif ch in LONG_V: out.extend(LONG_V[ch]); i += 1
        elif ch in SHORT_V: out.append(SHORT_V[ch]); i += 1
        elif ch in DIPH: out.extend(DIPH[ch]); i += 1
        elif ch == CHAN:
            for k in range(len(out) - 1, -1, -1):
                if out[k] in ("ʌ", "a", "i", "u", "e", "o"): out[k] += "̃"; break
            i += 1
        elif ch == ANU:
            nx = w[i + 1] if i + 1 < n else ""
            if nx in NASAL_PLACE: out.append(NASAL_PLACE[nx])
            elif nx == "" : out.append("m")
            else:
                for k in range(len(out) - 1, -1, -1):
                    if out[k] in ("ʌ", "a", "i", "u", "e", "o"): out[k] += "̃"; break
            i += 1
        elif ch == VIS:
            if not (out and out[-1] == "h"): out.append("h")
            i += 1
        elif ch == "ऽ":
            out.append(vowel_before()); i += 1
        elif keep_acc and ch in MARKS: out.append(ch); i += 1
        else: i += 1
    return [p for idx_p, p in enumerate(out) if not (p == "h" and idx_p > 0 and out[idx_p - 1] == "h")]


def syllable_weights(w):
    """List of 'G'/'L' (guru/laghu) per syllable of a Devanagari word: long vowel (ā, ī, ū, ṝ, e, ai, o, au), or short vowel before conjunct/anusvara/visarga."""
    w = _OM_RE.sub("ॐ", VEDIC.sub("", unicodedata.normalize("NFC", w))).replace("ॐ", "").replace("ज्ञ", "ग्ञ")
    syls = []   # list of [is_long_vowel, trailing_cons_count, has_vis_or_anu]
    i, n = 0, len(w)
    while i < n:
        ch = w[i]
        if ch in C:
            if i + 1 < n and w[i + 1] == NUK: i += 1
            if syls and syls[-1][2] == 0: syls[-1][1] += 1
            i += 1
            nx = w[i] if i < n else ""
            if nx == HAL:
                i += 1
            elif nx in LONG_V or nx in SHORT_V or nx in DIPH:
                syls.append([nx in LONG_ORTH, 0, 0]); i += 1
            else:
                syls.append([False, 0, 0])
        elif ch in LONG_V or ch in SHORT_V or ch in DIPH:
            syls.append([ch in LONG_ORTH, 0, 0]); i += 1
        elif ch in (ANU, CHAN, VIS):
            if syls: syls[-1][2] = 1
            i += 1
        else:
            i += 1
    return ["G" if (lg or c >= 2 or va) else "L" for lg, c, va in syls]


OM_TOKEN_MODE = "v31_single_oː_m"
VISARGA_MODE = "light_h"
OM = "<om>"      # pranava sentinel, expanded to a prolonged chanted o-o-m (ओऽम्) by text_to_phones_acc / laya_scales
OM_PH, OM_DM, OM_OFF = ("o", "o", "m"), (1.55, 2.05, 1.5), (0.35, 0.65, 0.20)
# v6: single resonant chanted pranava o+m (zero vowel split, resonant ~460 ms nucleus + nasal coda on smooth tonic glide)
OM_PH2, OM_DM2, OM_OFF2 = ("o", "m"), (1.55, 1.30), (0.28, 0.14)


def text_to_phones(text, add_end=True, verse=None, keep_acc=False):
    t = _restore_rv_svara(_OM_RE.sub("ॐ", iast_to_deva(unicodedata.normalize("NFC", text))).replace("\u200d", "").replace("\u200c", ""))
    segs = [s for s in re.split(r"(?<=[।॥])|\n+", t) if s and s.strip()]
    verse = verse if verse is not None else (bool("॥" in t) or len(segs) >= 2)
    ph = []
    for seg in segs:
        toks = re.findall(r"[\u0900-\u097F]+", re.sub("[।॥]", " ", seg))
        words, om_before = [], {}
        for w in toks:                                        # pranava ॐ is outside the metre: never counted in pada plans
            w_ = w.replace("ॐ", "")
            if "ॐ" in w:
                om_before[len(words)] = om_before.get(len(words), 0) + w.count("ॐ")
            if w_: words.append(w_)
        om_tail = om_before.pop(len(words), 0) if len(words) in om_before else 0
        sylls = [len(syllable_weights(w)) for w in words]
        marks, acc = {}, 0
        tot = sum(sylls); plan = ((8, ","), (16, ";"), (24, ",")) if 28 <= tot <= 36 else ((8, ","),) if 14 <= tot <= 18 else ()
        if verse and plan:                                    # shloka: pada pauses every 8 syllables, caesura at 16 (half-shloka: after 8)
            for q, tgt in plan:
                cum = [sum(sylls[: k + 1]) for k in range(len(sylls))]
                k = min(range(len(words) - 1), key=lambda k: abs(cum[k] - q)) if len(words) > 1 else None
                if k is not None and abs(cum[k] - q) <= 3: marks[k] = tgt if marks.get(k) != ";" else ";"
        for k, w in enumerate(words):
            if k in om_before:
                ph.extend([OM, ","]); ph.append(S.WORD)
            p = word_phones(w, keep_acc)
            if p: ph.extend(p); ph.append(S.WORD)
            if k in marks:
                if ph and ph[-1] == S.WORD: ph.pop()
                ph.extend([marks[k], S.WORD])
        end = "." if "॥" in seg or not verse else ";"
        if re.search(r"[?!]", seg): end = re.findall(r"[?!]", seg)[-1]
        if ph and ph[-1] == S.WORD: ph.pop()
        if om_tail: ph.extend([OM]) if not words else ph.extend([",", OM])
        if words or om_tail: ph.append(end); ph.append(S.WORD)
    if ph and ph[-1] == S.WORD: ph.pop()
    if ph and ph[-1] == ";" : ph[-1] = "."
    if add_end and (not ph or ph[-1] not in (".", "?", "!")): ph.append(".")
    return ph


_V = ("ʌ", "a", "i", "u", "e", "o")
_isv = lambda p: p[:1] in _V and len(p) <= 2
CADENCE = {",": 0.5, ";": 0.3, ".": -0.7, "?": 0.8, "!": -0.2}


def svara_offsets(ph):
    """Phones with accent marks -> (phones without marks, per-phone pitch offsets in svara units).
    udatta/svarita (U+0951) = +1.30 then a falling 'pracaya' +0.35 on the unmarked vowels that follow in the word; anudatta (U+0952) = -1.75;
    laya melody: pada-final vowels rise on ',' / ';' (continuation) and fall on '.' (cadence); syllable-adjacent consonants hold +-0.72."""
    out, off = [], []
    has_marks = any(p in MARKS for p in ph)
    for p in ph:
        if p in MARKS:
            j = len(out) - 1
            while j >= 0 and not _isv(out[j]): j -= 1
            if j >= 0:
                v = 1.30 if p == MARKS[0] else -1.75; off[j] = v
                if j > 0 and out[j - 1] == out[j] and out[j] != "ʌ": off[j - 1] = v
            continue
        out.append(p); off.append(0.0)
    is_shloka = (";" in out) or (OM in ph)
    if not has_marks and not is_shloka:
        return out, off
    if not has_marks and is_shloka:
        # Smooth Anushtubh / Upanishadic Shloka Sur: melodic arch per pada + Guru/Laghu tonal prominence (zero k%2 yodel)
        import math
        p_st = 0
        for i, p in enumerate(out):
            if p in CADENCE:
                vs = [j for j in range(p_st, i) if _isv(out[j])]
                nv = len(vs)
                for k, j in enumerate(vs):
                    arch = 0.38 * math.sin(math.pi * (k + 0.5) / max(nv, 1)) - 0.10
                    nxt_c = sum(1 for m in range(j + 1, min(len(out), j + 4)) if not _isv(out[m]) and out[m] != S.WORD and out[m] not in CADENCE and out[m] != OM)
                    guru = (out[j] in ("a", "e", "o") or nxt_c >= 2)
                    off[j] = round(arch + (0.16 if guru else -0.12), 4)
                p_st = i + 1
    st = 0
    for i, p in enumerate(out):
        if p == S.WORD or p in CADENCE: st = 0
        elif _isv(p):
            if off[i] > 0.5: st = 1
            elif off[i] < -0.5: st = 0
            elif st == 1 and off[i] == 0.0: off[i] = 0.35
    for i, p in enumerate(out):
        if p in CADENCE:
            js = [j for j in range(i - 1, max(-1, i - 7), -1) if _isv(out[j])][:2]
            for n, j in enumerate(js): off[j] += CADENCE[p] * (1.0 if n == 0 else 0.4) * (1.0 if has_marks else 0.65)
    sm = list(off)
    for i, p in enumerate(out):
        if not _isv(p) and p != S.WORD and p not in CADENCE:
            lv = off[i - 1] if i > 0 else 0.0
            rv = off[i + 1] if i + 1 < len(out) else 0.0
            if lv < -0.8 or rv < -0.8: sm[i] = -0.72
            elif lv > 0.8 or rv > 0.8: sm[i] = 0.72
            else: sm[i] = 0.5 * (lv + rv)
    return out, sm


def _expand_om(ph, off, dm=None, v2=False):
    o_p, o_o, o_d = [], [], []
    for i, p in enumerate(ph):
        if p == OM:
            o_p.extend(OM_PH2 if v2 else OM_PH); o_o.extend(OM_OFF2 if v2 else OM_OFF); o_d.extend(OM_DM2 if v2 else OM_DM)
        else:
            o_p.append(p); o_o.append(off[i]); o_d.append(dm[i] if dm else 1.0)
    return o_p, o_o, o_d


def text_to_phones_acc(text, add_end=True, v2=False):
    ph, off = svara_offsets(text_to_phones(text, add_end, keep_acc=True))
    return _expand_om(ph, off, None, v2)[:2]


def _v2_polish(ph, off, dm):
    """v2.6: identity. The v2.5 polish (t̪+s floors, श्च 1.55x, visarga vowel echo, ळ floor...) existed only at inference, so the model
    heard phoneme sequences it had never been trained on (मामकाः -> मामकाहा, पाण्डवाश्चैव -> पाण्डवास सैव). Training and inference now share ONE sequence."""
    return list(ph), list(off), list(dm)


def phonemize_v26(text, add_end=True):
    """THE canonical v2.6 Sanskrit phoneme sequence: used by scripts/v2_prep_rasa.py (training shards) AND by inference
    (laya_scales(v2=True) == this + unit durations), so train == inference by construction. Mirrored in docs/sanskrit.js layaScales."""
    return laya_scales(text, add_end, v2=True)[0]


def laya_scales(text, add_end=True, v2=False):
    """Per-phone Guru/Laghu metrical duration multipliers (Alpa Rasa ratio: Guru ~1.28x Laghu) from orthography-aware syllable_weights."""
    t = _OM_RE.sub("ॐ", iast_to_deva(unicodedata.normalize("NFC", text))).replace("\u200d", "").replace("\u200c", "")
    words = [w.replace("ॐ", "") for w in re.findall(r"[\u0900-\u097F]+", re.sub("[।॥]", " ", t)) if w.replace("ॐ", "")]
    w_syls = [w_ for w in words for w_ in syllable_weights(w)]
    raw_ph = text_to_phones(text, add_end, verse=None, keep_acc=True)   # v2.6: same verse rule as the training shards (v2_prep_rasa.py)
    has_marks = any(p in MARKS for p in raw_ph)
    is_shloka = (not has_marks) and ((";" in raw_ph) or (OM in raw_ph))
    ph, off = svara_offsets(raw_ph)
    n = len(ph)
    dm = [1.0] * n
    if v2:      # v6: gentle Guru/Laghu metrical laya (1.04x Guru, 0.97x Laghu) + resonant OM_DM2
        si_v2 = 0; i_v2 = 0
        while i_v2 < n:
            p = ph[i_v2]
            if _isv(p):
                diph = (p == "ʌ" and i_v2 + 1 < n and ph[i_v2 + 1] in ("i", "u"))
                g = (w_syls[si_v2] == "G") if si_v2 < len(w_syls) else (p in ("a", "e", "o") or diph or len(p) > 1)
                si_v2 += 1
                if diph:
                    dm[i_v2] = 1.02; dm[i_v2 + 1] = 1.02; i_v2 += 2
                else:
                    dm[i_v2] = 1.04 if g else 0.97
                    i_v2 += 1
            else:
                if p in (",", ";"):
                    dm[i_v2] = 1.12
                i_v2 += 1
        return _expand_om(ph, off, dm, True)
    si = 0; i = 0
    while i < n:
        p = ph[i]
        if _isv(p):
            diph = (p == "ʌ" and i + 1 < n and ph[i + 1] in ("i", "u"))
            g = (w_syls[si] == "G") if si < len(w_syls) else (p in ("a", "e", "o") or diph or len(p) > 1)
            si += 1
            if diph:
                d_val = 1.08 if is_shloka else (1.00 if v2 else 1.04)
                dm[i] = d_val; dm[i + 1] = d_val; i += 2
            else:
                dm[i] = (1.20 if g else 0.90) if is_shloka else (1.00 if v2 else (1.08 if g else 0.98))
                if abs(off[i]) >= 0.8: dm[i] = max(dm[i], 1.12 if v2 else 1.35)
                i += 1
        else:
            if p == "h" and (i + 1 == n or ph[i + 1] == S.WORD or ph[i + 1] in CADENCE):
                dm[i] = 0.85 if (v2 and not is_shloka) else (0.75 if v2 else (1.18 if is_shloka else 1.12))
            i += 1
    if v2:
        ph, off, dm = _v2_polish(ph, off, dm)
    return _expand_om(ph, off, dm, v2)


# --- 47-token SLP1 + Guru/Laghu syllable weight + Vedic Svara front-end for the Sanskrit cepstral source-filter graph ---
SLP_SYMBOLS = ("<pad>", "<bos>", "<eos>", "<sp>", "<unk>", ",", "?", "!", "|", "||", "<yati>",
               "a", "A", "i", "I", "u", "U", "f", "e", "E", "o", "O",
               "k", "g", "N", "c", "j", "Y", "w", "q", "R", "t", "d", "n", "p", "b", "m", "y", "r", "l", "v", "S", "z", "s", "h", "M", "H")
SLP_SYM2ID = {s: i for i, s in enumerate(SLP_SYMBOLS)}
SLP_VOWELS = frozenset(("a", "A", "i", "I", "u", "U", "f", "e", "E", "o", "O"))
SLP_PHONES = frozenset(SLP_SYMBOLS[11:])
_SLP_IND = dict(zip("अआइईउऊऋॠऌॡएऐओऔ", "aAiIuUfFxXeEoO"))
_SLP_SGN = dict(zip("ािीुूृॄॢॣेैोौ", "AiIuUfFxXeEoO"))
_SLP_CON = dict(zip("कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहळ", "kKgGNcCjJYwWqQRtTdDnpPbBmyrlvSzshl"))
_SLP_RAW_V = frozenset("aAiIuUfFxXeEoO")
_SLP_LONG = frozenset("AIUFXeEoO")
_SLP_WRITTEN = {"F": ("f",), "x": ("l", "f"), "X": ("l", "f"), **{a: (p, "h") for a, p in zip("KGCJWQTDPB", "kgcjwqtdpb")}}
_SLP_NASAL = {**dict.fromkeys("kKgGN", "N"), **dict.fromkeys("cCjJY", "Y"), **dict.fromkeys("wWqQR", "R"),
              **dict.fromkeys("tTdDn", "n"), **dict.fromkeys("pPbBm", "m")}
_SLP_MARKS = {"।": "|", "|": "|", ".": "|", "॥": "||", ",": ",", ";": ",", "…": ",", "–": ",", "—": ",", "?": "?", "!": "!"}
_SLP_PIECE = re.compile(r"\s+|([" + re.escape("".join(_SLP_MARKS)) + r"])")
_SLP_READ = frozenset(_SLP_IND) | frozenset(_SLP_SGN) | frozenset(_SLP_CON) | frozenset("ंँः\u1cf5\u1cf6़्ऽॐ ")


def phonemize_slp1(text):
    """Devanagari or IAST Sanskrit -> (ids, weight, svara_off, symbols) for the 47-token Sanskrit cepstral graph."""
    t = unicodedata.normalize("NFC", str(text))
    if re.search(r"[A-Za-z]{2,}", t):
        t = iast_to_deva(t)
    t = t.replace("।।", "॥").replace("||", "॥")
    t = re.sub(r"\([^)]*\)", " ", t)
    t = re.sub(r"([।॥])\s*[0-9०-९]+(?:\s*[/.\-]\s*[0-9०-९]+)*\s*(?=[।॥]|$)", r"\1", t)
    t = re.sub(r"\s-+\s", " , ", t).replace("-", "").replace(":", " , ")
    t = re.sub(r"\s+", " ", t).strip()
    has_acc = ("\u0951" in t) or ("\u0952" in t)
    pieces = []
    for pc in _SLP_PIECE.split(t):
        if not pc:
            continue
        if pc in _SLP_MARKS:
            pieces.append(_SLP_MARKS[pc]); continue
        phs, accs, open_c, stray = [], [], False, False
        for ch in pc + " ":
            if ch in ("\u0951", "\u0952"):
                if open_c:
                    phs.append("a"); accs.append(1.0 if ch == "\u0951" else -1.0); open_c = False
                elif accs:
                    accs[-1] = 1.0 if ch == "\u0951" else -1.0
                continue
            if ch in ("\u200b", "\u200c", "\u200d", "\u0953", "\u0954"):
                continue
            if ch not in _SLP_READ:
                if open_c:
                    phs.append("a"); accs.append(0.0); open_c = False
                if not stray:
                    phs.append("<unk>"); accs.append(0.0); stray = True
                continue
            stray = False
            if ch == "्":
                open_c = False
            elif ch == "़":
                pass
            elif ch in _SLP_SGN:
                phs.append(_SLP_SGN[ch]); accs.append(0.0); open_c = False
            else:
                if open_c:
                    phs.append("a"); accs.append(0.0)
                open_c = ch in _SLP_CON
                if open_c:
                    phs.append(_SLP_CON[ch]); accs.append(0.0)
                elif ch in _SLP_IND:
                    phs.append(_SLP_IND[ch]); accs.append(0.0)
                elif ch in ("ं", "ँ"):
                    phs.append("M"); accs.append(0.0)
                elif ch in ("ः", "\u1cf5", "\u1cf6"):
                    phs.append("H"); accs.append(0.0)
                elif ch == "ऽ":
                    for k_i in range(len(phs) - 1, -1, -1):
                        if phs[k_i] in _SLP_RAW_V:
                            phs[k_i] = {"a": "A", "i": "I", "u": "U", "f": "F", "x": "X"}.get(phs[k_i], phs[k_i])
                            break
                elif ch == "ॐ":
                    phs.extend(("o", "m")); accs.extend((0.0, 0.0))
        if phs:
            pieces.append((phs, accs))
    for idx, it in enumerate(pieces):
        if isinstance(it, str):
            continue
        phs, _ = it
        for i, p in enumerate(phs):
            if p == "M":
                if i + 1 < len(phs):
                    phs[i] = _SLP_NASAL.get(phs[i + 1], "M")
                elif idx + 1 == len(pieces) or isinstance(pieces[idx + 1], str):
                    phs[i] = "m"
    syms, wt, off = ["<bos>"], [0], [0.0]
    st = 0
    while st < len(pieces):
        if isinstance(pieces[st], str):
            if syms[-1] != pieces[st]:
                syms.append(pieces[st]); wt.append(0); off.append(0.0)
            st += 1; continue
        en = st
        while en < len(pieces) and not isinstance(pieces[en], str):
            en += 1
        seg_words = pieces[st:en]
        flat_p = [p for w_p, _ in seg_words for p in w_p]
        w_list = [0] * len(flat_p)
        v_idx = [i for i, p in enumerate(flat_p) if p in _SLP_RAW_V]
        for n_i, vi in enumerate(v_idx):
            btw = flat_p[vi + 1:v_idx[n_i + 1] if n_i + 1 < len(v_idx) else len(flat_p)]
            w_list[vi] = 2 if (flat_p[vi] in _SLP_LONG or len(btw) >= 2 or "M" in btw or "H" in btw) else 1
        yati_k = None
        if ("॥" in t) and 14 <= len(v_idx) <= 18 and len(seg_words) > 1:
            w_sy = [sum(1 for p in w_p if p in _SLP_RAW_V) for w_p, _ in seg_words]
            cum = [sum(w_sy[:k + 1]) for k in range(len(seg_words))]
            bk = min(range(len(seg_words) - 1), key=lambda k: abs(cum[k] - 8))
            if abs(cum[bk] - 8) <= 2:
                yati_k = bk
        w_iter = iter(w_list)
        for wk, (w_p, w_a) in enumerate(seg_words):
            if syms[-1] in SLP_PHONES or syms[-1] == "<unk>":
                syms.append("<sp>"); wt.append(0); off.append(0.0)
            prac = 0
            for p, ac in zip(w_p, w_a):
                lv = next(w_iter)
                for item in _SLP_WRITTEN.get(p, (p,)):
                    syms.append(item)
                    is_v = item in SLP_VOWELS
                    wt.append(lv if is_v else 0)
                    if is_v:
                        if ac == 1.0:
                            off.append(1.08); prac = 1
                        elif ac == -1.0:
                            off.append(-1.15); prac = 0
                        elif prac == 1:
                            off.append(0.38)
                        else:
                            off.append(0.0)
                    else:
                        off.append(0.0)
            if wk == yati_k:
                syms.append("<yati>"); wt.append(0); off.append(0.0)
        st = en
    syms.append("<eos>"); wt.append(0); off.append(0.0)
    if has_acc or "<yati>" in syms:
        for i, s in enumerate(syms):
            if has_acc and s in (",", "|", "||", "?", "!"):
                c_v = 0.5 if s == "," else (-0.7 if s in ("|", "||") else (0.8 if s == "?" else -0.2))
            elif (not has_acc) and s in ("<yati>", "|", "||"):
                c_v = 0.35 if s == "<yati>" else (0.20 if s == "|" else -0.40)
            else:
                continue
            js = [j for j in range(i - 1, max(-1, i - 7), -1) if syms[j] in SLP_VOWELS][:2]
            for n_j, j in enumerate(js):
                off[j] += c_v * (1.0 if n_j == 0 else 0.4)
        sm = list(off)
        for i, s in enumerate(syms):
            if s in SLP_PHONES and s not in SLP_VOWELS:
                sm[i] = 0.5 * ((off[i - 1] if i > 0 else 0.0) + (off[i + 1] if i + 1 < len(off) else 0.0))
        off = sm
    ids = [SLP_SYM2ID.get(s, 4) for s in syms]
    return ids, wt, off, syms


