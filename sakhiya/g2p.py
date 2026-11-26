"""Nepali grapheme-to-phoneme with morphology-aware schwa handling.

Key differences from Hindi-style G2P (which we deliberately do NOT use):
  * medial inherent schwas are RETAINED (kʌsʌri, sʌmʌj, bʌdʌl) -- no Hindi VC_CV deletion;
  * word-final inherent schwa after a simple consonant is dropped (gʱʌr, kam, nam);
  * word-final schwa after a conjunct IS pronounced (d̪ʱʌrmʌ, gʌrtsʰʌ, ʌnt̪ʌ, mʌnt̪rʌ);
  * monosyllables keep the schwa (tsʰʌ 'is', nʌ, mʌ 'I', rʌ 'and');
  * stem + case-postposition / plural boundaries are pronounced stem-first, so the stem's
    final schwa is not resurrected before the suffix (dzʌs+le -> dzʌsle, ghʌr+le -> gʱʌrle,
    t̪jʌs+pʌtsʰi -> t̪jʌspʌtsʰi) while sʌmʌj+ma keeps its internal schwa.
"""
import re
import unicodedata
from . import numbers, symbols as S

HAL, ANU, CHAN, VIS, NUK = "्", "ं", "ँ", "ः", "़"
V31_G2P = True
C = {"क": "k", "ख": "kʰ", "ग": "g", "घ": "gʱ", "ङ": "ŋ", "च": "ts", "छ": "tsʰ", "ज": "dz", "झ": "dzʱ",
     "ञ": "ɲ", "ट": "ʈ", "ळ": "l", "ठ": "ʈʰ", "ड": "ɖ", "ढ": "ɖʱ", "ण": "ɳ", "त": "t̪", "थ": "t̪ʰ", "द": "d̪",
     "ध": "d̪ʱ", "न": "n", "प": "p", "फ": "pʰ", "ब": "b", "भ": "bʱ", "म": "m", "य": "j", "र": "r",
     "ल": "l", "व": "w", "श": "ʃ", "ष": "ʃ", "स": "s", "ह": "h"}
NUKTA_MAP = {"ड": "ɽ", "ढ": "ɽʱ", "ज": "dz", "फ": "pʰ", "क": "k", "ख": "kʰ", "ग": "g"}
IV = {"अ": "ʌ", "आ": "a", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ए": "e", "ओ": "o", "ऑ": "o"}
IV2 = {"ऐ": ["ʌ", "i"], "औ": ["ʌ", "u"], "ऋ": ["r", "i"]}
MATRA = {"ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "े": "e", "ो": "o", "ॉ": "o"}
MATRA2 = {"ै": ["ʌ", "i"], "ौ": ["ʌ", "u"], "ृ": ["r", "i"]}
NASAL_PLACE = {"क": "ŋ", "ख": "ŋ", "ग": "ŋ", "घ": "ŋ", "च": "n", "छ": "n", "ज": "n", "झ": "n", "ट": "n", "ठ": "n",
               "ड": "n", "ढ": "n", "त": "n", "थ": "n", "द": "n", "ध": "n", "प": "m", "फ": "m", "ब": "m", "भ": "m"}
VOWELS = set(S.VOW) | set(S.NASV)

SUFFIXES = sorted("""हरूलाई हरूले हरूको हरूका हरूमा हरूबाट हरूसँग हरूतिर हरूभित्र हरूबाहिर हरूमाथि हरू हरुलाई हरुले हरुको हरुका हरुमा हरुबाट हरुसँग हरुतिर हरु
ज्यूहरूलाई ज्यूहरूले ज्यूहरूको ज्यूलाई ज्यूले ज्यूको ज्यूका ज्यूमा ज्यूबाट ज्यूसँग ज्यू ज्युलाई ज्युले ज्युको ज्यु हुक्रे हुक्रेन
लाई बाटै बाट देखि सम्म सँगै सँग संग सँगे संगे ले को का की मा मे से के हेन सन लग केर कर पछि अघि पहिले
भित्र बाहिर माथि तल नजिक भरि भर पट्टि पटि जस्तो जस्तै जति मात्र पनि नै चाहिँ चाहिं पिच्छे वरिपरि बारे विरुद्ध अनुसार समेत बित्तिकै अगाडि पछाडि प्रति तर्फ पूर्वक घर स्थान स्थल पत्र""".split(),
                  key=len, reverse=True)


def _parse(w, lang="ne"):
    """-> list of (phoneme, kind); kind in C (consonant), V (vowel), S (pending inherent schwa)."""
    out, i, n = [], 0, len(w)
    def nasalize():
        for k in range(len(out) - 1, -1, -1):
            if out[k][1] in "VS":
                ph = "ʌ" if out[k][0] == "ʌ?" else out[k][0]
                out[k] = (ph + "̃" if not ph.endswith("̃") else ph, "V")
                # Khatiwada (2009): nasal diphthongs (औँ, ऐँ, आउँ, आइँ, ओउँ, एउँ) nasalize both nucleus and glide
                if ph in ("i", "u", "ĩ", "ũ") and k >= 1 and out[k - 1][1] in "VS":
                    p0 = "ʌ" if out[k - 1][0] == "ʌ?" else out[k - 1][0]
                    if p0 in ("ʌ", "a", "o", "e"):
                        out[k - 1] = (p0 + "̃", "V")
                return
    while i < n:
        ch = w[i]
        if ch in C:
            ph = C[ch]
            if i + 1 < n and w[i + 1] == NUK:
                ph = NUKTA_MAP.get(ch, ph); i += 1
            out.append((ph, "C")); i += 1
            if i < n and w[i] == HAL:
                i += 1
            elif i < n and (w[i] in MATRA or w[i] in MATRA2):
                if w[i] in MATRA:
                    out.append((MATRA[w[i]], "V"))
                else:
                    out.extend((p, "V") for p in MATRA2[w[i]])
                i += 1
            else:
                out.append(("ʌ?", "S"))
        elif ch in IV:
            v_ph = IV[ch]
            if i > 0 and out and out[-1][1] == "V" and out[-1][0].rstrip("̃") == v_ph:
                out.append((S.WORD, "C"))  # explicit hiatus boundary for reduplicated vowels (आआफ्नो -> a | a pʰ n o)
            out.append((v_ph, "V")); i += 1
        elif ch in IV2:
            seq = IV2[ch]
            if ch in ("औ", "ऐ") and out and out[-1][1] == "V" and out[-1][0] in ("a", "ã"):
                seq = seq[1:]  # जाऔँ -> dz ã ũ, खाऔँ -> kʰ ã ũ (no 3-vowel a+ʌ+u hiatus)
            out.extend((p, "C" if p == "r" else "V") for p in seq); i += 1
        elif ch == CHAN:
            nasalize(); i += 1
        elif ch == ANU:
            nxt = w[i + 1] if i + 1 < n else ""
            if nxt in NASAL_PLACE:
                # keep preceding schwa (cluster follows), insert homorganic nasal
                if out and out[-1][1] == "S":
                    out[-1] = ("ʌ", "V")
                out.append((NASAL_PLACE[nxt], "C"))
            elif nxt in C:
                nasalize()
            else:
                nasalize()
            i += 1
        elif ch == "ॐ":
            out.extend([("o", "V"), ("m", "C")]); i += 1
        elif ch == VIS:
            out.append(("h", "C")); i += 1
        else:
            i += 1  # ZWJ etc.
    return out


# Native Nepali words & suffixes that ALWAYS retain word-final inherent schwa 'ʌ' (unlike Hindi schwa deletion):
# - finite verbs ending in -छ (रहेछ, गर्दछ, आउँछ, हुनेछ, जानेछ...)
# - postpositions/adverbs (आज, बाट, सँग, तल, तर, तिर, जब, तब, अब, अझ, वर, पर, बाहिर, भित्र, नजिक)
# - infinitives in -उन / core vowel-stem infinitives (आउन, पाउन, गाउन, बनाउन, देखाउन, खान, जान, लिन, दिन, हुन, रहन)
KEEP_FINAL_SCHWA = {
    "आज", "बाट", "सँग", "संग", "तल", "तर", "तिर", "जब", "तब", "अब", "अझ", "वर", "पर", "सुख", "दुख", "दुःख",
    "बाहिर", "भित्र", "हेर", "गर", "भन", "सुन", "पढ", "लेख", "राख", "छैन", "होइन", "सय", "समय", "निर्णय",
    "प्राय", "समुदाय", "उपाय", "बिहान", "सकल", "रुन", "पख", "साँझपख", "विषय", "परिचय", "आशय", "हृदय", "महोदय", "न्याय", "अध्याय",
    "खान", "जान", "लिन", "हुन", "रहन", "आउन", "पाउन", "गाउन", "लाउन", "धाउन", "छाउन",
    "गरेर", "भनेर", "हेरेर", "आएर", "गएर", "खाएर", "लिएर", "दिएर", "बसेर", "सुनेर", "पढेर",
}

G2P_LEX = {
    "काठमाडौँ": ["k", "a", "ʈʰ", "m", "a", "ɖ", "ʌ̃", "ũ"],
    "काठमाडौं": ["k", "a", "ʈʰ", "m", "a", "ɖ", "ʌ̃", "ũ"],
    "माछापुच्छ्रे": ["m", "a", "tsʰ", "a", "p", "u", "ts", "tsʰ", "r", "e"],
    "साँझपख": ["s", "ã", "dzʱ", "ʌ", "p", "ʌ", "kʰ", "ʌ"],
    "सुनजस्तै": ["s", "u", "n", "dz", "ʌ", "s", "t̪", "ʌ", "i"],
    "तहले": ["t̪", "ʌ", "h", "ʌ", "l", "e"],
    "चम्किरहेको": ["ts", "ʌ", "m", "k", "i", "r", "ʌ", "h", "e", "k", "o"],
    "खेतबारीमा": ["kʰ", "e", "t̪", "b", "a", "r", "i", "m", "a"],
    "थामिएपछि": ["t̪ʰ", "a", "m", "i", "e", "p", "ʌ", "tsʰ", "i"],
    "पहाडको": ["p", "ʌ", "h", "a", "ɖ", "k", "o"],
    "टुप्पोबाट": ["ʈ", "u", "p", "o", "b", "a", "ʈ", "ʌ"],
    "प्रयोगकर्ता": ["p", "r", "ʌ", "j", "o", "g", "k", "ʌ", "r", "t̪", "a"],
    "एप्लिकेसन": ["e", "p", "l", "i", "k", "e", "s", "ʌ", "n"],
    "मोबाइल": ["m", "o", "b", "a", "i", "l"],
    "वृद्धि": ["b", "r", "i", "d̪", "d̪ʱ", "i"],
    "हजार": ["h", "ʌ", "dz", "a", "r"],
    "छब्बीस": ["tsʰ", "ʌ", "b", "i", "s"],
    "छब्बिस": ["tsʰ", "ʌ", "b", "i", "s"],
    "पन्ध्र": ["p", "ʌ", "n", "d̪", "r", "ʌ"],
    "वाह": ["w", "a", "h", "ʌ"],
    "आआफ्नो": ["a", "|", "a", "pʰ", "n", "o"],
    "आआफ्ना": ["a", "|", "a", "pʰ", "n", "a"],
    "आआफ्नै": ["a", "|", "a", "pʰ", "n", "ʌ", "i"],
    "राजधानी": ["r", "a", "dz", "d̪ʱ", "a", "n", "i"],
    "प्रधानमन्त्री": ["p", "r", "ʌ", "d̪ʱ", "a", "n", "m", "ʌ", "n", "t̪", "r", "i"],
    "सगरमाथा": ["s", "ʌ", "g", "ʌ", "r", "m", "a", "t̪ʰ", "a"],
    "सुदूरपश्चिम": ["s", "u", "d̪", "u", "r", "p", "ʌ", "ʃ", "ts", "i", "m"],
    "लगभग": ["l", "ʌ", "g", "bʱ", "ʌ", "g"],
    "फुटबल": ["pʰ", "u", "ʈ", "b", "ʌ", "l"],
    "इन्टरनेट": ["i", "n", "ʈ", "ʌ", "r", "n", "e", "ʈ"],
    "बसपार्क": ["b", "ʌ", "s", "p", "a", "r", "k"],
    "वेबसाइट": ["w", "e", "b", "s", "a", "i", "ʈ"],
    "रेलवे": ["r", "e", "l", "w", "e"],
    "देवराज": ["d̪", "e", "w", "r", "a", "dz"],
    "डडेलधुरा": ["ɖ", "ʌ", "ɖ", "e", "l", "d̪ʱ", "u", "r", "a"],
    "सिलसिला": ["s", "i", "l", "s", "i", "l", "a"],
    "कोबोर्ड": ["k", "o", "b", "o", "r", "ɖ"],
    "किचकवध": ["k", "i", "ts", "ʌ", "k", "w", "ʌ", "d̪ʱ"],
    "सुरतगढ": ["s", "u", "r", "ʌ", "t̪", "g", "ʌ", "ɖʱ"],
    "किमडाँडा": ["k", "i", "m", "ɖ", "ã", "ɖ", "a"],
    "अर्घाखाँची": ["ʌ", "r", "gʱ", "a", "kʰ", "ã", "ts", "i"],
    "फिल्मसम्बन्धि": ["pʰ", "i", "l", "m", "s", "ʌ", "m", "b", "ʌ", "n", "d̪ʱ", "i"],
    "फ्रान्स": ["pʰ", "r", "a", "n", "s"],
    "एण्ड": ["e", "n", "ɖ"],
    "एकदम": ["e", "k", "d̪", "ʌ", "m"],
    "कलकल": ["k", "ʌ", "l", "k", "ʌ", "l"],
    "चिरबिर": ["ts", "i", "r", "b", "i", "r"],
    "किनमेल": ["k", "i", "n", "m", "e", "l"],
    "फलफूल": ["pʰ", "ʌ", "l", "pʰ", "u", "l"],
    "कागजात": ["k", "a", "g", "dz", "a", "t̪"],
    "जनसंख्या": ["dz", "ʌ", "n", "s", "ʌ", "ŋ", "kʰ", "j", "a"],
    "तापक्रम": ["t̪", "a", "p", "k", "r", "ʌ", "m"],
    "अधिकतम": ["ʌ", "d̪ʱ", "i", "k", "t̪", "ʌ", "m"],
    "अठहत्तर": ["ʌ", "ʈʰ", "h", "ʌ", "t̪", "t̪", "ʌ", "r"],
    "नेक्स्टजेन": ["n", "e", "k", "s", "ʈ", "dz", "e", "n"],
    "कति": ["k", "ʌ", "t̪", "i"],
    "कत्ति": ["k", "ʌ", "t̪", "t̪", "i"],
    "पकै": ["p", "ʌ", "k", "ʌ", "i"],
    "पक्कै": ["p", "ʌ", "k", "k", "ʌ", "i"],
    "चिटिक्क": ["ts", "i", "ʈ", "i", "k", "k", "ʌ"],
    "ठ्ठयाक्कै": ["ʈʰ", "ʈʰ", "j", "a", "k", "k", "ʌ", "i"],
    "ठ्याक्कै": ["ʈʰ", "j", "a", "k", "k", "ʌ", "i"],
    "सक्कली": ["s", "ʌ", "k", "k", "ʌ", "l", "i"],
    "अड्डा": ["ʌ", "ɖ", "ɖ", "a"],
    "अड्डाका": ["ʌ", "ɖ", "ɖ", "a", "k", "a"],
    "छक्क": ["tsʰ", "ʌ", "k", "k", "ʌ"],
    "सफा": ["s", "ʌ", "pʰ", "a"],
    "चिसो": ["ts", "i", "s", "o"],
    "सास": ["s", "a", "s"],
    "शान्ति": ["ʃ", "a", "n", "t̪", "i"],
    "छाता": ["tsʰ", "a", "t̪", "a"],
    "प्रविधि": ["p", "r", "ʌ", "b", "i", "d̪ʱ", "i"],
    "अर्थतन्त्र": ["ʌ", "r", "t̪ʰ", "ʌ", "t̪", "ʌ", "n", "t̪", "r", "ʌ"],
    "अन्तर्राष्ट्रिय": ["ʌ", "n", "t̪", "ʌ", "r", "r", "a", "ʃ", "ʈ", "r", "i", "j", "ʌ"],
    "प्रतिशत": ["p", "r", "ʌ", "t̪", "i", "ʃ", "ʌ", "t̪"],
    "डिजिटल": ["ɖ", "i", "dz", "i", "ʈ", "ʌ", "l"],
    "सफ्टवेयर": ["s", "ʌ", "pʰ", "ʈ", "w", "e", "j", "ʌ", "r"],
    "कम्प्युटर": ["k", "ʌ", "m", "p", "j", "u", "ʈ", "ʌ", "r"],
}

# English loanwords written in Devanagari: NO word-final inherent schwa after the final cluster (टेक्स्ट -> ʈ e k s ʈ, never 'text-a').
LOAN_NO_SCHWA = {"टेक्स्ट", "टेस्ट", "पोस्ट", "स्पीच", "फास्ट", "लिस्ट", "कन्टेन्ट", "प्रिन्ट", "ब्रान्ड", "वर्ल्ड",
                 "फिल्म", "इन्टेलिजेन्स", "एम्बुलेन्स", "जेम्स", "फ्रान्स", "पार्क", "बसपार्क", "वार्ड", "कार्ड", "बोर्ड", "एण्ड", "सरपंच"}
LOAN_FINAL_CL = ("क्स्ट", "स्ट", "स्क", "स्प", "क्स", "ल्ड", "न्ड", "न्ट", "क्ट", "न्स", "ल्म", "म्स", "र्क", "र्ड")


def _keep_final_schwa(w, units, idx, n_syl):
    if w and (w in LOAN_NO_SCHWA or w.endswith(LOAN_FINAL_CL)):
        return False
    if n_syl == 1:
        return True
    conj = idx >= 2 and units[idx - 1][1] == "C" and units[idx - 2][1] == "C"
    if conj:
        return True
    if not w:
        return False
    # All Nepali finite verbs ending in bare 'छ' (रहेछ, गर्दछ, आउँछ, हुनेछ, जानेछ, पाउनेछ...) keep final 'ʌ'
    if w.endswith("छ"):
        return True
    # Native Nepali adverbs, postpositions, converbs, imperatives, -आलय/-दय/-शय/-पख stems & core infinitives where Hindi drops schwa but Nepali pronounces 'ʌ'
    if w in KEEP_FINAL_SCHWA or w.endswith(("बाट", "सँग", "संग", "तिर", "तल", "बाहिर", "भित्र", "ीय", "प्रिय", "ष्ट्रिय", "लय", "दय", "शय", "पख")):
        return True
    # Causative / vowel-stem Nepali infinitives (-उन), converbs (-एर/-ेर: पुगेर, बनाएर), and negative verbs (-दैन: हुँदैन; -एन: भएन; छैन/होइन)
    if len(w) >= 3 and w.endswith(("उन", "एर", "ेर", "दैन", "एन", "छैन", "होइन")):
        return True
    return False


def _resolve(units, w=""):
    n_syl = sum(1 for _, k in units if k in "VS")
    res = []
    for idx, (ph, k) in enumerate(units):
        if k != "S":
            res.append(ph); continue
        last = all(u[1] == "C" for u in units[idx + 1:]) and idx == len(units) - 1
        if not last:
            res.append("ʌ")                    # medial: always retained
        elif _keep_final_schwa(w, units, idx, n_syl):
            res.append("ʌ")                    # conjunct-final / monosyllable / native Nepali schwa-retaining endings
    return res


def _honorific_glide(ph):
    """-नुभए- (बनाउनुभएको, गर्नुभएको, भन्नुभएको): palatal glide between ʌ and e keeps the compound smooth (no hiatus pause, no 'भेको' collapse)."""
    out = []
    for i, p in enumerate(ph):
        if p == "e" and i >= 3 and ph[i - 3:i] == ["u", "bʱ", "ʌ"]:
            out.append("j")
        out.append(p)
    return out


def _word_core(w, lang):
    ph = _resolve(_parse(w, lang), w)
    return _honorific_glide(ph) if "नुभ" in w else ph


def word_to_phones(w, lang="ne"):
    w = unicodedata.normalize("NFC", w)
    if w in G2P_LEX:
        return list(G2P_LEX[w])
    for sfx in SUFFIXES:
        if len(w) > len(sfx) and w.endswith(sfx) and sfx[0] in C:
            stem = w[: -len(sfx)]
            if stem in G2P_LEX or stem[-1] in C or stem.endswith(("हरू", "हरु", "ज्यू", "ज्यु")):
                return word_to_phones(stem, lang) + word_to_phones(sfx, lang) if sfx != w else _word_core(w, lang)
            break
    return _word_core(w, lang)


_TOK = re.compile(r"[,.?!;।॥]|[\u0900-\u0963\u0966-\u097F]+|[A-Za-z][A-Za-z'~]*|\S")
_PUNC = {",": ",", ".": ".", "।": ".", "॥": ".", "?": "?", "!": "!", ";": ";", ":": ",", "—": ",", "-": ","}


def normalize(text, lang="ne", fast=False):
    t = unicodedata.normalize("NFC", text)
    t = t.replace("\u200d", "").replace("\u200c", "")
    t = numbers.expand(t)
    if not fast and re.search(r"[A-Za-z]{2,}", t):
        import os
        if os.environ.get("ROMAN_ORT"):
            try:
                from .roman_neural import convert_many
                return convert_many([(t, lang)], node_path=os.environ.get("ROMAN_ORT"))[0]
            except Exception:
                pass
    from .translit import roman_to_deva
    t = re.sub(r"[A-Za-z][A-Za-z'~]*", lambda m: roman_to_deva(m[0], lang), t)
    return t


def text_to_phones(text, lang="ne", add_end=True):
    if lang == "sa":
        from . import sanskrit
        return sanskrit.text_to_phones(text, add_end)
    t = normalize(text, lang)
    phones, first = [], True
    toks = _TOK.findall(t)
    for idx_t, m in enumerate(toks):
        if m[0] in _PUNC and len(m) == 1:
            if phones and phones[-1] == S.WORD:
                phones.pop()
            phones.append(_PUNC[m]); phones.append(S.WORD); continue
        if not ("\u0900" <= m[0] <= "\u097F"):
            continue
        ph = word_to_phones(m, lang)
        if ph:
            # Collapse any accidental duplicate consecutive 'h' inside a word
            ph_clean = [p for idx_p, p in enumerate(ph) if not (p == "h" and idx_p > 0 and ph[idx_p - 1] == "h")]
            phones.extend(ph_clean); phones.append(S.WORD)
            nxt_w = toks[idx_t + 1] if idx_t + 1 < len(toks) else ""
            if m.endswith(("नुहोस्", "नोस्")) and nxt_w in ("र", "अनि", "तर", "वा"):
                phones.pop(); phones.extend([",", S.WORD])
    if phones and phones[-1] == S.WORD:
        phones.pop()
    if add_end and (not phones or phones[-1] not in (".", "?", "!")):
        phones.append(".")
    return phones


PLACE = {}
for _pl, _grp in {"ret": "ʈ ʈʰ ɖ ɖʱ", "den": "t̪ t̪ʰ d̪ d̪ʱ", "vel": "k kʰ g gʱ", "lab": "p pʰ b bʱ", "aff": "ts tsʰ dz dzʱ"}.items():
    for _x in _grp.split(): PLACE[_x] = _pl


def phone_dm(ph):
    """v6: Prosody-aware duration multipliers for punctuation breaths (',', '!', '?') and interjection nuclei ('वाह!'),
    while keeping all regular lexical phonemes at 1.0 (zero consonant/flap distortion)."""
    n = len(ph)
    dm = [1.0] * n
    for i, p in enumerate(ph):
        if p == ",":
            dm[i] = 1.25
        elif p in ("!", "?"):
            dm[i] = 1.32
        elif p == "a" and i >= 1 and ph[i - 1] == "w" and i + 2 < n and ph[i + 1] == "h" and ph[i + 2] in ("!", "|"):
            dm[i] = 1.38
            dm[i + 1] = 0.72
    return dm


def text_to_phones_acc(text, lang="ne", add_end=True):
    """(phones, per-phone pitch offsets). v6: Emits expressive pitch contours for Nepali exclamations ('!'),
    interjections ('वाह!'), rising questions ('?'), and storytelling comma continuations (','), plus Sanskrit svara."""
    if lang == "sa":
        from . import sanskrit
        return sanskrit.text_to_phones_acc(text, add_end)
    ph = text_to_phones(text, lang, add_end)
    n = len(ph)
    off = [0.0] * n
    import math
    c_start = 0
    for i, p in enumerate(ph):
        if p in (",", ".", "?", "!", ";"):
            vs = [j for j in range(c_start, i) if ph[j][:1] in ("ʌ", "a", "i", "u", "e", "o")]
            nv = len(vs)
            if p == "!":
                for k, j in enumerate(vs):
                    off[j] = round(0.95 * math.sin(math.pi * (k + 0.5) / max(nv, 1)) + 0.30, 4)
                if nv == 1:
                    off[vs[0]] = 1.35
                elif nv >= 2:
                    off[vs[0]] = max(off[vs[0]], 1.15)
                    off[vs[-1]] = 0.55
            elif p == "?":
                for k, j in enumerate(vs):
                    off[j] = round(-0.20 + 0.35 * (k / max(nv - 1, 1)), 4)
                if nv >= 1:
                    off[vs[-1]] = 1.25
                if nv >= 2:
                    off[vs[-2]] = 0.68
            elif p == ",":
                if nv >= 1:
                    off[vs[-1]] = 0.24
                if nv >= 2:
                    off[vs[-2]] = 0.12
            c_start = i + 1
    sm = list(off)
    for i, p in enumerate(ph):
        if p not in (",", ".", "?", "!", ";", "|") and p[:1] not in ("ʌ", "a", "i", "u", "e", "o"):
            lv = off[i - 1] if i > 0 else 0.0
            rv = off[i + 1] if i + 1 < n else 0.0
            sm[i] = round(0.5 * (lv + rv), 4)
    return ph, sm
