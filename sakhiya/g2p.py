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

SUFFIXES = sorted("""हरूबारे हरुबारे हरूद्वारा हरुद्वारा हरूसहित हरुसहित हरूलाई हरूले हरूको हरूका हरूमा हरूबाट हरूसँग हरूतिर हरूभित्र हरूबाहिर हरूमाथि हरू हरुलाई हरुले हरुको हरुका हरुमा हरुबाट हरुसँग हरुतिर हरु
ज्यूहरूलाई ज्यूहरूले ज्यूहरूको ज्यूलाई ज्यूले ज्यूको ज्यूका ज्यूमा ज्यूबाट ज्यूसँग ज्यू ज्युलाई ज्युले ज्युको ज्यु हुक्रे हुक्रेन
कर्ताहरूको कर्ताहरूका कर्ताहरूले कर्ताहरूलाई कर्ताको कर्ताका कर्ताले कर्तालाई कर्ता बाजीका बाजीको बाजीमा बाजी
लाई बाटै बाट देखि सम्म सँगै सँग संग सँगे संगे भन्दा ले को का की मा मै मे से के हेन सन लग केर कर पछि अघि पहिले
भित्र बाहिर माथि तल नजिक भरि भर पट्टि पटि जस्तो जस्तै जति मात्र पनि नै चाहिँ चाहिं पिच्छे वरिपरि बारेमा बारे विरुद्ध अनुसार समेत बित्तिकै अगाडि पछाडि प्रति तर्फ पूर्वक घर स्थान स्थल पत्र
चढाव लाइन शील शिल द्वारा सहित रहित युक्त मूलक जन्य पूर्ण सम्बन्धी सम्बन्धित मार्फत अन्तर्गत लगायत
पालिका शाला वाला करी नगर पुर गढ कोट चोक टोल बजार ग्राम बार""".split(),
                  key=len, reverse=True)
HEAVY_SFX = ("हरूसहित", "हरुसहित", "सहित", "मार्फत", "अनुसार", "अन्तर्गत", "विरुद्ध", "सम्बन्धी", "लगायत")
NUM_FOLLOWERS = ("प्रतिशत", "जना", "बजे", "लाख", "हजार", "करोड", "अर्ब", "खर्ब", "सय", "महिना", "वर्ष", "दिन", "हप्ता", "साता",
                 "घण्टा", "मिनेट", "सेकेन्ड", "वटा", "पटक", "चोटि", "तारिख", "तारीख", "गते", "दशमलव", "पोइन्ट", "डिग्री",
                 "जिबी", "मेगाबाइट", "किलोमिटर", "मिटर", "रुपैयाँ", "डलर", "देश", "राज्य")


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
                out.append((S.WORD if v_ph == "a" else "j", "C"))  # आआफ्नो -> a | a pʰ n o; लिइरहेको -> l i j i r ʌ h e k o
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
            if out and out[-1][1] == "S":
                out[-1] = ("ʌ", "V")
            out.append(("h", "C")); i += 1
        else:
            i += 1  # ZWJ etc.
    return out


# Native Nepali words & suffixes that ALWAYS retain word-final inherent schwa 'ʌ' (unlike Hindi schwa deletion):
# - finite verbs ending in -छ (रहेछ, गर्दछ, आउँछ, हुनेछ, जानेछ...)
# - postpositions/adverbs (आज, बाट, सँग, तल, तर, तिर, जब, तब, अब, अझ, वर, पर, बाहिर, भित्र, नजिक)
# - infinitives in -उन / core vowel-stem infinitives (आउन, पाउन, गाउन, बनाउन, देखाउन, खान, जान, लिन, दिन, हुन, रहन)
KEEP_FINAL_SCHWA = {
    "आज", "गत", "विगत", "अर्क", "सतर्क", "बाट", "सँग", "संग", "तल", "तर", "तिर", "जब", "तब", "अब", "अझ", "वर", "पर", "सुख", "दुख", "दुःख",
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
    "वृद्ध": ["b", "r", "i", "d̪", "d̪ʱ", "ʌ"],
    "बृहत": ["b", "r", "i", "h", "ʌ", "t̪"],
    "बृहत्": ["b", "r", "i", "h", "ʌ", "t̪"],
    "कृत्रिम": ["k", "r", "i", "t̪", "r", "i", "m"],
    "हजार": ["h", "ʌ", "dz", "a", "r"],
    "मोडेल": ["m", "o", "ɖ", "e", "l"],
    "कोड": ["k", "o", "ɖ"],
    "ओपन": ["o", "p", "ʌ", "n"],
    "पोइन्ट": ["p", "o", "i", "n", "ʈ"],
    "अनलाइन": ["ʌ", "n", "l", "a", "i", "n"],
    "वालेट": ["w", "a", "l", "e", "ʈ"],
    "सेयर": ["s", "e", "j", "ʌ", "r"],
    "शेयर": ["ʃ", "e", "j", "ʌ", "r"],
    "बजार": ["b", "ʌ", "dz", "a", "r"],
    "कारोबार": ["k", "a", "r", "o", "b", "a", "r"],
    "करिब": ["k", "ʌ", "r", "i", "b"],
    "गिरावट": ["g", "i", "r", "a", "w", "ʌ", "ʈ"],
    "समेत": ["s", "ʌ", "m", "e", "t̪"],
    "उतारचढाव": ["u", "t̪", "a", "r", "ts", "ʌ", "ɖʱ", "a", "w"],
    "अनुसार": ["ʌ", "n", "u", "s", "a", "r"],
    "डलर": ["ɖ", "ʌ", "l", "ʌ", "r"],
    "अर्थात": ["ʌ", "r", "t̪ʰ", "a", "t̪"],
    "अर्थात्": ["ʌ", "r", "t̪ʰ", "a", "t̪"],
    "हरेक": ["h", "ʌ", "r", "e", "k"],
    "दिन": ["d̪", "i", "n"],
    "दैनिक": ["d̪", "ʌ", "i", "n", "i", "k"],
    "जीवन": ["dz", "i", "w", "ʌ", "n"],
    "प्रभाव": ["p", "r", "ʌ", "bʱ", "a", "w"],
    "भएका": ["bʱ", "ʌ", "e", "k", "a"],
    "भएको": ["bʱ", "ʌ", "e", "k", "o"],
    "भएकी": ["bʱ", "ʌ", "e", "k", "i"],
    "भइरहेको": ["bʱ", "ʌ", "i", "r", "ʌ", "h", "e", "k", "o"],
    "भइरहेका": ["bʱ", "ʌ", "i", "r", "ʌ", "h", "e", "k", "a"],
    "भइरहेकी": ["bʱ", "ʌ", "i", "r", "ʌ", "h", "e", "k", "i"],
    "लिइरहेको": ["l", "i", "j", "i", "r", "ʌ", "h", "e", "k", "o"],
    "दिइरहेको": ["d̪", "i", "j", "i", "r", "ʌ", "h", "e", "k", "o"],
    "मामकाः": ["m", "a", "m", "ʌ", "k", "a", "h"],
    "युयुत्सवः": ["j", "u", "j", "u", "t̪", "s", "ʌ", "w", "ʌ", "h"],
    "प्रायः": ["p", "r", "a", "j", "ʌ", "h"],
    "क्रमशः": ["k", "r", "ʌ", "m", "ʌ", "ʃ", "ʌ", "h"],
    "अतः": ["ʌ", "t̪", "ʌ", "h"],
    "दुःख": ["d̪", "u", "h", "kʰ", "ʌ"],
    "च्याटजिपिटी": ["ts", "j", "a", "ʈ", "dz", "i", "p", "i", "ʈ", "i"],
    "जीपीटी": ["dz", "i", "p", "i", "ʈ", "i"],
    "एन्थ्रोपिक": ["e", "n", "t̪ʰ", "r", "o", "p", "i", "k"],
    "एन्थ्रोपिकको": ["e", "n", "t̪ʰ", "r", "o", "p", "i", "k", "k", "o"],
    "तपाईं": ["t̪", "ʌ", "p", "a", "ĩ"],
    "तपाईंको": ["t̪", "ʌ", "p", "a", "ĩ", "k", "o"],
    "तपाईंलाई": ["t̪", "ʌ", "p", "a", "ĩ", "l", "a", "i"],
    "तपाईंले": ["t̪", "ʌ", "p", "a", "ĩ", "l", "e"],
    "चौध": ["ts", "ʌ", "u", "d̪ʱ", "ʌ"],
    "सत्र": ["s", "ʌ", "t̪", "r", "ʌ"],
    "तर्क": ["t̪", "ʌ", "r", "k", "ʌ"],
    "सम्पर्क": ["s", "ʌ", "m", "p", "ʌ", "r", "k", "ʌ"],
    "छब्बीस": ["tsʰ", "ʌ", "b", "b", "i", "s"],
    "छब्बिस": ["tsʰ", "ʌ", "b", "b", "i", "s"],
    "पन्ध्र": ["p", "ʌ", "n", "d̪ʱ", "r", "ʌ"],
    "आएका": ["a", "j", "e", "k", "a"],
    "झरना": ["dzʱ", "ʌ", "r", "n", "a"],
    "झरनाको": ["dzʱ", "ʌ", "r", "n", "a", "k", "o"],
    "शिविर": ["ʃ", "i", "b", "i", "r"],
    "बचा": ["b", "ʌ", "ts", "a"],
    "बच्चा": ["b", "ʌ", "ts", "ts", "a"],
    "वाह": ["w", "a", "h"],
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
    "प्रतिशतले": ["p", "r", "ʌ", "t̪", "i", "ʃ", "ʌ", "t̪", "l", "e"],
    "डिजिटल": ["ɖ", "i", "dz", "i", "ʈ", "ʌ", "l"],
    "सफ्टवेयर": ["s", "ʌ", "pʰ", "ʈ", "w", "e", "j", "ʌ", "r"],
    "कम्प्युटर": ["k", "ʌ", "m", "p", "j", "u", "ʈ", "ʌ", "r"],
    # Surgical Sakhiya-One Polish lexicon entries (100% Py <-> JS parity):
    "वेब": ["w", "e", "b"],
    "वर्ल्ड": ["w", "ʌ", "r", "l", "ɖ"],
    "वाचक": ["w", "a", "ts", "ʌ", "k"],
    "विश्व": ["w", "i", "ʃ", "w", "ʌ"],
    "विश्वभर": ["w", "i", "ʃ", "w", "ʌ", "bʱ", "ʌ", "r"],
    "विशेष": ["w", "i", "ʃ", "e", "ʃ"],
    "विकास": ["w", "i", "k", "a", "s"],
    "व्यवसाय": ["w", "j", "ʌ", "w", "ʌ", "s", "a", "j"],
    "व्यवसायमा": ["w", "j", "ʌ", "w", "ʌ", "s", "a", "j", "m", "a"],
    "संवेदनशील": ["s", "ʌ̃", "w", "e", "d̪", "ʌ", "n", "ʃ", "i", "l"],
    "बुद्धिमत्ता": ["b", "u", "d̪", "d̪ʱ", "i", "m", "ʌ", "t̪", "t̪", "a"],
    "उल्लेखनीय": ["u", "l", "l", "e", "kʰ", "ʌ", "n", "i", "j", "ʌ"],
    "गर्नेछौँ": ["g", "ʌ", "r", "n", "e", "tsʰ", "ʌ̃", "ũ"],
    "गर्नेछौं": ["g", "ʌ", "r", "n", "e", "tsʰ", "ʌ̃", "ũ"],
    "लागौँ": ["l", "a", "g", "ʌ̃", "ũ"],
    "लागौं": ["l", "a", "g", "ʌ̃", "ũ"],
    "चलौँ": ["ts", "ʌ", "l", "ʌ̃", "ũ"],
    "चलौं": ["ts", "ʌ", "l", "ʌ̃", "ũ"],
    "पाँचौँ": ["p", "ã", "ts", "ʌ̃", "ũ"],
    "अर्बौँ": ["ʌ", "r", "b", "ʌ̃", "ũ"],
    "ल्याउँदै": ["l", "j", "ã", "ũ", "d̪", "ʌ", "i"],
    "अनुसन्धानात्मक": ["ʌ", "n", "u", "s", "ʌ", "n", "d̪ʱ", "a", "n", "a", "t̪", "m", "ʌ", "k"],
    "प्रयोगकर्ताको": ["p", "r", "ʌ", "j", "o", "g", "k", "ʌ", "r", "t̪", "a", "k", "o"],
    "लगानीकर्ताहरूको": ["l", "ʌ", "g", "a", "n", "i", "k", "ʌ", "r", "t̪", "a", "h", "ʌ", "r", "u", "k", "o"],
    "त्रैमासिकमा": ["t̪", "r", "ʌ", "i", "m", "a", "s", "i", "k", "m", "a"],
    "उपलब्धिहरू": ["u", "p", "ʌ", "l", "ʌ", "b", "d̪ʱ", "i", "h", "ʌ", "r", "u"],
    "विश्लेषकहरूका": ["w", "i", "ʃ", "l", "e", "ʃ", "ʌ", "k", "h", "ʌ", "r", "u", "k", "a"],
    "पूर्वाधारमा": ["p", "u", "r", "w", "a", "d̪ʱ", "a", "r", "m", "a"],
    "नवीनतम": ["n", "ʌ", "w", "i", "n", "t̪", "ʌ", "m"],
    "प्राविधिक": ["p", "r", "a", "w", "i", "d̪ʱ", "i", "k"],
    "विस्तृत": ["w", "i", "s", "t̪", "r", "i", "t̪"],
    "सुरक्षित": ["s", "u", "r", "ʌ", "k", "ʃ", "i", "t̪"],
    "संस्करणहरू": ["s", "ʌ̃", "s", "k", "ʌ", "r", "ʌ", "ɳ", "h", "ʌ", "r", "u"],
    "जेमिनाई": ["dz", "e", "m", "i", "n", "a", "i"],
    "फोर": ["pʰ", "o", "r"],
    "नियोन": ["n", "i", "j", "o", "n"],
    "जिपिटी": ["dz", "i", "p", "i", "ʈ", "i"],
    "सिक्स": ["s", "i", "k", "s"],
    "टु": ["ʈ", "u"],
    "एआई": ["e", "j", "a", "a", "i"],
    "एआईका": ["e", "j", "a", "a", "i", "k", "a"],
    "एआईको": ["e", "j", "a", "a", "i", "k", "o"],
    "एआईले": ["e", "j", "a", "a", "i", "l", "e"],
    "एआईमा": ["e", "j", "a", "a", "i", "m", "a"],
    "एआईलाई": ["e", "j", "a", "a", "i", "l", "a", "i"],
    "एआईबाट": ["e", "j", "a", "a", "i", "b", "a", "ʈ", "ʌ"],
    "एआईसँग": ["e", "j", "a", "a", "i", "s", "ʌ̃", "g", "ʌ"],
    "एआईहरू": ["e", "j", "a", "a", "i", "h", "ʌ", "r", "u"],
    "ओपनएआई": ["o", "p", "ʌ", "n", "|", "e", "j", "a", "a", "i"],
    "ओपनएआईले": ["o", "p", "ʌ", "n", "|", "e", "j", "a", "a", "i", "l", "e"],
    "ओपनएआईको": ["o", "p", "ʌ", "n", "|", "e", "j", "a", "a", "i", "k", "o"],
    "क्यारेक्टरएआई": ["k", "j", "a", "r", "e", "k", "ʈ", "ʌ", "r", "|", "e", "j", "a", "a", "i"],
    "गत": ["g", "ʌ", "t̪", "ʌ"],
    "विगत": ["b", "i", "g", "ʌ", "t̪", "ʌ"],
    "गजेट": ["g", "j", "a", "dz", "e", "ʈ"],
    "ग्याजेट": ["g", "j", "a", "dz", "e", "ʈ"],
    "ग्याजेटको": ["g", "j", "a", "dz", "e", "ʈ", "k", "o"],
    "सीईओ": ["s", "i", "|", "i", "|", "o"],
    "आईओएस": ["a", "i", "|", "o", "|", "e", "s"],
    "भिपिएन": ["bʱ", "i", "p", "i", "e", "n"],
    "एलएलएम": ["e", "l", "|", "e", "l", "|", "e", "m"],
    "एलएलएमको": ["e", "l", "|", "e", "l", "|", "e", "m", "k", "o"],
    "एमएल": ["e", "m", "|", "e", "l"],
    "एमएलले": ["e", "m", "|", "e", "l", "l", "e"],
    "एसएसडी": ["e", "s", "|", "e", "s", "|", "ɖ", "i"],
    "एसएसडीबाट": ["e", "s", "|", "e", "s", "|", "ɖ", "i", "b", "a", "ʈ", "ʌ"],
    "एफसीसी": ["e", "pʰ", "|", "s", "i", "|", "s", "i"],
    "एफसीसीले": ["e", "pʰ", "|", "s", "i", "|", "s", "i", "l", "e"],
    "एपिआई": ["e", "p", "i", "a", "i"],
    "सिपियु": ["s", "i", "p", "i", "j", "u"],
    "स्टारमाइन्ड": ["s", "ʈ", "a", "r", "m", "a", "i", "n", "ɖ"],
    "स्टारलिङ्क": ["s", "ʈ", "a", "r", "l", "i", "ŋ", "k"],
    "स्टारलिङ्कसँग": ["s", "ʈ", "a", "r", "l", "i", "ŋ", "k", "s", "ʌ̃", "g", "ʌ"],
    "स्पेसएक्स": ["s", "p", "e", "s", "e", "k", "s"],
    "स्पेसएक्सलाई": ["s", "p", "e", "s", "e", "k", "s", "l", "a", "i"],
    "एनभिडिया": ["e", "n", "bʱ", "i", "ɖ", "i", "j", "a"],
    "एनभिडियाको": ["e", "n", "bʱ", "i", "ɖ", "i", "j", "a", "k", "o"],
    "एनभिडियाले": ["e", "n", "bʱ", "i", "ɖ", "i", "j", "a", "l", "e"],
    "कमब्याक": ["k", "ʌ", "m", "b", "j", "a", "k"],
    "कमब्याकसँगै": ["k", "ʌ", "m", "b", "j", "a", "k", "s", "ʌ̃", "g", "ʌ", "i"],
    "क्लड": ["k", "l", "ʌ", "ɖ"],
    "सिप": ["ʃ", "i", "p"],
    "लन्च": ["l", "ʌ", "n", "ts"],
    "हेल्थ": ["h", "e", "l", "t̪ʰ"],
    "प्रिज्म": ["p", "r", "i", "dz", "m"],
    "न्युयोर्क": ["n", "j", "u", "j", "o", "r", "k"],
    "न्युयोर्कले": ["n", "j", "u", "j", "o", "r", "k", "l", "e"],
    "वालमार्ट": ["w", "a", "l", "m", "a", "r", "ʈ"],
    "वालमार्टले": ["w", "a", "l", "m", "a", "r", "ʈ", "l", "e"],
    "अनामनगर": ["ʌ", "n", "a", "m", "n", "ʌ", "g", "ʌ", "r"],
    "अनामनगरमा": ["ʌ", "n", "a", "m", "n", "ʌ", "g", "ʌ", "r", "m", "a"],
    "रोलआउट": ["r", "o", "l", "a", "u", "ʈ"],
    "इयरबड्स": ["i", "j", "ʌ", "r", "b", "ʌ", "ɖ", "s"],
    "फायरफक्स": ["p", "ʰ" if False else "pʰ", "a", "j", "ʌ", "r", "pʰ", "ʌ", "k", "s"],
    "च्याटबट": ["ts", "j", "a", "ʈ", "b", "ʌ", "ʈ"],
    "फिनसेन": ["pʰ", "i", "n", "s", "e", "n"],
    "फिनसेनका": ["pʰ", "i", "n", "s", "e", "n", "k", "a"],
    "सरकार": ["s", "ʌ", "r", "k", "a", "r"],
    "सरकारले": ["s", "ʌ", "r", "k", "a", "r", "l", "e"],
    "सरोकारवाला": ["s", "ʌ", "r", "o", "k", "a", "r", "w", "a", "l", "a"],
    "सरोकारवालासँग": ["s", "ʌ", "r", "o", "k", "a", "r", "w", "a", "l", "a", "s", "ʌ̃", "g", "ʌ"],
    "छलफल": ["tsʰ", "ʌ", "l", "pʰ", "ʌ", "l"],
    "छोटकरीमा": ["tsʰ", "o", "ʈ", "k", "ʌ", "r", "i", "m", "a"],
    "बालबालिकाको": ["b", "a", "l", "b", "a", "l", "i", "k", "a", "k", "o"],
    "दक्षिणपूर्वी": ["d̪", "ʌ", "k", "ʃ", "i", "ɳ", "p", "u", "r", "w", "i"],
    "शुक्रबार": ["ʃ", "u", "k", "r", "ʌ", "b", "a", "r"],
    "शनिबार": ["ʃ", "ʌ", "n", "i", "b", "a", "r"],
    "बिहीबार": ["b", "i", "h", "i", "b", "a", "r"],
}

# English loanwords written in Devanagari: NO word-final inherent schwa after the final cluster (टेक्स्ट -> ʈ e k s ʈ, never 'text-a').
LOAN_NO_SCHWA = {"टेक्स्ट", "टेस्ट", "पोस्ट", "स्पीच", "फास्ट", "लिस्ट", "कन्टेन्ट", "प्रिन्ट", "ब्रान्ड", "वर्ल्ड",
                 "फिल्म", "इन्टेलिजेन्स", "एम्बुलेन्स", "जेम्स", "फ्रान्स", "पार्क", "बसपार्क", "मार्क", "क्लर्क", "नेटवर्क", "वर्क", "शार्क", "वार्ड", "कार्ड", "बोर्ड", "एण्ड", "सरपंच",
                 "लन्च", "हेल्थ", "प्रिज्म", "भिपिएन", "न्युयोर्क", "वालमार्ट", "सेल्फ", "इयरबड्स", "सोर्स"}
LOAN_FINAL_CL = ("क्स्ट", "स्ट", "स्ड", "स्क", "स्प", "क्स", "ल्ड", "र्ल्ड", "न्ड", "न्ट", "क्ट", "न्स", "ल्म", "म्स", "र्ड",
                 "र्क", "र्ट", "न्च", "ल्थ", "ज्म", "फ्ट", "ल्फ", "ड्स", "र्स", "न्ज", "र्च", "ल्भ", "र्भ", "न्क", "ङ्क")
NEG_VERBS_EN = {"भएन", "गएन", "आएन", "पाएन", "दिएन", "लिएन", "थिएन", "खाएन", "छोडेन", "मानेन", "जानेन", "सकेन", "बुझेन", "देखेन", "भेटेन"}


def _keep_final_schwa(w, units, idx, n_syl):
    if w and w in KEEP_FINAL_SCHWA:
        return True
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
    if w.endswith(("बाट", "सँग", "संग", "तिर", "तल", "बाहिर", "भित्र", "ीय", "प्रिय", "ष्ट्रिय", "लय", "दय", "शय", "पख")):
        return True
    # Causative / vowel-stem Nepali infinitives (-उन), converbs (-एर/-ेर: पुगेर, बनाएर), and negative verbs (-दैन: हुँदैन; भएन/थिएन; छैन/होइन)
    if len(w) >= 3 and (w.endswith(("उन", "एर", "ेर", "दैन", "छैन", "होइन", "िएन", "एनन्")) or w in NEG_VERBS_EN):
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
            if stem in G2P_LEX or stem[-1] in C or stem.endswith(("हरू", "हरु", "ज्यू", "ज्यु", "बाजी", "कर्ता")):
                if sfx == w:
                    return _word_core(w, lang)
                sep = [S.WORD] if sfx in HEAVY_SFX else []
                return word_to_phones(stem, lang) + sep + word_to_phones(sfx, lang)
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
            nxt_w = toks[idx_t + 1] if idx_t + 1 < len(toks) else ""
            # Numeral 'छ' (6) before a counter/unit/noun: cliticize onto next word so model never treats 'छ' as clause-final copula verb
            if m == "छ" and nxt_w and (nxt_w.startswith(NUM_FOLLOWERS) or nxt_w in numbers._U):
                phones.extend(ph_clean)
                continue
            phones.extend(ph_clean); phones.append(S.WORD)
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
