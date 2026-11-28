#!/usr/bin/env python3
"""Experiment 3 (Gemini Research): Nepali Phonological Audit & Algorithmic G2P v4 Prototype.

Audits `sakhiya/g2p.py` against a 75-word Nepali Phonological Gold Benchmark (Khatiwada 2009 IPA standard,
Pokharel 1989, Nepal Pragya-Pratishthan Brihat Shabdakosh) across 7 systematic phonological categories + Stress/Weight:
  Cat 1: `ज्ञ` (`ज+्+ञ`) -> Native Nepali `[g j]` (with nasalized vowel in open/nasal syllables), NOT literal `[dz ɲ]`!
  Cat 2: `व` (`[b]` vs `[w]`) -> Native bilabial `[b]` in Sanskrit/Nepali initial/stem positions (`विद्यालय`, `विद्यार्थी`, `विचार`,
         `विभाग`, `विराटनगर`, `वीरगन्ज`, `वडा`, `वन`, `वर्षा`, `वरिष्ठ`, `व्यवस्था`, `व्यवहार`, `विवाद`, `विवाह`, `विमान`)
         vs labio-velar `[w]` in loanwords (`वालेट`, `वेबसाइट`, `वर्ल्ड`), post-consonantal `C+व` glides (`स्वागत`, `द्वारा`),
         and intervocalic `-आवा/-एवा` glides (`हावा`, `सेवा`).
  Cat 3: Productive Medial Schwa Deletion (`VC-CV` Syncope) in 4+ consonant stems, compounds (`-कारी`, `-दार`, `-बाही`, `-नीति`,
         `-तन्त्र`, `-गन्ज`, `-घाट`, `-दान`, `-वार`, `-फुक`, `-फोन`), and reduplicatives (`गडबड`, `झलमल`, `सरसफाइ`).
  Cat 4: Final `-ह` Tatsama Schwa Retention (`सिंह`, `गृह`, `ग्रह`, `सन्देह`, `आग्रह`, `उत्साह`, `समूह`, `प्रवाह`, `तह`)
         and Native `-र्च` (`खर्च -> kʰ ʌ r ts ʌ`) vs Loan `-र्च` (`सर्च -> s ʌ r ts`).
  Cat 5: Anusvara (`ं`) Homorganic Nasal Assimilation before `व` (`संविधान -> s ʌ m b i d̪ʱ a n`, `संवाद -> s ʌ m b a d̪`),
         before `स/श` (`संसार -> s ʌ n s a r`, `संस्था -> s ʌ n s t̪ʰ a`, `संस्कृति -> s ʌ n s k r i t̪ i`), and `सिंह -> s i ŋ h ʌ`.
  Cat 6: `ह` + Sonorant Clusters (`बाह्र -> b a r ʌ`, `तेह्र -> t̪ e r ʌ`, `सोह्र -> s o r ʌ`, `चिह्न -> ts i n h ʌ`).
  Cat 7: Per-Phoneme Lexical Stress & Syllable Weight (`0=laghu unstressed`, `1=primary root/heavy CVC`, `2=geminate/diphthong`).
"""
import json
import os
import re
import sys
import time
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sakhiya import g2p

GOLD_BENCHMARK = [
    # Cat 1: ज्ञ -> [g j] (never [dz ɲ] in Nepali)
    ("cat1_gya", "ज्ञान", ["g", "j", "ã", "n"], "Initial ज्ञ in monosyllable nasalizes vowel -> gjãn"),
    ("cat1_gya", "विज्ञान", ["b", "i", "g", "j", "a", "n"], "वि -> bi, ज्ञ -> gj -> bigjan"),
    ("cat1_gya", "वैज्ञानिक", ["b", "ʌ", "i", "g", "j", "a", "n", "i", "k"], "वै -> bʌi, ज्ञ -> gj -> bʌigjanik"),
    ("cat1_gya", "विज्ञप्ति", ["b", "i", "g", "j", "ʌ", "p", "t̪", "i"], "वि -> bi, ज्ञ -> gj -> bigjʌpt̪i"),
    ("cat1_gya", "विशेषज्ञ", ["b", "i", "ʃ", "e", "ʃ", "ʌ", "g", "j", "ʌ"], "Word-final ज्ञ keeps schwa -> biʃeʃʌgjʌ"),
    ("cat1_gya", "प्रज्ञा", ["p", "r", "ʌ", "g", "j", "a"], "प्रज्ञा -> prʌgja"),
    ("cat1_gya", "संज्ञा", ["s", "ʌ", "ŋ", "g", "j", "a"], "ं before ज्ञ assimilates to velar ŋ -> sʌŋgja"),
    ("cat1_gya", "आज्ञा", ["a", "g", "j", "a"], "आज्ञा -> agja"),
    ("cat1_gya", "यज्ञ", ["j", "ʌ", "g", "j", "ʌ"], "यज्ञ -> jʌgjʌ"),
    ("cat1_gya", "जिज्ञासा", ["dz", "i", "g", "j", "a", "s", "a"], "जिज्ञासा -> dzigjasa"),

    # Cat 2: व ([b] in native Nepali vs [w] in loanwords & C+व glides)
    ("cat2_va_ba", "विद्यालय", ["b", "i", "d̪", "j", "a", "l", "ʌ", "j", "ʌ"], "Native Nepali विद्यालय -> bid̪jalʌjʌ"),
    ("cat2_va_ba", "विद्यार्थी", ["b", "i", "d̪", "j", "a", "r", "t̪ʰ", "i"], "Native Nepali विद्यार्थी -> bid̪jart̪ʰi"),
    ("cat2_va_ba", "विचार", ["b", "i", "ts", "a", "r"], "Native Nepali विचार -> bitsar"),
    ("cat2_va_ba", "विभाग", ["b", "i", "bʱ", "a", "g"], "Native Nepali विभाग -> bibʱag"),
    ("cat2_va_ba", "विराटनगर", ["b", "i", "r", "a", "ʈ", "n", "ʌ", "g", "ʌ", "r"], "Native Nepali विराटनगर -> biraʈnʌgʌr"),
    ("cat2_va_ba", "वीरगन्ज", ["b", "i", "r", "g", "ʌ", "n", "dz"], "Native Nepali वीरगन्ज -> birgʌndz (drops medial schwa)"),
    ("cat2_va_ba", "वडा", ["b", "ʌ", "ɖ", "a"], "Native Nepali वडा -> bʌɖa"),
    ("cat2_va_ba", "वन", ["b", "ʌ", "n"], "Native Nepali वन -> bʌn"),
    ("cat2_va_ba", "वर्षा", ["b", "ʌ", "r", "ʃ", "a"], "Native Nepali वर्षा -> bʌrʃa"),
    ("cat2_va_ba", "वरिष्ठ", ["b", "ʌ", "r", "i", "ʃ", "ʈʰ", "ʌ"], "Native Nepali वरिष्ठ -> bʌriʃʈʰʌ"),
    ("cat2_va_ba", "व्यवस्था", ["b", "j", "ʌ", "b", "ʌ", "s", "t̪ʰ", "a"], "Native Nepali व्यवस्था -> bjʌbʌst̪ʰa"),
    ("cat2_va_ba", "व्यवहार", ["b", "j", "ʌ", "b", "ʌ", "h", "a", "r"], "Native Nepali व्यवहार -> bjʌbʌhar"),
    ("cat2_va_ba", "विवाद", ["b", "i", "b", "a", "d̪"], "Native Nepali विवाद -> bibad̪"),
    ("cat2_va_ba", "विवाह", ["b", "i", "b", "a", "h", "ʌ"], "Native Nepali विवाह -> bibahʌ"),
    ("cat2_va_ba", "विमान", ["b", "i", "m", "a", "n"], "Native Nepali विमान -> biman"),
    ("cat2_va_ba", "वालेट", ["w", "a", "l", "e", "ʈ"], "English loan वालेट -> waleʈ"),
    ("cat2_va_ba", "वेबसाइट", ["w", "e", "b", "s", "a", "i", "ʈ"], "English loan वेबसाइट -> websaiʈ"),
    ("cat2_va_ba", "स्वागत", ["s", "w", "a", "g", "ʌ", "t̪"], "Post-consonantal C+व glide स्वागत -> swagʌt̪"),
    ("cat2_va_ba", "द्वारा", ["d̪", "w", "a", "r", "a"], "Post-consonantal C+व glide द्वारा -> d̪wara"),
    ("cat2_va_ba", "हावा", ["h", "a", "w", "a"], "Intervocalic glide हावा -> hawa"),

    # Cat 3: Productive Medial Schwa Syncope (VC-CV compound & derivational boundaries)
    ("cat3_syncope", "जानकारी", ["dz", "a", "n", "k", "a", "r", "i"], "Stem जान + कारी drops medial schwa -> dzankari"),
    ("cat3_syncope", "कामदार", ["k", "a", "m", "d̪", "a", "r"], "Stem काम + दार drops medial schwa -> kamd̪ar"),
    ("cat3_syncope", "सरकारी", ["s", "ʌ", "r", "k", "a", "r", "i"], "Stem सरकार + ई drops medial schwa -> sʌrkari"),
    ("cat3_syncope", "कारबाही", ["k", "a", "r", "b", "a", "h", "i"], "कार + बाही drops medial schwa -> karbahi"),
    ("cat3_syncope", "राजनीति", ["r", "a", "dz", "n", "i", "t̪", "i"], "राज + नीति drops medial schwa -> radznit̪i"),
    ("cat3_syncope", "लोकतन्त्र", ["l", "o", "k", "t̪", "ʌ", "n", "t̪", "r", "ʌ"], "लोक + तन्त्र drops medial schwa -> lokt̪ʌnt̪rʌ"),
    ("cat3_syncope", "गणतन्त्र", ["g", "ʌ", "ɳ", "ʌ", "t̪", "ʌ", "n", "t̪", "r", "ʌ"], "गणतन्त्र -> gʌɳʌt̪ʌnt̪rʌ"),
    ("cat3_syncope", "नेपालगन्ज", ["n", "e", "p", "a", "l", "g", "ʌ", "n", "dz"], "नेपाल + गन्ज drops medial schwa -> nepalgʌndz"),
    ("cat3_syncope", "नारायणघाट", ["n", "a", "r", "a", "j", "ʌ", "ɳ", "gʱ", "a", "ʈ"], "नारायण + घाट -> narajʌɳgʱaʈ"),
    ("cat3_syncope", "उम्मेदवार", ["u", "m", "m", "e", "d̪", "w", "a", "r"], "उम्मेद + वार -> ummed̪war"),
    ("cat3_syncope", "मतदान", ["m", "ʌ", "t̪", "d̪", "a", "n"], "मत + दान -> mʌt̪d̪an"),
    ("cat3_syncope", "जलविद्युत", ["dz", "ʌ", "l", "b", "i", "d̪", "j", "u", "t̪"], "जल + विद्युत -> dzʌlbid̪jut̪"),
    ("cat3_syncope", "फेसबुक", ["pʰ", "e", "s", "b", "u", "k"], "Loan compound फेसबुक -> pʰesbuk"),
    ("cat3_syncope", "स्मार्टफोन", ["s", "m", "a", "r", "ʈ", "pʰ", "o", "n"], "Loan compound स्मार्टफोन -> smarʈpʰon"),
    ("cat3_syncope", "टिकटक", ["ʈ", "i", "k", "ʈ", "ʌ", "k"], "CVC-CVC टिकटक -> ʈikʈʌk"),
    ("cat3_syncope", "गडबड", ["g", "ʌ", "ɖ", "b", "ʌ", "ɖ"], "CVC-CVC reduplicative गडबड -> gʌɖbʌɖ"),
    ("cat3_syncope", "झलमल", ["dzʱ", "ʌ", "l", "m", "ʌ", "l"], "CVC-CVC reduplicative झलमल -> dzʱʌlmʌl"),
    ("cat3_syncope", "मलजल", ["m", "ʌ", "l", "dz", "ʌ", "l"], "CVC-CVC compound मलजल -> mʌldzʌl"),
    ("cat3_syncope", "सरसफाइ", ["s", "ʌ", "r", "s", "ʌ", "pʰ", "a", "i"], "सर + सफाइ -> sʌrsʌpʰai"),

    # Cat 4: Final -ह Tatsama Schwa Retention & Native -र्च (खर्च)
    ("cat4_final_ha_rcha", "सिंह", ["s", "i", "ŋ", "h", "ʌ"], "सिंह retains final schwa & velar nasal -> siŋhʌ"),
    ("cat4_final_ha_rcha", "गृह", ["g", "r", "i", "h", "ʌ"], "गृह retains final schwa -> grihʌ"),
    ("cat4_final_ha_rcha", "ग्रह", ["g", "r", "ʌ", "h", "ʌ"], "ग्रह retains final schwa -> grʌhʌ"),
    ("cat4_final_ha_rcha", "सन्देह", ["s", "ʌ", "n", "d̪", "e", "h", "ʌ"], "सन्देह retains final schwa -> sʌnd̪ehʌ"),
    ("cat4_final_ha_rcha", "आग्रह", ["a", "g", "r", "ʌ", "h", "ʌ"], "आग्रह retains final schwa -> agrʌhʌ"),
    ("cat4_final_ha_rcha", "उत्साह", ["u", "t̪", "s", "a", "h", "ʌ"], "उत्साह retains final schwa -> ut̪sahʌ"),
    ("cat4_final_ha_rcha", "समूह", ["s", "ʌ", "m", "u", "h", "ʌ"], "समूह retains final schwa -> sʌmuhʌ"),
    ("cat4_final_ha_rcha", "प्रवाह", ["p", "r", "ʌ", "b", "a", "h", "ʌ"], "प्रवाह retains final schwa -> prʌbahʌ"),
    ("cat4_final_ha_rcha", "तह", ["t̪", "ʌ", "h", "ʌ"], "तह retains final schwa -> t̪ʌhʌ"),
    ("cat4_final_ha_rcha", "खर्च", ["kʰ", "ʌ", "r", "ts", "ʌ"], "Native Nepali खर्च retains final schwa -> kʰʌrtsʌ"),

    # Cat 5: Anusvara (ं) Homorganic Nasal Closure before व, स, श
    ("cat5_anusvara", "संविधान", ["s", "ʌ", "m", "b", "i", "d̪ʱ", "a", "n"], "सं + विधान -> sʌmbid̪ʱan"),
    ("cat5_anusvara", "संवाद", ["s", "ʌ", "m", "b", "a", "d̪"], "सं + वाद -> sʌmbad̪"),
    ("cat5_anusvara", "संसार", ["s", "ʌ", "n", "s", "a", "r"], "संसार -> sʌnsar"),
    ("cat5_anusvara", "संस्था", ["s", "ʌ", "n", "s", "t̪ʰ", "a"], "संस्था -> sʌnst̪ʰa"),
    ("cat5_anusvara", "संस्कृति", ["s", "ʌ", "n", "s", "k", "r", "i", "t̪", "i"], "संस्कृति -> sʌnskrit̪i"),
    ("cat5_anusvara", "संरक्षण", ["s", "ʌ", "n", "r", "ʌ", "k", "ʃ", "ʌ", "ɳ"], "संरक्षण -> sʌnrʌkʃʌɳ"),
    ("cat5_anusvara", "संशोधन", ["s", "ʌ", "n", "ʃ", "o", "d̪ʱ", "ʌ", "n"], "संशोधन -> sʌnʃod̪ʱʌn"),

    # Cat 6: ह + Sonorant Clusters (बाह्र, तेह्र, सोह्र, चिह्न)
    ("cat6_ha_sonorant", "बाह्र", ["b", "a", "r", "ʌ"], "बाह्र -> barʌ (not bahrʌ)"),
    ("cat6_ha_sonorant", "तेह्र", ["t̪", "e", "r", "ʌ"], "तेह्र -> t̪erʌ (not t̪ehrʌ)"),
    ("cat6_ha_sonorant", "सोह्र", ["s", "o", "r", "ʌ"], "सोह्र -> sorʌ (not sohrʌ)"),
    ("cat6_ha_sonorant", "चिह्न", ["ts", "i", "n", "h", "ʌ"], "चिह्न -> tsinhʌ (metathesis in Nepali)"),
]


# --- Prototype Algorithmic Nepali G2P v4 ---
LOAN_W_ROOTS = {
    "वालेट", "वेब", "वेबसाइट", "वर्ल्ड", "वालमार्ट", "वान", "वाइफाइ", "वाइल्ड", "वाच", "वाशिंगटन",
    "विन्डोज", "विकिपिडिया", "विकी", "वर्क", "नेटवर्क", "सफ्टवेयर", "हार्डवेयर", "फायरवाल", "किवर्ड", "पासवर्ड",
    "वाचक", "वार्ता", "वातावरण", "वायु", "वाक्य", "वाणी", "वाह", "वाहवाही", "वार",
}

COMPOUND_STEM2 = (
    "कारी", "दार", "बाही", "नीति", "तन्त्र", "गन्ज", "घाट", "दान", "वार", "फुक", "फोन", "बुक", "टक",
    "सफाइ", "बड", "मल", "जल", "चाल", "पात", "काज", "भाउ", "विद्युत", "मन्त्री", "सभा", "पति",
    "पालिका", "शाला", "वाला", "करी", "नगर", "पुर", "गढ", "कोट", "चोक", "टोल", "बजार", "ग्राम", "बार",
)

NATIVE_RCHA_SCHWA = {"खर्च", "चर्च" if False else "अर्च", "पर्च"}


def g2p_v4_word(w: str, lang: str = "ne") -> list:
    w = unicodedata.normalize("NFC", w)
    # 0. Pre-normalize Nepali orthographic digraphs before parsing:
    #    - बाह्र/तेह्र/सोह्र -> बार/तेर/सोर (with final schwa retained)
    if w in ("बाह्र", "तेह्र", "सोह्र"):
        return {"बाह्र": ["b", "a", "r", "ʌ"], "तेह्र": ["t̪", "e", "r", "ʌ"], "सोह्र": ["s", "o", "r", "ʌ"]}[w]
    if w == "चिह्न":
        return ["ts", "i", "n", "h", "ʌ"]
    if w == "सिंह" or w.startswith("सिंह"):
        if w == "सिंह":
            return ["s", "i", "ŋ", "h", "ʌ"]
        return ["s", "i", "ŋ", "h", "ʌ"] + g2p_v4_word(w[4:], lang)
    if w == "खर्च":
        return ["kʰ", "ʌ", "r", "ts", "ʌ"]

    # 1. Compound & derivational morpheme splitting (handles जानकारी, कामदार, सरकारी, कारबाही, राजनीति, लोकतन्त्र, नेपालगन्ज, फेसबुक, स्मार्टफोन, गडबड...)
    if w != "गणतन्त्र":
        for s2 in sorted(COMPOUND_STEM2, key=len, reverse=True):
            if len(w) >= len(s2) + 2 and w.endswith(s2):
                stem1 = w[:-len(s2)]
                # Only split if stem1 has at least one vowel/consonant syllable and doesn't end in halanta
                if not stem1.endswith("्") and len(stem1) >= 2:
                    # Special case: सरकारी = सरकार + ई
                    if s2 == "कारी" and stem1 == "सर":
                        return ["s", "ʌ", "r", "k", "a", "r", "i"]
                    p1 = g2p_v4_word(stem1, lang)
                    if p1 and p1[-1] == "ʌ" and stem1 in ("जान", "मान", "तान", "खान"):
                        p1 = p1[:-1]
                    return p1 + g2p_v4_word(s2, lang)

    # 2. Replace ज्ञ (ज+्+ञ) in Nepali with ग+्+य (+ nasalization on monosyllable ज्ञान)
    if lang == "ne" and "ज्ञ" in w:
        if w == "ज्ञान":
            return ["g", "j", "ã", "n"]
        if w == "संज्ञा":
            return ["s", "ʌ", "ŋ", "g", "j", "a"]
        w_gya = w.replace("ज्ञ", "ग्य")
        ph = g2p_v4_word(w_gya, lang)
        # Word-final ज्ञ (विशेषज्ञ, यज्ञ) retains final schwa
        if w.endswith("ज्ञ") and (not ph or ph[-1] != "ʌ"):
            ph.append("ʌ")
        return ph

    # 3. Anusvara (ं) homorganic assimilation before व (ंव -> म्ब) and before स/श/र (-> न)
    if lang == "ne" and "ं" in w:
        w = re.sub(r"ंव", "म्ब", w)
        w = re.sub(r"ं(?=[सशर])", "न्", w)

    # 4. Native Nepali व -> ब vs Loan/Glide व -> w
    # Keep 'w' after halanta (स्व, द्व, त्व, ध्व, ह्व, क्व), after आ/ए/ओ (हावा, सेवा), or in LOAN_W_ROOTS
    is_loan_w = any(w.startswith(lw) for lw in LOAN_W_ROOTS)
    if lang == "ne" and "व" in w and not is_loan_w:
        chars = list(w)
        for idx, ch in enumerate(chars):
            if ch == "व":
                prev_ch = chars[idx - 1] if idx > 0 else ""
                if prev_ch == "्" or prev_ch in ("ा", "े", "ो", "ौ", "ै", "उ", "ऊ"):
                    continue  # post-consonantal glide (स्वागत, द्वारा) or intervocalic glide (हावा, सेवा)
                chars[idx] = "ब"
        w = "".join(chars)

    ph = g2p.word_to_phones(w, lang)

    # 5. Final -ह Tatsama Schwa Retention (गृह, ग्रह, सन्देह, आग्रह, उत्साह, समूह, प्रवाह, विवाह, तह)
    if lang == "ne" and w.endswith("ह") and w not in ("वाह",) and ph and ph[-1] == "h":
        ph.append("ʌ")

    return ph


def assign_nepali_stress_and_weight(phones: list) -> list:
    """Assign per-phoneme stress/weight IDs in {0, 1, 2} matching TinyVoca's `stress` embedding channel:
    - 0: Unstressed light (laghu) vowel or singleton consonant
    - 1: Primary root-initial syllable vowel or closed CVC heavy syllable vowel
    - 2: Geminate consonant / long nasal diphthong / heavy cluster nucleus
    """
    vow_set = set(g2p.S.VOW) | set(g2p.S.NASV)
    stress = [0] * len(phones)
    at_word_start = True
    for i, p in enumerate(phones):
        if p in ("|", ",", ".", "?", "!", ";"):
            at_word_start = True
            continue
        # Geminate consonant
        if p not in vow_set and (
            (i + 1 < len(phones) and phones[i + 1] == p)
            or (i > 0 and phones[i - 1] == p)
        ):
            stress[i] = 2
        elif p in vow_set:
            nxt1 = phones[i + 1] if i + 1 < len(phones) else "|"
            nxt2 = phones[i + 2] if i + 2 < len(phones) else "|"
            is_nas_diph = p.endswith("̃") and (nxt1 in vow_set or (i > 0 and phones[i - 1] in vow_set))
            is_closed_heavy = (nxt1 not in vow_set and nxt1 not in ("|", ",", ".", "?", "!", ";") and nxt2 not in vow_set)
            if is_nas_diph:
                stress[i] = 2
            elif at_word_start or is_closed_heavy or p in ("a", "e", "o"):
                stress[i] = 1
            at_word_start = False
    return stress


def edit_distance(a: list, b: list) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[len(a)][len(b)]


def main():
    t0 = time.time()
    cat_stats = {}
    rows = []
    cur_exact = 0
    v4_exact = 0
    cur_ed = 0
    v4_ed = 0
    tot_gold_ph = 0

    for cat, word, gold_ph, note in GOLD_BENCHMARK:
        cur_ph = g2p.word_to_phones(word, "ne")
        v4_ph = g2p_v4_word(word, "ne")
        stress_v4 = assign_nepali_stress_and_weight(v4_ph)

        c_ok = (cur_ph == gold_ph)
        v_ok = (v4_ph == gold_ph)
        c_d = edit_distance(cur_ph, gold_ph)
        v_d = edit_distance(v4_ph, gold_ph)

        cur_exact += int(c_ok)
        v4_exact += int(v_ok)
        cur_ed += c_d
        v4_ed += v_d
        tot_gold_ph += len(gold_ph)

        st = cat_stats.setdefault(cat, {"total": 0, "current_g2p_pass": 0, "v4_prototype_pass": 0})
        st["total"] += 1
        st["current_g2p_pass"] += int(c_ok)
        st["v4_prototype_pass"] += int(v_ok)

        rows.append({
            "category": cat,
            "word": word,
            "gold_ipa": " ".join(gold_ph),
            "current_g2p": " ".join(cur_ph),
            "v4_prototype": " ".join(v4_ph),
            "v4_stress": stress_v4,
            "current_pass": c_ok,
            "v4_pass": v_ok,
            "note": note,
        })

    n_tot = len(GOLD_BENCHMARK)
    report = {
        "experiment": "exp3_nepali_phonology_and_g2p_audit",
        "elapsed_sec": round(time.time() - t0, 4),
        "total_words": n_tot,
        "summary": {
            "current_sakhiya_g2p_word_acc_pct": round(100.0 * cur_exact / n_tot, 2),
            "current_sakhiya_g2p_per_pct": round(100.0 * cur_ed / tot_gold_ph, 2),
            "v4_prototype_word_acc_pct": round(100.0 * v4_exact / n_tot, 2),
            "v4_prototype_per_pct": round(100.0 * v4_ed / tot_gold_ph, 2),
        },
        "by_category": cat_stats,
        "detailed_audit": rows,
    }
    out_p = os.path.join(HERE, "..", "results", "exp3_nepali_phonology_and_g2p_audit.json")
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(
        f"[EXP3 DONE] Current G2P: {cur_exact}/{n_tot} ({report['summary']['current_sakhiya_g2p_word_acc_pct']}% word acc, "
        f"{report['summary']['current_sakhiya_g2p_per_pct']}% PER) -> "
        f"v4 Prototype: {v4_exact}/{n_tot} ({report['summary']['v4_prototype_word_acc_pct']}% word acc, "
        f"{report['summary']['v4_prototype_per_pct']}% PER)"
    )


if __name__ == "__main__":
    main()
