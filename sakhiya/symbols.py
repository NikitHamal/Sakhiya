"""Phoneme inventory for Nepali (+ slots for Tharu / regional languages).

Tokens are IPA-like strings. Breathy-voiced aspirates (gʱ dzʱ ɖʱ d̪ʱ bʱ), dental vs
retroflex stops and nasalised vowels are all distinct symbols.
"""
PAD = "_"
WORD = "|"          # word boundary
PUNCT = [",", ".", "?", "!", ";"]   # ';' = long pause / clause break
BOS, EOS = "<s>", "</s>"

CONS = ["k", "kʰ", "g", "gʱ", "ŋ", "ts", "tsʰ", "dz", "dzʱ", "ɲ",
        "ʈ", "ʈʰ", "ɖ", "ɖʱ", "ɳ", "t̪", "t̪ʰ", "d̪", "d̪ʱ", "n",
        "p", "pʰ", "b", "bʱ", "m", "j", "r", "l", "w", "ʃ", "s", "h", "ɽ", "ɽʱ", "z", "f"]
VOW = ["ʌ", "a", "i", "u", "e", "o", "ɛ", "ɔ"]
NASV = [v + "̃" for v in VOW]
LANG_TOKENS = ["<ne>", "<thr>", "<thr_rana>", "<thr_chit>", "<mai>", "<bho>", "<new>", "<tam>"]

SYMBOLS = [PAD, WORD, BOS, EOS] + PUNCT + CONS + VOW + NASV
SYM2ID = {s: i for i, s in enumerate(SYMBOLS)}
N_SYMBOLS = len(SYMBOLS)

LANG2ID = {"ne": 0, "thr": 1, "thr_dangaura": 1, "thr_rana": 2, "thr_chitwania": 3,
           "sa": 4, "mai": 5, "bho": 6, "new": 7, "taj": 7}


def encode(phones):
    return [SYM2ID[p] for p in phones if p in SYM2ID]
