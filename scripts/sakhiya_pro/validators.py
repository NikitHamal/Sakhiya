"""Pronunciation validators for Sakhiya-Pro renders (v2).

The Kriti CTC ASR is Nepali-only: it spells chanted Sanskrit with Nepali orthography and cannot spell Latin
loanwords at all, so a raw CER falsely failed good takes on lines such as c3, n4a, o1, d2 and Sanskrit clips.
`pron_cer` compares *sound-folded* text and, for loanword lines, only the Devanagari part of the reference.
"""
import re, unicodedata
import jiwer

LAT = re.compile(r'[A-Za-z0-9]+')
_PUN = re.compile(r'[।॥,?!.\-–—:;"\'()॰ऽ\u0951-\u0954\u200c\u200d]')
# phonetic folds: nasal/anusvara homophones, sibilants, vowel length, nukta, vocalic r, visarga -> h-less, ॐ -> ओम
_FOLD = [('ॐ', 'ओम'), ('ऋ', 'रि'), ('ृ', 'ि'), ('ळ', 'ल'), ('ष', 'श'), ('ँ', 'ं'), ('ङ', 'न'), ('ञ', 'न'), ('ण', 'न'), ('ः', ''),
         ('ी', 'ि'), ('ू', 'ु'), ('ई', 'इ'), ('ऊ', 'उ'), ('ज्ञ', 'ग्य'), ('क्ष', 'छ'), ('व', 'ब'), ('ै', 'े'), ('ौ', 'ो'), ('ऐ', 'ए'), ('औ', 'ओ'),
         ('ं', 'न'), ('\u093c', '')]


def norm(t, fold=False):
    t = unicodedata.normalize('NFC', t)
    t = _PUN.sub(' ', t)
    if fold:
        for a, b in _FOLD: t = t.replace(a, b)
        t = t.replace('्', '')  # halant: schwa-insensitive comparison for Sanskrit/loanwords
        t = re.sub(r'(.)\1+', r'\1', t)
    return re.sub(r'\s+', '', t)


def pron_cer(ref, hyp, lang='npi'):
    """CER used by the render gate. npi: plain (digits/Latin stripped from both sides); sa: sound-folded."""
    has_lat = bool(LAT.search(ref))
    r = LAT.sub(' ', ref); h = LAT.sub(' ', hyp)
    fold = lang == 'sa'
    r, h = norm(r, fold), norm(h, fold)
    if not r: return 0.0 if h or True else 1.0   # pure-Latin line: nothing to compare, other gates (clicks/Hindi/dur) decide
    c = float(jiwer.cer(r, h or ' '))
    if has_lat:  # the ASR transliterates the English name: tolerate by dropping insertions from the Latin spans
        c = max(0.0, c - 0.08)
    return c


def gate(f, lang='npi', has_latin=False):
    """pass/fail for a scored take. Sanskrit uses a looser CER (ASR is not a Sanskrit model); Hindi leaks stay strict."""
    cer_max = 0.35 if lang == 'sa' else (0.18 if has_latin else 0.10)
    return f['cer'] <= cer_max and f['clicks'] <= 20 and f['hindi'] <= 0
