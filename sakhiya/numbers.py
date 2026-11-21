"""Nepali number / date / currency / acronym expansion (text -> Devanagari words)."""
import re

_U = ("शून्य एक दुई तीन चार पाँच छ सात आठ नौ दश एघार बाह्र तेह्र चौध पन्ध्र सोह्र सत्र अठार उन्नाइस "
      "बिस एक्काइस बाइस तेइस चौबिस पच्चिस छब्बिस सत्ताइस अठ्ठाइस उनन्तिस तिस "
      "एकतिस बत्तिस तेत्तिस चौँतिस पैँतिस छत्तिस सैँतिस अठतिस उनन्चालिस चालिस "
      "एकचालिस बयालिस त्रिचालिस चवालिस पैँतालिस छयालिस सतचालिस अठचालिस उनन्चास पचास "
      "एकाउन्न बाउन्न त्रिपन्न चौवन्न पचपन्न छपन्न सन्ताउन्न अन्ठाउन्न उनान्न साठी "
      "एकसट्ठी बैसट्ठी त्रिसट्ठी चौँसट्ठी पैँसट्ठी छैसट्ठी सतसट्ठी अठसट्ठी उनान्सत्तरी सत्तरी "
      "एकहत्तर बहत्तर त्रिहत्तर चौहत्तर पचहत्तर छयाहत्तर सतहत्तर अठहत्तर उनासी असी "
      "एकासी बयासी त्रियासी चौरासी पचासी छयासी सतासी अठासी उनान्नब्बे नब्बे "
      "एकान्नब्बे बयान्नब्बे त्रियान्नब्बे चौरान्नब्बे पन्चान्नब्बे छयान्नब्बे सन्तान्नब्बे अन्ठान्नब्बे उनान्सय").split()
assert len(_U) == 100, len(_U)
_BIG = [(10**11, "खर्ब"), (10**9, "अर्ब"), (10**7, "करोड"), (10**5, "लाख"), (1000, "हजार"), (100, "सय")]
MONTHS = "बैशाख जेठ असार श्रावण भदौ असोज कात्तिक मंसिर पौष माघ फागुन चैत".split()
_DIG = str.maketrans("०१२३४५६७८९", "0123456789")
LETTERS = {c: n for c, n in zip("abcdefghijklmnopqrstuvwxyz",
           "ए बी सी डी इ एफ जी एच आई जे के एल एम एन ओ पी क्यु आर एस टी यु भी डब्लु एक्स वाई जेड".split())}


def num_words(n):
    if n < 100:
        return _U[n]
    for v, w in _BIG:
        if n >= v:
            q, r = divmod(n, v)
            head = num_words(q) + " " + w
            return head + (" " + num_words(r) if r else "")
    return ""


def digits_words(s):
    return " ".join(_U[int(c)] for c in s)


def _number(tok):
    tok = tok.replace(",", "")
    if "." in tok:
        a, b = tok.split(".", 1)
        return num_words(int(a or 0)) + " दशमलव " + digits_words(b)
    if len(tok) > 1 and tok[0] == "0":
        return digits_words(tok)
    if len(tok) > 13:
        return digits_words(tok)
    return num_words(int(tok))


def expand(text):
    t = text.translate(_DIG)
    # dates: YYYY-MM-DD or YYYY/MM/DD (Bikram Sambat style)
    def date(m):
        y, mo, d = int(m[1]), int(m[2]), int(m[3])
        if 1 <= mo <= 12 and 1 <= d <= 32:
            return f"{num_words(y)} साल {MONTHS[mo-1]} {num_words(d)} गते"
        return m[0]
    t = re.sub(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", date, t)
    t = re.sub(r"\b(\d{1,2}):(\d{2})\b", lambda m: num_words(int(m[1])) + " बजेर " + num_words(int(m[2])) + " मिनेट"
               if int(m[2]) else num_words(int(m[1])) + " बजे", t)
    t = re.sub(r"(?:रु\.?|रू\.?|Rs\.?|NPR)\s*(\d[\d,]*(?:\.\d+)?)", lambda m: _number(m[1]) + " रुपैयाँ", t)
    t = re.sub(r"(\d[\d,]*(?:\.\d+)?)\s*%", lambda m: _number(m[1]) + " प्रतिशत", t)
    t = re.sub(r"(\d[\d,]*(?:\.\d+)?)\s*(?:किमी|km)\b", lambda m: _number(m[1]) + " किलोमिटर", t)
    t = re.sub(r"\d[\d,]*(?:\.\d+)?", lambda m: _number(m[0].rstrip(",")), t)
    # acronyms: 2-5 capital Latin letters -> spelled letters
    t = re.sub(r"\b([A-Z]{2,5})\b", lambda m: " ".join(LETTERS[c.lower()] for c in m[1]), t)
    return t
