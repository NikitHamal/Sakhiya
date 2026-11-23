#!/usr/bin/env python3
"""Pre-TTS Nepali news text normalizer for Sakhiya-Pro v2.

* Devanagari / ASCII digits -> Nepali number words (0-99 irregular table, सय/हजार/लाख/करोड).
* decimals (३.५ -> तीन दशमलव पाँच), percentages (% -> प्रतिशत), simple dates (2082/08/12 -> साल/महिना/गते).
* digits glued to / next to Latin brand tokens (GPT-6.2, Gemini 4 Neon) become English words, so OmniVoice
  reads the proper noun naturally; the Latin spelling itself is never touched (Anthropic, Jensen Huang, Liquid Glass Dock).
"""
import re, unicodedata

N99 = ("शून्य एक दुई तीन चार पाँच छ सात आठ नौ दस एघार बाह्र तेह्र चौध पन्ध्र सोह्र सत्र अठार उन्नाइस "
       "बीस एक्काइस बाइस तेइस चौबीस पच्चीस छब्बीस सत्ताइस अट्ठाइस उनन्तिस तीस एकतीस बत्तीस तेत्तीस चौँतीस पैँतीस छत्तीस सैँतीस अठत्तीस उनन्चालीस "
       "चालीस एकचालीस बयालीस त्रिचालीस चौवालीस पैंतालीस छयालीस सतचालीस अठचालीस उनन्चास पचास एकाउन्न बाउन्न त्रिपन्न चौवन्न पचपन्न छपन्न सन्ताउन्न अन्ठाउन्न उनान्साठी "
       "साठी एकसट्ठी बैसट्ठी त्रिसट्ठी चौसट्ठी पैँसट्ठी छयसट्ठी सतसट्ठी अठसट्ठी उनान्सत्तरी सत्तरी एकहत्तर बहत्तर त्रिहत्तर चौहत्तर पचहत्तर छयहत्तर सतहत्तर अठहत्तर उनासी "
       "असी एकासी बयासी त्रियासी चौरासी पचासी छयासी सतासी अठासी उनान्नब्बे नब्बे एकान्नब्बे बयान्नब्बे त्रियान्नब्बे चौरान्नब्बे पन्चानब्बे छयान्नब्बे सन्तानब्बे अन्ठानब्बे उनान्सय").split()
assert len(N99) == 100, len(N99)
MONTHS = "बैशाख जेठ असार श्रावण भदौ असोज कार्तिक मंसिर पौष माघ फागुन चैत".split()
EN = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_DEV = str.maketrans('०१२३४५६७८९', '0123456789')


def ne_int(n: int) -> str:
    if n < 100: return N99[n]
    parts = []
    for unit, name in ((10 ** 7, 'करोड'), (10 ** 5, 'लाख'), (1000, 'हजार'), (100, 'सय')):
        q, n = divmod(n, unit)
        if q: parts.append(ne_int(q) + ' ' + name)
    if n: parts.append(N99[n])
    return ' '.join(parts)


def en_int(n: int) -> str:
    if n < 20: return EN[n]
    if n < 100: return TENS[n // 10] + ('-' + EN[n % 10] if n % 10 else '')
    if n < 1000: return EN[n // 100] + ' hundred' + (' ' + en_int(n % 100) if n % 100 else '')
    if n < 10 ** 6: return en_int(n // 1000) + ' thousand' + (' ' + en_int(n % 1000) if n % 1000 else '')
    return ' '.join(EN[int(c)] for c in str(n))


def _num(s: str, lang='ne') -> str:
    s = s.translate(_DEV).replace(',', '')
    f = ne_int if lang == 'ne' else en_int
    if '.' in s:
        a, b = s.split('.', 1)
        dec = 'दशमलव' if lang == 'ne' else 'point'
        dig = [N99[int(c)] if lang == 'ne' else EN[int(c)] for c in b]
        return f(int(a or 0)) + ' ' + dec + ' ' + ' '.join(dig)
    if len(s) > 1 and s[0] == '0':  # leading zeros -> digit by digit
        return ' '.join(N99[int(c)] if lang == 'ne' else EN[int(c)] for c in s)
    if len(s) > 12: return ' '.join(N99[int(c)] if lang == 'ne' else EN[int(c)] for c in s)
    return f(int(s))


_D = r'[0-9०-९]'
NUM = rf'{_D}+(?:,{_D}{{2,3}})*(?:\.{_D}+)?'
LAT = r'[A-Za-z]'


def normalize(t: str) -> str:
    t = unicodedata.normalize('NFC', t).replace('\u200b', '').replace('\ufeff', '')
    # dates  2082/08/12 | 2082-08-12 | १२/०८/२०८२
    def date(m):
        a, b, c = (x.translate(_DEV) for x in m.groups())
        y, mo, d = (a, b, c) if len(a) == 4 else (c, b, a)
        try:
            mo_i, d_i = int(mo), int(d)
            if not (1 <= mo_i <= 12 and 1 <= d_i <= 32): return m.group(0)
        except ValueError:
            return m.group(0)
        return f'{ne_int(int(y))} साल {MONTHS[mo_i - 1]} {ne_int(d_i)} गते'
    t = re.sub(rf'({_D}{{4}}|{_D}{{1,2}})[/\-.]({_D}{{1,2}})[/\-.]({_D}{{4}}|{_D}{{1,2}})', lambda m: date(m) if (len(m.group(1)) == 4 or len(m.group(3)) == 4) else m.group(0), t)
    # digits touching Latin brand tokens -> English
    t = re.sub(rf'(?<={LAT})[\s\-]?({NUM})', lambda m: ' ' + _num(m.group(1), 'en'), t)
    t = re.sub(rf'({NUM})(?=[\s\-]?{LAT})(?!\s*(?:%|प्रतिशत))', lambda m: _num(m.group(1), 'en') + ' ', t)
    # percentages / currency
    t = re.sub(rf'({NUM})\s*%', lambda m: _num(m.group(1)) + ' प्रतिशत', t)
    t = re.sub(rf'(?:रु\.?|Rs\.?|रू\.?)\s*({NUM})', lambda m: 'रुपैयाँ ' + _num(m.group(1)), t)
    t = re.sub(rf'({NUM})', lambda m: _num(m.group(1)), t)
    t = re.sub(r'\bdr\.?\b', 'डा.', t) if False else t
    t = re.sub(r'\s+', ' ', t).strip()
    return t


if __name__ == '__main__':
    for s in ['नेपालको जनसंख्या २ करोड ९० लाख छ।', 'वृद्धिदर ३.५% रह्यो', '२०८२/०८/१२ मा', 'GPT-6.2 र Gemini 4 Neon आए', 'रु. १,२५,००० पर्‍यो', 'Anthropic को Jensen Huang']:
        print(s, '=>', normalize(s))
