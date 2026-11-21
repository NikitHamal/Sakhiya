"""Phonology regression tests: python -m pytest tests -q  (or python tests/test_frontend.py)"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sakhiya.g2p import word_to_phones, text_to_phones
from sakhiya import numbers, translit

P = lambda w: " ".join(word_to_phones(w))
CASES = {
    "घर": "gʱ ʌ r",                       # final schwa dropped after simple C
    "धर्म": "d̪ʱ ʌ r m ʌ",               # final schwa kept after conjunct
    "गर्छ": "g ʌ r tsʰ ʌ",
    "छ": "tsʰ ʌ",                         # monosyllable keeps schwa
    "कसरी": "k ʌ s ʌ r i",                # medial schwa kept (no Hindi deletion)
    "समय": "s ʌ m ʌ j ʌ",                 # Khatiwada (2009): final -य retains schwa
    "जसले": "dz ʌ s l e",                 # stem + postposition boundary
    "त्यसपछि": "t̪ j ʌ s p ʌ tsʰ i",
    "घरमा": "gʱ ʌ r m a",
    "भाइ": "bʱ a i",
    "छैन": "tsʰ ʌ i n ʌ",                 # negative verb retains final schwa
    "बाट": "b a ʈ ʌ",                     # postposition बाट retains final schwa
    "काठमाडौँ": "k a ʈʰ m a ɖ ʌ̃ ũ",       # compound schwa deletion + clean oral-onset nasal diphthong
    "गाउँ": "g ã ũ",                      # nasal diphthong
    "आउँछ": "ã ũ tsʰ ʌ",                  # nasal diphthong + finite -छ schwa
    "भैँसी": "bʱ ʌ̃ ĩ s i",                # nasal diphthong
    "सँग": "s ʌ̃ g ʌ",                    # native Nepali postposition retains final schwa
    "आज": "a dz ʌ",                       # native Nepali adverb retains final schwa (unlike Hindi 'aaj')
    "गर्दछ": "g ʌ r d̪ ʌ tsʰ ʌ",          # finite -छ verb retains final schwa
    "रहेछ": "r ʌ h e tsʰ ʌ",
    "बनाउन": "b ʌ n a u n ʌ",             # -उन infinitive retains final schwa
    "पुगेर": "p u g e r ʌ",               # -एर converb retains final schwa (unlike Hindi -kar)
    "हुँदैन": "h ũ d̪ ʌ i n ʌ",            # -दैन negative verb retains final schwa
    "भएन": "bʱ ʌ e n ʌ",                  # -एन negative past verb retains final schwa
    "घरहरूसँग": "gʱ ʌ r h ʌ r u s ʌ̃ g ʌ", # stacked plural + postposition without stem-final schwa resurrection
    "दिनभरि": "d̪ i n bʱ ʌ r i",           # -भरि suffix boundary without medial schwa resurrection
    "सुख": "s u kʰ ʌ",                    # native Nepali सुख retains final schwa
    "दुख": "d̪ u kʰ ʌ",                    # native Nepali दुख retains final schwa
    "बजारमे": "b ʌ dz a r m e",           # Tharu locative clitic -मे without spurious stem-final schwa
    "घरसे": "gʱ ʌ r s e",                 # Tharu ablative clitic -से
    "गामके": "g a m k e",                 # Tharu genitive/dative clitic -के
}


def test_schwa():
    for w, exp in CASES.items():
        assert P(w) == exp, (w, P(w), exp)


def test_numbers():
    assert numbers.num_words(1250) == "एक हजार दुई सय पचास"
    assert numbers.expand("रु. 100") == "एक सय रुपैयाँ"
    assert "भदौ" in numbers.expand("2081-05-12")


def test_roman():
    assert translit.roman_to_deva("ramro") == "राम्रो"
    assert translit.roman_to_deva("kasari") == "कसरी"
    assert translit.roman_to_deva("gaunma") == "गाउँमा"
    assert translit.roman_to_deva("gaoon") == "गाउँ"
    assert translit.roman_to_deva("gaaun") == "गाउँ"
    assert translit.roman_to_deva("baschu") == "बस्छु"
    assert translit.roman_to_deva("thauma") == "ठाउँमा"
    assert translit.roman_to_deva("tohar", "thr_dangaura") == "तोहर"
    assert translit.roman_to_deva("paanime", "thr_rana") == "पानीमे"


def test_text():
    ph = text_to_phones("तपाईं कहाँ जानुहुन्छ?")
    assert ph[-1] == "?" and "|" in ph


if __name__ == "__main__":
    test_schwa(); test_numbers(); test_roman(); test_text(); print("all front-end tests passed")
