"""Nepal-native respelling of English loanwords for OmniVoice (pure Devanagari, Kathmandu news-anchor phonology).

Kathmandu anchors say loanwords with Nepali vowel quantity / consonant values: short English /i/ -> long ी in acronyms
(जीपीटी), 'to' -> टू, 'model' -> मोडल, sh/zh -> स in 'motion' (मोसन), final ङ/ँ kept thick (छौँ, पाँचौँ, भेटौँला).
`nativize()` is applied to the *input text* only (never to training targets).
"""
import re

_PAIRS = [
    ('जिपिटी', 'जीपीटी'), ('च्याटजिपिटी', 'च्याटजीपीटी'), ('च्याटजीपिटी', 'च्याटजीपीटी'),
    ('सिक्स पोइन्ट टु', 'सिक्स प्वाइन्ट टू'), ('पोइन्ट', 'प्वाइन्ट'),
    ('टेक्स्ट टु स्पीच', 'टेक्स्ट टू स्पिच'), ('स्पीच', 'स्पिच'),
    ('मोडेलहरू', 'मोडलहरू'), ('मोडेल', 'मोडल'),
    ('लिक्विड ग्लास', 'लिक्विड ग्लास'), ('स्प्रिङ मोसन', 'स्प्रिङ मोसन'), ('पाथवेज', 'प्याथवेज'),
    ('जेन्सेन हुआङ', 'जेन्सन ह्वाङ'), ('जेन्सेन', 'जेन्सन'), ('हुआङ', 'ह्वाङ'),
    ('छौं', 'छौँ'), ('गरौं', 'गरौँ'), ('पाँचौं', 'पाँचौँ'), ('भेटौंला', 'भेटौँला'), ('औंला', 'औँला'),
]
_PAIRS.sort(key=lambda p: -len(p[0]))


def nativize(t: str) -> str:
    for a, b in _PAIRS:
        t = t.replace(a, b)
    return t
