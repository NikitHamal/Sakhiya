import json
import os
import subprocess
from sakhiya import g2p

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

OWNER_AUDITED_WORDS = {
    # 1. व -> ब leak words (must use true labio-velar approximant 'w', NEVER 'b'):
    "वालेट": ["w", "a", "l", "e", "ʈ"],
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
    # 2. Geminates & halanta stops:
    "बुद्धिमत्ता": ["b", "u", "d̪", "d̪ʱ", "i", "m", "ʌ", "t̪", "t̪", "a"],
    "उल्लेखनीय": ["u", "l", "l", "e", "kʰ", "ʌ", "n", "i", "j", "ʌ"],
    "पक्कै": ["p", "ʌ", "k", "k", "ʌ", "i"],
    "चिटिक्क": ["ts", "i", "ʈ", "i", "k", "k", "ʌ"],
    "बृहत्": ["b", "r", "i", "h", "ʌ", "t̪"],
    "बृहत": ["b", "r", "i", "h", "ʌ", "t̪"],
    "अर्थात्": ["ʌ", "r", "t̪ʰ", "a", "t̪"],
    "अर्थात": ["ʌ", "r", "t̪ʰ", "a", "t̪"],
    # 3. Chandrabindu nasal diphthongs:
    "गर्नेछौँ": ["g", "ʌ", "r", "n", "e", "tsʰ", "ʌ̃", "ũ"],
    "लागौँ": ["l", "a", "g", "ʌ̃", "ũ"],
    "चलौँ": ["ts", "ʌ", "l", "ʌ̃", "ũ"],
    "पाँचौँ": ["p", "ã", "ts", "ʌ̃", "ũ"],
    "काठमाडौँ": ["k", "a", "ʈʰ", "m", "a", "ɖ", "ʌ̃", "ũ"],
    "अर्बौँ": ["ʌ", "r", "b", "ʌ̃", "ũ"],
    # 4. Long compounds:
    "अनुसन्धानात्मक": ["ʌ", "n", "u", "s", "ʌ", "n", "d̪ʱ", "a", "n", "a", "t̪", "m", "ʌ", "k"],
    "प्रयोगकर्ताको": ["p", "r", "ʌ", "j", "o", "g", "k", "ʌ", "r", "t̪", "a", "k", "o"],
    "उतारचढाव": ["u", "t̪", "a", "r", "ts", "ʌ", "ɖʱ", "a", "w"],
    "त्रैमासिकमा": ["t̪", "r", "ʌ", "i", "m", "a", "s", "i", "k", "m", "a"],
    "गिरावट": ["g", "i", "r", "a", "w", "ʌ", "ʈ"],
    "उपलब्धिहरू": ["u", "p", "ʌ", "l", "ʌ", "b", "d̪ʱ", "i", "h", "ʌ", "r", "u"],
    "सुरक्षित": ["s", "u", "r", "ʌ", "k", "ʃ", "i", "t̪"],
    # 5. Tech loanwords:
    "जेमिनाई": ["dz", "e", "m", "i", "n", "a", "i"],
    "फोर": ["pʰ", "o", "r"],
    "नियोन": ["n", "i", "j", "o", "n"],
    "जिपिटी": ["dz", "i", "p", "i", "ʈ", "i"],
    "जीपीटी": ["dz", "i", "p", "i", "ʈ", "i"],
    "सिक्स": ["s", "i", "k", "s"],
    "पोइन्ट": ["p", "o", "i", "n", "ʈ"],
    "टु": ["ʈ", "u"],
    "ओपन": ["o", "p", "ʌ", "n"],
    "एआई": ["e", "|", "a", "i"],
    "एआईका": ["e", "|", "a", "i", "k", "a"],
    "एआईले": ["e", "|", "a", "i", "l", "e"],
    "च्याटजिपिटी": ["ts", "j", "a", "ʈ", "dz", "i", "p", "i", "ʈ", "i"],
    "एन्थ्रोपिक": ["e", "n", "t̪ʰ", "r", "o", "p", "i", "k"],
    "एन्थ्रोपिकको": ["e", "n", "t̪ʰ", "r", "o", "p", "i", "k", "k", "o"],
    "मोबाइल": ["m", "o", "b", "a", "i", "l"],
    "एप्लिकेसन": ["e", "p", "l", "i", "k", "e", "s", "ʌ", "n"],
}

OWNER_PHRASES = {
    "च्याटजिपिटी वालेट": ["ts", "j", "a", "ʈ", "dz", "i", "p", "i", "ʈ", "i", "|", "w", "a", "l", "e", "ʈ", "."],
    "त चलौँ": ["t̪", "ʌ", "|", "ts", "ʌ", "l", "ʌ̃", "ũ", "."],
    "गिरावट आयो": ["g", "i", "r", "a", "w", "ʌ", "ʈ", "|", "a", "j", "o", "."],
    "१७ प्रतिशतले": ["s", "ʌ", "t̪", "r", "ʌ", "|", "p", "r", "ʌ", "t̪", "i", "ʃ", "ʌ", "t̪", "l", "e", "."],
    "१४ प्रतिशतले": ["ts", "ʌ", "u", "d̪ʱ", "ʌ", "|", "p", "r", "ʌ", "t̪", "i", "ʃ", "ʌ", "t̪", "l", "e", "."],
    "जेमिनाई फोर नियोन": ["dz", "e", "m", "i", "n", "a", "i", "|", "pʰ", "o", "r", "|", "n", "i", "j", "o", "n", "."],
    "जिपिटी सिक्स पोइन्ट टु": ["dz", "i", "p", "i", "ʈ", "i", "|", "s", "i", "k", "s", "|", "p", "o", "i", "n", "ʈ", "|", "ʈ", "u", "."],
    "ओपन एआई": ["o", "p", "ʌ", "n", "|", "e", "|", "a", "i", "."],
    "मोबाइल एप्लिकेसन": ["m", "o", "b", "a", "i", "l", "|", "e", "p", "l", "i", "k", "e", "s", "ʌ", "n", "."],
}


def test_python_owner_words_and_phrases():
    for w, expected in OWNER_AUDITED_WORDS.items():
        got = g2p.word_to_phones(w, "ne")
        assert got == expected, f"Mismatch on {w}: got {got}, expected {expected}"
    # Verify वालेट never starts with 'b'
    assert g2p.word_to_phones("वालेट", "ne")[0] == "w"
    assert "b" not in g2p.word_to_phones("वालेट", "ne")

    for phrase, expected in OWNER_PHRASES.items():
        got = g2p.text_to_phones(phrase, "ne")
        assert got == expected, f"Phrase mismatch on {phrase}: got {got}, expected {expected}"


def test_python_js_100pct_parity_on_owner_words_and_clips():
    owner_clips = json.load(open(os.path.join(REPO, "eval", "owner_4clips.json")))["clips"][:3]
    all_inputs = list(OWNER_AUDITED_WORDS.keys()) + list(OWNER_PHRASES.keys()) + [c["text"] for c in owner_clips]

    js_code = f"""
const fs = require('fs');
global.window = {{}};
const fe = JSON.parse(fs.readFileSync({ json.dumps(os.path.join(REPO, "docs", "data", "frontend.json")) }, 'utf8'));
const g2pSrc = fs.readFileSync({ json.dumps(os.path.join(REPO, "docs", "g2p.js")) }, 'utf8');
eval(g2pSrc + "\\nF = fe; window.SK.F = fe;");
const inputs = { json.dumps(all_inputs, ensure_ascii=False) };
const out = {{}};
for (const inp of inputs) {{
  out[inp] = window.SK.textToPhones(inp, 'ne');
}}
console.log(JSON.stringify(out));
"""
    res = subprocess.run(["node", "-e", js_code], capture_output=True, text=True, check=True)
    js_map = json.loads(res.stdout.strip())
    for inp in all_inputs:
        py_ph = g2p.text_to_phones(inp, "ne")
        js_ph = js_map[inp]
        assert py_ph == js_ph, f"Py <-> JS mismatch on {inp!r}:\n  Py: {py_ph}\n  JS: {js_ph}"
