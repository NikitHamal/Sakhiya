"""Phase 1 G2P Schwa Deletion, Hiatus, ृ, व, Visarga & Chunker Parity Tests (Python <-> docs/g2p.js + docs/tts.js)."""
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sakhiya import g2p, onnx_tts

AUDITED_30_WORDS = {
    "मोडेल": ["m", "o", "ɖ", "e", "l"],
    "कोड": ["k", "o", "ɖ"],
    "ओपन": ["o", "p", "ʌ", "n"],
    "पोइन्ट": ["p", "o", "i", "n", "ʈ"],
    "अनलाइन": ["ʌ", "n", "l", "a", "i", "n"],
    "किनमेल": ["k", "i", "n", "m", "e", "l"],
    "वालेट": ["w", "a", "l", "e", "ʈ"],
    "सेयर": ["s", "e", "j", "ʌ", "r"],
    "बजार": ["b", "ʌ", "dz", "a", "r"],
    "कारोबार": ["k", "a", "r", "o", "b", "a", "r"],
    "करिब": ["k", "ʌ", "r", "i", "b"],
    "गिरावट": ["g", "i", "r", "a", "w", "ʌ", "ʈ"],
    "समेत": ["s", "ʌ", "m", "e", "t̪"],
    "उतारचढाव": ["u", "t̪", "a", "r", "ts", "ʌ", "ɖʱ", "a", "w"],
    "अनुसार": ["ʌ", "n", "u", "s", "a", "r"],
    "डलर": ["ɖ", "ʌ", "l", "ʌ", "r"],
    "अर्थात": ["ʌ", "r", "t̪ʰ", "a", "t̪"],
    "मोबाइल": ["m", "o", "b", "a", "i", "l"],
    "एप्लिकेसन": ["e", "p", "l", "i", "k", "e", "s", "ʌ", "n"],
    "हरेक": ["h", "ʌ", "r", "e", "k"],
    "दिन": ["d̪", "i", "n"],
    "दैनिक": ["d̪", "ʌ", "i", "n", "i", "k"],
    "जीवन": ["dz", "i", "w", "ʌ", "n"],
    "प्रभाव": ["p", "r", "ʌ", "bʱ", "a", "w"],
    "भएका": ["bʱ", "ʌ", "e", "k", "a"],
    "भएको": ["bʱ", "ʌ", "e", "k", "o"],
    "भइरहेको": ["bʱ", "ʌ", "i", "r", "ʌ", "h", "e", "k", "o"],
    "भइरहेका": ["bʱ", "ʌ", "i", "r", "ʌ", "h", "e", "k", "a"],
    "बृहत": ["b", "r", "i", "h", "ʌ", "t̪"],
    "मामकाः": ["m", "a", "m", "ʌ", "k", "a", "h"],
}

CONSONANT_FINAL_WORDS_NO_SCHWA = [
    "मोडेल", "कोड", "ओपन", "पोइन्ट", "अनलाइन", "किनमेल", "वालेट", "सेयर",
    "बजार", "कारोबार", "करिब", "गिरावट", "समेत", "उतारचढाव", "अनुसार", "डलर",
    "अर्थात", "मोबाइल", "एप्लिकेसन", "हरेक", "दिन", "दैनिक", "जीवन", "प्रभाव",
    "हजार", "बृहत",
]

CONJUNCT_FINAL_KEEP_SCHWA = {
    "प्रसिद्ध": ["p", "r", "ʌ", "s", "i", "d̪", "d̪ʱ", "ʌ"],
    "स्पष्ट": ["s", "p", "ʌ", "ʃ", "ʈ", "ʌ"],
    "शब्द": ["ʃ", "ʌ", "b", "d̪", "ʌ"],
    "राष्ट्र": ["r", "a", "ʃ", "ʈ", "r", "ʌ"],
    "चन्द्र": ["ts", "ʌ", "n", "d̪", "r", "ʌ"],
    "तर्क": ["t̪", "ʌ", "r", "k", "ʌ"],
}


def test_30_audited_words_and_schwa_rules():
    for w, expected in AUDITED_30_WORDS.items():
        got = g2p.word_to_phones(w, "ne")
        assert got == expected, f"Word {w}: expected {expected}, got {got}"

    for w in CONSONANT_FINAL_WORDS_NO_SCHWA:
        ph = g2p.word_to_phones(w, "ne")
        assert ph[-1] != "ʌ", f"Consonant-final word {w} must NOT emit trailing schwa: {ph}"

    for w, expected in CONJUNCT_FINAL_KEEP_SCHWA.items():
        got = g2p.word_to_phones(w, "ne")
        assert got == expected, f"Conjunct-final word {w}: expected {expected}, got {got}"

    # Verify suffixes and compounds drop stem-final schwa cleanly
    assert g2p.word_to_phones("दिनमै", "ne") == ["d̪", "i", "n", "m", "ʌ", "i"]
    assert g2p.word_to_phones("होडबाजीका", "ne") == ["h", "o", "ɖ", "b", "a", "dz", "i", "k", "a"]
    assert g2p.word_to_phones("संवेदनशील", "ne") == ["s", "ʌ̃", "w", "e", "d̪", "ʌ", "n", "ʃ", "i", "l"]
    assert g2p.word_to_phones("लिइरहेको", "ne") == ["l", "i", "j", "i", "r", "ʌ", "h", "e", "k", "o"]
    assert "|" not in g2p.word_to_phones("लिइरहेको", "ne")


def test_python_js_parity_on_30_words_and_4_owner_clips():
    clips_data = json.load(open(os.path.join(ROOT, "eval", "owner_4clips.json")))["clips"]
    clip_texts = [c["text"] for c in clips_data]
    words = list(AUDITED_30_WORDS.keys()) + ["दिनमै", "होडबाजीका", "संवेदनशील", "लिइरहेको", "प्रसिद्ध", "स्पष्ट", "शब्द", "राष्ट्र", "चन्द्र", "कृत्रिम", "वृद्धि", "दुःख", "प्रायः", "क्रमशः", "अतः", "युयुत्सवः"]

    js_code = f"""
    const fs = require("fs");
    global.window = global;
    global.fetch = async (u) => ({{ json: async () => JSON.parse(fs.readFileSync("docs/" + u, "utf8")) }});
    require("./docs/g2p.js");
    require("./docs/sanskrit.js");
    (async () => {{
      await window.SK.load();
      const words = {json.dumps(words, ensure_ascii=False)};
      const clips = {json.dumps(clip_texts, ensure_ascii=False)};
      const w_out = {{}};
      for (const w of words) {{
        w_out[w] = window.SK.textToPhones(w, "ne");
      }}
      const c_out = clips.map(c => window.SK.textToPhones(c, "ne"));
      console.log(JSON.stringify({{ w_out, c_out }}));
    }})();
    """
    res = json.loads(subprocess.check_output(["node", "-e", js_code], text=True, cwd=ROOT).strip())
    for w in words:
        py_ph = g2p.text_to_phones(w, "ne")
        js_ph = res["w_out"][w]
        assert py_ph == js_ph, f"Python <-> JS mismatch on word {w}: py={py_ph} vs js={js_ph}"

    for c_txt, js_ph in zip(clip_texts, res["c_out"]):
        py_ph = g2p.text_to_phones(c_txt, "ne")
        assert py_ph == js_ph, f"Python <-> JS mismatch on clip {c_txt[:40]}: py={py_ph[:30]} vs js={js_ph[:30]}"


def test_chunker_no_mid_phrase_slicing_on_4_owner_clips():
    clips_data = json.load(open(os.path.join(ROOT, "eval", "owner_4clips.json")))["clips"]
    forbidden_breaks = [
        ("कामहरू", "छिटो"),
        ("संस्करणहरू", "ल्याउँदै"),
        ("सुविधा", "सुरु"),
        ("नयाँ", "परिवर्तनहरूबारे"),
        ("प्रभाव", "हाम्रो"),
    ]
    for c in clips_data:
        segs = onnx_tts._segment(c["text"])
        for s_prev, s_next in zip(segs, segs[1:]):
            for w_end, w_start in forbidden_breaks:
                assert not (s_prev.endswith(w_end) and s_next.startswith(w_start)), (
                    f"Forbidden mid-clause slice between '{w_end}' and '{w_start}': [{s_prev}] | [{s_next}]"
                )
