"""Sakhiya v5 G2P + Sanskrit + JS parity regression suite (`tests/test_g2p_v5.py`)."""
import json, os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sakhiya import g2p, sanskrit as sa


def test_v5_g2p_fixes_and_js_parity():
    # 1. काठमाडौँ dual-nasal diphthong (ʌ̃ ũ)
    ktm = g2p.word_to_phones("काठमाडौँ")
    assert ktm == ["k", "a", "ʈʰ", "m", "a", "ɖ", "ʌ̃", "ũ"], ktm
    # 2. आआफ्नो vowel hiatus boundary
    aafno = g2p.word_to_phones("आआफ्नो")
    assert aafno == ["a", "|", "a", "pʰ", "n", "o"], aafno
    # 3. वाह! crisp single 'h' with no spurious final schwa
    wah = g2p.text_to_phones("वाह! के अद्भुत दृश्य हो यो!", "ne")
    assert wah[:4] == ["w", "a", "h", "!"], wah[:6]
    # 4. गहिरो श्वास लिनुहोस् र आजको पललाई clause break before र
    warm = g2p.text_to_phones("गहिरो श्वास लिनुहोस् र आजको पललाई", "ne")
    assert "," in warm, f"Expected prosodic comma after लिनुहोस् before र: {warm}"
    # 5. भूर्भुवः single 'h' visarga (zero duplicate 'h')
    bhur = sa.word_phones("भूर्भुवः")
    assert bhur == ["bʱ", "u", "r", "bʱ", "u", "w", "ʌ", "h"], bhur

    # 6. 100% Python <-> docs/g2p.js + docs/sanskrit.js parity
    sents = [
        "नेपालको राजधानी काठमाडौँ हिमालको काखमा रहेको एउटा ऐतिहासिक सहर हो।",
        "शान्त मन र मधुर मुस्कानले हरेक कठिन दिनलाई सहज बनाउँछ। गहिरो श्वास लिनुहोस् र आजको पललाई हृदयदेखि अनुभव गर्नुहोस्।",
        "वाह! के अद्भुत दृश्य हो यो!",
        "गाउँलेहरू आआफ्नो खेतबारीमा काम गर्न निस्किए।",
    ]
    js_code = f"""
    const fs = require("fs");
    global.window = global;
    global.fetch = async (u) => ({{ json: async () => JSON.parse(fs.readFileSync("docs/" + u, "utf8")) }});
    require("./docs/g2p.js");
    require("./docs/sanskrit.js");
    (async () => {{
      await window.SK.load();
      const sents = {json.dumps(sents)};
      const out = sents.map(s => window.SK.textToPhones(s, "ne"));
      console.log(JSON.stringify(out));
    }})();
    """
    js_out = json.loads(subprocess.check_output(["node", "-e", js_code], text=True).strip())
    for s, j_ph in zip(sents, js_out):
        p_ph = g2p.text_to_phones(s, "ne")
        assert p_ph == j_ph, f"Parity mismatch on {s}: py={p_ph} vs js={j_ph}"


if __name__ == "__main__":
    test_v5_g2p_fixes_and_js_parity()
    print("test_g2p_v5 OK")
