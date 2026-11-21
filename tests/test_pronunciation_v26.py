"""v2.6 owner pronunciation: exact phoneme strings, NO duplicate phones, ALL duration multipliers 1.0, Python<->Node parity"""
import json, os, subprocess, sys
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from sakhiya import sanskrit as sa
from sakhiya.g2p import text_to_phones, phone_dm

NE = ["टेक्स्ट टु स्पीच छोट टेक्स्ट संघरियाहुक्रे हामी हों", "नेपालको राजधानी काठमाडौँ हिमालको काखमा रहेको एउटा ऐतिहासिक सहर हो।",
      "एक समयको कुरा हो, एउटा सानो गाउँमा एउटी बुढी आमा बस्थिन्। उनी बिहानै उठेर चरालाई दाना हाल्थिन्।"]
SA = ["ॐ पूर्णमदः पूर्णमिदं पूर्णात्पूर्णमुदच्यते। पूर्णस्य पूर्णमादाय पूर्णमेवावशिष्यते॥",
      "अ॒ग्निमी॑ळे पु॒रोहि॑तं य॒ज्ञस्य॑ दे॒वमृ॒त्विज॑म् । होता॑रं रत्न॒धात॑मम् ॥",
      "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥"]



def _s(t, lang="ne"):
    return " ".join(sa.laya_scales(t, v2=True)[0] if lang == "sa" else text_to_phones(t, "ne"))


def test_no_duplicate_adjacent_phones():
    for t in NE:
        ph = text_to_phones(t, "ne")
        assert all(not (a == b and a not in (".", ",", "|")) for a, b in zip(ph, ph[1:])), (t, ph)
    for t in SA:
        ph = sa.laya_scales(t, v2=True)[0]
        assert all(not (a == b and a not in (".", ",", "|", ";")) for a, b in zip(ph, ph[1:])), (t, ph)


def test_exact_owner_words():
    assert "k a kʰ m a" in _s("काखमा")                       # no काख्वा
    assert _s("उठेर") == "u ʈʰ e r ʌ ."                      # single flap r, no उठेरर
    assert _s("एक समयको").startswith("e k |")                # crisp onset, no एर
    assert "ʃ ts ʌ i w ʌ" in _s("पाण्डवाश्चैव", "sa")         # श्च keeps both ʃ and च(ts)
    assert _s("मामकाः", "sa") == "m a m ʌ k a h ."          # light h, NO echo vowel (no मामकाहा)
    assert "j u j u t̪ s ʌ w ʌ h ." in _s("युयुत्सवः", "sa")
    assert "d̪ ʌ ts j ʌ" in _s("उदच्यते", "sa") and "d̪ʱ ʌ ts" not in _s("उदच्यते", "sa")
    assert _s("ॐ", "sa").startswith("o m")                   # resonant o + m, no 'o o m'


def test_no_duration_hacks():
    for t in NE:
        ph = text_to_phones(t, "ne")
        dm = phone_dm(ph)
        for p, d in zip(ph, dm):
            if p not in (",", "!", "?"):
                assert d == 1.0, (t, p, d)
    for t in SA:
        ph, _, dm = sa.laya_scales(t, v2=True)
        assert len(ph) == len(dm) and all(0.7 <= d <= 3.5 for d in dm), (t, set(dm))


def test_word_boundary_and_nasal():
    ph = text_to_phones("टेक्स्ट टु"); i = ph.index("|")
    assert ph[i - 1] == "ʈ" and ph[i + 1] == "ʈ"             # boundary token survives -> no टेक्सटु fusion
    for lg in ("ne", "thr_dangaura"):
        ph = text_to_phones("हों", lg); assert any(p.endswith("\u0303") for p in ph) and "o" not in ph, ph
    assert "|" not in text_to_phones("संघरियाहुक्रे")


def test_js_parity():
    code = f"""global.window = global; global.fetch = async u => ({{json: async () => JSON.parse(require("fs").readFileSync("docs/" + u, "utf8"))}});
require("./docs/g2p.js"); require("./docs/sanskrit.js"); (async () => {{ await window.SK.load();
const ne = {json.dumps(NE)}, sa = {json.dumps(SA)};
console.log(JSON.stringify({{ne: ne.map(t => {{ const ph = window.SK.textToPhones(t, "ne"); return {{ph, dm: window.SK.phoneDm(ph)}}; }}), sa: sa.map(t => window.SA.layaScales(t, true, true))}})); }})();"""
    out = json.loads(subprocess.check_output(["node", "-e", code], text=True, cwd=ROOT))
    rd = lambda xs: [round(x, 4) for x in xs]
    for t, o in zip(NE, out["ne"]):
        ph = text_to_phones(t, "ne"); assert o["ph"] == ph, t; assert rd(o["dm"]) == rd(phone_dm(ph)), t
    for t, o in zip(SA, out["sa"]):
        p_ph, _, p_dm = sa.laya_scales(t, True, True); assert o["ph"] == p_ph, t; assert rd(o["dm"]) == rd(p_dm), t


if __name__ == "__main__":
    for f in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        f()
    print("v2.6 pronunciation + parity tests passed")
