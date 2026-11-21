"""Auto (Mixed) router, owner pronunciation regressions and Python<->Node parity: python3 tests/test_mixed.py"""
import json, os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sakhiya import mixed, sanskrit as sa
from sakhiya.g2p import word_to_phones, text_to_phones

OWNER = ("हेलो साथीहरू। म सखिया हुँ, निकित हमालज्यूले बनाउनुभएको सानो टेक्स्ट टु स्पीच मोडेल। म संस्कृत पनि बोल्न सक्छु। "
         "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥ र थारू भाषामा पनि बोल्न सक्छु। "
         "हेलो संघरियाहुक्रे। मै सखिया हों, निकित हमाल जी बनाइल छोट टेक्स्ट टु स्पीच मोडेल।")
import unicodedata
D = lambda x: unicodedata.normalize("NFD", x)
P = lambda w, l="ne": " ".join(word_to_phones(w, l))


def test_router():
    segs = mixed.route(OWNER)
    assert [s[0] for s in segs] == ["ne", "sa", "ne", "thr_dangaura"], [s[0] for s in segs]
    assert "युयुत्सवः" in segs[1][1] and "सञ्जय ॥" in segs[1][1]            # both shloka halves stay ONE Sanskrit segment
    assert segs[2][1].startswith("र थारू") and segs[3][1].startswith("हेलो संघरियाहुक्रे") and "मै सखिया हों" in segs[3][1]
    assert mixed.detect_lang("दुःख लाग्यो।") == "ne" and mixed.detect_lang("अतः म गएँ।") == "ne"   # Nepali words with visarga are not Sanskrit
    assert mixed.detect_lang("ॐ पूर्णमदः") == "sa" and mixed.detect_lang("अ॒ग्निमी॑ळे") == "sa"
    assert [c for c in mixed.split_clauses("मूल्य १.५ रुपैयाँ। ठीक छ?")] == ["मूल्य १.५ रुपैयाँ।", "ठीक छ?"]


def test_nasal_o():
    for l in ("ne", "thr_dangaura", "thr_rana", "thr_chitwania"):
        assert D(P("हों", l)) == D("h õ"), (l, P("हों", l))
    assert D(P("छौं")) == D("tsʰ ʌ̃ ũ") and D(P("गएँ")) == D("g ʌ ẽ") and D(P("हुँ")) == D("h ũ")
    assert "o m" not in P("हों")


def test_dzju_honorific():
    for w in ("हमालज्यूले", "दाजुज्यू", "गुरुज्यू"):
        assert "dz j u" in P(w), (w, P(w))
    assert P("हमालज्यूले") == "h ʌ m a l dz j u l e", P("हमालज्यूले")


def test_loanwords():
    assert P("टेक्स्ट") == "ʈ e k s ʈ" and P("स्पीच") == "s p i ts" and P("मोडेल") == "m o ɖ e l"
    assert P("कम्प्युटर").endswith("ʈ ʌ r") and P("फोन") == "pʰ o n"
    for w in ("स्पीच", "स्टेशन", "स्कूल"):
        assert word_to_phones(w)[0] == "s" and word_to_phones(w)[1] in ("p", "ʈ", "k"), w     # no prothetic vowel
    assert P("धर्म") == "d̪ʱ ʌ r m ʌ" and P("अन्त") == "ʌ n t̪ ʌ"                              # native schwa retention untouched


def test_honorific_compound():
    for w in ("बनाउनुभएको", "गर्नुभएको", "भन्नुभएको"):
        ph = word_to_phones(w)
        assert "|" not in ph and ph[-6:] == ["bʱ", "ʌ", "j", "e", "k", "o"], (w, ph)


def test_sanskrit_v2():
    ph = sa.laya_scales("धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥", v2=True)[0]
    s = " ".join(ph)
    assert "j u j u t̪ s ʌ w ʌ h" in s and "h ʌ" not in s.replace("ʌ w ʌ h", ""), s   # v2.6: dental t̪+s kept, plain visarga h (no echo)
    assert "m a m ʌ k a h" in s and "k a h a" not in s
    assert sa.laya_scales("ॐ पूर्णमदः", v2=True)[0][:2] == ["o", "m"]


def test_js_parity():
    code = f"""global.window = global; global.document = undefined;
require("./docs/sanskrit.js"); require("./docs/mixed.js");
const t = {json.dumps(OWNER)};
const sh = {json.dumps("धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥")};
console.log(JSON.stringify({{r: window.MIX.route(t), c: window.MIX.splitClauses(t), l: window.SA.layaScales(sh, true, true)}}));"""
    out = json.loads(subprocess.check_output(["node", "-e", code], text=True, cwd=os.path.join(os.path.dirname(__file__), "..")))
    assert [list(x) for x in mixed.route(OWNER)] == out["r"] and mixed.split_clauses(OWNER) == out["c"]
    p_ph, _, p_dm = sa.laya_scales("धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥", v2=True)
    assert out["l"]["ph"] == p_ph and [round(x, 4) for x in out["l"]["dm"]] == [round(x, 4) for x in p_dm]


def test_js_g2p_parity():
    from sakhiya.g2p import phone_dm
    words = ["हों", "छौं", "हमालज्यूले", "टेक्स्ट", "स्पीच", "मोडेल", "बनाउनुभएको", "गर्नुभएको", "भन्नुभएको", "संघरियाहुक्रे", "धर्म", "निकित",
             "काठमाडौँ", "जाऔँ", "खाऔँ", "विद्यालय", "विद्यालयको", "पुस्तकालय", "माछापुच्छ्रे", "साँझपख", "वाह"]
    code = f"""global.window = global; global.fetch = async u => ({{json: async () => JSON.parse(require("fs").readFileSync("docs/" + u, "utf8"))}});
require("./docs/g2p.js"); (async () => {{ await window.SK.load(); const ws = {json.dumps(words)}; console.log(JSON.stringify({{ph: ws.map(w => window.SK.textToPhones(w, "ne").slice(0, -1)), dm: ws.map(w => window.SK.phoneDm(window.SK.textToPhones(w, "ne")))}})); }})();"""
    out = json.loads(subprocess.check_output(["node", "-e", code], text=True, cwd=os.path.join(os.path.dirname(__file__), "..")))
    for w, js, js_dm in zip(words, out["ph"], out["dm"]):
        assert js == word_to_phones(w), (w, js, word_to_phones(w))
        assert [round(x, 4) for x in js_dm] == [round(x, 4) for x in phone_dm(text_to_phones(w, "ne"))], (w, js_dm)


if __name__ == "__main__":
    for f in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        f()
    print("mixed-router + pronunciation + JS parity tests passed")
