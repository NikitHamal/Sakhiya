"""Sanskrit front-end regression tests: python3 tests/test_sanskrit.py"""
import json, os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sakhiya import sanskrit as sa
from sakhiya.g2p import text_to_phones

P = lambda w: " ".join(sa.word_phones(w))
CASES = {
    "रामः": "r a m ʌ h",                # single-token long ā ('a'), inherent a ('ʌ') kept, final visarga ('h')
    "हरिः": "h ʌ r i h",
    "धर्म": "d̪ʱ ʌ r m ʌ",               # no Nepali-style schwa deletion
    "संस्कृतम्": "s ʌ̃ s k r i t̪ ʌ m",   # anusvara before sibilant nasalises, final m via halant
    "ज्ञानम्": "g ɲ a n ʌ m",
    "गच्छति": "g ʌ ts tsʰ ʌ t̪ i",
    "वेदः": "w e d̪ ʌ h",
}
BAD = [(w, P(w), e) for w, e in CASES.items() if P(w) != e]
assert not BAD, BAD
assert sa.iast_to_deva("dharmakṣetre kurukṣetre") == "धर्मक्षेत्रे कुरुक्षेत्रे", sa.iast_to_deva("dharmakṣetre kurukṣetre")
assert sa.iast_to_deva("asato mā sadgamaya") == "असतो मा सद्गमय"
assert "ईळे" in sa.iast_to_deva("agnim īḷe purohitam")
assert "A" in sa.phonemize_slp1("तत्रऽस्ति")[3]
assert sa.syllable_weights("समवेता") == ["L", "L", "G", "G"]
sh = "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥"
ph = text_to_phones(sh, "sa")
assert ph.count(",") + ph.count(";") >= 3 and ph[-1] == "."          # pada pauses inserted in a 32-syllable shloka

# SLP1 47-token + Guru/Laghu + Vedic Svara + Anushtubh <yati> tests
ids, wt, off, syms = sa.phonemize_slp1(sh)
assert syms[0] == "<bos>" and syms[-1] == "<eos>"
assert "<yati>" in syms, f"expected Anushtubh <yati> in {syms}"
assert ("d", "h") == tuple(syms[1:3]) and wt[3] == 2                 # 'dharma': 'a' before 'rm' is Guru (2)
rv = "अ॒ग्निमी॑ळे पु॒रोहि॑तं य॒ज्ञस्य॑ दे॒वमृ॒त्विज॑म् । होता॑रं रत्न॒धात॑मम् ॥"
r_ids, r_wt, r_off, r_syms = sa.phonemize_slp1(rv)
assert max(r_off) >= 1.0 and min(r_off) <= -1.0 and len(r_ids) == len(r_wt) == len(r_off) == len(r_syms)

# Python <-> JS byte-for-byte parity on phonemizeSlp1 and layaScales
js_code = f"""
global.window = global;
require("./docs/sanskrit.js");
const s1 = window.SA.phonemizeSlp1({json.dumps(sh)});
const s2 = window.SA.phonemizeSlp1({json.dumps(rv)});
const s3 = window.SA.phonemizeSlp1("agnim īḷe purohitam tatra'sti");
const l1 = window.SA.layaScales({json.dumps(sh)});
console.log(JSON.stringify({{s1, s2, s3, l1}}));
"""
out = json.loads(subprocess.check_output(["node", "-e", js_code], text=True))
assert out["s1"]["ids"] == ids and out["s1"]["weight"] == wt and out["s1"]["syms"] == syms
assert out["s2"]["ids"] == r_ids and out["s2"]["weight"] == r_wt and [round(x, 4) for x in out["s2"]["off"]] == [round(x, 4) for x in r_off]
assert out["s3"]["syms"] == sa.phonemize_slp1("agnim īḷe purohitam tatra'sti")[3]
p_ph, p_off, p_dm = sa.laya_scales(sh)
assert out["l1"]["ph"] == p_ph and [round(x, 4) for x in out["l1"]["dm"]] == [round(x, 4) for x in p_dm]
print("sanskrit tests + JS parity ok")


def test_pranava_om_is_spoken_not_dropped():
    """Session 33: U+0950 was silently dropped by the Sanskrit front-end (the verse started at 'p u r ɳ…')."""
    from sakhiya import sanskrit as sa, g2p
    ph, off, dm = sa.laya_scales("ॐ पूर्णमदः पूर्णमिदं पूर्णात्पूर्णमुदच्यते।पूर्णस्य पूर्णमादाय पूर्णमेवावशिष्यते॥")
    assert ph[:3] == ["o", "o", "m"] and ph[3] == "," and len(ph) == len(off) == len(dm)
    assert dm[1] >= 2.0 and off[1] > 0.5                  # prolonged chanted vowel with positive pranava pitch arc
    # Verify Om does not shift Guru/Laghu syllable weights of following words (पूर्णमदः -> G G L G)
    _, _, dm_no_om = sa.laya_scales("पूर्णमदः पूर्णमिदं पूर्णात्पूर्णमुदच्यते।पूर्णस्य पूर्णमादाय पूर्णमेवावशिष्यते॥")
    assert dm[5:] == dm_no_om
    assert sa.laya_scales("ॐ")[0][:3] == ["o", "o", "m"]
    assert "o" not in sa.text_to_phones("पूर्णमदः")        # no phantom om without the sign
    # Spaceless danda '।' (U+0964) must split words and emit '.' pause in Nepali G2P too
    ne_ph = g2p.text_to_phones("पूर्णमुदच्यते।पूर्णस्य", "ne")
    assert "." in ne_ph[:-1], f"expected '.' pause at spaceless danda in {ne_ph}"


test_pranava_om_is_spoken_not_dropped()
