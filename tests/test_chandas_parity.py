import json
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from sakhiya import chandas as CH


TEST_VERSES = [
    ("sa_gayatri", "ॐ भूर्भुवः स्वः। तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि। धियो यो नः प्रचोदयात्॥", "Gayatri"),
    ("sa_rv", "अ॒ग्निमी॑ळे पु॒रोहि॑तं य॒ज्ञस्य॑ दे॒वमृ॒त्विज॑म् । होता॑रं रत्न॒धात॑मम् ॥", "Gayatri"),
    ("sa_gita", "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः। मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय॥", "Anushtubh"),
    ("sa_om", "ॐ पूर्णमदः पूर्णमिदं पूर्णात्पूर्णमुदच्यते। पूर्णस्य पूर्णमादाय पूर्णमेवावशिष्यते॥", "Anushtubh"),
    ("sa_unseen", "गुरूर्ब्रह्मा गुरूर्विष्णुः गुरुर्देवो महेश्वरः। गुरुः साक्षात् परं ब्रह्म तस्मै श्रीगुरवे नमः॥", "Anushtubh"),
]


def test_chandas_laghu_guru_and_meters():
    for cid, text, expected_meter in TEST_VERSES:
        res = CH.compile_chandas(text, bpm=75.0, tonic_hz=215.0)
        assert res["meter"] == expected_meter, f"{cid}: expected {expected_meter}, got {res['meter']}"
        weights = [a["weight"] for a in res["aksharas"]]
        assert "G" in weights and "L" in weights
        assert len(res["ph"]) == len(res["off"]) == len(res["dm"]) == len(res["dur_frames"])
        assert len(res["f0_hz_contour"]) == sum(res["dur_frames"])
        # Check zero F0 discontinuities (> 3.5st frame-to-frame jump)
        import math
        f0 = res["f0_hz_contour"]
        max_jump_st = max(abs(12.0 * math.log2(f0[i + 1] / f0[i])) for i in range(len(f0) - 1))
        assert max_jump_st < 1.5, f"{cid}: F0 discontinuity {max_jump_st:.2f} st"


def test_python_js_1to1_parity():
    node_code = """
    const fs = require('fs');
    global.window = global;
    eval(fs.readFileSync('docs/sanskrit.js', 'utf8'));
    eval(fs.readFileSync('docs/chandas.js', 'utf8'));
    const verses = JSON.parse(process.argv[1]);
    const out = {};
    for (const [cid, text] of verses) {
      out[cid] = window.CHANDAS.compileChandas(text, 75.0, 215.0, true);
    }
    console.log(JSON.stringify(out));
    """
    payload = json.dumps([[cid, txt] for cid, txt, _ in TEST_VERSES], ensure_ascii=False)
    r = subprocess.run(["node", "-e", node_code, payload], cwd=REPO, capture_output=True, text=True, check=True)
    js_out = json.loads(r.stdout)

    for cid, txt, _ in TEST_VERSES:
        py = CH.compile_chandas(txt, bpm=75.0, tonic_hz=215.0, add_end=True)
        js = js_out[cid]
        assert py["meter"] == js["meter"], f"{cid} meter mismatch"
        assert py["ph"] == js["ph"], f"{cid} ph mismatch: {py['ph']} vs {js['ph']}"
        assert [a["weight"] for a in py["aksharas"]] == [a["weight"] for a in js["aksharas"]], f"{cid} Laghu/Guru mismatch"
        assert [a["ak"] for a in py["aksharas"]] == [a["ak"] for a in js["aksharas"]], f"{cid} Akshara text mismatch"
        assert py["dur_frames"] == js["dur_frames"], f"{cid} dur_frames mismatch"
        for k, (po, jo) in enumerate(zip(py["off"], js["off"])):
            assert abs(po - jo) <= 1e-3, f"{cid} off[{k}] mismatch: {po} vs {jo}"
        for k, (pd, jd) in enumerate(zip(py["dm"], js["dm"])):
            assert abs(pd - jd) <= 1e-3, f"{cid} dm[{k}] mismatch: {pd} vs {jd}"
        for k, (pf, jf) in enumerate(zip(py["f0_hz_contour"], js["f0_hz_contour"])):
            assert abs(pf - jf) <= 1e-2, f"{cid} f0_hz_contour[{k}] mismatch: {pf} vs {jf}"
