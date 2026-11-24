import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import leak_loop_gate as G

SR = 24000
GAYATRI = "ॐ भूर्भुवः स्वः । तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि । धियो यो नः प्रचोदयात् ॥"


def tone(dur, f=180.0, amp=0.3):
    t = np.arange(int(dur * SR)) / SR
    env = np.minimum(1, np.minimum(t, dur - t) / 0.03)
    return (amp * env * (np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * 2 * f * t))).astype(np.float32)


def test_leak_tail_rejected():
    # v4 failure: Shanti/Gayatri opened with ref tail "प्रचोदयात्" which is not in the target
    target = "ॐ पूर्णमदः पूर्णमिदं पूर्णात् पूर्णमुदच्यते ॥"
    hyp = "ओम् नम् नम् नम् प्रचोदयात् पूर्णमदः पूर्णमिदं"
    assert "प्रचोदयात्" in G.leaked_words(GAYATRI, target, hyp)


def test_no_leak_when_word_in_target():
    assert G.leaked_words(GAYATRI, "ॐ भूर्भुवः स्वः", "ॐ भूर्भुवः स्वः") == []


def test_word_loop_rejected():
    r = G.loop_report("अग्निमीळे पुरोहितं यज्ञस्य", "अग्नि अग्निमीळे पुरोहितं अग्नि अग्नि यज्ञस्य")
    assert r and r[0]["kind"] == "word"
    assert G.loop_report("नमः नमः शिवाय", "नमः नमः शिवाय") == []   # allowed: target itself repeats


def test_namnamnam_loop():
    assert G.loop_report("ॐ भूर्भुवः", "ओम् नम् नम् नम् भूर्भुवः")


def test_clean_clip_passes():
    w = np.concatenate([np.zeros(int(.2 * SR), np.float32), tone(1.0), np.zeros(int(.2 * SR), np.float32)])
    r = G.check_clip(w, SR, "ॐ भूर्भुवः", GAYATRI, hyp="ॐ भूर्भुवः", mode="vedic")
    assert r["ok"], r


def test_hard_splice_detected():
    a = tone(0.5, amp=0.4)
    b = np.concatenate([np.zeros(int(.4 * SR), np.float32)])
    w = np.concatenate([np.zeros(int(.1 * SR), np.float32), a[:int(.37 * SR)], b])   # cut mid-voice, no fade
    assert G.find_splices(w, SR)


def test_collapsed_om_rejected():
    w = np.concatenate([tone(0.30), np.zeros(int(.3 * SR), np.float32), tone(1.2)])
    assert G.first_voiced_run_s(w, SR) < 0.45
    r = G.check_clip(w, SR, "ॐ भूर्भुवः", "", hyp=None, mode="vedic")
    assert any(x.startswith("OM_COLLAPSED") for x in r["rejects"])


def test_rich_om_accepted():
    w = np.concatenate([np.zeros(int(.15 * SR), np.float32), tone(1.7), np.zeros(int(.2 * SR), np.float32)])
    assert G.first_voiced_run_s(w, SR) >= 1.5


def test_clicks_rejected():
    w = tone(1.0)
    for p in (3000, 8000, 12000):
        w[p] = 0.9
    assert G.count_clicks(w, SR) >= 3
