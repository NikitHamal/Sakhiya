"""Unit tests for `sakhiya.voca` (Nepali Phonological G2P v4 + SakhiyaVoca model & source-filter head)."""

import torch

from sakhiya.voca import (
    SakhiyaVoca,
    SakhiyaVocaConfig,
    evaluate_gold_70,
    g2p_v4_word,
    phonemize_utterance,
    sentence_split_bucket,
)


def test_nepali_phonological_gold_70():
    rep = evaluate_gold_70()
    assert rep["correct"] == rep["total"] == 70, f"Gold-70 failures: {rep['failures']}"


def test_v_mode_kathmandu_vs_rasa():
    assert g2p_v4_word("विद्यालय", v_mode="kathmandu")[0] == "b"
    assert g2p_v4_word("विद्यालय", v_mode="rasa")[0] == "w"
    assert g2p_v4_word("वालेट", v_mode="kathmandu")[0] == "w"
    assert g2p_v4_word("संविधान", v_mode="rasa")[:4] == ["s", "ʌ", "m", "b"]


def test_sentence_split_bucket_deterministic():
    s = "नमस्ते, म सखिया हुँ। विद्यालय र विज्ञानको विकास महत्त्वपूर्ण छ।"
    b1 = sentence_split_bucket(s)
    b2 = sentence_split_bucket(s)
    assert 0 <= b1 < 1000 and b1 == b2


def test_sakhiya_voca_4_5m_synthesize():
    cfg = SakhiyaVocaConfig.tiny_4_5m()
    model = SakhiyaVoca(cfg).eval()
    n_params = sum(p.numel() for p in model.parameters())
    assert 4_400_000 <= n_params <= 4_700_000

    _, ids, stress = phonemize_utterance("नमस्ते नेपाल।")
    tok = torch.tensor([ids], dtype=torch.long)
    st = torch.tensor([stress], dtype=torch.long)
    wave, durs = model.synthesize(tok, st, speaker=0)
    assert wave.shape[0] == 1 and wave.shape[1] == int(durs.sum().item()) * cfg.hop
    assert not torch.isnan(wave).any()


def test_sakhiya_voca_onnx_export(tmp_path):
    from sakhiya.voca import export_onnx_models
    cfg = SakhiyaVocaConfig.tiny_4_5m()
    model = SakhiyaVoca(cfg).eval()
    rep = export_onnx_models(model, tmp_path / "onnx", prefix="test_voca")
    assert rep["nan_count"] == 0
    assert rep["len_diff_int8"] == 0
    assert rep["int8_mb"] < 6.0
    assert rep["rtf_1t_cpu"] < 0.05
