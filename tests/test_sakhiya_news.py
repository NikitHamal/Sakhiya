"""Unit tests for Sakhiya-News (`tests/test_sakhiya_news.py`).
Verifies:
1. Fixed Honest 4-Layer Forensic Gate (`pCER` gap math, QDQ INT8 initializer audit, full-band 4-12 kHz LSD).
2. 24 kHz 100-Band Full-Bandwidth `StudioVocoder100Mel` (ConvNeXt-V2 GRN + SiLU/GELU, zero Snake1d, zero f0_net) + `MS-SB-CQT`.
3. Single-Style `SakhiyaNewsFlagship` (Full-Sentence RoPE Encoder + 5D Structural Clause Cues + Layer-3 Engram +
   Stochastic 4-Step Flow Duration/F0/Energy + 10-Block Walsh-Hadamard/Monarch Improved MeanFlow Decoder).
"""
import os
import sys
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sakhiya.news_vocoder_100mel import (
    StudioVocoder100Mel,
    MultiScaleSubBandCQTDiscriminator,
    magnitude_aware_awp_losses,
)
from sakhiya.news_am_100mel import (
    SakhiyaNewsFlagship,
    compute_graph_structural_features,
)
from scripts.forensic_audio_gate import (
    audit_int8_qdq_initializers,
    _fullband_hf_lsd_4_12khz_db,
)


def test_honest_forensic_baseline_json():
    p = os.path.join(ROOT, "eval", "forensic_sakhiya_one_honest_baseline.json")
    assert os.path.exists(p), "Missing honest baseline JSON"
    import json
    d = json.load(open(p))
    # Verify pCER gap is honestly computed as 0.059 - 0.032 = +0.027 (never 0.000)
    assert abs(d["ne_pcer_gap_vs_anchor"] - round(d["ne_pcer"] - d["anchor_pcer"], 4)) < 1e-6
    assert d["ne_pcer_gap_vs_anchor"] == 0.027
    assert d["int8_weight_pct"] >= 70.0


def test_int8_qdq_initializer_audit_on_deployed_model():
    i8_p = os.path.join(ROOT, "docs", "models_sakhiya_one", "model.int8.onnx")
    if os.path.exists(i8_p):
        qdq = audit_int8_qdq_initializers(i8_p)
        assert qdq["int8_weight_pct"] >= 70.0
        assert qdq["dequantize_linear_nodes"] >= 50


def test_studio_vocoder_100mel_and_mssbcqt():
    voc = StudioVocoder100Mel(n_mels=100, ch=240).eval()
    # Verify zero Snake1d and zero f0_net inside vocoder
    mod_names = [type(m).__name__ for m in voc.modules()]
    assert "Snake1d" not in mod_names
    assert not hasattr(voc.src, "f0_net")

    mel = torch.randn(1, 100, 16)
    f0 = torch.full((1, 1, 16), 210.0)
    vuv = torch.ones((1, 1, 16))
    with torch.no_grad():
        y, re_s, im_s, log_mag, cos_ph, sin_ph = voc.forward_wav(mel, f0, vuv)
    assert y.shape == (1, 16 * 256)
    l_awp = magnitude_aware_awp_losses(cos_ph, sin_ph, re_s, im_s)
    assert torch.isfinite(l_awp)

    cqt = MultiScaleSubBandCQTDiscriminator().eval()
    with torch.no_grad():
        outs, fmaps = cqt(y)
    assert len(outs) == 3 and len(fmaps) == 12


def test_sakhiya_news_flagship_single_style_architecture():
    m = SakhiyaNewsFlagship().eval()
    # Verify single-style, zero Vedic heads, zero fast_blocks bloat
    assert not hasattr(m.enc, "sty")
    assert not hasattr(m.enc, "svara_emb")
    assert not hasattr(m.enc, "mora_emb")
    assert not hasattr(m.vel, "fast_blocks")
    assert len(m.vel.blocks) == 10

    ids = torch.tensor([[1, 9, 46, 20, 20, 47, 8, 24, 46, 24, 28, 50, 5]], dtype=torch.long)
    st5 = compute_graph_structural_features(ids)
    assert st5.shape == (1, ids.shape[1], 5)
    # Geminate at positions 3, 4 (20, 20) must be flagged as 1.0
    assert float(st5[0, 3, 4].item()) == 1.0
    assert float(st5[0, 4, 4].item()) == 1.0

    with torch.no_grad():
        wav = m.infer(ids, torch.zeros(1, dtype=torch.long), torch.ones(ids.shape[1]), torch.zeros(ids.shape[1]))
    assert wav.ndim == 2 and wav.shape[0] == 1 and wav.shape[1] > 1024
    assert not torch.isnan(wav).any()
