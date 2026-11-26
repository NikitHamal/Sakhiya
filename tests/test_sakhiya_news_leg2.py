import json
import os
from sakhiya import g2p

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_nepali_geminate_contrast_pairs():
    pairs = [
        ("कति", "कत्ति", "t̪"),
        ("पकै", "पक्कै", "k"),
        ("बचा", "बच्चा", "ts"),
    ]
    for sing_w, gem_w, cons in pairs:
        p_s = g2p.word_to_phones(sing_w, "ne")
        p_g = g2p.word_to_phones(gem_w, "ne")
        assert p_s != p_g, f"{sing_w} and {gem_w} must have distinct phoneme sequences"
        assert p_g.count(cons) == p_s.count(cons) + 1, f"{gem_w} must contain geminate {cons}"

    for gem_w, cons in [("चिटिक्क", "k"), ("ठ्ठयाक्कै", "k"), ("सक्कली", "k"), ("छब्बीस", "b")]:
        ph = g2p.word_to_phones(gem_w, "ne")
        has_adj = any(ph[i] == cons and ph[i + 1] == cons for i in range(len(ph) - 1))
        assert has_adj, f"{gem_w} -> {ph} missing adjacent geminate {cons}"


def test_sakhiya_news_leg2_final_gate_and_onnx():
    onnx_p = os.path.join(REPO, "docs", "models_sakhiya_news", "model.int8.onnx")
    gate_p = os.path.join(REPO, "eval", "forensic_sakhiya_news_final.json")
    raw_b_p = os.path.join(REPO, "eval", "raw_anchor_physics_baseline.json")
    e2e_b_p = os.path.join(REPO, "eval", "full_e2e_node_ort_benchmark.json")
    tiers_p = os.path.join(REPO, "docs", "tiers.json")

    assert os.path.isfile(onnx_p), "Missing docs/models_sakhiya_news/model.int8.onnx"
    sz_mb = os.path.getsize(onnx_p) / 1e6
    assert 8.0 <= sz_mb <= 9.80, f"ONNX size {sz_mb:.3f} MB outside [8.0, 9.80] MB"

    fg = json.load(open(gate_p))
    assert fg["gate_pass"] is True
    assert fg["int8_mb"] <= 9.80
    assert fg["int8_weight_pct"] >= 88.0
    assert fg["distinct_from_fp32_graph"] is True
    assert fg["stft_lsd_int8_fp32_dB"] <= 0.30
    assert fg["python_js_parity_stft_lsd_dB"] <= 0.05
    assert fg["sim_Anchor"] >= 0.88
    assert fg["margin_Anchor_OldV1"] >= 0.48
    assert fg["HF_LSD_4_12kHz_dB"] <= 1.15
    assert fg["silence_floor_SNR_dB"] >= 58.0
    assert fg["voiced_vowel_hnr_delta_vs_anchor_dB"] <= 1.50
    assert fg["ne_pcer"] <= 0.049
    assert fg["ne_pcer_gap_vs_anchor"] <= 0.017
    assert fg["word_drop_count"] == 0
    assert fg["f0_shutters"] == 0
    assert fg["clicks"] == 0

    e2e = json.load(open(e2e_b_p))
    assert e2e["step2_cfg0"]["warm_median_rtf_1thread"] <= 0.080
    assert e2e["step2_cfg0"]["peak_rss_mb"] > 0

    raw_b = json.load(open(raw_b_p))
    assert "ltas_4band_raw_anchor_dB" in raw_b
    assert "oracle_5rung_ladder_summary" in raw_b

    tiers = json.load(open(tiers_p))["tiers"]
    assert tiers[0]["id"] == "sakhiya_news"
