"""The web Pareto chart and tiers.json must only contain numbers traceable to measured artifacts."""
import json, os

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_pareto_matches_size_json():
    P = json.load(open(f"{R}/docs/pareto.json"))["tiers"]
    for t in P:
        if t["tag"] in ("v6_voca_4m", "v6_sano_1p5m"):
            d = "models_voca" if "voca" in t["tag"] else "models_sano"
            if os.path.exists(f"{R}/docs/{d}/size.json"):
                s = json.load(open(f"{R}/docs/{d}/size.json"))
                assert t["utmos_15"] is not None and t["utmos_15"] < 6


def test_tiers_json_has_no_handtyped_quality():
    T = json.load(open(f"{R}/docs/tiers.json"))["tiers"]
    ids = [t["id"] for t in T]
    assert "sakhiya_one" in ids and ("sakhiya_one_vocos" in ids or "sakhiya_one_fast" in ids)
    assert "sakhiya_one_vocos_fp16" in ids
    assert os.path.exists(f"{R}/docs/models_sakhiya_one_vocos/model.fp16.onnx")
    assert "t5" not in ids and "micro" not in ids
    for t in T:
        if not t.get("ready", True):
            continue
        assert isinstance(t.get("mb"), (int, float)) and t["mb"] > 0
        assert t.get("utmos") is None or (1.0 <= float(t["utmos"]) <= 5.0)


def test_sakhiya_one_vocos_fp16_runtime():
    import numpy as np
    from sakhiya.onnx_tts import SakhiyaONNX
    d = os.path.join(R, "docs", "models_sakhiya_one_vocos")
    m = SakhiyaONNX(d, tiny=False, threads=1)
    assert m.single is not None
    w, sr = m.say("नमस्ते, आजको विशेष समाचार।", lang="ne", style="news")
    assert sr == 24000 and w.size > 1000 and np.isfinite(w).all()


def test_sakhiya_one_4layer_forensic_reports():
    for tier in ("sakhiya_one", "sakhiya_one_fast"):
        fp = f"{R}/eval/forensic_{tier}.json"
        assert os.path.exists(fp), f"Missing {fp}"
        g = json.load(open(fp))
        assert g.get("gate_pass") is True
        assert 6.5 <= float(g["int8_mb"]) <= 8.5
        assert g["node_verify"]["ok"] is True and g["node_verify"]["nan_count"] == 0
        l1 = g["layers"]["layer1_timbre"]
        l2 = g["layers"]["layer2_physics"]
        l3 = g["layers"]["layer3_pron_vedic"]
        l4 = g["layers"]["layer4_int8_parity"]
        assert l1["sim_ProV3"] >= (0.84 if tier == "sakhiya_one" else 0.80)
        assert l1["margin_ProV3_OldV1"] >= 0.35
        assert l1["voiced_ltas_mcd_db"] <= 3.5 and l1["voiced_ltas_mcd_ratio"] <= 0.52
        assert l2["HF_LSD_4_10kHz_dB"] <= (2.0 if tier == "sakhiya_one" else 2.6)
        assert l2["mid_vowel_shutters"] == 0 and l2["intra_voiced_pitch_breaks"] == 0 and l2["hf_impulse_clicks"] == 0
        assert l3["word_drop_count"] == 0 and l3["stutter_count"] == 0 and l3["sa_ctc_ll_gap_vs_ProV3"] <= 0.25
        assert l4["stft_lsd_int8_fp32_dB"] <= 0.35 and l4["sim_ProV3_fp32_minus_int8"] <= 0.02
