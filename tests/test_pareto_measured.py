"""The web Pareto chart must only contain numbers traceable to measured artifacts."""
import json, os
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def test_pareto_matches_size_json():
    P = json.load(open(f"{R}/docs/pareto.json"))["tiers"]
    for t in P:
        if t["tag"] in ("v6_voca_4m", "v6_sano_1p5m"):
            d = "models_voca" if "voca" in t["tag"] else "models_sano"
            s = json.load(open(f"{R}/docs/{d}/size.json"))
            assert abs(t["int8_mb"] - s["total_tiny_mb"]) < 1e-6
            assert t["utmos_15"] is not None and t["utmos_15"] < 6

def test_tiers_json_has_no_handtyped_quality():
    T = json.load(open(f"{R}/docs/tiers.json"))["tiers"]
    P = {t["tag"]: t for t in json.load(open(f"{R}/docs/pareto.json"))["tiers"]}
    for t in T:
        if t["id"] in ("v6nano", "v6teacher"):  # new v6 tiers: no quality number may be typed by hand; must be null until measured
            assert t.get("utmos") is None
            continue
        src = P["v6_teacher" if t["id"] in ("int8", "fp16") else ("v6_voca_4m" if t["id"] == "voca" else "v6_sano_1p5m")]
        assert t["utmos"] == src["utmos_15"]
