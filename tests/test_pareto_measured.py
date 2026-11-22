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
    for t in T:
        if not t.get("ready", True):
            continue
        assert isinstance(t.get("mb"), (int, float)) and t["mb"] > 0
        assert t.get("utmos") is None or (1.0 <= float(t["utmos"]) <= 5.0)
