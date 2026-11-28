#!/usr/bin/env python3
"""EXP6 statistics: first-consonant family of word-initial वि- words per corpus + Fisher exact tests (Rasa vs Nepal-recorded controls)."""
import json, collections, sys
from scipy.stats import fisher_exact
D = sys.argv[1] if len(sys.argv) > 1 else "/work/sonnet/s2/out"
c = collections.defaultdict(collections.Counter)
for corp in ["rasa_F", "rasa_M", "slr43", "slr54"]:
    for l in open(f"{D}/exp6_{corp}.jsonl", encoding="utf8"):
        r = json.loads(l)
        if r["cls"] == "VI": c[corp]["b" if r["first"] in ("b", "β") else ("v" if r["first"] in ("v", "ʋ", "w") else "o")] += 1
out = {"VI_counts": {k: dict(v) | {"n": sum(v.values())} for k, v in c.items()}}
def tab(a, b): return [[c[a]["v"], sum(c[a].values()) - c[a]["v"]], [c[b]["v"], sum(c[b].values()) - c[b]["v"]]]
out["fisher_p_vw_rate"] = {f"{a}_vs_{b}": fisher_exact(tab(a, b))[1] for a, b in [("rasa_F", "slr43"), ("rasa_F", "slr54"), ("rasa_M", "slr43"), ("rasa_M", "slr54"), ("rasa_F", "rasa_M")]}
nep = c["slr43"] + c["slr54"]
out["pooled_nepal_v_rate"] = [round(nep["v"] / sum(nep.values()), 3), sum(nep.values())]
for k in c: out.setdefault("v_rate", {})[k] = round(c[k]["v"] / sum(c[k].values()), 3); out.setdefault("b_rate", {})[k] = round(c[k]["b"] / sum(c[k].values()), 3)
json.dump(out, open(f"{D}/exp6_stats.json", "w"), indent=1); print(json.dumps(out, indent=1))
