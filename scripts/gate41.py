"""Relative RSI gate: candidate must be intelligible AND beat the incumbent on a combined naturalness/intelligibility score for BOTH languages.
usage: gate41.py candidate_metrics.json incumbent_metrics.json export_dir repo_dir"""
import json, shutil, sys, os
cand, inc, exp, repo = sys.argv[1:5]; c = json.load(open(cand)); i = json.load(open(inc))


def pick(m):
    fp = next(s for s in m["systems"] if s["system"] == "Sakhiya-fp16")["sentence"]
    sa = next((v for k, v in m.get("sanskrit", {}).items() if k.startswith("Sakhiya one voice")), None) or next((v for k, v in m.get("sanskrit", {}).items() if "Sakhiya" in k and "female" in k), None)
    return fp, sa


def score(fp, sa): return (fp["mos"] - 8 * fp["pcer"]) + 1.3 * (sa["mos"] - 8 * sa["pcer"])   # Sanskrit weighted up: it is the open gap


cf, cs = pick(c); inf, ins = pick(i)
sc, si = score(cf, cs), score(inf, ins)
ok = cf["pcer"] <= 0.07 and cs["pcer"] <= 0.22 and sc > si + 0.02
print(f"ne pCER {cf['pcer']} MOS {cf['mos']} | sa pCER {cs['pcer']} MOS {cs['mos']} | score {sc:.3f} vs incumbent {si:.3f}", "PROMOTE" if ok else "REVERT", flush=True)
json.dump(c, open(f"{repo}/eval/candidate_wave41.json", "w"), ensure_ascii=False, indent=1)
if ok:
    shutil.rmtree(f"{repo}/docs/models"); shutil.copytree(exp, f"{repo}/docs/models"); shutil.copy(cand, f"{repo}/eval/metrics.json")
else:
    os.system(f"cd {repo} && git checkout -- docs/models docs/samples docs/metrics.json eval/metrics.json 2>/dev/null")
