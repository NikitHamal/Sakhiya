#!/usr/bin/env python3
"""v10 ship: a staged nanov10/atomv10 export replaces docs/models_<tier> ONLY IF, on the real forensic gate scored on the
int8 graphs the website serves, it is not worse than the currently shipped tier AND it is smaller. Then freezes the
sample pack from those int8 graphs, updates tiers.json, commits and pushes.
usage: v10_ship.py --tier nano|atom --stage DIR --gate JSON --tag v10_atom --label 'Sakhiya-V1-Atom (v10)' [--desc ..]"""
import argparse, json, os, shutil, subprocess, sys
R = "/work/tts"; REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GIT = "GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0 git -c user.name=Sakhiya -c user.email=sakhiya@example.com"
DIR = {"nano": "models_voca", "atom": "models_sano"}


def sh(c): return subprocess.run(c, shell=True, cwd=REPO, capture_output=True, text=True)
def score(f): return (f.get("utmos_15") or 0) - 0.5 * (f.get("word_drop_count") or 0) - 0.3 * (f.get("stutter_count") or 0) - 0.02 * (f.get("clicks") or 0) / 10.0
def int8_mb(d): return round(sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d) if f.endswith(".int8.onnx")) / 1e6, 3)


def main():
    ap = argparse.ArgumentParser()
    for k in ("tier", "stage", "gate", "tag", "label"): ap.add_argument("--" + k, required=True)
    ap.add_argument("--desc", default=None); ap.add_argument("--tol", type=float, default=0.02)
    a = ap.parse_args()
    g = json.load(open(a.gate))
    bp = f"{REPO}/eval/forensic_v9_{a.tier}.json"
    if not os.path.exists(bp): bp = f"{REPO}/eval/forensic_v8_{a.tier}.json"
    base = json.load(open(bp)); d = f"{REPO}/docs/{DIR[a.tier]}"
    old_mb, new_mb = int8_mb(d), int8_mb(a.stage)
    rec = dict(tier=a.tier, tag=a.tag, new_score=round(score(g), 3), base_score=round(score(base), 3), base_file=os.path.basename(bp),
               new_utmos=g.get("utmos_15"), base_utmos=base.get("utmos_15"), new_mb=new_mb, old_mb=old_mb,
               new_drops=g.get("word_drop_count"), new_stutters=g.get("stutter_count"), new_clicks=g.get("clicks"))
    ok = rec["new_score"] >= rec["base_score"] - a.tol and new_mb < old_mb \
        and (g.get("stutter_count") or 0) <= (base.get("stutter_count") or 0) and (g.get("word_drop_count") or 0) <= (base.get("word_drop_count") or 0)
    rec["shipped"] = bool(ok); os.makedirs(f"{R}/v10", exist_ok=True)
    with open(f"{R}/v10/decisions.jsonl", "a") as f: f.write(json.dumps(rec) + "\n")
    print("DECISION", json.dumps(rec), flush=True)
    if not ok: sys.exit(f"NOT SHIPPED {a.tag}: gate or size condition not met")
    shutil.rmtree(d, ignore_errors=True); shutil.copytree(a.stage, d)
    for f in os.listdir(d):
        if f.endswith(".onnx") and not f.endswith(".int8.onnx"): os.remove(f"{d}/{f}")
    ej = f"eval/forensic_{a.tag}.json"; shutil.copyfile(a.gate, f"{REPO}/{ej}")
    r = sh(f"{sys.executable} scripts/freeze_version_pack.py --tag {a.tag} --label '{a.label}' --models docs/{DIR[a.tier]} --metrics-json {ej} --tiny "
           f"--notes 'v10: same acoustic student + shared 1.2M-param vocoder densely distilled from V1; gated on the int8 graphs the site serves'")
    print(r.stdout[-400:], r.stderr[-400:])
    tj = json.load(open(f"{REPO}/docs/tiers.json"))
    for t in tj["tiers"]:
        if t["id"] == a.tier: t.update(mb=new_mb, utmos=g.get("utmos_15"), desc=a.desc or t["desc"])
    json.dump(tj, open(f"{REPO}/docs/tiers.json", "w"), indent=1, ensure_ascii=False)
    print(f"SHIPPED {a.tag} {old_mb} MB -> {new_mb} MB int8", flush=True)
    if os.environ.get("V10_NOPUSH"): return
    print(sh(f"{GIT} add docs eval scripts monitor && {GIT} commit -qm 'v10: ship {a.tag} ({old_mb} -> {new_mb} MB, gate {rec['base_score']} -> {rec['new_score']})'").stderr[-300:])
    print(sh(f"{GIT} pull -q --rebase origin main; {GIT} push -q origin HEAD:main").stderr[-300:])


if __name__ == "__main__":
    main()
