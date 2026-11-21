#!/usr/bin/env python3
"""v9 one-shot ship: export (ONNX) -> real forensic gate -> replace docs/models_<tier> ONLY IF it beats the shipped v7 score
-> freeze a sample pack rendered from the int8 graphs the site serves -> update tiers.json -> commit + push.
usage: v9_ship.py --tier voca|sano --am am.pt --voc voc.pt --tag v8_nano --label 'Sakhiya-V1-Nano (v8)'"""
import argparse, json, os, shutil, subprocess, sys
R = "/work/tts"; REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GIT = "GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0 git"
TIER_ID = {"voca": "nano", "sano": "atom"}
def sh(c): return subprocess.run(c, shell=True, cwd=REPO, capture_output=True, text=True)
def score(f): return (f.get("utmos_15") or 0) - 0.5 * (f.get("word_drop_count") or 0) - 0.3 * (f.get("stutter_count") or 0) - 0.02 * (f.get("clicks") or 0) / 10.0
def main():
    ap = argparse.ArgumentParser()
    for k in ("tier", "am", "voc", "tag", "label"): ap.add_argument("--" + k, required=True)
    ap.add_argument("--force", action="store_true"); ap.add_argument("--target", default=None); ap.add_argument("--desc", default=None); a = ap.parse_args(); a.target = a.target or a.tier
    EXP = {"voca": "nanoplus", "sano": "atomplus"}[a.tier]; stg = f"{R}/v9/stage_{a.tier}"; os.makedirs(f"{R}/v9", exist_ok=True); shutil.rmtree(stg, ignore_errors=True)
    r = sh(f"{sys.executable} scripts/export_v2_onnx.py --tier {EXP} --am {a.am} --voc {a.voc} --out {stg}"); print(r.stdout[-600:], r.stderr[-600:])
    if r.returncode: sys.exit("export failed")
    gj = f"{R}/v9/gate_{a.tier}.json"
    r = sh(f"{sys.executable} scripts/forensic_audio_gate.py --model {stg} --limit 30 --out {gj}"); print(r.stdout[-500:])
    g = json.load(open(gj)); base = json.load(open(f"{REPO}/eval/forensic_v8_{TIER_ID[a.target]}.json"))
    print(f"NEW score={score(g):.3f} utmos={g.get('utmos_15')} drops={g.get('word_drop_count')} stutters={g.get('stutter_count')} clicks={g.get('clicks')} | v8 score={score(base):.3f} utmos={base.get('utmos_15')}", flush=True)
    if score(g) <= score(base) + 0.05 and not a.force: sys.exit("NOT SHIPPED: did not beat v8 on the real gate")
    d = f"docs/models_{a.target}"; shutil.rmtree(f"{REPO}/{d}", ignore_errors=True); shutil.copytree(stg, f"{REPO}/{d}")
    for f in os.listdir(f"{REPO}/{d}"):
        if f.endswith(".onnx") and not f.endswith(".int8.onnx"): os.remove(f"{REPO}/{d}/{f}")   # fp graphs live in the release, not the repo
    ej = f"eval/forensic_{a.tag}.json"; shutil.copyfile(gj, f"{REPO}/{ej}")
    # int8 is what the site serves: render the frozen samples from it
    sh(f"{sys.executable} scripts/freeze_version_pack.py --tag {a.tag} --label '{a.label}' --models {d} --metrics-json {ej} --tiny --notes 'v9: cached-teacher distillation, fully trained acoustic student + V1 vocoder; measured on the real forensic gate'")
    mb = round(sum(os.path.getsize(f"{REPO}/{d}/{f}") for f in os.listdir(f"{REPO}/{d}") if f.endswith(".int8.onnx")) / 1e6, 3)
    tj = json.load(open(f"{REPO}/docs/tiers.json"))
    for t in tj["tiers"]:
        if t["id"] == TIER_ID[a.target]:
            t.update(mb=mb, utmos=g.get("utmos_15"), desc=a.desc or t["desc"])
    json.dump(tj, open(f"{REPO}/docs/tiers.json", "w"), indent=1, ensure_ascii=False)
    print(f"SHIPPED {a.tag} {mb} MB int8", flush=True)
    if os.environ.get("V8_NOPUSH"): return
    sh(f"git add docs eval scripts && git commit -qm 'v9: ship {a.tag} (UTMOS {g.get('utmos_15')} vs v8 {base.get('utmos_15')})'")
    print(sh(f"{GIT} pull -q --rebase origin main; {GIT} push -q origin HEAD:main").stderr[-300:])
if __name__ == "__main__": main()
