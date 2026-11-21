#!/usr/bin/env python3
"""v7 supervisor: every --every minutes export the newest v7 student checkpoints (Voca, Sano) to a *staging* dir, gate them with
forensic_audio_gate, and ONLY if the measured score beats the currently shipped one, replace docs/models_{voca,sano}, freeze an
immutable v7 version pack, refresh docs/pareto.json + tiers.json, archive checkpoints to GitHub Release v6-ckpt and push Pages.
Never touches the training processes. Resumable: state lives in /work/tts/v7/ship_state.json."""
import argparse, json, os, shutil, subprocess, sys, time
R = "/work/tts"; REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from sakhiya.v6_pipeline import archive_v6_release  # noqa: E402
ST = f"{R}/v7/ship_state.json"
TIERS = [("voca", "docs/models_voca", "Voca (4.4 MB int8)", "eval/forensic_v6_voca_4m.json"),
         ("sano", "docs/models_sano", "Sano (1.8 MB int8)", "eval/forensic_v6_sano_1p5m.json")]
GIT = "GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0 git"

def score(f): return (f.get("utmos_15") or 0) - 0.5 * (f.get("word_drop_count") or 0) - 0.3 * (f.get("stutter_count") or 0)
def sh(c, check=False): return subprocess.run(c, shell=True, cwd=REPO, check=check, capture_output=True, text=True)
def rj(p, d=None):
    try: return json.load(open(p))
    except Exception: return d

def cycle(n):
    state = rj(ST, {}) ; shipped = False
    for tier, mdir, lbl, base_f in TIERS:
        am, voc = f"{R}/runs/v7_{tier}_am/am.pt", f"{R}/runs/v7_{tier}_voc/voc.pt"
        if not (os.path.exists(am) and os.path.exists(voc)): continue
        stg = f"{R}/v7/stage_{tier}"; shutil.rmtree(stg, ignore_errors=True)
        cur = state.get(tier, {}).get("score")
        if cur is None: cur = score(rj(f"{REPO}/{base_f}", {}) or {})
        a, v = f"{R}/v7/snap_{tier}_am.pt", f"{R}/v7/snap_{tier}_voc.pt"
        shutil.copyfile(am, a); shutil.copyfile(voc, v)  # snapshot so a mid-export torch.save can't tear the files
        r = sh(f"{sys.executable} scripts/export_v2_onnx.py --tier {tier} --am {a} --voc {v} --out {stg}")
        if r.returncode: print(tier, "export failed", r.stderr[-400:], flush=True); continue
        fj = f"{REPO}/eval/forensic_v7_{tier}.json"; tmp = f"{R}/v7/gate_{tier}.json"
        r = sh(f"{sys.executable} scripts/forensic_audio_gate.py --model {stg} --limit 30 --out {tmp}")
        g = rj(tmp, {}) or {}
        sc = score(g); print(f"[cycle {n}] {tier} score={sc:.3f} (utmos {g.get('utmos_15')} drops {g.get('word_drop_count')}) vs shipped {cur:.3f}", flush=True)
        if g.get("utmos_15") and sc > cur + 0.03:
            shutil.rmtree(f"{REPO}/{mdir}", ignore_errors=True); shutil.copytree(stg, f"{REPO}/{mdir}"); shutil.copyfile(tmp, fj)
            tag = f"v7_{tier}_c{n}"
            sh(f"{sys.executable} scripts/freeze_version_pack.py --tag {tag} --label 'v7 {lbl} cycle {n}' --models {mdir} --metrics-json {fj} --notes 'v7 long-run student, measured UTMOS {g.get('utmos_15')}'")
            state[tier] = dict(score=sc, utmos=g.get("utmos_15"), when=time.strftime("%H:%MZ", time.gmtime()), tag=tag); shipped = True
            json.dump(state, open(ST, "w"))
        archive_v6_release({f"v7_{tier}_am.pt": a, f"v7_{tier}_voc.pt": v, "v7_state.json": ST})
    if shipped:
        sh(f"{sys.executable} scripts/build_pareto.py")
        sh(f"{GIT} add docs eval; git commit -qm 'v7: ship improved student (cycle {n})'")
        out = sh(f"{GIT} pull -q --rebase origin main; {GIT} push -q origin HEAD:main")
        print("push:", (out.stderr or out.stdout)[-200:], flush=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--every", type=float, default=50); ap.add_argument("--first", type=float, default=45)
    a = ap.parse_args(); os.makedirs(f"{R}/v7", exist_ok=True)
    time.sleep(a.first * 60); n = len([k for k in rj(ST, {})]) + 1
    while True:
        try: cycle(n)
        except Exception as e: print("cycle error", repr(e), flush=True)
        n += 1; time.sleep(a.every * 60)

if __name__ == "__main__":
    main()
