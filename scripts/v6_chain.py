#!/usr/bin/env python3
"""Autonomous, wall-clock-capped v6 chain (runs on the MoLab box, detached).

    v6_istft_teacher  ->  real forensic gate  ->  ship ONLY if it passes
    v6_nano           ->  real forensic gate + int8/fp32 quantisation gate  ->  ship ONLY if it passes

Rules enforced here (owner law):
  * every training stage gets --max-minutes <= 150 (teacher 100, nano 100 by default);
  * the training scripts run the step-50 throughput gate themselves; this wrapper
    reads their log and aborts the stage when the projected ETA is > 150 min;
  * nothing is frozen / pushed unless the audio gate passes. A failed gate is recorded
    honestly in eval/ and in runs/v6_chain.json (which the MoLab cards read).
New models are shipped under NEW tags (v6_istft_teacher, v6_nano) and NEW model dirs
(docs/models_v6_teacher, docs/models_v6_nano); the old frozen v6_teacher pack is never touched.
"""
import argparse, json, os, re, shutil, subprocess, sys, time

R = "/work/tts"
REPO = f"{R}/repo"
STATE = f"{R}/runs/v6_chain.json"
ENV = dict(os.environ, GIT_ASKPASS="/tmp/gitenv.sh", GIT_TERMINAL_PROMPT="0")

TEACHER_GATE = dict(word_drop_count=0, stutter_count=0, f0_shutters=0, ne_pcer=0.028, sa_pcer=0.032, kathmandu_nasal=0.5)
NANO_GATE = dict(word_drop_count=0, stutter_count=0, f0_shutters=0, ne_pcer=0.040, sa_pcer=0.060, quant_corr=0.98, int8_mb=3.0)


def save(**kw):
    st = json.load(open(STATE)) if os.path.exists(STATE) else {"stages": {}}
    for k, v in kw.items():
        if k == "stage_update":
            name, upd = v
            st["stages"].setdefault(name, {}).update(upd)
        else:
            st[k] = v
    st["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(st, open(STATE + ".tmp", "w"), indent=1, ensure_ascii=False)
    os.replace(STATE + ".tmp", STATE)


def run_logged(cmd, log, cap_minutes, name, eta_gate=True):
    """Run a command, tee to log, kill if it exceeds the hard cap or the step-50 gate says ETA > 150 min."""
    t0 = time.time()
    save(stage_update=(name, dict(state="running", started=t0, cap_min=cap_minutes, cmd=cmd)))
    with open(log, "w") as lf:
        p = subprocess.Popen(cmd, shell=True, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT, env=ENV, executable="/bin/bash")
        aborted = None
        while p.poll() is None:
            time.sleep(10)
            txt = open(log, errors="ignore").read()[-6000:]
            m = re.findall(r"projected ETA=([0-9.]+) min", txt)
            if eta_gate and m and float(m[-1]) > 150.0:
                aborted = f"step-50 gate: projected ETA {m[-1]} min > 150"
            if time.time() - t0 > (cap_minutes + 25) * 60:
                aborted = f"hard wall-clock cap {cap_minutes}+25 min exceeded"
            if aborted:
                p.terminate()
                time.sleep(5)
                p.kill()
                break
    rc = p.returncode
    save(stage_update=(name, dict(state="aborted" if aborted else ("done" if rc == 0 else "failed"), rc=rc, abort_reason=aborted, minutes=round((time.time() - t0) / 60, 1))))
    return rc == 0 and not aborted


def sh(cmd, check=False):
    r = subprocess.run(cmd, shell=True, cwd=REPO, capture_output=True, text=True, env=ENV, executable="/bin/bash")
    if check and r.returncode:
        raise RuntimeError(r.stderr[-500:])
    return r


def teacher_verdict(js):
    why = []
    for k in ("word_drop_count", "stutter_count", "f0_shutters"):
        if js.get(k, 99) != 0:
            why.append(f"{k}={js.get(k)}")
    for k, lim in (("ne_pcer", TEACHER_GATE["ne_pcer"]), ("sa_pcer", TEACHER_GATE["sa_pcer"])):
        v = js.get(k)
        if v is None:
            why.append(f"{k} missing")
        elif v > lim:
            why.append(f"{k}={v} > {lim}")
    n = js.get("nasal_scores", {}).get("काठमाडौँ", {}).get("ctc_nasal_prob", 0)
    if n < TEACHER_GATE["kathmandu_nasal"]:
        why.append(f"kathmandu nasal prob {n} < 0.5")
    return not why, why


def nano_verdict(js, export_manifest):
    why = []
    for k in ("word_drop_count", "stutter_count", "f0_shutters"):
        if js.get(k, 99) != 0:
            why.append(f"{k}={js.get(k)}")
    for k, lim in (("ne_pcer", NANO_GATE["ne_pcer"]), ("sa_pcer", NANO_GATE["sa_pcer"])):
        v = js.get(k)
        if v is None:
            why.append(f"{k} missing")
        elif v > lim:
            why.append(f"{k}={v} > {lim}")
    if export_manifest.get("quant_corr", 0) < NANO_GATE["quant_corr"]:
        why.append(f"quant_corr {export_manifest.get('quant_corr')} < 0.98")
    if export_manifest.get("int8_mb", 99) > NANO_GATE["int8_mb"]:
        why.append(f"int8_mb {export_manifest.get('int8_mb')} > 3.0")
    return not why, why


def ship(tag, label, export_dir, docs_dir, metrics_json, notes):
    dst = f"{REPO}/{docs_dir}"
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(export_dir, dst)
    r = sh(f"python3 scripts/freeze_version_pack.py --tag {tag} --label '{label}' --models {docs_dir} --metrics-json {metrics_json} --notes '{notes}'")
    ok = r.returncode == 0
    try:  # make the Studio tier selector offer the shipped model
        tp = f"{REPO}/docs/tiers.json"; t = json.load(open(tp)); mf = json.load(open(f"{dst}/manifest.json")) if os.path.exists(f"{dst}/manifest.json") else {}
        tid = "v6nano" if tag == "v6_nano" else "v6teacher"
        ent = next((x for x in t["tiers"] if x["id"] == tid), None)
        if ent is None:
            ent = dict(id=tid, label=label, prec="int8", variant="tiny", rtf=None, utmos=None, ne_pcer=None, sa_pcer=None); t["tiers"].append(ent)
        ent.update(dir=docs_dir.replace("docs/", "") + "/", ready=True, mb=mf.get("int8_mb") or ent.get("mb"), desc=notes); ent.pop("gate", None)
        json.dump(t, open(tp, "w"), ensure_ascii=False, indent=1)
    except Exception as e:
        print("tier registration failed", e)
    sh(f"git add docs/tiers.json {docs_dir} docs/samples/versions eval/{os.path.basename(metrics_json)} && git -c user.email=a@b -c user.name=NikitHamal commit -m 'Ship {tag}: passed real forensic gate' ; "
       f"git pull --rebase --autostash origin main; git push origin HEAD:main")
    return ok, (r.stdout + r.stderr)[-400:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-minutes", type=float, default=100)
    ap.add_argument("--teacher-steps", type=int, default=70000)
    ap.add_argument("--nano-minutes", type=float, default=100)
    ap.add_argument("--nano-am-steps", type=int, default=45000)
    ap.add_argument("--nano-voc-steps", type=int, default=25000)
    ap.add_argument("--skip-teacher", action="store_true")
    a = ap.parse_args()
    assert a.teacher_minutes <= 150 and a.nano_minutes <= 150
    os.makedirs(f"{R}/logs", exist_ok=True)
    save(chain_started=time.time(), plan=["v6_chandas", "v6_teacher", "v6_nano", "v6_studio"], v6_chandas="shipped (see eval/forensic_v6_chandas.json)")

    tdir = f"{R}/export/v6_istft_teacher"
    teacher_ok = False
    if not a.skip_teacher:
        ok = run_logged(f"python3 scripts/train_v6_teacher.py --max-minutes {a.teacher_minutes} --steps {a.teacher_steps} --bs 36 --export-dir {tdir}", f"{R}/logs/v6_teacher_long.log", a.teacher_minutes, "v6_teacher")
        if ok:
            sh(f"python3 scripts/forensic_audio_gate.py --model {tdir} --limit 40 --out eval/forensic_v6_istft_teacher.json")
            js = json.load(open(f"{REPO}/eval/forensic_v6_istft_teacher.json"))
            teacher_ok, why = teacher_verdict(js)
            save(stage_update=("v6_teacher", dict(gate_pass=teacher_ok, gate_fail_reasons=why, gate={k: js.get(k) for k in ("word_drop_count", "stutter_count", "f0_shutters", "ne_pcer", "sa_pcer", "clicks", "utmos_15")})))
            if teacher_ok:
                s_ok, tail = ship("v6_istft_teacher", "v6 Teacher (Single-Stage Direct-iSTFT + Chandas)", tdir, "docs/models_v6_teacher", "eval/forensic_v6_istft_teacher.json", "Passed real forensic gate; chandas wired for Sanskrit clips")
                save(stage_update=("v6_teacher", dict(shipped=s_ok)))
            else:
                sh("git add eval/forensic_v6_istft_teacher.json && git -c user.email=a@b -c user.name=NikitHamal commit -m 'v6_istft_teacher: forensic gate NOT passed (recorded, not shipped)'; git pull --rebase --autostash origin main; git push origin HEAD:main")

    ndir = f"{R}/export/v6_nano"
    tck = f"{R}/runs/v6_istft_teacher/teacher.pt" if teacher_ok else "/nonexistent_teacher.pt"
    ok = run_logged(f"python3 scripts/train_v6_nano.py --max-minutes {a.nano_minutes} --am-steps {a.nano_am_steps} --voc-steps {a.nano_voc_steps} --export-dir {ndir} --teacher-ckpt {tck}", f"{R}/logs/v6_nano_long.log", a.nano_minutes, "v6_nano")
    if ok:
        sh(f"python3 scripts/forensic_audio_gate.py --model {ndir} --limit 40 --out eval/forensic_v6_nano.json")
        js = json.load(open(f"{REPO}/eval/forensic_v6_nano.json"))
        mf = json.load(open(f"{ndir}/manifest.json")) if os.path.exists(f"{ndir}/manifest.json") else {}
        nano_ok, why = nano_verdict(js, mf)
        save(stage_update=("v6_nano", dict(gate_pass=nano_ok, gate_fail_reasons=why, int8_mb=mf.get("int8_mb"), quant_corr=mf.get("quant_corr"), gate={k: js.get(k) for k in ("word_drop_count", "stutter_count", "f0_shutters", "ne_pcer", "sa_pcer", "clicks", "utmos_15")})))
        if nano_ok:
            s_ok, tail = ship("v6_nano", "v6 Sakhiya-Nano (<3 MB int8)", ndir, "docs/models_v6_nano", "eval/forensic_v6_nano.json", "Passed real forensic + int8 quantisation gate")
            save(stage_update=("v6_nano", dict(shipped=s_ok)))
        else:
            sh("git add eval/forensic_v6_nano.json && git -c user.email=a@b -c user.name=NikitHamal commit -m 'v6_nano: forensic gate NOT passed (recorded, not shipped)'; git pull --rebase --autostash origin main; git push origin HEAD:main")
    save(chain_finished=time.time())


if __name__ == "__main__":
    main()
