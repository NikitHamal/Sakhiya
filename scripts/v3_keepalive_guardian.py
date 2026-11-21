"""Sakhiya-v3 Non-Duplicating 12-Hour Keep-Alive & Stage-2 Continuation Guardian.
Monitors active GPU processes every 45s:
- NEVER duplicates a running process (checks both `ps -eo cmd` and `progress.json` mtime > 125s).
- Refreshes `/work/tts/heartbeat.json` every 45s so the MoLab container never idles out.
- When `train_v3_am.py` finishes its 4.5h Phase 2B Stage 1 run, automatically resumes Phase 2B Stage 2
  (Pure 2-Voice Native Accent & Cross-Lingual Unification Specialization, `--specialize --lr 4.5e-5 --hours 6.0`).
- When `train_v3_voc.py` finishes its 270m Stage 3B window, automatically resumes Stage 3B (`--lr 3.0e-5 --minutes 360`).
- Ensures `v3_relay_and_gate.py watch` and `v3_milestone_daemon.py loop --limit 100` stay alive for the full 12h box."""
import json, os, subprocess, sys, time

R = "/work/tts"
REPO = "/work/tts/repo"
LOGS = "/work/tts/logs"


def running_cmds():
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,stat,args"], text=True)
        res = []
        for ln in out.splitlines()[1:]:
            parts = ln.strip().split(None, 2)
            if len(parts) == 3 and not parts[1].startswith("Z") and "<defunct>" not in parts[2]:
                res.append((int(parts[0]), parts[2]))
        return res
    except Exception:
        return []


def has_proc(substr):
    return [pid for pid, cmd in running_cmds() if substr in cmd and "v3_keepalive_guardian" not in cmd]


def prog_age(p):
    return (time.time() - os.path.getmtime(p)) if os.path.exists(p) else 999999.0


def spawn(cmd, log_name):
    os.makedirs(LOGS, exist_ok=True)
    full = f"setsid nohup {cmd} >> {LOGS}/{log_name} 2>&1 < /dev/null &"
    subprocess.run(full, shell=True, cwd=REPO)
    print(f"[guardian {time.strftime('%H:%M:%S')}] Spawned: {cmd}", flush=True)


def step_once():
    try:
        json.dump({"t": time.time(), "who": "v3-keepalive-guardian"}, open(f"{R}/heartbeat.json", "w"))
    except Exception:
        pass

    # 1. Keep 40-min GitHub Release checkpoint survival relay alive
    if not has_proc("v3_relay_and_gate.py watch"):
        spawn(f"python3 -u {REPO}/scripts/v3_relay_and_gate.py watch", "v3_relay.log")

    # 2. Keep 40-min 200-sentence milestone evaluator & auto-promoter alive
    if not has_proc("v3_milestone_daemon.py loop"):
        spawn(f"python3 -u {REPO}/scripts/v3_milestone_daemon.py loop --limit 100", "v3_milestone.log")

    # 3. Keep Sakhiya-v3-AM alive (continue into Phase 2B Stage 2 Specialization only after Stage 1 exits cleanly)
    am_pids = has_proc("train_v3_am.py")
    if not am_pids and prog_age(f"{R}/runs/v3_am/progress.json") > 125.0 and os.path.exists(f"{R}/feats/v3_24k/v3_studio_aligned_24k.pt"):
        spawn(
            f"python3 -u {REPO}/scripts/train_v3_am.py --hours 6.0 --lr 4.5e-5 --specialize --out {R}/runs/v3_am",
            "v3_am.log",
        )

    # 4. Keep Sakhiya-v3-Voc Stage 3B alive (continue only after current 270m window exits cleanly)
    voc_pids = has_proc("train_v3_voc.py")
    if not voc_pids and prog_age(f"{R}/runs/v3_voc_3b/progress.json") > 125.0 and prog_age(f"{R}/runs/v3_voc/progress.json") > 125.0:
        spawn(
            f"python3 -u {REPO}/scripts/train_v3_voc.py --init {R}/runs/v3_voc_3b/G_best.pt --gta {R}/feats/v3_gta_24k.pt --gta_p 0.75 --lr 3.0e-5 --minutes 360 --out {R}/runs/v3_voc_3b",
            "v3_voc_3b.log",
        )


def main():
    if len(has_proc("v3_keepalive_guardian.py")) > 1:
        print("v3_keepalive_guardian already running; exiting duplicate.", flush=True)
        return
    print("v3_keepalive_guardian started.", flush=True)
    while True:
        try:
            step_once()
        except Exception as e:
            print("guardian err:", e, flush=True)
        time.sleep(45)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "once":
        step_once()
    else:
        main()
