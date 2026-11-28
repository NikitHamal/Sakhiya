"""Drop-in progress reporter + detached job launcher for Sakhiya S4 (stdlib only).

In a training script (Gemini):
    sys.path.insert(0, "/work/tts/repo/scripts"); from s4_progress import Progress
    P = Progress("voca_t45_2v", total=120_000, stage="GAN phase")
    ...
    P.update(step, loss=float(l), mel=float(m), lr=lr, note="ckpt saved")   # call every N steps (cheap, rate-limited)
    P.sample("/work/tts/s4/samples/voca_t45_step10k.wav", "step 10k  F  'नमस्ते'")   # shows up as an audio card
    P.done()

Tick a plan gate:  python3 scripts/s4_progress.py gate G2 pass "transplant beat scratch"
Launch detached + register for the cards/guardian (restart=True resumes it if it dies; your script MUST resume from ckpt):
    python3 scripts/s4_progress.py launch NAME "python3 -u scripts/train.py --resume" [--restart] [--tag UNIQUETAG]
"""
import json, os, shutil, subprocess, sys, time

S4 = os.environ.get("S4_DIR", "/work/tts/s4")


def _atomic(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


class Progress:
    def __init__(self, name, total=None, stage="", min_interval=5.0):
        self.name, self.total, self.stage, self.mi = name, total, stage, min_interval
        self.t0, self.last, self.step0 = time.time(), 0.0, None
        self.p = f"{S4}/progress/{name}.json"
        self.m = f"{S4}/progress/{name}.metrics.jsonl"
        self.meta = {}

    def update(self, step, force=False, note=None, **metrics):
        now = time.time()
        if self.step0 is None:
            self.step0 = (step, now)
        if not force and now - self.last < self.mi:
            return
        self.last = now
        s0, t0 = self.step0
        rate = (step - s0) / max(now - t0, 1e-6) if step > s0 else 0.0
        eta = (self.total - step) / rate if (self.total and rate > 0) else None
        d = {"name": self.name, "t": now, "started": self.t0, "step": step, "total": self.total, "stage": self.stage,
             "rate": rate, "eta_s": eta, "metrics": {k: float(v) for k, v in metrics.items()}, "note": note or self.meta.get("note", ""), "done": False}
        self.meta["note"] = d["note"]
        _atomic(self.p, d)
        with open(self.m, "a") as f:
            f.write(json.dumps({"t": now, "step": step, **d["metrics"]}) + "\n")

    def set_stage(self, stage):
        self.stage = stage

    def sample(self, wav_path, label=""):
        os.makedirs(f"{S4}/samples", exist_ok=True)
        import re
        tag = re.sub(r"[^A-Za-z0-9]+", "", (label.split("·")[1] if label.count("·") >= 1 else ""))[:12]   # S4 fix: F/M renders in the same second used to overwrite each other
        dst = f"{S4}/samples/{self.name}__{time.time_ns()}_{tag}.wav"
        if os.path.abspath(wav_path) != os.path.abspath(dst):
            shutil.copy(wav_path, dst)
        with open(f"{S4}/samples/index.jsonl", "a") as f:
            f.write(json.dumps({"t": time.time(), "job": self.name, "file": os.path.basename(dst), "label": label}) + "\n")

    def done(self, note="finished"):
        d = json.load(open(self.p)) if os.path.exists(self.p) else {"name": self.name}
        d.update(t=time.time(), done=True, note=note)
        _atomic(self.p, d)


def launch(name, cmd, restart=False, tag=None, cwd="/work/tts/repo", max_restarts=5):
    tag = tag or f"s4job_{name}"
    os.makedirs("/work/tts/logs", exist_ok=True)
    log = f"/work/tts/logs/{name}.log"
    full = f"cd {cwd} && S4_JOB_TAG={tag} setsid nohup bash -c 'exec -a {tag} {cmd}' >> {log} 2>&1 < /dev/null & echo $!"
    # S4 fix (Sonnet): the old `subprocess.run(full, capture_output=True)` blocked until the JOB ended, because the backgrounded
    # `cd && setsid ... &` list kept the capture pipe open (launch hung -> kernel retried -> duplicate jobs). Use Popen + own session.
    env = dict(os.environ, S4_JOB_TAG=tag)
    lf = open(log, "ab")
    proc = subprocess.Popen(["bash", "-c", f"exec -a {tag} {cmd}"], cwd=cwd, env=env, stdout=lf, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    pid = proc.pid
    jp = f"{S4}/jobs.json"
    jobs = json.load(open(jp)) if os.path.exists(jp) else {"jobs": []}
    jobs["jobs"] = [j for j in jobs["jobs"] if j["name"] != name]
    jobs["jobs"].append({"name": name, "cmd": full, "pid": pid, "tag": tag, "restart": restart, "max_restarts": max_restarts,
                         "stale_s": 150, "cwd": cwd, "log": log, "t": time.time(), "state": "running"})
    _atomic(jp, jobs)
    print(f"launched {name} pid={pid} log={log}")


def gate(gid, state, note=""):
    """state in todo|doing|pass|fail  (shown on the Plan-gates card)"""
    gp = f"{S4}/gates.json"
    g = json.load(open(gp)) if os.path.exists(gp) else {}
    g[gid] = {"state": state, "note": note, "t": time.time()}
    _atomic(gp, g)
    print("gate", gid, state)


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) >= 3 and a[0] == "gate":
        gate(a[1], a[2], a[3] if len(a) > 3 else "")
    elif len(a) >= 3 and a[0] == "launch":
        launch(a[1], a[2], restart="--restart" in a, tag=(a[a.index("--tag") + 1] if "--tag" in a else None))
    else:
        print(__doc__)
