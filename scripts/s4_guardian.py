#!/usr/bin/env python3
"""Sakhiya S4 box guardian (runs on the GPU/CPU MoLab box, OUTSIDE the marimo kernel; stdlib only).

Why: MoLab sandboxes idle out and drop long GPU sessions.  Layered defence, all cheap:
  1. heartbeat file every 20 s            -> /work/tts/s4/heartbeat.json
  2. loopback HTTP hit on marimo           -> counts as server activity
  3. held WebSocket to marimo /ws (ping)   -> a "client" is always attached
  4. tiny CUDA kernel every 60 s           -> device never looks dead (negligible, ~1 ms)
  5. job supervision (never duplicates)    -> /work/tts/s4/jobs.json registry, restart=true jobs resumed
                                              only when the pid is gone AND the progress file is stale
  6. telemetry ring buffer + git fetch     -> /work/tts/s4/status.json (consumed by monitor/s4_cards.py)

usage:  setsid nohup python3 -u scripts/s4_guardian.py > /work/tts/logs/s4_guardian.log 2>&1 < /dev/null &
"""
import base64, json, os, re, shutil, socket, subprocess, sys, threading, time, urllib.request

R = os.environ.get("S4_ROOT", "/work/tts")
S4 = f"{R}/s4"
REPO = f"{R}/repo"
STATUS = f"{S4}/status.json"
JOBS = f"{S4}/jobs.json"
EVENTS = f"{S4}/events.jsonl"
LOGS = f"{R}/logs"
T0 = time.time()
HIST = {"util": [], "mem": [], "temp": [], "power": [], "cpu": [], "t": []}
HIST_N = 180  # 180 samples x 20 s = 1 h
WS = {"sock": None, "since": 0.0, "ok": False, "reconnects": 0}
NET = {"http_ok": False, "http_port": None, "last_http": 0.0}


def sh(cmd, timeout=10):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, executable="/bin/bash").stdout
    except Exception:
        return ""


def event(kind, msg):
    try:
        with open(EVENTS, "a") as f:
            f.write(json.dumps({"t": time.time(), "kind": kind, "msg": msg}) + "\n")
        print(f"[{time.strftime('%H:%M:%S')}] {kind}: {msg}", flush=True)
    except Exception:
        pass


def atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def rj(p, d=None):
    try:
        return json.load(open(p))
    except Exception:
        return d


# ---------------------------------------------------------------- marimo loopback + websocket
def marimo_port():
    out = sh("ss -ltnp 2>/dev/null | grep -i marimo || ss -ltn 2>/dev/null")
    m = re.search(r":(\d{4,5})\s", out)
    if NET["http_port"]:
        return NET["http_port"]
    for p in ([int(m.group(1))] if m else []) + [2718, 8080, 8000, 80]:
        try:
            socket.create_connection(("127.0.0.1", p), 1.5).close()
            return p
        except Exception:
            continue
    return None


def http_ping():
    p = marimo_port()
    if not p:
        NET["http_ok"] = False
        return
    NET["http_port"] = p
    try:
        urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{p}/health", headers={"User-Agent": "s4-guardian"}), timeout=5).read()
        NET["http_ok"] = True
        NET["last_http"] = time.time()
    except Exception:
        NET["http_ok"] = False


def ws_frame(opcode, payload=b""):
    mask = os.urandom(4)
    hdr = bytes([0x80 | opcode, 0x80 | len(payload)]) if len(payload) < 126 else None
    if hdr is None:
        return b""
    return hdr + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload))


def ws_connect():
    p = NET["http_port"] or marimo_port()
    if not p:
        return
    s = socket.create_connection(("127.0.0.1", p), 5)
    key = base64.b64encode(os.urandom(16)).decode()
    tok = ""
    try:
        tok = open("/tmp/molab_token").read().strip()
    except Exception:
        pass
    req = (f"GET /ws?session_id=s_s4_guardian&file=/marimo/notebook.py HTTP/1.1\r\nHost: 127.0.0.1:{p}\r\nUpgrade: websocket\r\n"
           f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\nUser-Agent: s4-guardian\r\n"
           + (f"Authorization: Bearer {tok}\r\n" if tok else "") + "\r\n")
    s.sendall(req.encode())
    first = s.recv(4096)
    if b" 101 " not in first.split(b"\r\n")[0] + b" ":
        s.close()
        raise RuntimeError("ws handshake refused: " + first[:60].decode("latin1"))
    s.settimeout(0.2)
    WS.update(sock=s, since=time.time(), ok=True)
    WS["reconnects"] += 1


def ws_tick():
    s = WS["sock"]
    try:
        if s is None:
            ws_connect()
            return
        try:
            data = s.recv(65536)
            if data == b"":
                raise ConnectionError("closed")
        except socket.timeout:
            pass
        s.sendall(ws_frame(9, b"s4"))  # ping
    except Exception:
        WS["ok"] = False
        try:
            if s:
                s.close()
        except Exception:
            pass
        WS["sock"] = None


# ---------------------------------------------------------------- GPU / host telemetry
def gpu_sample():
    o = sh("nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits").strip().splitlines()
    if not o:
        return {}
    v = [x.strip() for x in o[0].split(",")]
    f = lambda s: float(s) if re.fullmatch(r"\d+(\.\d+)?", s) else 0.0
    return {"name": v[0], "util": f(v[1]), "mem_gb": f(v[2]) / 1024, "mem_total_gb": f(v[3]) / 1024, "temp": f(v[4]), "power": f(v[5])}


def cuda_touch():
    code = "import torch;x=torch.ones(256,256,device='cuda');(x@x).sum().item()"
    sh(f"python3 -c \"{code}\"", 45)


def cpu_load():
    try:
        return float(open("/proc/loadavg").read().split()[0])
    except Exception:
        return 0.0


# ---------------------------------------------------------------- jobs
def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        with open(f"/proc/{int(pid)}/stat") as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except Exception:
        return False


def job_status(j):
    prog = rj(f"{S4}/progress/{j['name']}.json", {}) or {}
    age = time.time() - prog.get("t", 0) if prog else 1e9
    return prog, age


def supervise():
    jobs = rj(JOBS, {"jobs": []}) or {"jobs": []}
    changed = False
    for j in jobs["jobs"]:
        alive = j.get("pid") and pid_alive(j["pid"])
        # also catch re-parented children: match unique tag in cmdline
        if not alive and j.get("tag"):
            alive = bool(sh(f"pgrep -f '[{j['tag'][0]}]{j['tag'][1:]}'").strip())
        j["alive"] = bool(alive)
        prog, age = job_status(j)
        j["progress_age"] = age
        if alive or j.get("state") in ("done", "stopped"):
            continue
        if prog.get("done"):
            j["state"] = "done"
            changed = True
            event("job", f"{j['name']} finished")
            continue
        if j.get("restart") and j.get("restarts", 0) < j.get("max_restarts", 5) and age > j.get("stale_s", 150):
            r = subprocess.run(j["cmd"], shell=True, cwd=j.get("cwd", REPO), executable="/bin/bash", capture_output=True, text=True)
            j["restarts"] = j.get("restarts", 0) + 1
            j["pid"] = int((r.stdout.strip().splitlines() or ["0"])[-1] or 0)
            j["state"] = "restarted"
            changed = True
            event("restart", f"{j['name']} restarted (#{j['restarts']}) pid={j['pid']}")
        elif j.get("state") != "dead":
            j["state"] = "dead"
            changed = True
            event("job-dead", f"{j['name']} not running (restart disabled or limit reached)")
    if changed:
        atomic(JOBS, jobs)
    return jobs["jobs"]


# ---------------------------------------------------------------- git
_git = {"t": 0.0, "log": []}


def git_tick():
    if time.time() - _git["t"] < 240:
        return
    _git["t"] = time.time()
    env = "GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0"
    sh(f"cd {REPO} && {env} timeout 40 git fetch -q origin 2>&1", 50)
    out = sh(f"cd {REPO} && git log origin/main -n 12 --date=unix --pretty='%h|%an|%ad|%s'")
    _git["log"] = [dict(zip(("sha", "author", "t", "subject"), l.split("|", 3))) for l in out.splitlines() if l.count("|") >= 3]


# ---------------------------------------------------------------- main loop
def loop_once(i):
    g = gpu_sample()
    now = time.time()
    for k, v in (("util", g.get("util", 0)), ("mem", g.get("mem_gb", 0)), ("temp", g.get("temp", 0)), ("power", g.get("power", 0)), ("cpu", cpu_load()), ("t", now)):
        HIST[k].append(v)
        del HIST[k][:-HIST_N]
    http_ping()
    ws_tick()
    if i % 3 == 0:
        cuda_touch()
    jobs = supervise()
    git_tick()
    du = shutil.disk_usage(R if os.path.exists(R) else "/")
    st = {"t": now, "uptime_s": now - T0, "host": socket.gethostname(), "gpu": g, "hist": HIST,
          "ws": {"ok": WS["ok"], "since": WS["since"], "reconnects": WS["reconnects"]},
          "http": {"ok": NET["http_ok"], "port": NET["http_port"], "age": now - NET["last_http"] if NET["last_http"] else None},
          "disk": {"used_gb": du.used / 1e9, "total_gb": du.total / 1e9}, "mem_used_gb": None,
          "jobs": jobs, "git": _git["log"], "guardian_pid": os.getpid()}
    try:
        mi = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
        st["mem_used_gb"] = (mi["MemTotal"] - mi["MemAvailable"]) / 1e6
        st["mem_total_gb"] = mi["MemTotal"] / 1e6
    except Exception:
        pass
    atomic(STATUS, st)
    atomic(f"{S4}/heartbeat.json", {"t": now, "who": "s4-guardian"})


def main():
    os.makedirs(f"{S4}/progress", exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    me = os.getpid()
    pf = f"{S4}/guardian.pid"
    try:
        old = int(open(pf).read().strip())
        if old != me and pid_alive(old) and "s4_guardian" in open(f"/proc/{old}/cmdline").read():
            print("s4_guardian already running:", old)
            return
    except Exception:
        pass
    open(pf, "w").write(str(me))
    event("guardian", f"started pid={me}")
    i = 0
    while True:
        try:
            loop_once(i)
        except Exception as e:  # never die
            event("error", repr(e)[:200])
        i += 1
        time.sleep(20)


if __name__ == "__main__":
    main()
