"""Live telemetry for the MoLab cards (runs inside the marimo kernel, no daemon needed: snapshot() is called on every 3 s refresh).
Reads only local files + nvidia-smi: GPU, running training processes, loss curves parsed from /work/tts/logs, ASR proof-pack scoreboard,
tier gate state and the coordination file. Strict JSON-safe values only."""
import glob, json, os, re, subprocess, time

ROOT = os.environ.get("SAKHIYA_ROOT", "/work/tts/repo")
LOGS = "/work/tts/logs"
COORD = "/work/tts/COORD_V25.json"
STEP = re.compile(r"(?:^|\s)step\s+(\d+)(?:\s*/\s*(\d+))?.*?\bloss[=\s:]+(-?\d+(?:\.\d+)?)", re.I)
TAG = re.compile(r"^\[([^\]]+)\]")
TRAIN_HINT = re.compile(r"train|distill|v2_am|v2_dump|polish|gta|nano|atom|unify", re.I)


def _read_json(p, default=None):
    try:
        with open(p) as f: return json.load(f)
    except Exception:
        return default


def gpu():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip().split(",")
        n, u, mu, mt, t, p = [x.strip() for x in o]
        return {"name": n, "util": float(u), "mem_used": float(mu), "mem_total": float(mt), "temp": float(t), "power": float(p) if p.replace(".", "").isdigit() else None}
    except Exception:
        return {}


def procs():
    out = []
    try:
        for line in subprocess.run(["ps", "-eo", "pid,etimes,args"], capture_output=True, text=True, timeout=5).stdout.splitlines()[1:]:
            pid, et, args = line.strip().split(None, 2)
            if "python" in args and TRAIN_HINT.search(args) and "marimo" not in args and "defunct" not in args and "live.py" not in args:
                out.append({"pid": int(pid), "elapsed": int(et), "cmd": args[:150]})
    except Exception:
        pass
    return out


def curves(max_logs=6, max_pts=160, window_h=48):
    now, res = time.time(), []
    for p in sorted(glob.glob(LOGS + "/*.log"), key=os.path.getmtime, reverse=True):
        if now - os.path.getmtime(p) > window_h * 3600: continue
        try:
            with open(p, "rb") as f:
                f.seek(0, 2); n = f.tell(); f.seek(max(0, n - 400_000)); txt = f.read().decode("utf8", "ignore")
        except Exception:
            continue
        series = {}
        for ln in txt.splitlines():
            m = STEP.search(ln)
            if not m: continue
            t = TAG.match(ln.strip()); key = t.group(1) if t else os.path.basename(p)[:-4]
            series.setdefault(key, []).append((int(m.group(1)), float(m.group(3)), int(m.group(2)) if m.group(2) else None))
        for k, v in series.items():
            if len(v) < 2: continue
            stride = max(1, len(v) // max_pts); vv = v[::stride] + ([v[-1]] if v[-1] not in v[::stride] else [])
            res.append({"name": k, "log": os.path.basename(p), "step": v[-1][0], "total": v[-1][2], "loss": v[-1][1], "x": [a for a, _, _ in vv], "y": [b for _, b, _ in vv],
                        "age": now - os.path.getmtime(p)})
        if len(res) >= max_logs: break
    return res[:max_logs]


def proofpack():
    rows = []
    for p in sorted(glob.glob(ROOT + "/eval/asr_*proofpack*.json")):
        d = _read_json(p, {})
        if not isinstance(d, dict): continue
        items = {k: v for k, v in d.items() if isinstance(v, dict) and "hit" in v}
        if not items: continue
        hit = sum(len(v.get("hit", [])) for v in items.values()); miss = sum(len(v.get("miss", [])) for v in items.values())
        rows.append({"name": os.path.basename(p)[:-5].replace("asr_", "").replace("_proofpack", "").replace("proofpack_", ""), "hit": hit, "miss": miss,
                     "items": {k: {"hit": v.get("hit", []), "miss": v.get("miss", [])} for k, v in items.items()}, "mtime": os.path.getmtime(p)})
    return rows


def snapshot():
    return {"t": time.time(), "gpu": gpu(), "procs": procs(), "curves": curves(), "proof": proofpack(),
            "tiers": (_read_json(ROOT + "/docs/tiers.json", {}) or {}).get("tiers", []), "coord": _read_json(COORD, {}) or {},
            "heartbeat": (_read_json("/work/tts/heartbeat.json", {}) or {}).get("t")}
