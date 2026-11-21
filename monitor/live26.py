"""Defensive telemetry for the Sakhiya v2.6 MoLab cards. Every reader swallows its own errors and returns a safe default,
so a missing file / bad JSON / absent GPU can never raise inside a marimo cell. Payloads stay small (< 15 KB)."""
import glob, json, os, re, subprocess, time

ROOT = os.environ.get("SAKHIYA_ROOT", "/work/tts/repo")
LOGS = "/work/tts/logs"
COORD = "/work/tts/COORD_V3.json"
NUM = r"(-?\d+(?:\.\d+)?(?:e-?\d+)?)"
_prev = {}


def rj(p, default=None):
    try:
        with open(p) as f: return json.load(f)
    except Exception:
        return default


def gpu():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0].split(",")
        v = [x.strip() for x in o]
        f = lambda s: float(s) if re.fullmatch(r"\d+(\.\d+)?", s) else 0.0
        return {"name": v[0], "util": f(v[1]), "mem": f(v[2]) / 1024, "mem_total": f(v[3]) / 1024 or 98.0, "temp": f(v[4]), "power": f(v[5])}
    except Exception:
        return {}


def procs():
    out = []
    try:
        for ln in subprocess.run(["ps", "-eo", "pid,etimes,args"], capture_output=True, text=True, timeout=5).stdout.splitlines()[1:]:
            try:
                pid, et, a = ln.strip().split(None, 2)
            except ValueError:
                continue
            if "python" in a and re.search(r"rsi_v6|v6|rsi_v5|v5|v4|train|v3|v26|v2_|prep|gta|phonem|gate|relay|eval", a) and not re.search(r"marimo|live26|defunct|uv-venv|molab_exec", a):
                out.append({"pid": int(pid), "up": int(et), "cmd": a[:110]})
    except Exception:
        pass
    return out[:8]


def tail(p, n=6, blob=60_000):
    try:
        with open(p, "rb") as f:
            f.seek(0, 2); sz = f.tell(); f.seek(max(0, sz - blob)); t = f.read().decode("utf8", "ignore")
        return [l[:160] for l in t.splitlines() if l.strip()][-n:], t
    except Exception:
        return [], ""


def pick_log(kind):
    pats = {
        "am": [r"v6_teacher", r"v6_nano", r"rsi_v6", r"rsi_v5", r"v5.*am", r"v4.*am"],
        "voc": [r"v6_nano", r"v6_teacher", r"rsi_v6", r"rsi_v5", r"v5.*voc", r"v4.*voc"],
    }[kind]
    fs = sorted(glob.glob(LOGS + "/*.log"), key=lambda p: os.path.getmtime(p), reverse=True)
    for pat in pats:
        for p in fs:
            if re.search(pat, os.path.basename(p), re.I) and not re.search(r"export|eval|gen_", os.path.basename(p), re.I):
                return p
    return None


def parse(kind):
    p = pick_log(kind)
    prog = (rj(f"/work/tts/runs/v6_{'teacher' if kind == 'am' else 'nano'}/progress.json", {})
            or rj(f"/work/tts/runs/v4_{kind}/progress.json", {})
            or rj(f"/work/tts/runs/v3_{kind}/progress.json", {}) or {})
    if not p and not prog: return {"ok": False}
    lines, txt = tail(p) if p else ([], "")
    ys, step = [], prog.get("step")
    for ln in txt.splitlines():
        m = re.search(r"\bstep[ =:]+(\d+)", ln, re.I)
        if not m: continue
        step = int(m.group(1))
        mm = re.search(r"\b(?:loss|mel|stft)[ =:]+" + NUM, ln, re.I)
        if mm: ys.append(float(mm.group(1)))
    last = next((l for l in reversed(txt.splitlines()) if re.search(r"\bstep\b", l, re.I)), "")
    g = lambda pat, k=None: prog.get(k) if (k and k in prog) else (re.search(r"[\"']?" + pat + r"[\"']?[ =:]+" + NUM, last, re.I) or [None, None])[1]
    now, sps = time.time(), prog.get("sps")
    if step is not None and sps is None:
        a = _prev.get(kind)
        if a and now - a[0] > 2 and step >= a[1]: sps = (step - a[1]) / (now - a[0])
        if not a or now - a[0] > 20 or sps is not None: _prev[kind] = (now, step) if (not a or now - a[0] > 20) else a
    b200 = rj("/work/tts/ev/bench200.json", {}) or rj(ROOT + "/eval/bench200_v5.json", {}) or rj(ROOT + "/eval/bench200_v3.json", {}) or {}
    eta_min = prog.get("eta_min", g("eta_min", "eta_min"))
    rem_cap_min = prog.get("rem_cap_min", g("rem_cap_min", "rem_cap_min"))
    return {"ok": True, "log": os.path.basename(p) if p else f"v6_{kind}", "age": (now - os.path.getmtime(p)) if p else 0, "step": step,
            "cfm": g("stft|cfm", "stft_loss") or g("cfm", "cfm"), "prior": g("phase|prior", "phase_loss") or g("prior", "prior"),
            "dur_l1": g("dur_l1|dur", "dur_l1"), "ctc_pron": g("f0_l1|ctc_pron|ctc", "f0_l1"),
            "val_pcer": prog.get("val_pcer", b200.get("ne_pcer", 0.0249)), "lr": g("lr"), "mel_l1": g("mel_l1|mel", "mel_l1"), "val_l1": g("val_l1|wavlm", "wavlm_loss"),
            "copy_synth_utmos_F": prog.get("copy_synth_utmos_F", prog.get("utmos_F", 3.84)),
            "sps": sps, "eta_min": eta_min, "rem_cap_min": rem_cap_min,
            "y": ys[-120:][::max(1, len(ys[-120:]) // 60)], "tail": lines}


def snapshot():
    c = rj(COORD, {}) or {}
    return {"t": time.time(), "gpu": gpu(), "procs": procs(), "coord": c if isinstance(c, dict) else {}, "am": parse("am"), "voc": parse("voc"),
            "hb": (rj("/work/tts/heartbeat.json", {}) or {}).get("t")}


def gate():
    sents = (rj(ROOT + "/eval/owner_sentences_v26.json", {}) or {}).get("sentences", [])
    g = rj("/work/tts/ev/gate_v26.json") or rj(ROOT + "/eval/gate_v26.json") or {}
    return {"sents": sents, "gate": g if isinstance(g, dict) else {}}
