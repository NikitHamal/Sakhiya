"""Sakhiya v2 live telemetry (Track B): shards progress, v2_am step/loss, v2_voc step/loss, GPU util/VRAM, COORD roles.
Called by scripts/status_daemon.build() every 3 s -> docs/data/status.json["v2"]; rendered by the `tts_v2` marimo card."""
import glob, json, os, re, subprocess, time
R = "/work/tts"; L = f"{R}/logs"


def _lines(p):
    try:
        return open(p, errors="ignore").read().splitlines()
    except Exception:
        return []


def _jl(p):
    try:
        return json.load(open(p))
    except Exception:
        return {}


def _count(pat):
    return len([f for f in glob.glob(pat) if not f.endswith(".tmp")])


def _running(key):
    try:
        o = subprocess.run(f"pgrep -af '{key}'", shell=True, capture_output=True, text=True, timeout=4).stdout
        return any("pgrep" not in l and "bash -c" not in l for l in o.splitlines())
    except Exception:
        return False


def _gpu_apps():
    try:
        return subprocess.run("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits",
                              shell=True, capture_output=True, text=True, timeout=4).stdout.strip()
    except Exception:
        return ""


def _series_am(path):
    xs, ys, last = [], [], {}
    for ln in _lines(path):
        m = re.match(r"^step (\d+) loss ([\d.]+) (\{.*?\}) lr ([\de.+-]+) vram ([\d.]+)G", ln)
        if m:
            xs.append(int(m[1])); ys.append(float(m[2]))
            try:
                parts = eval(m[3], {"__builtins__": {}})
            except Exception:
                parts = {}
            last = dict(step=int(m[1]), loss=float(m[2]), parts={k: round(float(v), 3) for k, v in parts.items()}, lr=m[4], vram=float(m[5]))
    return dict(x=xs[-300:], y=ys[-300:]), last


def _series_voc(path):
    xs, ys, last, val = [], [], {}, None
    for ln in _lines(path):
        m = re.match(r"^step (\d+) mel ([\d.]+) stft ([\d.]+) f0 ([\d.]+) D ([\d.]+) vram ([\d.]+)G (\d+)s", ln)
        if m:
            xs.append(int(m[1])); ys.append(float(m[2]))
            last = dict(step=int(m[1]), mel=float(m[2]), stft=float(m[3]), f0=float(m[4]), D=float(m[5]), elapsed_s=int(m[7]))
        m = re.match(r"^VAL step (\d+) ema_mel_l1 ([\d.]+)", ln)
        if m:
            val = float(m[2])
    return dict(x=xs[-300:], y=ys[-300:]), last, val


def _newest_log(pats, default):
    cs = [f for pat in pats for f in glob.glob(pat) if os.path.exists(f) and os.path.getsize(f) > 0]
    return max(cs, key=os.path.getmtime) if cs else default


def v2_status():
    now = time.time()
    am_s, am_last = _series_am(f"{L}/v2_am_A.log")
    voc_lp = _newest_log([f"{L}/v2_voc_3a*.log", f"{L}/v2_voc_3b*.log"], f"{L}/v2_voc_3a.log")
    amb_lp = _newest_log([f"{L}/v2_am_[BCD]*.log"], f"{L}/v2_am_B.log")
    vo_s, vo_last, vo_val = _series_voc(voc_lp)
    coord = _jl(f"{R}/COORD.json")
    g = _gpu_apps().split(",")
    gpu = dict(util=float(g[0]), mem_used=float(g[1]), mem_total=float(g[2]), temp=float(g[3]), power=float(g[4])) if len(g) >= 5 else {}
    pj = _jl(f"{R}/runs/v2_voc/progress.json")
    am_log = _lines(f"{L}/v2_am_A.log")
    amb_s, amb_last = _series_am(amb_lp)
    nat_meta = _jl(f"{R}/feats/v2_native_24k.pt.json")
    onnx_sz = _jl(f"{R}/export_v2_full/size.json")
    return dict(
        t=now, gpu=gpu,
        shards=dict(am_done=_count(f"{R}/feats/v2_shards/*.pt"), am_total=130, studio_ready=os.path.exists(f"{R}/feats/v2_studio_24k.pt"),
                    native_ready=os.path.exists(f"{R}/feats/v2_native_24k.pt"), native_n=nat_meta.get("n", 0), native_hours=nat_meta.get("total_hours", 0.0),
                    voc_done=_count(f"{R}/feats/v2_voc_shards/*.pt"), voc_total=131, prep_running=_running("v2_prep_rasa.py") and not os.path.exists(f"{R}/feats/v2_studio_24k.pt"),
                    wavs_running=_running("v2_voc_wavs.py")),
        am=dict(running=_running("v2_am.py --stage A"), series=am_s, last=am_last, tail=am_log[-3:], budget_s=2.5 * 3600),
        am_b=dict(running=_running("v2_am.py --stage B") or _running("v2_am_B") or _running("v2_am_C") or _running("v2_am_D"), series=amb_s, last=amb_last, tail=_lines(amb_lp)[-3:], budget_s=6.0 * 3600),
        voc=dict(running=_running("train_v2_voc.py"), series=vo_s, last=vo_last, val=vo_val, best=pj.get("best_val"),
                 tail=_lines(voc_lp)[-3:], budget_s=240 * 60),
        onnx_v2=onnx_sz,
        coord=dict(data_ready=coord.get("data_ready"), gta_ready=coord.get("gta_ready"),
                   a=(coord.get("track_a") or {}).get("stage"), b=(coord.get("track_b") or {}).get("stage")),
    )
