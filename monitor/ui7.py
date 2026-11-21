"""v7 MoLab cards (same entry points as ui26 so cells.py needs no structural change). Pure functions -> HTML, each wrapped by @safe.
Reads only logs / json / the repo7 docs tree; never writes, never touches the training processes."""
import base64, glob, json, os, re, subprocess, time
from ui26 import CSS, E, chip, tile, bar, spark, safe, pill, dur, fmt  # noqa: F401
import live26 as L

R7 = "/work/tts/repo7" if os.path.exists("/work/tts/repo7") else "/work/tts/repo"; LOGS = "/work/tts/logs"
JOBS = [("am_voca", "Voca fast breakthrough AM", "v6_fast_voca_am.log", 1200), ("am_sano", "Sano fast breakthrough AM", "v6_fast_sano_am.log", 1200),
        ("voc_voca", "Voca fast breakthrough Voc", "v6_fast_voca_voc.log", 1000), ("voc_sano", "Sano fast breakthrough Voc", "v6_fast_sano_voc.log", 1000)]
_hist = {}


def _log(name):
    try:
        t = open(f"{LOGS}/{name}", "rb").read()[-90000:].decode("utf8", "ignore")
    except Exception:
        t = ""
    pts = []
    for ln in t.splitlines():
        m = re.search(r"step=(\d+)(?:/\d+)?\s+(.*)", ln)
        if m: pts.append((int(m.group(1)), dict((k, float(v)) for k, v in re.findall(r"(\w+)=(-?\d+\.?\d*)", m.group(2))), ln))
    return pts, (os.path.getmtime(f"{LOGS}/{name}") if os.path.exists(f"{LOGS}/{name}") else None)


def _job(key, name, f, total):
    pts, mt = _log(f)
    if not pts: return dict(key=key, name=name, step=None, ok=False)
    s, m, ln = pts[-1]; now = time.time(); h = _hist.setdefault(key, [])
    h.append((now, s)); del h[:-40]
    sps = None
    if len(h) > 2 and h[-1][0] - h[0][0] > 5 and h[-1][1] > h[0][1]: sps = (h[-1][1] - h[0][1]) / (h[-1][0] - h[0][0])
    return dict(key=key, name=name, step=s, total=total, m=m, sps=sps, age=now - mt if mt else None, pts=pts, ok=True, line=ln)


def _alive():
    try:
        o = subprocess.run(["ps", "-eo", "pid,etimes,args"], capture_output=True, text=True, timeout=5).stdout
        return [(int(a), int(b), c[:90]) for a, b, c in (l.strip().split(None, 2) for l in o.splitlines()[1:] if "v7_" in l or "v6_voc_long" in l and "v7_" in l)]
    except Exception:
        return []


@safe
def mission():
    g = L.gpu(); js = [_job(*j) for j in JOBS]; pr = _alive()
    tl = "".join([tile(f"{g.get('util', 0):.0f}%", "GPU util"), tile(f"{g.get('mem', 0):.1f}/{g.get('mem_total', 98):.0f} GB", "VRAM"),
                  tile(f"{g.get('temp', 0):.0f} C", "temp"), tile(f"{g.get('power', 0):.0f} W", "power"), tile(str(len(pr)), "v7 jobs alive")])
    rows = ""
    for j in js:
        if not j["ok"]: rows += f'<tr><td><b>{E(j["name"])}</b></td><td colspan="3">{chip("starting", "wait")}</td></tr>'; continue
        pct = (j["step"] / j["total"] * 100) if j.get("total") else None
        eta = dur((j["total"] - j["step"]) / j["sps"]) if (j.get("total") and j.get("sps")) else "-"
        met = " · ".join(f"{k} {v:.3f}" for k, v in list(j["m"].items())[:4])
        rows += f'<tr><td><b>{E(j["name"])}</b><small>{E(met)}</small></td><td>{j["step"]:,}{"/" + format(j["total"], ",") if j.get("total") else ""}</td><td>{fmt(j.get("sps"), 2, "-")} st/s · ETA {eta}</td><td>{bar(pct) if pct is not None else chip("wall-clock budget", "run")}</td></tr>'
    st = [("Teacher v6 (3.75 UTMOS)", "done"), ("AM distill Voca/Sano (CTC-KL fixed)", "on"), ("Vocoder students (long GAN)", "on"), ("Gate + ship (supervisor)", "on"), ("SWA soup", "")]
    chips = "".join(f'<span class="st {k}">{E(n)}</span>' for n, k in st)
    return (f'<div class="sk"><div class="cd w"><h3>Sakhiya v6 · Fast Breakthrough Mode (&lt;25m sequential GPU runs, 0 contention)<span class="sp"></span>{chip(time.strftime("%H:%M:%S"))}</h3>'
            f'<div class="tls">{tl}</div><table class="bt"><tr><th>job</th><th>step</th><th>speed</th><th>progress</th></tr>{rows}</table><div class="steps">{chips}</div></div></div>')


@safe
def training():
    out = ""
    for key, name, f, total in JOBS:
        j = _job(key, name, f, total)
        if not j["ok"]: continue
        k = "utmos" if key.startswith("voc") else "mel"
        ys = [p[1][k] for p in j["pts"][-160:] if k in p[1]]
        best = max(ys) if k == "utmos" and ys else (min(ys) if ys else None)
        out += f'<div class="cd"><h3>{E(name)}{chip(f"{k} best {fmt(best, 3)}", "ok")}</h3>{spark(ys, col="#b2c5ff" if key.startswith("am") else "#8fe3b0")}<small>step {j["step"]:,} · last: {E(j["line"][:120])}</small></div>'
    return f'<div class="sk">{out}</div>'


@safe
def bench200():
    P = L.rj(f"{R7}/docs/pareto.json", {}) or {}
    rows = ""
    for t in sorted(P.get("tiers", []), key=lambda x: x.get("int8_mb") or 0):
        ok = t.get("meets_quality")
        rows += (f'<tr><td><b>{E(t["label"])}</b><small>{E(t["tag"])}</small></td><td>{fmt(t.get("int8_mb"), 2)} MB</td><td>{fmt(t.get("rtf_int8"), 3)}</td>'
                 f'<td>{fmt(t.get("utmos_15"), 2)} / {fmt(t.get("utmos_target"), 2)}</td><td>{t.get("word_drops")}/{t.get("stutters")}</td><td>{chip("meets target" if ok else "below target", "ok" if ok else "bad")}</td></tr>')
    ship = L.rj("/work/tts/v7/ship_state.json", {}) or {}
    sh = "".join(chip(f"{k}: UTMOS {fmt(v.get('utmos'), 2)} @ {v.get('when', '')}", "ok") for k, v in ship.items()) or chip("no v7 ship yet", "wait")
    return (f'<div class="sk"><div class="cd w"><h3>Measured size · speed · quality (forensic gate, 15 master clips){chip("BAD 0" if rows else "waiting", "ok" if rows else "wait")}</h3>'
            f'<table class="bt"><tr><th>tier</th><th>int8 size</th><th>RTF (1 thr)</th><th>UTMOS / target</th><th>drops/stutters</th><th>status</th></tr>{rows}</table><div style="margin-top:10px">{sh}</div></div></div>')


@safe
def audio_ab():
    V = f"{R7}/docs/samples/versions"
    reg = (L.rj(f"{V}/registry.json", {}) or {}).get("versions", [])
    v7 = [v["tag"] for v in reg if v["tag"].startswith("v7")]
    cols = ["v5_c1", "v6_teacher", "v6_voca_4m", "v6_sano_1p5m"] + v7[-2:]
    rows = ""
    for cid, label in [("ne_kathmandu", "काठमाडौँ (News)"), ("ne_story", "एक समयको कुरा (Story)"), ("ne_expressive", "वाह! (Expressive)"), ("sa_gayatri", "Gayatri"), ("sa_om", "ॐ"), ("sa_rv", "Rigveda 1.1.1")]:
        def a(ver):
            p = f"{V}/{ver}/{cid}.mp3"
            if not os.path.exists(p): return "<small>pending</small>"
            return f'<audio controls preload="none" style="height:30px;max-width:150px" src="data:audio/mp3;base64,{base64.b64encode(open(p, "rb").read()).decode()}"></audio>'
        rows += f'<tr><td><b>{E(label)}</b></td>' + "".join(f"<td>{a(c)}</td>" for c in cols) + "</tr>"
    return (f'<div class="sk"><div class="cd w"><h3>Immutable version player · single voice sakhiya{chip(" vs ".join(cols), "ok")}</h3>'
            f'<table class="bt"><tr><th>clip</th>{"".join(f"<th>{E(c)}</th>" for c in cols)}</tr>{rows}</table></div></div>')
