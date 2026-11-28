"""Sakhiya S4 live cards (Material 3 Expressive look) for the MoLab marimo notebook.
Read-only, stdlib only, every reader swallows its own errors (a card can never raise).
Data comes from /work/tts/s4 (written by scripts/s4_guardian.py + scripts/s4_progress.py)."""
import base64, glob, html, json, os, re, time

R = os.environ.get("S4_ROOT", "/work/tts")
S4 = f"{R}/s4"
LOGS = f"{R}/logs"

DEFAULT_GATES = [
    ("A1", "CPU", "G0 blind listen desk (Rasa-F / Rasa-M / SLR43 / FLEURS)"),
    ("A2", "CPU", "rasa_import: 48→24 kHz, LogSumExp align, sentence-hash split"),
    ("A3", "CPU", "g2p_v4 single-source front-end + 300-word native sheet"),
    ("A4", "CPU", "sakhiya/tiny fork: 32-clip overfit + ONNX parity ≤ 0.6 %"),
    ("A5", "CPU", "eval/frozen_v1 + baselines frozen"),
    ("G1", "GPU", "cache 24 kHz wav+mel+F0+energy+align, loader ≥ 40 it/s"),
    ("G2", "GPU", "3K smoke: transplant beats scratch (audible source-filter)"),
    ("G3", "GPU", "3 runs live (T4.5 2v / T10 2v / T10 2v+styles) past 20K"),
    ("G4", "GPU", "kill losers @40K/80K, blind renders every 10K"),
    ("G5", "GPU", "export INT8/FP16, len_diff=0, RTF ≤ 0.05 (1 thread)"),
    ("G6", "EARS", "CMOS ≥ +0.5 vs V1 & default · accent ≥ 90 % · WER ≤ real+2"),
]

CSS = """<style>
.s4{--p:#3b5ba9;--pc:#dce1ff;--on-pc:#00164e;--sc:#e6e9f4;--ter:#7d4c8e;--surf:#fbf8fd;--sc1:#f0edf1;--sc2:#eae7ec;--tx:#1b1b1f;--mu:#5f5f6a;--ol:#c6c6d0;--ok:#1b6e3a;--okc:#b9f4c4;--er:#ba1a1a;--erc:#ffdad6;--wa:#7a5900;--wac:#ffdf9d;
 font-family:'Google Sans','Roboto Flex','Segoe UI',system-ui,sans-serif;color:var(--tx);font-size:14px;line-height:1.4}
@media (prefers-color-scheme:dark){.s4{--p:#b5c4ff;--pc:#224190;--on-pc:#dce1ff;--sc:#434659;--ter:#eeb4ff;--surf:#121316;--sc1:#1b1b1f;--sc2:#25262a;--tx:#e4e1e6;--mu:#c6c6d0;--ol:#45464f;--ok:#9dd8a8;--okc:#00522a;--er:#ffb4ab;--erc:#93000a;--wa:#f4c14a;--wac:#5a4300}}
.s4 *{box-sizing:border-box}
.s4 .card{background:var(--sc1);border-radius:28px;padding:18px 20px;margin:2px 0}
.s4 .hero{background:var(--pc);color:var(--on-pc);border-radius:40px 40px 40px 12px;padding:20px 24px}
.s4 h2{margin:0;font-size:26px;font-weight:600;letter-spacing:-.01em}
.s4 h3{margin:0 0 10px;font-size:16px;font-weight:600;display:flex;align-items:center;gap:8px}
.s4 .row{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.s4 .chip{display:inline-flex;align-items:center;gap:6px;padding:5px 12px;border-radius:10px;background:var(--sc);font-size:12.5px;font-weight:500}
.s4 .chip.ok{background:var(--okc);color:var(--ok)}.s4 .chip.er{background:var(--erc);color:var(--er)}.s4 .chip.wa{background:var(--wac);color:var(--wa)}
.s4 .dot{width:8px;height:8px;border-radius:50%;background:currentColor;display:inline-block}
.s4 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px}
.s4 .stat{background:var(--sc2);border-radius:20px;padding:12px 14px}
.s4 .stat b{display:block;font-size:24px;font-weight:600;letter-spacing:-.02em}.s4 .stat span{color:var(--mu);font-size:12px}
.s4 .bar{height:14px;border-radius:7px;background:var(--sc);overflow:hidden;position:relative}
.s4 .bar i{position:absolute;inset:0 auto 0 0;background:var(--p);border-radius:7px;min-width:14px}
.s4 .mu{color:var(--mu);font-size:12.5px}
.s4 pre{background:var(--sc2);border-radius:16px;padding:10px 14px;margin:8px 0 0;font:12px/1.45 ui-monospace,Menlo,monospace;white-space:pre-wrap;word-break:break-all;max-height:190px;overflow:auto;color:var(--tx)}
.s4 .job{background:var(--sc2);border-radius:24px;padding:14px 16px;margin-bottom:8px}
.s4 .gate{display:flex;gap:10px;align-items:center;padding:7px 10px;border-radius:16px;background:var(--sc2);margin-bottom:5px}
.s4 .gate .id{font-weight:700;min-width:30px}.s4 .gate .tag{font-size:11px;color:var(--mu);min-width:38px}
.s4 .st{width:22px;height:22px;border-radius:7px 50% 50% 50%;display:grid;place-items:center;font-size:12px;font-weight:700;background:var(--sc);color:var(--mu)}
.s4 .st.pass{background:var(--okc);color:var(--ok)}.s4 .st.fail{background:var(--erc);color:var(--er)}.s4 .st.doing{background:var(--pc);color:var(--on-pc)}
.s4 audio{height:34px;width:100%;margin-top:4px}
.s4 .au{background:var(--sc2);border-radius:20px;padding:10px 12px}
.s4 svg{display:block;width:100%}
.s4 .ev{padding:4px 0;border-bottom:1px solid var(--ol)}.s4 .ev:last-child{border:0}
.s4 a{color:var(--p)}
</style>"""


def rj(p, d=None):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return d


def e(x):
    return html.escape(str(x))


def ago(t):
    if not t:
        return "never"
    d = max(0, time.time() - float(t))
    return f"{int(d)}s ago" if d < 90 else f"{int(d/60)}m ago" if d < 5400 else f"{d/3600:.1f}h ago"


def dur(s):
    if s is None:
        return "—"
    s = int(s)
    return f"{s//3600}h {s%3600//60:02d}m" if s >= 3600 else f"{s//60}m {s%60:02d}s"


def chip(txt, kind=""):
    return f'<span class="chip {kind}"><span class="dot"></span>{e(txt)}</span>'


def spark(vals, color="var(--p)", h=44, vmax=None, fill=True):
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return f'<svg viewBox="0 0 300 {h}" height="{h}"><text x="4" y="{h/2}" fill="var(--mu)" font-size="11">collecting…</text></svg>'
    hi = vmax if vmax else (max(vals) or 1)
    lo = 0 if vmax else min(vals)
    span = (hi - lo) or 1
    pts = [(i * 300 / (len(vals) - 1), h - 3 - (v - lo) / span * (h - 8)) for i, v in enumerate(vals)]
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f'<path d="{d} L300,{h} L0,{h} Z" fill="{color}" opacity=".14"/>' if fill else ""
    return f'<svg viewBox="0 0 300 {h}" height="{h}" preserveAspectRatio="none">{area}<path d="{d}" fill="none" stroke="{color}" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/></svg>'


def status():
    return rj(f"{S4}/status.json", {}) or {}


def box_age(st):
    return time.time() - st.get("t", 0) if st else 1e9


def header():
    st = status()
    g = st.get("gpu", {})
    age = box_age(st)
    alive = age < 90
    ws = st.get("ws", {})
    jobs = st.get("jobs", [])
    run = sum(1 for j in jobs if j.get("alive"))
    c = [chip("guardian alive" if alive else f"guardian silent {dur(age)}", "ok" if alive else "er"),
         chip("keep-alive socket held" if ws.get("ok") else "socket reconnecting", "ok" if ws.get("ok") else "wa"),
         chip(f"{run}/{len(jobs)} jobs running", "ok" if run else "wa"),
         chip(g.get("name", "no GPU").replace("NVIDIA ", "").replace(" Server Edition", ""))]
    return (f'<div class="s4"><div class="hero"><div class="mu" style="color:inherit;opacity:.7">Sakhiya · Session 4 · GPU build</div>'
            f'<h2>Live build console</h2><div class="row" style="margin-top:12px">{"".join(c)}</div>'
            f'<div class="mu" style="color:inherit;opacity:.7;margin-top:10px">Box up {dur(st.get("uptime_s"))} · updated {ago(st.get("t"))} · reconnects {ws.get("reconnects", 0)}</div></div></div>{CSS}')


def gpu_card():
    st = status()
    g, h = st.get("gpu", {}), st.get("hist", {})
    if not g:
        return f'<div class="s4"><div class="card"><h3>GPU</h3><span class="mu">no telemetry yet — is s4_guardian running?</span></div></div>{CSS}'
    memp = 100 * g.get("mem_gb", 0) / (g.get("mem_total_gb", 96) or 96)
    tiles = [("util", f'{g.get("util", 0):.0f}%', "GPU util"), ("mem", f'{g.get("mem_gb", 0):.1f} GB', f"VRAM of {g.get('mem_total_gb', 0):.0f} ({memp:.0f}%)"),
             ("t", f'{g.get("temp", 0):.0f}°C', "temperature"), ("p", f'{g.get("power", 0):.0f} W', "power draw")]
    s = "".join(f'<div class="stat"><b>{a}</b><span>{b}</span></div>' for _, a, b in tiles)
    util_avg = (sum(h.get("util", [0])[-30:]) / max(1, len(h.get("util", [0])[-30:])))
    warn = ""
    if util_avg < 5 and any(j.get("alive") for j in st.get("jobs", [])):
        warn = chip("job running but GPU ≈ idle (last 10 min) — check dataloader", "wa")
    elif util_avg < 5:
        warn = chip("GPU idle — nothing is training right now", "er")
    cpu = f'CPU load {h.get("cpu", [0])[-1]:.1f} · RAM {st.get("mem_used_gb") or 0:.0f}/{st.get("mem_total_gb") or 0:.0f} GB · disk {st.get("disk", {}).get("used_gb", 0):.0f} GB'
    return (f'<div class="s4"><div class="card"><h3>GPU</h3><div class="grid">{s}</div>'
            f'<div style="margin-top:10px" class="mu">utilisation · last hour</div>{spark(h.get("util", []), vmax=100)}'
            f'<div class="mu">VRAM · last hour</div>{spark(h.get("mem", []), "var(--ter)", vmax=g.get("mem_total_gb") or 96)}'
            f'<div class="row" style="margin-top:6px">{warn}<span class="mu">{e(cpu)}</span></div></div></div>{CSS}')


def _parse_log(path):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            sz = f.tell()
            f.seek(max(0, sz - 12000))
            t = f.read().decode("utf8", "ignore")
    except Exception:
        return [], None
    lines = [l[:200] for l in t.splitlines() if l.strip()]
    m = None
    for l in reversed(lines):
        m = re.search(r"(?:step|it|iter)[ =:]*(\d+)\s*/\s*(\d+)", l, re.I)
        if m:
            break
    return lines[-6:], (int(m.group(1)), int(m.group(2))) if m else None


def jobs_card():
    st = status()
    jobs = st.get("jobs", [])
    if not jobs:
        return f'<div class="s4"><div class="card"><h3>Training jobs</h3><span class="mu">No jobs registered yet. Launch with <code>python3 scripts/s4_progress.py launch NAME "cmd" --restart</code> (or call <code>Progress(...)</code> from the trainer).</span></div></div>{CSS}'
    out = []
    for j in jobs:
        p = rj(f"{S4}/progress/{j['name']}.json", {}) or {}
        loglines, lp = _parse_log(j.get("log", ""))
        step, total = p.get("step"), p.get("total")
        if step is None and lp:
            step, total = lp
        pct = 100 * step / total if (step is not None and total) else None
        alive = j.get("alive")
        state = "done" if p.get("done") else "running" if alive else j.get("state", "stopped")
        kind = "ok" if state in ("running", "done") else "wa" if state == "restarted" else "er"
        stale = j.get("progress_age", 1e9)
        stale_chip = chip(f"no progress {dur(stale)}", "wa") if (alive and stale > 300 and stale < 1e8) else ""
        mets = p.get("metrics", {})
        mchips = "".join(f'<span class="chip">{e(k)} {v:.4g}</span>' for k, v in list(mets.items())[:6])
        # loss sparkline from metrics jsonl
        series = []
        try:
            with open(f"{S4}/progress/{j['name']}.metrics.jsonl") as f:
                rows = [json.loads(l) for l in f.readlines()[-240:]]
            key = "loss" if any("loss" in r for r in rows) else next((k for k in rows[-1] if k not in ("t", "step")), None) if rows else None
            series = [r.get(key) for r in rows if key and key in r]
        except Exception:
            key = None
        bar = f'<div class="bar"><i style="width:{pct:.1f}%"></i></div>' if pct is not None else '<div class="bar"><i style="width:14px;opacity:.4"></i></div>'
        info = f'{step:,} / {total:,} · {pct:.1f}%' if pct is not None else "no step counter yet"
        eta = f' · ETA {dur(p.get("eta_s"))}' if p.get("eta_s") else ""
        rate = f' · {p["rate"]:.2f} it/s' if p.get("rate") else ""
        out.append(f'<div class="job"><div class="row" style="justify-content:space-between"><b>{e(j["name"])}</b>'
                   f'<span class="row">{chip(state, kind)}{stale_chip}{chip(p["stage"]) if p.get("stage") else ""}</span></div>'
                   f'<div style="margin:10px 0 6px">{bar}</div><div class="mu">{e(info)}{e(eta)}{e(rate)} · restarts {j.get("restarts", 0)}</div>'
                   f'<div class="row" style="margin-top:6px">{mchips}</div>'
                   + (f'<div class="mu" style="margin-top:8px">{e(key)} trend</div>{spark(series, "var(--ter)")}' if len(series) > 2 else "")
                   + (f'<div class="mu">{e(p.get("note"))}</div>' if p.get("note") else "")
                   + (f'<pre>{e(chr(10).join(loglines))}</pre>' if loglines else "") + "</div>")
    return f'<div class="s4"><div class="card"><h3>Training jobs</h3>{"".join(out)}</div></div>{CSS}'


def gates_card():
    g = rj(f"{S4}/gates.json")
    if not g:
        g = {i: {"state": "todo", "note": ""} for i, _, _ in DEFAULT_GATES}
    rows, done = [], 0
    for i, tag, title in DEFAULT_GATES:
        s = g.get(i, {}).get("state", "todo")
        done += s == "pass"
        sym = {"pass": "✓", "fail": "✕", "doing": "●"}.get(s, "")
        note = g.get(i, {}).get("note", "")
        rows.append(f'<div class="gate"><span class="st {s}">{sym}</span><span class="id">{i}</span><span class="tag">{tag}</span><span style="flex:1">{e(title)}'
                    + (f'<div class="mu">{e(note)}</div>' if note else "") + "</span></div>")
    pct = 100 * done / len(DEFAULT_GATES)
    return (f'<div class="s4"><div class="card"><h3>Plan gates <span class="chip">{done}/{len(DEFAULT_GATES)}</span></h3>'
            f'<div class="bar" style="margin-bottom:10px"><i style="width:{pct:.0f}%"></i></div>{"".join(rows)}</div></div>{CSS}')


def samples_card(n=6):
    idx = []
    try:
        idx = [json.loads(l) for l in open(f"{S4}/samples/index.jsonl")][-n:][::-1]
    except Exception:
        pass
    if not idx:
        return f'<div class="s4"><div class="card"><h3>Newest audio</h3><span class="mu">No samples yet. Trainers call <code>Progress.sample(wav, label)</code> at every checkpoint — they appear here for blind listening.</span></div></div>{CSS}'
    items = []
    for r in idx:
        p = f"{S4}/samples/{r['file']}"
        try:
            if os.path.getsize(p) > 900_000:
                raise ValueError
            b = base64.b64encode(open(p, "rb").read()).decode()
            items.append(f'<div class="au"><b>{e(r.get("job"))}</b> <span class="mu">{ago(r.get("t"))}</span><div class="mu">{e(r.get("label"))}</div><audio controls preload="none" src="data:audio/wav;base64,{b}"></audio></div>')
        except Exception:
            items.append(f'<div class="au"><b>{e(r.get("job"))}</b> <span class="mu">{e(r["file"])} (too large to embed)</span></div>')
    return f'<div class="s4"><div class="card"><h3>Newest audio</h3><div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(260px,1fr))">{"".join(items)}</div></div></div>{CSS}'


def activity_card():
    st = status()
    commits = []
    for c in st.get("git", [])[:8]:
        who = c.get("author", "")
        commits.append(f'<div class="ev"><span class="chip {"ok" if "emini" in who else ""}">{e(who[:14])}</span> <code>{e(c.get("sha"))}</code> {e(c.get("subject", "")[:110])} <span class="mu">{ago(c.get("t"))}</span></div>')
    evs = []
    try:
        evs = [json.loads(l) for l in open(f"{S4}/events.jsonl")][-8:][::-1]
    except Exception:
        pass
    ev_h = "".join(f'<div class="ev"><span class="mu">{ago(x["t"])}</span> <b>{e(x["kind"])}</b> {e(x["msg"])}</div>' for x in evs) or '<span class="mu">no events</span>'
    return (f'<div class="s4"><div class="card"><h3>Repo (origin/main)</h3>{"".join(commits) or "<span class=mu>fetching…</span>"}'
            f'<h3 style="margin-top:14px">Guardian events</h3>{ev_h}</div></div>{CSS}')
