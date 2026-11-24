"""Sakhiya Pro v3 / Shrink-Ladder — 5-panel live Marimo dashboard (zero-error by construction).

Install (on the box, inside the marimo kernel with top-level await):
    exec(open('/work/sp/repo/monitor/cells_pro5.py').read()); await install()

Every card body lives inside a function and is wrapped in try/except, so a missing file or half-written
json never turns into a cell error. Private (underscore) names only -> no cross-cell redefinition errors.
Panels: 1 box health + countdown, 2 accent-fix / corpus status, 3 live ladder progress (it/s, ETA),
4 size-vs-quality leaderboard, 5 interactive listening desk (auto-discovers newest INT8 / Pro audio).
"""

TICK = r"""
import marimo as _mo
sp5_tick = _mo.ui.refresh(options=["10s", "15s", "30s", "1m"], default_interval="15s", label="Sakhiya live auto-refresh")
sp5_tick
"""

COMMON = r"""
def _sp5_helpers():
    import json, os, subprocess, html
    def tail(p, n=40):
        try:
            return open(p, errors="ignore").read().splitlines()[-n:]
        except Exception:
            return []
    def sh(c, t=6):
        try:
            return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t).stdout.strip()
        except Exception:
            return ""
    def jl(p, d=None):
        try:
            if os.path.exists(p):
                return json.load(open(p, errors="ignore"))
        except Exception:
            pass
        return d
    box = lambda title, body: (
        "<div style='border:1px solid #d0d7de;border-radius:10px;padding:10px 14px;margin:4px 0;font-family:system-ui,sans-serif;font-size:13px'>"
        f"<div style='font-weight:700;font-size:15px;margin-bottom:6px'>{title}</div>{body}</div>")
    return dict(tail=tail, sh=sh, jl=jl, esc=html.escape, box=box)
"""

C1 = r"""
sp5_tick.value
def _card1():
    import marimo as _mo, datetime, os
    _g = {}
    exec(_COMMON_SRC, _g); H = _g["_sp5_helpers"]()
    sh, tail, box = H["sh"], H["tail"], H["box"]
    gpu = sh("nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader")
    npt = sh("TZ=Asia/Kathmandu date '+%a %d %b %H:%M:%S NPT'")
    now = datetime.datetime.now(datetime.timezone.utc)
    try:
        dl = datetime.datetime.fromisoformat(open("/work/sp/deadline_utc.txt").read().strip().rstrip("Z")).replace(tzinfo=datetime.timezone.utc)
    except Exception:
        dl = now + datetime.timedelta(hours=10)
    left = max(0, int((dl - now).total_seconds()))
    cd = f"{left//3600:d}h {(left%3600)//60:02d}m {left%60:02d}s"
    dl_npt = (dl + datetime.timedelta(hours=5, minutes=45)).strftime("%H:%M NPT")
    ram = sh("free -h | awk '/Mem:/ {print $3 \" / \" $2}'")
    disk = sh("df -h / | awk 'NR==2 {print $3 \" used\"}'")
    procs = sh("ps -eo pid,etime,pcpu,args --sort=-pcpu | grep -E '[t]rain_student|[l]adder_daemon|[g]en_corpus|[b]ackup_daemon|[t]rain_voc|[g]ta_polish' | cut -c1-150 | head -12")
    bk = tail("/work/tts/logs/backup_daemon_v3.log", 2)
    bk_alive = bool(sh("pgrep -f '[b]ackup_daemon.py'"))
    body = (f"<div>🕒 {npt} &nbsp;|&nbsp; ⏳ <b style='font-size:18px'>{cd}</b> to hard cutoff ({dl_npt})</div>"
            f"<div>🖥️ {H['esc'](gpu) or 'nvidia-smi unavailable'}</div>"
            f"<div>RAM {ram} · disk {disk}</div>"
            f"<div>{'🟢' if bk_alive else '🔴'} HF backup daemon (15 min): {H['esc'](' | '.join(bk)) or 'no log yet'}</div>"
            f"<pre style='font-size:11px;margin:6px 0 0;white-space:pre-wrap'>{H['esc'](procs) or 'no training processes'}</pre>")
    return _mo.Html(box("1 · Box health & countdown", body))
try:
    _out1 = _card1()
except Exception as _e:
    import marimo as _mo
    _out1 = _mo.md(f"card 1 soft-error: `{type(_e).__name__}: {_e}`")
_out1
"""

C2 = r"""
sp5_tick.value
def _card2():
    import marimo as _mo, os, glob, json
    _g = {}
    exec(_COMMON_SRC, _g); H = _g["_sp5_helpers"]()
    jl, tail, box, esc, sh = H["jl"], H["tail"], H["box"], H["esc"], H["sh"]
    cr = jl("/work/sp/distill/corpus_report.json", {}) or {}
    gr = jl("/work/sp/distill/gt_report.json", {}) or {}
    ac = jl("/work/sp/final/sakhiya_pro_v3/eval_accent.json", {}) or {}
    rows = "".join(f"<tr><td>{esc(str(k))}</td><td align=right>{v} h</td></tr>" for k, v in sorted((cr.get('by_src') or {}).items(), key=lambda kv: -float(kv[1] or 0)))
    gt_rows = "".join(f"<tr><td>{esc(str(k))}</td><td align=right>{v} h</td></tr>" for k, v in sorted((gr.get('by_src') or {}).items(), key=lambda kv: -float(kv[1] or 0)))
    gen_alive = sh("pgrep -fc '[g]en_corpus.py work'") or "0"
    acc = ""
    if isinstance(ac, dict) and ac:
        acc = "<pre style='font-size:11px;white-space:pre-wrap;max-height:160px;overflow:auto'>" + esc(json.dumps(ac, ensure_ascii=False)[:1400]) + "</pre>"
    body = (f"<div><b>Teacher corpus (Pro v3 synth + voice-converted real speech):</b> {cr.get('utts','?')} utts · <b>{cr.get('hours','?')} h</b> · mean UTMOS {cr.get('mean_utmos','?')}</div>"
            f"<div><b>Student training set (GT cache):</b> {gr.get('items','?')} items · <b>{gr.get('hours','?')} h</b></div>"
            f"<div>gen_corpus workers alive: <b>{esc(gen_alive)}</b> (0 = paused/finished)</div>"
            "<div style='display:flex;gap:24px;margin-top:6px'>"
            f"<table style='font-size:12px'><tr><th colspan=2>teacher corpus</th></tr>{rows}</table>"
            f"<table style='font-size:12px'><tr><th colspan=2>GT cache</th></tr>{gt_rows}</table></div>"
            "<div style='margin-top:6px'><b>Accent-fix audit (Pro v3):</b> language tag <code>npi</code> (never auto), <code>instruct=None</code>, anchor = SLR43 6834 (downward closure).</div>" + acc)
    return _mo.Html(box("2 · Accent-fix & corpus status", body))
try:
    _out2 = _card2()
except Exception as _e:
    import marimo as _mo
    _out2 = _mo.md(f"card 2 soft-error: `{type(_e).__name__}: {_e}`")
_out2
"""

C3 = r"""
sp5_tick.value
def _card3():
    import marimo as _mo, re, os, glob, time
    _g = {}
    exec(_COMMON_SRC, _g); H = _g["_sp5_helpers"]()
    tail, box, esc = H["tail"], H["box"], H["esc"]
    rx_v = re.compile(r"Step (\d+) mel ([\d.]+) stft ([\d.]+) (?:f0|teach) ([\d.]+) D ([\d.]+) \| ([\d.]+)/(\d+) min")
    vrows = ""
    for vlbl, vlg in (("awp_voc224 (Phase 1)", "/work/tts/logs/awp_voc224.log"), ("awp_voc136 (Phase 1)", "/work/tts/logs/awp_voc136.log")):
        vlast = None
        for ln in reversed(tail(vlg, 30)):
            m = rx_v.search(ln)
            if m:
                vlast = m
                break
        if vlast:
            vst, vmel, vstft, vf0, vd, vel, vbud = int(vlast[1]), float(vlast[2]), float(vlast[3]), float(vlast[4]), float(vlast[5]), float(vlast[6]), float(vlast[7])
            vdone = any("VOC_GT_DONE" in ln for ln in tail(vlg, 5))
            vstate = "✅ done" if vdone else ("🟢 running" if int(time.time() - os.path.getmtime(vlg)) < 120 else "⏸️ idle")
            vrows += f"<tr><td><b>{vlbl}</b></td><td align=right>{vst}</td><td align=right>mel {vmel:.3f}</td><td align=right>stft {vstft:.3f}</td><td align=right>{vel:.1f}/{vbud:.0f} min</td><td align=right>{max(0.0, vbud-vel):.0f} min</td><td>{vstate}</td></tr>"
    rx = re.compile(r"Step (\d+) \| ([\d.]+) it/s \| mel ([\d.]+).*?\| ([\d.]+)/(\d+) min")
    rows = ""
    for tier in ("t30", "t15", "t10", "t1"):
        lg = f"/work/sp/distill/runs/{tier}/train.log"
        last = None
        for ln in reversed(tail(lg, 30)):
            m = rx.search(ln)
            if m:
                last = m
                break
        if not os.path.exists(lg):
            rows += f"<tr><td>{tier}</td><td colspan=7>not started</td></tr>"
            continue
        if not last:
            rows += f"<tr><td>{tier}</td><td colspan=7>starting…</td></tr>"
            continue
        step, its, mel, el, bud = int(last[1]), float(last[2]), float(last[3]), float(last[4]), float(last[5])
        age = int(time.time() - os.path.getmtime(lg))
        alive = "🟢" if age < 120 else "🔴 stalled"
        eta = max(0.0, bud - el)
        pct = min(100, 100 * el / bud)
        bar = f"<div style='background:#eee;width:140px;height:8px;border-radius:4px'><div style='background:#2da44e;width:{pct:.0f}%;height:8px;border-radius:4px'></div></div>"
        rows += (f"<tr><td><b>{tier}</b></td><td align=right>{step}</td><td align=right>{its:.2f}</td><td align=right>{mel:.3f}</td>"
                 f"<td align=right>{el:.0f}/{bud:.0f} min</td><td align=right>{eta:.0f} min</td><td>{bar}</td><td>{alive}</td></tr>")
    ld = tail("/work/tts/logs/ladder_daemon.log", 4) + tail("/work/sp/distill/ladder_daemon.log", 4)
    vtbl = (f"<div style='margin-bottom:6px'><b>Stage V · Pro-v3 Harmonic-iSTFT Vocoder GANs:</b>"
            f"<table style='font-size:12px;border-spacing:10px 2px'>{vrows}</table></div>") if vrows else ""
    body = (vtbl + "<table style='font-size:12.5px;border-spacing:10px 2px'><tr><th>tier</th><th>step</th><th>step/s</th><th>mel</th><th>elapsed</th><th>ETA</th><th></th><th>state</th></tr>"
            + rows + "</table>"
            "<div style='margin-top:6px;color:#555'>INT8 forensic gate (ladder daemon, every 20 min):</div>"
            f"<pre style='font-size:11px;white-space:pre-wrap'>{esc(chr(10).join(ld[-5:])) or 'waiting for first gate'}</pre>")
    return _mo.Html(box("3 · Shrink-ladder live progress", body))
try:
    _out3 = _card3()
except Exception as _e:
    import marimo as _mo
    _out3 = _mo.md(f"card 3 soft-error: `{type(_e).__name__}: {_e}`")
_out3
"""

C4 = r"""
sp5_tick.value
def _card4():
    import marimo as _mo
    _g = {}
    exec(_COMMON_SRC, _g); H = _g["_sp5_helpers"]()
    jl, box, esc = H["jl"], H["box"], H["esc"]
    lb = jl("/work/sp/tiers/leaderboard.json", []) or []
    v1 = next((r for r in lb if str(r.get("name", "")).startswith("Sakhiya-V1 ")), {})
    def f(x, nd=3):
        try:
            return f"{float(x):.{nd}f}"
        except Exception:
            return esc(str(x))
    def mbkey(r):
        try:
            return float(str(r.get("size_mb")).lstrip("~"))
        except Exception:
            return 1e9
    rows = ""
    for r in sorted(lb, key=mbkey):
        try:
            d = float(r["utmos"]) - float(v1.get("utmos"))
            delta = f"<span style='color:{'#1a7f37' if d >= 0 else '#cf222e'}'>{d:+.2f}</span>"
        except Exception:
            delta = "–"
        rows += (f"<tr><td>{esc(str(r.get('name')))}</td><td align=right>{f(str(r.get('size_mb')).lstrip('~'), 2)}</td><td align=right>{f(r.get('utmos'))}</td>"
                 f"<td align=right>{delta}</td><td align=right>{f(r.get('cer'))}</td><td align=right>{r.get('drops','–')}</td><td align=right>{r.get('clicks','–')}</td>"
                 f"<td align=right>{f(r.get('spk'),2)}</td><td align=right>{f(r.get('rtf'),3)}</td><td>{esc(str(r.get('status',''))[:90])}</td></tr>")
    head = "<tr><th align=left>model</th><th>MB</th><th>UTMOS</th><th>Δ vs V1</th><th>CER</th><th>drops</th><th>clicks</th><th>spk</th><th>RTF</th><th align=left>status</th></tr>"
    body = ("<div style='color:#555;margin-bottom:4px'>Same metric stack for every row (UTMOS, Nepali CER, clicks, word drops, ECAPA spk-sim). Sorted by size. Δ is UTMOS vs the live V1.</div>"
            f"<table style='font-size:12px;border-spacing:8px 2px'>{head}{rows}</table>")
    return _mo.Html(box("4 · Size-vs-Quality leaderboard", body))
try:
    _out4 = _card4()
except Exception as _e:
    import marimo as _mo
    _out4 = _mo.md(f"card 4 soft-error: `{type(_e).__name__}: {_e}`")
_out4
"""

C5_SCAN = r"""
import marimo as _mo
sp5_rescan = _mo.ui.run_button(label="🔄 Rescan audio")
sp5_rescan
"""

C5_PICK = r"""
import marimo as _mo
sp5_rescan.value
def _discover():
    import glob, os
    pats = ["/work/sp/tiers/*/best/wav/*", "/work/sp/tiers/*/c_*/wav/*", "/work/sp/listen/*/*", "/work/sp/out/listen/*/*",
            "/work/sp/repo/docs/samples/versions/*/*.mp3", "/work/sp/repo/docs/samples/versions/*/*.wav", "/work/sp/repo/docs/samples/versions/*/*/*.mp3"]
    seen = {}
    for p in pats:
        for f in glob.glob(p):
            if f.lower().endswith((".wav", ".mp3")) and os.path.isfile(f) and os.path.getsize(f) < 6_000_000:
                seen[f] = os.path.getmtime(f)
    items = sorted(seen.items(), key=lambda kv: -kv[1])[:300]
    return {f.replace("/work/sp/", ""): f for f, _ in items}
sp5_files = _discover()
sp5_pick = _mo.ui.dropdown(options=sp5_files, value=(next(iter(sp5_files)) if sp5_files else None), label=f"Clip ({len(sp5_files)} found, newest first)", full_width=True)
sp5_pick
"""

C5_PLAY = r"""
def _play():
    import marimo as _mo, base64, os, time
    p = sp5_pick.value
    if not p or not os.path.isfile(p):
        return _mo.md("_No audio yet — INT8 gate WAVs appear after the first ladder gate (about 20 min)._")
    mime = "audio/mpeg" if p.lower().endswith(".mp3") else "audio/wav"
    b = base64.b64encode(open(p, "rb").read()).decode()
    st = os.stat(p)
    return _mo.Html(f"<div style='font-family:system-ui;font-size:13px'><b>{os.path.basename(p)}</b> · {st.st_size/1024:.0f} KB · modified {time.strftime('%H:%M:%S', time.gmtime(st.st_mtime))} UTC"
                    f"<br><audio controls preload='auto' style='width:100%;margin-top:6px' src='data:{mime};base64,{b}'></audio></div>")
try:
    _out5 = _play()
except Exception as _e:
    import marimo as _mo
    _out5 = _mo.md(f"player soft-error: `{type(_e).__name__}: {_e}`")
_out5
"""

CELLS = {
    "tts_sp5_tick": TICK,
    "tts_sp5_c1_box": C1,
    "tts_sp5_c2_accent": C2,
    "tts_sp5_c3_ladder": C3,
    "tts_sp5_c4_board": C4,
    "tts_sp5_c5_scan": C5_SCAN,
    "tts_sp5_c5_pick": C5_PICK,
    "tts_sp5_c5_play": C5_PLAY,
}

# helper source is injected as a private string constant into each card cell
_COMMON_LINE = "_COMMON_SRC = " + repr(COMMON) + "\n"


def _with_common(src):
    return _COMMON_LINE + src if "_COMMON_SRC" in src else src


async def install():
    import marimo._code_mode as cm
    async with cm.get_context(skip_staleness_check=True) as ctx:
        by_name = {c.name: c.id for c in ctx.cells.values()}
        for c in list(ctx.cells.values()):
            if c.name not in CELLS:
                ctx.delete_cell(c.id)
        ids = {}
        for name, src in CELLS.items():
            code = _with_common(src).strip()
            if name in by_name:
                ids[name] = by_name[name]
                ctx.edit_cell(ids[name], code, hide_code=True)
            else:
                ids[name] = ctx.create_cell(code, name=name, hide_code=True)
        for name in CELLS:
            ctx.run_cell(ids[name])
    print("installed", list(CELLS))
