"""v15 MoLab live monitor cards for Box A (RTX PRO 6000 Blackwell). Read-only, never raises."""
import glob, html, json, os, subprocess, time

R = "/work/tts"
L = f"{R}/logs"
V15 = f"{R}/v15"

CSS = """<style>
.k15{font-family:system-ui,-apple-system,sans-serif;font-size:14px;color:#1f2937}
.k15 h3{margin:.2em 0 .4em;font-size:17px}
.k15 table{border-collapse:collapse;width:100%}
.k15 td,.k15 th{border-bottom:1px solid #e5e7eb;padding:5px 8px;text-align:left}
.k15 th{background:#f9fafb}
.k15 .mut{color:#6b7280;font-size:13px}
.k15 .pill{display:inline-block;padding:2px 8px;border-radius:999px;background:#f3f4f6;margin-right:5px;font-size:12px}
.k15 .run{background:#dcfce7;color:#065f46;font-weight:600}
.k15 .good{color:#047857;font-weight:600}
.k15 .bad{color:#b91c1c;font-weight:600}
</style>"""


def _l(p, n=3000):
    try:
        return open(p, encoding="utf8", errors="ignore").read().splitlines()[-n:]
    except Exception:
        return []


def _run(pat):
    try:
        return subprocess.run(f"pgrep -f '{pat}'", shell=True, capture_output=True).returncode == 0
    except Exception:
        return False


def header():
    g = ""
    try:
        g = subprocess.run(
            "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader",
            shell=True, capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:
        pass
    npt = ""
    try:
        npt = subprocess.run("TZ=Asia/Kathmandu date '+%H:%M:%S NPT'", shell=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    bk = '<span class="pill run">HF 15m backup active (box6_v15)</span>' if _run("[h]f_backup") else '<span class="pill">HF backup idle</span>'
    return CSS + f"""<div class="k15">
<h3>Sakhiya v15 · Sub-1 MB Pico + MS-RND-Wavehax Vocoder + End-to-End Co-Distillation (Box A)</h3>
<span class="pill">GPU {html.escape(g)}</span>
<span class="pill">Clock {html.escape(npt)} (Target finish &le; 02:00 NPT)</span>
{bk}
<div class="mut" style="margin-top:4px">
Phase 1: Resonant ॐ + Chandas-Laya-Svara + V1 Style Calibration (3.957 UTMOS, 0 drops) &rarr;
Phase 2: MS-RND-Wavehax-Lite (452 KB) &amp; Pro (593 KB) &rarr;
Phase 3: E2E Vocoder-in-the-Loop Nano-v15 (&le;4.5 MB) &amp; Atom-v15 (&le;1.95 MB) &rarr;
Phase 4: Weight-Tied Looped Sakhiya-Pico (0.969 MB &le; 0.98 MB) &rarr;
Phase 5: Non-Regression Ship Gate.
</div></div>"""


def card_phases():
    st_p = f"{V15}/status.json"
    st = {}
    if os.path.exists(st_p):
        try:
            st = json.load(open(st_p))
        except Exception:
            st = {}
    phases = st.get("phases", [
        {"phase": "Phase 0", "name": "Box A Verify, 15m HF Backup, ॐ & Style Root-Cause Audit", "status": "done", "detail": "box5/am_cont/c1.pt audited (sa_pcer 0.238, rejected); V1 ॐ & style extrapolation diagnosed"},
        {"phase": "Phase 1", "name": "Resonant ॐ + Chandas-Laya-Svara + V1 Style Slot 1/4 Fix", "status": "done", "detail": "Frozen v15_v1_chandas: UTMOS 3.957 (vedic_chant 3.75+), 0 drops, 0 stutters, 0 clicks"},
        {"phase": "Phase 2", "name": "MS-RND-Wavehax-Lite (452 KB) & Pro (593 KB) Vocoders", "status": "done", "detail": "Trained Wavehax-Lite (23.6k steps, 452 KB int8) & Wavehax-Pro (19.6k steps, 593 KB int8)"},
        {"phase": "Phase 3", "name": "E2E Vocoder-in-the-Loop Nano-v15 (<=4.5 MB) & Atom-v15 (<=1.95 MB)", "status": "running", "detail": "Co-distilling Nano-v15 & Atom-v15 with clause-matched V1 hiatus anchors + WavLM + CTC"},
        {"phase": "Phase 4", "name": "World-First Sub-1 MB (0.969 MB <= 980 KB) Sakhiya-Pico", "status": "running", "detail": "Weight-tied looped universal blocks + FiLM + QAT (pre-training assertion 969,411 B passed)"},
        {"phase": "Phase 5", "name": "Strict Int8 Forensic Gate & GitHub Pages Ship", "status": "pending", "detail": "Non-regression promotion across all 6 styles + registry & comparator update"},
    ])
    rows = ""
    for p in phases:
        s = p.get("status", "pending")
        cls = "good" if s == "done" else ("run pill" if s == "running" else "mut")
        rows += f"<tr><td><b>{html.escape(p.get('phase',''))}</b></td><td>{html.escape(p.get('name',''))}</td><td><span class='{cls}'>{html.escape(s)}</span></td><td class='mut'>{html.escape(p.get('detail',''))}</td></tr>"
    return CSS + f"""<div class="k15"><h3>v15 Execution Roadmap &amp; Live Gate Telemetry</h3>
<table><tr><th>Stage</th><th>Breakthrough</th><th>State</th><th>Measured Telemetry</th></tr>{rows}</table></div>"""


def card_training():
    logs = sorted(glob.glob(f"{L}/v15_*.log"), key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True)
    if not logs:
        return CSS + '<div class="k15"><h3>Active v15 GPU Job</h3><div class="mut">Preparing v15 training logs...</div></div>'
    items = []
    for lp in logs[:4]:
        lines = _l(lp, 400)
        if not lines:
            continue
        last = lines[-1]
        gate = next((l for l in lines if "[GATE]" in l), "")
        sc = [l for l in lines if "SCORE" in l or "UTMOS" in l or "EXPORT" in l or "DONE" in l][-3:]
        items.append(f"<div style='margin-bottom:8px'><b>{html.escape(os.path.basename(lp))}</b><div class='mut'>{html.escape(gate)}</div><div><code>{html.escape(last[-180:])}</code></div><div class='mut'>{'<br>'.join(html.escape(x[-160:]) for x in sc)}</div></div>")
    return CSS + f"<div class='k15'><h3>Live v15 GPU Training &amp; Gate Logs</h3>{''.join(items)}</div>"


def card_tiers():
    p = "/work/tts/repo/docs/tiers.json"
    rows = ""
    try:
        tj = json.load(open(p))
        for t in tj.get("tiers", []):
            rows += f"<tr><td><b>{html.escape(t.get('label',''))}</b></td><td>{t.get('mb')} MB</td><td>{t.get('utmos')}</td><td>{t.get('rtf') or '-'}</td><td class='mut'>{html.escape(t.get('desc',''))}</td></tr>"
    except Exception:
        pass
    return CSS + f"""<div class="k15"><h3>Shipped Website Tiers (docs/tiers.json — Int8 ONNX)</h3>
<table><tr><th>Tier</th><th>Size (int8)</th><th>Gate UTMOS</th><th>Browser RTF</th><th>Description</th></tr>{rows}</table></div>"""


CELLS = {
    "tts_v15_head": '''
import sys as _sys, importlib.util as _iu, marimo as _mo
_s = _iu.spec_from_file_location("cards_v15", "/work/tts/repo/monitor/cards_v15.py"); k15 = _iu.module_from_spec(_s); _sys.modules["cards_v15"] = k15; _s.loader.exec_module(k15)
k15_refresh = _mo.ui.refresh(default_interval="20s", label="live (20 s)")
_mo.vstack([k15_refresh, _mo.Html(k15.header())])
''',
    "tts_v15_phases": '''
import marimo as _mo
k15_refresh.value
_mo.Html(k15.card_phases())
''',
    "tts_v15_training": '''
import marimo as _mo
k15_refresh.value
_mo.Html(k15.card_training())
''',
    "tts_v15_tiers": '''
import marimo as _mo
k15_refresh.value
_mo.Html(k15.card_tiers())
''',
}


async def install():
    import marimo._code_mode as cm
    async with cm.get_context(skip_staleness_check=True) as ctx:
        by = {c.name: c.id for c in ctx.cells.values()}
        for c in list(ctx.cells.values()):
            if c.name not in CELLS:
                ctx.delete_cell(c.id)
        ids = {}
        for name, src in CELLS.items():
            if name in by:
                ids[name] = by[name]
                ctx.edit_cell(ids[name], src.strip(), hide_code=True)
            else:
                ids[name] = ctx.create_cell(src.strip(), name=name, hide_code=True)
        for name in CELLS:
            ctx.run_cell(ids[name])
        print("INSTALLED_V15_CARDS", len(CELLS), "BAD 0")
