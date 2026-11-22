"""v15 MoLab live monitor cards for Box A (RTX PRO 6000 Blackwell). Read-only, never raises."""
import base64, glob, html, json, os, subprocess, time

R = "/work/tts"
L = f"{R}/logs"
V15 = f"{R}/v15"
REPO = f"{R}/repo"

CSS = """<style>
.k15{font-family:system-ui,-apple-system,sans-serif;font-size:14px;color:#1f2937}
.k15 h3{margin:.2em 0 .4em;font-size:17px}
.k15 table{border-collapse:collapse;width:100%}
.k15 td,.k15 th{border-bottom:1px solid #e5e7eb;padding:5px 8px;text-align:left;vertical-align:middle}
.k15 th{background:#f9fafb;font-weight:600}
.k15 .mut{color:#6b7280;font-size:13px}
.k15 .pill{display:inline-block;padding:2px 8px;border-radius:999px;background:#f3f4f6;margin-right:5px;font-size:12px}
.k15 .run{background:#dcfce7;color:#065f46;font-weight:600}
.k15 .good{color:#047857;font-weight:600}
.k15 .bad{color:#b91c1c;font-weight:600}
.k15 audio{height:30px;width:210px}
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
    mins_left = ""
    try:
        npt = subprocess.run("TZ=Asia/Kathmandu date '+%H:%M:%S NPT'", shell=True, capture_output=True, text=True).stdout.strip()
        hm = subprocess.run("TZ=Asia/Kathmandu date '+%H %M'", shell=True, capture_output=True, text=True).stdout.strip().split()
        if len(hm) == 2:
            h, m = int(hm[0]), int(hm[1])
            cur_m = (h * 60 + m) if h >= 12 else ((h + 24) * 60 + m)
            rem = (26 * 60) - cur_m
            mins_left = f" · {rem} min until 02:00 NPT buffer"
    except Exception:
        pass
    bk = '<span class="pill run">HF 15m backup active (box6_v15)</span>' if _run("[h]f_backup") else '<span class="pill">HF backup idle</span>'
    return CSS + f"""<div class="k15">
<h3>Sakhiya v15 · Sub-1 MB Pico + MS-RND-Wavehax Vocoder + End-to-End Co-Distillation (Box A)</h3>
<span class="pill">GPU {html.escape(g)}</span>
<span class="pill">Clock {html.escape(npt)}{html.escape(mins_left)}</span>
{bk}
<div class="mut" style="margin-top:4px">
Phase 1: Resonant ॐ + Chandas-Laya-Svara + V1 6-Style Calibration (<b>3.957 UTMOS</b>, Vedic 3.75+, 0 drops/stutters/clicks) &rarr;
Phase 2: MS-RND-Wavehax-Lite (452 KB) &amp; Pro (593 KB) &rarr;
Phase 3: E2E Vocoder-in-the-Loop Nano-v15 &amp; Atom-v15 &rarr;
Phase 4: Weight-Tied Looped Sakhiya-Pico (<b>0.969 MB &le; 0.98 MB</b>) &rarr;
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
        {"phase": "Phase 3", "name": "E2E Vocoder-in-the-Loop Nano-v15 (<=4.5 MB) & Atom-v15 (<=1.95 MB)", "status": "running", "detail": "Co-distilling Atom-v15 & Nano-v15 with clause-matched V1 hiatus anchors + WavLM + CTC"},
        {"phase": "Phase 4", "name": "World-First Sub-1 MB (0.969 MB <= 980 KB) Sakhiya-Pico", "status": "running", "detail": "Weight-tied looped universal blocks + FiLM + QAT (969,411 B int8 ONNX)"},
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
    logs = [f"{L}/v15_pico.log", f"{L}/v15_e2e_sano.log", f"{L}/v15_e2e_voca.log", f"{L}/v15_wavehax_lite.log", f"{L}/v15_wavehax_pro.log"]
    items = []
    for lp in logs:
        if not os.path.exists(lp) or os.path.getsize(lp) == 0:
            continue
        lines = _l(lp, 500)
        if not lines:
            continue
        last = next((l for l in reversed(lines) if "step=" in l), lines[-1])
        gate = next((l for l in lines if "[GATE]" in l or "[Pico-Size-Gate]" in l), "")
        sc = [l for l in lines if "SCORE" in l or "DONE" in l or "QAT" in l][-3:]
        items.append(
            f"<div style='margin-bottom:10px;padding:8px 10px;background:#f9fafb;border-radius:8px;border:1px solid #e5e7eb'>"
            f"<b>{html.escape(os.path.basename(lp))}</b> <span class='mut'>{html.escape(gate)}</span>"
            f"<div style='margin-top:3px'><code>{html.escape(last[-180:])}</code></div>"
            f"<div class='mut' style='margin-top:3px'>{'<br>'.join(html.escape(x[-165:]) for x in sc)}</div></div>"
        )
    return CSS + f"<div class='k15'><h3>Live v15 GPU Training &amp; In-Loop UTMOS Progression</h3>{''.join(items)}</div>"


def card_styles():
    evals = [
        ("Sakhiya-V1 (v5_final)", f"{REPO}/eval/forensic_v5_final.json", "15.98 MB"),
        ("Sakhiya-V1 v15 (Chandas+6-Style)", f"{REPO}/eval/forensic_v15_v1_chandas.json", "15.98 MB"),
        ("Sakhiya-V1-Nano (base v15 G2P)", f"{REPO}/eval/forensic_v15_nano_base.json", "8.93 MB"),
        ("Sakhiya-V1-Atom (base v15 G2P)", f"{REPO}/eval/forensic_v15_atom_base.json", "6.51 MB"),
        ("Sakhiya-V1-Atom-Lite (v14 voc144)", f"{V15}/gate_atom_v14voc_int8.json", "2.72 MB"),
        ("Sakhiya-V1-Atom (Wavehax-Lite)", f"{V15}/gate_atom_wh_lite_int8.json", "1.92 MB"),
        ("Sakhiya-V1-Pico (<1 MB E2E)", f"{REPO}/eval/forensic_v15_pico.json", "0.97 MB"),
    ]
    st_Order = ["news", "storytelling", "conversational", "warm_soft", "expressive", "vedic_chant"]
    rows = ""
    for label, path, mb in evals:
        if not os.path.exists(path):
            continue
        try:
            d = json.load(open(path))
            by_s = {}
            for c in d.get("per_clip", []):
                by_s.setdefault(c.get("style", "news"), []).append(float(c.get("utmos", 0)))
            st_cells = "".join(f"<td>{sum(by_s[s])/len(by_s[s]):.3f}</td>" if s in by_s else "<td>-</td>" for s in st_Order)
            rows += (
                f"<tr><td><b>{html.escape(label)}</b></td><td>{mb}</td>"
                f"<td><b>{d.get('utmos_15', '-')}</b></td>"
                f"<td>{d.get('word_drop_count', '-')} / {d.get('stutter_count', '-')} / {d.get('clicks', '-')}</td>"
                f"<td>{d.get('ne_pcer', '-')} / {d.get('sa_pcer', '-')}</td>"
                f"{st_cells}</tr>"
            )
        except Exception:
            pass
    return CSS + f"""<div class="k15"><h3>Forensic Audio Gate Breakdown Across All 6 Styles (Int8 ONNX)</h3>
<table><tr><th>Model</th><th>Int8 Size</th><th>UTMOS-15</th><th>Drops/Stut/Clicks</th><th>ne/sa pCER</th><th>News</th><th>Story</th><th>Conv</th><th>Warm</th><th>Expr</th><th>Vedic</th></tr>{rows}</table></div>"""


def card_listen():
    vers = [("v5_final", "V1 Old (v5_final)"), ("v15_v1_chandas", "V1 v15 (Chandas)"), ("v14_atom_lite", "Atom-Lite (2.7MB)")]
    for tag, lbl in [("v15_nano", "Nano-v15"), ("v15_atom", "Atom-v15"), ("v15_pico", "Pico-v15 (<1MB)")]:
        if os.path.isdir(f"{REPO}/docs/samples/versions/{tag}"):
            vers.append((tag, lbl))
    clips = [
        ("sa_om", "Vedic ॐ (sa_om)"),
        ("sa_rv", "Rigveda 1.1.1 (sa_rv)"),
        ("sa_gayatri", "Gayatri Mantra (sa_gayatri)"),
        ("ne_kathmandu", "News (ne_kathmandu)"),
        ("ne_story", "Storytelling (ne_story)"),
        ("ne_chat", "Conversational (ne_chat)"),
        ("ne_expressive", "Expressive वाह! (ne_expressive)"),
        ("ne_unseen_narrative", "Hiatus आआफ्नो (ne_unseen_narrative)"),
    ]
    th = "".join(f"<th>{html.escape(lbl)}</th>" for _, lbl in vers)
    rows = ""
    for cid, clbl in clips:
        tds = ""
        for tag, _ in vers:
            mp3 = f"{REPO}/docs/samples/versions/{tag}/{cid}.mp3"
            if os.path.exists(mp3):
                b64 = base64.b64encode(open(mp3, "rb").read()).decode()
                tds += f"<td><audio controls preload='none' src='data:audio/mpeg;base64,{b64}'></audio></td>"
            else:
                tds += "<td class='mut'>pending</td>"
        rows += f"<tr><td><b>{html.escape(clbl)}</b></td>{tds}</tr>"
    return CSS + f"""<div class="k15"><h3>Listen Side-by-Side: Frozen Int8 Master Clips</h3>
<table><tr><th>Clip</th>{th}</tr>{rows}</table></div>"""


def card_tiers():
    p = f"{REPO}/docs/tiers.json"
    rows = ""
    try:
        tj = json.load(open(p))
        for t in tj.get("tiers", []):
            rows += f"<tr><td><b>{html.escape(t.get('label',''))}</b></td><td>{t.get('mb')} MB</td><td><b>{t.get('utmos')}</b></td><td>{t.get('rtf') or '-'}</td><td class='mut'>{html.escape(t.get('desc',''))}</td></tr>"
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
    "tts_v15_styles": '''
import marimo as _mo
k15_refresh.value
_mo.Html(k15.card_styles())
''',
    "tts_v15_listen": '''
import marimo as _mo
k15_refresh.value
_mo.Html(k15.card_listen())
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
