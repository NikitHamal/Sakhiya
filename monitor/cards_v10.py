"""Sakhiya v10 MoLab cards: live view of the small-model breakthrough work (sizes vs the 3 MB goal, live runs, real-gate
quality, listen A/B). Read-only: every card only reads logs/json/wav files on the box; nothing here starts or stops jobs."""
import base64, glob, html, json, os, re, subprocess, time
R = "/work/tts"; REPO = f"{R}/repo9"; LOGS = f"{R}/logs"; TARGET_MB = 3.0
CSS = """<style>.sk{font-family:system-ui,sans-serif;font-size:14px;color:#1f2937}.sk h3{margin:.2em 0 .5em;font-size:17px}
.sk table{border-collapse:collapse;width:100%}.sk td,.sk th{border-bottom:1px solid #e5e7eb;padding:5px 8px;text-align:left}
.sk th{background:#f9fafb;font-weight:600}.sk .bar{background:#eef2ff;border-radius:6px;height:10px;position:relative;min-width:120px}
.sk .bar>i{position:absolute;left:0;top:0;bottom:0;border-radius:6px;background:#6366f1}.sk .ok{color:#047857;font-weight:600}
.sk .bad{color:#b91c1c;font-weight:600}.sk .mut{color:#6b7280}.sk .pill{display:inline-block;padding:1px 8px;border-radius:999px;background:#f3f4f6;margin-right:4px}
.sk .run{background:#dcfce7}.sk .idle{background:#f3f4f6}.sk audio{height:32px;width:260px}</style>"""


def _sh(c):
    try: return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=8).stdout
    except Exception: return ""


def _j(p):
    try: return json.load(open(p))
    except Exception: return None


def _tail(p, n=400):
    try:
        with open(p, "rb") as f:
            f.seek(0, 2); s = f.tell(); f.seek(max(0, s - 60000)); return f.read().decode("utf8", "ignore").splitlines()[-n:]
    except Exception: return []


def _bar(frac, color="#6366f1"):
    frac = max(0.0, min(1.0, frac)); return f'<div class="bar"><i style="width:{frac*100:.0f}%;background:{color}"></i></div>'


def header():
    g = _sh("nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader").strip()
    return CSS + f"""<div class="sk"><h3>Sakhiya v10 · small-model breakthrough</h3>
<div class="mut">Goal: Nano &amp; Atom with a clean, natural voice, and Atom under {TARGET_MB:.0f} MB. Each stage is capped under 2 h,
with a step-50 speed check and a real forensic gate before anything ships.</div>
<div style="margin-top:6px"><span class="pill">GPU: {html.escape(g or 'n/a')}</span><span class="pill">box time {time.strftime('%H:%M:%S', time.gmtime())} UTC</span></div></div>"""


JOBS = [("v9 Nano acoustic (cached teacher)", "v9_voca.log", "v9_am_cached.py --tier voca"),
        ("v9 Atom acoustic (cached teacher)", "v9_sano.log", "v9_am_cached.py --tier sano"),
        ("v10 student vocoder (dense distill)", "v10_voc.log", "v10_voc_distill.py"),
        ("v9/v10 ship chain", "v9_chain.log", "v9_chain.sh|v10_chain.sh")]


def card_jobs():
    ps = _sh("ps -eo args")
    rows = []
    for name, log, pat in JOBS:
        lines = _tail(f"{LOGS}/{log}")
        running = any(re.search(p, ps) for p in pat.split("|"))
        last = next((l for l in reversed(lines) if "step=" in l or "SCORE" in l or "DONE" in l or "[chain]" in l or "SHIPPED" in l), lines[-1] if lines else "no log yet")
        gate = next((l for l in lines if "[GATE]" in l), "")
        m = re.search(r"elapsed=([\d.]+)m|t=([\d.]+)m", last); el = float(next(x for x in m.groups() if x)) if m else 0.0
        cap = 100.0 if "v9_" in log and "chain" not in log else 80.0
        st = '<span class="pill run">running</span>' if running else '<span class="pill idle">idle</span>'
        rows.append(f"<tr><td><b>{name}</b><br>{st}</td><td>{_bar(el / cap) if el else ''}<span class='mut'>{el:.0f} / {cap:.0f} min</span></td>"
                    f"<td><code style='font-size:12px'>{html.escape(last[-170:])}</code><br><span class='mut'>{html.escape(gate[-110:])}</span></td></tr>")
    return CSS + '<div class="sk"><h3>Live runs</h3><table><tr><th>Job</th><th>Time vs cap</th><th>Latest line</th></tr>' + "".join(rows) + "</table></div>"


def _sizes(d):
    if not os.path.isdir(d): return None
    f = {k: os.path.getsize(os.path.join(d, k)) / 1e6 for k in os.listdir(d) if k.endswith(".int8.onnx")}
    if not f: return None
    am = sum(v for k, v in f.items() if not k.startswith("vocoder")); voc = sum(v for k, v in f.items() if k.startswith("vocoder"))
    return am, voc, am + voc


def card_size():
    rows = []
    cands = [("Sakhiya-V1 (shipped)", f"{REPO}/docs/models_v2"), ("Nano (shipped)", f"{REPO}/docs/models_voca"), ("Atom (shipped)", f"{REPO}/docs/models_sano")]
    for d in sorted(glob.glob(f"{R}/v10/stage_*")): cands.append((f"candidate {os.path.basename(d)[6:]}", d))
    for name, d in cands:
        s = _sizes(d)
        if not s: continue
        am, voc, tot = s; ok = tot < TARGET_MB
        rows.append(f"<tr><td>{name}</td><td>{am:.2f}</td><td>{voc:.2f}</td><td class='{'ok' if ok else 'bad'}'>{tot:.2f} MB</td><td>{_bar(tot / 16.0, '#10b981' if ok else '#f59e0b')}</td></tr>")
    return CSS + f"""<div class="sk"><h3>Size vs the {TARGET_MB:.0f} MB goal (int8, what the site downloads)</h3>
<table><tr><th>Model</th><th>Acoustic MB</th><th>Vocoder MB</th><th>Total</th><th></th></tr>{''.join(rows)}</table>
<div class="mut" style="margin-top:6px">The V1 vocoder is 5.04 MB in every shipped tier, so no acoustic-model change alone can get under 3 MB. v10 distils one shared 1.2 M-param vocoder from it.</div></div>"""


def _score(f):
    return (f.get("utmos_15") or 0) - 0.5 * (f.get("word_drop_count") or 0) - 0.3 * (f.get("stutter_count") or 0) - 0.02 * (f.get("clicks") or 0) / 10.0


def card_quality():
    rows = []
    for name, p in [("Nano shipped (v8 gate)", f"{REPO}/eval/forensic_v8_nano.json"), ("Atom shipped (v8 gate)", f"{REPO}/eval/forensic_v8_atom.json"),
                    ("Nano v9 gate", f"{R}/v9/gate_voca.json"), ("Atom v9 gate", f"{R}/v9/gate_sano.json")] + \
                   [(f"v10 {os.path.basename(p)[5:-5]}", p) for p in sorted(glob.glob(f"{R}/v10/gate_*.json"))]:
        f = _j(p)
        if not f: continue
        rows.append(f"<tr><td>{name}</td><td>{f.get('utmos_15')}</td><td>{f.get('word_drop_count')}</td><td>{f.get('stutter_count')}</td><td>{f.get('clicks')}</td><td><b>{_score(f):.3f}</b></td></tr>")
    sc = [l for l in _tail(f"{LOGS}/v10_voc.log") if "SCORE" in l or "TEACHER" in l][-6:]
    v10 = "".join(f"<div><code style='font-size:12px'>{html.escape(l[-200:])}</code></div>" for l in sc) or "<span class='mut'>no v10 vocoder scores yet</span>"
    return CSS + f"""<div class="sk"><h3>Quality (real forensic gate: UTMOS minus word drops, stutters and clicks)</h3>
<table><tr><th>Run</th><th>UTMOS</th><th>Drops</th><th>Stutters</th><th>Clicks</th><th>Score</th></tr>{''.join(rows)}</table>
<h3 style="margin-top:12px">v10 vocoder on the real text → acoustic → vocoder path (UTMOS)</h3>{v10}</div>"""


def card_listen():
    wavs = sorted(glob.glob(f"{R}/v10/samples/*.wav"))[:12]
    if not wavs: return CSS + '<div class="sk"><h3>Listen</h3><span class="mut">Samples appear here after the v10 export renders them (teacher vocoder vs. new small vocoder, same sentence).</span></div>'
    items = []
    for w in wavs:
        b = base64.b64encode(open(w, "rb").read()).decode()
        items.append(f"<tr><td>{html.escape(os.path.basename(w)[:-4])}</td><td><audio controls preload='none' src='data:audio/wav;base64,{b}'></audio></td></tr>")
    return CSS + '<div class="sk"><h3>Listen (same sentences)</h3><table>' + "".join(items) + "</table></div>"


def card_events():
    lines = []
    for log in ("v9_chain.log", "v10_chain.log"):
        lines += [f"{log}: {l}" for l in _tail(f"{LOGS}/{log}", 12) if l.strip() and "Warning" not in l]
    body = "".join(f"<div><code style='font-size:12px'>{html.escape(l[-220:])}</code></div>" for l in lines[-18:]) or "<span class='mut'>no events yet</span>"
    return CSS + f'<div class="sk"><h3>Ship / gate events</h3>{body}</div>'
