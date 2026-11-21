"""Sakhiya v11 cards: the V1 fine-tune on mined data (live). Works on both boxes: on the mining box it reads the run's own
log; on the training box it reads the copy the mining box mirrors every minute (/work/tts/v11_mirror). Read-only, never raises."""
import ast, glob, html, json, os, re, time
R = "/work/tts"
SRC = [(f"{R}/logs/v11_ft.log", f"{R}/v11/v11_mined_24k_report.json", f"{R}/logs/v11_mine_chain.log"),
       (f"{R}/v11_mirror/v11_ft.log", f"{R}/v11_mirror/v11_mined_24k_report.json", f"{R}/v11_mirror/v11_mine_chain.log")]
CAP_MIN = 60.0
CSS = """<style>.s11{font-family:system-ui,sans-serif;font-size:14px;color:#1f2937}.s11 h3{margin:.2em 0 .5em;font-size:17px}
.s11 table{border-collapse:collapse;width:100%}.s11 td,.s11 th{border-bottom:1px solid #e5e7eb;padding:5px 8px;text-align:left}
.s11 th{background:#f9fafb;font-weight:600}.s11 .bar{background:#eef2ff;border-radius:6px;height:10px;position:relative}
.s11 .bar>i{position:absolute;left:0;top:0;bottom:0;border-radius:6px;background:#6366f1}.s11 .up{color:#047857;font-weight:600}
.s11 .dn{color:#b91c1c;font-weight:600}.s11 .mut{color:#6b7280}.s11 .pill{display:inline-block;padding:1px 8px;border-radius:999px;background:#f3f4f6;margin-right:4px}
.s11 .run{background:#dcfce7}.s11 .ok{background:#dcfce7}.s11 .no{background:#fee2e2}.s11 audio{height:32px;width:250px}</style>"""
LABEL = {"cfm": "Mel error (main)", "dur_l1": "Timing error", "f0": "Pitch error", "prior": "Prior error"}


def _paths():
    for lg, rp, ch in SRC:
        if os.path.exists(lg): return lg, rp, ch
    return SRC[1]


def _lines(p):
    try:
        with open(p, "rb") as f:
            f.seek(0, 2); n = f.tell(); f.seek(max(0, n - 400000)); return f.read().decode("utf8", "ignore").splitlines()
    except Exception: return []


def _dicts(l):
    out = {}
    for k in ("val_sakhiya", "val_mined"):
        m = re.search(k + r"=(\{[^}]*\})", l)
        if m:
            try: out[k] = ast.literal_eval(m.group(1))
            except Exception: pass
    return out


def parse():
    lg, rp, ch = _paths(); L = _lines(lg)
    st = dict(start=None, scores=[], last=None, gate=None, done=None, log=lg, age=None)
    try: st["age"] = time.time() - os.path.getmtime(lg)
    except Exception: pass
    for l in L:
        if "V1 start" in l: st["start"] = _dicts(l)
        elif "SCORE step=" in l:
            d = _dicts(l); m = re.search(r"step=(\d+)", l)
            if m and d: d["step"] = int(m.group(1)); st["scores"].append(d)
        elif "[GATE]" in l: st["gate"] = l.strip()
        elif "[v11-ft] DONE" in l: st["done"] = l.strip()
        elif "[v11-ft] step=" in l:
            m = re.search(r"step=(\d+) ([\d.]+) st/s t=([\d.]+)m", l)
            if m: st["last"] = dict(step=int(m.group(1)), sps=float(m.group(2)), t=float(m.group(3)))
    try: st["report"] = json.load(open(rp))
    except Exception: st["report"] = None
    st["chain"] = [l for l in _lines(ch) if l.strip() and "setlocale" not in l][-8:]
    return st


def _bar(f, c="#6366f1"):
    f = max(0.0, min(1.0, f)); return f'<div class="bar"><i style="width:{f*100:.0f}%;background:{c}"></i></div>'


def _spark(vals, w=220, h=40, color="#6366f1"):
    if len(vals) < 2: return ""
    lo, hi = min(vals), max(vals); rng = (hi - lo) or 1e-6
    pts = " ".join(f"{i*(w-4)/(len(vals)-1)+2:.1f},{h-2-(v-lo)/rng*(h-4):.1f}" for i, v in enumerate(vals))
    return f'<svg width="{w}" height="{h}"><polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/></svg>'


def card_header():
    s = parse(); last = s["last"] or {}
    if s["done"]: state = '<span class="pill ok">finished</span>'
    elif s["age"] is not None and s["age"] < 120: state = '<span class="pill run">training</span>'
    else: state = '<span class="pill">waiting / not started</span>'
    t = last.get("t", 0.0)
    where = "mining box (live log)" if s["log"].startswith(f"{R}/logs") else "mirrored from the mining box every minute"
    return CSS + f"""<div class="s11"><h3>Sakhiya-V1 fine-tune on the new mined data {state}</h3>
<div class="mut">V1's acoustic model, trained further on AssemblyAI/Gemini-transcribed Nepali and verse-matched Vedic. Other voices sit in
speaker slots 1-15; Sakhiya stays on slot 0, and half of every batch is original Sakhiya data. Capped at {CAP_MIN:.0f} min. Source: {where}.</div>
<div style="margin:8px 0 2px">Step <b>{last.get('step', 0):,}</b> · {last.get('sps', 0):.1f} steps/s · {t:.1f} / {CAP_MIN:.0f} min</div>{_bar(t / CAP_MIN)}
<div class="mut" style="margin-top:4px">{html.escape(s['gate'] or 'speed check pending')}</div></div>"""


def card_progress():
    s = parse(); v0 = s["start"] or {}; sc = s["scores"]
    if not v0 or not sc: return CSS + '<div class="s11"><h3>Is it better than V1?</h3><span class="mut">No validation scores yet (first one at step 1,000).</span></div>'
    cur = sc[-1]; rows = []
    for grp, title in (("val_sakhiya", "Sakhiya's own held-out clips (the voice we ship)"), ("val_mined", "Held-out mined expressive speech")):
        rows.append(f"<tr><th colspan=5>{title}</th></tr>")
        for k in ("cfm", "dur_l1", "f0", "prior"):
            a, b = v0.get(grp, {}).get(k), cur.get(grp, {}).get(k)
            if a is None or b is None: continue
            d = (b - a) / a * 100 if a else 0.0; cls = "up" if d < -0.5 else ("dn" if d > 0.5 else "mut")
            word = "better" if d < -0.5 else ("worse" if d > 0.5 else "same")
            hist = [x.get(grp, {}).get(k) for x in sc if x.get(grp, {}).get(k) is not None]
            rows.append(f"<tr><td>{LABEL[k]}</td><td>{a:.4f}</td><td>{b:.4f}</td><td class='{cls}'>{abs(d):.1f}% {word}</td><td>{_spark([a] + hist)}</td></tr>")
    return CSS + f"""<div class="s11"><h3>Is it better than V1? (lower is better · step {cur['step']:,})</h3>
<table><tr><th>Measure</th><th>V1</th><th>Now</th><th>Change</th><th>Trend</th></tr>{''.join(rows)}</table>
<div class="mut" style="margin-top:6px">These are training-side errors on clips the model never trained on. They show direction, not sound quality:
the ship decision comes only from the real audio gate after export.</div></div>"""


def card_data():
    r = (parse().get("report")) or {}
    if not r: return CSS + '<div class="s11"><h3>Training data</h3><span class="mut">Pack report not available here yet.</span></div>'
    hrs = r.get("hours", {}); tot = sum(hrs.values())
    hr = "".join(f"<tr><td>{k}</td><td>{v:.2f} h</td><td>{_bar(v / max(hrs.values()))}</td></tr>" for k, v in sorted(hrs.items(), key=lambda x: -x[1]))
    ts = " ".join(f'<span class="pill">{k}: {v:,}</span>' for k, v in r.get("text_source", {}).items())
    rj = " ".join(f'<span class="pill">{k.replace("_", " ")}: {v:,}</span>' for k, v in sorted(r.get("rejected", {}).items(), key=lambda x: -x[1]))
    return CSS + f"""<div class="s11"><h3>New training data: {r.get('items', 0):,} clips · {tot:.1f} h · {r.get('speakers', 0)} speakers</h3>
<table>{hr}</table><div style="margin-top:8px"><b>Transcripts</b> {ts}</div>
<div style="margin-top:6px" class="mut"><b>Rejected by the filters</b> {rj}</div></div>"""


def card_gate():
    s = parse(); chain = "".join(f"<div><code style='font-size:12px'>{html.escape(l[-200:])}</code></div>" for l in s["chain"]) or "<span class='mut'>waiting for the fine-tune to finish</span>"
    gl = [l for l in _lines(f"{R}/logs/v11_gate_chain.log") if l.strip()][-6:]
    gate = "".join(f"<div><code style='font-size:12px'>{html.escape(l[-220:])}</code></div>" for l in gl)
    dec = ""
    for p in glob.glob(f"{R}/repo9/eval/v11_v1_decision.json"):
        try:
            d = json.load(open(p)); v1, c = d["v1"], d["candidate"]
            ck = " ".join(f'<span class="pill {"ok" if v else "no"}">{k.replace("_", " ")}</span>' for k, v in d["checks"].items())
            st = sorted(set(v1.get("styles", {})) | set(c.get("styles", {})))
            sr = "".join(f"<tr><td>{k}</td><td>{v1['styles'].get(k, '-')}</td><td>{c['styles'].get(k, '-')}</td></tr>" for k in st)
            dec = (f"<h3 style='margin-top:10px'>Decision: {'SHIPPED' if d['ship'] else 'not shipped'} ({d['pick']} checkpoint)</h3><div>{ck}</div>"
                   f"<table style='margin-top:6px'><tr><th></th><th>V1</th><th>Fine-tuned</th></tr>"
                   f"<tr><td><b>Gate score</b></td><td>{v1['score']}</td><td>{c['score']}</td></tr><tr><td>UTMOS</td><td>{v1['utmos']}</td><td>{c['utmos']}</td></tr>"
                   f"<tr><td>Word drops / stutters / clicks</td><td>{v1['drops']} / {v1['stutters']} / {v1['clicks']}</td><td>{c['drops']} / {c['stutters']} / {c['clicks']}</td></tr>"
                   f"<tr><td>Voice match</td><td>{v1['sv_cos']}</td><td>{c['sv_cos']}</td></tr>{sr}</table>")
        except Exception as e:
            dec = f"<span class='mut'>decision file unreadable: {html.escape(str(e))}</span>"
    return CSS + f"""<div class="s11"><h3>After training: hand-off, audio gate, ship decision</h3>
<div class="mut">Ships only if the gate score rises by 0.03+, no extra word drops or stutters, the Sakhiya voice match holds, storytelling and
Vedic get no worse and at least one improves by 0.05+.</div>
<div style="margin-top:6px"><b>Mining box</b>{chain}</div>{f'<div style="margin-top:6px"><b>Training box gate</b>{gate}</div>' if gate else ''}{dec}</div>"""


def card_listen():
    w = sorted(glob.glob(f"{R}/v10/samples/v11_*.wav"))
    if not w: return CSS + '<div class="s11"><h3>Listen: V1 vs fine-tuned</h3><span class="mut">Rendered on the training box after the gate (same sentences, int8 graphs the site serves).</span></div>'
    import base64
    by = {}
    for p in w:
        sid, name = os.path.basename(p)[:-4].split("__"); by.setdefault(sid, {})[name] = p
    rows = []
    for sid, d in sorted(by.items()):
        cells = "".join(f"<td>{n}<br><audio controls preload='none' src='data:audio/wav;base64,{base64.b64encode(open(d[n], 'rb').read()).decode()}'></audio></td>" for n in ("V1", "V1-v11") if n in d)
        rows.append(f"<tr><td>{html.escape(sid[4:])}</td>{cells}</tr>")
    return CSS + '<div class="s11"><h3>Listen: V1 vs fine-tuned (same sentence)</h3><table>' + "".join(rows) + "</table></div>"
