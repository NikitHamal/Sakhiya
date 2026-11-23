"""Sakhiya-Pro v2 + tiny-tier live Marimo cards. Idempotent install (touches ONLY tts_sakhiya_pro2* cells):
   exec(open('/work/sp/repo/monitor/cells_pro2.py').read()); await install()
State is read from files so any job can update the cards just by writing them:
   /work/sp/stage/<name>.json   {"label","state":"todo|run|done","note"}   -> stage checklist (sorted by name)
   /work/sp/tiers/leaderboard.json  [{"name","size_mb","utmos","cer","spk","rtf","status"}]  -> size-vs-quality table
   /work/sp/listen/<group>/*.mp3|wav -> listening desk (one accordion per group)
"""
CARD = r'''
sp2_tick.value
import json, glob, os, re, subprocess, html, time, datetime
import marimo as mo
def _t(p, n=40):
    try: return open(p, errors="ignore").read().splitlines()[-n:]
    except Exception: return []
def _sh(c):
    try: return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=8).stdout.strip()
    except Exception: return ""
gpu = _sh("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader")
npt = _sh("TZ=Asia/Kathmandu date '+%a %H:%M NPT'")
now = datetime.datetime.now(datetime.timezone.utc); stop = datetime.datetime.fromisoformat((open('/work/sp/deadline_utc.txt').read().strip() if os.path.exists('/work/sp/deadline_utc.txt') else '2026-11-24T04:30:00') ).replace(tzinfo=datetime.timezone.utc)
left = (stop - now).total_seconds(); lh, lm = int(left // 3600), int(left % 3600 // 60)
logs = sorted(glob.glob("/work/tts/logs/train_*.log"), key=os.path.getmtime)
job, step_line, loss, sps, eta = "-", "-", "-", "-", "-"
if logs:
    lg = logs[-1]; job = os.path.basename(lg)[6:-4]; ls = [l for l in _t(lg, 400) if re.search(r"Step \d+", l)]
    if ls:
        step_line = ls[-1][:170]; m = re.search(r"Step (\d+)", ls[-1]); cur = int(m.group(1)) if m else 0
        ml = re.search(r"loss[^0-9]*([0-9.]+)", ls[-1]); loss = ml.group(1) if ml else "-"
        try:
            tot = 0
            for cf in (f"/work/sp/exp/{job}/train_cfg.json", f"/work/sp/student/{job}/cfg.json"):
                if os.path.exists(cf): tot = json.load(open(cf)).get("steps", 0)
            msp = re.search(r"([0-9.]+) *(?:steps/s|it/s|steps_per_sec)", ls[-1]) or re.search(r"steps_per_sec: ([0-9.]+)", ls[-1]); v = float(msp.group(1)) if msp else 0
            if v > 0 and os.path.getmtime(lg) > time.time() - 180: sps = f"{v:.2f}"; eta = f"{(tot-cur)/v/60:.1f} min left ({cur}/{tot})" if tot else f"step {cur}"
        except Exception: pass
running = _sh("pgrep -fa 'omnivoice[.]cli[.]train|train_micro|train_pico|distill' | head -2")
stages = []
for p in sorted(glob.glob("/work/sp/stage/*.json")):
    try: stages.append(json.load(open(p)))
    except Exception: pass
ic = {"done": "✅", "run": "🔄", "todo": "⬜", "fail": "❌"}
st = "".join(f"<li>{ic.get(s.get('state'), '⬜')} <b>{html.escape(str(s.get('label','')))}</b> <span style='color:#666'>{html.escape(str(s.get('note','')))}</span></li>" for s in stages)
try: lb = json.load(open("/work/sp/tiers/leaderboard.json"))
except Exception: lb = []
tb = "".join(f"<tr><td>{html.escape(str(r.get('name')))}</td><td align=right>{r.get('size_mb','-')}</td><td align=right>{r.get('utmos','-')}</td><td align=right>{r.get('cer','-')}</td><td align=right>{r.get('spk','-')}</td><td align=right>{r.get('rtf','-')}</td><td>{html.escape(str(r.get('status','')))}</td></tr>" for r in lb)
lad = []
for t in ("t30", "t15", "t10", "t5", "t1"):
    lg = f"/work/sp/distill/runs/{t}/train.log"
    ls_ = [l for l in _t(lg, 30) if l.startswith("Step ")]
    if ls_:
        mm = re.search(r"Step (\d+) \| ([0-9.]+) it/s .*?\| ([0-9.]+)/([0-9.]+) min", ls_[-1])
        fin = os.path.exists(f"/work/sp/distill/runs/{t}/ema_final.pt")
        if mm: lad.append(f"<tr><td>{t}</td><td align=right>{mm.group(1)}</td><td align=right>{mm.group(2)}</td><td align=right>{mm.group(3)} / {mm.group(4)} min</td><td>{'✅ finished' if fin else ('🔄 training' if os.path.getmtime(lg) > time.time() - 180 else '⏸ stopped')}</td><td><code>{html.escape(ls_[-1].split('|')[2][:110])}</code></td></tr>")
ladtb = "<table style='border-collapse:collapse' border=1 cellpadding=4><tr><th>student</th><th>step</th><th>it/s</th><th>progress</th><th>state</th><th>losses</th></tr>" + "".join(lad) + "</table>" if lad else ""
bk = _t("/work/tts/logs/backup_daemon_v2.log", 1)
col = "#c00" if left < 3600 else "#080"
mo.Html(f"""<div style="font-family:system-ui;font-size:14px;line-height:1.5">
<h3 style="margin:0">Sakhiya-Pro v2 → Micro / Pico — LIVE · {npt}</h3>
<p><b style="color:{col}">⏱ {lh}h {lm}m until hard stop (final pushes done by then; box expiry is later)</b> · GPU util/mem/total/temp <code>{html.escape(gpu)}</code></p>
<p>job: <b>{html.escape(job)}</b> {'(running)' if running else '(idle)'}<br>Step: <code>{html.escape(step_line)}</code><br>speed: <b>{sps}</b> steps/s · ETA: <b>{eta}</b> · loss: {loss}</p>
<ul style="list-style:none;padding-left:0">{st}</ul>{ladtb}
<table style="border-collapse:collapse" border="1" cellpadding="4"><tr><th>tier</th><th>INT8 size MB</th><th>UTMOS</th><th>CER</th><th>spk-match</th><th>1-thr RTF</th><th>status</th></tr>{tb}</table>
<p>HF backup (15 min → sakhiya-v1-train/sakhiya_pro_v2): <code>{html.escape(' | '.join(bk) if bk else '-')}</code></p>
<p style="color:#666">Scores come from synthesized / exported-INT8 audio (ASR CER, UTMOS22, speaker similarity, click forensics), never training loss.</p></div>""")
'''
AUDIO = r'''
sp2_tick.value
import os, glob
import marimo as mo
def _aud(p):
    try: return mo.audio(open(p, "rb").read())
    except Exception as e: return mo.md(f"n/a ({e})")
groups = {}
for d in sorted(glob.glob("/work/sp/listen/*/")):
    fs = sorted(glob.glob(d + "*.mp3") + glob.glob(d + "*.wav"))[:24]
    if fs: groups[os.path.basename(d.rstrip("/"))] = mo.accordion({os.path.basename(f).rsplit(".", 1)[0]: _aud(f) for f in fs})
mo.vstack([mo.md("### 🔊 Sakhiya listening desk (auto-updated after every stage)")] + ([mo.accordion(groups)] if groups else [mo.md("_no clips yet_")]))
'''
TICK = '''import marimo as mo
sp2_tick = mo.ui.refresh(options=["15s", "30s", "1m"], default_interval="30s", label="Sakhiya v2 live refresh")
sp2_tick'''
def _wrap(code):
    import textwrap, re as _re
    body = textwrap.indent(code.strip(), '    ')
    body = _re.sub(r'^    (mo\.Html\(|mo\.vstack\(\[mo\.md\(")', lambda m: '    return ' + m.group(1), body, flags=_re.M)
    return 'def _main():\n' + body + '\n_main()'
async def install():
    import marimo._code_mode as cm
    async with cm.get_context(skip_staleness_check=True) as ctx:
        ex = {c.name: c.id for c in ctx.cells.values()}
        for name, code in (('tts_sakhiya_pro2_tick', TICK), ('tts_sakhiya_pro2', CARD), ('tts_sakhiya_pro2_audio', AUDIO)):
            code = code.strip() if name.endswith('_tick') else _wrap(code)
            if name in ex: ctx.edit_cell(ex[name], code, hide_code=True); cid = ex[name]
            else: cid = ctx.create_cell(code, name=name, hide_code=True)
            ctx.run_cell(cid); print('installed', name)
