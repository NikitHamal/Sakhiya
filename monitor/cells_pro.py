"""Sakhiya-Pro Marimo monitor card. Install (idempotent, touches ONLY the tts_sakhiya_pro cell) via marimo._code_mode:
   exec(open('/work/sp/repo/monitor/cells_pro.py').read()); await install()"""
CARD = r'''
import json, glob, os, re, subprocess, html
import marimo as mo
def _t(p, n=40):
    try: return open(p, errors="ignore").read().splitlines()[-n:]
    except Exception: return []
def _sh(c):
    try: return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=8).stdout.strip()
    except Exception: return ""
gpu = _sh("nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader")
npt = _sh("TZ=Asia/Kathmandu date '+%H:%M NPT'")
rows = []
for sj in sorted(glob.glob("/work/sp/evals/*/scores.json")):
    s = json.load(open(sj))["summary"]
    rows.append(f"<tr><td>{s['tag']}</td><td>{s['cer']:.3f}</td><td>{s['cer_tech']:.3f}</td><td>{s['cer_vedic']:.3f}</td><td>{s['utmos']:.2f}</td><td>{s['spk_sim']:.2f}</td><td>{s['hindi']}</td></tr>")
tr = [l for l in _t("/work/sp/exp/ft1/train.log", 400) if "Step" in l][-1:] if os.path.exists("/work/sp/exp/ft1/train.log") else []
run = "RUNNING" if _sh("pgrep -f 'omnivoice[.]cli[.]train'") else "idle"
bk = _t("/work/tts/logs/backup_daemon.py.log", 1)
mo.Html(f"""<div style="font-family:system-ui;font-size:14px"><h3>Sakhiya-Pro (OmniVoice-600M teacher) · {npt}</h3>
<p>GPU {gpu} · training: <b>{run}</b> · last: {html.escape(tr[0] if tr else '-')} · backup: {html.escape(bk[0] if bk else '-')}</p>
<table style="border-collapse:collapse"><tr><th>eval</th><th>CER</th><th>tech</th><th>vedic</th><th>UTMOS</th><th>spk</th><th>hindi</th></tr>{''.join(rows)}</table></div>""")
'''
async def install():
    import marimo._code_mode as cm
    async with cm.get_context(skip_staleness_check=True) as ctx:
        ex = {c.name: c.id for c in ctx.cells.values()}
        if 'tts_sakhiya_pro' in ex: ctx.edit_cell(ex['tts_sakhiya_pro'], CARD.strip(), hide_code=True); cid = ex['tts_sakhiya_pro']
        else: cid = ctx.create_cell(CARD.strip(), name='tts_sakhiya_pro', hide_code=True)
        ctx.run_cell(cid); print('installed tts_sakhiya_pro')
