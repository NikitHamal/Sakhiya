"""Sakhiya-Pro live Marimo cards (2 cells). Install (idempotent, touches ONLY the tts_sakhiya_pro* cells) via marimo._code_mode:
   exec(open('/work/sp/repo/monitor/cells_pro.py').read()); await install()
Card 1 `tts_sakhiya_pro`        : live status — clock, hard-deadline countdown, GPU, running job + step/ETA, stage checklist,
                                  HF backup, eval leaderboard (synthesized-audio metrics).
Card 2 `tts_sakhiya_pro_audio`  : listen in the browser — base zero-shot vs fine-tune on the 8 hard sentences + 15-clip pack + EP01.
"""
CARD = r'''
import json, glob, os, re, subprocess, html, time, datetime
import marimo as mo
def _t(p, n=40):
    try: return open(p, errors="ignore").read().splitlines()[-n:]
    except Exception: return []
def _sh(c):
    try: return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=8).stdout.strip()
    except Exception: return ""
gpu = _sh("nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader")
npt = _sh("TZ=Asia/Kathmandu date '+%a %H:%M NPT'")
# hard stop: 06:25 NPT on 24 Nov (= 00:40 UTC); box expires ~06:55 NPT
now = datetime.datetime.now(datetime.timezone.utc); stop = datetime.datetime(2026, 11, 24, 0, 40, tzinfo=datetime.timezone.utc)
left = (stop - now).total_seconds(); lh, lm = int(left // 3600), int(left % 3600 // 60)
# ---- running job
logs = sorted(glob.glob("/work/sp/exp/*/train.log"), key=os.path.getmtime)
job, step_line, loss, sps, eta = "-", "-", "-", "-", "-"
if logs:
    lg = logs[-1]; job = lg.split("/")[-2]; ls = [l for l in _t(lg, 600) if re.search(r"Step \d+", l)]
    if ls:
        step_line = ls[-1][-160:]; m = re.search(r"Step (\d+)", ls[-1]); cur = int(m.group(1)) if m else 0
        ml = re.search(r"[Ll]oss[:= ]+([0-9.]+)", ls[-1]); loss = ml.group(1) if ml else "-"
        try:
            cfg = json.load(open(f"/work/sp/exp/{job}/train_cfg.json")); tot = cfg.get("steps", 0)
            dt = time.time() - os.path.getmtime(f"/work/sp/exp/{job}/train_cfg.json")
            if cur > 0 and dt > 0 and os.path.getmtime(lg) > time.time() - 120:
                sps = f"{cur/dt:.2f}"; eta = f"{(tot-cur)/(cur/dt)/60:.1f} min of {tot} steps"
        except Exception: pass
running = bool(_sh("pgrep -f 'omnivoice[.]cli[.]train'"))
render_run = bool(_sh("pgrep -f 'sakhiya_pro_render[.]py'"))
pip = _sh("ls /tmp/pipeline_state.txt >/dev/null 2>&1 && tail -n 6 /tmp/pipeline_state.txt")
# ---- stages
def ok(c): return "✅" if c else "⬜"
stages = [
 ("Deps + assets (HF, SLR43)", os.path.exists("/work/sp/models/OmniVoice")),
 ("Bake-off: OmniVoice vs Swar F5 (winner: OmniVoice-600M)", os.path.exists("/work/sp/bakeoff")),
 ("Single-voice dataset (SLR43 6834 + VC-unified)", os.path.exists("/work/sp/ft/dataset_report.json")),
 ("Fine-tune ft2 → ft2_s900 picked", os.path.exists("/work/sp/exp/ft2")),
 ("Gentle sweep ft3 (low LR, 100-step ckpts)", os.path.exists("/work/sp/exp/ft3/train.log")),
 ("Render pack + EP01", os.path.exists("/work/sp/render/ep01/voiceover_ep01.mp3")),
 ("Final dir published to HF", os.path.exists("/work/sp/final/published.txt")),
]
st = "".join(f"<li>{ok(b)} {html.escape(n)}</li>" for n, b in stages)
# ---- leaderboard
rows = []
for sj in sorted(glob.glob("/work/sp/evals/*/scores.json")):
    s = json.load(open(sj))["summary"]
    c = -3*s["cer"] - s["cer_vedic"] - s["cer_tech"] + 0.3*s["utmos"] + s["spk_sim"] - 0.5*s["hindi"] - 0.002*s["clicks"]
    rows.append((c, s))
rows.sort(key=lambda r: -r[0])
tb = "".join(f"<tr><td>{'🏆 ' if i==0 else ''}{s['tag']}</td><td>{c:.3f}</td><td>{s['cer']:.3f}</td><td>{s['cer_tech']:.3f}</td><td>{s['cer_vedic']:.3f}</td><td>{s['utmos']:.2f}</td><td>{s['spk_sim']:.2f}</td><td>{s['hindi']}</td><td>{s['clicks']:.0f}</td></tr>" for i, (c, s) in enumerate(rows))
bk = _t("/work/tts/logs/backup_daemon.py.log", 2)
col = "#c00" if left < 3600 else "#080"
mo.Html(f"""<div style="font-family:system-ui;font-size:14px;line-height:1.5">
<h3 style="margin:0">Sakhiya-Pro · OmniVoice-600M anchor teacher — LIVE · {npt}</h3>
<p><b style="color:{col}">⏱ {lh}h {lm}m until hard stop 06:25 NPT</b> · GPU <code>{html.escape(gpu)}</code> · training: <b>{'RUNNING '+job if running else 'idle'}</b> · render: <b>{'RUNNING' if render_run else 'idle'}</b></p>
<p>Step: <code>{html.escape(step_line)}</code><br>speed: <b>{sps}</b> steps/s · ETA: <b>{eta}</b> · loss: {loss}</p>
<ul style="list-style:none;padding-left:0">{st}</ul>
<p>HF backup (every 15 min → sakhiya-v1-train/sakhiya_pro): <code>{html.escape(' | '.join(bk) if bk else '-')}</code></p>
<table style="border-collapse:collapse" border="1" cellpadding="4"><tr><th>eval (audio-based)</th><th>score</th><th>CER</th><th>tech</th><th>vedic</th><th>UTMOS</th><th>spk-sim</th><th>hindi</th><th>clicks</th></tr>{tb}</table>
<p style="color:#666">Scores come from synthesized audio (Kriti ASR CER, UTMOS22, ECAPA similarity, click forensics) — not from training loss. {html.escape(pip)}</p></div>""")
'''
AUDIO = r'''
import os, glob, json
import marimo as mo
def _aud(p, label):
    try: return mo.vstack([mo.md(f"**{label}**"), mo.audio(open(p, "rb").read())])
    except Exception as e: return mo.md(f"{label}: n/a ({e})")
tags = [t for t in ("base_zeroshot", "ft2_s900", "ft3_best") if os.path.isdir(f"/work/sp/evals/{t}")]
ids = ["s1_anthropic_lat", "s2_gemini_lat", "s3_jensen_lat", "s4_gpt_lat", "s5_liquid_lat", "s6_convo_dev", "s7_conjunct_dev", "s8_vedic_dev"]
cmp_ = mo.accordion({i: mo.hstack([_aud(f"/work/sp/evals/{t}/{i}.wav", t) for t in tags]) for i in ids})
pack = sorted(glob.glob("/work/sp/render/pack/*.mp3"))
packv = mo.accordion({os.path.basename(p)[:-4]: mo.audio(open(p, "rb").read()) for p in pack}) if pack else mo.md("pack: n/a")
ep = "/work/sp/render/ep01/voiceover_ep01.mp3"
epv = mo.audio(open(ep, "rb").read()) if os.path.exists(ep) else mo.md("EP01: n/a")
mo.vstack([mo.md("### 🔊 Sakhiya-Pro listening desk"), mo.md("**A/B: zero-shot base vs fine-tune (same sentence, same seed)**"), cmp_,
           mo.md("**15-clip master pack (Best-of-3, mastered -14 LUFS)**"), packv, mo.md("**NEBians News Episode 01 voiceover**"), epv])
'''
async def install():
    import marimo._code_mode as cm
    async with cm.get_context(skip_staleness_check=True) as ctx:
        ex = {c.name: c.id for c in ctx.cells.values()}
        for name, code in (('tts_sakhiya_pro', CARD), ('tts_sakhiya_pro_audio', AUDIO)):
            if name in ex: ctx.edit_cell(ex[name], code.strip(), hide_code=True); cid = ex[name]
            else: cid = ctx.create_cell(code.strip(), name=name, hide_code=True)
            ctx.run_cell(cid); print('installed', name)
