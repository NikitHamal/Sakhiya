"""Sakhiya-Pro v1/v2 + Distillation Ladder 5-Panel Live Marimo Dashboard.
Idempotent install via marimo._code_mode (touches ONLY tts_sakhiya_pro2* cells):
   exec(open('/work/sp/repo/monitor/cells_pro2.py').read()); await install()

Five crisp auto-refreshing panels:
1. Real-time Box Health & Countdown Timer to 06:25 NPT cutoff (+ GPU, RAM, Disk, 15-min HF Backup status)
2. Live Teacher Corpus Generation & Single-Voice Cache Progress (clips, hours, clips/min, ETA, sources)
3. Live Distillation Ladder Training Monitor (active tier, step, steps/sec, ETA, QAT, best exported INT8 gate score)
4. Apples-to-Apples Size-vs-Quality Leaderboard comparing live site tiers (V1 15.98 MB, Nano 8.93 MB, Atom 6.51 MB,
   Atom-Lite 2.72 MB) and Sakhiya-Pro v1/v2 against ladder students (29.6, 14.5, 9.5, 4.8, 1.05 MB) on UTMOS, CER,
   drops, stutters, clicks, and 1-thread CPU RTF
5. Interactive Listening Desk auto-discovering latest exported INT8 WAVs/MP3s from every tier and Episode 01 with 0 errors
"""

TICK = '''import marimo as mo
sp2_tick = mo.ui.refresh(options=["10s", "15s", "30s", "1m"], default_interval="15s", label="Sakhiya Live Auto-Refresh")
sp2_tick'''

CARD = r'''
sp2_tick.value
import json, glob, os, re, subprocess, html, time, datetime
import marimo as mo

def _t(p, n=40):
    try:
        return open(p, errors="ignore").read().splitlines()[-n:]
    except Exception:
        return []

def _sh(c):
    try:
        return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=6).stdout.strip()
    except Exception:
        return ""

def _j(p, default=None):
    try:
        if os.path.exists(p):
            return json.load(open(p, errors="ignore"))
    except Exception:
        pass
    return default

# --- Panel 1: Real-time Box Health & Countdown to 06:25 NPT Cutoff ---
gpu = _sh("nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader")
npt = _sh("TZ=Asia/Kathmandu date '+%a %d %b %Y · %H:%M:%S NPT'")
utc_now = datetime.datetime.now(datetime.timezone.utc)
# 06:25 NPT on 24 Nov = 00:40 UTC (plus deadline_utc.txt if extended)
cutoff_0625 = datetime.datetime(2026, 11, 24, 0, 40, tzinfo=datetime.timezone.utc)
box_exp_0655 = datetime.datetime(2026, 11, 24, 1, 10, tzinfo=datetime.timezone.utc)
dl_str = open('/work/sp/deadline_utc.txt').read().strip() if os.path.exists('/work/sp/deadline_utc.txt') else '2026-11-24T04:30:00'
try:
    dl_utc = datetime.datetime.fromisoformat(dl_str.rstrip('Z')).replace(tzinfo=datetime.timezone.utc)
except Exception:
    dl_utc = cutoff_0625
target_stop = cutoff_0625 if utc_now < cutoff_0625 else dl_utc
left_s = max(0.0, (target_stop - utc_now).total_seconds())
lh, lm, ls_sec = int(left_s // 3600), int((left_s % 3600) // 60), int(left_s % 60)
exp_s = max(0.0, (box_exp_0655 - utc_now).total_seconds())
eh, em = int(exp_s // 3600), int((exp_s % 3600) // 60)

ram_info = _sh("free -h | awk '/Mem:/ {print $3 \" / \" $2}'")
disk_info = _sh("df -h /work 2>/dev/null | awk 'NR==2 {print $3 \" used / \" $4 \" avail\"}' || df -h / | awk 'NR==2 {print $3 \" / \" $4}'")
bk_lines = _t("/work/tts/logs/backup_daemon_v2.log", 2)
bk_alive = bool(_sh("pgrep -f '[b]ackup_daemon.py'"))
bk_status = ("🟢 ACTIVE · " if bk_alive else "🔴 STOPPED · ") + (" | ".join(bk_lines) if bk_lines else "no log yet")

stages = []
for p in sorted(glob.glob("/work/sp/stage/*.json")):
    s = _j(p)
    if s:
        stages.append(s)
ic = {"done": "✅", "run": "🔄", "todo": "⬜", "fail": "❌"}
stage_html = "".join(
    f"<li style='margin:2px 0'>{ic.get(s.get('state'), '⬜')} <b>{html.escape(str(s.get('label','')))}</b> "
    f"<span style='color:#555'>{html.escape(str(s.get('note','')))}</span></li>"
    for s in stages
)

# --- Panel 2: Live Teacher Corpus Generation & Single-Voice Cache Progress ---
n_workers = int(_sh("pgrep -fc '[g]en_corpus.py' || true") or "0")
part_files = sorted(glob.glob("/work/sp/distill/corpus/part_*.jsonl"))
clips_by_id = {}
first_mtime, last_mtime = None, None
for pf in part_files:
    try:
        mt = os.path.getmtime(pf)
        last_mtime = mt if last_mtime is None else max(last_mtime, mt)
        for line in open(pf, errors="ignore"):
            line = line.strip()
            if line.endswith("}"):
                r = json.loads(line)
                clips_by_id[r["id"]] = r
    except Exception:
        pass

# Also check manifest.jsonl if merged
mf_path = "/work/sp/distill/manifest.jsonl"
mf_rows = []
if os.path.exists(mf_path):
    for line in open(mf_path, errors="ignore"):
        if line.strip().endswith("}"):
            try:
                mf_rows.append(json.loads(line))
            except Exception:
                pass

active_rows = mf_rows if len(mf_rows) > len(clips_by_id) else list(clips_by_id.values())
n_clips = len(active_rows)
tot_hrs = sum(float(r.get("dur", 0.0)) for r in active_rows) / 3600.0
mean_utmos = (sum(float(r.get("utmos", 0.0)) for r in active_rows) / n_clips) if n_clips else 0.0
mean_cer = (sum(float(r.get("cer", 0.0)) for r in active_rows) / n_clips) if n_clips else 0.0
mean_spk = (sum(float(r.get("spk_sim", 0.0)) for r in active_rows) / n_clips) if n_clips else 0.0
by_src = {}
for r in active_rows:
    sc = r.get("src", "unknown")
    by_src[sc] = by_src.get(sc, 0.0) + float(r.get("dur", 0.0)) / 3600.0
src_str = ", ".join(f"{k}: {v:.2f}h" for k, v in sorted(by_src.items(), key=lambda x: -x[1])) or "none yet"

# Calculate clips/min from worker start time or log mtimes
w_etimes = _sh("ps -eo etimes,args | awk '/[g]en_corpus[.]py/ {print $1; exit}'")
elapsed_min = max(0.5, float(w_etimes) / 60.0) if w_etimes.isdigit() else 15.0
clips_per_min = n_clips / elapsed_min if n_workers > 0 else 0.0
target_hours = 10.0
rem_hours = max(0.0, target_hours - tot_hrs)
hrs_per_min = tot_hrs / elapsed_min if (n_workers > 0 and tot_hrs > 0) else 0.05
corpus_eta = f"{rem_hours / max(1e-4, hrs_per_min):.0f} min to {target_hours:.0f}h target" if n_workers > 0 else ("MERGED & CACHED" if os.path.exists("/work/sp/distill/cache_train.pt") else "idle")
cache_info = "pending"
if os.path.exists("/work/sp/distill/cache_train.pt"):
    csz = os.path.getsize("/work/sp/distill/cache_train.pt") / 1e6
    cache_info = f"✅ READY ({csz:.1f} MB in RAM/SSD)"

# --- Panel 3: Live Distillation Ladder Training Monitor ---
TIER_SPECS = [
    ("t30", "Sakhiya-T30 (Flagship 30M)", "29.6 MB"),
    ("t15", "Sakhiya-T15 (Compact 15M)", "14.5 MB"),
    ("t10", "Sakhiya-T10 (Compact 10M)", "9.5 MB"),
    ("t5",  "Sakhiya-Micro (Hero Sub-5M)", "4.8 MB"),
    ("t1",  "Sakhiya-Pico (Sub-1.5M Looped)", "1.05 MB"),
]
lad_rows = []
for tid, tlabel, tmb in TIER_SPECS:
    lg = f"/work/sp/distill/runs/{tid}/train.log"
    ls_ = [l for l in _t(lg, 50) if l.startswith("Step ")]
    fin = os.path.exists(f"/work/sp/distill/runs/{tid}/ema_final.pt")
    lat = os.path.exists(f"/work/sp/distill/runs/{tid}/ema_latest.pt")
    best_gate = _j(f"/work/sp/tiers/{tid}/best/gate.json")
    g_str = "-"
    if best_gate:
        bs = best_gate.get("summary", {})
        g_str = f"<b>score {best_gate.get('score','-')}</b > · UTMOS {bs.get('utmos','-')} · CER {bs.get('cer_ne','-')} · {best_gate.get('int8_mb','-')} MB"
    if ls_:
        last_l = ls_[-1]
        mm = re.search(r"Step (\d+) \| ([0-9.]+) it/s \| (.*?) \| ([0-9.]+)/([0-9.]+) min qat=(\d)", last_l)
        is_live = os.path.exists(lg) and (time.time() - os.path.getmtime(lg) < 120)
        st_badge = "🟢 TRAINING" if is_live else ("✅ DONE" if fin else ("⏸ PAUSED/CKPT" if lat else "⬜ QUEUED"))
        if mm:
            stp, sps_v, lss, el_m, tot_m, qat_v = mm.groups()
            rem_m = max(0.0, float(tot_m) - float(el_m))
            eta_s = f"{rem_m:.1f}m left ({el_m}/{tot_m}m)" if is_live else f"{el_m}/{tot_m}m"
            lad_rows.append(
                f"<tr><td><b>{tid}</b> ({tmb})</td><td>{st_badge}</td><td align=right><b>{stp}</b></td>"
                f"<td align=right>{sps_v}</td><td>{eta_s}</td><td align=center>{'ON' if qat_v=='1' else 'warmup'}</td>"
                f"<td><code>{html.escape(lss[:85])}</code></td><td>{g_str}</td></tr>"
            )
            continue
    lad_rows.append(
        f"<tr><td><b>{tid}</b> ({tmb})</td><td>⬜ QUEUED</td><td align=right>0</td>"
        f"<td align=right>-</td><td>waiting</td><td align=center>-</td><td><code>-</code></td><td>{g_str}</td></tr>"
    )

# --- Panel 4: Apples-to-Apples Size-vs-Quality Leaderboard ---
# Always include live site baselines + Sakhiya-Pro v1/v2 + all 5 ladder students
 SITE_BASELINES = [
    dict(name="Sakhiya-Pro v1 (ft3_s200 · 600M Teacher)", size_mb="2400 (bf16)", utmos=3.973, cer=0.064, sa_cer=0.137, drops=0, stutters=0, clicks=35, rtf="GPU", status="🏆 Best Teacher (Anchor)"),
    dict(name="Sakhiya-Pro v2 (s2_300 · 600M Teacher)", size_mb="2400 (bf16)", utmos=3.796, cer=0.066, sa_cer=0.147, drops=0, stutters=0, clicks=44, rtf="GPU", status="✅ Frozen Pack + EP01 v2"),
    dict(name="Sakhiya-V1 (Live Site Flagship)", size_mb=15.98, utmos=3.957, cer=0.014, sa_cer=0.025, drops=0, stutters=0, clicks=0, rtf=0.1884, status="🟢 Live Site Baseline (models_v2)"),
    dict(name="Sakhiya-V1-Nano (Live Site)", size_mb=8.93, utmos=3.253, cer=0.027, sa_cer=0.046, drops=0, stutters=0, clicks=0, rtf=0.0217, status="🟢 Live Site Baseline (models_voca)"),
    dict(name="Sakhiya-V1-Atom (Live Site)", size_mb=6.51, utmos=3.262, cer=0.017, sa_cer=0.068, drops=0, stutters=0, clicks=0, rtf=0.0189, status="🟢 Live Site Baseline (models_sano)"),
    dict(name="Sakhiya-V1-Atom-Lite (Live Site)", size_mb=2.72, utmos=2.944, cer=0.017, sa_cer=0.040, drops=0, stutters=0, clicks=0, rtf=0.0155, status="🟢 Live Site Baseline (models_atom_lite)"),
]
 student_targets = [
    ("t30", "Student-T30 (Ladder 30 MB)", "~29.6"),
    ("t15", "Student-T15 (Ladder 14.5 MB)", "~14.5"),
    ("t10", "Student-T10 (Ladder 9.5 MB)", "~9.5"),
    ("t5",  "Sakhiya-Micro t5 (Ladder 4.8 MB)", "~4.8"),
    ("t1",  "Sakhiya-Pico t1 (Ladder 1.05 MB)", "~1.05"),
]
lb_all = list(SITE_BASELINES)
for tid, tname, est_mb in student_targets:
    g = _j(f"/work/sp/tiers/{tid}/best/gate.json")
    if not g:
        # Check if any candidate c_* has a gate.json
        cands = sorted(glob.glob(f"/work/sp/tiers/{tid}/c_*/gate.json"))
        if cands:
            g = _j(cands[-1])
    if g:
        s = g.get("summary", {})
        pass_str = "✅ PASS" if g.get("gate_pass") else "⚠️ GATE FAIL"
        lb_all.append(dict(
            name=f"{tname} (step {g.get('step',0)})",
            size_mb=g.get("int8_mb", est_mb),
            utmos=s.get("utmos", "-"),
            cer=s.get("cer_ne", "-"),
            sa_cer=s.get("cer_sa", "-"),
            drops=s.get("word_drop_count", s.get("word_drop_proxy", 0)),
            stutters=s.get("stutter_count", 0),
            clicks=s.get("clicks", 0),
            rtf=g.get("rtf_1thread", "-"),
            status=f"{pass_str} · corr={g.get('int8_vs_fp32_mel_corr','-')} · spk={s.get('spk_sim','-')}"
        ))
    else:
        lb_all.append(dict(
            name=tname, size_mb=est_mb, utmos="-", cer="-", sa_cer="-",
            drops="-", stutters="-", clicks="-", rtf="-", status="🔄 Awaiting first INT8 gate"
        ))

lb_html = "".join(
    f"<tr><td><b>{html.escape(str(r['name']))}</b></td>"
    f"<td align=right>{r['size_mb']}</td>"
    f"<td align=right><b>{r['utmos']}</b></td>"
    f"<td align=right>{r['cer']}</td>"
    f"<td align=right>{r['sa_cer']}</td>"
    f"<td align=right>{r['drops']}</td>"
    f"<td align=right>{r['stutters']}</td>"
    f"<td align=right>{r['clicks']}</td>"
    f"<td align=right>{r['rtf']}</td>"
    f"<td>{html.escape(str(r['status']))}</td></tr>"
    for r in lb_all
)

col = "#c00" if left_s < 1800 else "#0a6b31"
mo.Html(f"""<div style="font-family:system-ui,-apple-system,sans-serif;font-size:13.5px;line-height:1.45;color:#111">
<div style="background:#f4fbf7;border:1px solid #b7e1c7;border-radius:8px;padding:10px 14px;margin-bottom:10px">
  <h3 style="margin:0 0 4px 0;color:#084c24">🟢 Panel 1 — Box Health & Hard-Cutoff Countdown · {html.escape(npt)}</h3>
  <div><b style="color:{col}">⏱ Countdown to 06:25 NPT Safety Cutoff: {lh}h {lm:02d}m {ls_sec:02d}s</b>
  (06:55 NPT window: {eh}h {em:02d}m · Autopilot target: <code>{html.escape(dl_str)} UTC</code>)</div>
  <div><b>GPU:</b> <code>{html.escape(gpu)}</code> · <b>RAM:</b> <code>{html.escape(ram_info)}</code> · <b>Disk:</b> <code>{html.escape(disk_info)}</code></div>
  <div><b>15-Min HF Backup Daemon:</b> <code>{html.escape(bk_status)}</code></div>
  <ul style="list-style:none;padding-left:0;margin:6px 0 0 0">{stage_html}</ul>
</div>

<div style="background:#f5f8ff;border:1px solid #bfd4ff;border-radius:8px;padding:10px 14px;margin-bottom:10px">
  <h3 style="margin:0 0 4px 0;color:#103a8c">🎙️ Panel 2 — Live Teacher Corpus Generation & Single-Voice Cache Progress</h3>
  <div><b>Active Workers:</b> {n_workers} · <b>Verified Clips:</b> <b>{n_clips}</b> · <b>Audio Volume:</b> <b>{tot_hrs:.2f} hours</b> (24 kHz mono) · <b>Speed:</b> {clips_per_min:.1f} clips/min · <b>ETA:</b> {html.escape(corpus_eta)}</div>
  <div><b>Quality Gate Means:</b> UTMOS <b>{mean_utmos:.3f}</b> · CER <b>{mean_cer:.3f}</b> · ECAPA Spk-Sim <b>{mean_spk:.3f}</b> · <b>RAM/SSD Shard Cache:</b> {html.escape(cache_info)}</div>
  <div style="color:#444;font-size:12.5px"><b>Sources Breakdown:</b> {html.escape(src_str)}</div>
</div>

<div style="background:#fffaf2;border:1px solid #f5d69b;border-radius:8px;padding:10px 14px;margin-bottom:10px">
  <h3 style="margin:0 0 6px 0;color:#7a4b00">🪜 Panel 3 — Live Distillation Ladder Training Monitor (30 → 15 → 10 → 5 → 1 MB)</h3>
  <table style="border-collapse:collapse;width:100%;font-size:12.5px" border="1" cellpadding="4">
    <tr style="background:#fef3dc"><th>Tier</th><th>Status</th><th>Step</th><th>Steps/s</th><th>Progress / ETA</th><th>QAT</th><th>Live Losses</th><th>Best Exported INT8 Gate</th></tr>
    {"".join(lad_rows)}
  </table>
</div>

<div style="background:#fff;border:1px solid #d0d7de;border-radius:8px;padding:10px 14px">
  <h3 style="margin:0 0 6px 0;color:#1f2328">📊 Panel 4 — Apples-to-Apples Size-vs-Quality Leaderboard (Real Exported INT8 ONNX Gate)</h3>
  <table style="border-collapse:collapse;width:100%;font-size:12.5px" border="1" cellpadding="4">
    <tr style="background:#f6f8fa"><th>Model / Tier</th><th>INT8 Size (MB)</th><th>UTMOS</th><th>Nepali CER</th><th>Sanskrit CER</th><th>Drops</th><th>Stutters</th><th>Clicks</th><th>1-Thr CPU RTF</th><th>Gate Status & Forensics</th></tr>
    {lb_html}
  </table>
</div>
</div>""")
'''

AUDIO = r'''
sp2_tick.value
import os, glob, json
import marimo as mo

def _safe_aud(p, label=None):
    try:
        if os.path.isfile(p) and os.path.getsize(p) > 256:
            aud = mo.audio(open(p, "rb").read())
            return mo.vstack([mo.md(f"**{label}**"), aud]) if label else aud
    except Exception:
        pass
    return None

sections = {}

# 1. Episode 01 Broadcast Voiceover & Surgical Lines (c3, n1a, n2a, n4a, f1, o1)
ep_items = {}
for tag, ep_p in (("EP01 v2 Mastered (-14 LUFS)", "/work/sp/render/ep01_v2/voiceover_ep01.mp3"),
                  ("EP01 v1 Mastered (-14 LUFS)", "/work/sp/render/ep01/voiceover_ep01.mp3"),
                  ("EP01 Repo Frozen v2", "/work/sp/repo/docs/sakhiya_pro/ep01_v2/voiceover_ep01.mp3")):
    a = _safe_aud(ep_p)
    if a is not None:
        ep_items[tag] = a
for lid in ("c3", "n1a", "n2a", "n4a", "f1", "o1"):
    lp = f"/work/sp/render/ep01_v2/lines/{lid}.wav"
    a = _safe_aud(lp)
    if a is not None:
        ep_items[f"EP01 line {lid}"] = a
if ep_items:
    sections["🎙️ NEBians News Episode 01 Broadcast Voiceover & Key Lines"] = mo.accordion(ep_items)

# 2. Live Listening Desk folders (/work/sp/listen/*)
for d in sorted(glob.glob("/work/sp/listen/*/")):
    gname = os.path.basename(d.rstrip("/"))
    fs = sorted(glob.glob(d + "*.mp3") + glob.glob(d + "*.wav"))[:24]
    items = {}
    for f in fs:
        a = _safe_aud(f)
        if a is not None:
            items[os.path.basename(f).rsplit(".", 1)[0]] = a
    if items:
        sections[f"🔊 Live Tier: {gname} ({len(items)} clips)"] = mo.accordion(items)

# 3. Exported Student INT8 WAVs/MP3s from /work/sp/tiers/t*/best/wav/
for tid in ("t30", "t15", "t10", "t5", "t1"):
    for sub in ("best/wav",):
        wdir = f"/work/sp/tiers/{tid}/{sub}"
        if os.path.isdir(wdir):
            fs = sorted(glob.glob(wdir + "/*.mp3") or glob.glob(wdir + "/*.wav"))[:20]
            items = {}
            for f in fs:
                a = _safe_aud(f)
                if a is not None:
                    items[os.path.basename(f).rsplit(".", 1)[0]] = a
            if items and f"🔊 Live Tier: student_{tid} ({len(items)} clips)" not in sections:
                sections[f"🧪 Exported INT8 Student: {tid} ({len(items)} clips)"] = mo.accordion(items)

# 4. Frozen Reference Packs in repo (Sakhiya-Pro v1, Sakhiya-Pro v2, Sakhiya-V1 Chandas, Nano/Atom)
for vtag, vtitle in (("sakhiya_pro", "Sakhiya-Pro v1 (ft3_s200 Anchor)"),
                     ("sakhiya_pro_v2", "Sakhiya-Pro v2 (s2_300)"),
                     ("v15_v1_chandas", "Sakhiya-V1 (15.98 MB Flagship Baseline)")):
    vdir = f"/work/sp/repo/docs/samples/versions/{vtag}"
    if os.path.isdir(vdir):
        fs = sorted(glob.glob(vdir + "/*.mp3") + glob.glob(vdir + "/*.wav"))[:15]
        items = {}
        for f in fs:
            a = _safe_aud(f)
            if a is not None:
                items[os.path.basename(f).rsplit(".", 1)[0]] = a
        if items:
            sections[f"📦 Frozen Master Pack: {vtitle} ({len(items)} clips)"] = mo.accordion(items)

mo.vstack([
    mo.md("### 🎧 Panel 5 — Interactive Listening Desk (Auto-Discovers Latest INT8 Exports, Baselines & EP01)"),
    mo.accordion(sections) if sections else mo.md("_No audio clips discovered yet_")
])
'''

def _wrap(code):
    import textwrap, re as _re
    body = textwrap.indent(code.strip(), '    ')
    body = _re.sub(r'^    (mo\.Html\(|mo\.vstack\(\[)', lambda m: '    return ' + m.group(1), body, flags=_re.M)
    return 'def _main():\n' + body + '\n_main()'

async def install():
    import marimo._code_mode as cm
    async with cm.get_context(skip_staleness_check=True) as ctx:
        ex = {c.name: c.id for c in ctx.cells.values()}
        for name, code in (('tts_sakhiya_pro2_tick', TICK), ('tts_sakhiya_pro2', CARD), ('tts_sakhiya_pro2_audio', AUDIO)):
            code = code.strip() if name.endswith('_tick') else _wrap(code)
            if name in ex:
                ctx.edit_cell(ex[name], code, hide_code=True)
                cid = ex[name]
            else:
                cid = ctx.create_cell(code, name=name, hide_code=True)
            ctx.run_cell(cid)
            print('installed', name)
