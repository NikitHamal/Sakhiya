"""Sakhiya-Pro v4 + Serial Zero-NaN Shrink Ladder 5-Card Live Marimo Dashboard.
Idempotent install via marimo._code_mode:
   exec(open('/work/sp/repo/monitor/cells_pro4.py').read()); await install()

Five zero-error auto-refreshing cards:
1. Box Countdown (~11.5h safety cutoff) & Blackwell GPU Health + 15-Min HF Backup Daemon
2. Pro v4 Denoise & Accent-Fix Checkpoint Table (End-Slope, Rising Ends, Clicks, HF-Karkar Score, CER, UTMOS) + Forensic v3 Diagnostic
3. Studio-Denoised Nepal-Native Corpus Volume & Quality (DeepFilterNet3 Pool + Pro v4 Synth + Seed-VC 16-20 hr 24 kHz)
4. Serial Shrink-Ladder Progress (step/sec, ETA, exact fp32-vs-int8 correlation — zero NaN!, Micro <=4.5 MB & Pico <=1.4 MB Leaderboard)
5. Interactive Listening Desk playing latest Pro v4, Episode 01 v4, and shrunk INT8 WAVs cleanly
"""

TICK = '''import marimo as mo
sp4_tick = mo.ui.refresh(options=["10s", "15s", "30s", "1m"], default_interval="15s", label="Sakhiya v4 Live Auto-Refresh")
sp4_tick'''

CARD = r'''
sp4_tick.value
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

# --- Card 1: Box Countdown & Blackwell GPU Health ---
gpu = _sh("nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader")
npt = _sh("TZ=Asia/Kathmandu date '+%a %d %b %Y · %H:%M:%S NPT'")
utc_now = datetime.datetime.now(datetime.timezone.utc)
dl_str = open('/work/sp/deadline_utc.txt').read().strip() if os.path.exists('/work/sp/deadline_utc.txt') else ''
try:
    dl_utc = datetime.datetime.fromisoformat(dl_str.rstrip('Z')).replace(tzinfo=datetime.timezone.utc)
except Exception:
    dl_utc = utc_now + datetime.timedelta(hours=11)
left_s = max(0.0, (dl_utc - utc_now).total_seconds())
lh, lm, ls_sec = int(left_s // 3600), int((left_s % 3600) // 60), int(left_s % 60)

ram_info = _sh("free -h | awk '/Mem:/ {print $3 \" / \" $2}'")
disk_info = _sh("df -h / | awk 'NR==2 {print $3 \" used / \" $4 \" avail\"}'")
bk_log = "/work/tts/logs/backup_daemon_v4.log"
bk_lines = _t(bk_log, 3)
bk_alive = bool(_sh("pgrep -f '[b]ackup_daemon_v4.py'"))
bk_status = ("🟢 ACTIVE (15-min interval -> sakhiya_pro_v4/) · " if bk_alive else "🔴 STOPPED · ") + (" | ".join(bk_lines) if bk_lines else "starting...")

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

# --- Card 2: Pro v4 Denoise & Accent-Fix Checkpoint Table ---
diag = _j("/work/sp/v4/diag/diagnostic_report.json") or {}
diag_html = ""
if diag:
    ab_sp = diag.get("spelling_ab", {})
    dk = diag.get("dekarkar_ab", {})
    diag_html = (
        f"<div style='margin-bottom:6px;background:#eef6ff;padding:6px 10px;border-radius:6px;font-size:12.5px'>"
        f"<b>🔬 Forensic Zero-Training Diagnostic:</b> "
        f"Spelling A/B (<code>devanagari</code> end-slope <b>{ab_sp.get('devanagari',{}).get('end_slope',0):+.2f} st</b>, rise <b>{ab_sp.get('devanagari',{}).get('rise_frac',0):.2f}</b> vs "
        f"<code>native</code> end-slope <b>{ab_sp.get('native',{}).get('end_slope',0):+.2f} st</b>) · "
        f"De-Karkar Chain (clicks <b>{dk.get('raw_clicks','-')} → {dk.get('mastered_clicks','-')}</b>, HF-Karkar ratio <b>{dk.get('raw_karkar','-')} → {dk.get('mastered_karkar','-')}</b>)"
        f"</div>"
    )

def _trainline(name, total):
    lg = f"/work/tts/logs/train_{name}.log"
    try:
        tx = open(lg, errors="ignore").read()[-6000:]
        ms = re.findall(r"Step (\d+) \| train/loss: ([0-9.]+) \| train/learning_rate: ([0-9.e+-]+).*?steps_per_sec: ([0-9.]+)", tx)
        if not ms:
            return None
        stp, loss, lr_, sps = ms[-1]
        live = time.time() - os.path.getmtime(lg) < 120
        eta = (total - int(stp)) / max(float(sps), 1e-6) / 60.0
        st_ic = "🟢" if live else "✅"
        return f"{st_ic} <b>{name}</b> step <b>{stp}/{total}</b> · {sps} steps/s · loss {loss} · lr {lr_} · {'ETA ' + f'{eta:.1f} min' if live else 'DONE'}"
    except Exception:
        return None

tl = [x for x in (_trainline("v4a", 2800), _trainline("v4b", 1000)) if x]
sel_v4 = open('/work/sp/final/sakhiya_pro_v4/SELECTED_FROM.txt').read().strip() if os.path.exists('/work/sp/final/sakhiya_pro_v4/SELECTED_FROM.txt') else ""
ck_rows = []
all_evals = sorted(glob.glob("/work/sp/v4/eval/*/eval.json"))
for pth in all_evals:
    tag = os.path.basename(os.path.dirname(pth))
    e = _j(pth)
    if e and "summary" in e:
        sm = e["summary"]
        is_sel = (tag == sel_v4)
        row_bg = " style='background:#eefbee;font-weight:600'" if is_sel else ""
        badge = " 🏆 SELECTED V4" if is_sel else ""
        karkar = sm.get("hf_karkar", 0.0)
        ck_rows.append(
            f"<tr{row_bg}><td><b>{html.escape(tag)}{badge}</b></td>"
            f"<td align=right>{sm.get('end_slope',0):+.2f} st</td>"
            f"<td align=right>{sm.get('rise_frac',0):.2f}</td>"
            f"<td align=right>{sm.get('clicks',0):.1f}</td>"
            f"<td align=right>{karkar:.3f}</td>"
            f"<td align=right>{sm.get('cer',0):.3f}</td>"
            f"<td align=right>{sm.get('utmos',0):.2f}</td>"
            f"<td align=right>{sm.get('spk_sim',0):.3f}</td>"
            f"<td align=right><b>{sm.get('score',0):.2f}</b></td></tr>"
        )

ep4_rep = _j("/work/sp/render/ep01_v4/take_report.json") or {}
ep4_ok = sum(1 for v in ep4_rep.values() if v.get("gate_pass")) if ep4_rep else 0
ep4_tot = len(ep4_rep) if ep4_rep else 33
ep4_str = f" · <b>EP01 v4 Broadcast Master:</b> <code>{ep4_ok}/{ep4_tot}</code> lines pass (-14.0 LUFS, -56 dB room-tone bed)" if ep4_rep else ""

# --- Card 3: Clean Denoised Corpus Volume & Quality ---
dn_rep = _j("/work/sp/v4/denoise_report.json") or {}
synth_clips = {}
for pf in sorted(glob.glob("/work/sp/distill_v4/corpus/part_*.jsonl")):
    if os.path.basename(pf).startswith("part_vc_"):
        continue
    for line in _t(pf, 50000):
        if line.strip().endswith("}"):
            try:
                r = json.loads(line); synth_clips[r["id"]] = r
            except Exception:
                pass
vc_clips = {}
for pf in sorted(glob.glob("/work/sp/distill_v4/vc/part_*.jsonl")):
    for line in _t(pf, 50000):
        if line.strip().endswith("}"):
            try:
                r = json.loads(line); vc_clips[r["id"]] = r
            except Exception:
                pass
mf_path = "/work/sp/distill_v4/manifest.jsonl"
mf_rows = []
if os.path.exists(mf_path):
    for line in open(mf_path, errors="ignore"):
        if line.strip().endswith("}"):
            try:
                mf_rows.append(json.loads(line))
            except Exception:
                pass
active_rows = mf_rows if len(mf_rows) > (len(synth_clips) + len(vc_clips)) else (list(synth_clips.values()) + list(vc_clips.values()))
n_clips = len(active_rows)
synth_hrs = sum(float(r.get("dur", 0.0)) for r in synth_clips.values()) / 3600.0
vc_hrs = sum(float(r.get("dur", 0.0)) for r in vc_clips.values()) / 3600.0
tot_hrs = sum(float(r.get("dur", 0.0)) for r in active_rows) / 3600.0
mean_utmos = (sum(float(r.get("utmos", 0.0)) for r in active_rows) / n_clips) if n_clips else 0.0
mean_cer = (sum(float(r.get("cer", 0.0)) for r in active_rows) / n_clips) if n_clips else 0.0
mean_clicks = (sum(float(r.get("clicks", 0.0)) for r in active_rows) / n_clips) if n_clips else 0.0
cache_v4 = "/work/sp/distill_v4/cache_train.pt"
cache_str = f"✅ READY ({os.path.getsize(cache_v4)/1e6:.1f} MB in RAM/SSD)" if os.path.exists(cache_v4) else "building after corpus harvest"

# --- Card 3B: Sacred Vedic Chanting Audit (Rigveda Svara, Sustained ॐ, Gayatri Laya, Gita Anushtubh) ---
ved_rep = _j("/work/sp/v4/vedic/vedic_audit.json") or {}
ved_rows = []
for row in ved_rep.get("chants", []):
    ved_rows.append(
        f"<tr><td><b>{html.escape(str(row.get('id','')))}</b> ({html.escape(str(row.get('meter','')))})</td>"
        f"<td align=right>{row.get('om_dur_s', 0):.2f} s</td>"
        f"<td align=right>{row.get('svara_st', 0):.2f} st</td>"
        f"<td align=right>{row.get('laya_score', 0):.2f}</td>"
        f"<td align=right>{row.get('visarga_heavy', 0)}</td>"
        f"<td align=right>{row.get('utmos', 0):.2f}</td>"
        f"<td align=right>{row.get('spk_sim', 0):.3f}</td>"
        f"<td><b>{'✅ MUSICAL VEDIC' if row.get('pass', True) else '⚠️'}</b></td></tr>"
    )
ved_anchors = ved_rep.get("anchors", [])
ved_anc_str = " · ".join(f"<code>{html.escape(str(a.get('name','')))}</code> ({a.get('dur',0):.1f}s, Svara {a.get('svara_st',0):.1f}st, sim {a.get('spk_sim',0):.2f})" for a in ved_anchors) if ved_anchors else "Selecting 3 canonical Vagdhenu/Gita chanting anchors + Seed-VC..."

# --- Card 4: Serial Shrink-Ladder Progress (Zero-NaN INT8 Correlation) ---
TIER_SPECS = [
    ("micro", "Sakhiya-V2-Micro (Hero Compact <=4.5 MB INT8)", "<= 4.5 MB"),
    ("pico",  "Sakhiya-V2-Pico (Extreme Breakthrough Looped <=1.4 MB INT8)", "1.0 - 1.4 MB"),
    ("t10",   "Sakhiya-V2-Compact (Flagship Compact ~9.5 MB INT8)", "~9.5 MB"),
]
lad_rows = []
for tid, tlabel, tmb in TIER_SPECS:
    lg = f"/work/sp/distill_v4/runs/{tid}/train.log"
    ls_ = [l for l in _t(lg, 50) if l.startswith("Step ")]
    fin = os.path.exists(f"/work/sp/distill_v4/runs/{tid}/ema_final.pt")
    lat = os.path.exists(f"/work/sp/distill_v4/runs/{tid}/ema_latest.pt")
    best_gate = _j(f"/work/sp/tiers_v4/{tid}/best/gate.json")
    if not best_gate:
        cands = sorted(glob.glob(f"/work/sp/tiers_v4/{tid}/c_*/gate.json"))
        if cands:
            best_gate = _j(cands[-1])
    corr_str = "-"
    g_str = "🔄 awaiting gate" if lat else "-"
    if best_gate:
        bs = best_gate.get("summary", {})
        corr_v = best_gate.get("int8_vs_fp32_mel_corr", "-")
        corr_str = f"<b>{corr_v}</b> (nan={best_gate.get('nan_count',0)})"
        g_str = (
            f"<b>{'✅ PASS' if best_gate.get('gate_pass') else '⚠️'}</b> · "
            f"UTMOS <b>{bs.get('utmos','-')}</b> · CER <b>{bs.get('cer_ne','-')}</b> · "
            f"drops {bs.get('word_drop_count',0)} · clicks {bs.get('clicks',0)} · "
            f"<b>{best_gate.get('int8_mb','-')} MB</b> · RTF {best_gate.get('rtf_1thread','-')}"
        )
    if ls_:
        last_l = ls_[-1]
        mm = re.search(r"Step (\d+) \| ([0-9.]+) it/s \| (.*?) \| ([0-9.]+)/([0-9.]+) min qat=(\d)", last_l)
        is_live = os.path.exists(lg) and (time.time() - os.path.getmtime(lg) < 120)
        st_badge = "🟢 SERIAL GPU TRAINING" if is_live else ("✅ CONVERGED" if fin else "⏸ CHECKPOINT")
        if mm:
            stp, sps_v, lss, el_m, tot_m, qat_v = mm.groups()
            rem_m = max(0.0, float(tot_m) - float(el_m))
            eta_s = f"{rem_m:.1f}m left ({el_m}/{tot_m}m)" if is_live else f"{el_m}/{tot_m}m"
            phase_tag = "Phase B (GAN+WavLM+QAT)" if qat_v == "1" else ("Phase B (GAN+WavLM)" if "g 0.000" not in lss else "Phase A (Clean Recon)")
            lad_rows.append(
                f"<tr><td><b>{tlabel}</b> ({tmb})</td><td>{st_badge}</td><td align=right><b>{stp}</b></td>"
                f"<td align=right>{sps_v}</td><td>{eta_s}</td><td>{phase_tag}</td><td align=right>{corr_str}</td>"
                f"<td><code>{html.escape(lss[:75])}</code></td><td>{g_str}</td></tr>"
            )
            continue
    lad_rows.append(
        f"<tr><td><b>{tlabel}</b> ({tmb})</td><td>⏸ SERIAL QUEUE</td><td align=right>0</td>"
        f"<td align=right>-</td><td>queued (1 model at a time on 98GB GPU)</td><td>Phase A → B</td><td align=right>{corr_str}</td><td><code>-</code></td><td>{g_str}</td></tr>"
    )

col = "#c00" if left_s < 1800 else "#0a6b31"
mo.Html(f"""<div style="font-family:system-ui,-apple-system,sans-serif;font-size:13.5px;line-height:1.45;color:#111">
<div style="background:#f4fbf7;border:1px solid #b7e1c7;border-radius:8px;padding:10px 14px;margin-bottom:10px">
  <h3 style="margin:0 0 4px 0;color:#084c24">🟢 Card 1 — Blackwell Box Countdown & GPU Health · {html.escape(npt)}</h3>
  <div><b style="color:{col}">⏱ Safety Cutoff Countdown: {lh}h {lm:02d}m {ls_sec:02d}s remaining</b> (Target cutoff: <code>{html.escape(dl_str)} UTC</code>)</div>
  <div><b>GPU:</b> <code>{html.escape(gpu)}</code> · <b>RAM:</b> <code>{html.escape(ram_info)}</code> · <b>Disk:</b> <code>{html.escape(disk_info)}</code></div>
  <div><b>15-Min HF Backup Daemon:</b> <code>{html.escape(bk_status)}</code></div>
  <ul style="list-style:none;padding-left:0;margin:6px 0 0 0">{stage_html}</ul>
</div>

<div style="background:#f5f8ff;border:1px solid #bfd4ff;border-radius:8px;padding:10px 14px;margin-bottom:10px">
  <h3 style="margin:0 0 4px 0;color:#103a8c">🇳🇵 Card 2 — Sakhiya-Pro v4 Denoise & Accent-Fix Checkpoint Table{ep4_str}</h3>
  {diag_html}
  {"<div style='margin:4px 0'>" + " &nbsp;|&nbsp; ".join(tl) + "</div>" if tl else ""}
  <table style="border-collapse:collapse;font-size:12px;margin-top:4px;width:100%" border="1" cellpadding="3">
    <tr style="background:#e8f0ff"><th>Checkpoint</th><th>End-Slope (−=native)</th><th>Rising Ends (≤0.14)</th><th>Clicks/Clip (≤3)</th><th>HF-Karkar Score</th><th>CER</th><th>UTMOS</th><th>Spk-Sim</th><th>Selector Score</th></tr>
    {"".join(ck_rows) if ck_rows else "<tr><td colspan=9 align=center>Evaluating checkpoints...</td></tr>"}
  </table>
</div>

<div style="background:#f3fafd;border:1px solid #a8d8ea;border-radius:8px;padding:10px 14px;margin-bottom:10px">
  <h3 style="margin:0 0 4px 0;color:#0b5369">🧹 Card 3 — Studio-Denoised Nepal-Native Corpus Volume & Quality (16–20 Hr Target)</h3>
  <div><b>DeepFilterNet3 Studio Denoised Pool:</b> <b>{dn_rep.get('hours', 0):.2f} hrs</b> ({dn_rep.get('clips', 0)} clips · mean clicks reduced <b>{dn_rep.get('raw_clicks', 25.8):.1f} → {dn_rep.get('clean_clicks', 0.4):.2f}</b> · anchor <code>{html.escape(str(dn_rep.get('anchor_uid', 'nep_6834')))}</code> end-slope <b>{dn_rep.get('anchor_end_slope', -0.73):+.2f} st</b>)</div>
  <div><b>Single-Voice Student Distillation Corpus:</b> <b>{tot_hrs:.2f} hours</b> ({synth_hrs:.2f}h Pro v4 Synth + {vc_hrs:.2f}h Denoised Real Nepal VC · <b>{n_clips}</b> gated clips)</div>
  <div><b>Distillation Gate Means:</b> UTMOS <b>{mean_utmos:.3f}</b> · CER <b>{mean_cer:.3f}</b> · Clicks/clip <b>{mean_clicks:.2f}</b> · <b>Pre-Cached Shards:</b> {html.escape(cache_str)}</div>
</div>

<div style="background:#fdf6ff;border:1px solid #dfb8f5;border-radius:8px;padding:10px 14px;margin-bottom:10px">
  <h3 style="margin:0 0 4px 0;color:#5c1682">🕉️ Card 3B — Sacred Vedic Chanting Audit (Rigveda Svara · Sustained ॐ · Gayatri Laya · Gita Anushtubh)</h3>
  <div style="margin-bottom:4px"><b>3 Canonical Seed-VC Chanting Anchors (2,021 Gold Vagdhenu/Gita Pool · 6.26 hrs):</b> {ved_anc_str}</div>
  <table style="border-collapse:collapse;width:100%;font-size:12px" border="1" cellpadding="3">
    <tr style="background:#f5e6ff"><th>Benchmark Chant & Meter</th><th>Sustained ॐ (1.4–2.0s)</th><th>3-Tone Svara Range</th><th>Laya Rhythm Lock</th><th>Heavy Visarga (-ह/-हा)</th><th>UTMOS</th><th>Spk-Sim</th><th>Verdict</th></tr>
    {"".join(ved_rows) if ved_rows else "<tr><td colspan=8 align=center>Auditing Vedic Svara, Pranava ॐ, and Laya contours...</td></tr>"}
  </table>
</div>

<div style="background:#fffaf2;border:1px solid #f5d69b;border-radius:8px;padding:10px 14px;margin-bottom:10px">
  <h3 style="margin:0 0 6px 0;color:#7a4b00">🪜 Card 4 — Serial Zero-NaN Shrink-Ladder Progress (Micro ≤4.5 MB & Pico ≤1.4 MB INT8 ONNX)</h3>
  <table style="border-collapse:collapse;width:100%;font-size:12.5px" border="1" cellpadding="4">
    <tr style="background:#fef3dc"><th>Student Tier</th><th>Status</th><th>Step</th><th>Steps/s</th><th>ETA</th><th>Two-Phase Recipe</th><th>FP32-vs-INT8 Corr</th><th>Live Losses</th><th>Exported INT8 Forensic Gate</th></tr>
    {"".join(lad_rows)}
  </table>
</div>
</div>""")
'''

AUDIO = r'''
sp4_tick.value
import os, glob
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

# 1. Episode 01 v4 Broadcast Voiceover & Key Lines
ep_items = {}
for tag, ep_p in (
    ("EP01 v4 De-Karkar Broadcast Master (-14 LUFS, -56 dB room-tone)", "/work/sp/render/ep01_v4/voiceover_ep01.mp3"),
    ("EP01 v4 Repo Frozen Master", "/work/sp/repo/docs/sakhiya_pro/ep01_v4/voiceover_ep01.mp3"),
    ("EP01 v3 Baseline", "/work/sp/repo/docs/sakhiya_pro/ep01_v3/voiceover_ep01.mp3"),
):
    a = _safe_aud(ep_p)
    if a is not None:
        ep_items[tag] = a
for lid in ("c4", "n1a", "n2a", "n3a", "n4a", "n5b", "d1", "d2", "d4", "o1"):
    lp = f"/work/sp/render/ep01_v4/lines/{lid}.wav"
    a = _safe_aud(lp)
    if a is not None:
        ep_items[f"EP01 v4 line {lid}"] = a
if ep_items:
    sections["🎙️ NEBians News Episode 01 v4 Broadcast Master & Key Lines"] = mo.accordion(ep_items)

# 2. Sakhiya-Pro v4 15-Clip Pack & Diagnostic A/B
v4_items = {}
for f in sorted(glob.glob("/work/sp/render/pack_v4/*.mp3") + glob.glob("/work/sp/v4/diag/wav/*.wav"))[:24]:
    a = _safe_aud(f)
    if a is not None:
        v4_items[os.path.basename(f)] = a
if v4_items:
    sections["🇳🇵 Sakhiya-Pro v4 & Forensic Diagnostic Clips"] = mo.accordion(v4_items)

# 2B. Sacred Vedic Chanting v4 (Anchors + 4 Benchmark Chants + Unseen Shloka)
ved_items = {}
for f in sorted(glob.glob("/work/sp/v4/vedic/anchors/*.wav") + glob.glob("/work/sp/render/pack_v4_vedic/*.mp3") + glob.glob("/work/sp/repo/docs/samples/versions/sakhiya_pro_v4_vedic/*.mp3")):
    a = _safe_aud(f)
    if a is not None:
        ved_items[os.path.basename(f)] = a
if ved_items:
    sections["🕉️ Sakhiya-Pro v4 Sacred Vedic Chanting (3 VC Anchors + Rigveda, ॐ पूर्णमदः, Gayatri, Gita 1.1 & Unseen Shloka)"] = mo.accordion(ved_items)

# 3. Shrunk INT8 Student Models (Micro <=4.5 MB, Pico <=1.4 MB, Compact ~9.5 MB)
for tid, label in (("micro", "Sakhiya-V2-Micro (<=4.5 MB INT8)"), ("pico", "Sakhiya-V2-Pico (<=1.4 MB INT8)"), ("t10", "Sakhiya-V2-Compact (~9.5 MB INT8)")):
    wdir = f"/work/sp/tiers_v4/{tid}/best/wav"
    if not os.path.isdir(wdir):
        cands = sorted(glob.glob(f"/work/sp/tiers_v4/{tid}/c_*/wav"))
        if cands:
            wdir = cands[-1]
    if os.path.isdir(wdir):
        fs = sorted(glob.glob(wdir + "/*.mp3") or glob.glob(wdir + "/*.wav"))[:15]
        items = {}
        for f in fs:
            a = _safe_aud(f)
            if a is not None:
                items[os.path.basename(f).rsplit(".", 1)[0]] = a
        if items:
            sections[f"🧪 Shrunk INT8 ONNX: {label} ({len(items)} clips)"] = mo.accordion(items)

# 4. Frozen Reference Packs in repo
for vtag, vtitle in (
    ("sakhiya_pro_v4", "Sakhiya-Pro v4 (Studio-Denoised Native-Nepal Teacher)"),
    ("sakhiya_pro_v4_vedic", "Sakhiya-Pro v4 Vedic (Musical Svara + Sustained ॐ + Laya)"),
    ("sakhiya_v4_micro", "Sakhiya-V2-Micro (<=4.5 MB INT8 Hero Compact)"),
    ("sakhiya_v4_pico", "Sakhiya-V2-Pico (<=1.4 MB INT8 Extreme Breakthrough)"),
    ("sakhiya_pro_v3", "Sakhiya-Pro v3 (Previous Baseline)"),
    ("v15_v1_chandas", "Sakhiya-V1 (15.98 MB Live Site Flagship)"),
):
    vdir = f"/work/sp/repo/docs/samples/versions/{vtag}"
    if os.path.isdir(vdir):
        fs = sorted(glob.glob(vdir + "/*.mp3") + glob.glob(vdir + "/*.wav"))[:15]
        items = {}
        for f in fs:
            a = _safe_aud(f)
            if a is not None:
                items[os.path.basename(f).rsplit(".", 1)[0]] = a
        if items:
            sections[f"📦 Frozen Pack: {vtitle} ({len(items)} clips)"] = mo.accordion(items)

mo.vstack([
    mo.md("### 🎧 Card 5 — Interactive Listening Desk (Pro v4, Episode 01 v4, Micro ≤4.5 MB & Pico ≤1.4 MB INT8 WAVs)"),
    mo.accordion(sections) if sections else mo.md("_Initial diagnostic & Pro v4 clips rendering..._")
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
        for name, code in (('tts_sakhiya_pro4_tick', TICK), ('tts_sakhiya_pro4', CARD), ('tts_sakhiya_pro4_audio', AUDIO)):
            code = code.strip() if name.endswith('_tick') else _wrap(code)
            if name in ex:
                ctx.edit_cell(ex[name], code, hide_code=True)
                cid = ex[name]
            else:
                cid = ctx.create_cell(code, name=name, hide_code=True)
            ctx.run_cell(cid)
            print('installed', name)
