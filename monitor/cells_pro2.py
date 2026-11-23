"""Sakhiya-Pro v1/v2/v3 + Shrink Ladder 5-Panel Live Marimo Dashboard.
Idempotent install via marimo._code_mode (touches ONLY tts_sakhiya_pro2* cells):
   exec(open('/work/sp/repo/monitor/cells_pro2.py').read()); await install()

Five crisp auto-refreshing panels:
1. Real-time Box Health & Countdown Timer to 06:25 NPT cutoff (+ GPU, RAM, Disk, 15-min HF Backup status)
2. Active Accent-Fix (Sakhiya-Pro v3) & Dual-Source Teacher Corpus Progress (v3 Synth + Seed-VC Real Nepal Speech)
3. Live Shrink-Ladder Training Monitor (active tier, step, steps/sec, ETA, QAT, best exported INT8 gate score)
4. Apples-to-Apples Size-vs-Quality Leaderboard comparing live site tiers (V1 15.98 MB, Nano 8.93 MB, Atom 6.51 MB,
   Atom-Lite 2.72 MB / 2.998 MB GGUF) and Sakhiya-Pro v1/v2/v3 against shrunk ladder tiers on UTMOS, CER,
   drops, stutters, clicks, and 1-thread CPU RTF
5. Interactive Listening Desk auto-discovering latest exported INT8 WAVs/MP3s from every tier, Pro v3, and Episode 01 with 0 errors
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
dl_str = open('/work/sp/deadline_utc.txt').read().strip() if os.path.exists('/work/sp/deadline_utc.txt') else '2026-11-24T00:40:00'
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
bk_log = "/work/tts/logs/backup_daemon_v3.log" if os.path.exists("/work/tts/logs/backup_daemon_v3.log") else "/work/tts/logs/backup_daemon_v2.log"
bk_lines = _t(bk_log, 3)
bk_alive = bool(_sh("pgrep -f '[b]ackup_daemon.py'"))
bk_status = ("🟢 ACTIVE (15-min interval) · " if bk_alive else "🔴 STOPPED · ") + (" | ".join(bk_lines) if bk_lines else "no log yet")

stages = []
for p in sorted(glob.glob("/work/sp/stage/*.json")):
    bn = os.path.basename(p)
    if bn in ("10_corpus.json", "20_s1.json", "30_s2.json", "40_verify.json"):
        continue
    s = _j(p)
    if s:
        stages.append(s)
ic = {"done": "✅", "run": "🔄", "todo": "⬜", "fail": "❌"}
stage_html = "".join(
    f"<li style='margin:2px 0'>{ic.get(s.get('state'), '⬜')} <b>{html.escape(str(s.get('label','')))}</b> "
    f"<span style='color:#555'>{html.escape(str(s.get('note','')))}</span></li>"
    for s in stages
)

# --- Panel 2: Active Accent-Fix (Pro v3) & Dual-Source Teacher-Corpus Status ---
n_gen_workers = int(_sh("pgrep -fc '[g]en_corpus.py work' || true") or "0")
n_vc_workers = int(_sh("pgrep -fc '[v]c_corpus.py work' || true") or "0")
n_workers = n_gen_workers + n_vc_workers

synth_clips = {}
for pf in sorted(glob.glob("/work/sp/distill/corpus/part_*.jsonl")):
    if os.path.basename(pf).startswith("part_vc_"):
        continue
    try:
        for line in open(pf, errors="ignore"):
            line = line.strip()
            if line.endswith("}"):
                r = json.loads(line)
                synth_clips[r["id"]] = r
    except Exception:
        pass

vc_clips = {}
for pf in sorted(glob.glob("/work/sp/distill/vc/part_*.jsonl")):
    try:
        for line in open(pf, errors="ignore"):
            line = line.strip()
            if line.endswith("}"):
                r = json.loads(line)
                vc_clips[r["id"]] = r
    except Exception:
        pass

all_live_clips = dict(synth_clips)
all_live_clips.update(vc_clips)

mf_path = "/work/sp/distill/manifest.jsonl"
mf_rows = []
if os.path.exists(mf_path):
    for line in open(mf_path, errors="ignore"):
        if line.strip().endswith("}"):
            try:
                mf_rows.append(json.loads(line))
            except Exception:
                pass

active_rows = mf_rows if len(mf_rows) > len(all_live_clips) else list(all_live_clips.values())
n_clips = len(active_rows)
synth_hrs = sum(float(r.get("dur", 0.0)) for r in synth_clips.values()) / 3600.0
vc_hrs = sum(float(r.get("dur", 0.0)) for r in vc_clips.values()) / 3600.0
tot_hrs = sum(float(r.get("dur", 0.0)) for r in active_rows) / 3600.0
mean_utmos = (sum(float(r.get("utmos", 0.0)) for r in active_rows) / n_clips) if n_clips else 0.0
mean_cer = (sum(float(r.get("cer", 0.0)) for r in active_rows) / n_clips) if n_clips else 0.0
mean_spk = (sum(float(r.get("spk_sim", 0.0)) for r in active_rows) / n_clips) if n_clips else 0.0
by_src = {}
for r in active_rows:
    sc = r.get("src", "unknown")
    by_src[sc] = by_src.get(sc, 0.0) + float(r.get("dur", 0.0)) / 3600.0
src_str = ", ".join(f"{k}: {v:.2f}h" for k, v in sorted(by_src.items(), key=lambda x: -x[1])) or "none yet"

w_etimes = _sh("ps -eo etimes,args | awk '/[g]en_corpus[.]py work|[v]c_corpus[.]py work/ {print $1; exit}'")
elapsed_min = max(0.5, float(w_etimes) / 60.0) if w_etimes.isdigit() else 10.0
clips_per_min = n_clips / elapsed_min if n_workers > 0 else 0.0
hrs_per_hr = (tot_hrs / elapsed_min) * 60.0 if (n_workers > 0 and tot_hrs > 0) else 0.0
rem_window_min = max(0.0, 55.0 - elapsed_min)
corpus_eta = f"{rem_window_min:.0f} min left in 55m harvest window ({hrs_per_hr:.2f} yield h/hr)" if n_workers > 0 else ("MERGED & CACHED" if os.path.exists("/work/sp/distill/cache_train.pt") else "idle")
cache_info = "pending (builds after harvest)"
if os.path.exists("/work/sp/distill/cache_train.pt"):
    csz = os.path.getsize("/work/sp/distill/cache_train.pt") / 1e6
    cache_info = f"✅ READY ({csz:.1f} MB in RAM/SSD)"

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
tl = [x for x in (_trainline("v3a", 3200), _trainline("v3b", 600)) if x]
aud = _j("/work/sp/v3/audit_v2/audit.json") or {}
anc = _j("/work/sp/v3/ref/anchor_v3.json") or {}
sel_v3 = open('/work/sp/final/sakhiya_pro_v3/SELECTED_FROM.txt').read().strip() if os.path.exists('/work/sp/final/sakhiya_pro_v3/SELECTED_FROM.txt') else "v3b_checkpoint-600"
acc_rows = []
for tag, pth in [("v2 baseline", "/work/sp/v3/eval/v2/eval.json")] + [(os.path.basename(os.path.dirname(q)), q) for q in sorted(glob.glob("/work/sp/v3/eval/v3*/eval.json"))]:
    e = _j(pth)
    if e and "summary" in e:
        sm = e["summary"]
        is_sel = (tag == sel_v3)
        row_bg = " style='background:#eefbee;font-weight:600'" if is_sel else ""
        badge = " 🏆 SELECTED" if is_sel else ""
        acc_rows.append(f"<tr{row_bg}><td><b>{html.escape(tag)}{badge}</b></td><td align=right>{sm['end_slope']:+.2f}</td><td align=right>{sm['rise_frac']:.2f}</td><td align=right>{sm['spk_sim']:.3f}</td><td align=right>{sm['cer']:.3f}</td><td align=right>{sm['utmos']:.2f}</td><td align=right>{sm['f0_med_std']:.1f}</td><td align=right>{sm['rms_std_db']:.2f}</td><td align=right><b>{sm['score']:.2f}</b></td></tr>")

ep3_rep = _j("/work/sp/render/ep01_v3/take_report.json") or {}
ep3_ok = sum(1 for v in ep3_rep.values() if v.get("gate_pass")) if ep3_rep else 0
ep3_tot = len(ep3_rep) if ep3_rep else 33
ep3_summary = f" · <b>EP01 v3 Broadcast Master:</b> <code>{ep3_ok}/{ep3_tot}</code> lines pass (-14.0 LUFS)" if ep3_rep else ""

acc_html = (
    f"<div style='margin-top:6px;padding-top:6px;border-top:1px dashed #9db8f0'><b>🇳🇵 Accent-Fix (Sakhiya-Pro v3):</b> lang tag <code>{html.escape(str(aud.get('language','npi')))}</code> · instruct <code>None</code> · "
    f"anchor <code>{html.escape(str(anc.get('uid','-')))}</code> · guidance <b>{aud.get('best_gs','-')}</b> · spelling <b>{html.escape(str(aud.get('best_spelling','-')))}</b> · real-native end-slope median <b>{aud.get('real_end_slope_median', 0):+.2f} st</b>{ep3_summary}</div>"
    + ("<div style='margin-top:3px'>" + " &nbsp;|&nbsp; ".join(tl) + "</div>" if tl else "")
    + ("<table style='border-collapse:collapse;font-size:12px;margin-top:4px;width:100%' border=1 cellpadding=3><tr style='background:#e8f0ff'><th>Checkpoint</th><th>End-Slope st (−=native)</th><th>Rising Ends</th><th>Spk-Sim</th><th>CER</th><th>UTMOS</th><th>F0 Drift Hz</th><th>Level Drift dB</th><th>Accent Score</th></tr>" + "".join(acc_rows) + "</table>" if acc_rows else "")
)

# --- Panel 3: Live Shrink-Ladder Training Monitor ---
TIER_SPECS = [
    ("t15", "Sakhiya-T15 (Flagship Compact)", "<=14.5 MB"),
    ("t5",  "Sakhiya-Micro (Hero Sub-5 MB)", "<=4.8 MB"),
    ("t1",  "Sakhiya-Pico (Breakthrough Sub-1.5 MB Looped)", "<=1.45 MB"),
    ("t10", "Sakhiya-T10 (Compact 10M)", "<=9.5 MB"),
    ("t30", "Sakhiya-T30 (Teacher-Bridge 30M)", "<=29.6 MB"),
]
lad_rows = []
live_step_map = {}
for tid, tlabel, tmb in TIER_SPECS:
    lg = f"/work/sp/distill/runs/{tid}/train.log"
    ls_ = [l for l in _t(lg, 50) if l.startswith("Step ")]
    fin = os.path.exists(f"/work/sp/distill/runs/{tid}/ema_final.pt")
    lat = os.path.exists(f"/work/sp/distill/runs/{tid}/ema_latest.pt")
    best_gate = _j(f"/work/sp/tiers/{tid}/best/gate.json")
    if not best_gate:
        cands = sorted(glob.glob(f"/work/sp/tiers/{tid}/c_*/gate.json"))
        if cands:
            best_gate = _j(cands[-1])
    g_str = "🔄 awaiting 20m gate" if lat else "-"
    if best_gate:
        bs = best_gate.get("summary", {})
        p_ic = "✅" if best_gate.get("gate_pass") else "🔄"
        g_str = f"{p_ic} <b>{best_gate.get('int8_mb','-')} MB</b> · UTMOS <b>{bs.get('utmos','-')}</b> · CER <b>{bs.get('cer_ne','-')}</b> · corr {best_gate.get('int8_vs_fp32_mel_corr','-')} (s{best_gate.get('step',0)})"
    if ls_:
        last_l = ls_[-1]
        mm = re.search(r"Step (\d+) \| ([0-9.]+) it/s \| (.*?) \| ([0-9.]+)/([0-9.]+) min qat=(\d)", last_l)
        is_live = os.path.exists(lg) and (time.time() - os.path.getmtime(lg) < 120)
        st_badge = "🟢 TRAINING" if is_live else ("✅ DONE" if fin else ("⏸ PAUSED/CKPT" if lat else "⬜ QUEUED"))
        if mm:
            stp, sps_v, lss, el_m, tot_m, qat_v = mm.groups()
            lss_clean = lss.replace(" pitch ", " vel ").replace(" g ", " wlm ").replace(" fm ", " gn ")
            live_step_map[tid] = (int(stp), float(sps_v), is_live)
            rem_m = max(0.0, float(tot_m) - float(el_m))
            eta_s = f"{rem_m:.1f}m left ({el_m}/{tot_m}m)" if is_live else f"{el_m}/{tot_m}m"
            mode_tag = ("STE-INT8 QAT+WavLM" if qat_v == "1" else "Co-Distill+WavLM")
            lad_rows.append(
                f"<tr><td><b>{tlabel}</b> ({tmb})</td><td>{st_badge}</td><td align=right><b>{stp}</b></td>"
                f"<td align=right>{sps_v}</td><td>{eta_s}</td><td align=center>{mode_tag}</td>"
                f"<td><code>{html.escape(lss_clean[:90])}</code></td><td>{g_str}</td></tr>"
            )
            continue
    has_cache = os.path.exists("/work/sp/distill/cache_train.pt")
    q_state = "🔄 AWAITING CORPUS CACHE" if (n_workers > 0 and tid in ("t15", "t5", "t1")) else "⏸ STANDBY RUNG"
    q_note = "queued after priority rungs (t15, t5, t1)" if has_cache else "starts after cache_train.pt"
    lad_rows.append(
        f"<tr><td><b>{tlabel}</b> ({tmb})</td><td>{q_state}</td><td align=right>0</td>"
        f"<td align=right>-</td><td>{q_note}</td><td align=center>-</td><td><code>-</code></td><td>{g_str}</td></tr>"
    )

# --- Panel 4: Apples-to-Apples Size-vs-Quality Leaderboard ---
_v3 = _j("/work/sp/final/sakhiya_pro_v3/eval_accent.json")
SITE_BASELINES = [
    dict(name="Sakhiya-Pro v3 (v3b_s600 · Native-Nepal Lock Teacher)", size_mb="2400 (bf16)",
         utmos=round(_v3["utmos"], 3) if _v3 else 3.918, cer=round(_v3["cer"], 3) if _v3 else 0.120, sa_cer="0.095", drops=0, stutters=0,
         clicks=round(_v3["clicks"], 1) if _v3 else 25.8, rtf="GPU",
         status=(f"🇳🇵 Active v3 Teacher · spk-sim {_v3['spk_sim']:.3f} · end-slope {_v3['end_slope']:+.2f} st · EP01 {ep3_ok}/{ep3_tot}" if _v3 else "🇳🇵 Active v3 Teacher")),
    dict(name="Sakhiya-Pro v1 (ft3_s200 · 600M Teacher)", size_mb="2400 (bf16)", utmos=3.973, cer=0.064, sa_cer=0.137, drops=0, stutters=0, clicks=35, rtf="GPU", status="🏆 v1 Teacher (Anchor)"),
    dict(name="Sakhiya-Pro v2 (s2_300 · 600M Teacher)", size_mb="2400 (bf16)", utmos=3.796, cer=0.066, sa_cer=0.147, drops=0, stutters=0, clicks=44, rtf="GPU", status="✅ Frozen Pack + EP01 v2"),
    dict(name="Sakhiya-V1 (15.98 MB Flagship · Chandas v15 / v5 3.717)", size_mb=15.98, utmos=3.957, cer=0.014, sa_cer=0.025, drops=0, stutters=0, clicks=0, rtf=0.1884, status="🟢 Live Site Baseline (models_v2)"),
    dict(name="Sakhiya-V1-Nano (8.93 MB Live Site)", size_mb=8.93, utmos=3.253, cer=0.027, sa_cer=0.046, drops=0, stutters=0, clicks=0, rtf=0.0217, status="🟢 Live Site Baseline (models_voca)"),
    dict(name="Sakhiya-V1-Atom (6.51 MB Live Site)", size_mb=6.51, utmos=3.262, cer=0.017, sa_cer=0.068, drops=0, stutters=0, clicks=0, rtf=0.0189, status="🟢 Live Site Baseline (models_sano)"),
    dict(name="Sakhiya-V1-Atom-Lite (2.72 MB ONNX / 2.998 MB GGUF)", size_mb=2.72, utmos=2.944, cer=0.017, sa_cer=0.040, drops=0, stutters=0, clicks=0, rtf=0.0155, status="🟢 Live Site Baseline (models_atom_lite)"),
]
student_targets = [
    ("t15", "Student-T15 (Flagship Compact <=14.5 MB)", "~14.5"),
    ("t5",  "Sakhiya-Micro t5 (Hero Sub-5 MB <=4.8 MB)", "~4.8"),
    ("t1",  "Sakhiya-Pico t1 (Breakthrough Sub-1.5 MB Looped)", "~1.05"),
    ("t10", "Student-T10 (Compact <=9.5 MB)", "~9.5"),
    ("t30", "Student-T30 (Ladder 30 MB)", "~29.6"),
]
lb_all = list(SITE_BASELINES)
for tid, tname, est_mb in student_targets:
    g = _j(f"/work/sp/tiers/{tid}/best/gate.json")
    if not g:
        cands = sorted(glob.glob(f"/work/sp/tiers/{tid}/c_*/gate.json"))
        if cands:
            g = _j(cands[-1])
    live_info = live_step_map.get(tid)
    live_sfx = f" · 🟢 live step {live_info[0]} ({live_info[1]:.1f} it/s)" if (live_info and live_info[2]) else ""
    if g:
        s = g.get("summary", {})
        pass_str = "✅ PASS" if g.get("gate_pass") else "⚠️ GATE FAIL"
        nv = g.get("node_verify", {})
        nv_str = " · node✓" if nv.get("ok") else ""
        bs_map = s.get("by_style", {})
        nw_u = bs_map.get("news", {}).get("utmos")
        vc_u = bs_map.get("vedic_chant", {}).get("utmos")
        sty_sfx = f" · news {nw_u} / vedic {vc_u}" if (nw_u and vc_u) else ""
        lb_all.append(dict(
            name=f"{tname} (gate s{g.get('step',0)})",
            size_mb=g.get("int8_mb", est_mb),
            utmos=s.get("utmos", "-"),
            cer=s.get("cer_ne", "-"),
            sa_cer=s.get("cer_sa", "-"),
            drops=s.get("word_drop_count", s.get("word_drop_proxy", 0)),
            stutters=s.get("stutter_count", 0),
            clicks=s.get("clicks", 0),
            rtf=g.get("rtf_1thread", "-"),
            status=f"{pass_str} · corr={g.get('int8_vs_fp32_mel_corr','-')} · spk={s.get('spk_sim','-')}{sty_sfx}{nv_str}{live_sfx}"
        ))
    else:
        st_msg = f"🟢 Actively Training (step {live_info[0]} @ {live_info[1]:.1f} it/s · INT8 gate evaluating)" if (live_info and live_info[2]) else "⏸ Standby Rung (after priority tiers t15, t5, t1)"
        lb_all.append(dict(
            name=tname, size_mb=est_mb, utmos="-", cer="-", sa_cer="-",
            drops="-", stutters="-", clicks="-", rtf="-", status=st_msg
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
  (06:55 NPT Box Expiry Window: {eh}h {em:02d}m · Autopilot target: <code>{html.escape(dl_str)} UTC</code>)</div>
  <div><b>GPU:</b> <code>{html.escape(gpu)}</code> · <b>RAM:</b> <code>{html.escape(ram_info)}</code> · <b>Disk:</b> <code>{html.escape(disk_info)}</code></div>
  <div><b>15-Min HF Backup Daemon:</b> <code>{html.escape(bk_status)}</code></div>
  <ul style="list-style:none;padding-left:0;margin:6px 0 0 0">{stage_html}</ul>
</div>

<div style="background:#f5f8ff;border:1px solid #bfd4ff;border-radius:8px;padding:10px 14px;margin-bottom:10px">
  <h3 style="margin:0 0 4px 0;color:#103a8c">🎙️ Panel 2 — Active Accent-Fix (Pro v3) & Dual-Source Teacher-Corpus Status</h3>
  <div><b>Active Workers:</b> <b>{n_workers}</b> ({n_gen_workers} v3-Synth + {n_vc_workers} Seed-VC) · <b>Verified Clips:</b> <b>{n_clips}</b> ({len(synth_clips)} synth + {len(vc_clips)} VC) · <b>Audio Volume:</b> <b>{tot_hrs:.2f} hours</b> ({synth_hrs:.2f}h synth + {vc_hrs:.2f}h VC, 24 kHz mono) · <b>Speed:</b> {clips_per_min:.1f} clips/min · <b>ETA:</b> {html.escape(corpus_eta)}</div>
  <div><b>Quality Gate Means:</b> UTMOS <b>{mean_utmos:.3f}</b> · CER <b>{mean_cer:.3f}</b> · ECAPA Spk-Sim <b>{mean_spk:.3f}</b> · <b>RAM/SSD Shard Cache:</b> {html.escape(cache_info)}</div>
  <div style="color:#444;font-size:12.5px"><b>Sources Breakdown:</b> {html.escape(src_str)}</div>
  {acc_html}
</div>

<div style="background:#fffaf2;border:1px solid #f5d69b;border-radius:8px;padding:10px 14px;margin-bottom:10px">
  <h3 style="margin:0 0 6px 0;color:#7a4b00">🪜 Panel 3 — Live Shrink-Ladder Training Monitor (14.5 → 4.8 → 1.05 MB Priority Tiers)</h3>
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
for tag, ep_p in (("EP01 v3 Native-Nepal Mastered (-14 LUFS)", "/work/sp/render/ep01_v3/voiceover_ep01.mp3"),
                  ("EP01 v2 Mastered (-14 LUFS)", "/work/sp/render/ep01_v2/voiceover_ep01.mp3"),
                  ("EP01 v1 Mastered (-14 LUFS)", "/work/sp/render/ep01/voiceover_ep01.mp3"),
                  ("EP01 Repo Frozen v3", "/work/sp/repo/docs/sakhiya_pro/ep01_v3/voiceover_ep01.mp3")):
    a = _safe_aud(ep_p)
    if a is not None:
        ep_items[tag] = a
for lid in ("c3", "n1a", "n2a", "n4a", "f1", "o1"):
    lp = f"/work/sp/render/ep01_v3/lines/{lid}.wav"
    if not os.path.isfile(lp):
        lp = f"/work/sp/render/ep01_v2/lines/{lid}.wav"
    a = _safe_aud(lp)
    if a is not None:
        ep_items[f"EP01 line {lid}"] = a
if ep_items:
    sections["🎙️ NEBians News Episode 01 Broadcast Voiceover & Key Lines"] = mo.accordion(ep_items)

# 1b. Accent A/B (same line, v2 vs v3)
ab = {}
for i in range(12):
    for tag in ("v2", "v3a_checkpoint-3200", "v3b_checkpoint-600"):
        a = _safe_aud(f"/work/sp/v3/eval/{tag}/wav/l{i:02d}_s11.wav", f"{tag} line {i}")
        if a is not None:
            ab[f"line {i:02d} · {tag}"] = a
if ab:
    sections["🇳🇵 Accent A/B: v2 vs v3 (same lines)"] = mo.accordion(ab)

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
for tid in ("t15", "t5", "t1", "t10", "t30"):
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

# 4. Frozen Reference Packs in repo (Sakhiya-Pro v3, v2, v1, Sakhiya-V1 Chandas, Atom-Lite GGUF)
for vtag, vtitle in (("sakhiya_pro_v3", "Sakhiya-Pro v3 (v3b_s600 Native-Nepal Lock)"),
                     ("sakhiya_pro_v2", "Sakhiya-Pro v2 (s2_300)"),
                     ("sakhiya_pro", "Sakhiya-Pro v1 (ft3_s200 Anchor)"),
                     ("v15_v1_chandas", "Sakhiya-V1 (15.98 MB Flagship Baseline)"),
                     ("v15_ggml_atom_lite_tiny", "Sakhiya-V1-Atom-Lite (2.998 MB GGUF Tiny)")):
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
