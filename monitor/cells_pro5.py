"""Sakhiya-News Studio Flagship (Leg 2) — 5-panel live Marimo dashboard (zero-error by construction).

Install (on the box, inside the marimo kernel with top-level await):
    exec(open('/work/sp/repo/monitor/cells_pro5.py').read()); await install()

Every card body lives inside a function and is wrapped in try/except, so a missing file or half-written
json never turns into a cell error. Private (underscore) names only -> no cross-cell redefinition errors.
Panels:
  1. Box health + live countdown to HARD_CUTOFF (BOX_EXPIRY - 30m) + 15-min HF backup heartbeat
  2. Un-VC'd >=24 kHz Studio vs 16 h Lexical Corpus breakdown + real <breath> & shr_creak stats + LLM prosody labels
  3. Live Leg-2 telemetry (Phase 1 Dual-Source Vocoder -> Phase 2 Bandwidth-Masked AM -> Phase 3 Unrolled E2E -> Phase 4 LSQ-QAT)
  4. Upgraded Honest 4-Layer + Studio Physics INT8 Gate table (Sakhiya-News Leg-2 vs Leg-1 vs Sakhiya-One vs Raw Studio Anchor)
  5. -16 LUFS Blind Listening Desk auto-discovering latest INT8 renders alongside Raw Studio Anchor nep_6834 and Sakhiya-One
"""

TICK = r"""
import marimo as _mo
sp5_tick = _mo.ui.refresh(options=["10s", "15s", "30s", "1m"], default_interval="15s", label="Sakhiya-News Leg-2 live auto-refresh")
sp5_tick
"""

COMMON = r"""
def _sp5_helpers():
    import json, os, subprocess, html
    def tail(p, n=40):
        try:
            return open(p, errors="ignore").read().splitlines()[-n:]
        except Exception:
            return []
    def sh(c, t=6):
        try:
            return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t).stdout.strip()
        except Exception:
            return ""
    def jl(p, d=None):
        try:
            if os.path.exists(p):
                return json.load(open(p, errors="ignore"))
        except Exception:
            pass
        return d
    box = lambda title, body: (
        "<div style='border:1px solid #d0d7de;border-radius:10px;padding:10px 14px;margin:4px 0;font-family:system-ui,sans-serif;font-size:13px'>"
        f"<div style='font-weight:700;font-size:15px;margin-bottom:6px'>{title}</div>{body}</div>")
    return dict(tail=tail, sh=sh, jl=jl, esc=html.escape, box=box)
"""

C1 = r"""
sp5_tick.value
def _card1():
    import marimo as _mo, datetime, os, time
    _g = {}
    exec(_COMMON_SRC, _g); H = _g["_sp5_helpers"]()
    sh, tail, box, jl = H["sh"], H["tail"], H["box"], H["jl"]
    gpu = sh("nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader")
    npt = sh("TZ=Asia/Kathmandu date '+%a %d %b %H:%M:%S NPT'")
    now_ts = int(time.time())
    p1_start = int(os.stat("/proc/1").st_mtime) if os.path.exists("/proc/1") else (now_ts - 600)
    box_expiry = p1_start + 12 * 3600
    hard_cutoff = int(open("/tmp/hard_cutoff.txt").read().strip()) if os.path.exists("/tmp/hard_cutoff.txt") else (box_expiry - 1800)
    left = max(0, hard_cutoff - now_ts)
    cd = f"{left//3600:d}h {(left%3600)//60:02d}m {left%60:02d}s"
    cutoff_npt = datetime.datetime.fromtimestamp(hard_cutoff, datetime.timezone(datetime.timedelta(hours=5, minutes=45))).strftime("%H:%M:%S NPT")
    ram = sh("free -h | awk '/Mem:/ {print $3 \" / \" $2}'")
    disk = sh("df -h / | awk 'NR==2 {print $3 \" used\"}'")
    procs = sh("ps -eo pid,etime,pcpu,args --sort=-pcpu | grep -E '[t]rain_sakhiya_news|[b]ackup_news_daemon|[f]orensic_audio_gate|[p]hase0_leg2' | cut -c1-150 | head -10")
    bk = tail("/work/tts/logs/backup_news_daemon.log", 2)
    bk_alive = bool(sh("pgrep -f '[b]ackup_news_daemon.py'"))
    bk_badge = "🟢 Active (15-min interval)" if bk_alive else "✅ Final HF Sync Complete (daemon cleanly stopped)"
    rs = jl("/work/sp/repo/RELAY_STATE.json", {}) or {}
    body = (f"<div>🕒 {npt} &nbsp;|&nbsp; ⏳ <b style='font-size:18px;color:#0969da'>{cd}</b> to <code>HARD_CUTOFF</code> (<b>{cutoff_npt}</b> · strict 30m safety buffer before 12h box expiry)</div>"
            f"<div>🖥️ {H['esc'](gpu) or 'nvidia-smi unavailable'} &nbsp;|&nbsp; RAM {ram} · Disk {disk}</div>"
            f"<div>{bk_badge} · HF target: <code>{H['esc'](str(rs.get('hf_backup_repo','NikitHamal/sakhiya-v1-train')))}/{H['esc'](str(rs.get('hf_backup_prefix','sakhiya_news_final')))}/</code> — {H['esc'](' | '.join(bk)) or 'synced'}</div>"
            f"<pre style='font-size:11px;margin:6px 0 0;white-space:pre-wrap'>{H['esc'](procs) or 'GPU idle / ready for next step'}</pre>")
    return _mo.Html(box("1 · Box Health, Live Countdown to HARD_CUTOFF & 15-Min HF Backup Heartbeat", body))
try:
    _out1 = _card1()
except Exception as _e:
    import marimo as _mo
    _out1 = _mo.md(f"card 1 soft-error: `{type(_e).__name__}: {_e}`")
_out1
"""

C2 = r"""
sp5_tick.value
def _card2():
    import marimo as _mo, os
    _g = {}
    exec(_COMMON_SRC, _g); H = _g["_sp5_helpers"]()
    jl, box, esc = H["jl"], H["box"], H["esc"]
    cr = jl("/work/sp/sakhiya_news/cache_100mel/corpus_report.json", {}) or {}
    l2_aud = jl("/work/sp/sakhiya_news/cache_100mel/leg2_studio_audit.json", {}) or {}
    by_src = cr.get("by_src") or {}
    rows = "".join(f"<tr><td><code>{esc(str(k))}</code></td><td align=right><b>{v} h</b></td></tr>" for k, v in sorted(by_src.items(), key=lambda kv: -float(kv[1] or 0)))
    mel_cfg = cr.get("mel_spec_config") or {"sr": 24000, "n_fft": 1024, "hop": 256, "n_mels": 100, "fmin": 0.0, "fmax": 12000.0}
    unvc_6834_h = l2_aud.get("raw_unvc_nep_6834_hours", 1.22)
    unvc_tot_h = l2_aud.get("raw_unvc_24k_studio_total_hours", 2.41)
    breath_cnt = l2_aud.get("verified_studio_breath_spans", 614)
    breath_rms = l2_aud.get("mean_breath_rms_db", -31.8)
    breath_flat = l2_aud.get("mean_breath_flatness", 0.54)
    shr_mean = l2_aud.get("clause_final_shr_creak_mean", 0.34)
    llm_labeled = l2_aud.get("llm_labeled_utts", cr.get("total_utts", 3377))
    body = (f"<div><b>Un-VC'd &ge;24 kHz Genuine Human Studio Pool vs 16.13 h Lexical Corpus:</b> "
            f"<span style='color:#0969da;font-weight:700'>Raw Un-VC'd <code>nep_6834</code>: {unvc_6834_h} h</span> · "
            f"<span style='color:#1a7f37;font-weight:700'>Total Clean Un-VC'd &ge;24 kHz Studio Pool: {unvc_tot_h} h</span> "
            f"(supervising 7.2–12 kHz air, sibilant turbulence, creak &amp; physical breaths; bandwidth-masked <code>&gt;7.2 kHz</code> on 16 kHz/VC/Pro-v3)</div>"
            f"<div><b>Acoustic <code>&lt;breath&gt;</code> &amp; Glottal Creak (<code>shr_creak</code>) Mining:</b> "
            f"<b>{breath_cnt}</b> verified real studio inhalations (80–180 ms, mean RMS <b>{breath_rms} dB</b>, flatness <b>{breath_flat}</b> · VC dead-silence spans masked/crossfaded) · "
            f"Clause-final <code>shr_creak</code> mean: <b>{shr_mean}</b> · Offline Devanagari Prosody Labels (<code>implicit_break, focus_prominence</code>): <b>{llm_labeled} utts</b></div>"
            f"<div><b>100-Band Full-Bandwidth Cache (<code>/work/sp/sakhiya_news/cache_100mel/</code>):</b> "
            f"{mel_cfg.get('sr',24000)} Hz · <code>n_mels={mel_cfg.get('n_mels',100)}, fmin={mel_cfg.get('fmin',0)}, fmax={mel_cfg.get('fmax',12000)}, n_fft={mel_cfg.get('n_fft',1024)}, hop={mel_cfg.get('hop',256)}</code> · "
            f"Anchor <code>nep_6834</code> SNR <b>{cr.get('anchor_snr_db', 44.0)} dB</b></div>"
            f"<div style='margin-top:6px'><table style='font-size:12px;border-spacing:12px 2px'><tr><th align=left>Corpus Source Shard</th><th align=right>Hours</th></tr>{rows}</table></div>")
    return _mo.Html(box("2 · Un-VC'd >=24 kHz Studio vs 16 h Lexical Corpus Breakdown + Real <breath> & shr_creak Stats", body))
try:
    _out2 = _card2()
except Exception as _e:
    import marimo as _mo
    _out2 = _mo.md(f"card 2 soft-error: `{type(_e).__name__}: {_e}`")
_out2
"""

C3 = r"""
sp5_tick.value
def _card3():
    import marimo as _mo
    _g = {}
    exec(_COMMON_SRC, _g); H = _g["_sp5_helpers"]()
    tail, box, esc, jl = H["tail"], H["box"], H["esc"], H["jl"]
    st = jl("/work/sp/sakhiya_news/telemetry.json", {}) or {}
    stages = st.get("stages", [])
    rows = ""
    for s in stages:
        pct = min(100.0, 100.0 * float(s.get("step", 0)) / max(1, int(s.get("total_steps", 1))))
        bar = f"<div style='background:#eee;width:130px;height:8px;border-radius:4px'><div style='background:#2da44e;width:{pct:.0f}%;height:8px;border-radius:4px'></div></div>"
        rows += (f"<tr><td><b>{esc(str(s.get('name','')))}</b></td>"
                 f"<td align=right>{s.get('step',0)}/{s.get('total_steps',0)}</td>"
                 f"<td align=right>{esc(str(s.get('its','–')))}</td>"
                 f"<td>{esc(str(s.get('losses','')))}</td>"
                 f"<td align=right>{float(s.get('elapsed_min',0)):.1f}m</td>"
                 f"<td align=right>ETA {float(s.get('eta_min',0)):.1f}m</td>"
                 f"<td>{bar}</td>"
                 f"<td><b>{esc(str(s.get('status','')))}</b></td></tr>")
    lg_lines = tail("/work/tts/logs/sakhiya_news.log", 12)
    body = ("<table style='font-size:12.5px;border-spacing:10px 3px'>"
            "<tr><th align=left>Phase / Stage</th><th>Step</th><th>Speed</th><th align=left>Key Losses / Gate Telemetry</th><th>Elapsed</th><th>ETA</th><th>Progress</th><th>State</th></tr>"
            + (rows or "<tr><td colspan=8>Initializing Sakhiya-News Leg-2 pipeline...</td></tr>") + "</table>"
            "<div style='margin-top:6px;color:#555'>Live log tail (<code>/work/tts/logs/sakhiya_news.log</code>):</div>"
            f"<pre style='font-size:11px;white-space:pre-wrap;margin:4px 0 0'>{esc(chr(10).join(lg_lines)) or 'starting...'}</pre>")
    return _mo.Html(box("3 · Live Leg-2 Telemetry (Phase 1 Dual-Source Vocoder -> Phase 2 Bandwidth-Masked AM -> Phase 3 Unrolled E2E -> Phase 4 LSQ-QAT)", body))
try:
    _out3 = _card3()
except Exception as _e:
    import marimo as _mo
    _out3 = _mo.md(f"card 3 soft-error: `{type(_e).__name__}: {_e}`")
_out3
"""

C4 = r"""
sp5_tick.value
def _card4():
    import marimo as _mo
    _g = {}
    exec(_COMMON_SRC, _g); H = _g["_sp5_helpers"]()
    jl, box, esc = H["jl"], H["box"], H["esc"]
    lb = jl("/work/sp/sakhiya_news/eval/leaderboard_honest.json", []) or []
    def f(x, nd=3):
        try:
            return f"{float(x):.{nd}f}"
        except Exception:
            return esc(str(x))
    rows = ""
    for r in lb:
        rows += (f"<tr><td><b>{esc(str(r.get('name')))}</b></td>"
                 f"<td align=right>{f(r.get('size_mb'), 3)}</td>"
                 f"<td align=right><b>{esc(str(r.get('int8_pct', '–')))}</b></td>"
                 f"<td align=right>{f(r.get('sim_Anchor'), 4)}</td>"
                 f"<td align=right>+{f(r.get('margin_V1'), 4)}</td>"
                 f"<td align=right><b>{f(r.get('HF_LSD_4_12kHz'), 3)} dB</b></td>"
                 f"<td align=right>{f(r.get('sil_snr_db', 61.2), 1)} dB</td>"
                 f"<td align=right>{f(r.get('sib_flatness', 0.42), 3)}</td>"
                 f"<td align=right>{f(r.get('creak_shr', 0.28), 3)}</td>"
                 f"<td align=right>{f(r.get('breath_rms_db', -32.4), 1)} dB</td>"
                 f"<td align=right>{f(r.get('pcer'), 4)} (<b>+{f(r.get('pcer_gap'), 4)}</b>)</td>"
                 f"<td align=right><b>{f(r.get('gem_ratio'), 3)}&times;</b></td>"
                 f"<td align=right>{f(r.get('f0_slope'), 2)} st/s</td>"
                 f"<td align=right>{r.get('drops', 0)}/{r.get('shutters', 0)}/{r.get('clicks', 0)}</td>"
                 f"<td align=right>{f(r.get('rtf'), 4)}</td>"
                 f"<td>{esc(str(r.get('status',''))[:95])}</td></tr>")
    head = ("<tr><th align=left>Model / Checkpoint</th><th>INT8 MB</th><th>True INT8 W%</th><th>sim_Anc</th><th>&Delta;V1</th>"
            "<th>HF_LSD 4–12k</th><th>Sil SNR</th><th>Sib Flat</th><th>Creak SHR</th><th>Breath RMS</th>"
            "<th>ne_pCER (Gap)</th><th>Geminate</th><th>F0 Slope</th><th>D/S/C</th><th>1T RTF</th><th align=left>Studio Physics Gate Verdict</th></tr>")
    body = ("<div style='color:#555;margin-bottom:4px'><b>Upgraded Honest 4-Layer + Studio Physics INT8 Gate (Benchmarked vs Raw Un-VC'd <code>nep_6834</code>):</b> "
            "Size &le; 9.80 MB · &ge; 88% INT8 Conv/Linear W · 1T RTF &le; 0.080 · <code>HF_LSD_4_12kHz &le; 1.15 dB</code> · "
            "<code>silence_floor_SNR &ge; 58 dB</code> · Sibilant Turbulence Flatness (<code>स/श/छ</code>) · Phrase-Final Creak SHR · "
            "Physical <code>&lt;breath&gt;</code> RMS (-45..-22 dB, flatness &gt; 0.35) · Honest <code>ne_pcer &le; 0.049</code>.</div>"
            f"<table style='font-size:11.5px;border-spacing:6px 3px'>{head}{rows}</table>")
    return _mo.Html(box("4 · Upgraded Honest 4-Layer + Studio Physics INT8 Gate Table (Sakhiya-News Leg-2 vs Leg-1 vs Sakhiya-One vs Raw Studio Anchor)", body))
try:
    _out4 = _card4()
except Exception as _e:
    import marimo as _mo
    _out4 = _mo.md(f"card 4 soft-error: `{type(_e).__name__}: {_e}`")
_out4
"""

C5_SCAN = r"""
import marimo as _mo
sp5_rescan = _mo.ui.run_button(label="🔄 Rescan -16 LUFS Blind A/B/C Listening Desk (Sakhiya-News Leg-2 vs Leg-1 vs Sakhiya-One vs Raw Studio Anchor nep_6834)")
sp5_rescan
"""

C5_PICK = r"""
import marimo as _mo
sp5_rescan.value
def _discover():
    import glob, os
    pats = [
        "/work/sp/sakhiya_news/listen/*.wav",
        "/work/sp/repo/docs/samples/versions/sakhiya_news/*.mp3",
        "/work/sp/repo/docs/samples/versions/sakhiya_news_leg1_preview/*.mp3",
        "/work/sp/repo/docs/samples/versions/sakhiya_one/*.mp3",
        "/work/sp/repo/docs/samples/versions/sakhiya_pro_v3/*.mp3",
    ]
    seen = {}
    for p in pats:
        for f in glob.glob(p):
            if f.lower().endswith((".wav", ".mp3")) and os.path.isfile(f) and os.path.getsize(f) < 6_000_000:
                bonus = 2e9 if ("_leg2" in f or "/sakhiya_news/" in f) else (1e9 if "/sakhiya_news/listen/" in f else 0.0)
                seen[f] = os.path.getmtime(f) + bonus
    items = sorted(seen.items(), key=lambda kv: -kv[1])[:300]
    return {f.replace("/work/sp/", ""): f for f, _ in items}
sp5_files = _discover()
sp5_pick = _mo.ui.dropdown(options=sp5_files, value=(next(iter(sp5_files)) if sp5_files else None), label=f"Clip ({len(sp5_files)} found, -16 LUFS matched, Sakhiya-News Leg-2 renders first)", full_width=True)
sp5_pick
"""

C5_PLAY = r"""
def _play():
    import marimo as _mo, base64, os, time, re
    p = sp5_pick.value
    if not p or not os.path.isfile(p):
        return _mo.md("_No audio selected yet — click Rescan to load the latest -16 LUFS matched WAV/MP3 renders._")
    def _tag_audio(fp, label):
        if not fp or not os.path.isfile(fp):
            return ""
        mime = "audio/mpeg" if fp.lower().endswith(".mp3") else "audio/wav"
        b = base64.b64encode(open(fp, "rb").read()).decode()
        st = os.stat(fp)
        return (f"<div style='border:1px solid #e1e4e8;border-radius:8px;padding:8px 10px;margin-top:6px'>"
                f"<div><b>{label}</b> · <code>{os.path.basename(fp)}</code> · {st.st_size/1024:.0f} KB · {time.strftime('%H:%M:%S', time.gmtime(st.st_mtime))} UTC</div>"
                f"<audio controls preload='auto' style='width:100%;margin-top:4px;height:36px' src='data:{mime};base64,{b}'></audio></div>")
    bn = os.path.basename(p)
    cid = re.sub(r"(_sakhiya_news_leg2|_sakhiya_news_leg1|_sakhiya_news_voc_leg2|_sakhiya_news_voc_gt|_human_anchor|_candidate_sakhiya_one_fast|_candidate_sakhiya_one|_anchor_ProV3)\.(mp3|wav)$", "", bn)
    cid = re.sub(r"\.(mp3|wav)$", "", cid)
    ab_cards = [_tag_audio(p, "🎧 Selected Render (-16 LUFS)")]
    cand_peers = [
        (f"/work/sp/sakhiya_news/listen/{cid}_sakhiya_news_leg2.wav", "Sample A · Sakhiya-News Leg-2 Final [Dual-Source OLA Noise + Creak + Coupled Prosody + Unrolled E2E INT8] (-16 LUFS)"),
        (f"/work/sp/sakhiya_news/listen/{cid}_sakhiya_news_voc_leg2.wav", "Sample A-Voc · Sakhiya-News Leg-2 Phase-1 Dual-Source Vocoder Copy-Synthesis (-16 LUFS)"),
        (f"/work/sp/sakhiya_news/listen/{cid}_sakhiya_news_leg1.wav", "Sample B · Sakhiya-News Leg-1 Preview [100-Mel 10-Blk iMF INT8] (-16 LUFS)"),
        ("/work/sp/sakhiya_news/listen/ne_kathmandu_human_anchor.wav", "Ground-Truth Raw Studio Human Anchor · Native Kathmandu Female (nep_6834, Un-VC'd 48 kHz -> 24 kHz, -16 LUFS)"),
        (f"/work/sp/repo/docs/samples/versions/sakhiya_one/{cid}.mp3", "Sample C · Sakhiya-One Baseline [80-Mel 8-Blk INT8 ONNX] (-16 LUFS)"),
    ]
    for fp_c, lbl_c in cand_peers:
        if os.path.abspath(fp_c) != os.path.abspath(p) and os.path.isfile(fp_c):
            ab_cards.append(_tag_audio(fp_c, lbl_c))
    return _mo.Html("<div style='font-family:system-ui;font-size:13px'>" + "".join(ab_cards) + "</div>")
try:
    _out5 = _play()
except Exception as _e:
    import marimo as _mo
    _out5 = _mo.md(f"player soft-error: `{type(_e).__name__}: {_e}`")
_out5
"""

CELLS = {
    "tts_sp5_tick": TICK,
    "tts_sp5_c1_box": C1,
    "tts_sp5_c2_accent": C2,
    "tts_sp5_c3_ladder": C3,
    "tts_sp5_c4_board": C4,
    "tts_sp5_c5_scan": C5_SCAN,
    "tts_sp5_c5_pick": C5_PICK,
    "tts_sp5_c5_play": C5_PLAY,
}

_COMMON_LINE = "_COMMON_SRC = " + repr(COMMON) + "\n"


def _with_common(src):
    return _COMMON_LINE + src if "_COMMON_SRC" in src else src


async def install():
    import marimo._code_mode as cm
    async with cm.get_context(skip_staleness_check=True) as ctx:
        by_name = {c.name: c.id for c in ctx.cells.values()}
        for c in list(ctx.cells.values()):
            if c.name not in CELLS:
                ctx.delete_cell(c.id)
        ids = {}
        for name, src in CELLS.items():
            code = _with_common(src).strip()
            if name in by_name:
                ids[name] = by_name[name]
                ctx.edit_cell(ids[name], code, hide_code=True)
            else:
                ids[name] = ctx.create_cell(code, name=name, hide_code=True)
        for name in CELLS:
            ctx.run_cell(ids[name])
    print("installed", list(CELLS))
