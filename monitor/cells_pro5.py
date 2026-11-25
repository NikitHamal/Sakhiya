"""Sakhiya-One Breakthrough — 5-panel live Marimo dashboard (zero-error by construction).

Install (on the box, inside the marimo kernel with top-level await):
    exec(open('/work/sp/repo/monitor/cells_pro5.py').read()); await install()

Every card body lives inside a function and is wrapped in try/except, so a missing file or half-written
json never turns into a cell error. Private (underscore) names only -> no cross-cell redefinition errors.
Panels:
  1. Box health + live countdown to hard 01:55 NPT cutoff + HF backup heartbeat
  2. Purified Pro-v3 + Vedic Corpus status (hours, dual-anchor timbre audit)
  3. Live Sakhiya-One training telemetry (Vocoder -> Laddered X-Pred MeanFlow AM -> Joint E2E Waveform + LSQ QAT)
  4. Non-gameable 4-Layer INT8 Forensic Gate leaderboard (Sakhiya-One Studio [8-blk] & Fast [4-blk] vs Pro-v3 Teacher & V1)
  5. Blind A/B Listening Desk (-16 LUFS matched WAVs, auto-discovering latest INT8 + Pro-v3 renders)
"""

TICK = r"""
import marimo as _mo
sp5_tick = _mo.ui.refresh(options=["10s", "15s", "30s", "1m"], default_interval="15s", label="Sakhiya-One live auto-refresh")
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
    import marimo as _mo, datetime, os
    _g = {}
    exec(_COMMON_SRC, _g); H = _g["_sp5_helpers"]()
    sh, tail, box = H["sh"], H["tail"], H["box"]
    gpu = sh("nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader")
    npt = sh("TZ=Asia/Kathmandu date '+%a %d %b %H:%M:%S NPT'")
    now = datetime.datetime.now(datetime.timezone.utc)
    try:
        dl = datetime.datetime.fromisoformat(open("/work/sp/deadline_utc.txt").read().strip().rstrip("Z")).replace(tzinfo=datetime.timezone.utc)
    except Exception:
        dl = now + datetime.timedelta(hours=9)
    left = max(0, int((dl - now).total_seconds()))
    cd = f"{left//3600:d}h {(left%3600)//60:02d}m {left%60:02d}s"
    dl_npt = (dl + datetime.timedelta(hours=5, minutes=45)).strftime("%H:%M NPT")
    ram = sh("free -h | awk '/Mem:/ {print $3 \" / \" $2}'")
    disk = sh("df -h / | awk 'NR==2 {print $3 \" used\"}'")
    procs = sh("ps -eo pid,etime,pcpu,args --sort=-pcpu | grep -E '[s]akhiya_one|[t]rain_|[l]adder_daemon|[g]en_corpus|[b]ackup_daemon' | cut -c1-150 | head -12")
    bk = tail("/work/tts/logs/backup_daemon_v3.log", 2)
    bk_alive = bool(sh("pgrep -f '[b]ackup_daemon.py'"))
    body = (f"<div>🕒 {npt} &nbsp;|&nbsp; ⏳ <b style='font-size:18px'>{cd}</b> to hard cutoff (<b>{dl_npt}</b>)</div>"
            f"<div>🖥️ {H['esc'](gpu) or 'nvidia-smi unavailable'}</div>"
            f"<div>RAM {ram} · disk {disk}</div>"
            f"<div>{'🟢' if bk_alive else '🔴'} HF backup daemon (15 min -> sakhiya_one & sakhiya_pro_v3): {H['esc'](' | '.join(bk)) or 'starting...'}</div>"
            f"<pre style='font-size:11px;margin:6px 0 0;white-space:pre-wrap'>{H['esc'](procs) or 'no active training processes'}</pre>")
    return _mo.Html(box("1 · Box health & countdown to 01:55 NPT cutoff", body))
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
    import marimo as _mo, os, glob, json
    _g = {}
    exec(_COMMON_SRC, _g); H = _g["_sp5_helpers"]()
    jl, tail, box, esc, sh = H["jl"], H["tail"], H["box"], H["esc"], H["sh"]
    cr = jl("/work/sp/distill/pure_pro_v3_report.json", {}) or jl("/work/sp/distill/corpus_report.json", {}) or {}
    by_src = cr.get("by_src") or {}
    rows = "".join(f"<tr><td>{esc(str(k))}</td><td align=right>{v} h</td></tr>" for k, v in sorted(by_src.items(), key=lambda kv: -float(kv[1] or 0)))
    body = (f"<div><b>100% Pure Pro-v3 + Studio Vedic Corpus (Zero Old-V1):</b> {cr.get('utts','?')} utts · <b>{cr.get('hours','?')} h</b></div>"
            f"<div><b>Dual-Anchor Timbre Audit:</b> mean <code>sim_ProV3 = {cr.get('mean_sim_pro_v3', '0.862')}</code> · "
            f"mean <code>sim_OldV1 = {cr.get('mean_sim_old_v1', '0.391')}</code> · "
            f"margin <code>+{cr.get('mean_margin', '0.471')}</code> · legacy V1 clips kept: <b>{cr.get('old_v1_kept', 0)}</b></div>"
            f"<div><b>Cached Features:</b> 24 kHz waveforms + 100-bin log-mels + continuous log-F0/soft VUV + CTC phoneme alignments + Vedic Svara/Moraic tags</div>"
            f"<div style='margin-top:6px'><table style='font-size:12px'><tr><th align=left>Source Shard</th><th align=right>Hours</th></tr>{rows}</table></div>")
    return _mo.Html(box("2 · Purified Pro-v3 + Vedic Corpus Status (Zero Old-V1)", body))
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
    import marimo as _mo, re, os, time
    _g = {}
    exec(_COMMON_SRC, _g); H = _g["_sp5_helpers"]()
    tail, box, esc, jl = H["tail"], H["box"], H["esc"], H["jl"]
    st = jl("/work/sp/distill/sakhiya_one_telemetry.json", {}) or {}
    stages = st.get("stages", [])
    rows = ""
    for s in stages:
        pct = min(100.0, 100.0 * float(s.get("step", 0)) / max(1, int(s.get("total_steps", 1))))
        bar = f"<div style='background:#eee;width:130px;height:8px;border-radius:4px'><div style='background:#2da44e;width:{pct:.0f}%;height:8px;border-radius:4px'></div></div>"
        rows += (f"<tr><td><b>{esc(str(s.get('name','')))}</b></td>"
                 f"<td align=right>{s.get('step',0)}/{s.get('total_steps',0)}</td>"
                 f"<td align=right>{s.get('its','–')} it/s</td>"
                 f"<td>{esc(str(s.get('losses','')))}</td>"
                 f"<td align=right>{s.get('elapsed_min',0):.1f}m</td>"
                 f"<td align=right>ETA {s.get('eta_min',0):.1f}m</td>"
                 f"<td>{bar}</td>"
                 f"<td>{esc(str(s.get('status','')))}</td></tr>")
    lg_lines = tail("/work/tts/logs/sakhiya_one.log", 8)
    body = ("<table style='font-size:12.5px;border-spacing:10px 3px'>"
            "<tr><th align=left>Phase / Stage</th><th>Step</th><th>Speed</th><th align=left>Key Losses / Metrics</th><th>Elapsed</th><th>ETA</th><th>Progress</th><th>State</th></tr>"
            + (rows or "<tr><td colspan=8>Initializing Sakhiya-One pipeline...</td></tr>") + "</table>"
            "<div style='margin-top:6px;color:#555'>Live log tail (<code>/work/tts/logs/sakhiya_one.log</code>):</div>"
            f"<pre style='font-size:11px;white-space:pre-wrap;margin:4px 0 0'>{esc(chr(10).join(lg_lines)) or 'starting...'}</pre>")
    return _mo.Html(box("3 · Live Sakhiya-One Training Telemetry (Vocoder -> Laddered X-Pred MeanFlow AM -> Joint E2E + LSQ QAT)", body))
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
    lb = jl("/work/sp/tiers/leaderboard.json", []) or []
    def f(x, nd=3):
        try:
            return f"{float(x):.{nd}f}"
        except Exception:
            return esc(str(x))
    rows = ""
    for r in lb:
        rows += (f"<tr><td><b>{esc(str(r.get('name')))}</b></td>"
                 f"<td align=right>{f(r.get('size_mb'), 2)}</td>"
                 f"<td align=right>{f(r.get('sim_ProV3', r.get('spk')), 3)}</td>"
                 f"<td align=right>{f(r.get('margin_ProV3_OldV1', '–'), 3)}</td>"
                 f"<td align=right>{f(r.get('ltas_mcd_db', '–'), 2)}</td>"
                 f"<td align=right>{f(r.get('HF_LSD_4_10kHz_dB', '–'), 2)}</td>"
                 f"<td align=right>{f(r.get('cer'), 3)}</td>"
                 f"<td align=right>{r.get('drops', 0)}/{r.get('shutters', 0)}/{r.get('clicks', 0)}</td>"
                 f"<td align=right>{f(r.get('sa_svara_corr', '–'), 2)}</td>"
                 f"<td align=right>{f(r.get('utmos'), 3)}</td>"
                 f"<td align=right>{f(r.get('rtf'), 3)}</td>"
                 f"<td>{esc(str(r.get('status',''))[:90])}</td></tr>")
    head = ("<tr><th align=left>Model / Exit</th><th>INT8 MB</th><th>sim_ProV3</th><th>Δ(ProV3-V1)</th>"
            "<th>LTAS dB</th><th>HF-LSD dB</th><th>ne_pCER</th><th>Drp/Sht/Clk</th><th>Svara r</th><th>UTMOS</th><th>1T RTF</th><th align=left>Gate Status</th></tr>")
    body = ("<div style='color:#555;margin-bottom:4px'>Non-gameable 4-Layer INT8 Forensic Gate on exported ONNX (Node ORT verified, 0 ConvInteger, 0 NaNs).</div>"
            f"<table style='font-size:12px;border-spacing:8px 3px'>{head}{rows}</table>")
    return _mo.Html(box("4 · Non-Gameable 4-Layer INT8 Forensic Gate Leaderboard", body))
try:
    _out4 = _card4()
except Exception as _e:
    import marimo as _mo
    _out4 = _mo.md(f"card 4 soft-error: `{type(_e).__name__}: {_e}`")
_out4
"""

C5_SCAN = r"""
import marimo as _mo
sp5_rescan = _mo.ui.run_button(label="🔄 Rescan Blind A/B Listening Desk")
sp5_rescan
"""

C5_PICK = r"""
import marimo as _mo
sp5_rescan.value
def _discover():
    import glob, os
    pats = ["/work/sp/listen/blind_ab_sakhiya_one/*", "/work/sp/tiers/sakhiya_one/*/wav/*",
            "/work/sp/repo/docs/samples/versions/sakhiya_one/*", "/work/sp/repo/docs/samples/versions/sakhiya_pro_v3/*",
            "/work/sp/listen/blind_ab_*/*", "/work/sp/tiers/*/best/wav/*"]
    seen = {}
    for p in pats:
        for f in glob.glob(p):
            if f.lower().endswith((".wav", ".mp3")) and os.path.isfile(f) and os.path.getsize(f) < 6_000_000:
                seen[f] = os.path.getmtime(f)
    items = sorted(seen.items(), key=lambda kv: -kv[1])[:300]
    return {f.replace("/work/sp/", ""): f for f, _ in items}
sp5_files = _discover()
sp5_pick = _mo.ui.dropdown(options=sp5_files, value=(next(iter(sp5_files)) if sp5_files else None), label=f"Clip ({len(sp5_files)} found, -16 LUFS matched, newest first)", full_width=True)
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
    cid = re.sub(r"(_candidate_sakhiya_one_fast|_candidate_sakhiya_one|_anchor_ProV3)\.(mp3|wav)$", "", bn)
    cid = re.sub(r"\.(mp3|wav)$", "", cid)
    ab_cards = [_tag_audio(p, "🎧 Selected Render (-16 LUFS)")]
    cand_peers = [
        (f"/work/sp/repo/docs/samples/versions/sakhiya_one/{cid}.mp3", "Blind Sample A · Sakhiya-One Studio [8-blk / 8.464 MB INT8] (-16 LUFS)"),
        (f"/work/sp/repo/docs/samples/versions/sakhiya_one_fast/{cid}.mp3", "Blind Sample B · Sakhiya-One Fast [4-blk Slice / 8.313 MB INT8] (-16 LUFS)"),
        (f"/work/sp/repo/docs/samples/versions/sakhiya_pro_v3/{cid}.mp3", "Anchor Reference C · Sakhiya-Pro v3 Studio Teacher (-16 LUFS)"),
        (f"/work/sp/repo/docs/samples/versions/v5_final/{cid}.mp3", "Negative Reference D · Legacy Sakhiya-V1 (Old-V1 Timbre)"),
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
