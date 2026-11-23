"""Sakhiya-Pro Studio card (bottom of the MoLab notebook): type any text -> Sakhiya-Pro voice -> captioned MP4.
Install (idempotent, touches ONLY the tts_sakhiya_studio* cells, appended at the bottom) via marimo._code_mode:
   exec(open('/work/sp/repo/monitor/cells_studio.py').read()); await install()
Cells:
  tts_sakhiya_studio         : the form (text prompt, model, takes, pace, format, title, Generate button)
  tts_sakhiya_studio_launch  : starts scripts/sakhiya_pro_studio.py detached when the form is submitted (no output)
  tts_sakhiya_studio_status  : live progress (auto-refresh), per-sentence take log
  tts_sakhiya_studio_result  : the finished MP4 player + MP4 / MP3 / WAV downloads + recent renders
"""

FORM = r'''
import marimo as _mo, os as _os, glob as _glob
def _studio_models():
    o = {}
    for d in sorted(_glob.glob("/work/sp/final/*/model.safetensors")):
        o["Sakhiya-Pro final · " + _os.path.basename(_os.path.dirname(d))] = _os.path.dirname(d)
    cks = sorted(_glob.glob("/work/sp/exp/*/checkpoint-*/model.safetensors"), key=_os.path.getmtime, reverse=True)[:6]
    for d in cks:
        p = _os.path.dirname(d); o["checkpoint · " + p.split("/exp/")[1]] = p
    return o
_studio_opts = _studio_models() or {"Sakhiya-Pro final": "/work/sp/final/sakhiya_pro_v1"}
_studio_default = "Sakhiya-Pro final · sakhiya_pro_v1" if "Sakhiya-Pro final · sakhiya_pro_v1" in _studio_opts else list(_studio_opts)[0]
_studio_text = _mo.ui.text_area(
    value="नमस्कार, म सखिया हुँ। आजको मुख्य समाचारमा Anthropic ले नयाँ मोडल सार्वजनिक गरेको छ।\n\nकाठमाडौँमा आज मौसम सफा रहनेछ।",
    rows=7, full_width=True, label="✍️ Text prompt — Nepali, Sanskrit (ॐ, ॥) or Nepali + English tech terms. Blank line = longer pause.")
studio_form = _mo.md("""{text}

{model}  {lang}  {fmt}

{takes}  {pace}  {title}
""").batch(
    text=_studio_text,
    model=_mo.ui.dropdown(_studio_opts, value=_studio_default, label="Model"),
    lang=_mo.ui.dropdown({"Auto (Nepali / Sanskrit by text)": "auto", "Nepali": "npi", "Sanskrit / Vedic": "sa"}, value="Auto (Nepali / Sanskrit by text)", label="Language"),
    fmt=_mo.ui.dropdown({"16:9 landscape 1280×720": "land", "9:16 vertical 720×1280": "vert"}, value="16:9 landscape 1280×720", label="Video"),
    takes=_mo.ui.slider(1, 5, value=3, label="Best-of-N takes"),
    pace=_mo.ui.slider(0.85, 1.15, step=0.01, value=1.0, label="Pace"),
    title=_mo.ui.text(value="NEBians News", label="Video title"),
).form(submit_button_label="🎬 Generate MP4", bordered=True, show_clear_button=False)
_mo.vstack([
    _mo.md("## 🎙️ Sakhiya-Pro Studio — text → natural voice → MP4"),
    _mo.md("Type anything, press **Generate MP4**. Best-of-N takes are scored by ASR / clicks / Hindi-leak / UTMOS, mastered to **-14 LUFS** "
           "(80 Hz HPF + warmth/presence EQ + gentle compression) and rendered into a captioned MP4. Progress and the finished video appear below."),
    studio_form,
])
'''

LAUNCH = r'''
studio_launched = None
if studio_form.value:
    import os as _os, json as _json, time as _time, subprocess as _sp
    _id = _time.strftime("%Y%m%d_%H%M%S")
    _job = f"/work/sp/studio/jobs/{_id}"; _os.makedirs(_job, exist_ok=True)
    _v = studio_form.value
    _json.dump(dict(text=_v["text"], model=_v["model"], lang=_v["lang"], takes=int(_v["takes"]), pace=float(_v["pace"]), fmt=_v["fmt"], title=_v["title"], steps=32),
               open(f"{_job}/job.json", "w"), ensure_ascii=False)
    _json.dump(dict(stage="queued", done=0, total=0, log=[], started=_time.time()), open(f"{_job}/status.json", "w"))
    _lf = open(f"{_job}/worker.log", "w")
    _sp.Popen(["python3", "/work/sp/repo/scripts/sakhiya_pro_studio.py", _job], stdout=_lf, stderr=_sp.STDOUT, stdin=_sp.DEVNULL, start_new_session=True)
    open("/work/sp/studio/latest.txt", "w").write(_id)
    studio_launched = _id
'''

TICK = '''import marimo as _mo
studio_tick = _mo.ui.refresh(options=["3s", "5s", "10s"], default_interval="5s", label="Studio live refresh")
studio_tick'''

# status/result use mo.state so the video player is NOT re-created on every refresh tick
STATUS = r'''
import marimo as _mo
studio_done_get, studio_done_set = _mo.state(None)
studio_shown = {"id": None}
'''

STATUS_VIEW = r'''
studio_tick.value; studio_launched
import os as _os, json as _json, time as _time, html as _html
import marimo as _mo
_lt = "/work/sp/studio/latest.txt"
_id = open(_lt).read().strip() if _os.path.exists(_lt) else None
_out = _mo.md("*No render yet — write some text above and press **Generate MP4**.*")
if _id:
    _job = f"/work/sp/studio/jobs/{_id}"
    try: _s = _json.load(open(f"{_job}/status.json"))
    except Exception: _s = dict(stage="starting", done=0, total=0, log=[], started=_time.time())
    _tot = max(1, _s.get("total") or 1); _pct = int(100 * (_s.get("done", 0)) / _tot)
    _stage = _s.get("stage", "?"); _el = (_s.get("finished") or _time.time()) - _s.get("started", _time.time())
    if _stage == "done": _pct = 100
    _icon = "✅" if _stage == "done" else ("❌" if _stage == "error" else "⏳")
    _bar = f'<div style="background:#1e293b;border-radius:6px;height:14px;width:100%"><div style="background:#14b8a6;height:14px;border-radius:6px;width:{_pct}%"></div></div>'
    _log = "<br>".join(_html.escape(l) for l in _s.get("log", [])[-12:])
    _err = f'<p style="color:#c00"><b>{_html.escape(str(_s.get("error")))}</b><br><code>{_html.escape("".join(open(_job + "/worker.log", errors="ignore").readlines()[-6:]) if _os.path.exists(_job + "/worker.log") else "")}</code></p>' if _stage == "error" else ""
    _out = _mo.Html(f"""<div style="font-family:system-ui;font-size:14px;line-height:1.5">
<b>{_icon} Job {_id}</b> · {_html.escape(_stage)} · {_pct}% · {_el:.0f}s elapsed{(' · ' + str(_s.get('duration_s')) + 's audio · ' + str(_s.get('lufs')) + ' LUFS') if _stage == 'done' else ''}
{_bar}<p style="font-family:ui-monospace,monospace;font-size:12px;color:#555">{_log}</p>{_err}</div>""")
    if _stage == "done" and _os.path.exists(f"{_job}/video.mp4") and studio_shown["id"] != _id:
        studio_shown["id"] = _id; studio_done_set(_id)
_out
'''

RESULT = r'''
import os as _os, json as _json, glob as _glob
import marimo as _mo
_id = studio_done_get()
if not _id:
    _res = _mo.md("*Finished MP4 appears here.*")
else:
    _job = f"/work/sp/studio/jobs/{_id}"
    _mp4 = open(f"{_job}/video.mp4", "rb").read()
    _dls = [_mo.download(data=_mp4, filename=f"sakhiya_pro_{_id}.mp4", mimetype="video/mp4", label="⬇️ MP4")]
    for _ext, _mt in (("mp3", "audio/mpeg"), ("wav", "audio/wav")):
        _dls.append(_mo.download(data=open(f"{_job}/voice.{_ext}", "rb").read(), filename=f"sakhiya_pro_{_id}.{_ext}", mimetype=_mt, label=f"⬇️ {_ext.upper()}"))
    _meta = _json.load(open(f"{_job}/meta.json")) if _os.path.exists(f"{_job}/meta.json") else {}
    _rows = "".join(f"<tr><td>{i+1}</td><td>{c['lang']}</td><td>{c['dur']}s</td><td>{c['cer']}</td><td>{c['utmos']}</td><td>{c['clicks']:.0f}</td><td>{c['hindi']}</td></tr>" for i, c in enumerate(_meta.get("chunks", [])))
    _res = _mo.vstack([
        _mo.md(f"### 🎬 Result · {_id} · {_meta.get('duration_s','?')}s · {_meta.get('lufs','?')} LUFS · model `{_os.path.basename(_meta.get('model',''))}` · rendered in {_meta.get('secs','?')}s"),
        _mo.video(src=_mp4, controls=True, rounded=True, max_height="560px"),
        _mo.hstack(_dls, justify="start"),
        _mo.accordion({"Per-sentence quality (CER · UTMOS · clicks · Hindi leak)": _mo.Html(f'<table border="1" cellpadding="4" style="border-collapse:collapse;font-size:12px"><tr><th>#</th><th>lang</th><th>dur</th><th>CER</th><th>UTMOS</th><th>clicks</th><th>hindi</th></tr>{_rows}</table>'),
                       "Files on the box": _mo.md(f"`{_job}/video.mp4`  \n`{_job}/voice.wav`  \n`{_job}/voice.mp3`")}),
    ])
_res
'''


def _wrap(code):
    """Cell-local scope for status/result so marimo sees no multiply-defined globals (names starting with _ are private anyway)."""
    return code.strip()


async def install():
    import marimo._code_mode as cm
    async with cm.get_context(skip_staleness_check=True) as ctx:
        ex = {c.name: c.id for c in ctx.cells.values()}
        for name, code in (('tts_sakhiya_studio', FORM), ('tts_sakhiya_studio_launch', LAUNCH), ('tts_sakhiya_studio_tick', TICK),
                           ('tts_sakhiya_studio_state', STATUS), ('tts_sakhiya_studio_status', STATUS_VIEW), ('tts_sakhiya_studio_result', RESULT)):
            code = code.strip()
            if name in ex: ctx.edit_cell(ex[name], code, hide_code=True); cid = ex[name]
            else: cid = ctx.create_cell(code, name=name, hide_code=True)
            ctx.run_cell(cid); print('installed', name)
