"""Sakhiya-Pro Studio card (bottom of the MoLab notebook): type any text -> Sakhiya-Pro voice -> MP3 (optional captioned MP4).
Install (idempotent, touches ONLY the tts_sakhiya_studio* cells, appended at the bottom) via marimo._code_mode:
   exec(open('/work/sp/repo/monitor/cells_studio.py').read()); await install()
Cells:
  tts_sakhiya_studio         : the form (text, model, language, pace, takes, quality, MP3 bitrate, mastering, optional MP4, Generate button)
  tts_sakhiya_studio_launch  : queues the job and makes sure the warm TTS server (scripts/sakhiya_pro_studio.py --serve) is running
  tts_sakhiya_studio_tick    : live-refresh timer + cancel button
  tts_sakhiya_studio_state   : tiny state holder (so the audio player is not re-created on every tick)
  tts_sakhiya_studio_status  : live progress bar, ETA, per-sentence take log, server health
  tts_sakhiya_studio_result  : MP3 player + MP3/WAV(/MP4) downloads + quality table + recent renders
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
    rows=8, full_width=True, label="✍️ Text prompt — Nepali, Sanskrit (ॐ, ॥) or Nepali + English tech terms. Blank line = longer pause. Long scripts are fine (split per sentence).")
studio_form = _mo.md("""{text}

{model}  {lang}  {bitrate}

{takes}  {steps}  {pace}

{variation}  {master}  {video}  {fmt}  {title}
""").batch(
    text=_studio_text,
    model=_mo.ui.dropdown(_studio_opts, value=_studio_default, label="Model"),
    lang=_mo.ui.dropdown({"Auto (Nepali / Sanskrit by text)": "auto", "Nepali": "npi", "Sanskrit / Vedic": "sa"}, value="Auto (Nepali / Sanskrit by text)", label="Language"),
    bitrate=_mo.ui.dropdown({"MP3 320 kbps (max)": "320k", "MP3 256 kbps (default)": "256k", "MP3 192 kbps": "192k", "MP3 128 kbps (small)": "128k"}, value="MP3 256 kbps (default)", label="Quality"),
    takes=_mo.ui.slider(1, 5, value=3, label="Best-of-N takes"),
    steps=_mo.ui.slider(16, 64, step=8, value=32, label="Diffusion steps"),
    pace=_mo.ui.slider(0.85, 1.15, step=0.01, value=1.0, label="Pace"),
    variation=_mo.ui.number(0, 9, value=0, label="Variation (new seeds)"),
    master=_mo.ui.checkbox(value=True, label="Broadcast mastering (-14 LUFS)"),
    video=_mo.ui.checkbox(value=False, label="Also make captioned MP4"),
    fmt=_mo.ui.dropdown({"16:9 landscape 1280×720": "land", "9:16 vertical 720×1280": "vert"}, value="16:9 landscape 1280×720", label="Video shape"),
    title=_mo.ui.text(value="NEBians News", label="Title / ID3 tag"),
).form(submit_button_label="🎧 Generate MP3", bordered=True, show_clear_button=False)
_mo.vstack([
    _mo.md("## 🎙️ Sakhiya-Pro Studio — text → natural voice → MP3"),
    _mo.md("Type anything and press **Generate MP3**. The warm Sakhiya-Pro server renders Best-of-N takes per sentence (picked by ASR / clicks / Hindi-leak / UTMOS / speaker match, with an automatic re-roll on a bad take), "
           "masters with the broadcast chain (80 Hz HPF · warmth/presence EQ · gentle compression · **-14 LUFS**) and gives you a playable, downloadable **MP3**. Tick *Also make captioned MP4* for a video."),
    studio_form,
    _mo.accordion({"💡 Example prompts (copy into the box)": _mo.md(
        "- `नमस्कार दर्शकवृन्द, NEBians News मा तपाईंलाई स्वागत छ। आज Gemini 4 Neon र Liquid Glass Dock का बारेमा कुरा गरौँ।`\n"
        "- `Jensen Huang ले GPT-6.2 जस्तै शक्तिशाली चिप सार्वजनिक गरे। यसले एआई उद्योगमा ठूलो परिवर्तन ल्याउने अपेक्षा गरिएको छ।`\n"
        "- `ॐ पूर्णमदः पूर्णमिदं पूर्णात्पूर्णमुदच्यते। पूर्णस्य पूर्णमादाय पूर्णमेवावशिष्यते॥`\n"
        "- `के तपाईंले आज बिहान समाचार हेर्नुभयो? कुरा त साँच्चै रमाइलो छ!`")}),
])
'''

LAUNCH = r'''
studio_launched = None
if studio_form.value:
    import os as _os, json as _json, time as _time, subprocess as _sp
    _root = "/work/sp/studio"; _os.makedirs(_root + "/jobs", exist_ok=True)
    _id = _time.strftime("%Y%m%d_%H%M%S")
    _job = f"{_root}/jobs/{_id}"; _os.makedirs(_job, exist_ok=True)
    _v = studio_form.value
    _json.dump(dict(text=_v["text"], model=_v["model"], lang=_v["lang"], takes=int(_v["takes"]), steps=int(_v["steps"]), pace=float(_v["pace"]),
                    bitrate=_v["bitrate"], master=bool(_v["master"]), video=bool(_v["video"]), fmt=_v["fmt"], title=_v["title"], variation=int(_v["variation"])),
               open(f"{_job}/job.json", "w"), ensure_ascii=False)
    _json.dump(dict(stage="queued", done=0, total=0, log=[], started=_time.time()), open(f"{_job}/status.json", "w"))
    open(f"{_root}/latest.txt", "w").write(_id)
    def _up(pid):   # alive and not a zombie (a killed child of this kernel keeps a /proc entry until reaped)
        try: return open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0] != "Z"
        except Exception: return False
    _alive = False
    try:
        _hb = _json.load(open(f"{_root}/server.json")); _alive = (_time.time() - _hb["ts"] < 20) and _up(_hb["pid"])
    except Exception: pass
    if not _alive:   # (re)start the warm server detached from this kernel
        _sp.Popen(["python3", "/work/sp/repo/scripts/sakhiya_pro_studio.py", "--serve", _root], stdout=open(f"{_root}/server.log", "a"), stderr=_sp.STDOUT, stdin=_sp.DEVNULL, start_new_session=True)
    studio_launched = _id
'''

TICK = r'''
import marimo as _mo
studio_tick = _mo.ui.refresh(options=["2s", "3s", "5s", "10s"], default_interval="3s", label="Studio live refresh")
def _studio_cancel(_):
    try:
        _id = open("/work/sp/studio/latest.txt").read().strip()
        open(f"/work/sp/studio/jobs/{_id}/cancel", "w").write("1")
    except Exception: pass
studio_cancel = _mo.ui.button(label="⛔ Cancel current job", kind="danger", on_click=_studio_cancel)
_mo.hstack([studio_tick, studio_cancel], justify="start", align="center")
'''

# status/result use mo.state so the audio player is NOT re-created on every refresh tick
STATE = r'''
import marimo as _mo
studio_done_get, studio_done_set = _mo.state(None)
studio_shown = {"id": None}
'''

STATUS_VIEW = r'''
studio_tick.value; studio_launched
import os as _os, json as _json, time as _time, html as _html
import marimo as _mo
def _up(pid):
    try: return open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0] != "Z"
    except Exception: return False
_root = "/work/sp/studio"; _lt = _root + "/latest.txt"
_id = open(_lt).read().strip() if _os.path.exists(_lt) else None
try:
    _hb = _json.load(open(_root + "/server.json")); _age = _time.time() - _hb["ts"]
    _srv = f"🟢 warm server · {_hb.get('state')}" if _age < 20 and _up(_hb["pid"]) else "🟠 server stopped (starts automatically on the next Generate)"
except Exception: _srv = "⚪ server not started yet (starts on first Generate; the first job loads the model ≈ 20 s)"
_out = _mo.md(f"*No render yet — write some text above and press **Generate MP3**.*  \n<small>{_srv}</small>")
if _id:
    _job = f"{_root}/jobs/{_id}"
    try: _s = _json.load(open(f"{_job}/status.json"))
    except Exception: _s = dict(stage="starting", done=0, total=0, log=[], started=_time.time())
    _tot = max(1, _s.get("total") or 1); _pct = int(100 * (_s.get("done", 0)) / _tot)
    _stage = _s.get("stage", "?"); _el = (_s.get("finished") or _time.time()) - _s.get("started", _time.time())
    if _stage == "done": _pct = 100
    _icon = "✅" if _stage == "done" else ("❌" if _stage == "error" else ("🕒" if _stage == "queued" else "⏳"))
    _eta = f" · ETA ≈ {_s['eta']:.0f}s" if _s.get("eta") else ""
    _fin = (" · " + str(_s.get("duration_s")) + "s audio · " + str(_s.get("lufs")) + " LUFS") if _stage == "done" else ""
    _bar = f'<div style="background:#1e293b;border-radius:6px;height:14px;width:100%"><div style="background:#14b8a6;height:14px;border-radius:6px;width:{_pct}%"></div></div>'
    _log = "<br>".join(_html.escape(l) for l in _s.get("log", [])[-12:])
    _err = ""
    if _stage == "error":
        _wl = _root + "/server.log"
        _tail = "".join(open(_wl, errors="ignore").readlines()[-6:]) if _os.path.exists(_wl) else ""
        _err = f'<p style="color:#c00"><b>{_html.escape(str(_s.get("error")))}</b><br><code style="white-space:pre-wrap">{_html.escape(_tail)}</code></p>'
    _out = _mo.Html(f"""<div style="font-family:system-ui;font-size:14px;line-height:1.5">
<b>{_icon} Job {_id}</b> · {_html.escape(_stage)} · {_pct}% · {_el:.0f}s elapsed{_eta}{_fin}
<br><small>{_srv}</small>
{_bar}<p style="font-family:ui-monospace,monospace;font-size:12px;color:#555">{_log}</p>{_err}</div>""")
    if _stage == "done" and _os.path.exists(f"{_job}/voice.mp3") and studio_shown["id"] != _id:
        studio_shown["id"] = _id; studio_done_set(_id)
_out
'''

RESULT = r'''
import os as _os, json as _json, glob as _glob, html as _html, base64 as _b64
import marimo as _mo
_id = studio_done_get()
def _aud(b):   # explicit audio/mpeg data-URI: the MP3 plays in every browser (mo.audio would label it .wav)
    return _mo.Html('<audio controls style="width:100%;max-width:640px" src="data:audio/mpeg;base64,' + _b64.b64encode(b).decode() + '"></audio>')
_root = "/work/sp/studio/jobs"
def _hist():
    _items = []
    for _d in sorted(_glob.glob(_root + "/*/voice.mp3"), reverse=True)[:5]:
        _jid = _os.path.basename(_os.path.dirname(_d))
        try:
            _m = _json.load(open(_os.path.dirname(_d) + "/meta.json")); _t = _m.get("text_used", "")[:70].replace("\n", " "); _du = _m.get("duration_s", "?")
        except Exception: _t, _du = "", "?"
        _items.append(_mo.vstack([_mo.md(f"`{_jid}` · {_du}s · {_html.escape(_t)}"), _aud(open(_d, "rb").read())]))
    return _mo.accordion({"🕘 Recent renders (last 5)": _mo.vstack(_items) if _items else _mo.md("none yet")})
if not _id:
    _res = _mo.vstack([_mo.md("*Finished MP3 appears here.*"), _hist()])
else:
    _job = f"{_root}/{_id}"
    _mp3 = open(f"{_job}/voice.mp3", "rb").read()
    _dls = [_mo.download(data=_mp3, filename=f"sakhiya_pro_{_id}.mp3", mimetype="audio/mpeg", label="⬇️ MP3"),
            _mo.download(data=open(f"{_job}/voice.wav", "rb").read(), filename=f"sakhiya_pro_{_id}.wav", mimetype="audio/wav", label="⬇️ WAV (lossless)")]
    _vid = []
    if _os.path.exists(f"{_job}/video.mp4"):
        _mp4 = open(f"{_job}/video.mp4", "rb").read()
        _dls.append(_mo.download(data=_mp4, filename=f"sakhiya_pro_{_id}.mp4", mimetype="video/mp4", label="⬇️ MP4"))
        _vid = [_mo.video(src=_mp4, controls=True, rounded=True, height=520)]
    _meta = _json.load(open(f"{_job}/meta.json")) if _os.path.exists(f"{_job}/meta.json") else {}
    _rows = "".join(f"<tr><td>{i+1}</td><td>{c['lang']}</td><td>{c['dur']}s</td><td>{c['cer']}</td><td>{c['utmos']}</td><td>{c['clicks']:.0f}</td><td>{c['hindi']}</td></tr>" for i, c in enumerate(_meta.get("chunks", [])))
    _res = _mo.vstack([
        _mo.md(f"### 🎧 Result · {_id} · {_meta.get('duration_s','?')}s · {_meta.get('lufs','?')} LUFS · model `{_os.path.basename(_meta.get('model',''))}` · rendered in {_meta.get('secs','?')}s"),
        _aud(_mp3),
        _mo.hstack(_dls, justify="start"),
        *_vid,
        _mo.accordion({"Text actually spoken (after cleanup)": _mo.md("```\n" + _meta.get("text_used", "") + "\n```"),
                       "Per-sentence quality (CER · UTMOS · clicks · Hindi leak)": _mo.Html(f'<table border="1" cellpadding="4" style="border-collapse:collapse;font-size:12px"><tr><th>#</th><th>lang</th><th>dur</th><th>CER</th><th>UTMOS</th><th>clicks</th><th>hindi</th></tr>{_rows}</table>'),
                       "Files on the box": _mo.md(f"`{_job}/voice.mp3`  \n`{_job}/voice.wav`  \n`{_job}/meta.json`")}),
        _hist(),
    ])
_res
'''

NAMES = (('tts_sakhiya_studio', 'FORM'), ('tts_sakhiya_studio_launch', 'LAUNCH'), ('tts_sakhiya_studio_tick', 'TICK'),
         ('tts_sakhiya_studio_state', 'STATE'), ('tts_sakhiya_studio_status', 'STATUS_VIEW'), ('tts_sakhiya_studio_result', 'RESULT'))


async def install():
    import marimo._code_mode as cm
    async with cm.get_context(skip_staleness_check=True) as ctx:
        ex = {c.name: c.id for c in ctx.cells.values()}
        for name, var in NAMES:
            code = globals()[var].strip()
            if name in ex: ctx.edit_cell(ex[name], code, hide_code=True); cid = ex[name]
            else: cid = ctx.create_cell(code, name=name, hide_code=True)
            ctx.run_cell(cid); print('installed', name)
