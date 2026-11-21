CELLS = {
"tts_keepalive": '''
import marimo as mo, json, os, sys, time, threading, importlib.util as _iu
_s = _iu.spec_from_file_location("cards_v10", "/work/tts/repo9/monitor/cards_v10.py"); ui10 = _iu.module_from_spec(_s); sys.modules["cards_v10"] = ui10; _s.loader.exec_module(ui10)
def _hb():
    while True:
        try: json.dump({"t": time.time(), "who": "tts-keepalive"}, open("/work/tts/heartbeat.json", "w"))
        except Exception: pass
        time.sleep(60)
if not getattr(sys, "_SAKHIYA_HB", False):
    sys._SAKHIYA_HB = True; threading.Thread(target=_hb, daemon=True).start()
refresh = mo.ui.refresh(default_interval="10s", label="live (10 s)")
listen_btn = mo.ui.button(label="Reload samples", kind="success")
mo.vstack([mo.Html(ui10.header()), mo.hstack([refresh, listen_btn], justify="start")])
''',
"tts_v10_jobs": '''
refresh.value
mo.Html(ui10.card_jobs())
''',
"tts_v10_size": '''
refresh.value
mo.Html(ui10.card_size())
''',
"tts_v10_quality": '''
refresh.value
mo.Html(ui10.card_quality())
''',
"tts_v10_listen": '''
listen_btn.value
mo.Html(ui10.card_listen())
''',
"tts_v10_events": '''
refresh.value
mo.Html(ui10.card_events())
''',
}
