"""Marimo cells for the Sakhiya MoLab notebook (owner cards v9): header + 5 plain-English cards.
Stats cards refresh every 5 s; audio cards only change when the owner presses 'New clips' (so playback is never interrupted). Defensive: no cell can raise."""
CELLS = {}
CELLS["tts_keepalive"] = '''
import marimo as mo, json, os, sys, time, threading, importlib.util as _iu
for _p in ("/work/tts/repo/monitor", "/work/tts/repo"):
    if _p not in sys.path: sys.path.insert(0, _p)
_s = _iu.spec_from_file_location("ui13_cards", "/work/tts/repo/monitor/ui13_cards.py"); ui9 = _iu.module_from_spec(_s); sys.modules["ui13_cards"] = ui9; _s.loader.exec_module(ui9)
def _hb():
    while True:
        try: json.dump({"t": time.time(), "who": "tts-keepalive"}, open("/work/tts/heartbeat.json", "w"))
        except Exception: pass
        time.sleep(60)
if not getattr(sys, "_SAKHIYA_HB", False):
    sys._SAKHIYA_HB = True; threading.Thread(target=_hb, daemon=True).start()
refresh = mo.ui.refresh(default_interval="5s", label="live (5 s)")
shuffle = mo.ui.button(label="New clips", kind="success")
kind_dd = mo.ui.dropdown(list(ui9.LISTS), value="Kept clips (passed every check)", label="Show")
style_dd = mo.ui.dropdown(list(ui9.STYLE_OPTS), value="All styles", label="Style")
page_n = mo.ui.number(start=1, stop=999, value=1, label="Page")
mo.vstack([mo.Html(ui9.header()), mo.hstack([shuffle, refresh], justify="start")])
'''
CELLS["tts_card1"] = '''
refresh.value
mo.Html(ui9.card1())
'''
CELLS["tts_card2"] = '''
shuffle.value
mo.Html(ui9.card2())
'''
CELLS["tts_card3"] = '''
mo.vstack([mo.hstack([kind_dd, style_dd, page_n], justify="start"), mo.Html(ui9.card3(kind_dd.value, style_dd.value, page_n.value))])
'''
CELLS["tts_card4"] = '''
shuffle.value
mo.Html(ui9.card4())
'''
CELLS["tts_card5"] = '''
refresh.value
mo.Html(ui9.card5())
'''

CELLS["tts_card6_pick"] = '''
audit_kind = mo.ui.dropdown(ui9.AUDIT_LISTS, value=ui9.AUDIT_LISTS[0], label="Check")
audit_style = mo.ui.dropdown(list(ui9.STYLE_OPTS), value="All styles", label="Style")
audit_page = mo.ui.number(start=1, stop=9999, value=1, label="Page")
mo.vstack([mo.Html(ui9.card6_head()), mo.hstack([audit_kind, audit_style, audit_page], justify="start")])
'''
CELLS["tts_card6_edit"] = '''
audit_items, _n, _pages, _pg = ui9.audit_items(audit_kind.value, audit_style.value, audit_page.value)
audit_v = mo.ui.array([mo.ui.dropdown(ui9.VERD, value=o["saved_v"], label="Verdict") for o in audit_items])
audit_t = mo.ui.array([mo.ui.text_area(value=o["saved_text"], rows=2, full_width=True, label="Words (edit here)") for o in audit_items])
mo.vstack([mo.md(f"**{_n:,} clips** in this list · page {_pg} of {_pages}")] + [mo.vstack([mo.Html(ui9.clip_html(o)), audit_t[i], audit_v[i]]) for i, o in enumerate(audit_items)]) if audit_items else mo.md("No clips in this list yet.")
'''
CELLS["tts_card6_save"] = '''
mo.Html(ui9.vtext(audit_items, audit_v.value, audit_t.value)) if audit_items else mo.Html(ui9.status())
'''
CELLS["tts_card6_import"] = '''
imp_box = mo.ui.text_area(value="", rows=3, full_width=True, label="Import fixes from the website: paste the text you copied from the Clip Library page")
imp_go = mo.ui.run_button(label="Import fixes")
mo.vstack([imp_box, imp_go])
'''
CELLS["tts_card6_imp_do"] = '''
mo.Html(ui9.import_fixes(imp_box.value)) if (imp_go.value and imp_box.value.strip()) else mo.md("_Nothing imported yet._")
'''

CELLS["tts_card7_vedic"] = '''
shuffle.value
mo.Html(ui9.card7())
'''

CELLS["tts_card8_stats"] = '''
refresh.value
mo.Html(ui9.card8_stats())
'''
CELLS["tts_card8_clips"] = '''
shuffle.value
mo.Html(ui9.card8_clips())
'''
