"""Marimo cells for the Sakhiya S4 live console. Installed by scripts/s4_install_cards.py (uses marimo._code_mode)."""
CELLS = {}
CELLS["s4_head"] = '''
import marimo as mo, sys, importlib.util as _iu
_s = _iu.spec_from_file_location("s4_cards", "/work/tts/repo/monitor/s4_cards.py"); S4 = _iu.module_from_spec(_s); sys.modules["s4_cards"] = S4; _s.loader.exec_module(S4)
refresh = mo.ui.refresh(default_interval="10s", label="live · 10 s")
mo.vstack([mo.Html(S4.header()), refresh], gap=0.5)
'''
CELLS["s4_gpu"] = '''
refresh.value
mo.Html(S4.gpu_card())
'''
CELLS["s4_jobs"] = '''
refresh.value
mo.Html(S4.jobs_card())
'''
CELLS["s4_gates"] = '''
refresh.value
mo.Html(S4.gates_card())
'''
CELLS["s4_audio"] = '''
new_clips = mo.ui.button(label="Reload newest audio")
new_clips
'''
CELLS["s4_audio_list"] = '''
new_clips.value
mo.Html(S4.samples_card())
'''
CELLS["s4_activity"] = '''
refresh.value
mo.Html(S4.activity_card())
'''
