"""v11 V1 fine-tune cards. Self-contained (own refresh, cell-local imports), so they can be added to ANY Sakhiya notebook without
touching its other cards. Install: exec(open(__file__).read()); await install_v11()  (adds/updates only tts_v11_* cells, at the top)."""
CELLS_V11 = {
"tts_v11_head": '''
import sys as _sys, importlib.util as _iu, marimo as _mo
_s = _iu.spec_from_file_location("cards_v11", "/work/tts/repo9/monitor/cards_v11.py"); v11ui = _iu.module_from_spec(_s); _sys.modules["cards_v11"] = v11ui; _s.loader.exec_module(v11ui)
v11_refresh = _mo.ui.refresh(default_interval="15s", label="V1 fine-tune live (15 s)")
v11_listen = _mo.ui.button(label="Reload V1 A/B samples", kind="success")
_mo.vstack([_mo.hstack([v11_refresh, v11_listen], justify="start"), _mo.Html(v11ui.card_header())])
''',
"tts_v11_progress": '''
import marimo as _mo
v11_refresh.value
_mo.Html(v11ui.card_progress())
''',
"tts_v11_data": '''
import marimo as _mo
_mo.Html(v11ui.card_data())
''',
"tts_v11_gate": '''
import marimo as _mo
v11_refresh.value
_mo.Html(v11ui.card_gate())
''',
"tts_v11_listen": '''
import marimo as _mo
v11_listen.value
_mo.Html(v11ui.card_listen())
''',
}


async def install_v11():
    import marimo._code_mode as cm
    async with cm.get_context(skip_staleness_check=True) as ctx:
        cells = list(ctx.cells.values()); by = {c.name: c.id for c in cells}
        first = next((c.id for c in cells if not c.name.startswith("tts_v11_")), None)
        ids, prev = {}, None
        for name, src in CELLS_V11.items():
            if name in by:
                ids[name] = by[name]; ctx.edit_cell(ids[name], src.strip(), hide_code=True)
            elif prev is None and first is not None:
                ids[name] = ctx.create_cell(src.strip(), name=name, hide_code=True, before=first)
            else:
                ids[name] = ctx.create_cell(src.strip(), name=name, hide_code=True, after=prev) if prev else ctx.create_cell(src.strip(), name=name, hide_code=True)
            prev = ids[name]
        for name in CELLS_V11: ctx.run_cell(ids[name])
        print("v11 cards:", list(CELLS_V11))
