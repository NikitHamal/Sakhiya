"""v5 recovery: 5 live Marimo cards. Install on the box (kernel top-level await):
    exec(open('/work/tts/repo/monitor/cells_v5.py').read()); await install(); await check()
Only marimo._code_mode is used (never the editor 'Keep change' buttons). Idempotent: wipes every non-v5 cell, edits in place."""
CELLS = {}
CELLS["v5_tick"] = '''
import marimo as mo, sys, importlib
if "/work/tts/repo/monitor" not in sys.path: sys.path.insert(0, "/work/tts/repo/monitor")
import v5_ui as v5
importlib.reload(v5)
v5_refresh = mo.ui.refresh(options=["10s", "15s", "30s", "1m"], default_interval="15s", label="Sakhiya v5 live refresh")
v5_refresh
'''
CELLS["v5_card1_box"] = '''
v5_refresh.value
mo.Html(v5.card1())
'''
CELLS["v5_card2_rollback"] = '''
v5_refresh.value
mo.Html(v5.card2())
'''
CELLS["v5_card3_vedic"] = '''
v5_refresh.value
mo.Html(v5.card3())
'''
CELLS["v5_card4_shrink"] = '''
v5_refresh.value
mo.Html(v5.card4())
'''
CELLS["v5_card5_desk_pick"] = '''
v5_names = v5.clip_names()
v5_pick = mo.ui.dropdown(v5_names, value=v5_names[0], label="Clip")
mo.vstack([mo.Html(v5.card5_head()), v5_pick])
'''
CELLS["v5_card5_desk_play"] = '''
v5.desk(v5_pick.value, mo)
'''

# ---INSTALL---
import marimo._code_mode as cm


async def install():
    async with cm.get_context(skip_staleness_check=True) as ctx:
        by_name = {c.name: c.id for c in ctx.cells.values()}
        for c in list(ctx.cells.values()):
            if c.name not in CELLS:
                ctx.delete_cell(c.id)
        ids = {}
        for name, src in CELLS.items():
            if name in by_name:
                ids[name] = by_name[name]; ctx.edit_cell(ids[name], src.strip(), hide_code=True)
            else:
                ids[name] = ctx.create_cell(src.strip(), name=name, hide_code=True)
        for name in CELLS:
            ctx.run_cell(ids[name])
    return list(CELLS)


async def check():
    import time; time.sleep(6)
    async with cm.get_context() as ctx:
        bad = 0
        for c in ctx.cells.values():
            errs = [str(e)[:200] for e in (c.errors or [])]
            if errs or "error" in str(c.status).lower():
                bad += 1
            print(c.name, c.status, "ERR" if errs else "ok", errs)
        print("CELLS", len(ctx.cells), "BAD", bad)
