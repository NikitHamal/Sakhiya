#!/usr/bin/env python3
"""Pull repo on the box, (re)start the S4 guardian, install the S4 live cards into the marimo notebook, print cell health (must be BAD: 0).
usage: MOLAB_URL=... MOLAB_TOKEN_FILE=... python3 scripts/s4_install_cards.py"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import molab_exec as M

_t = open(os.environ.get("MOLAB_TOKEN_FILE", "/tmp/molab_token")).read().strip()
M.run_py(f"import os; open('/tmp/molab_token','w').write({_t!r}); os.chmod('/tmp/molab_token',0o600)")  # guardian needs it for its loopback websocket (box-local, never committed)
print(M.run_sh("cd /work/tts/repo && export GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0 && git stash -q 2>/dev/null; git pull -q --rebase origin main 2>&1 | tail -1; git log --oneline | head -1"))
print(M.run_sh("mkdir -p /work/tts/s4/progress /work/tts/logs; cd /work/tts/repo && (setsid nohup python3 -u scripts/s4_guardian.py >> /work/tts/logs/s4_guardian.log 2>&1 < /dev/null &) ; sleep 4; tail -3 /work/tts/logs/s4_guardian.log; cat /work/tts/s4/guardian.pid"))
code = '''
import importlib.util as _u
_s = _u.spec_from_file_location("s4_cells", "/work/tts/repo/monitor/s4_cells.py"); C = _u.module_from_spec(_s); _s.loader.exec_module(C)
import marimo._code_mode as cm
async with cm.get_context(skip_staleness_check=True) as ctx:
    by_name = {c.name: c.id for c in ctx.cells.values()}
    for c in list(ctx.cells.values()):
        if c.name not in C.CELLS:
            ctx.delete_cell(c.id)
    ids = {}
    for name, src in C.CELLS.items():
        if name in by_name:
            ids[name] = by_name[name]; ctx.edit_cell(ids[name], src.strip(), hide_code=True)
        else:
            ids[name] = ctx.create_cell(src.strip(), name=name, hide_code=True)
    for name in C.CELLS:
        ctx.run_cell(ids[name])
import time; time.sleep(6)
async with cm.get_context() as ctx:
    bad = 0
    for c in ctx.cells.values():
        errs = [str(x)[:200] for x in (c.errors or [])]
        st = str(c.status)
        if errs or "error" in st.lower(): bad += 1
        print(c.name, st, "ERR" if errs else "ok", errs)
    print("CELLS", len(ctx.cells), "BAD:", bad)
'''
print(M.run_py(code, 170))
