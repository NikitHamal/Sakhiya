#!/usr/bin/env python3
"""Print status/output/errors of every MoLab notebook cell (expects ZERO errors)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import molab_exec as M
print(M.run_py('''
import marimo._code_mode as cm
async with cm.get_context() as ctx:
    bad = 0
    for c in ctx.cells.values():
        errs = [str(e)[:300] for e in (c.errors or [])]
        o = c.output; d = str(getattr(o, "data", ""))
        con = " | ".join(str(getattr(x, "data", ""))[:200] for x in (c.console_outputs or []))
        isbad = bool(errs) or "Traceback" in d or "Traceback" in con or "error" in str(c.status).lower()
        bad += isbad
        print(c.name, c.status, getattr(o, "mimetype", None), len(d), "ERR" if isbad else "ok", errs, con[:300])
    print("CELLS", len(ctx.cells), "BAD", bad)
''', 120))
