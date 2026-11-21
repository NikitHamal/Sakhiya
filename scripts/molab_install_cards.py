#!/usr/bin/env python3
"""Pull the repo on MoLab, install the v6 cards, then print every cell's status/errors (must be ZERO errors)."""
import subprocess, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import molab_exec as M
M.run_sh("cd /work/tts/repo && GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0 git stash -q; GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0 git pull --rebase origin main 2>&1 | tail -1")
code = '''
exec(open("/work/tts/repo/monitor/install.py").read().split('if __name__')[0])
await main()
import time; time.sleep(6)
async with cm.get_context() as ctx:
    bad = 0
    for c in ctx.cells.values():
        errs = [str(e)[:200] for e in (c.errors or [])]
        outs = [str(o.data)[:80] if o else "" for o in (c.console_outputs or [])]
        o = c.output; ot = str(getattr(o, "mimetype", "")) if o else "none"
        st = str(c.status)
        if errs or "error" in st.lower() or "traceback" in str(getattr(o, "data", "")).lower(): bad += 1
        print(c.name, st, ot, "ERR" if errs else "ok", errs)
    print("CELLS", len(ctx.cells), "BAD", bad)
'''
print(M.run_py(code, 170))
