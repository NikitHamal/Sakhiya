#!/usr/bin/env python3
"""Pull the repo on MoLab, (re)install the 5 v5 cards via marimo._code_mode, print per-cell status (must be 0 BAD)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import molab_exec as M
code = '''
import subprocess
subprocess.run("cd /work/tts/repo && GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0 git pull -q --rebase origin sakhiya-v5-recovery-synced", shell=True)
exec(open("/work/tts/repo/monitor/cells_v5.py").read())
await install()
await check()
'''
print(M.run_py(code, 170))
