#!/usr/bin/env python3
"""molab_put.py LOCAL REMOTE [LOCAL REMOTE ...] : copy small text files to the MoLab box (base64 over kernel exec)."""
import sys, base64; sys.path.insert(0, __file__.rsplit('/', 1)[0]); import molab_exec as m
a = sys.argv[1:]
for i in range(0, len(a), 2):
    b = base64.b64encode(open(a[i], 'rb').read()).decode()
    print(a[i + 1], m.run_py(f"import base64,os;os.makedirs(os.path.dirname({a[i+1]!r}),exist_ok=True);open({a[i+1]!r},'wb').write(base64.b64decode({b!r}));print('ok')").strip())
