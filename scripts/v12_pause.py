#!/usr/bin/env python3
"""Freeze / unfreeze the GPU-heavy YouTube mining workers (never this script, its parents, or the cloud-ASR jobs).
usage: pause.py stop | cont"""
import os, signal, sys
PAT = ("rsi_youtube_daemon.py", "yt_stage2_loop.sh", "yt_separate_segment.py", "yt_gate_speakers.py", "gpucap.py")
F = "/work/tts/paused_for_v12.pids"
me = {os.getpid(), os.getppid()}
if sys.argv[1] == "stop":
    P = []
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) in me: continue
        try: cmd = open(f"/proc/{d}/cmdline", "rb").read().replace(b"\0", b" ").decode("utf8", "ignore")
        except Exception: continue
        if any(p in cmd for p in PAT) and "pause.py" not in cmd: P.append(int(d))
    for p in P:
        try: os.kill(p, signal.SIGSTOP)
        except Exception: pass
    open(F, "w").write(" ".join(map(str, P))); print("frozen", len(P), flush=True)
else:
    P = [int(x) for x in open(F).read().split()] if os.path.exists(F) else []
    n = 0
    for p in P:
        try: os.kill(p, signal.SIGCONT); n += 1
        except Exception: pass
    print("resumed", n, flush=True)
