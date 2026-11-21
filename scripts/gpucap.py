#!/usr/bin/env python3
"""Run a script under a hard per-process CUDA memory cap so concurrent stages (gate x4, segmentation x6, ASR, VC)
share the 98 GB Blackwell without OOM-ing each other.   usage: gpucap.py FRACTION script.py [args...]"""
import os, runpy, sys
frac = float(sys.argv[1]); script = sys.argv[2]; sys.argv = sys.argv[2:]
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
try:
    import torch
    if torch.cuda.is_available(): torch.cuda.set_per_process_memory_fraction(frac)
except Exception as e:
    print("gpucap: no cap", e, flush=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
