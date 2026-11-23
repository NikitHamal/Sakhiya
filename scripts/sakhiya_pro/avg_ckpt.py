#!/usr/bin/env python3
"""Weight-average (EMA-equivalent) several fine-tune checkpoints into one standalone OmniVoice dir.
usage: avg_ckpt.py <out_dir> <ckpt1> <ckpt2> ...   (later checkpoints get linearly larger weight)"""
import sys, os, shutil, torch
from safetensors.torch import load_file, save_file
out, cks = sys.argv[1], sys.argv[2:]
w = [i + 1 for i in range(len(cks))]; tot = float(sum(w))
acc = None
for c, wi in zip(cks, w):
    fs = [f for f in os.listdir(c) if f.endswith('.safetensors') and f.startswith('model')]
    sd = {}
    for f in fs: sd.update(load_file(os.path.join(c, f)))
    if acc is None: acc = {k: v.float() * wi / tot for k, v in sd.items()}
    else:
        for k, v in sd.items(): acc[k] += v.float() * wi / tot
os.makedirs(out, exist_ok=True)
for f in os.listdir(cks[-1]):
    if not (f.endswith('.safetensors') or f.endswith('.bin') or f.endswith('.pkl') or f.startswith('optimizer')) and os.path.isfile(os.path.join(cks[-1], f)):
        shutil.copy(os.path.join(cks[-1], f), out)
save_file({k: v.to(torch.bfloat16).contiguous() for k, v in acc.items()}, os.path.join(out, 'model.safetensors'), metadata={'format': 'pt'})
print('averaged', len(cks), '->', out)
