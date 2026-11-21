"""Rebuild warm-start PyTorch checkpoints from the shipped ONNX models (docs/models/), for when the GPU box lost its .pt files.
python scripts/onnx2pt.py --models docs/models --out /work/tts/runs/warm
Writes <out>/am.pt ({"M","spk2id",...}; the training-only aligner stays randomly initialised) and <out>/G.pt (vocoder)."""
import os, sys, json, argparse
import numpy as np, onnx, torch
from onnx import numpy_helper
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sakhiya.acoustic import Acoustic
from sakhiya.vocoder import Vocoder

ap = argparse.ArgumentParser(); ap.add_argument("--models", default="docs/models"); ap.add_argument("--out", default="/work/tts/runs/warm")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)


def read(path):
    """-> {param_name: float32 ndarray}; resolves the fp16-weight 'Cast' trick and MatMul weights (stored transposed, named by node)."""
    m = onnx.load(path); init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}; W = {}
    for n in m.graph.node:
        if n.op_type == "Cast" and n.input[0] in init: W[n.output[0]] = init[n.input[0]].astype(np.float32)
    for k, v in init.items():
        if not k.endswith("_h"): W[k] = v.astype(np.float32)
    P = {k[2:]: v for k, v in W.items() if k.startswith("m.")}
    for n in m.graph.node:
        if n.op_type == "MatMul" and len(n.input) == 2 and n.input[1] in W and n.name.endswith("/MatMul"):
            mod = ".".join(n.name.strip("/").split("/")[:-1]); P[mod + ".weight"] = W[n.input[1]].T.copy()
    return P


cfg = json.load(open(f"{a.models}/config.json"))
am = Acoustic(n_spk=max(32, len(cfg["pitch_stats"]))); sd = am.state_dict(); P = {}
for n in ("encoder", "decoder"): P.update(read(f"{a.models}/{n}.onnx"))
got = {k: torch.from_numpy(v) for k, v in P.items() if k in sd and tuple(sd[k].shape) == tuple(v.shape)}
miss = [k for k in sd if k not in got and not k.startswith("aligner")]
bad = [k for k, v in P.items() if k in sd and tuple(sd[k].shape) != tuple(v.shape)]
print("AM loaded", len(got), "missing (non-aligner)", miss, "shape-mismatch", bad)
sd.update(got); am.load_state_dict(sd)
spks = sorted(cfg["pitch_stats"]); spk2id = {s: i for i, s in enumerate(spks)}
torch.save({"M": am.state_dict(), "spk2id": spk2id, "pitch_stats": cfg["pitch_stats"], "en": tuple(cfg.get("en", (0.0, 1.0))), "step": 0}, f"{a.out}/am.pt")

voc = Vocoder(); vs = voc.state_dict(); V = read(f"{a.models}/vocoder.onnx")
vg = {k: torch.from_numpy(v) for k, v in V.items() if k in vs and tuple(vs[k].shape) == tuple(v.shape)}
print("VOC loaded", len(vg), "of", len(vs), "missing", [k for k in vs if k not in vg])
vs.update(vg); voc.load_state_dict(vs, strict=False); torch.save(voc.state_dict(), f"{a.out}/G.pt")
print("OK", a.out)
