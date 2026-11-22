#!/usr/bin/env python3
"""v14 diagnosis: where does the small (1.2M) vocoder lose 0.5 UTMOS between training and the website?
Same Nano-v9 / Atom-v9 student mels throughout:
  A teacher V1 vocoder (torch)            B small vocoder, torch fp32 (= training-time check)
  C exported graphs, fp32 ONNX (forensic gate, no --tiny)   D exported graphs, int8 ONNX (forensic gate --tiny, = website)
A-B = capacity / mel mismatch, B-C = export path (pitch-shift input, decoder sharpening), C-D = int8 quantisation."""
import json, os, subprocess, sys
import torch
REPO = "/work/tts/repo"; sys.path.insert(0, REPO)
from sakhiya.v2_vocoder import VocoderV2
from scripts.v8_voc_gta import load_am, val_mels, export_mel
from scripts.v2_full_eval import load_utmos
from scripts.train_v3_voc import resamp16
R = "/work/tts"; C = f"{R}/ck"; dev = "cuda"; out = {}
D = torch.load(f"{R}/feats/v6_expressive_vedic_24k.pt", weights_only=False, mmap=True); mean, std = float(D["mean"]), float(D["std"]); del D
ut = load_utmos(dev)
def score(voc, vm):
    with torch.no_grad():
        return round(sum(float(ut(resamp16(voc(export_mel(m[None])).float()), torch.full((1,), 16000, device=dev)).mean()) for m in vm) / len(vm), 3)
t = VocoderV2().to(dev); ck = torch.load(f"{R}/runs/v5_c1/voc.pt", map_location=dev, weights_only=False); t.load_state_dict({k: v.float() for k, v in ck["G_ema"].items()}); t.train(False)
smalls = {}
for name in ("v10_voc144", "v10_voc144_adapt"):
    s = VocoderV2(ch=144, res_kernels=(3, 7, 11)).to(dev); k = torch.load(f"{C}/{name}.pt", map_location=dev, weights_only=False); s.load_state_dict(k["G_ema"]); s.train(False); smalls[name] = s
for tier, amf in (("voca", "v9_voca_am"), ("sano", "v9_sano_am")):
    am, st = load_am(tier, f"{C}/{amf}.pt", dev, mean, std); vm = val_mels(am, dev, mean, std)
    out[tier] = dict(A_teacher_torch=score(t, vm), **{f"B_{n}_torch": score(s, vm) for n, s in smalls.items()})
    print(tier, out[tier], flush=True)
for tier, exp, amf in (("nano", "nanov10", "v9_voca_am"), ("atom", "atomv10", "v9_sano_am")):
    stg = f"{R}/diag/stage_{tier}"
    subprocess.run(f"cd {REPO} && {sys.executable} scripts/export_v2_onnx.py --tier {exp} --am {C}/{amf}.pt --voc {C}/v10_voc144_adapt.pt --out {stg} > {R}/logs/diag_export_{tier}.log 2>&1", shell=True)
    for tiny in (False, True):
        gj = f"{R}/diag/gate_{tier}_{'int8' if tiny else 'fp32'}.json"
        subprocess.run(f"cd {REPO} && {sys.executable} scripts/forensic_audio_gate.py --model {stg} {'--tiny' if tiny else ''} --limit 15 --out {gj} > {R}/logs/diag_gate.log 2>&1", shell=True)
        g = json.load(open(gj)); out.setdefault(tier, {})[f"{'D_int8' if tiny else 'C_fp32'}_onnx_gate"] = dict(utmos=g.get("utmos_15"), drops=g.get("word_drop_count"), clicks=g.get("clicks"))
        print(tier, "tiny" if tiny else "fp32", out[tier], flush=True)
json.dump(out, open(f"{R}/diag/diag.json", "w"), indent=1); print("DIAG", json.dumps(out), flush=True)
