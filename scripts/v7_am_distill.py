#!/usr/bin/env python3
"""v7 long-run acoustic-student distillation (Voca / Sano). Resumable; runs beside (never inside) the v6 run dirs.

Why this exists: sakhiya/v6_distill.py crashed at the CTC-KL term (`ctc()` returns [T,B,V]; the mask is [B,T]) so the
v6 daemon never got past ~2k student steps. This runner loads that module's source, applies the one-line transpose fix
in memory (so the shared file is untouched), and trains the student for tens of thousands of steps on the GPU."""
import argparse, os, sys, types, torch
R = "/work/tts"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from sakhiya.ctc_aligner import MelCTCAligner  # noqa: E402

BAD = "l_kl = (l_kl * ym.float()).sum() / ym.sum().clamp(min=1.0)"
GOOD = "l_kl = ((l_kl.transpose(0, 1) if l_kl.shape[0] != ym.shape[0] else l_kl) * ym.float()).sum() / ym.sum().clamp(min=1.0)"

def load_fixed():
    src = open(os.path.join(REPO, "sakhiya", "v6_distill.py"), encoding="utf8").read()
    if BAD in src: src = src.replace(BAD, GOOD)
    m = types.ModuleType("sakhiya.v7_distill_fixed"); m.__package__ = "sakhiya"; m.__file__ = os.path.join(REPO, "sakhiya", "v7_distill_fixed.py")
    sys.modules[m.__name__] = m
    exec(compile(src, m.__file__, "exec"), m.__dict__)
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, choices=["voca", "sano"])
    ap.add_argument("--steps", type=int, default=26000)
    ap.add_argument("--teacher_am", default=f"{R}/runs/v6_teacher/am.pt")
    ap.add_argument("--teacher_voc", default=f"{R}/runs/v5_c1/voc.pt")
    a = ap.parse_args()
    dev = "cuda"
    D = torch.load(f"{R}/feats/v6_expressive_vedic_24k.pt", weights_only=False)
    ctc = MelCTCAligner().to(dev).train()  # cudnn RNN backward needs train mode; params frozen below
    ck = torch.load(f"{R}/feats/v31_ctc_recognizer.pt", map_location=dev, weights_only=False)
    ctc.load_state_dict(ck.get("model", ck)); [p.requires_grad_(False) for p in ctc.parameters()]
    M = load_fixed()
    out = f"{R}/runs/v7_{a.tier}_am"
    print(f"[v7-{a.tier}] start steps={a.steps} out={out}", flush=True)
    M.distill_v6_student(a.tier, D, a.teacher_am, a.teacher_voc, ctc, out, am_steps=a.steps, voc_steps=0, dev=dev)
    print(f"[v7-{a.tier}] DONE", flush=True)

if __name__ == "__main__":
    main()
