#!/usr/bin/env python3
"""v8 acoustic-student distillation with a HARD wall-clock cap and a step-50 throughput gate.

Why: the shipped v7 Nano/Atom acoustic students were exported from checkpoints at step ~4,000 of a planned 52,000
(the job was killed), so the text->mel stage was badly under-trained. That is the main source of the robotic / glitchy
voice. This runner resumes from the last checkpoint and runs until --max-minutes (<= 150) with the LR schedule
re-fitted to what the measured speed can actually finish (cosine ends exactly at the cap).
"""
import argparse, os, sys, time, types, torch
R = "/work/tts"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from sakhiya.ctc_aligner import MelCTCAligner  # noqa: E402

BAD = "l_kl = (l_kl * ym.float()).sum() / ym.sum().clamp(min=1.0)"
GOOD = "l_kl = ((l_kl.transpose(0, 1) if l_kl.shape[0] != ym.shape[0] else l_kl) * ym.float()).sum() / ym.sum().clamp(min=1.0)"
LOOP = "    for step in range(start_step, am_steps + 1):\n"
GATED = (
    "    _t0 = time.time(); _cap = float(os.environ.get('V8_MAX_SEC', '6000')); step = start_step - 1\n"
    "    while True:\n"
    "        step += 1\n"
    "        if step > am_steps: break\n"
    "        if step == start_step + 50:\n"
    "            _sps = 50.0 / max(time.time() - _t0, 1e-6); _eta = (am_steps - start_step) / _sps / 60.0\n"
    "            print(f'[GATE] step-50 throughput {_sps:.2f} steps/s, projected {_eta:.1f} min for {am_steps - start_step} steps (cap {_cap/60:.0f} min)', flush=True)\n"
    "            if _eta * 60.0 > _cap: am_steps = start_step + int(_sps * _cap * 0.97); print(f'[GATE] re-fitting schedule to {am_steps} total steps so it ends at the cap', flush=True)\n"
    "        if step > start_step + 50 and (time.time() - _t0) > _cap * 1.02: am_steps = step; print('[GATE] wall-clock cap hit, saving and exiting', flush=True)\n"
)


def load_fixed():
    src = open(os.path.join(REPO, "sakhiya", "v6_distill.py"), encoding="utf8").read()
    if BAD in src: src = src.replace(BAD, GOOD)
    assert LOOP in src, "AM loop marker missing"
    src = src.replace(LOOP, GATED, 1)
    # the body is indented one level (8 spaces) under `for`/`while`, matching the original loop body
    m = types.ModuleType("sakhiya.v8_distill_fixed"); m.__package__ = "sakhiya"; m.__file__ = os.path.join(REPO, "sakhiya", "v8_distill_fixed.py")
    sys.modules[m.__name__] = m
    exec(compile(src, m.__file__, "exec"), m.__dict__)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, choices=["voca", "sano"])
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--max-minutes", type=float, default=110.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--teacher_am", default=f"{R}/runs/v6_teacher/am.pt")
    ap.add_argument("--teacher_voc", default=f"{R}/runs/v5_c1/voc.pt")
    a = ap.parse_args()
    assert a.max_minutes <= 150, "hard cap is 150 minutes"
    os.environ["V8_MAX_SEC"] = str(a.max_minutes * 60.0)
    dev = "cuda"
    D = torch.load(f"{R}/feats/v6_expressive_vedic_24k.pt", weights_only=False)
    ctc = MelCTCAligner().to(dev).train()
    ck = torch.load(f"{R}/feats/v31_ctc_recognizer.pt", map_location=dev, weights_only=False)
    ctc.load_state_dict(ck.get("model", ck)); [p.requires_grad_(False) for p in ctc.parameters()]
    M = load_fixed()
    out = a.out or f"{R}/runs/v8_{a.tier}_am"
    if not os.path.exists(f"{out}/am.pt"):
        os.makedirs(out, exist_ok=True)
        src_ck = f"{R}/runs/v7_{a.tier}_am/am.pt"
        if os.path.exists(src_ck):
            import shutil; shutil.copyfile(src_ck, f"{out}/am.pt")
            print(f"[v8-{a.tier}] warm-started from {src_ck}", flush=True)
    print(f"[v8-{a.tier}] start steps<={a.steps} cap={a.max_minutes} min out={out}", flush=True)
    M.distill_v6_student(a.tier, D, a.teacher_am, a.teacher_voc, ctc, out, am_steps=a.steps, voc_steps=0, dev=dev)
    print(f"[v8-{a.tier}] DONE", flush=True)


if __name__ == "__main__":
    main()
