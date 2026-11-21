"""Publish the newest Sakhiya-v2 weights to the GitHub Pages demo (channel "v2 preview").

1. picks the newest acoustic-model checkpoint (v2_am_D > v2_am_C > v2_am_B) and the best vocoder checkpoint,
2. exports fp16 ONNX (encoder/decoder/vocoder) with scripts/export_v2_onnx.py,
3. copies it to docs/models_v2/ together with manifest.json (steps, losses, size, time, honest gate status),
4. optionally pushes through /work/tts/pushsync2.sh.

usage: python3 scripts/publish_v2.py [--push] [--min-minutes 80]
Raw fp16 .pt checkpoints are uploaded separately by scripts/relay.py to the GitHub release `ckpt-wip`."""
import os, sys, json, time, shutil, argparse, subprocess

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RUNS = "/work/tts/runs"
ap = argparse.ArgumentParser()
ap.add_argument("--push", action="store_true"); ap.add_argument("--force", action="store_true"); ap.add_argument("--min-minutes", type=float, default=0)
ap.add_argument("--out", default="/work/tts/export_v2_pub"); a = ap.parse_args()
DOC = os.path.join(ROOT, "docs", "models_v2"); MAN = os.path.join(DOC, "manifest.json")


def newest_am():
    for d in ("v2_am_uni", "v2_am_D", "v2_am_C", "v2_am_B"):
        p = f"{RUNS}/{d}/last.pt"
        if os.path.exists(p): return d, p
    raise SystemExit("no AM checkpoint")


old = json.load(open(MAN)) if os.path.exists(MAN) else {}
if not a.force and a.min_minutes and old.get("published_at") and (time.time() - old["published_at"]) / 60 < a.min_minutes:
    print("recent publish exists, skipping"); sys.exit(0)
run, am = newest_am()
for vd in ("v2_voc_3b2", "v2_voc_3b1", "v2_voc_3b", "v2_voc"):
    if os.path.exists(f"{RUNS}/{vd}/G_best.pt"):
        voc = f"{RUNS}/{vd}/G_best.pt"; break
import torch
c = torch.load(am, map_location="cpu", weights_only=False); am_step = int(c.get("step", 0)); del c
prog = {}
try: prog = json.load(open(os.path.join(os.path.dirname(voc), "progress.json")))
except Exception: pass
if not a.force and old.get("am_step") == am_step and old.get("voc_step") == prog.get("step"):
    print("nothing new"); sys.exit(0)
subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "export_v2_onnx.py"), "--am", am, "--voc", voc, "--out", a.out], check=True)
os.makedirs(DOC, exist_ok=True)
for f in ("encoder.onnx", "decoder.onnx", "vocoder.onnx", "config.json", "size.json"):
    shutil.copyfile(os.path.join(a.out, f), os.path.join(DOC, f))
for f in os.listdir(DOC):   # fp16 only on the hosted preview (keeps the repo small); int8 variant falls back to fp16 in the browser
    if f.endswith(".int8.onnx"): os.remove(os.path.join(DOC, f))
cs = {}
try: cs = json.load(open(os.path.join(ROOT, "eval", "v2_copysynth_loop.json")))
except Exception: pass
man = dict(channel="v2-preview", published_at=time.time(), published_utc=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
           am_run=run, am_step=am_step, voc_step=prog.get("step"), voc_val_l1=prog.get("best_val"),
           copysynth_utmos=cs.get("utmos_copy"), raw_utmos=cs.get("utmos_raw"),
           size_mb=json.load(open(os.path.join(DOC, "size.json"))).get("total_fp16_mb"),
           release="https://github.com/NikitHamal/Sakhiya/releases/tag/ckpt-wip",
           status="PREVIEW — in training; not promoted over Wave 42 until the ship gate passes")
json.dump(man, open(MAN, "w"), indent=1); print("published", man)
if a.push:
    subprocess.run(["/work/tts/pushsync2.sh", f"v2 preview checkpoint: AM {run}@{am_step}, voc@{prog.get('step')}"], check=False)
