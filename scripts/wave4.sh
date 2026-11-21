#!/bin/bash
# Wave 4 (session 7): Harvests clean warm_am.pt + am4 aligner weights -> runs vectorised GPU AM trainer (precomputed MAS, contour/delta losses, 2-voice specialization)
# -> GTA vocoder fine-tune warm-started with trained discriminator D -> ONNX export (2 flagship voices: Female s3614 & Male cv_13b830fe) -> full evaluation -> promote & archive fp16 ckpt/.
R=/work/tts; L=$R/logs; cd $R/repo; mkdir -p $R/runs $L
AM=${AM_MIN:-10}; GTA=${GTA_MIN:-5}; ALIGN=${ALIGN:-200}
push() { export GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0; git add -A docs eval ckpt STATUS.md LEDGER.json scripts sakhiya monitor 2>/dev/null; git commit -qm "$1"; git pull -q --rebase -X ours origin main; git push -q origin HEAD:main; }

python3 - <<'PY'
import torch
R = "/work/tts"
w = torch.load(f"{R}/runs/warm_am.pt", weights_only=False)
a4 = torch.load(f"{R}/runs/am4/last.pt", weights_only=False)
for k, v in a4["M"].items():
    if k.startswith("aligner."):
        w["M"][k] = v
torch.save(w, f"{R}/runs/warm_am5.pt")
g = torch.load(f"{R}/runs/warm_G.pt", weights_only=False)
v3 = torch.load(f"{R}/runs/voc3/last.pt", weights_only=False)
torch.save({"G": g, "D": v3["D"]}, f"{R}/runs/warm_G5.pt")
print("Prepared warm_am5.pt (with am4 aligner) and warm_G5.pt (with voc3 D)")
PY

mkdir -p $R/best; rm -rf $R/best/models $R/best/samples; cp -r docs/models $R/best/models; cp -r docs/samples $R/best/samples 2>/dev/null; cp docs/metrics.json $R/best/metrics.json
export ROMAN_ORT=$R/node/package.json
python3 scripts/train_am.py --set v2 --minutes $AM --bs 64 --lr 3e-4 --align_warm $ALIGN --init $R/runs/warm_am5.pt --out $R/runs/am5 > $L/am5.log 2>&1
python3 scripts/train_vocoder.py --set v2 --minutes $GTA --out $R/runs/voc_gta5 --bs 32 --gta $R/feats/gta_v2.pt --init $R/runs/warm_G5.pt --lr 5e-5 --phase 1.0 > $L/gta5.log 2>&1
python3 scripts/export_onnx.py --am $R/runs/am5/last.pt --voc $R/runs/voc_gta5/G.pt --out $R/export5 > $L/export5.log 2>&1 || { echo EXPORT_FAIL >> $L/wave4.log; exit 1; }
cp $R/export5/*.onnx $R/export5/config.json $R/export5/size.json docs/models/
git checkout -- docs/samples 2>/dev/null; python3 scripts/evaluate.py --export $R/export5 --out $R/eval5 > $L/eval5.log 2>&1
python3 scripts/eval_roman.py > $L/roman.log 2>&1; cp eval/roman_metrics.json docs/roman_metrics.json 2>/dev/null
python3 - <<'PY' > $L/promote.log 2>&1
import json, shutil, torch, os
R = "/work/tts"; g = lambda p: next(s for s in json.load(open(p))["systems"] if s["system"] == "Sakhiya-fp16")["sentence"]
new, old = g(f"{R}/repo/docs/metrics.json"), g(f"{R}/best/metrics.json")
better = new["pcer"] <= old["pcer"] + 0.003 and new["mos"] >= old["mos"] - 0.05 and (new["pcer"] < old["pcer"] or new["mos"] > old["mos"])
print("new", new["pcer"], new["mos"], "old", old["pcer"], old["mos"], "PROMOTE" if better else "REVERT")
if better:
    os.makedirs(f"{R}/repo/ckpt", exist_ok=True); c = torch.load(f"{R}/runs/am5/last.pt", weights_only=False)
    c["M"] = {k: v.half() for k, v in c["M"].items()}; torch.save(c, f"{R}/repo/ckpt/am.fp16.pt")
    torch.save({k: v.half() for k, v in torch.load(f"{R}/runs/voc_gta5/G.pt", weights_only=False).items()}, f"{R}/repo/ckpt/G.fp16.pt")
else:
    for d in ("models", "samples"): shutil.rmtree(f"{R}/repo/docs/{d}"); shutil.copytree(f"{R}/best/{d}", f"{R}/repo/docs/{d}")
    shutil.copy(f"{R}/best/metrics.json", f"{R}/repo/docs/metrics.json"); shutil.copy(f"{R}/best/metrics.json", f"{R}/repo/eval/metrics.json")
PY
cat $L/promote.log >> $L/wave4.log; push "wave 4: vectorised AM + harmonic-contrast GTA vocoder, evaluated ($(tail -n 1 $L/promote.log | awk '{print $NF}'))"
echo WAVE4_DONE >> $L/wave4.log
