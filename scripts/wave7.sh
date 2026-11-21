#!/bin/bash
# Wave 7 (session 9): AM fine-tune on v3 = v2 + mined OpenSLR-54 (CC-BY-SA) speakers; vocoder = wave-6 best (best_voc6_G.pt). Same promote gate.
R=/work/tts; L=$R/logs; cd $R/repo; mkdir -p $R/runs $L
AM=${AM_MIN:-8}; VOC=${VOC_MIN:-11}
push() { export GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0; git checkout -- docs/data/status.json 2>/dev/null || true; git add -A docs eval ckpt STATUS.md LEDGER.json scripts sakhiya monitor 2>/dev/null; git commit -qm "$1"; git pull -q --rebase -X ours origin main; git push -q origin HEAD:main; }

echo "WAVE7_START $(date)" > $L/wave7.log
mkdir -p $R/best7; rm -rf $R/best7/models $R/best7/samples; cp -r docs/models $R/best7/models; cp -r docs/samples $R/best7/samples 2>/dev/null; cp docs/metrics.json $R/best7/metrics.json
export ROMAN_ORT=$R/node/package.json

python3 scripts/train_am.py --set v3 --minutes $AM --bs 64 --lr 1.2e-4 --align_warm 0 --init $R/runs/am7/last.pt --out $R/runs/am8 > $L/am8.log 2>&1

python3 scripts/export_onnx.py --am $R/runs/am8/last.pt --voc $R/runs/best_voc6_G.pt --out $R/export8 > $L/export8.log 2>&1 || { echo EXPORT_FAIL >> $L/wave7.log; exit 1; }
cp $R/export8/*.onnx $R/export8/config.json $R/export8/size.json docs/models/
git checkout -- docs/samples 2>/dev/null; python3 scripts/evaluate.py --export $R/export8 --out $R/eval8 > $L/eval8.log 2>&1
python3 scripts/eval_roman.py > $L/roman.log 2>&1; cp eval/roman_metrics.json docs/roman_metrics.json 2>/dev/null

python3 - <<'PY' > $L/promote7.log 2>&1
import json, shutil, torch, os
R = "/work/tts"; g = lambda p: next(s for s in json.load(open(p))["systems"] if s["system"] == "Sakhiya-fp16")["sentence"]
new, old = g(f"{R}/repo/docs/metrics.json"), g(f"{R}/best7/metrics.json")
better = new["pcer"] <= old["pcer"] + 0.003 and new["mos"] >= old["mos"] - 0.03 and (new["pcer"] < old["pcer"] or new["mos"] > old["mos"])
print("new", new["pcer"], new["mos"], "old", old["pcer"], old["mos"], "PROMOTE" if better else "REVERT")
if better:
    os.makedirs(f"{R}/repo/ckpt", exist_ok=True); c = torch.load(f"{R}/runs/am8/last.pt", weights_only=False)
    c["M"] = {k: v.half() for k, v in c["M"].items()}; torch.save(c, f"{R}/repo/ckpt/am.fp16.pt")
    torch.save({k: v.half() for k, v in torch.load(f"{R}/runs/best_voc6_G.pt", weights_only=False).items()}, f"{R}/repo/ckpt/G.fp16.pt")
else:
    for d in ("models", "samples"): shutil.rmtree(f"{R}/repo/docs/{d}"); shutil.copytree(f"{R}/best7/{d}", f"{R}/repo/docs/{d}")
    shutil.copy(f"{R}/best7/metrics.json", f"{R}/repo/docs/metrics.json"); shutil.copy(f"{R}/best7/metrics.json", f"{R}/repo/eval/metrics.json")
PY
cat $L/promote7.log >> $L/wave7.log
push "wave 7: AM on v3 (+OpenSLR-54 mined speakers) ($(tail -n 1 $L/promote7.log | awk '{print $NF}'))"
echo WAVE7_DONE >> $L/wave7.log
