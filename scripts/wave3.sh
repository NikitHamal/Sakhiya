#!/bin/bash
# Wave 3 (session 4): box rebuilt -> curate -> warm-start from shipped ONNX (onnx2pt) -> AM (aligner warm-up, 18 min) || vocoder GT fine-tune with
# phase loss -> GTA vocoder -> export -> evaluate -> promote ONLY if sentence pCER does not regress -> archive fp16 checkpoints in ckpt/.
R=/work/tts; L=$R/logs; cd $R/repo; mkdir -p $R/runs
AM=${AM_MIN:-18}; VOC=${VOC_MIN:-18}; GTA=${GTA_MIN:-10}
push() { export GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0; git add -A docs eval ckpt STATUS.md LEDGER.json scripts 2>/dev/null; git commit -qm "$1"; git pull -q --rebase origin main; git push -q origin HEAD:main; }
[ -f $L/dl.done ] || { echo waiting_dl >> $L/wave3.log; until [ -f $L/dl.done ]; do sleep 10; done; }
[ -f $L/curate.done ] || { python3 scripts/curate.py > $L/curate.log 2>&1 && touch $L/curate.done || { echo CURATE_FAIL >> $L/wave3.log; exit 1; }; }
if [ -f ckpt/am.fp16.pt ]; then cp ckpt/am.fp16.pt $R/runs/warm_am.pt; cp ckpt/G.fp16.pt $R/runs/warm_G.pt; else
  python3 scripts/onnx2pt.py --models docs/models --out $R/runs/warm > $L/onnx2pt.log 2>&1 || { echo ONNX2PT_FAIL >> $L/wave3.log; exit 1; }
  cp $R/runs/warm/am.pt $R/runs/warm_am.pt; cp $R/runs/warm/G.pt $R/runs/warm_G.pt; fi
mkdir -p $R/best; rm -rf $R/best/models; cp -r docs/models $R/best/models; cp -r docs/samples $R/best/samples 2>/dev/null     # baseline = models on main now
export ROMAN_ORT=$R/node/package.json; [ -d $R/node/node_modules ] || npm i --silent --prefix $R/node onnxruntime-node > $L/node.log 2>&1
( python3 scripts/evaluate.py --export $R/best/models --out $R/eval_base > $L/eval_base.log 2>&1; cp docs/metrics.json $R/best/metrics.json; rm -rf $R/best/samples; cp -r docs/samples $R/best/samples; touch $L/eval_base.done ) &
python3 scripts/train_vocoder.py --set v2 --minutes $VOC --out $R/runs/voc3 --bs 32 --init $R/runs/warm_G.pt --lr 1e-4 --phase 3 > $L/voc3.log 2>&1 &
VP=$!
python3 scripts/train_am.py --set v2 --minutes $AM --bs 64 --lr 4e-4 --align_warm 1500 --init $R/runs/warm_am.pt --out $R/runs/am4 > $L/am4.log 2>&1
wait $VP
python3 scripts/train_vocoder.py --set v2 --minutes $GTA --out $R/runs/voc_gta3 --bs 32 --gta $R/feats/gta_v2.pt --init $R/runs/voc3/G.pt --lr 5e-5 --phase 3 > $L/gta3.log 2>&1
python3 scripts/export_onnx.py --am $R/runs/am4/last.pt --voc $R/runs/voc_gta3/G.pt --out $R/export4 > $L/export4.log 2>&1 || { echo EXPORT_FAIL >> $L/wave3.log; exit 1; }
cp $R/export4/*.onnx $R/export4/config.json $R/export4/size.json docs/models/
until [ -f $L/eval_base.done ]; do sleep 10; done     # baseline (shipped wave-1 model) scored by the same evaluator
git checkout -- docs/samples 2>/dev/null; python3 scripts/evaluate.py --export $R/export4 --out $R/eval4 > $L/eval4.log 2>&1
python3 scripts/eval_roman.py > $L/roman.log 2>&1; cp eval/roman_metrics.json docs/roman_metrics.json 2>/dev/null
python3 - <<'PY' > $L/promote.log 2>&1
import json, shutil, torch
R = "/work/tts"; g = lambda p: next(s for s in json.load(open(p))["systems"] if s["system"] == "Sakhiya-fp16")["sentence"]
new, old = g(f"{R}/repo/docs/metrics.json"), g(f"{R}/best/metrics.json")
better = new["pcer"] <= old["pcer"] + 0.003 and new["mos"] >= old["mos"] - 0.05
print("new", new["pcer"], new["mos"], "old", old["pcer"], old["mos"], "PROMOTE" if better else "REVERT")
if better:
    import os; os.makedirs(f"{R}/repo/ckpt", exist_ok=True); c = torch.load(f"{R}/runs/am4/last.pt", weights_only=False)
    c["M"] = {k: v.half() for k, v in c["M"].items()}; torch.save(c, f"{R}/repo/ckpt/am.fp16.pt")
    torch.save({k: v.half() for k, v in torch.load(f"{R}/runs/voc_gta3/G.pt").items()}, f"{R}/repo/ckpt/G.fp16.pt")
else:
    for d in ("models", "samples"): shutil.rmtree(f"{R}/repo/docs/{d}"); shutil.copytree(f"{R}/best/{d}", f"{R}/repo/docs/{d}")
    shutil.copy(f"{R}/best/metrics.json", f"{R}/repo/docs/metrics.json"); shutil.copy(f"{R}/best/metrics.json", f"{R}/repo/eval/metrics.json")
PY
cat $L/promote.log >> $L/wave3.log; push "wave 3: warm-started AM + phase-loss GTA vocoder, evaluated ($(tail -n 1 $L/promote.log | awk '{print $NF}'))"
echo WAVE3_DONE >> $L/wave3.log
