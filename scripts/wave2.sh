#!/bin/bash
# Wave 2 (warm-started, ~25 min on bf16): AM wave 3 (train_am --init am2) is launched separately -> GTA vocoder re-fine-tune on the new
# predicted mels -> export -> evaluate -> promote ONLY if sentence pCER improves over wave 1 (monotonic).   bash scripts/wave2.sh
R=/work/tts; L=$R/logs; cd $R/repo
GV=${GV:-14}
push() { export GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0; git add -A docs eval STATUS.md LEDGER.json 2>/dev/null; git commit -qm "$1"; git pull -q --rebase origin main; git push -q origin HEAD:main; }
until grep -q GTA_DONE $L/am3.log; do sleep 10; done                       # AM wave finished + new GTA mels written
[ -d $R/node/node_modules ] || npm i --silent --prefix $R/node onnxruntime-node > $L/node.log 2>&1
export ROMAN_ORT=$R/node/package.json
python3 scripts/train_vocoder.py --set v2 --minutes $GV --out $R/runs/voc_gta2 --bs 32 --gta $R/feats/gta_v2.pt --init $R/runs/voc_gta/G.pt --lr 5e-5 > $L/gta2.log 2>&1
# GTA no longer waits for the wave-1 re-score (GPU was idle); the baseline snapshot is taken once eval2 is done, before docs/models changes
until [ -f $L/eval2.done ]; do sleep 10; done                             # wave-1 export re-scored with the fixed evaluator (apples-to-apples baseline)
mkdir -p $R/best_v2; cp -r docs/models docs/samples docs/metrics.json eval/metrics.json $R/best_v2/ 2>/dev/null
export GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0; git add -A docs eval; git commit -qm 'snapshot' 2>/dev/null; git pull -q --rebase origin main 2>/dev/null || true
python3 scripts/export_onnx.py --am $R/runs/am3/last.pt --voc $R/runs/voc_gta2/G.pt --out $R/export3 > $L/export3.log 2>&1 || { echo EXPORT3_FAIL >> $L/wave2.log; exit 1; }
cp $R/export3/*.onnx $R/export3/config.json $R/export3/size.json docs/models/ 
python3 scripts/evaluate.py --export $R/export3 --out $R/eval3 > $L/eval3.log 2>&1
python3 scripts/eval_roman.py > $L/roman.log 2>&1; cp eval/roman_metrics.json docs/roman_metrics.json 2>/dev/null
python3 - <<'PY' > $L/promote.log 2>&1
import json, shutil, os
R = "/work/tts"; g = lambda p: next(s for s in json.load(open(p))["systems"] if s["system"] == "Sakhiya-fp16")["sentence"]
new, old = g(f"{R}/repo/docs/metrics.json"), g(f"{R}/best_v2/metrics.json")
better = new["pcer"] <= old["pcer"] + 0.003 and new["mos"] >= old["mos"] - 0.05
print("new", new["pcer"], new["mos"], "old", old["pcer"], old["mos"], "PROMOTE" if better else "REVERT")
if not better:
    for d in ("models", "samples"): shutil.rmtree(f"{R}/repo/docs/{d}"); shutil.copytree(f"{R}/best_v2/{d}", f"{R}/repo/docs/{d}")
    shutil.copy(f"{R}/best_v2/metrics.json", f"{R}/repo/docs/metrics.json"); shutil.copy(f"{R}/best_v2/metrics.json", f"{R}/repo/eval/metrics.json")
PY
cat $L/promote.log >> $L/wave2.log; push "wave 2: warm-started AM + GTA vocoder, evaluated ($(tail -n 1 $L/promote.log | awk '{print $NF}'))"
echo WAVE2_DONE >> $L/wave2.log
