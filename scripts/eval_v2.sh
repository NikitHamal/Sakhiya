#!/bin/bash
# Re-score the wave-1 export (export2) with the calibrated evaluator so wave 2 has an apples-to-apples baseline.   bash scripts/eval_v2.sh
R=/work/tts; L=$R/logs; cd $R/repo
[ -d $R/node/node_modules ] || npm i --silent --prefix $R/node onnxruntime-node > $L/node.log 2>&1
export ROMAN_ORT=$R/node/package.json
python3 scripts/evaluate.py --export $R/export2 --out $R/eval2 > $L/eval2.log 2>&1; touch $L/eval2.done
export GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0; git add -A docs eval; git commit -qm "wave 1 re-scored: UTMOS22 + DNSMOS fixed, calibrated MMS-1b ASR"; git pull -q --rebase origin main; git push -q origin HEAD:main
