#!/bin/bash
# Wave 3b (session 6): vocoder-only candidate. The wave-3 AM only reached 585 steps (CPU-bound MAS/aligner warm-up), so its GTA vocoder saw poor predicted mels.
# Candidate = shipped AM (warm_am.pt from onnx2pt) + wave-3 GT-mel phase-loss vocoder (runs/voc3). Chained after wave 3; promoted ONLY if sentence pCER does not regress and MOS does not drop.
R=/work/tts; L=$R/logs; cd $R/repo
push() { export GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0; git add -A docs eval ckpt STATUS.md LEDGER.json scripts 2>/dev/null; git commit -qm "$1"; git pull -q --rebase origin main; git push -q origin HEAD:main; }
until grep -q WAVE3_DONE $L/wave3.log 2>/dev/null; do sleep 10; done
rm -rf $R/best3b; mkdir -p $R/best3b; cp -r docs/models $R/best3b/models; cp -r docs/samples $R/best3b/samples; cp docs/metrics.json $R/best3b/metrics.json
python3 scripts/export_onnx.py --am $R/runs/warm_am.pt --voc $R/runs/voc3/G.pt --out $R/export3b > $L/export3b.log 2>&1 || { echo EXPORT_FAIL >> $L/wave3b.log; exit 1; }
cp $R/export3b/*.onnx $R/export3b/config.json $R/export3b/size.json docs/models/
export ROMAN_ORT=$R/node/package.json
git checkout -- docs/samples 2>/dev/null; python3 scripts/evaluate.py --export $R/export3b --out $R/eval3b > $L/eval3b.log 2>&1
python3 - <<'PY' > $L/promote3b.log 2>&1
import json, shutil
R = "/work/tts"; g = lambda p: next(s for s in json.load(open(p))["systems"] if s["system"] == "Sakhiya-fp16")["sentence"]
new, old = g(f"{R}/repo/docs/metrics.json"), g(f"{R}/best3b/metrics.json")
better = new["pcer"] <= old["pcer"] + 0.003 and new["mos"] >= old["mos"] - 0.05 and (new["pcer"] < old["pcer"] or new["mos"] > old["mos"])
print("new", new["pcer"], new["mos"], "old", old["pcer"], old["mos"], "PROMOTE" if better else "REVERT")
if not better:
    for d in ("models", "samples"): shutil.rmtree(f"{R}/repo/docs/{d}"); shutil.copytree(f"{R}/best3b/{d}", f"{R}/repo/docs/{d}")
    for d in (f"{R}/repo/docs/metrics.json", f"{R}/repo/eval/metrics.json"): shutil.copy(f"{R}/best3b/metrics.json", d)
PY
cat $L/promote3b.log >> $L/wave3b.log; push "wave 3b: vocoder-only candidate on shipped AM ($(tail -n 1 $L/promote3b.log | awk '{print $NF}'))"
echo WAVE3B_DONE >> $L/wave3b.log
