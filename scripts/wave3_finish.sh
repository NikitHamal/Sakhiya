#!/bin/bash
# Wave 3 finish (session 4): eval of the new export runs concurrently with the baseline eval (time), then roman eval -> gated promote -> push.
R=/work/tts; L=$R/logs; cd $R/repo
push() { export GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0; git add -A docs eval ckpt STATUS.md LEDGER.json scripts 2>/dev/null; git commit -qm "$1"; git pull -q --rebase origin main; git push -q origin HEAD:main; }
export ROMAN_ORT=$R/node/package.json
cp $R/export4/*.onnx $R/export4/config.json $R/export4/size.json docs/models/
python3 scripts/evaluate.py --export $R/export4 --out $R/eval4 > $L/eval4.log 2>&1
until [ -f $L/eval_base.done ]; do sleep 10; done
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
