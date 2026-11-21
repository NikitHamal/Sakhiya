#!/bin/bash
# Wave 30 post-stage: evaluate the own single-female-voice export on held-out REAL clips, then gate.
# PROMOTE if Nepali sentence pCER <= 0.07 and own Sanskrit pCER <= 0.22 (human Shrutilipi radio ceiling is ~0.17), i.e. intelligible in both.
# Otherwise REVERT the shipped files but keep the candidate numbers in eval/candidate_wave30.json (honest report, no silent swap).
R=/work/tts; L=$R/logs; cd $R/repo; W=$L/wave30.log
rm -rf $R/best30; mkdir -p $R/best30; cp -r docs/models $R/best30/models; cp -r docs/samples $R/best30/samples; cp docs/metrics.json $R/best30/metrics.json
python3 scripts/evaluate.py --export $R/export30 --out $R/eval30 > $L/eval30.log 2>&1
echo "EVAL_DONE $(date)" >> $W
python3 - <<'PY' > $L/promote30.log 2>&1
import json, shutil, os
R = "/work/tts"; m = json.load(open(f"{R}/repo/docs/metrics.json"))
fp = next(s for s in m["systems"] if s["system"] == "Sakhiya-fp16")["sentence"]
sa = next((v for k, v in m.get("sanskrit", {}).items() if k.startswith("Sakhiya one voice")), None)
ok = fp["pcer"] <= 0.07 and sa is not None and sa["pcer"] <= 0.22
print("ne pCER", fp["pcer"], "UTMOS", fp["mos"], "sa", sa, "PROMOTE" if ok else "REVERT", flush=True)
json.dump(m, open(f"{R}/repo/eval/candidate_wave30.json", "w"), ensure_ascii=False, indent=1)
if ok:
    shutil.rmtree(f"{R}/repo/docs/models"); shutil.copytree(f"{R}/export30", f"{R}/repo/docs/models")
    shutil.copy(f"{R}/repo/docs/metrics.json", f"{R}/repo/eval/metrics.json")
else:
    for d in ("models", "samples"): shutil.rmtree(f"{R}/repo/docs/{d}"); shutil.copytree(f"{R}/best30/{d}", f"{R}/repo/docs/{d}")
    shutil.copy(f"{R}/best30/metrics.json", f"{R}/repo/docs/metrics.json"); shutil.copy(f"{R}/best30/metrics.json", f"{R}/repo/eval/metrics.json")
PY
echo "PROMOTE_STAGE_DONE $(date) $(tail -1 $L/promote30.log | cut -c1-200)" >> $W
cd $R/repo && export GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0 && git add -A docs eval ckpt && git commit -qm "Wave 30 own single-female-voice result ($(tail -1 $L/promote30.log | cut -c1-120))" && git pull -q --rebase --autostash -X ours origin main && git push -q origin HEAD:main; echo "PUSHED $(date)" >> $W
