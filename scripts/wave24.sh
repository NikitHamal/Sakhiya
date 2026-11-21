#!/bin/bash
# Wave 24 (session 24): waves 22 and 23 (AM continuation + vocoder) both REVERTED; AM mel validation is flat (0.2996 at step 3000 -> 0.2965 at 9000), i.e. the AM is saturated at this size.
# So this wave keeps the *shipped* acoustic model (Wave 18a weights, 1 min at lr 1e-6 only to dump its own predicted mels) and spends the GPU on the real mismatch: GTA vocoder fine-tune from the shipped vocoder on the shipped AM's predicted mels.
R=/work/tts; L=$R/logs; cd $R/repo; W=$L/wave24.log; export ROMAN_ORT=$R/node/package.json
GTA=${GTA_MIN:-15}; echo "WAVE24_START $(date)" >> $W
G=$R/ckpt24; mkdir -p $G; cp ckpt/am.fp16.pt $G/am0.pt; cp ckpt/G.fp16.pt $G/G0.pt
rm -rf $R/best24; mkdir -p $R/best24; cp -r docs/models $R/best24/models; cp -r docs/samples $R/best24/samples; cp docs/metrics.json $R/best24/metrics.json
python3 scripts/train_am.py --set mix --minutes 1 --bs 64 --lr 1e-6 --w_pit 0.18 --init $G/am0.pt --out $R/runs/am24 > $L/am24.log 2>&1
echo "TRAIN_AM_DONE $(date)" >> $W
python3 scripts/train_vocoder.py --set mix --minutes $GTA --out $R/runs/voc24 --bs 48 --init $G/G0.pt --lr 1.5e-5 --phase 1.2 --gta $R/feats/gta_mix.pt --gta_p 0.7 --focus edge_f,edge_m,s3614 --focus_p 0.5 > $L/voc24.log 2>&1
echo "TRAIN_DONE $(date)" >> $W
python3 scripts/export_onnx.py --am $R/runs/am24/last.pt --voc $R/runs/voc24/G.pt --out $R/export24 > $L/export24.log 2>&1 || { echo EXPORT_FAIL >> $W; exit 1; }
cp $R/export24/*.onnx $R/export24/config.json $R/export24/size.json docs/models/
git checkout -- docs/samples 2>/dev/null; python3 scripts/evaluate.py --export $R/export24 --out $R/eval24 > $L/eval24.log 2>&1
export TAG=24
python3 - <<'PY' > $L/promote$TAG.log 2>&1
import json, os, shutil, torch
R = "/work/tts"; T = os.environ.get("TAG", "18")
g = lambda p, k="sentence": next(s for s in json.load(open(p))["systems"] if s["system"] == "Sakhiya-fp16")[k]
new, old = g(f"{R}/repo/docs/metrics.json"), g(f"{R}/best{T}/metrics.json")
better = (new["mos"] >= old["mos"] - 0.005 and new["pcer"] <= old["pcer"] + 0.002) and (new["mos"] > old["mos"] or new["pcer"] < old["pcer"])
print("new", new["pcer"], new["mos"], "old", old["pcer"], old["mos"], "PROMOTE" if better else "REVERT", flush=True)
if better:
    os.makedirs(f"{R}/repo/ckpt", exist_ok=True); c = torch.load(f"{R}/runs/am{T}/last.pt", weights_only=False)
    c["M"] = {k: v.half() for k, v in c["M"].items()}; torch.save(c, f"{R}/repo/ckpt/am.fp16.pt")
    torch.save({k: v.half() for k, v in torch.load(f"{R}/runs/voc{T}/G.pt", weights_only=False).items()}, f"{R}/repo/ckpt/G.fp16.pt")
    shutil.copy(f"{R}/repo/docs/metrics.json", f"{R}/repo/eval/metrics.json")
else:
    for d in ("models", "samples"): shutil.rmtree(f"{R}/repo/docs/{d}"); shutil.copytree(f"{R}/best{T}/{d}", f"{R}/repo/docs/{d}")
    for d in ("docs", "eval"): shutil.copy(f"{R}/best{T}/metrics.json", f"{R}/repo/{d}/metrics.json")
PY
cat $L/promote$TAG.log >> $L/wave$TAG.log
if grep -q PROMOTE $L/promote$TAG.log; then
    export GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0
    git checkout -- docs/data/status.json 2>/dev/null || true
    git add docs/models docs/samples docs/metrics.json eval/metrics.json ckpt
    git commit -qm "Promote Wave $TAG: $(cat $L/promote$TAG.log)"
    git pull -q --rebase --autostash -X ours origin main
    git push -q origin HEAD:main
fi
echo "WAVE24_DONE $(date)" >> $W
