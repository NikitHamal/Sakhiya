#!/bin/bash
# Wave 17: EMA-tracked (0.998) AM + consistent-phase Vocoder fine-tuning on the 37.1 h two-voice corpus,
# warm-started from Wave 17a (am17a SWA + voc17a SWA with resumed discriminator D).
R=/work/tts; L=$R/logs; cd $R/repo; mkdir -p $R/runs $L
AM=${AM_MIN:-12}; VOC=${VOC_MIN:-12}; TAG=${TAG:-17}
echo "WAVE${TAG}_START $(date)" > $L/wave$TAG.log
mkdir -p $R/best$TAG; rm -rf $R/best$TAG/models $R/best$TAG/samples; cp -r docs/models $R/best$TAG/models; cp -r docs/samples $R/best$TAG/samples; cp docs/metrics.json $R/best$TAG/metrics.json
export ROMAN_ORT=$R/node/package.json

python3 scripts/train_am.py --set edge --minutes $AM --bs 64 --lr 3e-5 --w_pit 0.15 --init $R/runs/am17a/last.pt --out $R/runs/am$TAG > $L/am$TAG.log 2>&1 &
python3 scripts/train_vocoder.py --set edge --minutes $VOC --out $R/runs/voc$TAG --bs 48 --init $R/runs/voc17a/G.pt --lr 3e-5 --phase 1.2 --gta $R/feats/gta_edge.pt --gta_p 0.5 --focus edge_f,edge_m --focus_p 1.0 > $L/voc$TAG.log 2>&1 &
wait
echo "TRAIN_DONE $(date)" >> $L/wave$TAG.log

python3 scripts/export_onnx.py --am $R/runs/am$TAG/last.pt --voc $R/runs/voc$TAG/G.pt --out $R/export$TAG > $L/export$TAG.log 2>&1 || { echo EXPORT_FAIL >> $L/wave$TAG.log; exit 1; }
cp $R/export$TAG/*.onnx $R/export$TAG/config.json $R/export$TAG/size.json docs/models/
git checkout -- docs/samples 2>/dev/null; python3 scripts/evaluate.py --export $R/export$TAG --out $R/eval$TAG > $L/eval$TAG.log 2>&1

python3 - <<'PY' > $L/promote$TAG.log 2>&1
import json, os, shutil, torch
R = "/work/tts"; T = os.environ.get("TAG", "17")
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
echo "WAVE${TAG}_DONE $(date)" >> $L/wave$TAG.log
