#!/bin/bash
# Wave 8 (session 10): TWO-VOICE DISTILLATION. AM + vocoder warm-start from wave-7 and fine-tune on ~10 h/voice of clean teacher speech
# (feats/edge.pt from scripts/distill_edge.py + feats_edge.py; research-only licence). Promote gate: UTMOS up, pCER not worse than +0.012.
R=/work/tts; L=$R/logs; cd $R/repo; mkdir -p $R/runs $L
AM=${AM_MIN:-18}; VOC=${VOC_MIN:-16}; AMI=${AM_INIT:-$R/runs/am8/last.pt}; VI=${VOC_INIT:-$R/runs/best_voc6_G.pt}; TAG=${TAG:-9}
push() { export GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0; git checkout -- docs/data/status.json 2>/dev/null || true; git add -A docs eval ckpt STATUS.md LEDGER.json scripts sakhiya monitor 2>/dev/null; git commit -qm "$1"; git pull -q --rebase -X ours origin main; git push -q origin HEAD:main; }

echo "WAVE8_START $(date)" > $L/wave8.log
[ -f $R/feats/edge.pt ] || python3 scripts/feats_edge.py > $L/feats_edge.log 2>&1
mkdir -p $R/best8; rm -rf $R/best8/models $R/best8/samples; cp -r docs/models $R/best8/models; cp -r docs/samples $R/best8/samples 2>/dev/null; cp docs/metrics.json $R/best8/metrics.json
export ROMAN_ORT=$R/node/package.json

python3 scripts/train_am.py --set edge --minutes $AM --bs 64 --lr 3e-4 --align_warm 500 --init $AMI --spk_init edge_f=s3614,edge_m=cv_13b830fe --out $R/runs/am$TAG > $L/am$TAG.log 2>&1 &
python3 scripts/train_vocoder.py --set edge --minutes $VOC --out $R/runs/voc$TAG --bs 48 --init $VI --lr 8e-5 --phase 1.0 --focus edge_f,edge_m --focus_p 1.0 > $L/voc$TAG.log 2>&1 &
wait
echo "TRAIN_DONE $(date)" >> $L/wave8.log
python3 scripts/export_onnx.py --am $R/runs/am$TAG/last.pt --voc $R/runs/voc$TAG/G.pt --out $R/export$TAG > $L/export$TAG.log 2>&1 || { echo EXPORT_FAIL >> $L/wave8.log; exit 1; }
cp $R/export$TAG/*.onnx $R/export$TAG/config.json $R/export$TAG/size.json docs/models/
git checkout -- docs/samples 2>/dev/null; python3 scripts/evaluate.py --export $R/export$TAG --out $R/eval$TAG > $L/eval$TAG.log 2>&1

python3 - <<'PY' > $L/promote8.log 2>&1
import json, shutil, torch, os
R = "/work/tts"; T = os.environ.get("TAG", "9"); g = lambda p: next(s for s in json.load(open(p))["systems"] if s["system"] == "Sakhiya-fp16")["sentence"]
new, old = g(f"{R}/repo/docs/metrics.json"), g(f"{R}/best8/metrics.json")
better = new["pcer"] <= old["pcer"] + 0.012 and new["mos"] > old["mos"] + 0.03
print("new", new["pcer"], new["mos"], "old", old["pcer"], old["mos"], "PROMOTE" if better else "REVERT")
if better:
    os.makedirs(f"{R}/repo/ckpt", exist_ok=True); c = torch.load(f"{R}/runs/am{T}/last.pt", weights_only=False)
    c["M"] = {k: v.half() for k, v in c["M"].items()}; torch.save(c, f"{R}/repo/ckpt/am.fp16.pt")
    torch.save({k: v.half() for k, v in torch.load(f"{R}/runs/voc{T}/G.pt", weights_only=False).items()}, f"{R}/repo/ckpt/G.fp16.pt")
    shutil.copy(f"{R}/repo/docs/metrics.json", f"{R}/repo/eval/metrics.json")
else:
    for d in ("models", "samples"): shutil.rmtree(f"{R}/repo/docs/{d}"); shutil.copytree(f"{R}/best8/{d}", f"{R}/repo/docs/{d}")
    shutil.copy(f"{R}/best8/metrics.json", f"{R}/repo/docs/metrics.json"); shutil.copy(f"{R}/best8/metrics.json", f"{R}/repo/eval/metrics.json")
PY
cat $L/promote8.log >> $L/wave8.log
push "wave 8: two-voice distillation ($(tail -n 1 $L/promote8.log | awk '{print $NF}'))"
echo WAVE8_DONE >> $L/wave8.log
