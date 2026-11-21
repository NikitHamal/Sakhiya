#!/bin/bash
# Wave 26 (session 26): weight-space soup instead of "more steps". Waves 21-25 showed that plain continuation / GTA / blend recipes lose to Wave 18a on the gate.
# Lever: fine-tune the shipped AM (higher F0 weight -> more natural pitch contour) and the shipped vocoder (GT mels, consistent-phase loss, per-voice focus) for a
# short time-boxed run, then WiSE-FT interpolate each with the shipped weights (alpha list) and let the gate pick the best candidate (each exported + evaluated).
R=/work/tts; L=$R/logs; cd $R/repo; mkdir -p $R/runs $L; W=$L/wave26.log; export ROMAN_ORT=$R/node/package.json
AM=${AM_MIN:-11}; VOC=${VOC_MIN:-11}; ALPHAS=${ALPHAS:-"0.5 1.0"}; T=26
echo "WAVE26_START $(date)" > $W
rm -rf $R/best$T; mkdir -p $R/best$T; cp -r docs/models $R/best$T/models; cp -r docs/samples $R/best$T/samples; cp docs/metrics.json $R/best$T/metrics.json
mkdir -p $R/ckpt0; cp ckpt/am.fp16.pt $R/ckpt0/am.pt; cp ckpt/G.fp16.pt $R/ckpt0/G.pt

[ -n "$SKIP_TRAIN" ] || python3 scripts/train_am.py --set ${SET:-mix} --minutes $AM --bs 64 --lr 1.2e-5 --w_pit 0.3 --init $R/ckpt0/am.pt --out $R/runs/am$T > $L/am$T.log 2>&1 &
[ -n "$SKIP_TRAIN" ] || python3 scripts/train_vocoder.py --set ${SET:-mix} --minutes $VOC --out $R/runs/voc$T --bs 48 --init $R/ckpt0/G.pt --lr 1.5e-5 --phase 1.2 --gta_p 0 --focus ${FOCUS:-edge_f,edge_m,s3614} --focus_p 0.6 > $L/voc$T.log 2>&1 &
wait
echo "TRAIN_DONE $(date)" >> $W

for A in $ALPHAS; do
  python3 scripts/soup.py $A $T >> $L/soup$T.log 2>&1
  python3 scripts/export_onnx.py --am $R/runs/am${T}s/last.pt --voc $R/runs/voc${T}s/G.pt --out $R/export$T > $L/export$T.log 2>&1 || { echo "EXPORT_FAIL $A" >> $W; continue; }
  cp $R/export$T/*.onnx $R/export$T/config.json $R/export$T/size.json docs/models/
  git checkout -- docs/samples 2>/dev/null; python3 scripts/evaluate.py --export $R/export$T --out $R/eval$T > $L/eval$T.log 2>&1
  AL=$A python3 - <<'PY' >> $W 2>&1
import json, os, shutil, torch
R = "/work/tts"; T = "26"
g = lambda p: next(s for s in json.load(open(p))["systems"] if s["system"] == "Sakhiya-fp16")["sentence"]
new, old = g(f"{R}/repo/docs/metrics.json"), g(f"{R}/best{T}/metrics.json")
better = (new["mos"] >= old["mos"] - 0.005 and new["pcer"] <= old["pcer"] + 0.002) and (new["mos"] > old["mos"] or new["pcer"] < old["pcer"])
print("alpha", os.environ["AL"], "new", new["pcer"], new["mos"], "old", old["pcer"], old["mos"], "PROMOTE" if better else "REVERT", flush=True)
if better:
    c = torch.load(f"{R}/runs/am{T}s/last.pt", weights_only=False); c["M"] = {k: v.half() for k, v in c["M"].items()}; torch.save(c, f"{R}/repo/ckpt/am.fp16.pt")
    torch.save({k: v.half() for k, v in torch.load(f"{R}/runs/voc{T}s/G.pt", weights_only=False).items()}, f"{R}/repo/ckpt/G.fp16.pt")
    shutil.copy(f"{R}/repo/docs/metrics.json", f"{R}/repo/eval/metrics.json")
    for d in ("models", "samples"): shutil.rmtree(f"{R}/best{T}/{d}"); shutil.copytree(f"{R}/repo/docs/{d}", f"{R}/best{T}/{d}")
    shutil.copy(f"{R}/repo/docs/metrics.json", f"{R}/best{T}/metrics.json")
else:
    for d in ("models", "samples"): shutil.rmtree(f"{R}/repo/docs/{d}"); shutil.copytree(f"{R}/best{T}/{d}", f"{R}/repo/docs/{d}")
    for d in ("docs", "eval"): shutil.copy(f"{R}/best{T}/metrics.json", f"{R}/repo/{d}/metrics.json")
PY
done
echo "WAVE26_DONE $(date)" >> $W
