#!/bin/bash
# Wave 27 (session 27):
# 1. Build studio Sanskrit (Alpa v0.2 Rasa female/male + high-SNR Shrutilipi) + Native Nepali (edge_f/edge_m + s3614/cv_b0487694) -> /work/tts/feats/edge_sa.pt
# 2. Stage 1 (8 min GPU bf16 AM + 8 min GPU bf16 Vocoder on edge_sa.pt, warm-started from Wave 18a with Alpa r-coloured retroflex prior on sa_emb)
# 3. Stage 2: WiSE-FT soup across alphas (0.0 0.08 0.20 0.40) + 2.5 min --sa_only Sanskrit adapter polish (freezing Nepali weights while polishing sa_emb, sa_f0, sa_m0, lang=sa)
# 4. Export ONNX (fp16 + tiny-int8), evaluate Nepali + Tharu + Sanskrit + Svara/Laya, and promote if better!
R=/work/tts; L=$R/logs; cd $R/repo; mkdir -p $R/runs $L; W=$L/wave27.log; export ROMAN_ORT=$R/node/package.json
AM=${AM_MIN:-7}; VOC=${VOC_MIN:-7}; SA_MIN=${SA_MIN:-2.5}; ALPHAS=${ALPHAS:-"0.0 0.12 0.25"}; T=27
echo "WAVE27_START $(date)" > $W
rm -rf $R/best$T; mkdir -p $R/best$T; cp -r docs/models $R/best$T/models; cp -r docs/samples $R/best$T/samples; cp docs/metrics.json $R/best$T/metrics.json
mkdir -p $R/ckpt0; cp ckpt/am.fp16.pt $R/ckpt0/am.pt; cp ckpt/G.fp16.pt $R/ckpt0/G.pt

[ -f $R/feats/edge_sa.pt ] || python3 scripts/distill_sanskrit.py > $L/distill_sa.log 2>&1

[ -n "$SKIP_TRAIN" ] || python3 scripts/train_am.py --set edge_sa --minutes $AM --bs 64 --lr 1.5e-5 --w_pit 0.25 --spk_init sa_f0=edge_f,sa_f1=edge_f,sa_m0=edge_m,sa_m1=edge_m,s3614=edge_f,cv_b0487694=edge_m,cv_13b830fe=edge_m --init $R/ckpt0/am.pt --out $R/runs/am$T > $L/am$T.log 2>&1 &
[ -n "$SKIP_TRAIN" ] || python3 scripts/train_vocoder.py --set edge_sa --minutes $VOC --out $R/runs/voc$T --bs 48 --init $R/ckpt0/G.pt --lr 1.2e-5 --phase 1.2 --gta_p 0 --focus edge_f,edge_m,sa_f0,sa_m0,s3614 --focus_p 0.7 > $L/voc$T.log 2>&1 &
wait
echo "STAGE1_DONE $(date)" >> $W

# Stage 2: Polish Sanskrit adapter (sa_emb, sa_f0, sa_m0, lang[4]) with frozen Nepali weights on top of am27
python3 scripts/train_am.py --set edge_sa --minutes $SA_MIN --bs 48 --lr 4e-4 --w_pit 0.25 --sa_only --init $R/runs/am$T/last.pt --out $R/runs/am${T}p > $L/am${T}p.log 2>&1
cp $R/runs/am${T}p/last.pt $R/runs/am$T/last.pt
echo "STAGE2_SA_POLISH_DONE $(date)" >> $W

for A in $ALPHAS; do
  python3 scripts/soup.py $A $T >> $L/soup$T.log 2>&1
  if [ "$A" = "0.0" ]; then cp $R/ckpt0/G.pt $R/runs/voc${T}s/G.pt; fi
  python3 scripts/export_onnx.py --am $R/runs/am${T}s/last.pt --voc $R/runs/voc${T}s/G.pt --out $R/export$T > $L/export$T.log 2>&1 || { echo "EXPORT_FAIL $A" >> $W; continue; }
  cp $R/export$T/*.onnx $R/export$T/config.json $R/export$T/size.json docs/models/
  git checkout -- docs/samples 2>/dev/null; python3 scripts/evaluate.py --export $R/export$T --out $R/eval$T > $L/eval$T.log 2>&1
  AL=$A python3 - <<'PY' >> $W 2>&1
import json, os, shutil, torch
R = "/work/tts"; T = "27"
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
echo "WAVE27_DONE $(date)" >> $W
