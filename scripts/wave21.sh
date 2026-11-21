#!/bin/bash
# Wave 21 (time-boxed to fit the ~30 min box lifetime; data tiers cached to a GitHub release via cache_data.py): rebuild the real-speech tiers + teacher corpus on a fresh box, then warm-started AM + vocoder (GT-mel, then GTA) on HOURS of data.
# Resumable (marker files).  usage: bash scripts/wave20.sh   (AM_MIN/VOC_MIN/GTA_MIN override durations)
R=/work/tts; L=$R/logs; cd $R/repo; mkdir -p $R/runs $R/feats $L; W=$L/wave21.log; export ROMAN_ORT=$R/node/package.json
AM=${AM_MIN:-16}; VOC=${VOC_MIN:-14}; GTA=${GTA_MIN:-6}; mk() { touch $L/w20.$1.done; echo "$1 $(date +%H:%M:%S)" >> $W; }; has() { [ -f $L/w20.$1.done ]; }
echo "WAVE21_START $(date)" >> $W
[ -f $L/cache_pulled ] && { for k in curate mine sa sadl distill feats_edge mix; do mk $k; done; touch $L/dl.done; grep -q SLR54DL_DONE $L/slr54dl.log 2>/dev/null || echo SLR54DL_DONE >> $L/slr54dl.log; }
while [ ! -f $L/dl.done ] || ! grep -q SLR54DL_DONE $L/slr54dl.log; do sleep 10; done
(has sadl || { python3 scripts/dl_sanskrit.py ${SA_SHARDS:-6} > $L/sadl.log 2>&1; mk sadl; }) &
has curate || { python3 scripts/curate.py > $L/curate.log 2>&1 && mk curate; }
has distill || { (python3 scripts/distill_edge.py --round 2 --n ${EDGE_N:-3200} --conc 40 > $L/distill.log 2>&1; touch $L/w20.distill.done) & }
has mine || { python3 scripts/mine_slr54.py --n ${MINE_N:-8000} --topspk 120 --percap 90 --minn 30 --maxspk 90 --cer 0.25 --snr 20 --mms --procs 18 > $L/mine.log 2>&1 && mk mine; }
while ! has sadl; do sleep 5; done
has sa || { python3 scripts/mine_sanskrit.py --shards ${SA_SHARDS:-6} --hours 14 --procs 18 > $L/sa.log 2>&1 && mk sa; }
while ! has distill; do sleep 10; done
has feats_edge || { python3 scripts/feats_edge.py --cer 0.30 --procs 18 > $L/feats_edge.log 2>&1 && mk feats_edge; }
has mix || { python3 scripts/build_mix.py > $L/mix.log 2>&1 && mk mix; [ -f $L/cache_pulled ] || python3 scripts/cache_data.py push > $L/cache_push.log 2>&1 & }
G=$R/ckpt21; mkdir -p $G; cp ckpt/am.fp16.pt $G/am0.pt; cp ckpt/G.fp16.pt $G/G0.pt; [ -f $R/runs/am20/last.pt ] && cp $R/runs/am20/last.pt $G/am0.pt; [ -f $R/runs/voc20/G.pt ] && cp $R/runs/voc20/G.pt $G/G0.pt; [ -f $R/relay/am.fp16.pt ] && [ ! -f $R/runs/am20/last.pt ] && cp $R/relay/am.fp16.pt $G/am0.pt; [ -f $R/relay/G.fp16.pt ] && [ ! -f $R/runs/voc20/G.pt ] && cp $R/relay/G.fp16.pt $G/G0.pt
rm -rf $R/best21; mkdir -p $R/best21; cp -r docs/models $R/best21/models; cp -r docs/samples $R/best21/samples; cp docs/metrics.json $R/best21/metrics.json
python3 scripts/train_am.py --set mix --minutes $AM --bs 64 --lr 5e-5 --w_pit 0.18 --init $G/am0.pt --out $R/runs/am21 > $L/am21.log 2>&1 &
python3 scripts/train_vocoder.py --set mix --minutes $VOC --out $R/runs/voc21a --bs 48 --init $G/G0.pt --lr 2e-5 --phase 1.2 --gta_p 0 --focus edge_f,edge_m,s3614 --focus_p 0.5 > $L/voc21a.log 2>&1 &
wait; echo "TRAIN_AM_DONE $(date)" >> $W
python3 scripts/train_vocoder.py --set mix --minutes $GTA --out $R/runs/voc21 --bs 48 --init $R/runs/voc21a/G.pt --lr 1.5e-5 --phase 1.2 --gta $R/feats/gta_mix.pt --gta_p 0.6 --focus edge_f,edge_m,s3614 --focus_p 0.5 > $L/voc21.log 2>&1
echo "TRAIN_DONE $(date)" >> $W
python3 scripts/export_onnx.py --am $R/runs/am21/last.pt --voc $R/runs/voc21/G.pt --out $R/export21 > $L/export21.log 2>&1 || { echo EXPORT_FAIL >> $W; exit 1; }
cp $R/export21/*.onnx $R/export21/config.json $R/export21/size.json docs/models/
git checkout -- docs/samples 2>/dev/null; python3 scripts/evaluate.py --export $R/export21 --out $R/eval21 > $L/eval21.log 2>&1
export TAG=21
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
echo "WAVE21_DONE $(date)" >> $W
