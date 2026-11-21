#!/bin/bash
# Wave 41 (session 32, Claude): RSI wave on the GPU, ~25 min, every stage resumable and checkpointed; only a better-than-incumbent model ships.
# 1) build own_ms.pt (REAL speaker ids kept, no teacher audio)  2) real-mel vocoder fine-tune (concurrent, same GPU)  3) AM warm-start from the best
# previous AM  4) GTA vocoder fine-tune on the new AM's predicted mels  5) export one female voice  6) evaluate on held-out real clips + gate.
# usage: AM_MIN=18 VOC_A_MIN=14 VOC_B_MIN=8 bash scripts/wave41.sh      (env INIT_AM / INIT_G override the warm start)
R=/work/tts; L=$R/logs; cd $R/repo41 || exit 1; W=$L/wave41.log; echo "WAVE41_START $(date)" >> $W
INIT_AM=${INIT_AM:-$R/runs/am40_best/last.pt}; [ -f $R/runs/am41/last.pt ] && INIT_AM=$R/runs/am41/last.pt
INIT_G=${INIT_G:-$R/runs/voc40_best/G.pt}; [ -f $R/runs/voc41b/G.pt ] && INIT_G=$R/runs/voc41b/G.pt
# stage 2 starts at once (real mels, GPU never idles while the CPU builds the set)
if [ -n "$VA_PID" ]; then VA=$VA_PID; else
python3 scripts/train_vocoder.py --set own --minutes ${VOC_A_MIN:-14} --out $R/runs/voc41a --bs 32 --init $INIT_G --lr 3e-5 --phase 1.2 --warm 0 > $L/voc41a.log 2>&1 &
VA=$!; fi
[ -f $R/feats/own_ms.pt ] || python3 scripts/build_own.py --raw_spk --sa_rep 1 --vag_rep 1 --ne_rep 2 --out $R/feats/own_ms.pt > $L/build_ms.log 2>&1
echo "BUILD_MS_DONE $(date)" >> $W
python3 scripts/train_am.py --set own_ms --minutes ${AM_MIN:-18} --bs ${BS:-64} --lr ${LR:-6e-5} --w_pit 0.2 --ema 0.995 --log_every 100 \
  --init $INIT_AM --sa_w ${SA_W:-3} --vag_w 1 --out $R/runs/am41 > $L/am41.log 2>&1
echo "TRAIN_AM_DONE $(date) $(grep -c step $L/am41.log)" >> $W
while kill -0 $VA 2>/dev/null; do sleep 3; done; GA=$R/runs/voc41a/G.pt; [ -f $GA ] || GA=$INIT_G
python3 scripts/train_vocoder.py --set own_ms --minutes ${VOC_B_MIN:-8} --out $R/runs/voc41b --bs 32 --init $GA --lr 1.5e-5 --phase 1.2 \
  --gta $R/feats/gta_own_ms.pt --gta_p 0.7 --warm 0 > $L/voc41b.log 2>&1
echo "VOC_B_DONE $(date)" >> $W
SAKHIYA_ONE_VOICE=1 python3 scripts/export_onnx.py --am $R/runs/am41/last.pt --voc $R/runs/voc41b/G.pt --out $R/export41 > $L/export41.log 2>&1 || { echo EXPORT_FAIL >> $W; exit 1; }
echo "EXPORT_DONE $(date)" >> $W
bash scripts/wave41_post.sh >> $L/wave41_post.out 2>&1
echo "WAVE41_DONE $(date)" >> $W
