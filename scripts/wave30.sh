#!/bin/bash
# Wave 30 (session 30): OWN single-female-voice model from REAL human speech only (no Edge-TTS, no Alpa audio in the training set).
# Warm-start the AM from the shipped checkpoint (its weights still carry teacher-era knowledge; disclosed in PROVENANCE.md),
# then fine-tune on own.pt (SLR43/FLEURS female Nepali + real Shrutilipi female Sanskrit, one speaker id `sakhiya`).
# CPU box (20 cores, no GPU): time-boxed, checkpoint every wave, vocoder reused (GTA fine-tune is GPU-class work).
R=/work/tts; L=$R/logs; cd $R/repo; W=$L/wave30.log; MIN=${AM_MIN:-22}; echo "WAVE30_START $(date)" >> $W
G=$R/ckpt30; mkdir -p $G $R/runs; INIT=${INIT_AM:-ckpt/am.fp16.pt}; [ -f $R/runs/am30/last.pt ] && INIT=$R/runs/am30/last.pt
cp ckpt/am.fp16.pt $G/am_prev.pt; cp ckpt/G.fp16.pt $G/G_prev.pt
[ -f $R/feats/own.pt ] || python3 scripts/build_own.py > $L/build_own.log 2>&1
echo "BUILD_OWN_DONE $(date)" >> $W
python3 scripts/train_am.py --set own --minutes $MIN --bs ${BS:-24} --lr ${LR:-4e-4} --w_pit 0.2 --ema ${EMA:-0.99} --log_every 20 --init $INIT --spk_init sakhiya=s0546 --out $R/runs/am30 > $L/am30.log 2>&1
echo "TRAIN_AM_DONE $(date)" >> $W
SAKHIYA_ONE_VOICE=1 python3 scripts/export_onnx.py --am $R/runs/am30/last.pt --voc $G/G_prev.pt --out $R/export30 > $L/export30.log 2>&1 || { echo EXPORT_FAIL >> $W; exit 1; }
echo "EXPORT_DONE $(date)" >> $W
