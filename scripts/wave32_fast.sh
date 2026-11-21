#!/bin/bash
# Wave 31 fast finish (session 31): resume the own-voice AM from runs/am30/last.pt for AM_MIN more minutes, wait for the concurrent
# real-mel vocoder fine-tune (runs/voc31/G.pt), export the single-female-voice model, evaluate on held-out clips and gate (wave30_post.sh).
# GTA vocoder fine-tune is deferred to the next wave (scripts/wave31.sh), so this wave fits a 25-minute session budget.
R=/work/tts; L=$R/logs; cd $R/repo; W=$L/wave30.log; G=$R/ckpt30
for pat in "wave_rs[i].sh" "wave3[01].sh" "train_a[m].py"; do for p in $(pgrep -f "$pat"); do [ $p != $$ ] && kill $p 2>/dev/null; done; done; sleep 2
echo "WAVE32_FAST_START $(date)" >> $W
python3 scripts/train_am.py --set own --minutes ${AM_MIN:-4} --bs 48 --lr ${LR:-2e-4} --w_pit 0.2 --ema 0.99 --log_every 20 --init $R/runs/am30/last.pt --spk_init sakhiya=s0546 --out $R/runs/am30 > $L/am30b.log 2>&1
echo "TRAIN_AM_DONE $(date)" >> $W
until [ -f $R/runs/voc31/G.pt ] && grep -q -E "DONE|saved" $L/voc_plain.log; do sleep 5; done
SAKHIYA_ONE_VOICE=1 python3 scripts/export_onnx.py --am $R/runs/am30/last.pt --voc $R/runs/voc31/G.pt --out $R/export30 > $L/export30.log 2>&1 || { echo EXPORT_FAIL >> $W; exit 1; }
echo "EXPORT_DONE $(date)" >> $W
bash scripts/wave30_post.sh >> $L/run31.out 2>&1
echo "WAVE32_FAST_DONE $(date)" >> $W
