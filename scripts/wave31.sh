#!/bin/bash
# Wave 31: the missing half of Wave 30. The own-voice acoustic model was trained on real speech, but the vocoder was still the
# teacher-era G_prev (HNR 1.4 dB on Sanskrit = buzzy). This stage fine-tunes the vocoder on GTA (predicted) mels of the own set,
# with phase/periodicity loss (--phase), then exports with that vocoder and runs the same held-out gate (wave30_post.sh).
# usage (after train_am wrote GTA_DONE):  VOC_MIN=14 bash scripts/wave31.sh   (CPU box: 20 cores; GPU: raise VOC_MIN)
R=/work/tts; L=$R/logs; cd $R/repo; W=$L/wave30.log; G=$R/ckpt30; VOC_MIN=${VOC_MIN:-14}
until grep -q GTA_DONE $L/am30.log 2>/dev/null; do sleep 10; done
echo "WAVE31_VOC_START $(date)" >> $W
INITG=$G/G_prev.pt; [ -f $R/runs/voc31/G.pt ] && INITG=$R/runs/voc31/G.pt
python3 scripts/train_vocoder.py --set own --minutes $VOC_MIN --out $R/runs/voc31 --bs 24 --init $INITG --lr 1.5e-5 --phase 1.2 \
  --gta $R/feats/gta_own.pt --gta_p 0.7 --warm 0 > $L/voc30.log 2>&1
echo "WAVE31_VOC_DONE $(date) $(tail -1 $L/voc30.log | cut -c1-160)" >> $W
VG=$R/runs/voc31/G.pt; [ -f $VG ] || VG=$G/G_prev.pt
SAKHIYA_ONE_VOICE=1 python3 scripts/export_onnx.py --am $R/runs/am30/last.pt --voc $VG --out $R/export30 > $L/export30.log 2>&1 || { echo EXPORT_FAIL >> $W; exit 1; }
echo "EXPORT_DONE $(date)" >> $W
bash scripts/wave30_post.sh >> $L/run31.out 2>&1
