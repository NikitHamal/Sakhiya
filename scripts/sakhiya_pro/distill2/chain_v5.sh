#!/bin/bash
# Unattended: (fixed) vocoder GANs finish -> 4 students (t15 t10 t5 t1) until END_UTC -> ladder_daemon gates every 20 min.
# Launch detached:  setsid nohup bash chain_v5.sh > /work/tts/logs/chain_v5.log 2>&1 < /dev/null &
L=/work/tts/logs; D=/work/sp/distill; R=/work/sp/repo/scripts/sakhiya_pro/distill2; Q=/work/sp/repo/scripts/sakhiya_pro/distill
export PYTHONPATH=/work/sp:/work/sp/repo HF_TOKEN=$(cat /tmp/hf_token) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
st(){ echo "{\"label\":\"$2\",\"state\":\"$3\",\"note\":\"$4\"}" > /work/sp/stage/$1.json; }
END_EPOCH=${END_EPOCH:-$(date -u -d "today 19:30" +%s)}
st 58_voc "Pro-v3 vocoder GAN (320 + 144 ch, fixed fuse bug)" run "55 min each"
for i in $(seq 1 200); do
  if grep -q VOC_GT_DONE $L/voc320.log && grep -q VOC_GT_DONE $L/voc144.log; then break; fi
  sleep 30
done
tail -n 2 $L/voc320.log $L/voc144.log
st 58_voc "Pro-v3 vocoder GAN (320 + 144 ch, fixed fuse bug)" done "$(grep VAL $L/voc320.log | tail -n 1 | cut -c1-60)"
MIN=$(( (END_EPOCH - $(date +%s)) / 60 ))
nohup python3 $Q/ladder_daemon.py --every 1200 --tiers t15,t10,t5,t1 > $L/ladder_daemon.log 2>&1 < /dev/null &
st 60_micro "Student ladder (GT Pro-v3, fixed vocoder, GAN+WavLM+QAT)" run "t15 t10 t5 t1 concurrent, ${MIN} min"
for T in t15 t5 t1 t10; do
  BS=40; [ $T = t1 ] && BS=48
  nohup python3 $R/train_student_gt.py --tier $T --minutes $MIN --bs $BS --resume > $L/stu_$T.log 2>&1 < /dev/null &
done
echo "students launched $(date -u) for ${MIN} min"
