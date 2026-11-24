#!/bin/bash
# Unattended ground-truth Pro-v3 ladder chain.  Launch detached:
#   setsid nohup bash /work/sp/repo/scripts/sakhiya_pro/distill2/chain_gt.sh > /work/tts/logs/chain_gt.log 2>&1 < /dev/null &
# corpus workers (already running) -> prep_gt -> Pro-v3 vocoder GANs (320 + 144) -> students (t15, t5, t1, t10) -> ladder_daemon gates + promotes.
L=/work/tts/logs; D=/work/sp/distill; R=/work/sp/repo/scripts/sakhiya_pro/distill2; Q=/work/sp/repo/scripts/sakhiya_pro/distill
export PYTHONPATH=/work/sp:/work/sp/repo HF_TOKEN=$(cat /tmp/hf_token) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
st(){ echo "{\"label\":\"$2\",\"state\":\"$3\",\"note\":\"$4\"}" > /work/sp/stage/$1.json; }
git_pull(){ (cd /work/sp/repo && GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0 git pull -q --rebase origin main); }
DL=$(cat /work/tts/gen_deadline 2>/dev/null || echo $(( $(date +%s) + 60 )))
STU_MIN=${STU_MIN:-290}

st 50_corpus_d "Pro-v3 teacher corpus (batched, strict filters)" run "6 workers until deadline"
while [ $(grep -l "DONE tried" $L/gen_w[0-5].log 2>/dev/null | wc -l) -lt 6 ] && [ $(date +%s) -lt $((DL+600)) ]; do sleep 20; done
pkill -f "[r]unpy.run_path"; sleep 3
git_pull
st 50_corpus_d "Pro-v3 teacher corpus (batched, strict filters)" done "workers finished"

st 55_gt_prep "Ground-truth features: Pro-v3 audio + CTC-aligned durations" run "prep_gt"
python3 $R/prep_gt.py --out $D/gt_feats.pt --nproc 18 > $L/prep_gt.log 2>&1
tail -n 3 $L/prep_gt.log
if ! grep -q PREP_GT_DONE $L/prep_gt.log; then st 55_gt_prep "Ground-truth features" fail "see prep_gt.log"; exit 1; fi
st 55_gt_prep "Ground-truth features: Pro-v3 audio + CTC-aligned durations" done "$(grep '^{' $L/prep_gt.log | tail -n 1 | cut -c1-160)"

st 58_voc "Pro-v3 vocoder GAN (320 + 144 ch)" run "40 min"
nohup python3 $R/train_voc_gt.py --ch 320 --init /work/tts/ck/v5_c1_voc.pt --minutes 40 > $L/voc320.log 2>&1 < /dev/null &
P1=$!
nohup python3 $R/train_voc_gt.py --ch 144 --init /work/tts/ck/v14_voc144.pt --minutes 36 > $L/voc144.log 2>&1 < /dev/null &
P2=$!
wait $P1 $P2
tail -n 2 $L/voc320.log $L/voc144.log
st 58_voc "Pro-v3 vocoder GAN (320 + 144 ch)" done "$(tail -n 1 $L/voc320.log | cut -c1-80)"

nohup python3 $Q/ladder_daemon.py --every 1200 --tiers t30,t15,t10,t5,t1 > $L/ladder_daemon.log 2>&1 < /dev/null &
st 60_micro "Student ladder (ground-truth Pro-v3, GAN + WavLM + QAT)" run "t15 t5 t1 t10 concurrent, ${STU_MIN} min"
for T in t15 t5 t1 t10; do
  BS=24; [ $T = t1 ] && BS=32
  nohup python3 $R/train_student_gt.py --tier $T --minutes $STU_MIN --bs $BS --resume > $L/stu_$T.log 2>&1 < /dev/null &
done
sleep 1; echo "students launched $(date -u)"
