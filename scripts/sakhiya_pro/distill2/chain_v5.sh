#!/bin/bash
# Unattended Sakhiya-Pro v3 Voice-Locked Shrink-Ladder Autopilot:
# 1. Builds OT-CFM Sakhiya-Pro v3 Teacher AM (/work/tts/ck/v3_t30_am.pt) + Rectified Cache (/work/sp/distill/v3_rectified_cache.pt)
# 2. Launches all 5 student tiers (t30, t15, t10, t5, t1) concurrently with fast cached targets + STE INT8 QAT
# 3. Runs ladder_daemon.py every 15 min to export INT8 ONNX, run full forensic + Node ORT gate, and promote winning tiers.
L=/work/tts/logs; D=/work/sp/distill; R=/work/sp/repo/scripts/sakhiya_pro/distill2; Q=/work/sp/repo/scripts/sakhiya_pro/distill
export PYTHONPATH=/work/sp/repo:/work/sp/repo/scripts/sakhiya_pro/distill:/work/sp HF_TOKEN=$(cat /tmp/hf_token) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
st(){ echo "{\"label\":\"$2\",\"state\":\"$3\",\"note\":\"$4\"}" > /work/sp/stage/$1.json; }
END_EPOCH=${END_EPOCH:-$(date -u -d "today 19:30" +%s)}

st 58_voc "Pro-v3 OT-CFM Teacher AM + Rectified Cache" run "verify OT-CFM teacher + rectified cache"
python3 $R/train_student_gt.py --tier t30 --prep_only >> $L/prep_v3_teacher.log 2>&1
st 58_voc "Pro-v3 OT-CFM Teacher AM + Rectified Cache" done "$(tail -n 2 $L/prep_v3_teacher.log | tr '\n' ' ' | cut -c1-80)"

MIN=$(( (END_EPOCH - $(date +%s)) / 60 ))
[ $MIN -gt 130 ] && MIN=130
[ $MIN -lt 25 ] && MIN=45

rm -rf $D/runs/t30 $D/runs/t15 $D/runs/t10 $D/runs/t1
mkdir -p $D/runs/t30 $D/runs/t15 $D/runs/t10 $D/runs/t1

st 60_micro "Pro-v3 Shrink Ladder (t30, t15, t10, t1)" run "concurrent ${MIN} min + INT8 QAT"
for T in t30 t15 t10 t1; do
  BS=32; [ $T = t1 ] && BS=48
  if [ $T = t30 ] || [ $T = t15 ]; then BS=18; fi
  nohup python3 $R/train_student_gt.py --tier $T --minutes $MIN --bs $BS > $L/stu_$T.log 2>&1 < /dev/null &
done

sleep 15
nohup python3 $Q/ladder_daemon.py --every 900 --tiers t30,t15,t10,t1 > $L/ladder_daemon.log 2>&1 < /dev/null &
echo "Pro-v3 shrink ladder launched $(date -u) for ${MIN} min"
