#!/bin/bash
# chain: wait for Seed-VC -> build dataset -> tokenize -> guarded full FT -> eval every 300 steps on synthesized audio
cd /work/sp; L=/work/tts/logs; export PYTHONPATH=/work/sp/OmniVoice_src:/work/sp
NAME=${1:-ft1}; STEPS=${2:-1500}; LR=${3:-5e-6}
while pgrep -f 'vc_[bv]' >/dev/null; do sleep 10; done
echo "[$(date +%T)] VC finished"
python3 /work/sp/build_ft_data.py > $L/build_ft_data.log 2>&1 || { echo build failed; exit 1; }
echo "[$(date +%T)] dataset built"; cat /work/sp/ft/dataset_report.json
bash /work/sp/repo/scripts/sakhiya_pro/ft_tokenize.sh > $L/tokenize.log 2>&1 || { echo tokenize failed; exit 1; }
echo "[$(date +%T)] tokenized"
python3 /work/sp/repo/scripts/sakhiya_pro/eval_ckpt.py /work/sp/models/OmniVoice base_zeroshot > $L/eval_base.log 2>&1
python3 /work/sp/repo/scripts/sakhiya_pro/guarded_train.py --name $NAME --steps $STEPS --lr $LR --save_steps 150
echo "[$(date +%T)] train rc=$?"
for s in 300 600 900 1200 1500; do
  [ -d /work/sp/exp/$NAME/checkpoint-$s ] && python3 /work/sp/repo/scripts/sakhiya_pro/eval_ckpt.py /work/sp/exp/$NAME/checkpoint-$s ${NAME}_s$s > $L/eval_${NAME}_s$s.log 2>&1
done
echo "[$(date +%T)] PIPELINE_DONE"
