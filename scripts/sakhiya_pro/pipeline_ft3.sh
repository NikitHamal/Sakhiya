#!/bin/bash
# ft3: gentle sweep (low LR, 100-step ckpts) to recover UTMOS/CER of the base model while keeping the anchor voice.
# train (guarded: 150-min cap + step-50 gate) -> eval every ckpt on synthesized audio -> rank vs ft2_s900 + base.
cd /work/sp; L=/work/tts/logs; export PYTHONPATH=/work/sp/OmniVoice_src:/work/sp
R=/work/sp/repo/scripts; N=ft3; STEPS=${1:-1200}; LR=${2:-3e-6}
python3 $R/sakhiya_pro/guarded_train.py --name $N --steps $STEPS --lr $LR --batch_tokens 8192 --save_steps 100
echo "[$(date +%T)] train rc=$?"
for d in /work/sp/exp/$N/checkpoint-*; do
  s=${d##*-}; [ $((s % 200)) -eq 0 ] || continue
  python3 $R/sakhiya_pro/eval_ckpt.py $d ${N}_s$s > $L/eval_${N}_s$s.log 2>&1
done
python3 $R/sakhiya_pro/select_best.py ${N}_ base_zeroshot ft2_s900 > $L/select_$N.log 2>&1; cat $L/select_$N.log
echo "[$(date +%T)] PIPELINE3_DONE"
