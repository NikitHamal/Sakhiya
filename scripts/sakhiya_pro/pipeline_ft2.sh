#!/bin/bash
# full run: wait SLR43 VC -> rebuild dataset -> tokenize -> guarded FT -> audio evals -> pick best -> avg -> final dir -> render ep01 + 15-clip pack
cd /work/sp; L=/work/tts/logs; export PYTHONPATH=/work/sp/OmniVoice_src:/work/sp
R=/work/sp/repo/scripts; N=ft2; STEPS=${1:-1800}; LR=${2:-8e-6}
while pgrep -f 'vc_slr[.]py' >/dev/null || pgrep -f 'pipeline_ft1[.]sh' >/dev/null; do sleep 10; done
echo "[$(date +%T)] VC + pilot finished"
cp $R/sakhiya_pro/build_ft_data.py /work/sp/build_ft_data.py
python3 /work/sp/build_ft_data.py > $L/build_ft_data2.log 2>&1 || { echo build failed; exit 1; }
cp /work/sp/ft/dataset_report.json /work/sp/ft/dataset_report_ft2.json; cat /work/sp/ft/dataset_report.json
bash $R/sakhiya_pro/ft_tokenize.sh > $L/tokenize2.log 2>&1 || { echo tokenize failed; exit 1; }
echo "[$(date +%T)] tokenized"
python3 $R/sakhiya_pro/guarded_train.py --name $N --steps $STEPS --lr $LR --batch_tokens 8192 --save_steps 150
echo "[$(date +%T)] train rc=$?"
for s in 600 900 1200 1500 1800; do
  [ -d /work/sp/exp/$N/checkpoint-$s ] && python3 $R/sakhiya_pro/eval_ckpt.py /work/sp/exp/$N/checkpoint-$s ${N}_s$s > $L/eval_${N}_s$s.log 2>&1
done
python3 $R/sakhiya_pro/select_best.py ${N}_ base_zeroshot > $L/select_$N.log 2>&1; cat $L/select_$N.log
BEST=$(grep '^BEST' $L/select_$N.log | awk '{print $2}'); S=${BEST##*_s}
if [[ "$BEST" == ${N}_s* ]]; then
  CK=""; for d in $((S-150)) $S $((S+150)); do [ -d /work/sp/exp/$N/checkpoint-$d ] && CK="$CK /work/sp/exp/$N/checkpoint-$d"; done
  python3 $R/sakhiya_pro/avg_ckpt.py /work/sp/exp/${N}_avg $CK > $L/avg.log 2>&1
  python3 $R/sakhiya_pro/eval_ckpt.py /work/sp/exp/${N}_avg ${N}_avg > $L/eval_${N}_avg.log 2>&1
  python3 $R/sakhiya_pro/select_best.py ${N}_ base_zeroshot > $L/select_$N.log 2>&1; cat $L/select_$N.log
  BEST=$(grep '^BEST' $L/select_$N.log | awk '{print $2}')
  if [ "$BEST" == "${N}_avg" ]; then SRC=/work/sp/exp/${N}_avg; else SRC=/work/sp/exp/$N/checkpoint-${BEST##*_s}; fi
else SRC=/work/sp/models/OmniVoice; fi
echo "[$(date +%T)] FINAL=$BEST from $SRC"
F=/work/sp/final/sakhiya_pro_v1; rm -rf $F; mkdir -p $F; for f in $SRC/*; do case $(basename $f) in optimizer*|scheduler*|random_states*|*.pkl|*.bin) ;; *) cp -r $f $F/;; esac; done
cp /work/sp/ref/anchor_ref.wav /work/sp/ref/anchor_ref.json $F/; cp $R/sakhiya_pro_render.py $R/sakhiya_pro/sp_common.py $R/sakhiya_pro/bake_sents.py $F/; echo $BEST > $F/SELECTED_FROM.txt
cd /work/sp/repo
python3 scripts/sakhiya_pro_render.py --model $F --pack --out /work/sp/render/pack > $L/render_pack.log 2>&1
python3 scripts/sakhiya_pro_render.py --model $F --ep01 --out /work/sp/render/ep01 > $L/render_ep01.log 2>&1
echo "[$(date +%T)] PIPELINE2_DONE"
