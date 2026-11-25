#!/bin/bash
# After Sakhiya-Pro v2 finishes: teacher corpus -> shards -> 5 ladder students concurrently (<=135 min rounds, resumed with lower LR)
# -> ladder_daemon gates the real int8 exports every 20 min.  Launch detached:  setsid nohup bash distill_chain.sh > /work/tts/logs/distill_chain.log 2>&1 < /dev/null &
L=/work/tts/logs; D=/work/sp/distill; R=/work/sp/repo/scripts/sakhiya_pro/distill; export PYTHONPATH=/work/sp/OmniVoice_src:/work/sp HF_TOKEN=$(cat /tmp/hf_token)
mkdir -p $D/runs $D/corpus /work/sp/stage /work/sp/tiers /work/sp/listen
st(){ echo "{\"label\":\"$2\",\"state\":\"$3\",\"note\":\"$4\"}" > /work/sp/stage/$1.json; }
DEAD=$(date -u -d "$(cat /work/sp/deadline_utc.txt)Z" +%s); NW=${NW:-7}; CORPUS_MIN=${CORPUS_MIN:-60}; TIERS=${TIERS:-"t30 t15 t10 t1"}
until grep -q PRO_V2_DONE $L/pipeline_v2.log 2>/dev/null; do sleep 30; done
[ -f $D/texts.jsonl ] || python3 $R/gen_corpus.py texts > $L/gen_texts.log 2>&1
st 50_corpus_d "Distillation corpus from Pro v2 (strict filters)" run "$NW teacher workers, up to ${CORPUS_MIN} min"
CD=$(( $(date +%s) + CORPUS_MIN*60 )); [ $CD -gt $((DEAD-3*3600)) ] && CD=$((DEAD-3*3600))
for i in $(seq 0 $((NW-1))); do setsid nohup python3 $R/gen_corpus.py work $i $NW $CD > $L/gen_w$i.log 2>&1 < /dev/null & done
sleep 20
while pgrep -f "[g]en_corpus.py work" >/dev/null; do
  H=$(python3 -c "import glob,json;print(round(sum(json.loads(l)['dur'] for p in glob.glob('$D/corpus/part_*.jsonl') for l in open(p) if l.strip().endswith('}'))/3600,2))"); st 50_corpus_d "Distillation corpus from Pro v2 (strict filters)" run "kept ${H} h so far"; sleep 60
done
python3 $R/gen_corpus.py merge > $L/gen_merge.log 2>&1; cat $L/gen_merge.log | tail -n 2
NPROC=18 python3 $R/prep_cache.py $D/manifest.jsonl $D/cache_train.pt > $L/prep_cache.log 2>&1; tail -n 1 $L/prep_cache.log
st 50_corpus_d "Distillation corpus from Pro v2 (strict filters)" done "$(tail -n 1 $L/prep_cache.log | cut -c1-120)"
nohup python3 $R/ladder_daemon.py --every 1200 > $L/ladder_daemon.log 2>&1 < /dev/null &
for r in 1 2 3 4; do
  left=$(( (DEAD - $(date +%s)) / 60 - 40 )); [ $left -lt 25 ] && break; M=$(( left > 135 ? 135 : left )); EX=""
  [ $r -gt 1 ] && EX="--resume --fresh_clock --lr_mult $(python3 -c "print(0.5**($r-1))")"
  st 60_micro "Student ladder round $r ($TIERS), ${M} min each" run "t30/t15/t10/t5/t1 train concurrently; int8 gate every 20 min"
  for t in $TIERS; do setsid nohup python3 $R/train_student.py --tier $t --data $D/cache_train.pt --out $D/runs/$t --minutes $M $EX > $L/student_${t}_r$r.log 2>&1 < /dev/null & done
  sleep 30; while pgrep -f "[t]rain_student.py" >/dev/null; do sleep 30; done
  echo "[$(date +%T)] round $r done"
done
st 60_micro "Student ladder" done "all rounds finished; final gate by ladder_daemon"
