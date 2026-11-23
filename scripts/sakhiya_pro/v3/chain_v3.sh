#!/bin/bash
# After PRO_V3_DONE: freeze v3 -> teacher (v3 synth, batched) + Seed-VC real speech -> merge -> RAM/SSD cache -> ladder (t15 t5 t1) -> ladder_daemon gates.
# setsid nohup bash chain_v3.sh > /work/tts/logs/chain_v3.log 2>&1 < /dev/null &
L=/work/tts/logs; D=/work/sp/distill; R=/work/sp/repo/scripts/sakhiya_pro/distill; V3=/work/sp/repo/scripts/sakhiya_pro/v3; export PYTHONPATH=/work/sp/OmniVoice_src:/work/sp HF_TOKEN=$(cat /tmp/hf_token)
st(){ echo "{\"label\":\"$2\",\"state\":\"$3\",\"note\":\"$4\"}" > /work/sp/stage/$1.json; }
DEAD=$(date -u -d "$(cat /work/sp/deadline_utc.txt)Z" +%s); CORPUS_MIN=${CORPUS_MIN:-55}; TIERS=${TIERS:-"t15 t5 t1"}
until grep -q PRO_V3_DONE $L/pipeline_v3.log 2>/dev/null; do sleep 20; done
python3 $V3/finalize_v3.py > $L/finalize_v3.log 2>&1; tail -n 3 $L/finalize_v3.log
st 50_corpus_d "Teacher corpus: v3 synthesis + Seed-VC real Nepal speech -> anchor tract" run "4 teacher + 6 VC workers, ${CORPUS_MIN} min"
CD=$(( $(date +%s) + CORPUS_MIN*60 )); mkdir -p $D/corpus $D/vc
for i in 0 1 2 3; do TEACHER=/work/sp/final/sakhiya_pro_v3 BATCH=16 NSTEP=20 SP_REF_WAV=/work/sp/v3/ref/anchor_v3.wav SP_REF_JSON=/work/sp/v3/ref/anchor_v3.json setsid nohup python3 $R/gen_corpus.py work $i 4 $CD > $L/gen_w$i.log 2>&1 < /dev/null & done
for i in 0 1 2 3 4 5; do VC_STEPS=16 setsid nohup python3 $V3/vc_corpus.py work $i 6 $CD > $L/vc_w$i.log 2>&1 < /dev/null & done
sleep 30
while pgrep -f "[g]en_corpus.py work|[v]c_corpus.py work" >/dev/null; do
  H=$(python3 -c "import glob,json;f=lambda g:sum(json.loads(l)['dur'] for p in glob.glob(g) for l in open(p) if l.strip().endswith('}'))/3600;print(round(f('$D/corpus/part_*.jsonl'),2),round(f('$D/vc/part_*.jsonl'),2))")
  st 50_corpus_d "Teacher corpus (v3 synth h | Seed-VC real h)" run "$H"; sleep 60
done
for p in $D/vc/part_*.jsonl; do cp $p $D/corpus/part_vc_$(basename $p); done
python3 $R/gen_corpus.py merge > $L/gen_merge.log 2>&1; tail -n 1 $L/gen_merge.log
NPROC=18 python3 $R/prep_cache.py $D/manifest.jsonl $D/cache_train.pt > $L/prep_cache.log 2>&1; tail -n 1 $L/prep_cache.log
st 50_corpus_d "Teacher corpus + cache" done "$(tail -n 1 $L/prep_cache.log | cut -c1-120)"
nohup python3 $R/ladder_daemon.py --every 1200 > $L/ladder_daemon.log 2>&1 < /dev/null &
for r in 1 2 3; do
  left=$(( (DEAD - $(date +%s)) / 60 - 40 )); [ $left -lt 25 ] && break; M=$(( left > 135 ? 135 : left )); EX=""
  [ $r -gt 1 ] && EX="--resume --fresh_clock --lr_mult $(python3 -c "print(0.5**($r-1))")"
  st 60_micro "Student ladder round $r ($TIERS), ${M} min" run "step-50 watchdog; int8 gate every 20 min"
  for t in $TIERS; do setsid nohup python3 $R/train_student.py --tier $t --data $D/cache_train.pt --out $D/runs/$t --minutes $M $EX > $L/student_${t}_r$r.log 2>&1 < /dev/null & done
  sleep 30; while pgrep -f "[t]rain_student.py" >/dev/null; do sleep 30; done
  echo "[$(date +%T)] round $r done"
done
st 60_micro "Student ladder" done "final gate by ladder_daemon"
