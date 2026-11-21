#!/bin/bash
# Wave 43 (session 33): Sanskrit-first RSI wave.  Two AM variants train CONCURRENTLY on the one GPU (A = balanced, B = Sanskrit/Vedic-heavy),
# both warm-start from the real-speaker-id am41 checkpoint on the ॐ-fixed + curated data (own_ms43), are folded to a REAL-HUMAN-ONLY single
# female voice (no Edge-TTS embedding), exported with the incumbent vocoder, checked for an audible ॐ, evaluated on held-out real clips,
# and the relative gate (scripts/gate41.py, score must beat the incumbent) promotes at most one.  ~22 min wall clock, every stage checkpointed.
# usage: AM_MIN=16 bash scripts/wave43.sh        (run from /work/tts/repo33)
R=/work/tts; L=$R/logs; cd $R/repo33 || exit 1; W=$L/wave43.log; echo "WAVE43_START $(date)" >> $W
INIT_AM=${INIT_AM:-$R/runs/am41/raw_last.pt}; INIT_G=${INIT_G:-$R/runs/voc42/G.pt}
[ -f $R/feats/own_ms43.pt ] || python3 scripts/build_own43.py > $L/build43.log 2>&1
echo "BUILD43_DONE $(date) $(tail -3 $L/build43.log | tr '\n' ' ' | cut -c1-300)" >> $W
python3 scripts/train_am.py --set own_ms43 --minutes ${AM_MIN:-16} --bs 64 --lr 5e-5 --w_pit 0.2 --ema 0.995 --log_every 100 --init $INIT_AM --sa_w 4 --vag_w 2 --out $R/runs/am43a > $L/am43a.log 2>&1 &
PA=$!
python3 scripts/train_am.py --set own_ms43 --minutes ${AM_MIN:-16} --bs 64 --lr 9e-5 --w_pit 0.3 --ema 0.995 --log_every 100 --init $INIT_AM --sa_w 8 --vag_w 3 --out $R/runs/am43b > $L/am43b.log 2>&1 &
PB=$!
wait $PA $PB; echo "TRAIN43_DONE $(date)" >> $W
for v in a b; do
  python3 scripts/fold43.py $R/runs/am43$v/last.pt $R/runs/am43$v/folded.pt >> $L/fold43.log 2>&1
  SAKHIYA_ONE_VOICE=1 python3 scripts/export_onnx.py --am $R/runs/am43$v/folded.pt --voc $INIT_G --out $R/export43$v > $L/export43$v.log 2>&1 || { echo "EXPORT43$v FAIL" >> $W; continue; }
  python3 scripts/check_om.py $R/export43$v $R/eval43$v/om > $L/om43$v.log 2>&1
  echo "OM43$v $(tr -d '\n ' < $R/eval43$v/om/om_check.json | cut -c1-400)" >> $W
done
python3 scripts/check_om.py $R/export42 $R/eval43_incumbent/om > $L/om42.log 2>&1; echo "OM42(incumbent) $(tr -d '\n ' < $R/eval43_incumbent/om/om_check.json | cut -c1-400)" >> $W
for v in a b; do
  [ -d $R/export43$v ] || continue
  python3 scripts/evaluate.py --export $R/export43$v --out $R/eval43$v > $L/eval43$v.log 2>&1
  cp docs/metrics.json $R/var/cand43$v.json 2>/dev/null; echo "EVAL43${v}_DONE $(date)" >> $W
  python3 scripts/gate41.py docs/metrics.json $R/incumbent_metrics.json $R/export43$v . > $L/gate43$v.log 2>&1
  echo "GATE43$v $(tail -1 $L/gate43$v.log | cut -c1-240)" >> $W
  grep -q PROMOTE $L/gate43$v.log && { echo "PROMOTED43$v" >> $W; cp docs/metrics.json $R/incumbent_metrics.json; mkdir -p docs/samples/om; cp $R/eval43$v/om/*.wav $R/eval43$v/om/om_check.json docs/samples/om/; break; }
done
echo "WAVE43_DONE $(date)" >> $W
