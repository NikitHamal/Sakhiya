#!/bin/bash
# Evaluate export41 on held-out REAL clips, gate against the incumbent (relative, so every promoted checkpoint is strictly better), save the candidate.
R=/work/tts; L=$R/logs; cd $R/repo41; W=$L/wave41.log
python3 scripts/evaluate.py --export $R/export41 --out $R/eval41 > $L/eval41.log 2>&1
echo "EVAL41_DONE $(date)" >> $W
python3 scripts/gate41.py $R/repo41/docs/metrics.json $R/incumbent_metrics.json $R/export41 $R/repo41 > $L/gate41.log 2>&1
echo "GATE41 $(tail -1 $L/gate41.log | cut -c1-240)" >> $W
