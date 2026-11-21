#!/bin/bash
# Keeps the GPU busy after the first v7 jobs finish: 2 more 200-min vocoder rounds per tier (resumes voc.pt, lr restarts at 1.2e-4)
# and a second acoustic-student pass (steps 26k -> 52k). Waits on exact job patterns; never kills anything.
cd /work/tts/repo7 || exit 1
LOG=/work/tts/logs
for t in voca sano; do
  ( while pgrep -f "v6_voc_long.py --tier $t " >/dev/null; do sleep 30; done
    for r in 2 3; do
      python scripts/v6_voc_long.py --tier $t --minutes 200 --out /work/tts/runs/v7_${t}_voc --bs 48 --seg 64 >> $LOG/v7_voc_$t.log 2>&1
    done ) &
  ( while pgrep -f "v7_am_distill.py --tier $t " >/dev/null; do sleep 30; done
    python scripts/v7_am_distill.py --tier $t --steps 52000 >> $LOG/v7_am_$t.log 2>&1 ) &
done
wait
