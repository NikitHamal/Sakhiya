#!/bin/bash
# Keeps N GPU segmentation workers alive until mining is finished and every raw file is processed.
N=${1:-5}; R=/work/tts; cd $R/repo; rm -f $R/segments/*/*.claim
for i in $(seq 0 $((N-1))); do
  ( while true; do
      python3 scripts/yt_separate_segment.py --shard 0/1 >> $R/logs/seg_w$i.log 2>&1
      rm -f $R/segments/*/*.claim.stale
      if [ -f $R/logs/mining.finished ]; then python3 scripts/yt_separate_segment.py --shard 0/1 >> $R/logs/seg_w$i.log 2>&1; break; fi
      sleep 20
    done ) &
  sleep 3
done
wait; touch $R/logs/stage2.finished
