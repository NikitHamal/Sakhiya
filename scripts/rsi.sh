#!/bin/bash
# Recursive self-improvement driver (RSI). One wave = train (resume) -> export -> evaluate on held-out REAL clips -> gate -> push.
# Every wave resumes from runs/am30/last.pt, so training compounds across waves and across sessions; the gate in
# wave30_post.sh only promotes a candidate that is intelligible (Nepali pCER <= 0.07, own Sanskrit pCER <= 0.22), and a
# candidate that is worse than the shipped model is reverted, with its numbers kept in eval/candidate_wave30.json.
# usage (detached, survives the notebook kernel):  setsid nohup bash scripts/rsi.sh > /work/tts/logs/rsi.out 2>&1 &
# stop after the current wave:  touch /work/tts/STOP_RSI
R=/work/tts; L=$R/logs; cd $R/repo || exit 1
WAVES=${WAVES:-6}; MIN=${AM_MIN:-24}
for i in $(seq 1 "$WAVES"); do
  [ -f $R/STOP_RSI ] && { echo "RSI_STOPPED $(date)" >> $L/rsi.log; break; }
  echo "RSI_WAVE_START $i $(date)" >> $L/rsi.log
  GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0 git pull -q --rebase --autostash -X ours origin main 2>/dev/null || true
  rm -f $L/am30.log
  AM_MIN=$MIN bash scripts/wave30.sh >> $L/rsi.out 2>&1 || { echo "RSI_WAVE_FAIL $i $(date)" >> $L/rsi.log; sleep 30; continue; }
  bash scripts/wave30_post.sh >> $L/rsi.out 2>&1
  # keep a rolling, monotone checkpoint archive (never overwritten by a worse wave: the promote gate decides what ships)
  cp $R/runs/am30/last.pt $R/runs/am30/wave_$i.pt 2>/dev/null
  echo "RSI_WAVE_DONE $i $(date) $(tail -1 $L/promote30.log | cut -c1-200)" >> $L/rsi.log
done
echo "RSI_END $(date)" >> $L/rsi.log
