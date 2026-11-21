#!/bin/bash
# usage: post_wave.sh TAG — waits for the wave's gate line, pushes models/samples/metrics/ckpt on PROMOTE, then marks WAVE<TAG>_DONE (finalize_wave.py waits for it).
R=/work/tts; L=$R/logs; T=$1; cd $R/repo; export GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0
while ! grep -qE "PROMOTE|REVERT|EXPORT_FAIL" $L/wave$T.log 2>/dev/null; do sleep 10; done
if grep -q "PROMOTE" $L/wave$T.log; then
  git checkout -- docs/data/status.json 2>/dev/null; git add -A docs eval ckpt scripts sakhiya monitor; git commit -qm "wave $T promoted: new models, samples and benchmark"; git pull -q --rebase -X ours origin main; git push -q origin HEAD:main
fi
echo "WAVE${T}_DONE $(date)" >> $L/wave$T.log
