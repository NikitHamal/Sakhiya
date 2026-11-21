#!/bin/bash
# RSI driver (session 31): vag.pt -> own.pt -> AM (warm start, resumable) -> GTA mels -> vocoder GTA fine-tune -> export -> eval -> gate -> push.
# Every stage is resumable (runs/am30/last.pt, runs/voc31/G.pt); each wave is time-boxed so results land inside the session.
R=/work/tts; L=$R/logs; cd $R/repo
until [ -f $R/feats/vag.pt ] && grep -q VAG_DONE $L/vag.log 2>/dev/null; do sleep 5; done
rm -f $R/feats/own.pt; python3 scripts/build_own.py --sa_rep ${SA_REP:-3} --vag_rep ${VAG_REP:-2} --ne_rep ${NE_REP:-1} > $L/build_own.log 2>&1; echo "RSI_START $(date)" >> $L/wave30.log
BS=${BS:-48} AM_MIN=${AM_MIN:-22} LR=${LR:-4e-4} bash scripts/wave30.sh
VOC_MIN=${VOC_MIN:-14} bash scripts/wave31.sh
echo "RSI_DONE $(date)" >> $L/wave30.log
