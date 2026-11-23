#!/bin/bash
# Launch the whole Pro v2 chain after bootstrap: v13 wavs -> 12 corpus workers -> pipeline_v2.sh (waits for workers, merges, trains, evals).
L=/work/tts/logs; cd /work/sp; export PYTHONPATH=/work/sp/OmniVoice_src:/work/sp HF_TOKEN=$(cat /tmp/hf_token)
P=/work/sp/repo/scripts/sakhiya_pro
python3 $P/build_s1.py v13wav > $L/v13wav.log 2>&1
for i in $(seq 0 ${NW:-11}); do setsid nohup python3 $P/build_s1.py work $i ${NW1:-12} > $L/s1_w$i.log 2>&1 < /dev/null & done
sleep 5
S1_DEADLINE=${S1_DEADLINE:-$(( $(date +%s) + 55*60 ))} bash $P/pipeline_v2.sh ${TARGET_H:-30} > $L/pipeline_v2.log 2>&1
