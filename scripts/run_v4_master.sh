#!/bin/bash
set -e
cd /work/tts/repo
export GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0
python3 -c 'import json,time; json.dump({"stage_idx":0,"right_now":"Phase 1: Extracting & CTC-aligning 24kHz Nepali+Sanskrit studio speech","updated_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}, open("/work/tts/COORD_V3.json","w")); json.dump({"t":time.time()}, open("/work/tts/heartbeat.json","w"))'
if [ ! -f /work/tts/feats/v2_studio_24k.pt ]; then
  NW=16 MAX_SHARDS_PER_LANG=18 python3 -u scripts/v2_prep_rasa.py /work/tts/data/rasa /work/tts/feats/v2_studio_24k.pt > /work/tts/logs/prep_rasa.log 2>&1
fi
if [ ! -f /work/tts/feats/v3_24k/v3_studio_aligned_24k.pt ]; then
  python3 -u scripts/v3_phase1_align.py > /work/tts/logs/phase1_align.log 2>&1
fi
if [ ! -f /work/tts/feats/v3_24k/v31_studio_aligned_24k.pt ]; then
  python3 -u scripts/run_v31_pipeline.py > /work/tts/logs/phase3_realign.log 2>&1
fi
python3 -u scripts/v2_voc_wavs.py 100 16 24 > /work/tts/logs/voc_wavs.log 2>&1
python3 -u scripts/v4_resume.py --fresh --m1_1a 1000 --m1_1b 600 --m2_steps 1000 --m3_steps 400 > /work/tts/logs/v4_resume.log 2>&1
