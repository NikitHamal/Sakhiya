#!/bin/bash
# keeps the Nepali cloud-ASR router alive; every 30 min folds all engines' words into gold_text and republishes the Clip Library (no secrets here).
cd /work/tts/repo || exit 1; export GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0; last=${FIRST_DELAY_START:-0}
while true; do
  if ! python3 - <<'PY'
import psutil, sys
sys.exit(0 if any('scripts/nepali_asr_router.py' in ' '.join(p.info['cmdline'] or []) for p in psutil.process_iter(['cmdline'])) else 1)
PY
  then (AAI_THREADS=${AAI_THREADS:-40} GROQ_THREADS=2 setsid nohup python3 scripts/nepali_asr_router.py >> /work/tts/logs/cloud_asr.log 2>&1 < /dev/null &); fi
  if [ $(( $(date +%s) - last )) -gt 1800 ]; then
    python3 scripts/apply_corrections.py >> /work/tts/logs/cloud_apply.log 2>&1; python3 scripts/apply_gemini.py >> /work/tts/logs/cloud_apply.log 2>&1
    python3 scripts/publish_clip_library.py --push >> /work/tts/logs/cloud_apply.log 2>&1
    last=$(date +%s)
  fi
  sleep 60
done
