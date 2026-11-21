#!/bin/bash
# keeps the Gemini transcriber alive and, every 20 min, folds results into gold_text/Vedic gate and republishes the Clip Library (no secrets here).
cd /work/tts/repo || exit 1; export GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0; last=0
while true; do
  if ! python3 - <<'PY'
import psutil, sys
sys.exit(0 if any('scripts/gemini_transcribe.py' in ' '.join(p.info['cmdline'] or []) for p in psutil.process_iter(['cmdline'])) else 1)
PY
  then (setsid nohup env N_TRANSCRIBE=4 N_FLASH=10 SESSION_SPACING=3.0 python scripts/gemini_transcribe.py >> /work/tts/logs/gemini.log 2>&1 < /dev/null &); fi
  if [ $(( $(date +%s) - last )) -gt 1200 ]; then
    python scripts/apply_corrections.py >> /work/tts/logs/gemini_apply.log 2>&1; python scripts/apply_gemini.py >> /work/tts/logs/gemini_apply.log 2>&1
    true; last=$(date +%s)
  fi
  sleep 60
done
