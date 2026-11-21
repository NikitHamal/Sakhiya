#!/bin/bash
# v12 (mining box): freeze the GPU-heavy mining workers -> cleaned-vocoder training in 10-min chunks, each exported + gated vs
# shipped V1 (stops on its own when a chunk is worse) -> ship decision + A/B samples -> resume mining (always, via trap).
cd /work/tts/repo11 || exit 1
L=/work/tts/logs; V=/work/tts/v12; mkdir -p $V
python3 scripts/v12_pause.py stop
trap 'python3 /work/tts/repo11/scripts/v12_pause.py cont; echo "[v12] mining resumed $(date -u +%H:%M)Z"' EXIT
nohup python3 scripts/hf_backup.py --dirs $V $L --prefix v12 --every 300 > $L/v12_backup.log 2>&1 &
BK=$!
echo "[v12] training started $(date -u +%H:%M)Z"
python3 scripts/v12_voc_clean.py --out $V > $L/v12_voc.log 2>&1
echo "[v12] training rc=$? $(date -u +%H:%M)Z $(grep DONE $V/train.log | tail -1)"
export GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0
git pull -q --rebase origin main
python3 scripts/v12_ship_voc.py 2>&1 | grep -E "DECISION|SHIPPED|NOT SHIPPED|samples|Error" | cut -c1-1500
kill $BK 2>/dev/null; timeout 240 python3 scripts/hf_backup.py --dirs $V --prefix v12 --every 100000 > $L/v12_backup_final.log 2>&1
echo "[v12] CHAIN DONE $(date -u +%H:%M)Z"
