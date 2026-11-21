#!/bin/bash
# v9 chain: wait for both capped AM trainings -> export + real forensic gate -> ship ONLY if it beats the shipped v8 -> prune selector -> push.
export GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0
cd /work/tts/repo9 || exit 1
while pgrep -f "scripts/v9_am_cached.py" >/dev/null; do sleep 20; done
echo "[chain] training finished $(date -u +%H:%M)Z"
git checkout -q . ; git clean -fdq docs eval; git pull -q --rebase origin main
VOC=/work/tts/runs/v5_c1/voc.pt
python3 scripts/v9_ship.py --tier voca --am /work/tts/runs/v9_voca_am/am.pt --voc $VOC --tag v9_nano --label "Sakhiya-V1-Nano (v9)" --desc "Small model · fully trained student + V1 vocoder · clean voice"
python3 scripts/v9_ship.py --tier sano --am /work/tts/runs/v9_sano_am/am.pt --voc $VOC --tag v9_atom --label "Sakhiya-V1-Atom (v9)" --desc "Smallest model · fully trained student + V1 vocoder · clean voice"
python3 scripts/v9_registry.py
git add docs eval scripts STATUS.md; git commit -qm "v9: selector = V1 / Nano / Atom (latest shipped), STATUS"; git pull -q --rebase origin main; git push -q origin HEAD:main
mkdir -p /work/tts/v9; mkdir -p /tmp/ck; cp /work/tts/runs/v9_voca_am/am.pt /tmp/ck/v9_nano_am.pt; cp /work/tts/runs/v9_sano_am/am.pt /tmp/ck/v9_atom_am.pt
echo "[chain] CHAIN DONE $(date -u +%H:%M)Z"
