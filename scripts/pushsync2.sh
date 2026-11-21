#!/bin/bash
# Push-safe sync (second clone so it never races the relay daemon's pushsync.sh). Includes docs/models_v2.
set -e
export GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0
P=/work/tts/pushclone2
[ -d $P/.git ] || git clone -q https://NikitHamal@github.com/NikitHamal/Sakhiya.git $P
cd $P && git fetch -q origin && git checkout -q -B main origin/main
cd /work/tts/repo
for d in sakhiya scripts tests eval monitor docs; do tar --exclude=data/status.json --exclude=__pycache__ --exclude='docs/models' --exclude='*.pyc' -cf - $d | tar -xf - -C $P; done
cp STATUS.md LEDGER.json $P/ 2>/dev/null || true
cd $P && git config user.name NikitHamal && git config user.email nikithamal@users.noreply.github.com
git add -A && (git diff --cached --quiet && echo NOTHING || git commit -qm "${1:-v2 progress}")
for i in 1 2 3; do git pull -q --rebase origin main && git push -q origin HEAD:main && echo PUSHED && break || sleep 5; done
git log --oneline | head -2
