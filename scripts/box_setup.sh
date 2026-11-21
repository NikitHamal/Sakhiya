#!/bin/bash
# One-shot GPU-box bootstrap (the MoLab sandbox is ephemeral: re-run after any restart). Needs /tmp/gitenv.sh (git askpass helper).
# usage: bash box_setup.sh   -> clones repo, installs deps, downloads corpora, then runs the resumable pipeline.
R=/work/tts; L=$R/logs; mkdir -p $L $R/runs $R/feats; cd $R
export GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0
[ -d repo ] || git clone -q https://NikitHamal@github.com/NikitHamal/Sakhiya.git repo
cd repo && git config user.name NikitHamal && git config user.email nikithamal@users.noreply.github.com && git pull -q --rebase origin main
[ -f $L/deps.done ] || { pip install -q pyworld jiwer librosa soundfile praat-parselmouth speechmos onnx onnxscript piper-tts scikit-learn > $L/pip.log 2>&1; touch $L/deps.done; }
nohup python3 scripts/status_daemon.py > $L/daemon.log 2>&1 &
[ -f $L/dl.done ] || { python3 scripts/dl_all.py > $L/dl.log 2>&1 && touch $L/dl.done; }
AM_MIN=${AM_MIN:-35} VOC_MIN=${VOC_MIN:-35} GTA_MIN=${GTA_MIN:-15} bash scripts/pipeline.sh > $L/pipeline.log 2>&1
