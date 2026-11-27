#!/bin/bash
# one-shot MoLab pipeline: fonts -> audio -> render -> release. Safe to re-run after a MoLab restart.
set -e; cd "$(dirname "$0")/.."
mkdir -p /work/fonts && (cd /work/fonts && [ -d node_modules/@fontsource/poppins ] || (npm init -y >/dev/null && npm i @fontsource/poppins @fontsource/noto-sans-devanagari >/dev/null 2>&1))
echo "[1/4] audio"; python3 video/gen_audio.py
echo "[2/4] render"; WIP_TAG=news-ep1-v6-wip WORKERS=${WORKERS:-4} python3 video/seg_render.py
echo "[3/4] release"; bash video/release.sh
echo "[4/4] done"
