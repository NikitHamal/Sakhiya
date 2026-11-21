#!/bin/bash
# v11 (training box): wait for the mining-box fine-tune checkpoints on the private HF repo (max 3 h) -> export V1 + candidates
# (tier v25 = V1's own graph, V1 vocoder) -> forensic gate on the INT8 graphs -> decide/ship (waits for the v10 chain first so
# the two never push at the same time).
cd /work/tts/repo9 || exit 1
L=/work/tts/logs; V=/work/tts/v11; mkdir -p $V
for i in $(seq 1 180); do
  python3 -c "from huggingface_hub import HfApi;import sys;sys.exit(0 if 'v11/READY' in HfApi(token=open('/tmp/hf_token').read().strip()).list_repo_files('NikitHamal/sakhiya-v1-train') else 1)" 2>/dev/null && break
  sleep 60
done
python3 -c "from huggingface_hub import snapshot_download as s;s('NikitHamal/sakhiya-v1-train',allow_patterns=['v11/*'],local_dir='/work/tts/v11/hf',token=open('/tmp/hf_token').read().strip())" || { echo "[v11] no checkpoints"; exit 1; }
echo "[v11] checkpoints downloaded $(date -u +%H:%M)Z"; cat $V/hf/v11/summary.json 2>/dev/null | head -c 400; echo
for k in v1 best last; do
  if [ $k = v1 ]; then AM=/work/tts/runs/v5_c1/am.pt; else AM=$V/hf/v11/$k.pt; fi
  [ -f $AM ] || continue
  rm -rf $V/stage_$k
  python3 scripts/export_v2_onnx.py --tier v25 --am $AM --voc /work/tts/runs/v5_c1/voc.pt --out $V/stage_$k 2>&1 | tail -1 | cut -c1-200
  python3 scripts/forensic_audio_gate.py --model $V/stage_$k --tiny --limit 30 --out $V/gate_$k.json 2>&1 | tail -1
done
while pgrep -f "[v]10_chain.sh" >/dev/null; do sleep 30; done
git stash -q 2>/dev/null; git pull -q --rebase origin main; git stash pop -q 2>/dev/null
python3 scripts/v11_ship_v1.py 2>&1 | grep -E "DECISION|SHIPPED|NOT SHIPPED"
echo "[v11] CHAIN DONE $(date -u +%H:%M)Z"
