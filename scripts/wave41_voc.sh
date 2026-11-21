#!/bin/bash
# Vocoder-only ablation on top of the PROMOTED acoustic model (ckpt/am.fp16.pt): does the real-mel GAN fine-tune (voc41c, no phase loss) or its
# average with the shipped vocoder raise UTMOS for BOTH languages?  Same held-out evaluator; results in /work/tts/var/<name>.json.
R=/work/tts; L=$R/logs; cd $R/repo41; mkdir -p $R/var
while pgrep -f "[t]rain_vocoder.py --set own_ms --minutes 14" > /dev/null; do sleep 5; done
git checkout -q -- . 2>/dev/null; export GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0; git pull -q --rebase origin main
python3 - <<'PY'
import torch
R = "/work/tts"; g0 = torch.load("ckpt/G.fp16.pt", map_location="cpu", weights_only=False); g1 = torch.load(f"{R}/runs/voc41c/G.pt", map_location="cpu", weights_only=False)
g0 = g0["G"] if "G" in g0 else g0; g1 = g1["G"] if "G" in g1 else g1
torch.save({k: (0.5 * v.float() + 0.5 * g1[k].float()) if v.is_floating_point() and k in g1 else v for k, v in g0.items()}, f"{R}/runs/voc41c_avg.pt")
PY
run() {
  SAKHIYA_ONE_VOICE=1 python3 scripts/export_onnx.py --am ckpt/am.fp16.pt --voc $2 --out $R/exp_$1 > $L/exp_$1.log 2>&1 || { echo "$1 EXPORT_FAIL" >> $L/variants.log; return; }
  python3 scripts/evaluate.py --export $R/exp_$1 --out $R/eval_$1 > $L/eval_$1.log 2>&1; cp docs/metrics.json $R/var/$1.json
  git checkout -q -- docs eval 2>/dev/null; echo "$1 DONE $(date +%H:%M:%S)" >> $L/variants.log; }
run am42_voc41c $R/runs/voc41c/G.pt
run am42_voc41c_avg $R/runs/voc41c_avg.pt
echo "VOC_VARIANTS_DONE" >> $L/variants.log
