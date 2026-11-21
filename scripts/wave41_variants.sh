#!/bin/bash
# Wave 41 ablation: which half of the candidate hurt? Same held-out evaluator for every (AM, vocoder) pair; numbers land in /work/tts/var/<name>.json.
R=/work/tts; L=$R/logs; cd $R/repo41; mkdir -p $R/var
python3 - <<'PY'
import torch
R = "/work/tts"; a = torch.load(f"{R}/runs/am41/last.pt", map_location="cpu", weights_only=False); b = torch.load(f"{R}/runs/am40_best/last.pt", map_location="cpu", weights_only=False)
for al in (0.5,):
    out = {k: (v if k in ("spk.weight", "lang.weight") or not v.is_floating_point() or k not in b["M"] or b["M"][k].shape != v.shape else (1 - al) * v.float() + al * b["M"][k].float()) for k, v in a["M"].items()}
    c = dict(a); c["M"] = out; torch.save(c, f"{R}/runs/am41_wise.pt")
PY
run() { # name am voc
  SAKHIYA_ONE_VOICE=1 python3 scripts/export_onnx.py --am $2 --voc $3 --out $R/exp_$1 > $L/exp_$1.log 2>&1 || { echo "$1 EXPORT_FAIL" >> $L/variants.log; return; }
  python3 scripts/evaluate.py --export $R/exp_$1 --out $R/eval_$1 > $L/eval_$1.log 2>&1; cp docs/metrics.json $R/var/$1.json
  git checkout -- docs eval 2>/dev/null; echo "$1 DONE $(date +%H:%M:%S)" >> $L/variants.log; }
run am41_voc40 $R/runs/am41/last.pt $R/runs/voc40_best/G.pt
run am41_voc41a $R/runs/am41/last.pt $R/runs/voc41a/G.pt
run wise_voc40 $R/runs/am41_wise.pt $R/runs/voc40_best/G.pt
echo "VARIANTS_DONE" >> $L/variants.log
