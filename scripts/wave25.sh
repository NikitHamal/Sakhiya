#!/bin/bash
# Wave 25: waves 22 and 23 (AM continuation + vocoder) both REVERTED; AM mel validation is flat (0.2996 at step 3000 -> 0.2965 at 9000), i.e. the AM is saturated at this size.
# So this wave keeps the *shipped* acoustic model (Wave 18a weights, 1 min at lr 1e-6 only to dump its own predicted mels) and spends the GPU on the real mismatch: GTA vocoder fine-tune from the shipped vocoder on the shipped AM's predicted mels.
R=/work/tts; L=$R/logs; cd $R/repo; W=$L/wave25.log; export ROMAN_ORT=$R/node/package.json
GTA=${GTA_MIN:-15}; echo "WAVE25_START $(date)" >> $W
G=$R/ckpt25; mkdir -p $G; cp ckpt/am.fp16.pt $G/am0.pt; cp ckpt/G.fp16.pt $G/G0.pt
rm -rf $R/best25; mkdir -p $R/best25; cp -r docs/models $R/best25/models; cp -r docs/samples $R/best25/samples; cp docs/metrics.json $R/best25/metrics.json
mkdir -p $R/runs/am25 $R/runs/voc25 $R/runs/am25; cp $R/runs/am24/last.pt $R/runs/am25/last.pt
python3 - > $L/blend25.log 2>&1 <<'PY'
import torch
R = "/work/tts"; w = float(__import__("os").environ.get("BLEND_W", "0.2"))
a = torch.load(f"{R}/ckpt24/G0.pt", weights_only=False, map_location="cpu"); b = torch.load(f"{R}/runs/voc24/G.pt", weights_only=False, map_location="cpu")
b = b["G"] if isinstance(b, dict) and "G" in b else b
out = {k: ((1 - w) * a[k].float() + w * b[k].float()) if a[k].is_floating_point() else a[k] for k in a}
torch.save(out, f"{R}/runs/voc25/G.pt"); print("blended", len(out), "w", w)
PY
echo "BLEND_DONE $(date)" >> $W
python3 scripts/export_onnx.py --am $R/runs/am25/last.pt --voc $R/runs/voc25/G.pt --out $R/export25 > $L/export25.log 2>&1 || { echo EXPORT_FAIL >> $W; exit 1; }
cp $R/export25/*.onnx $R/export25/config.json $R/export25/size.json docs/models/
git checkout -- docs/samples 2>/dev/null; python3 scripts/evaluate.py --export $R/export25 --out $R/eval25 > $L/eval25.log 2>&1
export TAG=25
python3 - <<'PY' > $L/promote$TAG.log 2>&1
import json, os, shutil, torch
R = "/work/tts"; T = os.environ.get("TAG", "18")
g = lambda p, k="sentence": next(s for s in json.load(open(p))["systems"] if s["system"] == "Sakhiya-fp16")[k]
new, old = g(f"{R}/repo/docs/metrics.json"), g(f"{R}/best{T}/metrics.json")
better = (new["mos"] >= old["mos"] - 0.005 and new["pcer"] <= old["pcer"] + 0.002) and (new["mos"] > old["mos"] or new["pcer"] < old["pcer"])
print("new", new["pcer"], new["mos"], "old", old["pcer"], old["mos"], "PROMOTE" if better else "REVERT", flush=True)
if better:
    os.makedirs(f"{R}/repo/ckpt", exist_ok=True); c = torch.load(f"{R}/runs/am{T}/last.pt", weights_only=False)
    c["M"] = {k: v.half() for k, v in c["M"].items()}; torch.save(c, f"{R}/repo/ckpt/am.fp16.pt")
    torch.save({k: v.half() for k, v in torch.load(f"{R}/runs/voc{T}/G.pt", weights_only=False).items()}, f"{R}/repo/ckpt/G.fp16.pt")
    shutil.copy(f"{R}/repo/docs/metrics.json", f"{R}/repo/eval/metrics.json")
else:
    for d in ("models", "samples"): shutil.rmtree(f"{R}/repo/docs/{d}"); shutil.copytree(f"{R}/best{T}/{d}", f"{R}/repo/docs/{d}")
    for d in ("docs", "eval"): shutil.copy(f"{R}/best{T}/metrics.json", f"{R}/repo/{d}/metrics.json")
PY
cat $L/promote$TAG.log >> $L/wave$TAG.log
if grep -q PROMOTE $L/promote$TAG.log; then
    export GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0
    git checkout -- docs/data/status.json 2>/dev/null || true
    git add docs/models docs/samples docs/metrics.json eval/metrics.json ckpt
    git commit -qm "Promote Wave $TAG: $(cat $L/promote$TAG.log)"
    git pull -q --rebase --autostash -X ours origin main
    git push -q origin HEAD:main
fi
echo "WAVE25_DONE $(date)" >> $W
