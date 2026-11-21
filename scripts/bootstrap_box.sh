#!/bin/bash
# Fresh MoLab box -> running pipeline. The sandbox is ephemeral (it was reset twice during session 19/20): everything is resumable and re-runnable.
# Needs /tmp/gitenv.sh (git askpass helper, never committed).  usage: bash bootstrap_box.sh [wave script, default wave19.sh]
R=/work/tts; L=$R/logs; mkdir -p $L $R/runs $R/feats; cd $R
export GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0
[ -d repo ] || git clone -q https://NikitHamal@github.com/NikitHamal/Sakhiya.git repo
cd repo && git config user.name NikitHamal && git config user.email nikithamal@users.noreply.github.com && git pull -q --rebase origin main
[ -f $L/deps.done ] || { pip install -q pyworld jiwer librosa soundfile praat-parselmouth speechmos onnx onnxscript piper-tts scikit-learn edge-tts pyarrow > $L/pip.log 2>&1; touch $L/deps.done; } &
[ -d $R/node/node_modules ] || npm i --silent --prefix $R/node onnxruntime-node > $L/node.log 2>&1 &
wait
[ -f $L/dl.done ] || { python3 scripts/cache_data.py pull > $L/cache_pull.log 2>&1 && python3 -c '
import os, torch
F = "/work/tts/feats"; mix = torch.load(f"{F}/mix.pt", weights_only=False)
if not os.path.exists(f"{F}/v2.pt"): torch.save([r for r in mix if r.get("src") in ("slr43", "fleurs", "cv")], f"{F}/v2.pt")
if not os.path.exists(f"{F}/sa.pt"): torch.save([r for r in mix if r.get("lang") == "sa" or r.get("src") == "shruti"], f"{F}/sa.pt")
if not os.path.exists(f"{F}/edge.pt"): torch.save([r for r in mix if r["spk"] in ("edge_f", "edge_m")], f"{F}/edge.pt")
' && touch $L/cache_pulled $L/dl.done && echo SLR54DL_DONE > $L/slr54dl.log; }
[ -f $L/dl.done ] || { python3 scripts/dl_all.py > $L/dl.log 2>&1 && touch $L/dl.done; } &
grep -q SLR54DL_DONE $L/slr54dl.log 2>/dev/null || python3 scripts/dl_slr54.py > $L/slr54dl.log 2>&1 &
wait
python3 scripts/relay.py pull > $L/relay_pull.log 2>&1 || true
nohup python3 scripts/relay.py watch > $L/relay.log 2>&1 &
nohup python3 scripts/status_daemon.py > $L/daemon.log 2>&1 &
WAVE=${1:-wave20.sh}; W=${WAVE%.sh}; T=${W#wave}
nohup bash scripts/$WAVE > $L/$W.out 2>&1 &
nohup python3 scripts/finalize_wave.py $T $T > $L/finalize$T.out 2>&1 &
echo BOOT_DONE
