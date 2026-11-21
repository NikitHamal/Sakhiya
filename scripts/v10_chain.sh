#!/bin/bash
# v10 chain (each stage capped < 2 h, nothing runs blind):
#   wait for the v9 acoustic runs -> v10 shared vocoder distillation (85 min cap, step-50 gate inside)
#   -> wait for the v9 ship chain -> export nanov10/atomv10 -> real forensic gate on the INT8 graphs -> A/B wavs
#   -> ship each tier only if not worse than what is shipped AND smaller -> push.
export GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0
cd /work/tts/repo9 || exit 1
L=/work/tts/logs; V=/work/tts/v10; VOC=/work/tts/runs/v10_voc144/voc.pt; mkdir -p $V
while pgrep -f "scripts/v9_am_cached.py" >/dev/null; do sleep 20; done
echo "[v10] acoustic runs finished $(date -u +%H:%M)Z, starting vocoder distillation"
python3 scripts/v10_voc_distill.py --ch 144 --out /work/tts/runs/v10_voc144 --max-minutes 85 --gan_after 8000 --eval_every 2000 > $L/v10_voc.log 2>&1
echo "[v10] vocoder rc=$? $(date -u +%H:%M)Z"; grep -E "GATE|TEACHER|DONE" $L/v10_voc.log | tail -3
[ -f $VOC ] || { echo "[v10] no vocoder checkpoint -> stop"; exit 1; }
while pgrep -f "v9_chain.sh" >/dev/null; do sleep 20; done
echo "[v10] v9 chain finished $(date -u +%H:%M)Z, exporting"
git stash -q 2>/dev/null; git pull -q --rebase origin main; git stash pop -q 2>/dev/null
for t in nano atom; do
  if [ $t = nano ]; then EXP=nanov10; AM=/work/tts/runs/v9_voca_am/am.pt; else EXP=atomv10; AM=/work/tts/runs/v9_sano_am/am.pt; fi
  rm -rf $V/stage_$t
  python3 scripts/export_v2_onnx.py --tier $EXP --am $AM --voc $VOC --out $V/stage_$t 2>&1 | tail -2
  python3 scripts/forensic_audio_gate.py --model $V/stage_$t --tiny --limit 30 --out $V/gate_$t.json 2>&1 | tail -1
done
python3 scripts/v10_ab.py 2>&1 | grep AB
python3 scripts/v10_ship.py --tier nano --stage $V/stage_nano --gate $V/gate_nano.json --tag v10_nano --label "Sakhiya-V1-Nano (v10)" --desc "Small model · shared distilled vocoder · clean voice" 2>&1 | grep -E "DECISION|SHIPPED"
python3 scripts/v10_ship.py --tier atom --stage $V/stage_atom --gate $V/gate_atom.json --tag v10_atom --label "Sakhiya-V1-Atom (v10)" --desc "Smallest model (under 3 MB) · shared distilled vocoder · clean voice" 2>&1 | grep -E "DECISION|SHIPPED"
echo "[v10] CHAIN DONE $(date -u +%H:%M)Z"
