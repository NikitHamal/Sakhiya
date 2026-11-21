#!/bin/bash
# box 3 (fresh MoLab sandbox; box 1 hit the 12 h limit and lost its un-backed-up runs at ~20:42Z).
# Safe to re-run: running acoustic jobs are not started twice.
# Rebuild the Nano/Atom acoustic students + the shared ~1.2 M-param vocoder that takes Atom under 3 MB. Every stage is capped
# (< 2 h) with the step-50 throughput gate inside each script, and every checkpoint/log is mirrored to the private HF repo.
#   A (parallel): v9 Nano AM (100 min) | v9 Atom AM (100 min, starts after the shared teacher cache) | v10 vocoder (85 min, on GT
#                 mels + v7-student mels as stand-ins)
#   B: vocoder adaptation on the NEW v9 student mels (25 min, warm start from A)
#   C: Nano v9 (+V1 vocoder) ship check; then nanov10/atomv10 export -> int8 forensic gate -> ship only if not worse AND smaller.
R=/work/tts; L=$R/logs; cd $R/repo9 || exit 1
export GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0
mkdir -p $R/runs/v9_voca_am $R/runs/v9_sano_am $R/runs/v10_voc144 $R/runs/v10_voc144_adapt $R/v10 $R/v9
pgrep -f "[h]f_backup.py" >/dev/null || nohup python3 $R/hf_backup.py --dirs $R/runs/v9_voca_am $R/runs/v9_sano_am $R/runs/v10_voc144 $R/runs/v10_voc144_adapt $R/v10 $L --prefix box3 > $L/hf_backup.log 2>&1 &
TEACH=$R/runs/v6_teacher/am.fp16.pt
pgrep -f "[v]9_am_cached.py --tier voca" >/dev/null || nohup python3 scripts/v9_am_cached.py --tier voca --max-minutes 100 --init $R/runs/v7/voca_am.pt --teacher_am $TEACH > $L/v9_voca.log 2>&1 &
echo "[box3] Nano AM started $(date -u +%H:%M)Z"
for i in $(seq 1 120); do grep -q "\[cache\] built\|\[cache\] loaded" $L/v9_voca.log 2>/dev/null && break; pgrep -f "[v]9_am_cached.py --tier voca" >/dev/null || { echo "[box3] Nano AM died before the cache"; tail -5 $L/v9_voca.log; break; }; sleep 15; done
pgrep -f "[v]9_am_cached.py --tier sano" >/dev/null || nohup python3 scripts/v9_am_cached.py --tier sano --max-minutes 100 --init $R/runs/v7/sano_am.pt --teacher_am $TEACH > $L/v9_sano.log 2>&1 &
echo "[box3] Atom AM started $(date -u +%H:%M)Z"
python3 scripts/v10_voc_distill.py --ch 144 --out $R/runs/v10_voc144 --max-minutes 85 --gan_after 8000 --eval_every 2000 --am_voca $R/runs/v7/voca_am.pt --am_sano $R/runs/v7/sano_am.pt > $L/v10_voc.log 2>&1
echo "[box3] vocoder A rc=$? $(date -u +%H:%M)Z $(grep -E 'DONE' $L/v10_voc.log | tail -1)"
while pgrep -f "[v]9_am_cached.py" >/dev/null; do sleep 20; done
echo "[box3] AMs finished $(date -u +%H:%M)Z"
VOC=$R/runs/v10_voc144/voc.pt
if [ -f $VOC ] && [ -f $R/runs/v9_voca_am/am.pt ] && [ -f $R/runs/v9_sano_am/am.pt ]; then
  python3 scripts/v10_voc_distill.py --ch 144 --init $VOC --out $R/runs/v10_voc144_adapt --max-minutes 25 --gan_after 300 --lr 2e-4 --eval_every 1000 > $L/v10_voc_adapt.log 2>&1
  echo "[box3] vocoder B (adapt) rc=$? $(grep -E 'DONE' $L/v10_voc_adapt.log | tail -1)"
  [ -f $R/runs/v10_voc144_adapt/voc.pt ] && VOC=$R/runs/v10_voc144_adapt/voc.pt
fi
git stash -q 2>/dev/null; git pull -q --rebase origin main; git stash pop -q 2>/dev/null
python3 scripts/v9_ship.py --tier voca --am $R/runs/v9_voca_am/am.pt --voc $R/runs/v5_c1/voc.pt --tag v9_nano --label "Sakhiya-V1-Nano (v9)" --desc "Small model · fully trained student + V1 vocoder · clean voice" 2>&1 | grep -E "NEW score|SHIPPED|NOT SHIPPED"
[ -f $VOC ] || { echo "[box3] no vocoder checkpoint -> stop"; exit 1; }
for t in nano atom; do
  if [ $t = nano ]; then EXP=nanov10; AM=$R/runs/v9_voca_am/am.pt; else EXP=atomv10; AM=$R/runs/v9_sano_am/am.pt; fi
  rm -rf $R/v10/stage_$t
  python3 scripts/export_v2_onnx.py --tier $EXP --am $AM --voc $VOC --out $R/v10/stage_$t > $L/v10_export_$t.log 2>&1; echo "[box3] export $t rc=$?"
  python3 scripts/forensic_audio_gate.py --model $R/v10/stage_$t --tiny --limit 30 --out $R/v10/gate_$t.json > $L/v10_gate_$t.log 2>&1; echo "[box3] gate $t rc=$?"
done
python3 scripts/v10_ab.py 2>&1 | grep AB
git stash -q 2>/dev/null; git pull -q --rebase origin main; git stash pop -q 2>/dev/null
python3 scripts/v10_ship.py --tier nano --stage $R/v10/stage_nano --gate $R/v10/gate_nano.json --tag v10_nano --label "Sakhiya-V1-Nano (v10)" --desc "Small model · shared distilled vocoder · clean voice" 2>&1 | grep -E "DECISION|SHIPPED"
python3 scripts/v10_ship.py --tier atom --stage $R/v10/stage_atom --gate $R/v10/gate_atom.json --tag v10_atom --label "Sakhiya-V1-Atom (v10)" --desc "Smallest model (under 3 MB) · shared distilled vocoder · clean voice" 2>&1 | grep -E "DECISION|SHIPPED"
echo "[box3] CHAIN DONE $(date -u +%H:%M)Z"
