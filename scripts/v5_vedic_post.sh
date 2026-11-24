#!/bin/bash
# Wait for the isolated Vedic LoRA chain, then gate-evaluate adapter checkpoints (pure prompt, de-karkar chain, leak/loop gate).
# Writes /work/tts/v5_recovery/state/vedic_lora.json (picked up by the 15-min HF backup of v5_recovery/).
export HF_TOKEN=$(cat /tmp/hf_token) PYTHONPATH=/work/sp; V=/work/tts/v5_vedic; R=/work/tts/repo/scripts; S=/work/tts/v5_recovery/state
until grep -q "CHAIN_DONE\|PREP_FAIL\|TOK_FAIL" /work/tts/logs/v5_vedic_chain.log 2>/dev/null; do sleep 20; done
ADS=$(ls -d $V/exp/vl1/adapter-* 2>/dev/null | sort -t- -k2 -n)
LAST=$(echo "$ADS" | tail -1); MID=$(echo "$ADS" | awk 'NR==int((NR_TOT+1)/2)' NR_TOT=$(echo "$ADS" | wc -l))
for A in $(echo "$ADS" | awk -F- '$NF%800==0') $LAST; do
  T=vl1_$(basename $A | sed 's/adapter-//'); [ -f $V/eval/$T/summary.json ] && continue
  python3 $R/v5_vedic_adapter.py --adapter $A --tag $T --nref 3 --seeds 3 > /work/tts/logs/v5_vedic_eval_$T.log 2>&1
done
python3 - <<EOF
import json,glob,os
out={"base_frozen":json.load(open("$V/exp/vl1/train_report.json")) if os.path.exists("$V/exp/vl1/train_report.json") else None,"evals":{}}
for f in sorted(glob.glob("$V/eval/*/summary.json")):
    s=json.load(open(f)); out["evals"][s["tag"]]={k:s[k] for k in ("passed","n","per_chant_pass","chants_with_a_clean_take")}
json.dump(out,open("$S/vedic_lora.json","w"),ensure_ascii=False,indent=1); print(json.dumps(out,ensure_ascii=False))
EOF
python3 $R/v5_vedic_state.py > /dev/null 2>&1
echo POST_DONE
