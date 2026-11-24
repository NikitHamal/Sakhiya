#!/bin/bash
# v5 Vedic-only LoRA chain: prep (strict gates) -> tokenize -> isolated LoRA on frozen v3b-600. Detached; logs to /work/tts/logs.
export HF_TOKEN=$(cat /tmp/hf_token); R=/work/tts/repo/scripts; V=/work/tts/v5_vedic; L=/work/tts/logs
BASE=/work/tts/v5_recovery/hf/sakhiya_pro_v3/exp/v3b/latest   # v3b step-600 = the frozen v5 Nepali master weights
MIN=${1:-40}
set -x; cd $V
python3 $R/v5_vedic_prep.py || { echo PREP_FAIL; exit 1; }
rm -rf $V/tokens
for s in train dev; do
  python3 -m omnivoice.scripts.extract_audio_tokens --input_jsonl $V/ft/$s.jsonl \
    --tar_output_pattern "$V/tokens/$s/audios/shard-%06d.tar" --jsonl_output_pattern "$V/tokens/$s/txts/shard-%06d.jsonl" \
    --tokenizer_path bosonai/higgs-audio-v2-tokenizer --nj_per_gpu 3 --shuffle True || { echo TOK_FAIL $s; exit 2; }
done
python3 - <<EOF
import json
json.dump(dict(train=[dict(manifest_path=["$V/tokens/train/data.lst"])], dev=[dict(manifest_path=["$V/tokens/dev/data.lst"])]), open("$V/data_cfg.json","w"))
EOF
python3 $R/v5_vedic_lora_train.py --name vl1 --base $BASE --data_cfg $V/data_cfg.json --minutes $MIN
echo "CHAIN_DONE rc=$?"
