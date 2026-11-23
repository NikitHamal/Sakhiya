#!/bin/bash
# Tokenize ft/{train,dev}.jsonl into OmniVoice WebDataset shards (Higgs audio tokenizer).
set -e; cd /work/sp/OmniVoice_src; export PYTHONPATH=$PWD:$PYTHONPATH HF_TOKEN=$(cat /tmp/hf_token)
T=/work/sp/ft/tokens; rm -rf $T
for s in train dev; do
  python -m omnivoice.scripts.extract_audio_tokens --input_jsonl /work/sp/ft/$s.jsonl \
    --tar_output_pattern "$T/$s/audios/shard-%06d.tar" --jsonl_output_pattern "$T/$s/txts/shard-%06d.jsonl" \
    --tokenizer_path bosonai/higgs-audio-v2-tokenizer --nj_per_gpu 3 --shuffle True
done
echo TOKENIZE_DONE
