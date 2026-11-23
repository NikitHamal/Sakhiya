#!/bin/bash
L=/work/tts/logs; mkdir -p /work/sp/models /work/sp/data; cd /work/sp
export HF_TOKEN=$(cat /tmp/hf_token)
python3 - > $L/dl2.log 2>&1 <<'PY'
import os
from huggingface_hub import snapshot_download, hf_hub_download
tok=os.environ['HF_TOKEN']
for rid,ign in [('k2-fsa/OmniVoice',None)]:
    d=snapshot_download(rid,local_dir='/work/sp/models/'+rid.split('/')[1],token=tok); print('OK',rid,d,flush=True)
p=hf_hub_download('NikitHamal/sakhiya-v1-train','v13/real_audio_24k.pt',local_dir='/work/sp/data',token=tok); print('OK real',p,flush=True)
d=snapshot_download('NikitHamal/sakhiya-nepali-vedic-private',repo_type='dataset',local_dir='/work/sp/data/vedic',token=tok,allow_patterns=['meta/*','README.md']); print('OK vedic meta',flush=True)
PY
echo dl2 done >> $L/setup.log
