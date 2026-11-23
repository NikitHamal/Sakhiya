#!/bin/bash
# Metric assets for Sakhiya-Pro: Kriti Nepali CTC ASR (Aarjan/swar), UTMOS22 (SpeechMOS), ECAPA (speechbrain).
L=/work/tts/logs; cd /work/sp; export HF_TOKEN=$(cat /tmp/hf_token)
python3 - > $L/dl_swar.log 2>&1 <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download('Aarjan/swar', local_dir='/work/sp/models/swar', token=os.environ['HF_TOKEN'], allow_patterns=['app/**', 'kriti_onnx/**', 'README.md'])
print('SWAR_OK')
PY
[ -d SpeechMOS ] || git clone -q https://github.com/tarepan/SpeechMOS.git SpeechMOS > $L/dl_speechmos.log 2>&1
pip install -q speechbrain onnxruntime-gpu > $L/pip_sb.log 2>&1; echo "metrics done $?" >> $L/setup.log
