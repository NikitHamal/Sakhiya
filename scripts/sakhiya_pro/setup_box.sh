#!/bin/bash
L=/work/tts/logs; cd /work/sp
export HF_TOKEN=$(cat /tmp/hf_token) HUGGING_FACE_HUB_TOKEN=$(cat /tmp/hf_token) GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0
pip install -q onnx onnxscript librosa soundfile jiwer praat-parselmouth pyloudnorm pyworld soxr transformers accelerate peft huggingface_hub hf_transfer scikit-learn speechmos > $L/pip1.log 2>&1; echo pip1 done $? >> $L/setup.log
pip install -q f5-tts > $L/pip_f5.log 2>&1; echo f5 done $? >> $L/setup.log
pip install -q omnivoice > $L/pip_omni.log 2>&1; echo omni done $? >> $L/setup.log
[ -d repo ] || git clone -q https://NikitHamal@github.com/NikitHamal/Sakhiya.git repo
echo clone done >> $L/setup.log
