#!/bin/bash
# Fresh-box bootstrap for Sakhiya-Pro v2 (Session 67). Launch detached:
#   setsid nohup bash /work/sp/repo/scripts/sakhiya_pro/bootstrap_v2.sh > /work/tts/logs/bootstrap_v2.log 2>&1 < /dev/null &
# Restores the v1 teacher + anchor + ft meta from HF, installs deps, downloads OmniVoice, SLR43 / SLR54 / v13 meta, then
# (optionally) chains pipeline_v2.sh.  Tokens only from /tmp/hf_token and /tmp/gitenv.sh on the box.
L=/work/tts/logs; mkdir -p $L /work/sp/stage /work/sp/final /work/sp/ref /work/sp/ft /work/tts/data
cd /work/sp
export HF_TOKEN=$(cat /tmp/hf_token) HUGGING_FACE_HUB_TOKEN=$(cat /tmp/hf_token) GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0
S=/work/sp/repo/scripts
st(){ echo "{\"label\":\"$2\",\"state\":\"$3\",\"note\":\"$4\"}" > /work/sp/stage/$1.json; }
st 00_boot "Bootstrap box (deps, teacher, corpora)" run "starting"
# 1) python deps (parallel with downloads)
( pip install -q onnx onnxscript librosa soundfile jiwer praat-parselmouth pyloudnorm pyworld soxr transformers accelerate peft hf_transfer scikit-learn speechmos > $L/pip1.log 2>&1; echo pip1 $? >> $L/setup.log
  pip install -q omnivoice > $L/pip_omni.log 2>&1; echo omni $? >> $L/setup.log ) &
# 2) OmniVoice sources (training CLI + token extractor) + base model
[ -d OmniVoice_src ] || git clone -q https://github.com/k2-fsa/OmniVoice.git OmniVoice_src > $L/clone_omni.log 2>&1
# 3) v1 teacher / anchor / ft meta from our HF backup (prefix sakhiya_pro_v2/)
python3 - > $L/restore_v1.log 2>&1 <<'PY'
import os, shutil
from huggingface_hub import snapshot_download
tok = os.environ['HF_TOKEN']
d = snapshot_download('NikitHamal/sakhiya-v1-train', allow_patterns=['sakhiya_pro_v2/final/sakhiya_pro_v1/*', 'sakhiya_pro_v2/ref/*', 'sakhiya_pro_v2/ft/*'], local_dir='/work/sp/hfpull', token=tok)
b = '/work/sp/hfpull/sakhiya_pro_v2'
for s, t in (('final/sakhiya_pro_v1', '/work/sp/final/sakhiya_pro_v1'), ('ref', '/work/sp/ref'), ('ft', '/work/sp/ft')):
    shutil.copytree(f'{b}/{s}', t, dirs_exist_ok=True)
print('RESTORE_OK')
PY
st 00_boot "Bootstrap box (deps, teacher, corpora)" run "teacher restored: $(tail -1 $L/restore_v1.log)"
# 4) base OmniVoice + v13 audio/meta + SLR43 (existing script) in background; SLR54 in parallel
bash $S/sakhiya_pro/dl_assets.sh > $L/dl_assets.log 2>&1 &
ln -sfn /work/tts/data/slr54 /work/sp/data/slr54 2>/dev/null || { mkdir -p /work/sp/data; ln -sfn /work/tts/data/slr54 /work/sp/data/slr54; }
python3 $S/dl_slr54.py > $L/dl54.log 2>&1 &
wait
mkdir -p /work/sp/data/v13
# v13 meta may live in the private dataset meta/ dir
for c in /work/sp/data/vedic/meta /work/sp/data/vedic/meta/v13; do [ -f $c/gold_text.json ] && ln -sfn $c /work/sp/data/v13/meta; done
ls /work/sp/data/v13/meta > $L/v13_meta_ls.log 2>&1
st 00_boot "Bootstrap box (deps, teacher, corpora)" done "BOOT_DONE $(date +%T)"
echo "BOOT_DONE $(date +%T)"
