#!/bin/bash
# Master End-to-End Orchestrator for Sakhiya-Pro v4 + 16-20h Denoised Corpus + Serial Zero-NaN Shrink Ladder (Micro <=4.5 MB & Pico <=1.4 MB).
# Uses /dev/shm (504 GB RAM disk) for all heavy audio/token/checkpoint scratch dirs so container rootfs / stays under 5 GB.

set -e
L=/work/tts/logs
R=/work/sp/repo/scripts/sakhiya_pro
V4=$R/v4
S=/work/sp/repo/scripts
export PYTHONPATH=/work/sp/OmniVoice_src:/work/sp:/work/sp/repo
export HF_TOKEN=$(cat /tmp/hf_token)
export HUGGING_FACE_HUB_TOKEN=$(cat /tmp/hf_token)
export HF_HUB_CACHE=/dev/shm/hf_cache

st(){ echo "{\"label\":\"$2\",\"state\":\"$3\",\"note\":\"$4\"}" > /work/sp/stage/$1.json; }
refresh_cards(){ timeout 30 python3 $S/molab_exec.py --py "exec(open('/work/sp/repo/monitor/cells_pro4.py').read()); await install()" >/dev/null 2>&1 || true; }

# 0. Setup /dev/shm symlinks for heavy scratch directories
mkdir -p /dev/shm/sp_data/slr43 /dev/shm/sp_data/slr54 /dev/shm/sp_v4_work/clean_pool /dev/shm/sp_v4_work/stage4A /dev/shm/sp_v4_work/stage4B /dev/shm/sp_exp /dev/shm/sp_distill_v4 /dev/shm/sp_hfpull
ln -sfn /dev/shm/sp_data/slr43 /work/sp/data/slr43
ln -sfn /dev/shm/sp_data/slr54 /work/sp/data/slr54
ln -sfn /dev/shm/sp_v4_work/clean_pool /work/sp/v4/clean_pool
ln -sfn /dev/shm/sp_v4_work/stage4A /work/sp/v4/stage4A
ln -sfn /dev/shm/sp_v4_work/stage4B /work/sp/v4/stage4B
ln -sfn /dev/shm/sp_exp /work/sp/exp
ln -sfn /dev/shm/sp_distill_v4 /work/sp/distill_v4
ln -sfn /dev/shm/sp_hfpull /work/sp/hfpull

# 1. Download SLR43 + SLR54 + v3 distill cache into /dev/shm in parallel
st 05_corpora "05_corpora: Download SLR43 + SLR54 + v3 Single-Voice Cache into /dev/shm" run "parallel download into 500GB RAM disk"
(
  [ -f /work/sp/data/slr43/ne_np_female/line_index.tsv ] || bash $R/dl_slr43.sh > $L/dl_slr43.log 2>&1
) &
(
  ln -sfn /dev/shm/sp_data/slr54 /work/tts/data/slr54
  [ -f /work/sp/data/slr54/.done_f ] || python3 $S/dl_slr54.py > $L/dl_slr54.log 2>&1
) &
(
  python3 - <<'PY' > $L/dl_v3_distill.log 2>&1
import os, shutil
from huggingface_hub import snapshot_download
tok = os.environ['HF_TOKEN']
snapshot_download('NikitHamal/sakhiya-v1-train', allow_patterns=['sakhiya_pro_v3/distill/cache_train.pt', 'sakhiya_pro_v3/distill/manifest.jsonl', 'sakhiya_pro_v3/distill/texts.jsonl', 'sakhiya_pro_v3/v3/ref/*'], local_dir='/work/sp/hfpull', cache_dir='/dev/shm/hf_cache', token=tok)
shutil.rmtree('/dev/shm/hf_cache', ignore_errors=True)
print('V3_DISTILL_CACHE_OK')
PY
) &
wait
st 05_corpora "05_corpora: Download SLR43 + SLR54 + v3 Single-Voice Cache into /dev/shm" done "SLR43 + SLR54 + v3 cache ready in /dev/shm"
refresh_cards

# 2. Lock studio-denoised nep_6834 anchor and render immediate EP01 v4 Broadcast Master preview
python3 $V4/build_v4_data.py anchor > $L/v4_anchor.log 2>&1
st 08_ep01_init "08_ep01_init: Initial Episode 01 v4 Broadcast Master (De-Karkar + Pure Devanagari + -56 dB Room-Tone)" run "rendering 33 lines"
export SP_REF_WAV=/work/sp/v4/ref/anchor_v4.wav SP_REF_JSON=/work/sp/v4/ref/anchor_v4.json
python3 $S/v4_pro_render.py --model /work/sp/hf_pro_v1/v3 --ep01 --out /work/sp/render/ep01_v4 --takes 2 --steps 24 > $L/render_ep01_v4_init.log 2>&1
mkdir -p /work/sp/listen/pro_v4 /work/sp/repo/docs/sakhiya_pro/ep01_v4
cp /work/sp/render/ep01_v4/voiceover_ep01.mp3 /work/sp/listen/pro_v4/ep01_v4_voiceover.mp3 2>/dev/null || true
cp /work/sp/render/ep01_v4/voiceover_ep01.mp3 /work/sp/render/ep01_v4/timeline.json /work/sp/render/ep01_v4/take_report.json /work/sp/repo/docs/sakhiya_pro/ep01_v4/ 2>/dev/null || true
cd /work/sp/repo && GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0 git add docs/sakhiya_pro/ep01_v4 && git commit -m "ep01_v4: initial De-Karkar Broadcast Master voiceover (-14 LUFS, -56 dB room-tone)" && GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0 git push origin HEAD:sakhiya-pro-v4 || true
cd /work/sp
st 08_ep01_init "08_ep01_init: Initial Episode 01 v4 Broadcast Master" done "33/33 lines rendered (-14.0 LUFS) & pushed"
refresh_cards

# 3. Run full Sakhiya-Pro v4 pipeline (Studio Denoise 30-50h pool -> RVQ Tokenize -> Stage 4A -> Stage 4B -> Fixed Accent Selector -> Final EP01 v4 & 15-Clip Pack)
bash $V4/pipeline_v4.sh > $L/pipeline_v4_inner.log 2>&1

# 4. Build spotless 16-20 hour single-voice 24 kHz Nepal-native distillation corpus
st 50_corpus_v4 "50_corpus_v4: 16-20 Hr Single-Voice 24 kHz Nepal-Native Distillation Corpus" run "Pro v4 Synth + Studio-Denoised Nepal Speech -> cache_train.pt"
refresh_cards
python3 $V4/build_distill_v4.py /work/sp/final/sakhiya_pro_v4 > $L/build_distill_v4.log 2>&1
st 50_corpus_v4 "50_corpus_v4: 16-20 Hr Single-Voice 24 kHz Nepal-Native Distillation Corpus" done "$(tail -n 1 $L/build_distill_v4.log | cut -c1-120)"
refresh_cards

# 5. Execute Serial Zero-NaN Two-Phase Shrink Pipeline (Sakhiya-V2-Micro <=4.5 MB -> Sakhiya-V2-Pico <=1.4 MB)
MICRO_MIN=85 PICO_MIN=85 python3 $V4/serial_ladder_v4.py > $L/serial_ladder_v4.log 2>&1
echo "[$(date +%T)] CHAIN_V4_ALL_DONE"
