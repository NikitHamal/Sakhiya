#!/bin/bash
# Sakhiya-Pro v4 pipeline:
# 1. Studio-denoise 30-50 hr Nepal-native pool (SLR43 + SLR54 female + v13 Nepal news/story/conv) & lock cleanest anchor
# 2. Extract Higgs-Audio-v2 RVQ tokens on studio-clean audio (zero phone-mic hiss in layers 5-8)
# 3. Stage 4A: broad denoised flowing Nepal-native phonology pass (6-15s stitched, pure Devanagari, lang="npi", instruct=None,
#    prompt_ratio=0.0, cosine LR 1.2e-5 -> 2e-6, strictly capped with Step-50 watchdog)
# 4. Stage 4B: tight clean-anchor cadence lock evaluating checkpoints every 200 steps (200, 400, 600, 800, 1000) with
#    FIXED Accent & Cleanliness Selector
# 5. Render & freeze Episode 01 v4 Broadcast Master + 15-clip pack (sakhiya_pro_v4), upload to HF, and push to sakhiya-pro-v4.

set -e
L=/work/tts/logs
R=/work/sp/repo/scripts/sakhiya_pro
V4=$R/v4
S=/work/sp/repo/scripts
export PYTHONPATH=/work/sp/OmniVoice_src:/work/sp
export HF_TOKEN=$(cat /tmp/hf_token)
export HUGGING_FACE_HUB_TOKEN=$(cat /tmp/hf_token)

st(){ echo "{\"label\":\"$2\",\"state\":\"$3\",\"note\":\"$4\"}" > /work/sp/stage/$1.json; }
refresh_cards(){ python3 $S/molab_exec.py --py "exec(open('/work/sp/repo/monitor/cells_pro4.py').read()); await install()" >/dev/null 2>&1 || true; }

cd /work/sp
st 01_diag "01_diag: Forensic Zero-Training Diagnostic (v3b_200/400 vs 600, spelling=devanagari, De-Karkar chain)" done "v3b_400 beats 600 (-0.57 vs -0.42 st); spelling=devanagari & De-Karkar verified"
refresh_cards

st 10_v4_data "10_v4_data: Studio-Denoise 30-50h Nepal Pool + Anchor Audition + RVQ Tokenization" run "DeepFilterNet3 / spectral denoise + de-click + 6-15s stitching"
python3 $V4/build_v4_data.py all > $L/v4_data.log 2>&1
tail -n 3 $L/v4_data.log

tok(){
  cd /work/sp/OmniVoice_src
  T=$1/tokens
  rm -rf $T
  for s in train dev; do
    python3 -m omnivoice.scripts.extract_audio_tokens \
      --input_jsonl $1/$s.jsonl \
      --tar_output_pattern "$T/$s/audios/shard-%06d.tar" \
      --jsonl_output_pattern "$T/$s/txts/shard-%06d.jsonl" \
      --tokenizer_path bosonai/higgs-audio-v2-tokenizer \
      --nj_per_gpu 8 \
      --shuffle True
  done
  cd /work/sp
}

tok /work/sp/v4/stage4A > $L/v4_tokA.log 2>&1
tok /work/sp/v4/stage4B > $L/v4_tokB.log 2>&1
st 10_v4_data "10_v4_data: Studio-Denoise 30-50h Nepal Pool + Anchor Audition + RVQ Tokenization" done "Stage 4A & 4B studio-clean RVQ tokens extracted"
refresh_cards

# Init from best v3b_checkpoint-400 if present, else /work/sp/hf_pro_v1/v3 or /work/sp/models/OmniVoice
INIT_4A="/work/sp/models/OmniVoice"
if [ -d "/work/sp/hf_pro_v1/v3" ]; then INIT_4A="/work/sp/hf_pro_v1/v3"; fi

st 20_v4_A "20_v4_A: Stage 4A Broad Denoised Nepal-Native Phonology Pass (prompt_ratio 0.0, lr 1.2e-5 -> 2e-6)" run "step-50 watchdog active, cap <=50 min"
python3 $R/guarded_train.py \
  --name v4a \
  --steps 2400 \
  --lr 1.2e-5 \
  --min_lr_ratio 0.1667 \
  --prompt_ratio 0.0 \
  --batch_tokens 16384 \
  --save_steps 800 \
  --budget_min 50 \
  --init "$INIT_4A" \
  --tokens /work/sp/v4/stage4A/tokens
st 20_v4_A "20_v4_A: Stage 4A Broad Denoised Nepal-Native Phonology Pass" done "2400 steps completed"
refresh_cards

st 30_v4_B "30_v4_B: Stage 4B Tight Clean-Anchor Cadence Lock (eval every 200 steps: 200..1000)" run "prompt_ratio 0.0, lr 3.5e-6 -> 1e-6, cap <=35 min"
python3 $R/guarded_train.py \
  --name v4b \
  --steps 1000 \
  --lr 3.5e-6 \
  --min_lr_ratio 0.2857 \
  --prompt_ratio 0.0 \
  --batch_tokens 8192 \
  --save_steps 200 \
  --budget_min 35 \
  --init /work/sp/exp/v4a/checkpoint-2400 \
  --tokens /work/sp/v4/stage4B/tokens
st 30_v4_B "30_v4_B: Stage 4B Tight Clean-Anchor Cadence Lock" done "checkpoints 200, 400, 600, 800, 1000 saved"
refresh_cards

st 40_v4_verify "40_v4_verify: Fixed Accent & Cleanliness Selector across 200/400/600/800/1000" run "penalizing rise_frac>0.14, end_slope>-0.55, clicks>3, hf_karkar"
for c in v4a/checkpoint-2400 v4b/checkpoint-200 v4b/checkpoint-400 v4b/checkpoint-600 v4b/checkpoint-800 v4b/checkpoint-1000; do
  t=${c//\//_}
  if [ -d "/work/sp/exp/$c" ]; then
    python3 $V4/accent_eval_v4.py "/work/sp/exp/$c" "/work/sp/v4/eval/$t" > "$L/eval_$t.log" 2>&1 &
  fi
  while [ $(jobs -r | wc -l) -ge 2 ]; do sleep 5; done
done
wait
refresh_cards

python3 - <<'PY'
import json, glob, shutil, os
best = None
for p in sorted(glob.glob('/work/sp/v4/eval/v4*/eval.json')):
    s = json.load(open(p))['summary']
    tag = os.path.basename(os.path.dirname(p))
    print(tag, 'score=', round(s['score'], 3), 'end=', round(s['end_slope'], 3), 'rise=', round(s['rise_frac'], 3), 'clicks=', round(s['clicks'], 2), 'karkar=', round(s['hf_karkar'], 4), 'cer=', round(s['cer'], 3), 'utmos=', round(s['utmos'], 3), flush=True)
    if best is None or s['score'] > best[0]:
        best = (s['score'], tag, s)
tag = best[1]
ck = '/work/sp/exp/' + tag.replace('_', '/', 1)
print('SELECTED_V4:', tag, ck, flush=True)
F = '/work/sp/final/sakhiya_pro_v4'
shutil.rmtree(F, ignore_errors=True)
base_src = '/work/sp/hf_pro_v1/v3' if os.path.isdir('/work/sp/hf_pro_v1/v3') else '/work/sp/models/OmniVoice'
shutil.copytree(base_src, F)
shutil.copy2(ck + '/model.safetensors', F + '/model.safetensors')
shutil.copy2('/work/sp/v4/ref/anchor_v4.wav', F + '/anchor_ref.wav')
shutil.copy2('/work/sp/v4/ref/anchor_v4.json', F + '/anchor_ref.json')
open(F + '/SELECTED_FROM.txt', 'w').write(tag + '\n')
json.dump(best[2], open(F + '/eval_accent.json', 'w'), indent=2)
PY

cp $S/v4_pro_render.py /work/sp/final/sakhiya_pro_v4/ 2>/dev/null || true
st 40_v4_verify "40_v4_verify: Rendering EP01 v4 Broadcast Master + 15-Clip Pack" run "De-Karkar chain + -56 dB room-tone bed"
export SP_REF_WAV=/work/sp/v4/ref/anchor_v4.wav SP_REF_JSON=/work/sp/v4/ref/anchor_v4.json
python3 $S/v4_pro_render.py --model /work/sp/final/sakhiya_pro_v4 --ep01 --out /work/sp/render/ep01_v4 > $L/render_ep01_v4.log 2>&1 &
python3 $S/v4_pro_render.py --model /work/sp/final/sakhiya_pro_v4 --pack --out /work/sp/render/pack_v4 > $L/render_pack_v4.log 2>&1 &
wait
mkdir -p /work/sp/listen/pro_v4
cp /work/sp/render/pack_v4/*.mp3 /work/sp/listen/pro_v4/ 2>/dev/null || true
cp /work/sp/render/ep01_v4/voiceover_ep01.mp3 /work/sp/listen/pro_v4/ep01_v4_voiceover.mp3 2>/dev/null || true
python3 $V4/finalize_v4.py freeze > $L/finalize_v4.log 2>&1
st 40_v4_verify "40_v4_verify: Sakhiya-Pro v4 + EP01 v4 Frozen & Uploaded" done "PRO_V4_DONE"
refresh_cards
echo "[$(date +%T)] PRO_V4_DONE"
