#!/bin/bash
# Sakhiya-Pro v3: native-Nepal weight lock.  setsid nohup bash pipeline_v3.sh > /work/tts/logs/pipeline_v3.log 2>&1 < /dev/null &
L=/work/tts/logs; R=/work/sp/repo/scripts/sakhiya_pro; V3=$R/v3; S=/work/sp/repo/scripts; export PYTHONPATH=/work/sp/OmniVoice_src:/work/sp HF_TOKEN=$(cat /tmp/hf_token)
st(){ echo "{\"label\":\"$2\",\"state\":\"$3\",\"note\":\"$4\"}" > /work/sp/stage/$1.json; }
cd /work/sp; mkdir -p /work/sp/v3/eval
st 05_v3_audit "v3 accent audit (npi tag, instruct=None, anchor + guidance + spelling sweep)" done "best: gs 2.0 + native spelling; anchor with downward closure"
python3 - <<'PY'
import json, collections, shutil
a = json.load(open('/work/sp/v3/audit_v2/audit.json')); D = '/work/sp/data/slr43/ne_np_female'
cnt = collections.Counter(l.split('\t')[0].split('_')[1] for l in open(D + '/line_index.tsv'))
pick = [k for k in ('cand0', 'cand1', 'cand2') if k in a['refs'] and cnt.get((a['refs'][k].get('uid') or '').split('_')[1], 0) >= 100]
k = pick[0] if pick else 'cur'; r = a['refs'][k]
uid = r.get('uid') or json.load(open('/work/sp/ref/anchor_ref.json'))['uid']; print('ANCHOR', k, uid, r['text'], flush=True)
shutil.copy(r['path'], '/work/sp/v3/ref/anchor_v3.wav')
json.dump(dict(ref_text=r['text'], uid=uid, src=r.get('src')), open('/work/sp/v3/ref/anchor_v3.json', 'w'), ensure_ascii=False)
PY
st 10_v3_data "v3 data: stitched 6-14 s native utterances + anchor lock set" run "building stage B"
python3 $V3/build_v3_data.py B > $L/v3_dataB.log 2>&1; tail -n 2 $L/v3_dataB.log
tok(){ cd /work/sp/OmniVoice_src; T=$1/tokens; rm -rf $T
  for s in train dev; do python -m omnivoice.scripts.extract_audio_tokens --input_jsonl $1/$s.jsonl --tar_output_pattern "$T/$s/audios/shard-%06d.tar" --jsonl_output_pattern "$T/$s/txts/shard-%06d.jsonl" \
    --tokenizer_path bosonai/higgs-audio-v2-tokenizer --nj_per_gpu 8 --shuffle True; done; cd /work/sp; }
tok /work/sp/v3/stageA > $L/v3_tokA.log 2>&1; tail -n 1 $L/v3_tokA.log | cut -c1-200
tok /work/sp/v3/stageB > $L/v3_tokB.log 2>&1; tail -n 1 $L/v3_tokB.log | cut -c1-200
st 10_v3_data "v3 data: stitched 6-14 s native utterances + anchor lock set" done "tokenized A+B"
# baseline: v2 under the same fixed config (for the before/after table)
(python3 $V3/accent_eval.py /work/sp/final/sakhiya_pro_v2 /work/sp/v3/eval/v2 > $L/eval_v2.log 2>&1 &)
st 20_v3_A "v3 stage A: native weight lock (lr 1.4e-5 cosine->2e-6, prompt 0.10, 3200 steps, bf16)" run "step-50 watchdog"
python3 $R/guarded_train.py --name v3a --steps 3200 --lr 1.4e-5 --min_lr_ratio 0.143 --prompt_ratio 0.10 --batch_tokens 16384 --save_steps 800 --budget_min 55 --init /work/sp/models/OmniVoice --tokens /work/sp/v3/stageA/tokens; echo "[$(date +%T)] v3a rc=$?"
st 20_v3_A "v3 stage A: native weight lock" done "3200 steps"
st 30_v3_B "v3 stage B: single-anchor voice lock (lr 4e-6 -> 1e-6, 600 steps)" run "70% anchor / 30% diversity"
python3 $R/guarded_train.py --name v3b --steps 600 --lr 4e-6 --min_lr_ratio 0.25 --prompt_ratio 0.10 --batch_tokens 8192 --save_steps 200 --budget_min 25 --init /work/sp/exp/v3a/checkpoint-3200 --tokens /work/sp/v3/stageB/tokens; echo "[$(date +%T)] v3b rc=$?"
st 30_v3_B "v3 stage B: single-anchor voice lock" done "600 steps"
st 40_v3_verify "v3 verify: accent / spk-sim / consistency eval of checkpoints" run "A-3200, B-200/400/600"
for c in v3a/checkpoint-2400 v3a/checkpoint-3200 v3b/checkpoint-200 v3b/checkpoint-400 v3b/checkpoint-600; do
  t=${c//\//_}; [ -d /work/sp/exp/$c ] && python3 $V3/accent_eval.py /work/sp/exp/$c /work/sp/v3/eval/$t > $L/eval_$t.log 2>&1 &
  [ $(jobs -r | wc -l) -ge 3 ] && wait -n
done; wait
python3 - <<'PY'
import json, glob, shutil, os
best = None
for p in glob.glob('/work/sp/v3/eval/v3*/eval.json'):
    s = json.load(open(p))['summary']; print(os.path.basename(os.path.dirname(p)), round(s['score'], 3), 'spk', round(s['spk_sim'], 3), 'end', round(s['end_slope'], 2), 'cer', round(s['cer'], 3), 'utmos', round(s['utmos'], 2))
    if best is None or s['score'] > best[0]: best = (s['score'], os.path.basename(os.path.dirname(p)), s)
tag = best[1]; ck = '/work/sp/exp/' + tag.replace('_', '/', 1); print('BEST', tag, ck)
F = '/work/sp/final/sakhiya_pro_v3'; shutil.rmtree(F, ignore_errors=True); shutil.copytree('/work/sp/final/sakhiya_pro_v2', F)
shutil.copy(ck + '/model.safetensors', F + '/model.safetensors')
for f in ('anchor_ref.wav',): shutil.copy('/work/sp/v3/ref/anchor_v3.wav', F + '/anchor_ref.wav')
r = json.load(open('/work/sp/v3/ref/anchor_v3.json')); json.dump(dict(r, dur=8.0), open(F + '/anchor_ref.json', 'w'), ensure_ascii=False)
open(F + '/SELECTED_FROM.txt', 'w').write(tag + '\n'); json.dump(best[2], open(F + '/eval_accent.json', 'w'), indent=1)
PY
cp $S/sakhiya_pro_render.py /work/sp/final/sakhiya_pro_v3/ 2>/dev/null
st 40_v3_verify "v3 verify: accent eval" done "selected; rendering EP01 + 15-clip pack"
export SP_REF_WAV=/work/sp/v3/ref/anchor_v3.wav SP_REF_JSON=/work/sp/v3/ref/anchor_v3.json SP_ACCENT=1 SP_NATIVIZE=1
python3 $S/sakhiya_pro_render.py --model /work/sp/final/sakhiya_pro_v3 --ep01 --out /work/sp/render/ep01_v3 > $L/render_ep01_v3.log 2>&1 &
python3 $S/sakhiya_pro_render.py --model /work/sp/final/sakhiya_pro_v3 --pack --out /work/sp/render/pack_v3 > $L/render_pack_v3.log 2>&1 &
wait
mkdir -p /work/sp/listen/pro_v3; cp /work/sp/render/pack_v3/*.mp3 /work/sp/listen/pro_v3/ 2>/dev/null; cp /work/sp/render/ep01_v3/voiceover_ep01.mp3 /work/sp/listen/pro_v3/ep01_voiceover.mp3 2>/dev/null
st 40_v3_verify "v3 verify + render" done "PRO_V3_DONE"; echo "[$(date +%T)] PRO_V3_DONE"
