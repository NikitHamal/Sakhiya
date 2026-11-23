#!/bin/bash
# Sakhiya-Pro v2 one-shot chain (launch detached on the box):  setsid nohup bash pipeline_v2.sh [TARGET_H=30] > /work/tts/logs/pipeline_v2.log 2>&1 < /dev/null &
# corpus workers -> merge -> tokenize -> Stage 1 (broad Nepal-native phonology) -> pron eval -> Stage 2 (70% anchor / 30% diversity, LR 2e-6) -> pron eval -> pick -> final/sakhiya_pro_v2 -> pack + EP01
cd /work/sp; L=/work/tts/logs; R=/work/sp/repo/scripts; P=$R/sakhiya_pro; export PYTHONPATH=/work/sp/OmniVoice_src:/work/sp HF_TOKEN=$(cat /tmp/hf_token)
TARGET=${1:-30}; mkdir -p /work/sp/stage /work/sp/listen /work/sp/tiers
st(){ echo "{\"label\":\"$2\",\"state\":\"$3\",\"note\":\"$4\"}" > /work/sp/stage/$1.json; }
st 10_corpus "Stage-1 corpus (SLR54 + SLR43 + relaxed v13)" run "$TARGET h target"
st 20_s1 "Stage 1: broad Nepal-native phonology" todo ""; st 30_s2 "Stage 2: anchor lock (LR 2e-6)" todo ""; st 40_verify "Verify: pron eval + 15 clips + EP01" todo ""
st 50_corpus_d "Distillation corpus (20-25 h from Pro v2)" todo ""; st 60_micro "Sakhiya-Micro (<=4.5 MB int8)" todo ""; st 70_pico "Sakhiya-Pico (<=1.5 MB int8)" todo ""
hrs(){ python3 - <<'PY'
import glob, json
print(sum(json.loads(l)['dur'] for p in glob.glob('/work/sp/ft2/part_*.jsonl') for l in open(p) if l.strip().endswith('}')) / 3600)
PY
}
until pgrep -f '[b]uild_s1.py work' >/dev/null && false; do
  H=$(hrs); echo "[$(date +%T)] corpus ${H}h"; [ "$(python3 -c "print(int($H>=$TARGET*1.3))")" = 1 ] && break
  pgrep -f '[b]uild_s1.py work' >/dev/null || break
  [ $(date +%s) -gt ${S1_DEADLINE:-9999999999} ] && break; sleep 30
done
for p in $(pgrep -f '[b]uild_s1.py work'); do kill $p; done; sleep 3
python3 $P/build_s1.py merge $TARGET > $L/s1_merge.log 2>&1; cat $L/s1_merge.log | tail -12
st 10_corpus "Stage-1 corpus (SLR54 + SLR43 + relaxed v13)" done "$(grep total_hours /work/sp/ft2/dataset_report.json)"
tok(){ # $1 = dir containing train.jsonl/dev.jsonl
  cd /work/sp/OmniVoice_src; T=$1/tokens; rm -rf $T
  for s in train dev; do python -m omnivoice.scripts.extract_audio_tokens --input_jsonl $1/$s.jsonl --tar_output_pattern "$T/$s/audios/shard-%06d.tar" --jsonl_output_pattern "$T/$s/txts/shard-%06d.jsonl" \
    --tokenizer_path bosonai/higgs-audio-v2-tokenizer --nj_per_gpu 8 --shuffle True; done; cd /work/sp; }
tok /work/sp/ft2 > $L/tokenize_s1.log 2>&1; tail -n 2 $L/tokenize_s1.log
st 20_s1 "Stage 1: broad Nepal-native phonology" run "bf16, step-50 watchdog"
python3 $P/guarded_train.py --name s1 --steps 1200 --lr 8e-6 --batch_tokens 16384 --save_steps 200 --budget_min 135 --tokens /work/sp/ft2/tokens; echo "[$(date +%T)] s1 rc=$?"
python3 $P/eval_ckpt.py /work/sp/final/sakhiya_pro_v1 v1_ref > /dev/null 2>&1; python3 $P/eval_pron.py /work/sp/final/sakhiya_pro_v1 pron_v1 > $L/pron_v1.log 2>&1; tail -n 1 $L/pron_v1.log
for d in /work/sp/exp/s1/checkpoint-*; do s=${d##*-}; [ $((s % 400)) -eq 0 ] || continue; python3 $P/eval_pron.py $d pron_s1_$s > $L/pron_s1_$s.log 2>&1; tail -n 1 $L/pron_s1_$s.log; done
pick(){ python3 - "$@" <<'PY'
import json, glob, sys
best = None
for sj in glob.glob('/work/sp/evals/pron_%s*/pron_scores.json' % sys.argv[1]):
    s = json.load(open(sj))['summary']
    c = -3 * s['cer_native'] + 1.0 * s['word_recall'] - s['cer_tech'] - 0.5 * s['cer_vedic'] + 0.3 * s['utmos'] + 1.5 * s['spk_sim'] - 0.5 * s['hindi'] - 0.002 * s['clicks']
    if best is None or c > best[0]: best = (c, s['tag'])
print(best[1])
PY
}
B1=$(pick s1); S=${B1#pron_}; echo "best stage-1: $B1"; N1=/work/sp/exp/s1/checkpoint-${B1##*_}
st 20_s1 "Stage 1: broad Nepal-native phonology" done "picked $B1"
python3 - <<'PY'
import json, random
rng = random.Random(9); O = '/work/sp/ft3v2'; import os; os.makedirs(O, exist_ok=True)
meta = json.load(open('/work/sp/ft/train_meta.json')); anchor = []; seen = set()
for r in meta:
    if r['src'] == 'slr43_6834' and r['id'].rsplit('_r', 1)[0] not in seen: seen.add(r['id'].rsplit('_r', 1)[0]); anchor.append(r)
div = [json.loads(l) for l in open('/work/sp/ft2/train.jsonl')]; div = [r for r in div if not r['id'].startswith('anchor')]
n_div = int(len(anchor) * 3 * 0.30 / 0.70)  # anchor x3 = 70 %
rows = []
for k in range(3): rows += [dict(id=f"{r['id'].rsplit('_r',1)[0]}_a{k}", audio_path=r['audio_path'], text=r['text'], language_id=r['language_id']) for r in anchor]
rows += rng.sample(div, min(n_div, len(div))); rng.shuffle(rows)
dev = [json.loads(l) for l in open('/work/sp/ft/dev.jsonl')][:10]
for name, rs in (('train', rows), ('dev', dev)):
    with open(f'{O}/{name}.jsonl', 'w', encoding='utf8') as f:
        for r in rs: f.write(json.dumps({k: r[k] for k in ('id', 'audio_path', 'text', 'language_id')}, ensure_ascii=False) + '\n')
print('stage2 rows', len(rows), 'anchor', len(anchor) * 3, 'diversity', n_div)
PY
tok /work/sp/ft3v2 > $L/tokenize_s2.log 2>&1
st 30_s2 "Stage 2: anchor lock (LR 2e-6)" run "70% anchor / 30% diversity"
python3 $P/guarded_train.py --name s2 --steps 300 --lr 2e-6 --batch_tokens 8192 --save_steps 100 --budget_min 40 --init $N1 --tokens /work/sp/ft3v2/tokens; echo "[$(date +%T)] s2 rc=$?"
for d in /work/sp/exp/s2/checkpoint-*; do s=${d##*-}; python3 $P/eval_pron.py $d pron_s2_$s > $L/pron_s2_$s.log 2>&1; tail -n 1 $L/pron_s2_$s.log; done
B2=$(pick s2); echo "best stage-2: $B2"; st 30_s2 "Stage 2: anchor lock (LR 2e-6)" done "picked $B2"
BEST=/work/sp/exp/s2/checkpoint-${B2##*_}
rm -rf /work/sp/final/sakhiya_pro_v2; cp -r /work/sp/final/sakhiya_pro_v1 /work/sp/final/sakhiya_pro_v2; cp $BEST/model.safetensors /work/sp/final/sakhiya_pro_v2/model.safetensors
echo "$B2 (stage-1 $B1)" > /work/sp/final/sakhiya_pro_v2/SELECTED_FROM.txt
st 40_verify "Verify: pron eval + 15 clips + EP01" run "rendering"
python3 $R/sakhiya_pro_render.py --model /work/sp/final/sakhiya_pro_v2 --pack --out /work/sp/render/pack_v2 > $L/render_pack_v2.log 2>&1
python3 $R/sakhiya_pro_render.py --model /work/sp/final/sakhiya_pro_v2 --ep01 --out /work/sp/render/ep01_v2 > $L/render_ep01_v2.log 2>&1
mkdir -p /work/sp/listen/pro_v2; cp /work/sp/render/pack_v2/*.mp3 /work/sp/listen/pro_v2/ 2>/dev/null; cp /work/sp/render/ep01_v2/voiceover_ep01.mp3 /work/sp/listen/pro_v2/ep01_voiceover.mp3 2>/dev/null
st 40_verify "Verify: pron eval + 15 clips + EP01" done "PRO_V2_DONE"; echo "[$(date +%T)] PRO_V2_DONE"
