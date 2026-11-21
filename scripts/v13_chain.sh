#!/bin/bash
# v13 EXPERIMENT (box 3): brand-new Nepali + Sanskrit models FROM SCRATCH on the mined real data. Nothing here ships to the site;
# results + listenable samples only. Each stage <= 110 min, step-50 gate inside, audio check every 20-min chunk.
R=/work/tts; L=$R/logs; V=$R/v13; mkdir -p $V; cd $R/repo9 || exit 1
dl() { python3 -c "from huggingface_hub import hf_hub_download as d;import shutil;shutil.copy(d('NikitHamal/sakhiya-v1-train','v13/$1',token=open('/tmp/hf_token').read().strip()),'$V/$1');print('got $1')"; }
have() { python3 -c "from huggingface_hub import HfApi;import sys;sys.exit(0 if 'v13/$1' in HfApi(token=open('/tmp/hf_token').read().strip()).list_repo_files('NikitHamal/sakhiya-v1-train') else 1)" 2>/dev/null; }
for i in $(seq 1 90); do have mined_24k.pt && break; sleep 40; done
dl mined_24k_report.json; dl mined_24k.pt || { echo "[v13] no mined pack"; exit 1; }
echo "[v13] mined pack: $(cat $V/mined_24k_report.json | head -c 400)"
pkill -f "[h]f_backup.py" ; sleep 1
nohup python3 $R/hf_backup.py --dirs $R/runs/v9_voca_am $R/runs/v9_sano_am $R/runs/v10_voc144 $R/runs/v10_voc144_adapt $R/v10 $V/am_scratch $V/voc_scratch $V --prefix box3 --every 900 > $L/hf_backup.log 2>&1 &
nohup python3 scripts/v13_scratch_am.py --out $V/am_scratch > $L/v13_am.log 2>&1 &
echo "[v13] scratch acoustic model started $(date -u +%H:%M)Z"
for i in $(seq 1 90); do have real_audio_24k.pt && break; sleep 40; done
dl real_audio_24k.pt || echo "[v13] no audio pack"
while pgrep -f "[v]12_box3_chain.sh" >/dev/null; do sleep 30; done
if [ -f $V/real_audio_24k.pt ]; then
  echo "[v13] scratch vocoder started $(date -u +%H:%M)Z"
  python3 scripts/v13_scratch_voc.py --scratch --no_stop --audio_pack $V/real_audio_24k.pt --voc_init $R/runs/v5_c1/voc.pt --am $R/runs/v5_c1/am.pt \
    --sak $R/feats/v6_expressive_vedic_24k.pt --shipped docs/models_v2 --out $V/voc_scratch --chunk-min 20 --max-chunks 5 --max-minutes 110 \
    --gate-limit 15 --lr 2e-4 --d_warm 0 --w_anchor 0.0 --bs 32 > $L/v13_voc.log 2>&1
  echo "[v13] scratch vocoder rc=$? $(grep DONE $V/voc_scratch/train.log | tail -1)"
fi
while pgrep -f "[v]13_scratch_am.py" >/dev/null; do sleep 30; done
echo "[v13] scratch acoustic model finished $(date -u +%H:%M)Z $(grep DONE $V/am_scratch/train.log | tail -1 | cut -c1-300)"
BV=$(python3 -c "import json;r=[json.loads(l) for l in open('$V/voc_scratch/chunks.jsonl')];print('$V/voc_scratch/c%d_voc.pt' % max(r,key=lambda x:x['cand']['score'])['chunk'])" 2>/dev/null)
python3 scripts/v13_samples.py --model docs/models_v2 --name V1 --out $V/samples > /dev/null 2>&1
if [ -n "$BV" ] && [ -f $V/am_scratch/final.pt ]; then
  python3 scripts/export_v2_onnx.py --tier v25 --am $V/am_scratch/final.pt --voc $BV --out $V/stage_pair > $L/v13_export_pair.log 2>&1
  python3 scripts/forensic_audio_gate.py --model $V/stage_pair --tiny --limit 30 --out $V/gate_pair.json > $L/v13_gate_pair.log 2>&1
  python3 scripts/v13_samples.py --model $V/stage_pair --name scratchAM+scratchVoc --out $V/samples
  python3 scripts/v13_samples.py --model $V/voc_scratch/stage_c$(basename $BV | tr -dc 0-9) --name V1AM+scratchVoc --out $V/samples
fi
python3 - <<'PY'
import json, os
V = "/work/tts/v13"; out = {}
def rd(p):
    try: return [json.loads(l) for l in open(p)]
    except Exception: return []
out["acoustic_from_scratch_chunks"] = rd(f"{V}/am_scratch/gates.jsonl")
out["vocoder_from_scratch_chunks"] = [dict(chunk=r["chunk"], score=r["cand"]["score"], utmos=r["cand"]["utmos"], drops=r["cand"]["drops"], v1=r["base"]["score"]) for r in rd(f"{V}/voc_scratch/chunks.jsonl")]
try:
    g = json.load(open(f"{V}/gate_pair.json")); out["pair"] = dict(utmos=g.get("utmos_15"), drops=g.get("word_drop_count"), stutters=g.get("stutter_count"), clicks=g.get("clicks"), voice=g.get("sv_cos_ne_sa"))
except Exception: out["pair"] = None
out["v1_reference_utmos"] = 3.717
for f in ("mined_24k_report.json",):
    try: out["data"] = json.load(open(f"{V}/{f}"))
    except Exception: pass
json.dump(out, open("/work/tts/repo9/eval/v13_scratch_results.json", "w"), indent=1, ensure_ascii=False); print("[v13] RESULTS", json.dumps(out)[:1500])
PY
export GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0
git add eval/v13_scratch_results.json scripts/v13_*.py scripts/v13_chain.sh 2>/dev/null; git commit -qm "v13 experiment: from-scratch acoustic model + vocoder on mined real data (results only, nothing shipped)"; git pull -q --rebase origin main; git push -q origin HEAD:main
echo "[v13] CHAIN DONE $(date -u +%H:%M)Z"
