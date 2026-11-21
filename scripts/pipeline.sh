#!/bin/bash
# Full Sakhiya v2 pipeline on the GPU box:  bash scripts/pipeline.sh   (resumable via marker files)
R=/work/tts; L=$R/logs; mkdir -p $L $R/runs $R/feats; cd $R/repo
AM_MIN=${AM_MIN:-50}; VOC_MIN=${VOC_MIN:-50}; GTA_MIN=${GTA_MIN:-25}
push() { export GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0; git add -A docs eval 2>/dev/null; git commit -qm "$1"; git pull -q --rebase origin main; git push -q origin HEAD:main; }
[ -f $L/deps.done ] || { pip install -q pyworld jiwer librosa soundfile praat-parselmouth speechmos piper-tts > $L/pip.log 2>&1; pip install -q --no-deps torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130 >> $L/pip.log 2>&1; touch $L/deps.done; }
[ -f $L/dl.done ] || { python3 scripts/dl_all.py > $L/dl.log 2>&1 && touch $L/dl.done; }
[ -f $L/curate.done ] || { python3 scripts/curate.py > $L/curate.log 2>&1 && touch $L/curate.done || exit 1; }
if [ ! -f $L/phase1.done ]; then
  python3 scripts/train_vocoder.py --set v2 --minutes $VOC_MIN --out $R/runs/voc2 --bs 32 > $L/voc2.log 2>&1 &
  python3 scripts/train_am.py --set v2 --minutes $AM_MIN --bs 64 --out $R/runs/am2 > $L/am2.log 2>&1
  wait; touch $L/phase1.done
fi
[ -f $L/gta.done ] || { python3 scripts/train_vocoder.py --set v2 --minutes $GTA_MIN --out $R/runs/voc_gta --bs 32 --gta $R/feats/gta_v2.pt --init $R/runs/voc2/G.pt --lr 1e-4 > $L/gta.log 2>&1 && touch $L/gta.done; }
[ -f $L/export.done ] || { python3 scripts/export_onnx.py --am $R/runs/am2/last.pt --voc $R/runs/voc_gta/G.pt --out $R/export2 > $L/export.log 2>&1 && touch $L/export.done; }
mkdir -p docs/models; cp $R/export2/*.onnx $R/export2/config.json $R/export2/size.json docs/models/ 2>/dev/null; push "v2 models exported (fp16 + int8)"
[ -f $L/eval.done ] || { python3 scripts/evaluate.py --export $R/export2 > $L/eval.log 2>&1; touch $L/eval.done; }
push "v2 evaluation, samples and metrics"
echo PIPELINE_DONE >> $L/pipeline.log
