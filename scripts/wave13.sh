#!/bin/bash
# Wave 13 (session 13): AM continues from am12 with a 3x pitch-contour loss; vocoder fine-tunes on predicted mels (GTA 60%) from voc12 (D resumed).
# Data: 37 h two-voice teacher corpus (feats/edge.pt), gta from am12 (feats/gta_edge.pt). 20 min bf16 each, concurrent. Naturalness-first promote gate.
R=/work/tts; L=$R/logs; cd $R/repo; mkdir -p $R/runs $L
AM=${AM_MIN:-22}; VOC=${VOC_MIN:-22}; TAG=${TAG:-13}; PREV=${PREV:-12}
echo "WAVE${TAG}_START $(date)" > $L/wave$TAG.log
mkdir -p $R/best$TAG; rm -rf $R/best$TAG/models $R/best$TAG/samples; cp -r docs/models $R/best$TAG/models; cp -r docs/samples $R/best$TAG/samples; cp docs/metrics.json $R/best$TAG/metrics.json
export ROMAN_ORT=$R/node/package.json
python3 scripts/train_am.py --set edge --minutes $AM --bs 64 --lr 1.5e-4 --w_pit 0.45 --init $R/runs/am$PREV/last.pt --out $R/runs/am$TAG > $L/am$TAG.log 2>&1 &
python3 scripts/train_vocoder.py --set edge --minutes $VOC --out $R/runs/voc$TAG --bs 48 --init $R/runs/voc$PREV/G.pt --lr 1.2e-4 --phase 1.0 --gta $R/feats/gta_edge.pt --gta_p 0.6 --focus edge_f,edge_m --focus_p 1.0 > $L/voc$TAG.log 2>&1 &
wait
echo "TRAIN_DONE $(date)" >> $L/wave$TAG.log
cp $R/feats/gta_edge.pt $R/feats/gta_edge_w$TAG.pt
python3 scripts/export_onnx.py --am $R/runs/am$TAG/last.pt --voc $R/runs/voc$TAG/G.pt --out $R/export$TAG > $L/export$TAG.log 2>&1 || { echo EXPORT_FAIL >> $L/wave$TAG.log; exit 1; }
cp $R/export$TAG/*.onnx $R/export$TAG/config.json $R/export$TAG/size.json docs/models/
git checkout -- docs/samples 2>/dev/null; python3 scripts/evaluate.py --export $R/export$TAG --out $R/eval$TAG > $L/eval$TAG.log 2>&1
TAG=$TAG python3 scripts/promote.py $TAG > $L/promote$TAG.log 2>&1; cat $L/promote$TAG.log >> $L/wave$TAG.log
