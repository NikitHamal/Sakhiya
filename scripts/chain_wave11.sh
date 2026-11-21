#!/bin/bash
# Session 11: when wave 10 (am10/voc10) is done and round-2 distillation (10k extra multi-sentence clips) is complete,
# rebuild feats on ALL teacher audio (~25 h) and run wave 11 warm-started from wave 10 with the naturalness-first gate.
R=/work/tts; L=$R/logs; cd $R/repo
while ! grep -q CHAIN9_DONE $L/chain9.log 2>/dev/null || [ "$(grep -l DISTILL_DONE $L/distill2_*.log 2>/dev/null | wc -l)" -lt 4 ]; do sleep 20; done
echo "CHAIN11_START $(date)" > $L/chain11.log
# keep wave-10 candidate artefacts before the gate output is overwritten
[ -d $R/export10 ] && cp -r $R/export10 $R/export10_keep 2>/dev/null; cp -f $R/eval10/metrics.json $R/eval10_metrics.json 2>/dev/null
cp -f $R/feats/edge.pt $R/feats/edge_w10.pt; rm -f $R/feats/edge.pt
TAG=11 AM_MIN=${AM_MIN:-24} VOC_MIN=${VOC_MIN:-22} AM_INIT=$R/runs/am10/last.pt VOC_INIT=$R/runs/voc10/G.pt bash scripts/wave11.sh >> $L/chain11.log 2>&1
echo "CHAIN11_DONE $(date)" >> $L/chain11.log
