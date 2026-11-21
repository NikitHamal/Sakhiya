#!/bin/bash
# Autonomous follow-up: when the distillation corpus is complete AND wave 8 has finished, rebuild feats on the FULL corpus and run wave 9
# (warm-start from the wave-8 AM + vocoder, same promote gate, self-pushing).  Launched detached on the GPU box.
R=/work/tts; L=$R/logs; cd $R/repo
while ! grep -q DISTILL_DONE $L/distill_0.log $L/distill_1.log $L/distill_2.log $L/distill_3.log 2>/dev/null || ! grep -q WAVE8_DONE $L/wave8.log 2>/dev/null; do sleep 20; done
echo "CHAIN9_START $(date)" > $L/chain9.log
cp -f $R/feats/edge.pt $R/feats/edge_w8.pt; rm -f $R/feats/edge.pt
TAG=10 AM_MIN=${AM_MIN:-20} VOC_MIN=${VOC_MIN:-18} AM_INIT=$R/runs/am9/last.pt VOC_INIT=$R/runs/voc9/G.pt bash scripts/wave8.sh >> $L/chain9.log 2>&1
echo "CHAIN9_DONE $(date)" >> $L/chain9.log
