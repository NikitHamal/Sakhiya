#!/bin/bash
# v2.6 chain (runs detached on the box): wait for AM -> balanced GTA dump -> 50/50 vocoder polish from the ALL-SPEAKER Stage 3A G_best -> handoff to export + gate.
# usage: setsid nohup bash scripts/v26_chain.sh > /work/tts/logs/v26_chain.log 2>&1 < /dev/null &
cd /work/tts/repo; L=/work/tts/logs; R=/work/tts/runs
upd() { python3 - "$1" "$2" <<'P'
import json, sys, time
p = "/work/tts/COORD_V26.json"; c = json.load(open(p)); c["right_now"] = sys.argv[2]; c["stage_idx"] = int(sys.argv[1]); c["updated"] = time.time(); json.dump(c, open(p, "w"), indent=1)
P
}
mkdir -p $R/v26_voc; [ -f $R/v26_voc_init_3A.pt ] || cp $R/v2_voc/G_best.pt $R/v26_voc_init_3A.pt
while ! grep -q "AM_DONE" $L/v26_am_train.log; do sleep 30; done
upd 2 "Stage 3: dumping balanced 50/50 GTA mels from the converged v2.6 AM"
python3 -u scripts/v26_dump_gta.py --ckpt $R/v26_am/polish_final.pt --out /work/tts/feats/v26_gta_2voice.pt > $L/v26_dump_gta.log 2>&1
upd 2 "Stage 3: balanced GTA vocoder polish (50% female / 50% male batches, warm start = all-speaker 3A)"
python3 -u scripts/v26_train_voc.py --out $R/v26_voc --init $R/v26_voc_init_3A.pt --gta /work/tts/feats/v26_gta_2voice.pt --gta_p 0.7 --lr 4e-5 --ema 0.999 --minutes 75 > $L/v26_voc_train.log 2>&1
upd 3 "Stage 4: AM + vocoder trained. Next: export 2-voice ONNX, run scripts/gate_pronunciation_v26.py, refresh docs/samples"
echo CHAIN_DONE
