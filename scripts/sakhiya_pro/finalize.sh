#!/bin/bash
# finalize.sh <SRC_CKPT_DIR> <EVAL_TAG>  — build final dir, render EP01 + 15-clip pack, EVAL.md, freeze into docs, push git, publish HF + GH release.
# Runs on the GPU box. Tokens only from /tmp/hf_token and /tmp/gitenv.sh (never committed).
set -e
SRC=$1; TAG=$2; L=/work/tts/logs; export PYTHONPATH=/work/sp/OmniVoice_src:/work/sp
R=/work/sp/repo; F=/work/sp/final/sakhiya_pro_v1; export GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0
cd $R; git pull -q --rebase origin main || true
rm -rf $F; mkdir -p $F
for f in $SRC/*; do case $(basename $f) in optimizer*|scheduler*|random_states*|*.pkl|*.bin) ;; *) cp -r $f $F/;; esac; done
cp /work/sp/ref/anchor_ref.wav /work/sp/ref/anchor_ref.json $F/
cp $R/scripts/sakhiya_pro_render.py $R/scripts/sakhiya_pro/sp_common.py $R/scripts/sakhiya_pro/bake_sents.py $F/
echo "$TAG from $SRC" > $F/SELECTED_FROM.txt
rm -rf /work/sp/render/pack /work/sp/render/ep01
python3 $R/scripts/sakhiya_pro_render.py --model $F --pack --out /work/sp/render/pack > $L/render_pack.log 2>&1
python3 $R/scripts/sakhiya_pro_render.py --model $F --ep01 --out /work/sp/render/ep01 > $L/render_ep01.log 2>&1
python3 - <<EOF
import json,glob
rows=[json.load(open(p))['summary'] for p in glob.glob('/work/sp/evals/*/scores.json')]
rows.sort(key=lambda s:-( -3*s['cer']-s['cer_vedic']-s['cer_tech']+0.3*s['utmos']+s['spk_sim']-0.5*s['hindi']-0.002*s['clicks']))
sel=[s for s in rows if s['tag']=="$TAG"][0]
json.dump({k:round(sel[k],4) for k in ('cer','cer_tech','cer_vedic','utmos','spk_sim','clicks')}|{'hindi_leaks':sel['hindi'],'eval_tag':"$TAG",'note':'automatic metrics on 8 hard sentences; no human MOS'}, open('/work/sp/render/metrics.json','w'), indent=1)
t='# Sakhiya-Pro v1 evaluation (synthesized audio, 8 hard sentences)\n\nSelected: **%s**\n\n| tag | CER | tech CER | Vedic CER | UTMOS22 | spk-sim | Hindi leaks | clicks |\n|---|---|---|---|---|---|---|---|\n'%"$TAG"
for s in rows: t+=f"| {s['tag']} | {s['cer']:.3f} | {s['cer_tech']:.3f} | {s['cer_vedic']:.3f} | {s['utmos']:.2f} | {s['spk_sim']:.2f} | {s['hindi']} | {s['clicks']:.0f} |\n"
open('$F/EVAL.md','w').write(t); open('$R/docs/sakhiya_pro/EVAL.md','w').write(t)
EOF
cd $R
rm -rf docs/samples/versions/sakhiya_pro
python3 scripts/sakhiya_pro/freeze_pro_pack.py /work/sp/render/pack /work/sp/render/metrics.json --notes "OmniVoice-600M full-backbone anchor fine-tune ($TAG); Best-of-3 + broadcast chain -14 LUFS; automatic metrics only"
mkdir -p docs/sakhiya_pro/ep01; cp /work/sp/render/ep01/voiceover_ep01.mp3 /work/sp/render/ep01/timeline.json /work/sp/render/ep01/take_report.json docs/sakhiya_pro/ep01/
git add -A docs; git commit -qm "sakhiya_pro: freeze 15-clip pack + EP01 voiceover + eval ($TAG)"; git pull -q --rebase origin main; git push -q origin HEAD:main; git log --oneline | head -1
python3 scripts/sakhiya_pro/publish_hf.py $F scripts/sakhiya_pro/MODEL_CARD.md /work/sp/render
cd /work/sp/render; rm -f /tmp/pro_pack.zip /tmp/pro_scripts.zip; (cd pack && zip -qr /tmp/pro_pack.zip *.mp3 pack_meta.json); (cd $F && zip -qr /tmp/pro_scripts.zip sakhiya_pro_render.py sp_common.py bake_sents.py anchor_ref.wav anchor_ref.json EVAL.md config.json)
printf "Sakhiya-Pro v1 — OmniVoice-600M anchor fine-tune (%s).\n\nWeights (2.4 GB, above GH's 2 GB asset cap): https://huggingface.co/NikitHamal/sakhiya-pro-v1\nAssets: EP01 voiceover (-14 LUFS), 15-clip pack, render scripts + anchor reference.\nAutomatic metrics only; see EVAL.md.\n" "$TAG" > /tmp/notes.md
python3 $R/scripts/sakhiya_pro/release_gh.py /tmp/notes.md /work/sp/render/ep01/voiceover_ep01.mp3 /work/sp/render/ep01/voiceover_ep01.wav /tmp/pro_pack.zip /tmp/pro_scripts.zip
touch /work/sp/final/published.txt; echo FINALIZE_DONE
