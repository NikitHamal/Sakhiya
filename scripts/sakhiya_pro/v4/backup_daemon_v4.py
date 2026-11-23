#!/usr/bin/env python3
"""15-minute HF backup of Sakhiya-Pro v4 artefacts -> NikitHamal/sakhiya-v1-train/sakhiya_pro_v4/ (never 5 min: avoids HF 429)."""
import os, time, glob, shutil
from huggingface_hub import HfApi

tok = open('/tmp/hf_token').read().strip()
api = HfApi(token=tok)
REPO_ID = 'NikitHamal/sakhiya-v1-train'
OUT_STAGE = '/work/sp/out_v4'


def stage():
    os.makedirs(OUT_STAGE + '/logs', exist_ok=True)
    os.system(f'cp -u /work/tts/logs/*.log {OUT_STAGE}/logs/ 2>/dev/null')
    for src, dst in (
        ('/work/sp/v4/diag', 'v4/diag'),
        ('/work/sp/v4/eval', 'v4/eval'),
        ('/work/sp/v4/audit', 'v4/audit'),
        ('/work/sp/v4/ref', 'v4/ref'),
        ('/work/sp/render/ep01_v4', 'render/ep01_v4'),
        ('/work/sp/render/pack_v4', 'render/pack_v4'),
        ('/work/sp/stage', 'stage'),
        ('/work/sp/listen', 'listen'),
        ('/work/sp/tiers_v4', 'tiers_v4'),
    ):
        if os.path.isdir(src):
            os.system(f'mkdir -p {OUT_STAGE}/{dst} && cp -ru {src}/. {OUT_STAGE}/{dst}/ 2>/dev/null')

    for f in ('texts.jsonl', 'manifest.jsonl', 'corpus_report.json', 'denoise_report.json'):
        if os.path.exists('/work/sp/distill_v4/' + f):
            os.makedirs(OUT_STAGE + '/distill_v4', exist_ok=True)
            shutil.copy2('/work/sp/distill_v4/' + f, f'{OUT_STAGE}/distill_v4/{f}')

    for d in glob.glob('/work/sp/distill_v4/runs/*/'):
        n = os.path.basename(d.rstrip('/'))
        os.makedirs(f'{OUT_STAGE}/distill_v4/runs/{n}', exist_ok=True)
        for fn in ('train.log', 'ema_latest.pt', 'ema_final.pt'):
            if os.path.exists(d + fn):
                shutil.copy2(d + fn, f'{OUT_STAGE}/distill_v4/runs/{n}/{fn}')

    for d in glob.glob('/work/sp/exp/v4*/'):
        n = os.path.basename(d.rstrip('/'))
        cks = sorted(glob.glob(d + 'checkpoint-*'), key=lambda p: int(p.rsplit('-', 1)[1]))
        os.makedirs(f'{OUT_STAGE}/exp/{n}', exist_ok=True)
        for fn in ('train.log', 'train_cfg.json'):
            if os.path.exists(d + fn):
                shutil.copy2(d + fn, f'{OUT_STAGE}/exp/{n}/{fn}')
        if cks:
            c = cks[-1]
            t = f'{OUT_STAGE}/exp/{n}/latest'
            os.makedirs(t, exist_ok=True)
            for fn in os.listdir(c):
                if fn.endswith('.safetensors') or fn.endswith('.json'):
                    shutil.copy2(os.path.join(c, fn), os.path.join(t, fn))
            open(t + '/STEP', 'w').write(c.rsplit('-', 1)[1])

    if os.path.isdir('/work/sp/final/sakhiya_pro_v4'):
        os.system(f'mkdir -p {OUT_STAGE}/final/sakhiya_pro_v4 && cp -ru /work/sp/final/sakhiya_pro_v4/. {OUT_STAGE}/final/sakhiya_pro_v4/')


if __name__ == '__main__':
    while True:
        try:
            stage()
            api.upload_folder(
                folder_path=OUT_STAGE,
                repo_id=REPO_ID,
                path_in_repo='sakhiya_pro_v4',
                commit_message='sakhiya_pro_v4 auto-backup ' + time.strftime('%H:%M:%S'),
                ignore_patterns=['*.tmp', '**/__pycache__/**'],
            )
            print(time.strftime('%H:%M:%S'), 'v4 backup ok', flush=True)
        except Exception as e:
            print(time.strftime('%H:%M:%S'), 'v4 backup ERR', str(e)[:200], flush=True)
        time.sleep(900)
