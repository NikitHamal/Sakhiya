"""15-minute HF backup of Sakhiya-Pro artefacts -> NikitHamal/sakhiya-v1-train/sakhiya_pro/ (never 5 min: HF 429)."""
import os, time, glob, shutil
from huggingface_hub import HfApi
tok = open('/tmp/hf_token').read().strip(); a = HfApi(token=tok); R = 'NikitHamal/sakhiya-v1-train'
O = '/work/sp/out'
def stage():
    os.makedirs(O + '/logs', exist_ok=True)
    os.system(f'cp -u /work/tts/logs/*.log {O}/logs/ 2>/dev/null')
    for src, dst in (('/work/sp/ref', 'ref'), ('/work/sp/evals', 'evals'), ('/work/sp/bakeoff', 'bakeoff'), ('/work/sp/render', 'render')):
        if os.path.isdir(src): os.system(f'mkdir -p {O}/{dst} && cp -ru {src}/. {O}/{dst}/ 2>/dev/null')
    for f in ('dataset_report.json', 'train_meta.json', 'dev_meta.json'):
        if os.path.exists('/work/sp/ft/' + f): os.system(f'mkdir -p {O}/ft && cp -u /work/sp/ft/{f} {O}/ft/')
    for d in glob.glob('/work/sp/exp/*/'):
        n = os.path.basename(d.rstrip('/')); cks = sorted(glob.glob(d + 'checkpoint-*'), key=lambda p: int(p.rsplit('-', 1)[1]))
        os.system(f'mkdir -p {O}/exp/{n} && cp -u {d}train.log {d}train_cfg.json {O}/exp/{n}/ 2>/dev/null')
        if cks:
            c = cks[-1]; t = f'{O}/exp/{n}/latest'; os.makedirs(t, exist_ok=True)
            for f in os.listdir(c):
                if f.endswith('.safetensors') or f.endswith('.json'): shutil.copy(os.path.join(c, f), t)
            open(t + '/STEP', 'w').write(c.rsplit('-', 1)[1])
    for d in glob.glob('/work/sp/final/*/'):
        n = os.path.basename(d.rstrip('/')); os.system(f'mkdir -p {O}/final/{n} && cp -ru {d}. {O}/final/{n}/')
while True:
    try:
        stage()
        a.upload_folder(folder_path=O, repo_id=R, path_in_repo='sakhiya_pro', commit_message='sakhiya_pro auto-backup ' + time.strftime('%H:%M:%S'), ignore_patterns=['*.tmp', '**/__pycache__/**'])
        print(time.strftime('%H:%M:%S'), 'backup ok', flush=True)
    except Exception as e: print(time.strftime('%H:%M:%S'), 'backup ERR', str(e)[:200], flush=True)
    time.sleep(900)
