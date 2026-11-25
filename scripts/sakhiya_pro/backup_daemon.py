"""15-minute HF backup of Sakhiya-Pro artefacts -> NikitHamal/sakhiya-v1-train/sakhiya_pro/ (never 5 min: HF 429)."""
import os, time, glob, shutil
from huggingface_hub import HfApi
tok = open('/tmp/hf_token').read().strip(); a = HfApi(token=tok); R = 'NikitHamal/sakhiya-v1-train'
O = '/work/sp/out'
def stage():
    os.makedirs(O + '/logs', exist_ok=True)
    os.system(f'cp -u /work/tts/logs/*.log {O}/logs/ 2>/dev/null')
    for src, dst in (('/work/sp/v3/eval', 'v3/eval'), ('/work/sp/v3/audit_v2', 'v3/audit'), ('/work/sp/v3/ref', 'v3/ref'), ('/work/sp/ref', 'ref'), ('/work/sp/evals', 'evals'), ('/work/sp/bakeoff', 'bakeoff'), ('/work/sp/render', 'render'), ('/work/sp/stage', 'stage'), ('/work/sp/listen', 'listen'), ('/work/sp/tiers', 'tiers'), ('/work/sp/pron', 'pron')):
        if os.path.isdir(src): os.system(f'mkdir -p {O}/{dst} && cp -ru {src}/. {O}/{dst}/ 2>/dev/null')
    for f in ('dataset_report.json', 'train_meta.json', 'dev_meta.json'):
        if os.path.exists('/work/sp/ft/' + f): os.system(f'mkdir -p {O}/ft && cp -u /work/sp/ft/{f} {O}/ft/')
    for d in glob.glob('/work/sp/distill/runs/*/'):   # student ladder: EMA checkpoints + logs (resume state.pt stays local)
        n = os.path.basename(d.rstrip('/')); os.system(f'mkdir -p {O}/distill/runs/{n} && cp -u {d}train.log {d}ema_latest.pt {d}ema_final.pt {O}/distill/runs/{n}/ 2>/dev/null')
    for f in ('texts.jsonl', 'manifest.jsonl', 'corpus_report.json', 'cache_train.pt'):
        if os.path.exists('/work/sp/distill/' + f): os.system(f'mkdir -p {O}/distill && cp -u /work/sp/distill/{f} {O}/distill/')
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
    os.makedirs('/work/sp/out_one', exist_ok=True)
    os.system('cp -u /work/tts/logs/*.log /work/sp/out_one/ 2>/dev/null')
    for f in ('sakhiya_one_voc.pt', 'sakhiya_one_am.pt', 'sakhiya_one_e2e.pt'):
        if os.path.exists('/work/tts/ck/' + f):
            shutil.copy('/work/tts/ck/' + f, '/work/sp/out_one/' + f)
    if os.path.isdir('/work/sp/tiers/sakhiya_one/best'):
        os.system('mkdir -p /work/sp/out_one/best && cp -ru /work/sp/tiers/sakhiya_one/best/. /work/sp/out_one/best/ 2>/dev/null')
while True:
    try:
        stage()
        a.upload_folder(folder_path=O, repo_id=R, path_in_repo='sakhiya_pro_v3', commit_message='sakhiya_pro_v3 auto-backup ' + time.strftime('%H:%M:%S'), ignore_patterns=['*.tmp', '**/__pycache__/**'])
        a.upload_folder(folder_path='/work/sp/out_one', repo_id=R, path_in_repo='sakhiya_one', commit_message='sakhiya_one auto-backup ' + time.strftime('%H:%M:%S'), ignore_patterns=['*.tmp', '**/__pycache__/**'])
        print(time.strftime('%H:%M:%S'), 'backup ok (sakhiya_pro_v3 + sakhiya_one)', flush=True)
    except Exception as e: print(time.strftime('%H:%M:%S'), 'backup ERR', str(e)[:200], flush=True)
    time.sleep(900)
