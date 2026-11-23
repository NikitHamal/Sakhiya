import os,time,sys,glob
from huggingface_hub import HfApi
tok=open('/tmp/hf_token').read().strip(); a=HfApi(token=tok)
R='NikitHamal/sakhiya-v1-train'
while True:
    try:
        os.makedirs('/work/sp/out',exist_ok=True)
        for f in glob.glob('/work/tts/logs/*.log'):
            pass
        os.system('mkdir -p /work/sp/out/logs && cp -u /work/tts/logs/*.log /work/sp/out/logs/ 2>/dev/null')
        a.upload_folder(folder_path='/work/sp/out',repo_id=R,path_in_repo='sakhiya_pro',commit_message='sakhiya_pro auto-backup '+time.strftime('%H:%M:%S'),ignore_patterns=['*.tmp','**/__pycache__/**'])
        print(time.strftime('%H:%M:%S'),'backup ok',flush=True)
    except Exception as e: print(time.strftime('%H:%M:%S'),'backup ERR',str(e)[:200],flush=True)
    time.sleep(900)
