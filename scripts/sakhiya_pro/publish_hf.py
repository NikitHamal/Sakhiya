#!/usr/bin/env python3
"""Publish Sakhiya-Pro to HF (NikitHamal/sakhiya-pro-v1) + final copy under sakhiya-v1-train/sakhiya_pro/final.
usage: publish_hf.py <final_dir> <model_card.md> [render_dir]"""
import sys, os, shutil
from huggingface_hub import HfApi
fin, card = sys.argv[1], sys.argv[2]; rend = sys.argv[3] if len(sys.argv) > 3 else None
a = HfApi(token=open('/tmp/hf_token').read().strip()); R = 'NikitHamal/sakhiya-pro-v1'
a.create_repo(R, repo_type='model', exist_ok=True, private=False)
shutil.copy(card, os.path.join(fin, 'README.md'))
a.upload_folder(folder_path=fin, repo_id=R, commit_message='Sakhiya-Pro v1: OmniVoice-600M anchor fine-tune + reference + render scripts', ignore_patterns=['__pycache__/**'])
if rend and os.path.isdir(rend):
    a.upload_folder(folder_path=rend, repo_id=R, path_in_repo='samples', commit_message='Sakhiya-Pro v1 samples: EP01 voiceover + 15-clip pack', allow_patterns=['*.mp3', '*.json', 'voiceover_ep01.wav', 'ep01/*.mp3', 'ep01/*.json', 'ep01/voiceover_ep01.wav', 'pack/*.mp3', 'pack/*.json'])
print('published', R)
