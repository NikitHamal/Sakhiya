#!/usr/bin/env python3
"""15-minute HF backup daemon for Sakhiya-Studio-80M (`scripts/sakhiya_pro/backup_studio_80m_daemon.py`).
Syncs clean mel cache metadata, Studio-80M checkpoints + EMA states, pretrained vocoder weights,
forensic gate JSONs, and clean truth matrix to `NikitHamal/sakhiya-v1-train` under `sakhiya_studio_80m_clean/`.
Never runs faster than 15 minutes (900s) to avoid HF 429 rate-limits.
"""
import glob
import os
import shutil
import sys
import time
from huggingface_hub import HfApi

TOK_FILE = "/tmp/hf_token"
REPO_ID = "NikitHamal/sakhiya-v1-train"
STAGE_DIR = "/work/sp/out_studio_80m_clean"


def stage_files():
    os.makedirs(STAGE_DIR + "/logs", exist_ok=True)
    os.makedirs(STAGE_DIR + "/checkpoints", exist_ok=True)
    os.makedirs(STAGE_DIR + "/eval", exist_ok=True)
    os.makedirs(STAGE_DIR + "/samples", exist_ok=True)

    for lg in glob.glob("/work/tts/logs/*.log"):
        try:
            shutil.copy2(lg, os.path.join(STAGE_DIR, "logs", os.path.basename(lg)))
        except Exception:
            pass

    for ck_src in glob.glob("/work/sp/studio_80m_clean/checkpoints/*.pt") + glob.glob("/work/sp/studio_80m_clean/*.json"):
        if os.path.isfile(ck_src):
            sub = "checkpoints" if ck_src.endswith(".pt") else "eval"
            dst = os.path.join(STAGE_DIR, sub, os.path.basename(ck_src))
            if not os.path.exists(dst) or os.path.getmtime(ck_src) > os.path.getmtime(dst):
                shutil.copy2(ck_src, dst)

    for ev_src in (
        glob.glob("/work/sp/sakhiya_news/eval/*.json")
        + glob.glob("/work/sp/repo/eval/*.json")
        + glob.glob("/work/Sakhiya/eval/*studio_80m*.json")
        + glob.glob("/work/Sakhiya/eval/clean_truth_matrix.json")
    ):
        try:
            shutil.copy2(ev_src, os.path.join(STAGE_DIR, "eval", os.path.basename(ev_src)))
        except Exception:
            pass

    for wav_src in (
        glob.glob("/work/sp/repo/docs/samples/versions/sakhiya_studio_80m_clean/*")
        + glob.glob("/work/Sakhiya/docs/samples/versions/sakhiya_studio_80m_clean/*")
    ):
        if os.path.isfile(wav_src):
            dst = os.path.join(STAGE_DIR, "samples", os.path.basename(wav_src))
            if not os.path.exists(dst) or os.path.getmtime(wav_src) > os.path.getmtime(dst):
                shutil.copy2(wav_src, dst)


def sync_once():
    tok = open(TOK_FILE).read().strip()
    api = HfApi(token=tok)
    stage_files()
    api.upload_folder(
        folder_path=STAGE_DIR,
        repo_id=REPO_ID,
        path_in_repo="sakhiya_studio_80m_clean",
        commit_message="sakhiya_studio_80m_clean 15m auto-backup " + time.strftime("%H:%M:%S"),
        ignore_patterns=["*.tmp", "**/__pycache__/**"],
    )
    print(f"[{time.strftime('%H:%M:%S')}] HF backup ok -> {REPO_ID}/sakhiya_studio_80m_clean", flush=True)


if __name__ == "__main__":
    if "--once" in sys.argv:
        sync_once()
        sys.exit(0)
    while True:
        try:
            sync_once()
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] HF backup ERR: {str(e)[:200]}", flush=True)
        time.sleep(900)
