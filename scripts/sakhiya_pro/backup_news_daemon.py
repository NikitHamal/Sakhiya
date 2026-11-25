#!/usr/bin/env python3
"""15-minute HF backup daemon for Sakhiya-News (`scripts/sakhiya_pro/backup_news_daemon.py`).
Syncs corpus shards (`cache_100mel/`), vocoder weights (`voc_100mel_best.pt`), AM checkpoints + optimizer/EMA states (`am_last.pt`, `am_best.pt`),
forensic gate JSONs, and `RELAY_STATE.json` to `NikitHamal/sakhiya-v1-train` under `sakhiya_news_final/`.
Never runs faster than 15 minutes (900s) to avoid HF 429 rate-limits.
"""
import glob
import os
import shutil
import time
from huggingface_hub import HfApi

TOK_FILE = "/tmp/hf_token"
REPO_ID = "NikitHamal/sakhiya-v1-train"
STAGE_DIR = "/work/sp/out_news_final"


def stage_files():
    os.makedirs(STAGE_DIR + "/logs", exist_ok=True)
    os.makedirs(STAGE_DIR + "/cache_100mel", exist_ok=True)
    os.makedirs(STAGE_DIR + "/checkpoints", exist_ok=True)
    os.makedirs(STAGE_DIR + "/eval", exist_ok=True)

    for lg in glob.glob("/work/tts/logs/*.log"):
        try:
            shutil.copy2(lg, os.path.join(STAGE_DIR, "logs", os.path.basename(lg)))
        except Exception:
            pass

    if os.path.isdir("/work/sp/sakhiya_news/cache_100mel"):
        for fn in os.listdir("/work/sp/sakhiya_news/cache_100mel"):
            if fn.endswith((".pt", ".json", ".jsonl")):
                src = os.path.join("/work/sp/sakhiya_news/cache_100mel", fn)
                dst = os.path.join(STAGE_DIR, "cache_100mel", fn)
                if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
                    shutil.copy2(src, dst)

    for ck_src in (
        "/work/tts/ck/voc_100mel_best.pt",
        "/work/tts/ck/voc_100mel_last.pt",
        "/work/tts/ck/am_last.pt",
        "/work/tts/ck/am_best.pt",
        "/work/sp/sakhiya_news/voc_100mel_best.pt",
        "/work/sp/sakhiya_news/am_last.pt",
        "/work/sp/sakhiya_news/am_best.pt",
    ):
        if os.path.exists(ck_src):
            dst = os.path.join(STAGE_DIR, "checkpoints", os.path.basename(ck_src))
            if not os.path.exists(dst) or os.path.getmtime(ck_src) > os.path.getmtime(dst):
                shutil.copy2(ck_src, dst)

    for ev_src in glob.glob("/work/sp/sakhiya_news/eval/*.json") + glob.glob("/work/sp/repo/eval/*.json"):
        try:
            shutil.copy2(ev_src, os.path.join(STAGE_DIR, "eval", os.path.basename(ev_src)))
        except Exception:
            pass

    for rs in ("/work/sp/repo/RELAY_STATE.json", "/work/sp/sakhiya_news/RELAY_STATE.json"):
        if os.path.exists(rs):
            shutil.copy2(rs, os.path.join(STAGE_DIR, "RELAY_STATE.json"))

    if os.path.isdir("/work/sp/sakhiya_news/leg1_preview"):
        os.makedirs(STAGE_DIR + "/leg1_preview", exist_ok=True)
        os.system(f"cp -ru /work/sp/sakhiya_news/leg1_preview/. {STAGE_DIR}/leg1_preview/ 2>/dev/null")


def sync_once():
    tok = open(TOK_FILE).read().strip()
    api = HfApi(token=tok)
    stage_files()
    api.upload_folder(
        folder_path=STAGE_DIR,
        repo_id=REPO_ID,
        path_in_repo="sakhiya_news_final",
        commit_message="sakhiya_news_final 15m auto-backup " + time.strftime("%H:%M:%S"),
        ignore_patterns=["*.tmp", "**/__pycache__/**"],
    )
    print(f"[{time.strftime('%H:%M:%S')}] HF backup ok -> {REPO_ID}/sakhiya_news_final", flush=True)


if __name__ == "__main__":
    import sys
    if "--once" in sys.argv:
        sync_once()
        sys.exit(0)
    while True:
        try:
            sync_once()
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] HF backup ERR: {str(e)[:200]}", flush=True)
        time.sleep(900)
