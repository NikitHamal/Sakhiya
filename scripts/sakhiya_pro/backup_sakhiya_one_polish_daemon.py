#!/usr/bin/env python3
"""15-minute HF backup daemon for Sakhiya-One Surgical Polish (`scripts/sakhiya_pro/backup_sakhiya_one_polish_daemon.py`).
Syncs checkpoints, adapters, ONNX models, forensic gate JSONs, and -16 LUFS sample packs to
`NikitHamal/sakhiya-v1-train` under `sakhiya_one_polish/`.
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
STAGE_DIR = "/work/sp/out_sakhiya_one_polish"


def stage_files():
    for sub in ("logs", "checkpoints", "onnx", "eval", "samples"):
        os.makedirs(os.path.join(STAGE_DIR, sub), exist_ok=True)

    for lg in glob.glob("/work/tts/logs/*.log"):
        try:
            shutil.copy2(lg, os.path.join(STAGE_DIR, "logs", os.path.basename(lg)))
        except Exception:
            pass

    for ck_src in (
        glob.glob("/work/sp/polish/ck/*.pt")
        + glob.glob("/work/sp/polish/*.json")
        + glob.glob("/work/sp/distill/runs/sakhiya_one_polish/*.pt")
    ):
        if os.path.isfile(ck_src):
            sub = "checkpoints" if ck_src.endswith(".pt") else "eval"
            dst = os.path.join(STAGE_DIR, sub, os.path.basename(ck_src))
            if not os.path.exists(dst) or os.path.getmtime(ck_src) > os.path.getmtime(dst):
                shutil.copy2(ck_src, dst)

    for ox_src in (
        glob.glob("/work/sp/polish/onnx/**/*", recursive=True)
        + glob.glob("/work/Sakhiya/docs/models_sakhiya_one*/*.onnx")
        + glob.glob("/work/Sakhiya/docs/models_sakhiya_one*/*.json")
    ):
        if os.path.isfile(ox_src):
            rel = os.path.basename(os.path.dirname(ox_src)) + "_" + os.path.basename(ox_src)
            dst = os.path.join(STAGE_DIR, "onnx", rel)
            if not os.path.exists(dst) or os.path.getmtime(ox_src) > os.path.getmtime(dst):
                shutil.copy2(ox_src, dst)

    for ev_src in (
        glob.glob("/work/Sakhiya/eval/*sakhiya_one*.json")
        + glob.glob("/work/Sakhiya/eval/*showdown*.json")
        + glob.glob("/work/sp/polish/eval/*.json")
    ):
        if os.path.isfile(ev_src):
            try:
                shutil.copy2(ev_src, os.path.join(STAGE_DIR, "eval", os.path.basename(ev_src)))
            except Exception:
                pass

    for wav_src in (
        glob.glob("/work/sp/polish/listen/*")
        + glob.glob("/work/Sakhiya/docs/samples/versions/sakhiya_one_polish/*")
        + glob.glob("/work/Sakhiya/docs/samples/versions/sakhiya_one_vocos/*")
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
        path_in_repo="sakhiya_one_polish",
        commit_message="sakhiya_one_polish 15m auto-backup " + time.strftime("%H:%M:%S"),
        ignore_patterns=["*.tmp", "**/__pycache__/**"],
    )
    print(f"[{time.strftime('%H:%M:%S')}] HF backup ok -> {REPO_ID}/sakhiya_one_polish", flush=True)


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
