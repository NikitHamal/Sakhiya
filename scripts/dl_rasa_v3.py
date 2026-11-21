"""v3 data: ai4bharat/Rasa Nepali + Sanskrit parquets -> /work/tts/data/rasa (resumable). Token from /tmp/hf_token (never committed)."""
import os, sys, time
from huggingface_hub import snapshot_download
tok = open("/tmp/hf_token").read().strip()
dst = sys.argv[1] if len(sys.argv) > 1 else "/work/tts/data/rasa"
for i in range(20):
    try:
        p = snapshot_download("ai4bharat/Rasa", repo_type="dataset", local_dir=dst, token=tok,
                              allow_patterns=["Nepali/*", "Sanskrit/*", "*.md", "*.json"], max_workers=8)
        print("RASA_DL_DONE", p, flush=True); break
    except Exception as e:
        print("retry", i, repr(e)[:200], flush=True); time.sleep(10)
