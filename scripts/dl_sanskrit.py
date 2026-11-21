import time, os, sys
from huggingface_hub import hf_hub_download, list_repo_files
fs = sorted(f for f in list_repo_files("surajp/shrutilipi_sanskrit", repo_type="dataset") if f.endswith(".parquet"))
n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
for f in fs[:n]:
    t = time.time()
    try: p = hf_hub_download("surajp/shrutilipi_sanskrit", f, repo_type="dataset", local_dir="/work/tts/data/sa/hf"); print(f, os.path.getsize(p) / 1e6, "MB", round(time.time() - t), "s", flush=True)
    except Exception as e: print("fail", f, e, flush=True)
print("DL_DONE", flush=True)
