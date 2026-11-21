#!/usr/bin/env python3
"""Checkpoint insurance: every --every seconds upload new/changed *.pt / *.json under the watched run dirs and the logs dir to the
private HF repo (under <prefix>/), so a terminated MoLab sandbox never loses a run. Runs until killed; never raises."""
import argparse, glob, os, time
from huggingface_hub import HfApi
ap = argparse.ArgumentParser(); ap.add_argument("--dirs", nargs="+", required=True); ap.add_argument("--prefix", default="box3")
ap.add_argument("--every", type=int, default=300); ap.add_argument("--repo", default="NikitHamal/sakhiya-v1-train"); a = ap.parse_args()
api = HfApi(token=open("/tmp/hf_token").read().strip()); seen = {}
while True:
    for d in a.dirs:
        for p in glob.glob(f"{d}/*.pt") + glob.glob(f"{d}/*.json") + glob.glob(f"{d}/*.log"):
            if p.endswith(".tmp"): continue
            try:
                st = os.stat(p); key = (st.st_mtime, st.st_size)
                if seen.get(p) == key or time.time() - st.st_mtime < 5: continue
                api.upload_file(path_or_fileobj=p, path_in_repo=f"{a.prefix}/{os.path.basename(d)}/{os.path.basename(p)}", repo_id=a.repo, repo_type="model")
                seen[p] = key; print(time.strftime("%H:%M:%S"), "backed up", p, st.st_size // 1000, "kB", flush=True)
            except Exception as e:
                print("backup failed", p, repr(e)[:160], flush=True)
    time.sleep(a.every)
