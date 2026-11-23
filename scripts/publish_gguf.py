#!/usr/bin/env python3
"""Upload verified GGUF models + SHA256SUMS + prebuilt Linux/WASM artifacts to Hugging Face and (optionally) a GitHub release.
Tokens are read from $HF_TOKEN (or --hf-token-file) and $GITHUB_TOKEN; never written to disk/logs.
  python scripts/publish_gguf.py --dir /tmp/release [--github-release v15-ggml-gguf]
"""
import argparse, json, os, sys, urllib.request
from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="/tmp/release")
ap.add_argument("--hf-token-file", default="/tmp/hf_token")
ap.add_argument("--repo", default="NikitHamal/sakhiya-gguf")
ap.add_argument("--train-repo", default="NikitHamal/sakhiya-v1-train")
ap.add_argument("--train-prefix", default="box6_v15/ggml")
ap.add_argument("--card", default=os.path.join(os.path.dirname(__file__), "..", "docs", "gguf_model_card.md"))
ap.add_argument("--github-release", default="")
ap.add_argument("--github-repo", default="NikitHamal/Sakhiya")
a = ap.parse_args()
tok = os.environ.get("HF_TOKEN") or open(a.hf_token_file).read().strip()
api = HfApi(token=tok)
api.create_repo(a.repo, repo_type="model", exist_ok=True)
files = sorted(f for f in os.listdir(a.dir) if os.path.isfile(os.path.join(a.dir, f)))
api.upload_folder(repo_id=a.repo, folder_path=a.dir, commit_message="Sakhiya GGUF v15: V1/Nano/Atom/Atom-Lite Q8_0 + exact + F16, Linux x86_64 + WASM")
if os.path.exists(a.card):
    api.upload_file(path_or_fileobj=a.card, path_in_repo="README.md", repo_id=a.repo, commit_message="model card")
api.upload_folder(repo_id=a.train_repo, folder_path=a.dir, path_in_repo=a.train_prefix, commit_message="box6_v15/ggml: multi-tier GGUF + binaries")
print("HF_UPLOADED", len(files), "files ->", a.repo, "and", a.train_repo + "/" + a.train_prefix)

if a.github_release:
    gt = os.environ["GITHUB_TOKEN"]
    H = {"Authorization": f"token {gt}", "Accept": "application/vnd.github+json", "User-Agent": "sakhiya"}
    def call(url, data=None, method=None, headers=None, raw=False):
        req = urllib.request.Request(url, data=data, method=method, headers={**H, **(headers or {})})
        try:
            with urllib.request.urlopen(req) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            return {"_err": e.code, "_body": e.read()[:300].decode()}
    base = f"https://api.github.com/repos/{a.github_repo}"
    rel = call(f"{base}/releases/tags/{a.github_release}")
    if "_err" in rel:
        rel = call(f"{base}/releases", json.dumps({"tag_name": a.github_release, "name": a.github_release,
              "body": "Sakhiya GGML C++ engine v15: single-file GGUF models for V1/Nano/Atom/Atom-Lite (Q8_0 compact, Q8_0 exact, F16), "
                      "SHA256SUMS, prebuilt Linux x86_64 and WASM.", "target_commitish": "main"}).encode(), "POST")
    up = rel["upload_url"].split("{")[0]
    have = {x["name"] for x in rel.get("assets", [])}
    for f in files:
        if f in have: continue
        with open(os.path.join(a.dir, f), "rb") as fh:
            r = call(f"{up}?name={f}", fh.read(), "POST", {"Content-Type": "application/octet-stream"})
        print("GH", f, r.get("size", r))
    print("GH_RELEASE", rel.get("html_url"))
