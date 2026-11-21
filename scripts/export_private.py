#!/usr/bin/env python3
"""Progressive PRIVATE export of the mined corpus to Hugging Face + Kaggle (never public, never logs secrets).

  HF     : NikitHamal/sakhiya-nepali-vedic-private   (dataset, private=True)   <- /tmp/hf_token
  Kaggle : iamnikithamal/sakhiya-nepali-vedic-private (dataset, private)       <- /tmp/kaggle_token (KAGGLE_API_TOKEN)
Both are incremental: HF dedups by content hash; Kaggle gets a new version only when new shards exist
(files are hard-linked into a staging dir, so no extra disk).  Writes /work/tts/v6_state.json.
usage: export_private.py [--cycle N] [--only hf|kaggle]
"""
import argparse, glob, json, os, shutil, subprocess, sys, time

ROOT = os.environ.get("TTS_ROOT", "/work/tts")
HF_REPO = "NikitHamal/sakhiya-nepali-vedic-private"
KG_OWNER, KG_SLUG = "iamnikithamal", "sakhiya-nepali-vedic-private"
V6_STATE = f"{ROOT}/v6_state.json"
META_FILES = [f"{ROOT}/dataset/meta.jsonl", f"{ROOT}/mining_state.json", f"{ROOT}/ia_manifest.jsonl", f"{ROOT}/gate/speakers.json"]


def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)


def tok(path): return open(path).read().strip()


def dataset_files():
    out = []
    for sub in ("raw", "vc"):
        out += sorted(glob.glob(f"{ROOT}/dataset/{sub}/*.parquet") + glob.glob(f"{ROOT}/dataset/{sub}/*.tar"))
    return out


def sig(files): return sorted((os.path.basename(os.path.dirname(f)) + "/" + os.path.basename(f), os.path.getsize(f)) for f in files)


def load_state():
    try: return json.load(open(V6_STATE))
    except Exception: return {}


def save_state(**kw):
    s = load_state(); s.update(kw, ts=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(s, open(V6_STATE + ".tmp", "w"), indent=1); os.replace(V6_STATE + ".tmp", V6_STATE)


def card():
    return ("---\nlicense: other\nlanguage: [ne, sa]\ntags: [tts, nepali, sanskrit, vedic, private]\n---\n"
            "# sakhiya-nepali-vedic-private\nPRIVATE corpus: demucs-isolated, VAD-segmented, speaker-pure, UTMOS-gated, dual-ASR verified clips of "
            "Nepali storytelling / news / conversation and Vedic chanting with acoustic punctuation (`, ? !`), F0/energy/phoneme durations.\n"
            "`raw/` = original multi-speaker cleaned audio. `vc/` = unified `sakhiya` single-voice conversion. Shards: Parquet + WebDataset tar.\n"
            "Source audio belongs to its original publishers; this private mirror is for research/model training only.\n")


def to_hf(files):
    from huggingface_hub import HfApi
    api = HfApi(token=tok("/tmp/hf_token"))
    api.create_repo(HF_REPO, repo_type="dataset", private=True, exist_ok=True)
    open(f"{ROOT}/README_hf.md", "w").write(card())
    api.upload_file(path_or_fileobj=f"{ROOT}/README_hf.md", path_in_repo="README.md", repo_id=HF_REPO, repo_type="dataset")
    for sub in ("raw", "vc"):
        d = f"{ROOT}/dataset/{sub}"
        if glob.glob(d + "/*"):
            api.upload_folder(folder_path=d, path_in_repo=sub, repo_id=HF_REPO, repo_type="dataset", commit_message=f"{sub} shards {time.strftime('%F %T')}")
    for f in META_FILES:
        if os.path.exists(f): api.upload_file(path_or_fileobj=f, path_in_repo="meta/" + os.path.basename(f), repo_id=HF_REPO, repo_type="dataset")
    info = api.dataset_info(HF_REPO)
    assert info.private, "HF repo must be private"
    return True


def to_kaggle(files, cycle):
    stage = f"{ROOT}/kaggle_stage"; shutil.rmtree(stage, ignore_errors=True)
    for f in files:
        dst = f"{stage}/{os.path.basename(os.path.dirname(f))}/{os.path.basename(f)}"; os.makedirs(os.path.dirname(dst), exist_ok=True)
        try: os.link(f, dst)
        except OSError: shutil.copy2(f, dst)
    os.makedirs(stage, exist_ok=True)
    for f in META_FILES:
        if os.path.exists(f): shutil.copy2(f, f"{stage}/{os.path.basename(f)}")
    open(f"{stage}/README.md", "w").write(card())
    json.dump({"title": "Sakhiya Nepali Vedic private corpus", "id": f"{KG_OWNER}/{KG_SLUG}", "licenses": [{"name": "other"}], "isPrivate": True},
              open(f"{stage}/dataset-metadata.json", "w"))
    env = dict(os.environ, KAGGLE_API_TOKEN=tok("/tmp/kaggle_token"))
    exists = subprocess.run(["kaggle", "datasets", "status", f"{KG_OWNER}/{KG_SLUG}"], env=env, capture_output=True, text=True)
    ok = exists.returncode == 0 and "ready" in (exists.stdout + exists.stderr).lower()
    cmd = ["kaggle", "datasets", "version", "-p", stage, "-m", f"cycle {cycle} {time.strftime('%F %T')}", "--dir-mode", "tar"] if ok else \
          ["kaggle", "datasets", "create", "-p", stage, "--dir-mode", "tar"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3000)
    out = (r.stdout + r.stderr).replace(env["KAGGLE_API_TOKEN"], "***")
    log("kaggle", " ".join(cmd[2:3]), out[-300:].replace("\n", " "))
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cycle", type=int, default=0); ap.add_argument("--only", choices=["hf", "kaggle"]); a = ap.parse_args()
    files = dataset_files(); s = sig(files); st = load_state()
    if not files: log("no dataset shards yet"); return 0
    res = {}
    for name, fn in (("hf", lambda: to_hf(files)), ("kaggle", lambda: to_kaggle(files, a.cycle))):
        if a.only and a.only != name: continue
        if st.get(f"{name}_sig") == s: log(name, "up to date"); continue
        try:
            res[name] = bool(fn())
        except Exception as e:
            res[name] = False; log(name, "FAIL", repr(e)[:200].replace(tok("/tmp/hf_token"), "***"))
        if res[name]: save_state(**{f"{name}_sig": s, f"{name}_last_export": time.strftime("%F %T"), f"{name}_shards": len(files)})
    log("export", res); return 0 if all(res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
