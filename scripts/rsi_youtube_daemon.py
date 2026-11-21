#!/usr/bin/env python3
"""Overnight resumable daemon: mine -> separate+segment -> gate -> cluster -> ASR/align -> punctuate+pack -> archive (HF private + GH release) -> VC -> fine-tune hook.
State: /work/tts/mining_state.json (stage, cycle, counts).  Launch:  setsid nohup python3 scripts/rsi_youtube_daemon.py > /work/tts/logs/daemon.log 2>&1 < /dev/null &
Secrets are read from /tmp/hf_token, /tmp/gitenv.sh, /tmp/kaggle_token (never logged or committed). A fresh box resumes: HF dataset is pulled back if local data is missing."""
import glob, json, os, re, subprocess, sys, time

ROOT = os.environ.get("TTS_ROOT", "/work/tts"); REPO = f"{ROOT}/repo"; ST = f"{ROOT}/mining_state.json"; HF_REPO = "NikitHamal/sakhiya-nepali-vedic-private"
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def sh(cmd, bg=False, **k):
    log("$", cmd[:140]); p = subprocess.Popen(cmd, shell=True, cwd=REPO, executable="/bin/bash", **k)
    return p if bg else p.wait()
def state(**kw):
    s = json.load(open(ST)) if os.path.exists(ST) else {}
    s.update(kw, ts=time.strftime("%Y-%m-%d %H:%M:%S")); json.dump(s, open(ST, "w"), ensure_ascii=False, indent=1); return s
def alive(pat): return subprocess.run(f"pgrep -f '{pat[0]}{pat[1:]}' | grep -v $$ | wc -l", shell=True, capture_output=True, text=True).stdout.strip() != "0"

def hf_api():
    from huggingface_hub import HfApi
    return HfApi(token=open("/tmp/hf_token").read().strip())

def archive(cycle):
    """Private HF dataset + private Kaggle dataset (scripts/export_private.py, incremental) + v6_state.json."""
    r = sh(f"python3 scripts/export_private.py --cycle {cycle} >> {ROOT}/logs/export.log 2>&1")
    log("private export", "ok" if r == 0 else f"rc={r} (see logs/export.log)"); return r == 0

def gh_release(cycle):
    try:
        tok = re.search(r"echo (\S+)", open("/tmp/gitenv.sh").read())[1]; H = ["-H", f"Authorization: Bearer {tok}", "-H", "User-Agent: curl/8.5.0"]
        rel = json.loads(subprocess.run(["curl", "-s", *H, "https://api.github.com/repos/NikitHamal/Sakhiya/releases/tags/v6-data"], capture_output=True, text=True).stdout)
        if "id" not in rel: rel = json.loads(subprocess.run(["curl", "-s", "-X", "POST", *H, "https://api.github.com/repos/NikitHamal/Sakhiya/releases", "-d", json.dumps({"tag_name": "v6-data", "name": "v6-data", "body": "Sakhiya YouTube/open-web Nepali+Vedic TTS data (meta/shards)", "prerelease": True})], capture_output=True, text=True).stdout)
        have = {a["name"] for a in rel.get("assets", [])}
        for f in [f"{ROOT}/dataset/meta.jsonl", ST] + sorted(glob.glob(f"{ROOT}/dataset/vc/*.tar"))[-3:]:
            nm = os.path.basename(f)
            if os.path.exists(f) and os.path.getsize(f) < 1.9e9 and (nm not in have or nm in ("meta.jsonl", "mining_state.json")):
                for a in rel.get("assets", []):
                    if a["name"] == nm: subprocess.run(["curl", "-s", "-X", "DELETE", *H, a["url"]])
                subprocess.run(["curl", "-s", "-X", "POST", *H, "-H", "Content-Type: application/octet-stream", "--data-binary", f"@{f}", f"https://uploads.github.com/repos/NikitHamal/Sakhiya/releases/{rel['id']}/assets?name={nm}"], capture_output=True)
        log("GH release v6-data ok")
    except Exception as e: log("GH release FAIL", repr(e)[:150])

def resume_if_empty():
    if not os.path.isdir(f"{ROOT}/dataset") or not glob.glob(f"{ROOT}/dataset/raw/*.parquet"):
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(HF_REPO, repo_type="dataset", local_dir=f"{ROOT}/dataset_hf", token=open("/tmp/hf_token").read().strip()); log("resumed dataset from HF")
        except Exception as e: log("no HF snapshot yet", repr(e)[:100])

LAST = [0.0]

def cycle(n):
    state(stage="mining", cycle=n); mf = f"{ROOT}/logs/mining.finished"
    if os.path.exists(mf) and n > 0: os.remove(mf)
    env = f"CAP_SCALE={1 + 0.6 * n} ROWS_SCALE={1 + n} "
    if not alive("mine_archive_org") and not os.path.exists(mf): sh(env + f"python3 scripts/mine_archive_org.py --max-hours {260 + 120 * n} --workers 12 >> {ROOT}/logs/mine_ia.log 2>&1", bg=True)
    if not alive("yt_stage2_loop"): sh(f"bash scripts/yt_stage2_loop.sh 6 > {ROOT}/logs/stage2_loop.log 2>&1", bg=True)
    for it in range(200):
        time.sleep(240); state(stage="gate")
        ps = [sh(f"HF_TOKEN=$(cat /tmp/hf_token) python3 scripts/gpucap.py 0.09 scripts/yt_gate_speakers.py --score --shard {i}/4 >> {ROOT}/logs/gate_{i}.log 2>&1", bg=True) for i in range(4) if not alive(f"yt_gate_speakers.py --score --shard {i}/4")]
        while alive("yt_gate_speakers.py --score"):
            time.sleep(20)
            if time.time() - LAST[0] > 1500: LAST[0] = time.time(); process_new(n)
        if not alive("mine_archive_org"): open(mf, "w").write("1")
        if os.path.exists(mf) and not alive("yt_stage2_loop"): break
    process_new(n)

def process_new(n):
    state(stage="gate-filter"); sh("python3 scripts/yt_gate_speakers.py --cluster --filter --utmos 2.6 --utmos-chant 2.4 >> logs.tmp 2>&1".replace("logs.tmp", f"{ROOT}/logs/filter.log"))
    state(stage="asr"); sh(f"HF_TOKEN=$(cat /tmp/hf_token) python3 scripts/gpucap.py 0.28 scripts/yt_asr_ensemble.py >> {ROOT}/logs/asr.log 2>&1")
    state(stage="pack"); sh(f"python3 scripts/yt_prosody_pack.py --agree-silver 0.80 >> {ROOT}/logs/pack.log 2>&1")
    if os.path.exists(f"{REPO}/scripts/yt_vc_sakhiya.py"): state(stage="vc"); sh(f"python3 scripts/gpucap.py 0.15 scripts/yt_vc_sakhiya.py >> {ROOT}/logs/vc.log 2>&1")
    state(stage="archive"); archive(n); gh_release(n)
    try:
        nm = sum(1 for _ in open(f"{ROOT}/dataset/meta.jsonl")) if os.path.exists(f"{ROOT}/dataset/meta.jsonl") else 0
        if nm >= 40 and not os.path.exists(f"{ROOT}/logs/samples.published"): sh("python3 scripts/publish_yt_samples.py") == 0 and open(f"{ROOT}/logs/samples.published", "w").write("1")
    except Exception as e: log("samples FAIL", repr(e)[:150])
    if os.environ.get("FT_CMD"): state(stage="finetune"); sh(os.environ["FT_CMD"] + f" >> {ROOT}/logs/finetune.log 2>&1")

if __name__ == "__main__":
    os.makedirs(f"{ROOT}/logs", exist_ok=True); resume_if_empty(); n = state().get("cycle", 0) if os.path.exists(ST) else 0
    while True:
        try: cycle(n)
        except Exception as e: log("cycle error", repr(e)[:300])
        n += 1; state(stage="sleep", cycle=n); time.sleep(600)
