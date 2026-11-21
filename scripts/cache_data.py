"""Durable cache for the mined training tiers (the MoLab box is ephemeral and resets every ~30 min).
push: split feats/mix.pt (+ small json sidecars) into <=1.9 GB parts and upload them as GitHub release assets (tag data-cache).
pull: download + re-assemble them into /work/tts/feats (resumable, parallel).  The token is read from /tmp/gitenv.sh at run time (never stored).
usage: python3 scripts/cache_data.py push|pull"""
import json, os, re, subprocess, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
F = "/work/tts/feats"; REPO = "NikitHamal/Sakhiya"; TAG = "data-cache"; PART = 1900 * 1024 * 1024
SIDE = ["real_voices.json", "slr54_speakers.json", "v2_speakers.json", "sa_hours.json", "mix_hours.json"]


def tok(): return re.search(r"echo (\S+)", open("/tmp/gitenv.sh").read())[1]


def api(path, data=None, method=None, host="api.github.com", extra=()):
    r = urllib.request.Request(f"https://{host}{path}", data=data, method=method, headers={"Authorization": f"Bearer {tok()}", "Accept": "application/vnd.github+json", **dict(extra)})
    return json.load(urllib.request.urlopen(r, timeout=120))


def release():
    try: return api(f"/repos/{REPO}/releases/tags/{TAG}")
    except Exception: return api(f"/repos/{REPO}/releases", json.dumps(dict(tag_name=TAG, name="training data cache", body="Mined real-speech feature cache (mel/f0/phones). Not for end users.", prerelease=True)).encode(), "POST")


def push():
    rel = release(); have = {a["name"] for a in rel["assets"]}
    for a in rel["assets"]:                                            # drop the previous generation
        if a["name"].startswith("mix.pt.part") or a["name"] in SIDE: subprocess.run(["curl", "-s", "-X", "DELETE", "-H", f"Authorization: Bearer {tok()}", a["url"]])
    parts = sorted(os.listdir(F)); os.system(f"cd {F} && rm -f mix.pt.part* && split -b {PART} -d -a 2 mix.pt mix.pt.part")
    files = sorted(f for f in os.listdir(F) if f.startswith("mix.pt.part")) + [s for s in SIDE if os.path.exists(f"{F}/{s}")]

    def up(n):
        u = f"https://uploads.github.com/repos/{REPO}/releases/{rel['id']}/assets?name={n}"
        r = subprocess.run(["curl", "-s", "-X", "POST", "-H", f"Authorization: Bearer {tok()}", "-H", "Content-Type: application/octet-stream", "--data-binary", f"@{F}/{n}", u], capture_output=True, text=True)
        print("uploaded", n, r.stdout[:60].replace("\n", " "), flush=True)
    with ThreadPoolExecutor(4) as ex: list(ex.map(up, files))
    os.system(f"rm -f {F}/mix.pt.part*"); print("CACHE_PUSH_DONE", files, flush=True)


def pull():
    try: rel = api(f"/repos/{REPO}/releases/tags/{TAG}")
    except Exception as e: print("no cache", e); sys.exit(1)
    A = {a["name"]: a for a in rel["assets"]}; parts = sorted(n for n in A if n.startswith("mix.pt.part"))
    if not parts: print("cache empty"); sys.exit(1)
    os.makedirs(F, exist_ok=True)
    td = f"{F}/.pull_{os.getpid()}"
    os.makedirs(td, exist_ok=True)

    def dn(n):
        dst = f"{td}/{n}" if n.startswith("mix.pt.part") else f"{F}/{n}"
        subprocess.run(["curl", "-sL", "-H", f"Authorization: Bearer {tok()}", "-H", "Accept: application/octet-stream", "-o", dst, A[n]["url"]], check=True); print("got", n, flush=True)
    with ThreadPoolExecutor(6) as ex: list(ex.map(dn, parts + [s for s in SIDE if s in A]))
    os.system(f"cat {td}/mix.pt.part* > {td}/mix.pt.tmp && mv -f {td}/mix.pt.tmp {F}/mix.pt && rm -rf {td}")
    print("CACHE_PULL_DONE", flush=True)


if __name__ == "__main__": {"push": push, "pull": pull}[sys.argv[1]]()
