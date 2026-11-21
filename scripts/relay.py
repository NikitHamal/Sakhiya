"""Checkpoint relay across ephemeral MoLab boxes (they reset every ~30 min).
push: fp16-pack the newest AM / vocoder checkpoints from runs/ and upload them to the GitHub release `ckpt-wip`.
pull: download them into /work/tts/relay/ (bootstrap_box.sh then warm-starts from them instead of the shipped weights).
watch: push every 60 s whenever the source checkpoints changed.  Token is read from /tmp/gitenv.sh at run time only.
usage: python3 scripts/relay.py push|pull|watch [tag]"""
import glob, json, os, re, subprocess, sys, time, urllib.request
import torch
REPO = "NikitHamal/Sakhiya"; TAG = "ckpt-wip"; R = "/work/tts"; OUT = f"{R}/relay"


def tok(): return re.search(r"echo (\S+)", open("/tmp/gitenv.sh").read())[1]


def api(path, data=None, method=None):
    r = urllib.request.Request(f"https://api.github.com{path}", data=data, method=method, headers={"Authorization": f"Bearer {tok()}", "Accept": "application/vnd.github+json"})
    return json.load(urllib.request.urlopen(r, timeout=120))


def release():
    try: return api(f"/repos/{REPO}/releases/tags/{TAG}")
    except Exception: return api(f"/repos/{REPO}/releases", json.dumps(dict(tag_name=TAG, name="checkpoint relay", body="Latest in-flight AM/vocoder weights (fp16) so a reset box can warm-start. Not for end users.", prerelease=True)).encode())


def newest(pat):
    fs = glob.glob(pat); return max(fs, key=os.path.getmtime) if fs else None


def sources():
    return {
        "am.fp16.pt": newest(f"{R}/runs/am*/last.pt"),
        "G.fp16.pt": newest(f"{R}/runs/voc*/G.pt"),
        "v2_am.fp16.pt": newest(f"{R}/runs/v2_am*/last.pt"),
        "v2_G.fp16.pt": newest(f"{R}/runs/v2_voc*/G_best.pt"),
    }


def push(meta=""):
    rel = release(); n = 0
    for name, src in sources().items():
        if not src: continue
        tmp = f"/tmp/relay_{name}"
        c = torch.load(src, weights_only=False, map_location="cpu")
        if "M" in c:
            c["M"] = {k: v.half() if torch.is_tensor(v) and v.is_floating_point() else v for k, v in c["M"].items()}
            torch.save(c, tmp)
        elif "ema" in c or "net" in c:
            sd = c.get("ema", c.get("net"))
            torch.save(dict(ema={k: v.half() if torch.is_tensor(v) and v.is_floating_point() else v for k, v in sd.items()},
                            step=c.get("step", 0), stage=c.get("stage", "A"), mean=c.get("mean", -6.177), std=c.get("std", 2.492), pitch_trained=c.get("pitch_trained", True)), tmp)
        else:
            torch.save({k: v.half() if torch.is_tensor(v) and v.is_floating_point() else v for k, v in c.items()}, tmp)
        for a in rel["assets"]:
            if a["name"] == name: subprocess.run(["curl", "-s", "-X", "DELETE", "-H", f"Authorization: Bearer {tok()}", a["url"]])
        u = f"https://uploads.github.com/repos/{REPO}/releases/{rel['id']}/assets?name={name}"
        r = subprocess.run(["curl", "-s", "-X", "POST", "-H", f"Authorization: Bearer {tok()}", "-H", "Content-Type: application/octet-stream", "--data-binary", f"@{tmp}", u], capture_output=True, text=True)
        print("relay up", name, src, os.path.getsize(tmp), r.stdout[:40].replace("\n", " "), flush=True); n += 1
    return n


def pull():
    rel = api(f"/repos/{REPO}/releases/tags/{TAG}"); os.makedirs(OUT, exist_ok=True)
    for a in rel["assets"]:
        subprocess.run(["curl", "-sL", "-H", f"Authorization: Bearer {tok()}", "-H", "Accept: application/octet-stream", "-o", f"{OUT}/{a['name']}", a["url"]], check=True)
        print("relay got", a["name"], os.path.getsize(f"{OUT}/{a['name']}"), flush=True)


def watch():
    seen = {}
    while True:
        cur = {k: (v, os.path.getmtime(v)) for k, v in sources().items() if v}
        if cur and cur != seen:
            try: push(); seen = cur
            except Exception as e: print("relay err", e, flush=True)
        time.sleep(60)


if __name__ == "__main__": {"push": push, "pull": pull, "watch": watch}[sys.argv[1]]()
