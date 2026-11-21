"""Sakhiya-v3 40-Minute Checkpoint Survival Relay & Milestone Exporter.
Uploads fp16 checkpoints (`v3_am.fp16.pt`, `v3_voc.fp16.pt`, `v3_ctc.fp16.pt`) to GitHub Release `v3-ckpt`
every 40 minutes so zero GPU hours are lost when the 12-hour box expires.
usage: python3 scripts/v3_relay_and_gate.py [push|pull|watch]"""
import glob, json, os, re, subprocess, sys, time, urllib.request
import torch

REPO = "NikitHamal/Sakhiya"
TAG = "v3-ckpt"
R = "/work/tts"


def tok():
    return re.search(r"echo (\S+)", open("/tmp/gitenv.sh").read())[1]


def api(path, data=None, method=None):
    r = urllib.request.Request(
        f"https://api.github.com{path}",
        data=data,
        method=method,
        headers={"Authorization": f"Bearer {tok()}", "Accept": "application/vnd.github+json"},
    )
    return json.load(urllib.request.urlopen(r, timeout=120))


def release():
    try:
        return api(f"/repos/{REPO}/releases/tags/{TAG}")
    except Exception:
        body = json.dumps(dict(
            tag_name=TAG,
            name="Sakhiya v3 Checkpoint Survival Relay",
            body="Automated 40-min fp16 checkpoint archive (v3_am, v3_voc, v3_ctc) from RTX PRO 6000 Blackwell.",
            prerelease=True,
        )).encode()
        return api(f"/repos/{REPO}/releases", body, "POST")


def sources():
    voc_best = f"{R}/runs/v3_voc_3b/G_best.pt" if os.path.exists(f"{R}/runs/v3_voc_3b/G_best.pt") else f"{R}/runs/v3_voc/G_best.pt"
    voc_last = f"{R}/runs/v3_voc_3b/last.pt" if os.path.exists(f"{R}/runs/v3_voc_3b/last.pt") else f"{R}/runs/v3_voc/last.pt"
    return {
        "v3_am.fp16.pt": f"{R}/runs/v3_am/last.pt",
        "v3_voc.fp16.pt": voc_best,
        "v3_voc_last.fp16.pt": voc_last,
        "v3_ctc.fp16.pt": f"{R}/feats/v3_ctc_recognizer.pt",
    }


def push():
    rel = release()
    uploaded = []
    for name, src in sources().items():
        if not src or not os.path.exists(src):
            continue
        tmp = f"/tmp/relay_{name}"
        c = torch.load(src, weights_only=False, map_location="cpu")
        if isinstance(c, dict) and ("ema" in c or "net" in c):
            sd = c.get("ema", c.get("net"))
            torch.save(dict(
                ema={k: v.half() if torch.is_tensor(v) and v.is_floating_point() else v for k, v in sd.items()},
                step=c.get("step", 0), mean=c.get("mean", -6.177), std=c.get("std", 2.492),
                pre_baked_v25=True, version="3.0",
            ), tmp)
        elif isinstance(c, dict) and "G_ema" in c:
            sd = c["G_ema"]
            torch.save(dict(
                G_ema={k: v.half() if torch.is_tensor(v) and v.is_floating_point() else v for k, v in sd.items()},
                step=c.get("step", 0), val_l1=c.get("val_l1", 0.0),
                utmos_F=c.get("utmos_F", 0.0), utmos_M=c.get("utmos_M", 0.0),
            ), tmp)
        elif isinstance(c, dict):
            torch.save({k: (v.half() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in c.items()}, tmp)
        for a in rel.get("assets", []):
            if a["name"] == name:
                subprocess.run(["curl", "-s", "-X", "DELETE", "-H", f"Authorization: Bearer {tok()}", a["url"]])
        u = f"https://uploads.github.com/repos/{REPO}/releases/{rel['id']}/assets?name={name}"
        subprocess.run(["curl", "-s", "-X", "POST", "-H", f"Authorization: Bearer {tok()}",
                        "-H", "Content-Type: application/octet-stream", "--data-binary", f"@{tmp}", u],
                       capture_output=True, text=True)
        uploaded.append(name)
        print("v3-ckpt relay uploaded:", name, os.path.getsize(tmp), flush=True)
    if uploaded:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            p = f"{R}/COORD_V3.json"
            cd = json.load(open(p)) if os.path.exists(p) else {}
            cd["last_release_ckpt_utc"] = ts
            cd["last_release_assets"] = uploaded
            json.dump(cd, open(p, "w"), indent=2)
        except Exception:
            pass
    return uploaded


def pull():
    rel = api(f"/repos/{REPO}/releases/tags/{TAG}")
    os.makedirs(f"{R}/relay_v3", exist_ok=True)
    for a in rel.get("assets", []):
        dst = f"{R}/relay_v3/{a['name']}"
        subprocess.run(["curl", "-sL", "-H", f"Authorization: Bearer {tok()}", "-H", "Accept: application/octet-stream", "-o", dst, a["url"]], check=True)
        print("v3-ckpt pulled:", a["name"], os.path.getsize(dst), flush=True)


def watch(interval_s=2400):
    seen = {}
    while True:
        cur = {k: os.path.getmtime(v) for k, v in sources().items() if v and os.path.exists(v)}
        if cur and cur != seen:
            try:
                push()
                seen = cur
            except Exception as e:
                print("v3 relay err:", e, flush=True)
        time.sleep(interval_s)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "push"
    {"push": push, "pull": pull, "watch": watch}[cmd]()
