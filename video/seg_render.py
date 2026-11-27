#!/usr/bin/env python3
"""Resumable render: 30 s segments are encoded and uploaded to the `news-ep1-wip` release as they finish,
so a MoLab restart never loses work. Re-run and finished segments are downloaded instead of re-rendered."""
import json, math, os, shutil, subprocess, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
import render as R
REPO, TAG, SEG, FPS = "NikitHamal/Sakhiya", os.environ.get("WIP_TAG", "news-ep1-v2-wip"), 900, 30
TOK = subprocess.check_output([os.environ.get("GIT_ASKPASS", "/tmp/gitenv.sh")]).decode().strip()
H = ["-H", f"Authorization: token {TOK}"]
def curl(*a): return subprocess.check_output(["curl", "-s", "-L", *H, *a])
def rel():
    r = json.loads(curl(f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}") or b"{}")
    if "id" not in r:
        r = json.loads(curl("-X", "POST", f"https://api.github.com/repos/{REPO}/releases", "-d", json.dumps(dict(tag_name=TAG, name="NEBians News Ep.1 work in progress", prerelease=True, target_commitish="main"))))
    return r
def main():
    R.prep(); tl = json.load(open(R.B + "/timeline.json")); n = int(tl["total"] * FPS) + 1
    S = R.V + "/out/segs"; os.makedirs(S, exist_ok=True); r = rel(); have = {a["name"]: a for a in r.get("assets", [])}
    wk = int(os.environ.get("WORKERS", 12))
    for s in range(math.ceil(n / SEG)):
        name = f"seg_{s:02d}.mp4"; path = f"{S}/{name}"; a0, a1 = s * SEG, min(n, (s + 1) * SEG)
        if name in have:
            open(path, "wb").write(curl("-H", "Accept: application/octet-stream", have[name]["url"])); print("downloaded", name, flush=True); continue
        shutil.rmtree(R.FR, ignore_errors=True); os.makedirs(R.FR)
        chunks = [[(i - a0, round(i / FPS, 4)) for i in range(a0 + k, a1, wk)] for k in range(wk)]
        with Pool(wk) as p: errs = p.map(R.work, [(c, k) for k, c in enumerate(chunks)])
        print("seg", s, "errors:", [e for e in errs if e][:2], flush=True)
        subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", R.FR + "/%05d.jpg", "-c:v", "libx264", "-preset", "medium", "-crf", "16",
            "-pix_fmt", "yuv420p", "-g", "30", "-r", str(FPS), path])
        up = curl("-X", "POST", "-H", "Content-Type: application/octet-stream", "--data-binary", "@" + path, f"https://uploads.github.com/repos/{REPO}/releases/{r['id']}/assets?name={name}")
        print("uploaded", name, len(up), flush=True)
    open(S + "/list.txt", "w").write("".join(f"file '{S}/seg_{s:02d}.mp4'\n" for s in range(math.ceil(n / SEG))))
    os.makedirs(R.V + "/out", exist_ok=True)
    subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", S + "/list.txt", "-i", R.B + "/mix.wav", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart", R.V + "/out/nebians-news-ep1.mp4"])
    print("done", os.path.getsize(R.V + "/out/nebians-news-ep1.mp4") // 1024, "KB")
if __name__ == "__main__":
    main()
