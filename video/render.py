#!/usr/bin/env python3
"""Render NEBians News Ep.1. usage: render.py [--preview t1,t2,...] [--workers N] [--fps 30]"""
import json, os, shutil, subprocess, sys, time
from multiprocessing import Pool
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
V = os.path.join(ROOT, "video"); B = V + "/build"; WEB = V + "/web"; FR = B + "/frames"
FONTS = os.environ.get("FONTS_DIR", "/work/fonts/node_modules/@fontsource")

def prep():
    os.makedirs(WEB + "/fonts", exist_ok=True)
    for pkg in ("poppins", "noto-sans-devanagari"):
        d = f"{FONTS}/{pkg}/files"
        for f in os.listdir(d):
            if f.endswith("-normal.woff2") and any(f"-{w}-" in f for w in (400, 500, 600, 700)) and ("latin-" in f or "devanagari-" in f) and "ext" not in f:
                shutil.copy(f"{d}/{f}", WEB + "/fonts/" + f)
    tl = open(B + "/timeline.json").read(); env = open(B + "/env.json").read()
    open(WEB + "/data.js", "w").write(f"window.TL={tl};window.ENV={env};")

def work(args):
    idx, wid = args
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        br = p.chromium.launch(args=["--allow-file-access-from-files", "--disable-gpu", "--font-render-hinting=none"])
        pg = br.new_page(viewport={"width": 1920, "height": 1080}); errs = []
        pg.on("pageerror", lambda e: errs.append(str(e))); pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.goto("file://" + WEB + "/index.html"); pg.evaluate("window.ready")
        for n, T in idx:
            try: pg.evaluate(f"window.render({T})")
            except Exception as e: errs.append(f"T={T}: {e}")
            pg.locator("#cv").screenshot(path=f"{FR}/{n:05d}.jpg", type="jpeg", quality=93)
        br.close()
    return errs[:5]

def main():
    a = sys.argv[1:]; fps = 30; wk = 18
    if "--workers" in a: wk = int(a[a.index("--workers") + 1])
    prep(); tl = json.load(open(B + "/timeline.json")); total = tl["total"]
    if "--preview" in a:
        ts = [float(x) for x in a[a.index("--preview") + 1].split(",")]; shutil.rmtree(FR, ignore_errors=True); os.makedirs(FR)
        jobs = [([(i, t)], i) for i, t in enumerate(ts)]
        with Pool(min(len(jobs), 10)) as p: errs = p.map(work, jobs)
        print("errors:", [e for e in errs if e])
        from PIL import Image
        cols = 3; w, h = 640, 360; rows = (len(ts) + cols - 1) // cols; sheet = Image.new("RGB", (cols * w, rows * h))
        for i in range(len(ts)): sheet.paste(Image.open(f"{FR}/{i:05d}.jpg").resize((w, h)), ((i % cols) * w, (i // cols) * h))
        sheet.save(B + "/preview.jpg", quality=80); return
    n = int(total * fps) + 1; shutil.rmtree(FR, ignore_errors=True); os.makedirs(FR); t0 = time.time()
    chunks = [[(i, round(i / fps, 4)) for i in range(k, n, wk)] for k in range(wk)]
    with Pool(wk) as p: errs = p.map(work, [(c, k) for k, c in enumerate(chunks)])
    print("frames", n, "in", round(time.time() - t0), "s; errors:", [e for e in errs if e])
    out = V + "/out"; os.makedirs(out, exist_ok=True)
    subprocess.check_call(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", FR + "/%05d.jpg", "-i", B + "/mix.wav", "-c:v", "libx264", "-preset", "slow",
        "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out + "/nebians-news-ep1.mp4"])
    print("done", os.path.getsize(out + "/nebians-news-ep1.mp4") // 1024, "KB")

if __name__ == "__main__":
    main()
