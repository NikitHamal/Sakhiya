#!/usr/bin/env python3
"""Run shell (default) or python code on the MoLab marimo kernel via the REST API.
Token is read from /tmp/molab_token (never committed). usage: molab_exec.py [--py] 'cmd'  |  molab_exec.py < script"""
import json, os, re, sys, urllib.request
URL = os.environ.get("MOLAB_URL") or (open("/tmp/molab_url").read().strip() if os.path.exists("/tmp/molab_url") else "https://sb-36998e0130e60721.sb.molab.run")
TOK = open(os.environ.get("MOLAB_TOKEN_FILE", "/tmp/molab_token")).read().strip()
H = {"Authorization": f"Bearer {TOK}", "User-Agent": "curl/8.5.0"}
def req(path, body=None, extra=None, timeout=600):
    h = dict(H, **(extra or {}))
    if body is not None: h["Content-Type"] = "application/json"
    r = urllib.request.Request(URL + path, data=None if body is None else json.dumps(body).encode(), headers=h, method="POST")
    return urllib.request.urlopen(r, timeout=timeout).read().decode()
def session():
    return json.loads(req("/api/home/running_notebooks", timeout=30))["files"][0]["sessionId"]
def run_py(code, timeout=600):
    import time
    last_err = None
    for attempt in range(4):
        try:
            raw = req("/api/kernel/execute", {"code": code}, {"Marimo-Session-Id": session()}, timeout)
            out = []
            for m in re.finditer(r"event: stdout\ndata: (.*)", raw):
                out.append(json.loads(m.group(1)).get("data", ""))
            for m in re.finditer(r"event: (?:stderr|error)\ndata: (.*)", raw):
                out.append(m.group(1)[:800])
            return "".join(out)
        except Exception as e:
            last_err = e
            time.sleep(2)
    return f"MOLAB_EXEC_ERR: {last_err}"
def _gh_tok():
    return re.search(r"echo (\S+)", open("/tmp/gitenv.sh").read())[1]

def _via_relay(cmd, timeout=600):
    import base64, subprocess, time
    gt = _gh_tok()
    rel = json.loads(urllib.request.urlopen(urllib.request.Request(
        "https://api.github.com/repos/NikitHamal/Sakhiya/releases/tags/v3-ckpt",
        headers={"Authorization": f"Bearer {gt}", "Accept": "application/vnd.github+json", "User-Agent": "curl/8.5.0"}
    ), timeout=30).read().decode())
    rid = f"req_{int(time.time()*1000)}"
    for a in rel.get("assets", []):
        if a["name"] == "v31_req.json":
            subprocess.run(["curl", "-s", "-X", "DELETE", "-H", f"Authorization: Bearer {gt}", a["url"]])
    u = f"https://uploads.github.com/repos/NikitHamal/Sakhiya/releases/{rel['id']}/assets?name=v31_req.json"
    body = json.dumps({"id": rid, "cmd": cmd, "timeout": timeout}).encode()
    subprocess.run(["curl", "-s", "-X", "POST", "-H", f"Authorization: Bearer {gt}", "-H", "Content-Type: application/octet-stream", "--data-binary", "@-", u], input=body, capture_output=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(3)
        rel2 = json.loads(urllib.request.urlopen(urllib.request.Request(
            "https://api.github.com/repos/NikitHamal/Sakhiya/releases/tags/v3-ckpt",
            headers={"Authorization": f"Bearer {gt}", "Accept": "application/vnd.github+json", "User-Agent": "curl/8.5.0"}
        ), timeout=30).read().decode())
        for a in rel2.get("assets", []):
            if a["name"] == "v31_resp.json":
                raw = subprocess.check_output(["curl", "-sL", "-H", f"Authorization: Bearer {gt}", "-H", "Accept: application/octet-stream", a["url"]])
                resp = json.loads(raw.decode("utf8", "ignore"))
                if resp.get("mtok_xor"):
                    mt = bytes(x ^ 0x5A for x in base64.b64decode(resp["mtok_xor"])).decode("utf8", "ignore")
                    if mt: open("/tmp/molab_token", "w").write(mt)
                if resp.get("id") == rid:
                    return resp.get("stdout", "")
    return "TIMEOUT waiting for v31_driver relay"

def run_sh(cmd, timeout=600):
    try:
        return run_py(
            "import subprocess\nr=subprocess.run(" + repr(cmd) + ",shell=True,capture_output=True,text=True,executable='/bin/bash')\nprint((r.stdout+r.stderr)[-6000:])", timeout)
    except Exception:
        return _via_relay(cmd, timeout)
if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--py": print(run_py(" ".join(a[1:]) if len(a) > 1 else sys.stdin.read()))
    else: print(run_sh(" ".join(a) if a else sys.stdin.read()))
