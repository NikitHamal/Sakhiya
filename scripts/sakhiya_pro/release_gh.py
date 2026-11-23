#!/usr/bin/env python3
"""Create/update GitHub Release `sakhiya-pro-v1` and upload assets (each < 2 GB). Token read from /tmp/gitenv.sh (never committed).
usage: release_gh.py <notes.md> <asset> [<asset> ...]"""
import json, os, re, subprocess, sys, urllib.request
TAG = 'sakhiya-pro-v1'; REPO = 'NikitHamal/Sakhiya'
tok = re.findall(r'ghp_\w+|github_pat_\w+', open('/tmp/gitenv.sh').read())[0]
H = {'Authorization': f'Bearer {tok}', 'Accept': 'application/vnd.github+json', 'User-Agent': 'sakhiya'}
def api(path, data=None, method=None):
    r = urllib.request.Request('https://api.github.com' + path, data=None if data is None else json.dumps(data).encode(), headers=H, method=method)
    return json.loads(urllib.request.urlopen(r, timeout=60).read().decode() or '{}')
notes = open(sys.argv[1]).read()
try: rel = api(f'/repos/{REPO}/releases/tags/{TAG}'); api(f'/repos/{REPO}/releases/{rel["id"]}', dict(body=notes), 'PATCH')
except Exception: rel = api(f'/repos/{REPO}/releases', dict(tag_name=TAG, name='Sakhiya-Pro v1', body=notes, target_commitish='main'))
have = {a['name']: a['id'] for a in rel.get('assets', [])}
for p in sys.argv[2:]:
    n = os.path.basename(p)
    if os.path.getsize(p) >= 2 * 1024 ** 3: print('SKIP >2GB', n); continue
    if n in have: subprocess.run(['curl', '-s', '-X', 'DELETE', '-H', f'Authorization: Bearer {tok}', f'https://api.github.com/repos/{REPO}/releases/assets/{have[n]}'])
    u = f'https://uploads.github.com/repos/{REPO}/releases/{rel["id"]}/assets?name={n}'
    r = subprocess.run(['curl', '-s', '-X', 'POST', '-H', f'Authorization: Bearer {tok}', '-H', 'Content-Type: application/octet-stream', '--data-binary', f'@{p}', u], capture_output=True, text=True)
    print('uploaded', n, 'ok' if '"id"' in r.stdout else r.stdout[:200])
print(rel.get('html_url'))
