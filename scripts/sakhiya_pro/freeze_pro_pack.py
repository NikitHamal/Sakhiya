#!/usr/bin/env python3
"""Freeze the Sakhiya-Pro 15-clip master comparison pack into docs/samples/versions/sakhiya_pro/ and register it.
usage: freeze_pro_pack.py <pack_dir_from_render_pack> <metrics.json> [--notes "..."]
Immutable: refuses to overwrite once `.frozen` exists."""
import argparse, datetime, json, os, shutil, subprocess
ap = argparse.ArgumentParser(); ap.add_argument('pack'); ap.add_argument('metrics'); ap.add_argument('--notes', default=''); ap.add_argument('--tag', default='sakhiya_pro'); ap.add_argument('--label', default='Sakhiya-Pro (OmniVoice-600M fine-tune · broadcast teacher, GPU)')
a = ap.parse_args()
TAG = a.tag; out = os.path.join('docs', 'samples', 'versions', TAG)
if os.path.exists(os.path.join(out, '.frozen')): raise SystemExit(f'[IMMUTABLE GUARD] {out} already frozen')
os.makedirs(out, exist_ok=True)
meta = json.load(open(os.path.join(a.pack, 'pack_meta.json'))); clips = []
for m in meta:
    shutil.copy(os.path.join(a.pack, m['mp3']), os.path.join(out, m['mp3']))
    clips.append(dict(id=m['id'], lang=m['lang'], style=m['style'], text=m['text'], mp3=f'samples/versions/{TAG}/{m["mp3"]}', duration_s=m['duration_s']))
sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], text=True).strip()
now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
mm = json.load(open(a.metrics))
entry = dict(tag=TAG, label=a.label, commit_sha=sha, timestamp_utc=now, notes=a.notes,
             metrics=mm, clips=clips)
json.dump(entry, open(os.path.join(out, 'manifest.json'), 'w'), indent=2, ensure_ascii=False)
open(os.path.join(out, '.frozen'), 'w').write(f'Frozen at {now} commit {sha}\n')
rp = os.path.join('docs', 'samples', 'versions', 'registry.json'); reg = json.load(open(rp))
reg['versions'] = [v for v in reg['versions'] if v['tag'] != TAG] + [entry]
json.dump(reg, open(rp, 'w'), indent=2, ensure_ascii=False)
print('FROZEN', TAG, len(clips), 'clips')
