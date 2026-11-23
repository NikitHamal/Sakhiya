#!/bin/bash
# Publish NEBians News Ep.1 as a GitHub release of NikitHamal/Sakhiya. Token is read from the GIT_ASKPASS helper, never stored.
set -e
cd "$(dirname "$0")"; TAG=${TAG:-news-ep1}; REPO=NikitHamal/Sakhiya
TOK=$(${GIT_ASKPASS:-/tmp/gitenv.sh}); B=build; O=out
python3 - <<'EOF'
import json
tl = json.load(open("build/timeline.json")); f = lambda t: "%02d:%02d:%02d,%03d" % (t // 3600, t % 3600 // 60, t % 60, int(t % 1 * 1000))
out = []; n = 1
for s in tl["scenes"]:
    for l in s["lines"]:
        out.append(f"{n}\n{f(l['t'])} --> {f(l['t'] + l['d'])}\n{l['ne']}\n{l['en']}\n"); n += 1
open("out/nebians-news-ep1.srt", "w").write("\n".join(out))
EOF
ffmpeg -y -loglevel error -i $B/voice.wav -c:a libmp3lame -b:a 160k $O/sakhiya-voice-only.mp3
ffmpeg -y -loglevel error -ss 4 -i $O/nebians-news-ep1.mp4 -frames:v 1 $O/poster.jpg
NOTES="NEBians News, episode 01: Meet Sakhiya. 3:47, 1080p30. Voice: Sakhiya V1 int8 (docs/models_v2), default news style only, no other styles. Early phase model, trained for a few days. News as reported on 23 Nov 2026 (CNBC, Yahoo Finance, Indian Express, FT, AI Weekly). Source: video/ in this repo. Rendered on MoLab (canvas + Playwright + ffmpeg)."
API=https://api.github.com/repos/$REPO/releases
OLD=$(curl -s -H "Authorization: token $TOK" "$API/tags/$TAG" | python3 -c "import sys,json;print(json.load(sys.stdin).get('id',''))")
[ -n "$OLD" ] && curl -s -X DELETE -H "Authorization: token $TOK" "$API/$OLD" >/dev/null
curl -s -X DELETE -H "Authorization: token $TOK" "https://api.github.com/repos/$REPO/git/refs/tags/$TAG" >/dev/null || true
ID=$(curl -s -X POST -H "Authorization: token $TOK" $API -d "$(python3 -c "import json,sys;print(json.dumps(dict(tag_name='$TAG',name='NEBians News Ep.1: Meet Sakhiya',body=sys.argv[1],target_commitish='main')))" "$NOTES")" | python3 -c "import sys,json;print(json.load(sys.stdin)['id'])")
for f in nebians-news-ep1.mp4 nebians-news-ep1.srt sakhiya-voice-only.mp3 poster.jpg; do
  curl -s -X POST -H "Authorization: token $TOK" -H "Content-Type: application/octet-stream" --data-binary @$O/$f "https://uploads.github.com/repos/$REPO/releases/$ID/assets?name=$f" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('name'),d.get('size'),d.get('browser_download_url'))"
done
