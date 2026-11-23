# NEBians News Ep.1 video: status

Goal: 3 to 4 minute motion-graphics episode, narrated by Sakhiya V1 (int8, `docs/models_v2`, default `news` style only), rendered on MoLab, released from this repo.

## Pipeline (all in `video/`)
1. `script.json`: Nepali narration (Devanagari) + English captions, per scene.
2. `gen_audio.py` + `music.py`: TTS clips, QC report, timeline.json, env.json (voice spectrum per frame), mix.wav.
3. `web/`: canvas motion graphics (M3X: tonal flat surfaces, no shadows, no emoji, no blue/purple gradients). Driven by timeline.json + env.json.
4. `render.py`: Playwright frame render in parallel, ffmpeg mux, writes `out/nebians-news-ep1.mp4`.
5. `release.sh`: uploads mp4 + audio + captions to a GitHub release of this repo.

## Run (MoLab, /work/Sakhiya)
`python3 video/gen_audio.py && python3 video/render.py && bash video/release.sh`

## Log
- Session start: repo + molab connected, keepalive running, fonts via npm @fontsource (Poppins, Noto Sans Devanagari).
- Script, audio generator, sound design pushed.
