# NEBians News Ep.1 — status

Weekly recap, Friday 27 Nov 2026, window **Sat 21 – Fri 27 Nov 2026**. Runtime ≈ 5:50, 1080p30. **Voice only: no background music.** Voice mastered to −16 LUFS. Release is a **draft** (not posted; no links shown in the video, Sakhiya is not open source).

## Script (`script.json`)
- Intro rewritten: namaste → Sakhiya (AI news anchor) → welcome to the first episode → small text-to-speech model, simple tech facts only (Nepali, ~23 MB, no big server, phase one, getting better) → show format → "अब लागौँ आजको पहिलो समाचारतर्फ।"
- Plain Nepali with common English words (स्याटेलाइट, सीईओ, एपिसोड). No "चलो", no model version names.
- Stories: Space & AI, Chips, AI Models, Security & Policy, Devices & Software, Nepal Education, Quick Roundup. Each line carries a `big`/`label` fact used by the visuals.

## Visuals (`web/`: `m3.js`, `parts.js`, `scenes.js`, `main.js`)
Clean Material 3 Expressive: tonal palettes from one seed hue per scene (light/dark), morphing shapes, spatial springs, shape-wipe transitions, wavy progress bar, one fact per spoken line. Compact bilingual subtitle pill (Nepali + English, sentence-aware pages) with a small Sakhiya avatar.

## Pipeline (MoLab)
`python3 video/gen_audio.py` → `python3 video/render.py --workers 18` → `bash video/release.sh` (draft release).
