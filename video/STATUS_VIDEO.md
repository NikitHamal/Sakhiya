# NEBians News Ep.1 — status

Weekly recap, Friday 27 Nov 2026, window **Sat 21 – Fri 27 Nov 2026**. Runtime ≈ 5:57, 1080p30. **Voice only: no background music.** Voice mastered to −16 LUFS. Release is a **draft** (not posted; no links in the video; Sakhiya is not open source and is just "Sakhiya", no version names).

## Script (`script.json`)
- Intro: namaste → Sakhiya (AI news anchor) → first episode → small text-to-speech model, simple facts only (Nepali, ~23 MB, no big server, phase one) → show format → "अब लागौँ आजको पहिलो समाचारतर्फ।"
- Plain Nepali with common English tech words kept as people say them (डाटा लिक, ह्याक, ब्यान, लन्च, रोलआउट, ग्याजेट...). No literal Nepali translations of tech terms. No "चलो".
- `pron` map in `script.json`: TTS-only respellings (e.g. गजेट → ग्याजेट) applied in `gen_audio.py`; subtitles keep the original text.

## Voice clean-up (`gen_audio.py: clean()`, code only, no retraining)
High-pass 75 Hz → STFT split-band de-esser (4.8–11.5 kHz vs 0.2–3.5 kHz, ≤9 dB) → −2.2 dB dip at ~3.9 kHz (metallic band) → soft air shelf and steep low-pass ~10.5 kHz (removes the 12–14 kHz vocoder imaging band) → +body at 130–260 Hz → tiny synthetic room (7% wet) → gentle compressor → −16 LUFS.

## Visuals (`web/`)
Material 3 Expressive on canvas: tonal palettes per scene, morphing shapes, spatial springs, shape-wipe transitions, wavy progress bar. Official NEBians mark (`web/neb-logo.png`, from the NEB brand assets) on a tonal plate in the header, intro and outro. Story scenes: headline + Nepali line + a recap rail that fills in as each point is covered (6-point Nepal education story uses a compact stepper). Compact bilingual subtitle pill with a small Sakhiya avatar.

## Social copy
`SOCIAL_CAPTIONS.md` — long, medium, short and bio captions for NEBians News + Sakhiya.

## Pipeline (MoLab)
`bash video/pipeline.sh` = `gen_audio.py` → `render.py --workers 18` → `release.sh` (draft release).
