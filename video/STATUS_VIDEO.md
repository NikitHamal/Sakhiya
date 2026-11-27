# NEBians News Ep.1 — status

Weekly recap, Friday 27 Nov 2026, window **Sat 21 – Fri 27 Nov 2026**. Runtime ≈ 5:57, 1080p30. **Voice only: no background music.** Voice mastered to −16 LUFS. Release is a **draft** (not posted; no links in the video; Sakhiya is not open source and is just "Sakhiya", no version names).

## Script (`script.json`) & G2P/Prosody Fixes (`sakhiya/g2p.py`, `sakhiya/onnx_tts.py`, `docs/g2p.js`)
- Intro: namaste → Sakhiya (AI news anchor) → first episode → `परिचय` (instead of `चिनारी`) → small text-to-speech model (`~23 MB` / `२३ मेगाबाइट`, no big server, phase one) → `"तर हरेक एपिसोडसँगै म अझै राम्रो हुँदै जानेछु भन्ने मेरा निर्माताहरूको आशा छ।"` → show format → `"अब लागौँ आजको पहिलो समाचारतर्फ।"`
- Proper Nepali Devanagari numerals (`२३`, `२७`, `१० लाख`, `३०`, `६`, `४`, `६.२`, `७०`, `३`, `५`, `४२`, `२१`, `१५`, `२७.३`, `१६१`, `५४`, `५.९`, `८१`, `९`) in on-screen Nepali captions (`l.ne`), automatically expanded to spoken Nepali words by `sakhiya/numbers.py`.
- Loanword-final conjunct schwa deletion (`लन्च`, `हेल्थ`, `प्रिज्म`, `भिपिएन`, `न्युयोर्कले`, `वालमार्टले`, `सेल्फ`, `इयरबड्स`, `सोर्स`), compound morpheme splitting (`अनामनगरमा`, `प्रयोगशाला`, `महानगरपालिकाले`, `सरोकारवालासँग`, `प्रतिशतभन्दा`), and heavy postposition boundary preservation (`खबरसहित` → `kʰ ʌ b ʌ r | s ʌ h i t̪`).
- Native tatsama `गत` retained as `g ʌ t̪ ʌ` (*gata*); `ग्याजेट`/`गजेट` mapped to `g j a dz e ʈ`; `क्लड सिप` (Claude Ship) fixed; acronym lexicon (`एआई`, `सीईओ`, `आईओएस`, `भिपिएन`, `एलएलएमको`, `एमएलले`, `एसएसडीबाट`, `एफसीसीले`) and cliticized numeral `छ` (`६ प्रतिशतले`) prosody guard.
- Per-sentence synthesis with ~425 ms guaranteed silence on `।` (`purna biram`), `[5, 1]` leading stop guard + sibilant-cluster prothetic-vowel trimmer (`टेक`, `स्टारमाइन्ड`), and 28 ms raised-cosine tail fade-out + 65 ms silence tail (`तर्फ`, `नमस्ते`).

## Voice clean-up (`gen_audio.py: clean()`, code only, no retraining)
High-pass 80 Hz → frequency-axis narrowband spike suppressor (2.3–11 kHz, removes metallic whistle bins on `स`/`श`/`छ`/`त`) → dual-band dynamic de-esser & burst-glare tamer (2.6–5.2 kHz and 5.0–10.2 kHz referenced to 150–1800 Hz vowel band) → −3.2 dB presence dip at 3.85 kHz + smooth high shelf above 6.2 kHz + 8th-order low-pass at 9.2 kHz → unvoiced sibilant phase dispersion above 4.5 kHz → +1.4 dB warm chest body (130–260 Hz, zero artificial white-noise comb reverb) → gentle compressor → −16 LUFS.

## Visuals (`web/`)
Material 3 Expressive on canvas: tonal palettes per scene, morphing shapes, spatial springs, shape-wipe transitions, wavy progress bar. Official NEBians mark (`web/neb-logo.png`, from the NEB brand assets) on a tonal plate in the header, intro and outro. Story scenes: headline + Nepali line + a recap rail that fills in as each point is covered (6-point Nepal education story uses a compact stepper). Compact bilingual subtitle pill with a small Sakhiya avatar.

## Social copy
`SOCIAL_CAPTIONS.md` — long, medium, short and bio captions for NEBians News + Sakhiya.

## Pipeline (MoLab)
`bash video/pipeline.sh` = `gen_audio.py` → `render.py --workers 18` → `release.sh` (draft release).
