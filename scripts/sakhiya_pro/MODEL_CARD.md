---
license: other
language: [ne, sa, en]
base_model: k2-fsa/OmniVoice
pipeline_tag: text-to-speech
tags: [tts, nepali, sanskrit, devanagari, zero-shot-voice-cloning, sakhiya]
---
# Sakhiya-Pro v1 — Nepal-native anchor teacher TTS (OmniVoice-600M fine-tune)

Large GPU teacher voice for the Sakhiya project (NEBians News voiceovers; later a teacher for tiny browser models).
Full-backbone bf16 fine-tune of `k2-fsa/OmniVoice` (Qwen3-0.6B + 8-codebook audio tokens, `language="npi"`) on a single
Nepal-native female anchor (OpenSLR SLR43 speaker 6834 + Seed-VC-unified SLR43/Vedic/news clips).

## Use
```bash
pip install omnivoice  # + soundfile jiwer librosa
python sakhiya_pro_render.py --model . --text "नमस्ते, म सखिया।" --out hello.wav   # Best-of-3 + broadcast chain (-14 LUFS)
```
`anchor_ref.wav` / `anchor_ref.json` is the reference prompt (the anchor timbre). Always pass it as `ref_audio` / `ref_text`.

## Honest evaluation (synthesized audio, 8 hard sentences; automatic metrics only — no human MOS)
See `EVAL.md` in this repo for the table (base zero-shot vs fine-tune checkpoints). Metrics: Kriti Nepali CTC-ASR CER, UTMOS22,
ECAPA similarity to the anchor, Hindi-leak count, click forensics.

## Known limits
* Fine-tune data is ~1.9 h, mostly read speech (SLR43); news-style prosody comes from the base model, not from new news data.
* The base model has higher UTMOS / lower Vedic CER than the fine-tune; the fine-tune buys speaker lock (ECAPA sim 0.55 → ~0.7).
* ai4bharat/IndicF5 was gated (403); Swar (F5-Base fine-tune) represented the F5 family in the bake-off.
* No human listening test was run by the assistant; please listen to `samples/` before broadcast use.
