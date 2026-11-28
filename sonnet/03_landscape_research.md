# 03 · Landscape research: what the best tiny TTS systems actually do (researched Nov 2026)

Sources: model cards/READMEs fetched during this session (tiny-voca, tiny-voca-hindi, Alpa, Indic Parler-TTS, Indic Tiny TTS, Rasa dataset card,
Kitten TTS / TinyTTS write-ups). Numbers are the authors' own claims unless marked *(measured here)*.

## 1. The model the owner is thinking of: **tiny-voca** (n4ze3m, Apache-2.0)

* **4,537,291 parameters total**, two voices (F2/M2), 24 kHz, 18 MB FP32, ≈70× real-time on 4 CPU threads, ONNX + onnxruntime-web, runs even on ESP32-S3.
* **Open TTS Leaderboard (20 Nov 2026, English, no cloning): F2 rank 5 by WER (0.74 %), UTMOS22 4.43** — ahead of 100 M-parameter models. Total cost **$9.56** (RTX 4090, 7.7 h).
* Architecture (FastPitch-family, but fully explicit and *end-to-end differentiable*):
  `phonemes+stress+voice → 4-block text encoder (192 ch, 4 heads, RoPE) → duration & pause predictor → length-regulate → frame transformer A (2 blocks) → pitch/voicing/energy → frame transformer B (5 blocks) → per-frame cepstral filter params → **source-filter synthesizer** (band-limited pulse train + noise, shaped by predicted time-varying filters; 25,862 params)`. No neural vocoder. Hop 256 (10.7 ms).
* Training: 16,018 clips / 17.6 h, **104.7 K steps**; spectral (multi-res STFT) losses for 8 K steps, then MPD+MRD adversarial losses; discriminators exist only in training (2.7 M params, discarded).
* **Key insight (their own limitation section): “the training data is itself synthetic, so the model cannot be more natural than the model that generated it.”** Two perfectly *consistent* voices from a strong teacher + validated forced alignment is why 4.5 M suffices. Consistency of target, not model size, is the lever.
* Timing lesson (annotations-v3.md): a *learned* aligner collapsed in all three attempts; they use `facebook/wav2vec2-lv-60-espeak-cv-ft` CTC forced alignment, then calibrate: a phone begins 19 ms before the first emission frame; pause tokens only for silence ≥ 75 ms; every token ≥ 1 frame; durations sum exactly to the frame count; reject clips where voicing is found in < 50 % of vowel frames. Also found: an eSpeak invocation bug had corrupted the last phonemes of 294/300 rows — i.e. front-end bugs hide for a long time unless explicitly audited.

## 2. The closest precedent for Nepali: **tiny-voca-hindi** (RL10x, CC-BY-4.0, 9.7 M)

* Devanagari port, **trained on Rasa Hindi (the same corpus family we have for Nepali)**, one female voice × 8 styles (neutral, conv, happy, sad, anger, fear, surprise, disgust — i.e. Rasa's style labels used as speaker/style ids).
* **Ranks 1st of 16 Hindi models on the Open TTS Leaderboard (WER 4.33, ahead of fishaudio s2-pro 4.83 and Kokoro-82M 5.08), RTFx 38.9 on CPU.** Q8 GGUF **11.1 MB**, C++/ggml engine and web page included.
* Config (from `config.json`): 80-mel/24 kHz/hop 256, channels 256, 4 heads, FFN 1024, kernel 7, **encoder 4 / prosody 2 / acoustic 6 layers**, synth FFT 512 hop 128, `cep_causal 80`, `cep_anti 16`, `noise_cep 32`, vocab 70 phone symbols.
* Front-end (`tiny_voca/text.py`): Devanagari → IPA tokens; aspiration/breathy-voice as part of the phoneme; **nasalization as a parallel feature (reusing the “stress” channel), not a token**; schwa deletion rule (word-final dropped; medial dropped in V C _ C V unless it makes a 3-consonant cluster, right-to-left) with compounds as known errors; punctuation/word-boundary tokens may last 0 frames (pause capable).
* ⇒ Direct proof that **Rasa + tiny-voca recipe works for a Devanagari language at ≈10 MB**. Nepali differs in phonology (schwa [ʌ] vs Hindi [ə], different deletion/retention, `ts`/`dz` affricates, `व` realisations, murmured stops) → our G2P knowledge is the differentiator.

## 3. Other relevant systems

| System | Size | What matters for us |
|---|---|---|
| **Alpa** (77ethers, Apache-2.0) | 4.5 M | tiny-voca converted to Sanskrit by *editing two small tables* + 100 CPU-minutes. Proves tiny-voca weights transfer across languages (text-embedding/phoneme tables) — a legitimate **warm-start for Nepali** (phoneme embeddings re-initialised from articulatory neighbours). This repo's earlier use of Alpa was withdrawn (PROVENANCE) — a fresh, properly attributed warm-start is allowed (Apache-2.0). |
| **Indic Tiny TTS** (sulabhkatiyar, CC-BY-4.0, 22.5 M) | 90 MB/format | Matcha-style flow decoder + frozen Vocos, per-language, Rasa F+M, **11 languages — no Nepali**. Confirms Nepali is an unserved slot; also a second recipe (flow+Vocos) to compare. |
| **Indic Parler-TTS** (ai4bharat, Apache-2.0) | 938 M | Has Nepali speakers (69 speakers/18 languages), trained on IndicTTS 382 h + Rasa 288 h. Too large to ship; useful as **teacher / baseline / text-augmentation voice** and for "what is possible" listening. |
| **Kokoro-82M** | 82 M | Open TTS #1 EN (WER 0.66, UTMOS 4.51). StyleTTS2+iSTFTNet; strong reference for quality-per-parameter at ~10× our budget. |
| **Paradee-8M-v1.3, Inflect Micro v3 (12 M)** | 8–12 M | Rank 2 and 6 on the same leaderboard — the sub-12 M class is competitive with 1 B-class models when data is consistent. |
| **Kitten TTS 0.8** (15/40/80 M, StyleTTS2-based, int8 ≈ 8–25 MB) | 15 M | English-only, 8 voices; shows the 10–25 MB band is commercially acceptable. |
| **TinyTTS** | 1.6 M | 3.4 MB ONNX FP16, 53× RT at 44.1 kHz — floor on size; quality not comparable for tonal/prosodic languages. |
| **MMS-TTS (VITS)**, SpeechT5-Nepali fine-tunes, Chatterbox-Nepali fine-tunes | 36 M–0.5 B | Community Nepali models exist (all trained on SLR43/143-class data; MMS is CC-BY-NC). Low bar; include as baselines in the blind test. |
| **Microsoft Edge `ne-NP-Hemkala/Sagar`** | cloud | De-facto commercial bar for Nepali; reference only (not licensed for training). *(measured here: see §5 of README / exp5)* |

## 4. Why these small models win — distilled design rules

1. **Predict explicit F0 / voicing / energy / duration / pause per token and frame** (FastSpeech-2 style), do not hide prosody inside a few-step stochastic decoder.
2. **Differentiable source-filter / iSTFT output trained end-to-end with GAN losses**; no borrowed vocoder with a mel-space adapter.
3. **Consistent, clean, single-style-per-id data**; use style/speaker ids instead of one mixed model of everything; filter bad clips aggressively (voicing, alignment confidence, ASR agreement).
4. **Pretrained phone recogniser CTC forced alignment**, calibrated, validated with a label-free voicing-flip probe; explicit pause tokens.
5. **Ship the evaluation next to the model** (WER vs real-recording WER, leaderboards, listening links) and write the limitations.
6. **Cheap, repeatable training** (single GPU, hours, ≈$10) so that many iterations are possible — vs our 8-architecture/8-day churn.

## 5. Unknowns I could not resolve without hearing audio

* Whether the Rasa Nepali speakers' accent is the "clear Kathmandu standard" the owner wants.
* Whether a pure source-filter head is enough for a *female* voice with median F0 ≈ 247 Hz (Rasa) — the English/Hindi female voices are in a similar range, so likely yes, but a hybrid (source-filter + small iSTFT post-net) is the fallback.
* Whether Nepali needs word-level lexical stress/pitch cues beyond what duration+F0 give. Nepali is weakly tonal on some syllables in Kathmandu speech *(unverified)* — keep a generic per-token "prominence" side channel (as tiny-voca's `stress`).
