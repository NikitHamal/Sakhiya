# Provenance and attribution (Sakhiya v5 — True Single-Voice Neural-VC Unification + Forensic Audio Gate + Autonomous RSI)

This file states, without spin, the exact provenance of every shipped component in `docs/models/` and `docs/samples/versions/`.

## Session 52 — Sakhiya v5 True Single-Voice Breakthrough + Autonomous 9.2-Hour RSI (`31.01 MB` fp16 / `15.99 MB` tiny-int8, 24-Order Phoneme-kNN VC Unification + Acoustic-Verified v5 G2P + Smooth-F0 HarmonicSource + E2E WavLM-SLM + LTAS Timbre Lock, Zero UTMOS Reward Hacking)

| Shipped component | Architecture & Training Provenance | Licence / Status |
|---|---|---|
| `docs/models/encoder.onnx` (`.int8.onnx`) + `decoder.onnx` (`.int8.onnx`) | **Sakhiya-v5-AM** (`sakhiya/cfm.py`, 10.26 M params): Trained exclusively on **one single canonical female vocal tract (`sakhiya`, `spk=0`)** across 8,043 24 kHz studio clips (`5,173` Rasa Nepali Female + `3,137` Rasa Sanskrit Female clips voice-converted into `sakhiya`'s exact vocal tract via 24-order Phoneme-Conditioned Neural kNN-VC in `sakhiya/v5_vc_rsi.py`). Features v5 G2P (`<wb>`/`|` glottal vowel-hiatus re-articulation for `आआफ्नो`, dual-nasal diphthong `k a ʈʰ m a ɖ ʌ̃ ũ` for `काठमाडौँ`, crisp single `h` visarga for `भूर्भुवः`/`वाह!`, and `>= 5` frame function-word floor for `र, त, न, म, हो, छ`), Smooth Continuous F0/Energy Prosody Predictor, Hinge-Capped `ctc_pron` (`floor = 0.24` -> annealed to `0.0`), and Zero-Gather Glitch-Free X-Pred 8-Step Midpoint ODE Decoder. **Zero UTMOS RL/DPO reward hacking, zero WORLD pitch-shifting, zero external TTS teachers.** | 100% our own weights. |
| `docs/models/vocoder.onnx` (`.int8.onnx`) | **Sakhiya-v5-Voc** (`sakhiya/v2_vocoder.py`, 4.98 M params): 24 kHz 16x Smooth-F0 Source-Filter NSF-iSTFTNet + FP32 `Snake1d` (`k=1..8` harmonics driven by 9-frame average-pooled F0 and 5-frame soft-ramped V/UV envelope to eliminate intra-voiced F0 shutters and clicks). Trained with Multi-Res STFT + MPD/MRD + Frozen `microsoft/wavlm-base-plus-sv` SLM Perceptual Discriminator (`D_SLM`, `0` extra ONNX bytes) + **24-Bin DCT LTAS Single-Timbre Consistency Loss**. | 100% our own weights. |
| `scripts/forensic_audio_gate.py` & `scripts/rsi_v5_daemon.py` | **4-Check Real Audio Forensic Gate** (`word_drop_count == 0`, `stutter_count == 0`, `f0_shutters == 0`, `ltas_mcd_ne_sa <= 9.5`, `min_func_word_frames >= 5`, `काठमाडौँ` `ctc_nasal_prob >= 0.50`) + **9.2-Hour Autonomous RSI Controller** (`Cycle 0` Neural VC -> `Cycle 1` `v5_c1` -> `Cycle 2` `v5_c2` -> `Cycle 3` `v5_c3` -> `Cycle 4` `v5_final` -> `Cycle 5+`). | 100% open-source in repo. |

## Session 51 — Sakhiya v4 Promoted (`31.01 MB` fp16 / `15.98 MB` tiny-int8, Disentangled Explicit-Prosody X-Pred Flow + E2E WavLM-SLM Joint Vocoder + 4-Term RL/DPO Polish, 100% Teacher-Free Studio Speech)

| Shipped component | Architecture & Training Provenance | Licence / Status |
|---|---|---|
| `docs/models/encoder.onnx` (`.int8.onnx`) + `decoder.onnx` (`.int8.onnx`) | **Sakhiya-v4-AM** (`sakhiya/cfm.py`, 10.26 M params): Disentangled RoPE Content/Language Encoder with GRL Speaker-Adversarial Classifier (`lambda_grl = 0.20`) + Fine-Grained Explicit-F0/Energy/Pause Prosody Predictor + Exact 4-Step Deterministic Duration Predictor (`2.0x` tail phoneme weight) + Glitch-Free X-Prediction 8-step Midpoint ODE ConvNeXt-1D OT-CFM Decoder locked to **one single unified female voice (`sakhiya`)** across all 6 expressive styles (`news`, `storytelling`, `conversational`, `warm_soft`, `expressive`, `vedic_chant`). Refined via 4-Term GRPO/DPO preference optimization (`sakhiya/v4_slm.py`). **Zero Edge-TTS weights/audio, zero Alpa weights/audio, zero Hindi data, zero WORLD pitch-shifting.** | 100% our own weights. |
| `docs/models/vocoder.onnx` (`.int8.onnx`) | **Sakhiya-v4-Voc** (`sakhiya/v2_vocoder.py`, 4.98 M params): 24 kHz 16x Source-Filter NSF-iSTFTNet + FP32-stabilized `Snake1d` Residual Blocks (`k=1..8` harmonics with voiced-anchored log-F0 raised-cosine smoothing, `n_fft=64, hop=16` `_ola64`). Trained end-to-end with a frozen `microsoft/wavlm-base-plus-sv` Multi-Layer SLM Perceptual Discriminator (`SLMDiscriminator`, `0` extra bytes in exported ONNX) + `WavLM-SV` Single-Voice Timbre Lock Loss + MPD/MRD + Multi-Res STFT Loss. | 100% our own weights. |
| Real human training corpora (`v3_studio_aligned_24k.pt`) | `ai4bharat/Rasa` Nepali Female & Male + Sanskrit Female & Male studio recordings (CC-BY-4.0, 48 kHz resampled to 24 kHz), `OpenSLR SLR43` Nepali female studio recordings (CC-BY-SA-4.0), and `FLEURS ne_np` (CC-BY-4.0). All 200 unseen sentences in `eval/bench_general_200.json` strictly excluded from training. | CC-BY-4.0 / CC-BY-SA-4.0 (attribution given below). |

---

## Historical Audit (Session 30 — v1 Wave 1–42 Lineage, Now Superseded by Sakhiya v2)

This section records the earlier v1 audit (sessions 27–30) before `Sakhiya v2` replaced the v1 FastPitch/Vocos lineage.

## What was found

| Shipped component | What it really is | Licence / status |
|---|---|---|
| `docs/models/sanskrit.onnx` (4.77 MB), "Sanskrit Female / Male" in sessions 27-29 | **The Alpa v0.2 4.5M "Rasa" model (`77ethers/alpa`, Apache-2.0) exported to ONNX**, wrapped with our DSP post-processing (formant sharpening, visarga floor, svara pitch offsets) and a Python/JS front-end. It was never trained by us. | Apache-2.0. Attribution required, given here. |
| "Sakhiya beats Alpa v0.2" rows in `eval/metrics.json` (sessions 28-29) | Alpa-vs-Alpa-plus-post-processing. **Not an independent win and not a valid claim.** Withdrawn. | - |
| `sanskrit_female` / `sanskrit_male` speaker rows in the AM (`sa_f0`, `sa_m0`) | Trained on 600 clips **synthesised by Alpa** (`scripts/distill_sanskrit.py`) plus 300 real Shrutilipi clips. | Apache-2.0 teacher output. |
| Nepali flagship voice `edge_f` / `edge_m` (acoustic model + vocoder) | Distilled from **Microsoft Edge neural TTS** (`ne-NP-HemkalaNeural`, `ne-NP-SagarNeural`), about 37 h. | Research / personal use only. **Not commercially licensed.** |
| Real human data | SLR43 (Nepali, CC-BY-SA 4.0), FLEURS, Common Voice, SLR54, Shrutilipi Sanskrit radio (`surajp/shrutilipi_sanskrit`). About 12.5 h in total. | Per-source licences. |

## What changes from session 30

1. The only voice exposed in the web studio and in the new export is **one female voice, `sakhiya`**, for Nepali, Tharu and Sanskrit. The male voices are removed from the UI.
2. The new training set (`scripts/build_own.py`) uses **real human recordings only**. It drops every Edge-TTS and Alpa-synthesised clip. Held-out validation and test clips are untouched.
3. Alpa is kept only as an **external baseline** in `scripts/evaluate.py`, labelled as such. It is no longer a component of the Sakhiya voice.
4. Any model that still inherits weights from the teacher-distilled checkpoints (the warm start for the AM and vocoder) is stated as such in STATUS.md. A fully teacher-free model is the goal, not yet the fact.

## Attribution

- Alpa v0.2: https://huggingface.co/77ethers/alpa (Apache-2.0), by 77ethers.
- Shrutilipi: AI4Bharat. Sanskrit subset mirrored as `surajp/shrutilipi_sanskrit`.
- Inflect (owenawsong): referenced by the owner as a size/quality bar only. No code or weights are used.

## Session 31 additions (RSI branch `rsi/sakhiya-own-voice`)

- **Vagdhenu chant corpus** (`prathoshap/vagdhenu-data`, CC-BY-4.0, about 5.3 h, one male reciter, studio recorded) is mined by `scripts/mine_vagdhenu.py`. Because the reciter is male and the product is one female voice, every clip is converted to a female register with WORLD (f0 x1.35, spectral-envelope warp x1.12, resulting median f0 about 231 Hz). This is a **disclosed signal-processing augmentation of real, licensed speech**; no synthetic-teacher audio is involved. Attribution: Prathosh A P, Vagdhenu Sanskrit Chant Corpus, CC-BY-4.0.
- **Rasa** (`ai4bharat/Rasa`, CC-BY-4.0, studio female Sanskrit) would be the best female Sanskrit source (Alpa's edge) but is gated on Hugging Face and no token is available to this project. It is **not** used. A token would unlock it (tracked in issue #1).
- Warm start: the acoustic model and vocoder are still initialised from the shipped checkpoints, which carry teacher-era weights (see the table above). The fine-tuning data is real human speech only.
