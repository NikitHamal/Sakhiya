# 05 · Master Blueprint: `Sakhiya-Voca` (4.5M & 9.5M Dual-Voice Female + Male Nepali & Sanskrit TTS) & 12-Hour RTX PRO 6000 Blackwell Runbook

## 1. Product & Engineering Targets

| Target Dimension | `Sakhiya-Voca-4.5M` (Flagship Tiny Tier) | `Sakhiya-Voca-Pro-9.5M` (Studio Tier) | Current Default `cand_b_v2` (To Beat) |
|---|---|---|---|
| **Total Parameters (Incl. Waveform Head)** | **`4,534,795` (`4.53M`)** | **`9,682,000` (`9.68M`)** | `17,652,875` (`17.65M`) |
| **Exported ONNX Size (`FP16` / `INT8+FP16`)** | **`~9.1 MB` FP16 / `~4.8 MB` INT8+FP16** | **`~18.8 MB` FP16 / `~9.8 MB` INT8+FP16** | `35.53 MB` FP16 / `23.01 MB` INT8 |
| **Voices Included in Single Checkpoint** | **2 Native Voices (`0 = F_sakhiya` Female, `1 = M_sagar` Male)** | **2 Native Voices (`F_sakhiya`, `M_sagar`) × 6 Styles** | 1 Female Voice only |
| **Languages Supported** | **Nepali (`ne`) + Sanskrit/Vedic (`sa`)** | **Nepali (`ne`) + Sanskrit/Vedic (`sa`)** | Nepali + Sanskrit |
| **1-Thread / 4-Thread CPU Speed (Measured `EXP4`)** | **`97.1×` (1T) / `161.4×` (4T) real-time** | **`~45×` (1T) / `~85×` (4T) real-time** | `27.9×` (1T) / `52.3×` (4T) |
| **Pauses $\ge 150\text{ ms}$ per Utterance** | **`1.4 – 1.9` (matches human `1.73`)** | **`1.4 – 1.9` (matches human `1.73`)** | `0.20` (breathless) ❌ |
| **Utterance Pitch Dynamic Spread (`|\Delta F_0|` & `IQR`)** | Within **$\pm 10\%$ of real human speech** | Within **$\pm 8\%$ of real human speech** | `-28%` flatter than human ❌ |
| **Nepali 70-Word Phonological Gold Benchmark (`EXP3`)** | **`70 / 70` (`100.0%` word acc, `0.0%` PER)** | **`70 / 70` (`100.0%` word acc, `0.0%` PER)** | `6 / 70` (`8.57%` word acc) ❌ |

---

## 2. Exact Architecture Specification (`SakhiyaVoca`)

Verified and benchmarked in `gemini/experiments/exp4_sakhiya_voca_prototype.py`:

```
Inputs:
  tokens      : int64   [B, N]   (64-symbol Nepali + Sanskrit IPA inventory, BOS/EOS/WORD/PUNCT)
  stress      : int64   [B, N]   (0 = laghu/unstressed, 1 = guru/primary root stress, 2 = geminate/diphthong)
  voice       : int64   [B]      (0 = F_sakhiya Female [215 Hz ref], 1 = M_sagar Male [125 Hz ref])
  lang        : int64   [B]      (0 = Nepali, 1 = Sanskrit/Vedic)
  speed       : float32 [B]      (1.0 default speaking rate)
  pitch_shift : float32 [B]      (log-F0 offset; 0.058 ≈ 1 semitone)
  pitch_range : float32 [B]      (utterance pitch movement multiplier around voiced mean; 1.0 default)
  pitch_add   : float32 [B, N]   (per-token Vedic svara / question / exclamation contour offset)
```

### 2.1 Component Parameter Table (`Sakhiya-Voca-4.5M` vs `Sakhiya-Voca-Pro-9.5M`)

| Stage | Module | `Sakhiya-Voca-4.5M` (`d=192, ff=640`) | `Sakhiya-Voca-Pro-9.5M` (`d=256, ff=1024`) | Key Mathematical Properties |
|---|---|---:|---:|---|
| **1. Token Conditioning** | `tokens` + `stress` + `voices` + `langs` (+ `styles`) | `13,056` | `20,480` | Additive embeddings at token level (`voice` injected at both token and frame levels). |
| **2. Text Encoder** | `RoPEStack` (`4` layers, `heads=4`, `k=7` DWConv + Dense FFN) | `1,588,864` | `3,168,768` | Full-rank 1D RoPE self-attention + `k=7` depthwise conv + dense GELU FFN. Zero SVD/Monarch. |
| **3. Duration & Pause Head** | `DurationPausePredictor` (`2` Conv1d + LayerNorm -> `2` outputs) | `123,906` | `148,738` | Output 0: `log1p(frames)` trained with MSE on active tokens (`dur > 0`).<br>Output 1: `pause_present_logit` trained with BCE on pause-capable tokens (`|`, `,`, `.`, `?`, `!`, `;`). |
| **4. Length Regulator** | Exact Frame Gather + `3D` Intra-Token Position `[rel, 1-rel, log1p(dur)/4]` | `0` *(analytic)* | `0` *(analytic)* | Pause-capable tokens with `pause_logit <= 0` get `0` frames (zero artificial word gaps); tokens with `pause_logit > 0` get their full natural breath pause (`75–380 ms`). |
| **5. Frame Transformer A + Prosody Head** | `frame_voices` + `frame_position` + `RoPEStack` (`2` layers) + `Linear(d, 3)` | `796,355` | `1,587,971` | Predicts frame-level `[log_f0_rel, voiced_logit, log_rms_energy]` every `10.67 ms` (`hop=256`). Trained with L1 pitch + **Global-Variance `pitch_spread` loss** + BCE voicing + L1 energy. |
| **6. Frame Transformer B (Acoustic Stack)** | `prosody_projection` + `RoPEStack` (`5` layers in 4.5M / `6` layers in 9.5M) | `1,986,752` | `4,721,664` | Full-utterance RoPE self-attention conditioned on `(pitch, voiced, energy)` (annealed `1.0 -> 0.0` teacher-forcing over first 10k steps). |
| **7. Waveform Generator** | `NepaliSourceFilterHead` (`cep_causal=80, cep_anti=16, noise_cep=32`) | **`25,862`** | **`34,446`** | Analytical Dirichlet band-limited harmonic pulse train (`synth_fft=512, hop=128`, `2` sub-frame gains) + `4` sub-frame noise cepstrum (`noise_fft=256, hop=64` = `2.67 ms` stop burst resolution) + 5-class Nepali phonation limits + `-21/-19 dB` vowel de-hoarsening floor. |
| **Total Inference Parameters** | — | **`4,534,795`** | **`9,682,067`** | **99.43% in full-rank RoPE Transformers · 0.57% in Source-Filter Waveform Head** |

---

## 3. Nepali Phonological G2P v4 & Disambiguated Corpus Annotation Pipeline

### 3.1 Algorithmic G2P Fixes (`sakhiya/g2p.py` <-> `docs/g2p.js`, Validated `70/70 = 100%` in `EXP3`)
Before any alignment or training runs, we upgrade `sakhiya/g2p.py` (and auto-compile `docs/g2p.js` from shared rule tables + unit tests) with the 6 rules proven in `gemini/experiments/exp3_nepali_phonology_and_g2p_audit.py`:
1. **`ज्ञ` (`ज+्+ञ`) Rule**: In `lang="ne"`, map `ज्ञ -> g j` (with nasalized vowel `ã` on monosyllable `ज्ञान -> g j ã n`, velar nasal `ŋ` before `ज्ञ` in `संज्ञा -> s ʌ ŋ g j a`, and final schwa `ʌ` on word-final `-ज्ञ` in `विशेषज्ञ, यज्ञ, सर्वज्ञ`).
2. **Productive Compound & Derivational Morpheme Splitter (`COMPOUND_STEM2`)**: Expand compound/derivational second-members (`कारी, दार, बाही, नीति, तन्त्र, गन्ज, घाट, दान, वार, फोन, बुक, टक, सफाइ, बड, मल, जल, विद्युत, मन्त्री, सभा, पति...`) so stem-final medial schwas delete cleanly (`जानकारी -> dz a n k a r i`, `कामदार -> k a m d̪ a r`, `सरकारी -> s ʌ r k a r i`, `राजनीति -> r a dz n i t̪ i`, `फेसबुक -> pʰ e s b u k`).
3. **Native `व -> ब` vs Loan/Glide `व -> w` Rule**:
   - Preserve `w` after halanta (`स्व, द्व, त्व, ध्व, ह्व, क्व` -> `स्वागत, द्वारा`), intervocalically after long vowels (`हावा, सेवा`), in `-वार` (`उम्मेदवार`), and in `LOAN_W_ROOTS` (`वालेट, वेबसाइट, वर्ल्ड, वालमार्ट, वाचक, वातावरण...`).
   - Map native Sanskrit/Nepali initial and stem `व` (`विद्यालय, विद्यार्थी, विचार, विभाग, विराटनगर, वीरगन्ज, वडा, वन, वर्षा, वरिष्ठ, व्यवस्था, व्यवहार, विवाद, विवाह, विमान...`) to bilabial **`b`**.
   - **Delete Session 80's `w_e[id_w]` embedding hack** so `w` never bleeds into vowel `u` (`उतारचढाव` stays crisp `-w`, never `-उ`).
4. **Final `-ह` Tatsama Schwa Retention & Native `खर्च` Protection**: Retain final `ʌ` on `-ह` tatsamas (`सिंह -> s i ŋ h ʌ`, `गृह -> g r i h ʌ`, `ग्रह -> g r ʌ h ʌ`, `सन्देह -> s ʌ n d̪ e h ʌ`, `आग्रह`, `उत्साह`, `समूह`, `प्रवाह`, `तह`) and keep `खर्च -> kʰ ʌ r ts ʌ` in `KEEP_FINAL_SCHWA`.
5. **Anusvara (`ं`) Homorganic Assimilation**: Map `ंव -> म्ब` (`संविधान -> s ʌ m b i d̪ʱ a n`, `संवाद -> s ʌ m b a d̪`) and `ं` before `स/श/र` to dental `n` (`संसार -> s ʌ n s a r`, `संस्था -> s ʌ n s t̪ʰ a`, `संस्कृति -> s ʌ n s k r i t̪ i`, `संरक्षण -> s ʌ n r ʌ k ʃ ʌ ɳ`).
6. **`ह` + Sonorant Normalization**: Map `बाह्र -> b a r ʌ`, `तेह्र -> t̪ e r ʌ`, `सोह्र -> s o r ʌ`, `चिह्न -> ts i n h ʌ`.
7. **Per-Phoneme Stress / Syllable Weight (`stress` in `{0, 1, 2}`)**: Emit `0` (unstressed laghu), `1` (root-initial or closed CVC heavy syllable), `2` (geminate consonant or nasal diphthong) for every token.

### 3.2 Calibrated Pretrained CTC Forced Alignment (`wav2vec2-lv-60-espeak-cv-ft` + LogSumExp Mapping)
Following `tiny-voca/src/tiny_voca/align.py` and our `EXP5` calibration:
1. Run `facebook/wav2vec2-lv-60-espeak-cv-ft` (`16 kHz`, `20 ms` stride) on every training recording to obtain frame log-posteriors `log_probs [T_50hz, V]`.
2. **Nepali LogSumExp Posterior Projection (`EXP5` Breakthrough)**:
   - Map Nepali `/ʌ/` (`अ`) to $\text{logsumexp}(\text{log\_probs}[:, \{\text{id}(\text{'ʌ'}), \text{id}(\text{'ə'}), \text{id}(\text{'ɐ'}), \text{id}(\text{'ɔ'}), \text{id}(\text{'o'}), \text{id}(\text{'ɑ'})\}])$.
   - Map breathy-voiced stops (`gʱ, dzʱ, ɖʱ, d̪ʱ, bʱ`) to $\text{logsumexp}$ of the corresponding voiced stop + `[ɦ]/[h]`.
   - For `ai4bharat/Rasa` clips containing Devanagari `व`, compare the aligned posterior of `[b]` vs `[v]/[ʋ]/[w]` **per token**: label the `Rasa` training annotation with `[w]` when the `Rasa` speaker articulated `[ʋ]` and `[b]` when they articulated `[b]`. This guarantees **zero cross-contamination between the `[b]` and `[w]` phoneme embeddings**!
3. **Voicing-Flip Onset Calibration & `-45 dB` Silence Pause Attribution (`align.py:158–209`)**:
   - Calibrate `onset_offset_seconds` on 300 Nepali recordings against `librosa.pyin` vowel-to-voiceless and fricative-to-vowel voicing flips.
   - Assign silent regions (`>= 7` frames = `75 ms` below `P95 - 45 dB`) strictly to pause-capable tokens (`BOS`, `EOS`, `,`, `.`, `?`, `!`, `;`, `|`). Non-silent word boundaries `|` receive **`0` frames**!
   - Reject any clip where `vowel_voiced < 0.50`, `weak_phone_fraction > 0.25`, `SNR < 28 dB`, or `MMS-1b-all` strict CER `> 0.28`.

---

## 4. Two-Stage Kathmandu-Native Multi-Voice Curriculum (Solving `Rasa` Volume + Kathmandu Accent)

How do we get **both** the acoustic richness of `55.1 hours` of `48 kHz -> 24 kHz` studio `ai4bharat/Rasa` (Female + Male) **and** a 100% authentic **Nepal Kathmandu accent** in both Female and Male voices?

### Stage 1: Multi-Speaker Studio Backbone Pretraining (`Steps 0 -> 85,000`)
Train on the combined, strictly speaker-ID-separated studio corpus (`n_speakers = 8`):
- `spk 0`: **Kathmandu Female Target (`F_sakhiya`)** — `SLR43` speaker `nep_6834` + top 4 studio Kathmandu female speakers (`s0546`, `s3614`, `s2099`, `s3960`, acoustically timbre-matched and F0-normalized, `~3.8 h` pure 24 kHz Kathmandu female speech, oversampled `3×`).
- `spk 1`: **Kathmandu Male Target (`M_sagar`)** — Top high-SNR wideband Nepal-born male speakers from `OpenSLR SLR54` + `FLEURS ne_np` male studio subset (`~4.2 h` pure Nepal male speech, oversampled `3×`).
- `spk 2`: **`ai4bharat/Rasa` Nepali Female** (`~24 h` curated 24 kHz studio neutral/news/conv/book).
- `spk 3`: **`ai4bharat/Rasa` Nepali Male** (`~22 h` curated 24 kHz studio neutral/news/conv/book).
- `spk 4`: **`ai4bharat/Rasa` Sanskrit Female + Gita/Vagdhenu Studio Vedic** (`~6 h` curated 24 kHz studio Sanskrit).
- `spk 5`: **`ai4bharat/Rasa` Sanskrit Male** (`~5 h` curated 24 kHz studio Sanskrit).

**Why this works:** In `SakhiyaVoca`, the Transformer layers (`encoder`, `duration`, `prosody_stack`, `acoustic_stack`, `vocoder`) share `99.7%` of parameters across all speakers while `voices(spk)` and `frame_voices(spk)` (`384` floats per speaker!) absorb speaker-specific F0 height and vocal-tract length! And because `Rasa`'s `[ʋ]` realizations are labeled as token `w` during Stage 1, the shared Transformer backbone learns crystal-clear `[b]` vs `[w]` articulation from `60+ hours` of studio audio without any Indian `[ʋ]` leak into `[b]`!

### Stage 2: Kathmandu Voice & Accent Lock (`Steps 85,000 -> 115,000`)
- Anneal sampling weights to **70% Kathmandu Native (`spk 0` Female + `spk 1` Male, with Kathmandu `[ts]/[tsʰ]` affricates and `g2p_v4` schwa/phonology)** + **30% Studio High-Band Regularizer (`spk 2..5` with `duration` and `prosody_stack` gradients masked on `spk 2..5` so prosody, pausing, and accent lock 100% to native Kathmandu speakers while keeping 48->24 kHz studio bandwidth)**.
- At export time, slice rows `0` (`F_sakhiya`) and `1` (`M_sagar`) of `voices` and `frame_voices` into the final 2-voice checkpoint!

---

## 5. Hour-by-Hour 12-Hour RTX PRO 6000 Blackwell (96 GB VRAM) Execution Runbook

### Pre-GPU CPU Preparation (Can Run on MoLab CPU Box Before Starting Blackwell Clock)
1. **Merge & Lock `g2p_v4` (`sakhiya/g2p.py` + auto-generated `docs/g2p.js`)**: Verify `70/70` on `exp3_nepali_phonology_and_g2p_audit.py` + `100%` Python $\leftrightarrow$ JS parity on all existing tests (`pytest -q tests/`).
2. **Download & Prepare Acoustic Shards (`prepare.py`)**: Extract 24 kHz waveforms, 80-band Slaney log-mels, `pyin` `(log_f0, voiced)` + `trusted` pitch mask, and `log_rms_energy` for `Rasa` (F & M), `SLR43` Kathmandu Female, and `SLR54`/`FLEURS` Nepal Male.
3. **Run Calibrated Forced Alignment (`align.py` with LogSumExp Nepali Posterior Mapping)**: Produce validated `train.jsonl`, `validation.jsonl`, and `test.jsonl` with exact frame-summing `durations` and explicit `0`-vs-`>=7` frame pause labels.

### On the 12-Hour RTX PRO 6000 Blackwell (96 GB VRAM, 20 CPUs, 160 GB RAM)

| Hours | Phase | Action & Exact Command / Hyperparameters | Hard Go / No-Go Gate |
|---|---|---|---|
| **00:00 – 00:45** | **Phase 0: Fast Alignment & Data Verification** | Pull cached acoustic shards from HF; run GPU-accelerated `wav2vec2-lv-60-espeak-cv-ft` LogSumExp forced alignment across all `~60 h` in `< 18 min` (`batch=64` in fp16). Verify voicing-flip boundary probe (`>= 62%` within 1 frame, `>= 84%` within 2 frames) and `0` text-hash leakage between train/val/test. | `durations.sum() == frames` on 100% of rows; `pause_voiced < 0.05`; `vowel_voiced >= 0.85`. |
| **00:45 – 04:45** | **Phase 1: Concurrent Stage-1 Backbone Training (`Sakhiya-Voca-4.5M` & `Sakhiya-Voca-Pro-9.5M`)** | Launch **both** models in parallel on the 96 GB Blackwell GPU (each uses `< 14 GB` VRAM at `batch_frames = 12,000`, running at `~8.5 steps/s` = `30,000 steps/hour`!):<br>• Steps `0 .. 8,000`: Spectral-only warm-up (`stft=2.0, mel=10.0, pitch=2.0, pitch_spread=2.0, voicing=1.0, energy=1.0, dur=1.0, pause=1.0`, `teacher_hold=3000`, `teacher_ramp=7000`).<br>• Steps `8,000 .. 85,000`: Enable MPD + MRD + MS-SB-CQT discriminators with **Dynamic Gradient-Norm Balancing (`gan_ratio = 0.7`, `gan_ramp = 3000`, `feature_matching = 30.0`)**. | At step `10,000`: `pitch_spread_ratio` in `[0.88, 1.12]`, `pause_accuracy >= 0.90`, `voicing_accuracy >= 0.92`. |
| **04:45 – 07:15** | **Phase 2: Stage-2 Kathmandu Voice & Accent Lock (`Steps 85,000 -> 115,000`)** | Branch from step `85,000` with `lr = 1.0e-4` (`lr_half_life = 30,000`), locking prosody/duration/accent to `spk 0` (`F_sakhiya` Kathmandu Female) and `spk 1` (`M_sagar` Nepal Male) + Sanskrit Vedic Svara conditioning (`pitch_add`). Maintain EMA (`decay = 0.9995`). | Held-out Kathmandu validation: pauses/utt in `[1.3, 2.0]`, speaking rate within `±8%` of human, `HPVR >= 24 dB`, `[ts]/[tsʰ]` centroid `>= 5,500 Hz`. |
| **07:15 – 09:30** | **Phase 3: Selective Mixed-Precision QAT & Multi-Runtime Export (ONNX + GGUF/WASM)** | Run 15,000 steps of STE QAT on `encoder`, `prosody_stack`, and `acoustic_stack` linear weights while keeping `tokens`, `duration`, `prosody_head`, and `NepaliSourceFilterHead` (`25.8K` params) permanently in FP16/FP32.<br>Export single-graph `model.fp16.onnx` (`~9.1 MB`) and `model.int8.onnx` (`~4.8 MB` for 4.5M / `~9.8 MB` for 9.5M) with analytic STFT/cepstrum kernels (`analytic_cep`, `analytic_stft`). | `STFT_LSD(INT8, FP16) <= 0.15 dB`, `len_diff(INT8, FP16) == 0`, `nan_count == 0`, Node ORT 1T RTF `<= 0.015`. |
| **09:30 – 11:30** | **Phase 4: Non-Gameable Evaluation Battery & Blind A/B/C/D Listening Desk** | Evaluate `Sakhiya-Voca-4.5M` (F & M) and `Sakhiya-Voca-Pro-9.5M` (F & M) on:<br>1. Frozen 100-sentence unseen Kathmandu + Rasa test set (real-speech-relative prosody bands).<br>2. 70-word Nepali Phonological Gold Benchmark (`ज्ञ, व/ब, VC-CV syncope, -ह schwa, ं assimilation, बाह्र/तेह्र, च/छ`).<br>3. Render `-16 LUFS` Blind Listening Desk on `docs/listen.html` + MoLab Card comparing Real Human, `Sakhiya-Voca-4.5M (F & M)`, `Sakhiya-Voca-Pro-9.5M (F & M)`, `Sakhiya-V1`, `cand_b_v2`, and `Edge Hemkala/Sagar`. | Promote only if all real-speech-relative prosody bands, `70/70` phonology checks, and blind listening checks pass; sync all checkpoints & shards to Hugging Face. |
