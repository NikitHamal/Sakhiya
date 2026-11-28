# 04 · Plan: **Sakhiya-Tiny-NE** — a 5–10 M-parameter, two-voice (F/M) Nepali TTS from real studio speech

Goal (owner): the tiniest model with the clearest, cleanest, most native Nepali, 2 voices, better than existing Nepali TTS.
Principle: **change the recipe, not the polish** — real data, explicit prosody, end-to-end waveform, human-judged.

## 0. Decision gates (cheap, ordered; all can run before the GPU session)

| Gate | Question | How | Pass → | Fail → |
|---|---|---|---|---|
| G0 | Is the Rasa Nepali accent acceptable? | Owner + ≥3 native listeners hear 12 raw clips per speaker (F, M) from Rasa, plus SLR43 6834, FLEURS ne_np (listening desk static page, randomised, blind) | Train on Rasa F/M | Train backbone on Rasa, **adapt** to SLR43-6834 (≈2 h) for the female voice; male stays Rasa |
| G1 | Is G2P the accent bottleneck? | 300-word native-speaker review sheet (§2.3) + acoustic scoring of pronunciation variants | fix list → regenerate lexicon | – |
| G2 | Does the recipe beat today's default in a blind test? | 24 sentences × {real, new, V1, default, Edge F, Edge M}, ≥8 listeners, CMOS + MUSHRA-lite | Replace default | Keep V1 as default; iterate |

Baseline fact to beat *(measured, EXP1/3/4)*: UTMOS22 of default 2.64–2.81 vs V1 4.07 vs real 3.79; prosody deltas in `02_root_causes.md`.

## 1. Data (teacher-free; no VC, no distillation)

| Source | Use | Hours |
|---|---|---|
| **ai4bharat/Rasa Nepali** F (16,016 utts) + M (15,239) — CC-BY-4.0, 48 kHz studio, token already has access (HF `Rasa` listing 200) | **Main**: neutral base (WIKI 10.4 h, CONV 3.2, NEWS 2.2, BOOK 2.1, PROPER NOUN 2.0 per voice) | F 28.7 h / M 26.4 h total |
| Rasa emotion styles (6) | Style ids in stage 2 (optional "expressive" voices) | ≈7.5 h / voice |
| Rasa command styles (ALEXA, BB, DIGI, UMANG) | Numbers/units/commands robustness | ≈1.3 h / voice |
| SLR43 spk 6834, FLEURS, CV ne-NP | Accent adaptation (G0 fail) and **held-out test text/audio only** | ≈2 h + test |

Curation (in this order, log every rejected clip with reason like tiny-voca's `*.rejected.jsonl`):
1. Resample 48→24 kHz (sinc), peak/LUFS normalise per voice; trim edges to ≤150 ms silence.
2. Length 0.8–20 s; text normaliser (numbers → words incl. Nepali numerals, units, dates, Latin → Devanagari lexicon).
3. **ASR agreement filter**: MMS-1b-all `npi` + a second Nepali ASR; drop clips with strict CER > 0.30 against transcript (catches mis-transcriptions; measured human CER mean is 0.09, so 0.30 is ≈3σ-ish loose).
4. Forced alignment (§2.2) confidence filter; voicing-in-vowel-frames ≥ 50 % (tiny-voca rule); SNR > 30 dB.
5. Hold out **sentence-disjoint** val/test (Rasa `test` split + 500 sentences we write; hash the list into the repo; **never tune on it**).
Expected yield 85–95 % → ≈ 17 h neutral + 17 h neutral, + styles.

## 2. Front-end (single source of truth)

### 2.1 Normaliser
Port `numbers.py` + ad-hoc rules into one module with a unit-test table (numbers, Nepali digits, `%`, dates, time, money, ordinals, acronyms, loanwords). Output Devanagari words only; Latin → lexicon → else spell-out.

### 2.2 Alignment and durations
Primary: **pretrained CTC forced alignment** (torchaudio MMS_FA / `wav2vec2-lv-60-espeak-cv-ft`) of our phone sequence (mapping table to the recogniser's labels, alignment-only), then tiny-voca's calibrations: onset shift (measure on Nepali with the voicing-flip probe — do not copy 19 ms), pauses ≥ 75 ms on boundary/punctuation tokens, ≥ 1 frame/token, exact frame-sum. Cross-check against a second aligner on 500 clips; flag disagreement > 40 ms. **Do not use a self-trained aligner as the only source** (tiny-voca: collapsed 3/3).
Note (EXP2): `wav2vec2-lv-60-espeak-cv-ft` never emitted a schwa class on Nepali; verify its label coverage before trusting it for ʌ boundaries; fall back to MMS grapheme-CTC alignment + G2P projection if needed.

### 2.3 G2P
* Keep the Nepali phone inventory (ʌ a i u e o; aspirated/murmured stops; retroflex/dental; ts/tsʰ/dz/dzʱ; nasal vowels as a **parallel feature**, not a token; diphthongs; geminates marked) — it is linguistically sound.
* Generate a **data-driven pronunciation lexicon** for the top ~30 k words in Rasa: for each word, candidate pronunciations (schwa kept/deleted at each medial site, `व`→w/b, ड़/ढ़ flap) → score each against the aligned audio (acoustic GOP with the aligner) → keep the best if margin > threshold; rules only for OOV. This replaces hand-patching with evidence and gives the owner a review sheet of *disagreements only* (G1).
* Compile Python rules + lexicon to a JSON table and **generate** `docs/g2p.js` from it (no hand duplication).

## 3. Model (tiny-voca topology, Nepali adaptations)

```
tokens(≈64 phones+punct+boundary) + nasal feat + prominence feat + voice id(F/M) + style id
 → text encoder (4 blocks, d=192→256, RoPE) → duration+pause predictor (log1p, per-token)
 → length-regulate → frame transformer A (2) → F0 / voicing / energy (explicit, per frame)
 → frame transformer B (5–6) → per-frame cepstral filter params → differentiable source-filter synth (pulse+noise, ~26 K params)
 → 24 kHz wave.   Training-only: aux mel head, MPD + MRD discriminators.
```
Variants: **T4.5** (d=192, ≈4.5 M, 2 voices) and **T10** (d=256, 6 acoustic layers, ≈9.7 M, 2 voices × styles).
Warm start (optional, Apache-2.0, attributed): tiny-voca/Alpa weights for the transformer stacks with phoneme embeddings re-initialised — compare against scratch; adopt only if it helps val loss at 10 K steps.
Losses: multi-res STFT + mel + F0/V/UV/energy/duration (log) + adversarial after 8 K steps (MPD/MRD) + feature matching; **teacher-forced durations for 100 % → 0 % annealing** (tiny-voca `--teacher-final 0`).
Prosody diversity (later, only if needed): sampled F0/duration offsets from a tiny VAE/flow on the *predictors only* (kept out of the waveform path), 0 ⇒ deterministic.

Size/speed targets: T4.5 ≈ 9 MB FP16 / ≈ 5 MB INT8; T10 ≈ 19 MB FP16 / ≈ 10–11 MB INT8 (tiny-voca-hindi Q8 = 11.1 MB). RTF ≤ 0.05 on 1 CPU thread. Export ONNX (ORT-web), keep the existing ggml/C++ engine option.

## 4. Training budget (RTX PRO 6000 Blackwell 96 GB, 12 h)

tiny-voca: 104 K steps in 7.7 h on a 4090 (≈ 4 h here). Run **three experiments concurrently** (96 GB is plentiful):
(a) T4.5 2-voice, (b) T10 2-voice, (c) T10 2-voice + 8 styles. Each: 8 K spectral-only → GAN to 120 K steps; checkpoint every 10 K; auto-render the frozen listening set at each checkpoint; ASR/prosody battery (EXP1 script) per checkpoint (CPU side).
Fallback if GAN source-filter has buzz: add a 1–2 M-param iSTFT post-net conditioned on the same hidden states (still end-to-end, still no borrowed vocoder).

## 5. Optional ceiling-raiser (only after Gate G2 shows quality plateau)

tiny-voca's secret is consistency. Train a large teacher on **all** Rasa Nepali (e.g. fine-tune Indic Parler-TTS Nepali speakers, or our existing 83 M DiT retrained on 55 h instead of 12.6 h), then synthesise ≈ 50 h of *new* text (news, numbers, loanwords, questions) in both voices and add to the tiny model's training set. Gate: teacher must beat real-data tiny model in blind test, else skip. Keep Edge TTS outputs out of training (licence).

## 6. What we stop doing
VC (Seed-VC/kNN-VC), teacher-of-teacher distillation, MeanFlow/Monarch/Hadamard/INT8-QAT before a quality baseline, per-session architecture swaps, promoting on pCER/UTMOS/grit, hard-coded metrics in `tiers.json` (generate from the eval script).

## 7. Risks
| Risk | Mitigation |
|---|---|
| Rasa accent not the owner's target | G0, adapt with SLR43-6834 |
| Source-filter buzz on female voice | iSTFT post-net fallback; compare with Matcha+Vocos "Indic Tiny" recipe on same data |
| Expressive Rasa styles pollute neutral | Style ids; stage-1 neutral only; styles in stage 2 |
| Aligner errors → mumbling | Two aligners, voicing probe, reject list, durations audited per phone class |
| G2P errors (loanwords, compounds) | Data-driven lexicon + review sheet + loanword lexicon; unit tests generated from sheet |
| Overfitting evaluation again | Frozen hashed test set; blind human test is the only promotion criterion |
