# 08 · Revised plan v2 (after session 2) — what changes, what is still unproven, what to do before spending GPU hours

Supersedes the data/gate parts of `04` and `06 §A`; the model topology recommendation (tiny-voca family, explicit prosody, source-filter/GAN, 2 voices) is unchanged and agrees with `gemini/05`.

## 1. Why we are behind (final, ranked, with the evidence class)
| # | Cause | Evidence | Status |
|---|---|---|---|
| 1 | Student-of-synthetic-teacher lineage; 55 h native studio speech unused as backbone | repo audit, `01`, `02` | measured |
| 2 | Gates blind to nativeness (`ne_pcer` deletes ʌ / geminates / w≠b; hard-coded UTMOS 3.94 vs re-measured 2.64–2.81) | `02`, EXP1/4 | measured |
| 3 | Default tier talks fast, never pauses (0.20 vs 1.73 pauses/utt), flat F0 (−28 %), jitter +41 % | EXP1 | measured |
| 4 | Parameter budget inverted: 76.7 % of the 17.7 M default is the borrowed Vocos; the 4.1 M acoustic model has no frame attention | Gemini EXP1 (`gemini/01`) | measured (structure) |
| 5 | G2P: `ज्ञ`, schwa syncope in compounds, anusvara place assimilation, `ह`-clusters (numerals 13–16), `व` b/w | Gemini EXP3, reproduced here (`07 §4`) | measured; adjudication list needed |
| 6 | Unconditioned-F0 harmonic blur in a deterministic mel regressor | Gemini EXP2 (synthetic proof) | **hypothesis H-F0, untested on shipped mels** |
| 7 | Accent/pronunciation of the *training* voice not controlled (Rasa-F `[v/w]` 73 % on `वि-`) | `07 §1` | measured |

## 2. Decisions taken this session (owner-approved direction: use Rasa)
1. **Backbone data = Rasa Nepali, both voices.** Accent managed by (a) variant phone `ʋ` tagged per utterance from the acoustic probe, `b` default at inference, (b) stage-2 adaptation of the female voice on SLR43 (Nepal), (c) re-run `exp6` on synthesized output as an accent gate (`[v/w]` on `वि-` ≤ 20 %, `ज्ञ → g` ≥ 90 %).
2. **Primary demo voice order:** Male first (closer to Nepal on `वि-`, 37 % vs 73 %), Female after SLR43 adaptation passes the gate.
3. **Held-out set:** our own sentence-hash split of the 23,648 unique Rasa sentences, whole-sentence disjoint for both voices (Rasa `test` is 43.5 % contaminated) + 500 written sentences (numbers, units, dates, loanwords, questions, names). Commit the hash list; never tune on it.
4. **Neutral base stage:** WIKI + NEWS + BOOK + CONV (≈ 17.9 h F / 17.4 h M raw); PROPER NOUN and emotions/commands as separate style ids in stage 2. Add a 80–150 ms silence token at clip edges (Rasa clips are trimmed to ≈ 10 ms).
5. **Front-end is a first-class deliverable**: one rules+lexicon source → generated `docs/g2p.js`; digits/units/Latin are handled there (the data has none).
6. **Alignment:** pretrained CTC (MMS-1b-all grapheme for word/char anchors — verified working in EXP6, ≈ 2 s/clip on 6 threads — and espeak-CV recogniser with LogSumExp `ʌ` mapping for phone boundaries); two aligners must agree within 40 ms on a 500-clip audit; self-trained aligner never the only source.
7. **No promotion without a blind human test** (`05`); automatic metrics only as tripwires with real-speech-relative bands. Probes that are **not** valid for ranking: strict CER (saturated at 0.08–0.10 for real speech too), the `exp6` affricate classes, `roll99`.

## 3. What is still unproven (and the cheapest test for each)
| Unknown | Test (CPU unless noted) | Cost |
|---|---|---|
| Does the tiny-voca-style topology learn Nepali from aligned Rasa at all? | 200-step overfit on 32 aligned clips; then 2–3 K steps on 2 K clips (CPU, ~1–2 h) → render + objective probes | 1 day |
| Is H-F0 (harmonic blur) real in `cand_b_v2`? | HPVR of `cand_b_v2` vs V1 vs real on 40 shared sentences | 1 h |
| Is the Rasa 48 kHz content real above 12 kHz? | long-term spectrum of 100 clips (0–24 kHz), compare to SLR43 | 30 min |
| How wrong is G2P on the *evidence-safe* categories? | native review sheet (300 words: 100 `ज्ञ/संव-/ह`-class certain, 100 schwa, 100 व/b-w) | owner 30 min |
| Do Rasa-F `ʋ` utterances cluster by sentence type/style? | extend `exp6` to all `वि-/व-` words in 2 train shards; tabulate by style and by speaker session | 2 h |
| Can SLR43 (2,064 clips, several female speakers, 16 kHz) lift Rasa-F to the accent gate without losing quality? | stage-2 on GPU only after the first model exists | GPU 1 h |

## 4. CPU prep list v2 (replaces `06 §A`; do before any GPU hour)
1. **A1′ Owner listening desk G0** — 12 clips each: Rasa F, Rasa M, SLR43, SLR54, and the `वि-` word crops from `exp6` (blind labels). Owner confirms the Rasa accent is acceptable for the target product.
2. **A2′ Data build** — pull Rasa Nepali (train + test, ≈ 18 GB) to persistent storage; 48→24 kHz; trim/normalise; length 0.8–20 s; ASR agreement (MMS strict CER ≤ 0.30); SNR filter (floor ≤ −55 dBFS); log every rejection to `rejected.jsonl`; produce `manifest.jsonl` with `{gender, style, text, norm_text, dur, shard, sent_hash, split}`.
3. **A3′ Alignment dry run** (500 clips): MMS + espeak aligners, disagreement histogram, voicing-flip onset probe, `ʌ` coverage report.
4. **A4′ Front-end**: normaliser + unit table; G2P fix list from `07 §4` (certain items only) with regression tests; native review sheet; generated JS.
5. **A5′ Skeleton + overfit** (Gemini's `exp4` skeleton, Nepali token table, F/M ids, `ʋ` variant token, silence token) → 200-step CPU overfit curve.
6. **A6′ Frozen eval** — `eval/frozen_v1/` (hash list, 500 written sentences, listening page), baseline bake-off (V1, default, Edge F/M, MMS-TTS).
7. **A7′ H-F0 and bandwidth probes** (table above).

## 5. GPU session gates (unchanged shape, tighter numbers)
* H0–1: env + cached features. H1–3: spectral-only warm-up (T4.5 and T10, both voices). H3–9: GAN phase (3 concurrent runs), checkpoint every 10 K and auto-render the frozen set. H9–11: stage-2 (female SLR43 adaptation; style ids). H11–12: export (ONNX FP16/INT8, ORT-web parity), render, package.
* **Stop rule:** at 40 K steps, if the objective tripwires (pauses/utt within ±30 % of real, F0 range within ±20 %, jitter within +20 %, `exp6` accent gate trending) are not met → diagnose data/alignment, **do not add architecture**.
* **Promotion:** blind CMOS vs V1 and Edge F/M with ≥ 8 listeners (≥ 3 native Nepal-resident). Otherwise V1 remains the default and the failure analysis goes to `sonnet/RESULTS_*`.

## 6. Owner decisions still needed (small)
1. Confirm on the listening desk that the Rasa accent is acceptable for *both* voices (or approve "male first").
2. Approve 30 min for the native-speaker G2P review sheet (the single highest-leverage human input).
3. Approve the sentence-hash held-out split (we ignore Rasa's own `test` split).
