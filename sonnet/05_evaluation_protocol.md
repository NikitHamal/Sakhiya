# 05 · Evaluation protocol (what replaces pCER/UTMOS/grit as promotion criteria)

Lesson of 91 sessions: **a model can win every automatic gate we had and still sound worse** (`02_root_causes.md`, RC1/RC3).
Automatic metrics become *tripwires*; **only a blind human test can promote a model.**

## 1. Frozen test set (created once, hashed, never trained or tuned on)
* 500 Nepali sentences: 150 Rasa-`test` neutral (sentence-disjoint from train), 100 news/wiki written by us, 100 conversational/questions, 80 numbers/dates/money/units, 70 loanwords/acronyms/proper nouns.
* 100 *minimal-pair/contrast* items (schwa retention, geminates `पक्कै/पक्के`, aspirated vs unaspirated, retroflex vs dental, `च/छ` affricates, `व` /w/ vs /b/, nasal diphthongs `ौँ`), recorded by a native speaker once (≈15 min) as the accent reference.
* Store `eval/frozen_v1/sentences.jsonl` + SHA-256 in the repo; CI refuses edits without a version bump.

## 2. Human tests (promotion criteria)
| Test | Design | Pass bar |
|---|---|---|
| **Blind CMOS vs incumbent** | 24 sentences, A/B randomised, ≥ 8 listeners incl. owner and ≥ 3 native Kathmandu speakers, 7-point scale | Mean CMOS ≥ +0.5 vs V1 **and** vs current default; no sentence class worse than −0.3 |
| **MUSHRA-lite** | hidden real recording (reference + hidden reference), new model, V1, default, Edge F/M, 3.5 kHz-lowpass anchor; 20 sentences | New model ≥ 75 % of hidden-reference score for its own gender |
| **Accent clarity checklist** | Native listener marks, per contrast item, "clearly native / acceptable / wrong" | ≥ 90 % clear-or-acceptable; 0 wrong on schwa/geminate/च-छ items |
| **Intelligibility** | Transcription by native listeners on 40 semantically unpredictable sentences | WER ≤ real-recording WER + 2 pt |
Tooling: static listening page (extend `docs/listen.html`) that emits a JSON of ratings; ratings JSON committed to `eval/human/`.

## 3. Automatic tripwires (must not regress; never sufficient)
Run `sonnet/experiments/exp1_real_vs_shipped.py` unchanged on the frozen set. **Targets are the real-speech statistics of the same speaker**, not "as clean as possible":

| Probe | Target (relative to real recordings of the same voice) |
|---|---|
| Pauses ≥150 ms / utterance, mean pause | within ±25 % (real F: 1.73, 0.30 s on Rasa test sentences) |
| Speaking rate (graphemes/s excl. pauses) and total duration | within ±7 % |
| Mean |ΔF0| (st/10 ms), F0 5–95 % range | within ±12 % |
| Jitter / shimmer / HNR | within ±20 % (**lower jitter than real is also a fail** — over-smoothing) |
| Tilt 4–8 kHz vs 0–2 kHz, 8–12 vs 4–8 kHz | within ±1.5 dB |
| Strict ASR CER/WER (MMS-1b-all `npi` **and** a second ASR), **no** schwa/geminate/`व→ब` normalisation | ≤ real + 1.5 pt (report real floor next to every number) |
| UTMOS22 / NISQA | sanity floor only (≥ real − 0.4); **never a promotion criterion** (default scored 2.64 and still "passed") |
| Duration/alignment audit | CTC/forced-alignment confidence on synthesized audio ≥ real − 0.05; geminate/singleton ratio within ±15 % of real |
| Old forensic gate (word drop, stutter, clicks, F0 shutters, comb grit, sibilant flatness) | keep as regression tripwires |

## 4. Reporting rules
1. Every number in `docs/tiers.json` is **generated** by the eval script (`utmos`, `ne_cer`, …) with the script SHA and test-set hash; hand-typed literals are forbidden (`promote_and_sync_cand_b_v2.py:92` is the counter-example).
2. Always print the **real-recording score next to the model score**.
3. A model card per release: data hours/source, params, size, limitations, human-test table, links to the listening page (tiny-voca's card is the template).
4. One baseline model stays "frozen incumbent" until beaten in §2.

## 5. Baselines to include in every blind test
Real Rasa recording (reference), `Sakhiya-V1` (`docs/models_v2`), current default (`docs/models_sakhiya_one_vocos`), Edge `ne-NP-HemkalaNeural`/`SagarNeural` (reference only, **not** a training source), MMS-TTS Nepali (if license allows research use), Indic Parler-TTS Nepali speaker.
