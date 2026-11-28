# 07 · Session 2 findings (Sonnet lane) — accent control, Rasa audits, verification of the Gemini lane

Scope: research + measurement only. No model, `docs/`, `cpp/`, `src/` file changed. All numbers below come from scripts in
`sonnet/experiments/exp6..exp8` and raw outputs in `sonnet/results/s2/` (MoLab CPU box, 20 cores, no GPU). Owner decision recorded: **Rasa is acceptable
as the main dataset, even if the accent is not perfectly Nepal-standard** — so the accent work below is about *managing* it, not rejecting it.

## 0. Headline
1. **Rasa Nepali is a real, strong acoustic backbone (very clean, 48 kHz, no clipping, −71 dBFS noise floor) but it is NOT a Nepal-standard accent for the
   female voice on `वि-` words.** Measured at word level with Nepal-recorded controls: Rasa-F says `[v/w]` 73 % of the time, Nepal controls 7–20 %. Rasa-M is in between (37 %).
2. **Gemini's direction is right, its number and method were wrong** (89.5 % came from counting any b/v before i/j anywhere in 20 clips, 1 speaker, no control; word-level it is 73 % F / 37 % M).
3. **Rasa's own `test` split cannot be used as a held-out set**: 43.5 % of its unique sentences also occur in `train`. We must build a sentence-disjoint test set ourselves (this corrects `04_plan §1.5`).
4. **Rasa teaches no numerals, Latin, dates or units** (2 utterances with digits, 0 with Latin, 72 Devanagari symbols). Every number/loanword behaviour must come from the front-end, not from data.
5. Female and Male read **the same sentences 48 % of the time** (7,282 shared) → natural parallel text for a 2-voice model.
6. Gemini's 34 listed G2P outputs were **reproduced exactly** from `sakhiya/g2p.py` (they are real), but Gemini's `Sakhiya-Voca-4.5M` prototype is **untrained** — size/speed are valid, quality evidence is zero.

## 1. EXP6 — word-level accent probe with controls (`exp6_accent_control.py`, `exp6_stats.py`)
Method: MMS-1b-all (`npi`) grapheme-CTC forced alignment → word boundary → crop word (−60/+40 ms) → `wav2vec2-lv-60-espeak-cv-ft` on the **crop only** → first consonant label.
Positive control: `ब-` words must come out `[b]`. Corpora: Rasa F, Rasa M (train shards 30/31/40), SLR43 female (Nepal), SLR54 shard 0 (Nepal crowd). Kathmandu form of `वि-` is `[bi-]` (Wiktionary: विकास `[bikäs]`; Hindi `[ʋɪkɑːs]`).

| Corpus | n `वि-` words | `[v/w/ʋ]` | `[b]` | control `ब-`→`[b]` |
|---|---|---|---|---|
| Rasa **Female** | 52 | **73.1 %** | 17.3 % | 97.5 % |
| Rasa **Male** | 60 | **36.7 %** | 51.7 % | 77.5 % |
| SLR43 female (Nepal) | 60 | 20.0 % | 70.0 % | 85 % |
| SLR54 (Nepal) | 56 | 7.1 % | 76.8 % | 77.5 % |

Fisher exact on `[v/w]` rate: Rasa-F vs SLR43 p = 1.7e-8, vs SLR54 p = 4.2e-13; Rasa-M vs SLR54 p = 1.2e-4, vs SLR43 p = 0.067 (n.s.); Rasa-F vs Rasa-M p = 1.4e-4.
Readings: the recogniser *can* tell `[b]` (control 78–98 %), so the Rasa-F effect is not recogniser bias. Rasa-M is closer to Nepal speech than Rasa-F.
`ज्ञ` words: `[g…]` is the majority in all four corpora (n = 3–12 per corpus, too small to rank) and **never** `dʒ/ɲ` (0 of 24) → the repo G2P `ज्ञ → dz ɲ` is wrong for every corpus.
`च/छ` first-consonant labels are too noisy with this recogniser (family spread over ts/ʃ/t/tɕ) to say anything — **do not use this probe for affricates** (needs a purpose-built classifier).
Limits: ≤ 60 words per cell, one recogniser, SLR43/54 are crowdsourced (varied mics, accents of western/eastern Nepal); the claim is "Rasa-F differs from Nepal speech on `वि-`", not a full accent profile.

### Consequence for the plan (adopts Gemini's token trick, rejects its "Indian accent" framing)
* Treat accent as a **controllable variant, not a defect**: in training transcripts of Rasa-F, mark `व` as a separate phone `ʋ` **only where the acoustic probe says the speaker said `[v/w]`**, `b` otherwise; inference default is `b` (Nepal form) → the model learns clean `b` and clean `ʋ` embeddings and we choose.
* **Male Rasa is the lower-risk primary voice for the first accent-sensitive demo; female needs stage-2 adaptation** on SLR43 (Nepal female, 2,064 clips, 800 MB) — measure again after adaptation with this very probe (target: `[v/w]` ≤ 20 % on `वि-`).
* Never *mine* the final lexicon from Rasa audio (this half of Gemini's Trap S2 stands); use it only for per-utterance variant tagging.

## 2. EXP7 — Rasa text audit (`exp7_rasa_text_audit.py`, `results/s2/exp7.json`)
31,255 utts / 55.09 h. Unique sentences: F 15,818, M 15,112, union 23,648; **shared F∩M 7,282 (48.2 % of M)**.
Test split 3,027 unique sentences (F 1,602 + M 1,525 utts), **1,317 (43.5 %) also in train**. Train vocabulary 47,377 word types / 306,638 tokens, 50.6 % hapax; test token OOV rate 8.06 %; only 8,153 types occur ≥ 5 times (2,061 ≥ 20).
Symbols: 81 distinct chars (72 Devanagari). Rare: `ड़ ढ़ ऽ ॉ ॆ ॠ ऑ` (< 10 each). Latin 0, ASCII digits 0, Devanagari digits in 4 characters of 2 utterances.
Styles per voice (hours F/M): WIKI 10.4/10.1, CONV 3.2/2.9, NEWS 2.2/2.2, BOOK 2.1/2.2, PROPER NOUN 2.0/1.7 (mean 1.8 s, 34 % of F clips < 1 s — list intonation), six emotions ≈ 1.2 each, commands ≈ 1.3 total.
Median words/s: WIKI 1.47 F / 1.69 M, CONV 2.06 / 2.15, NEWS 1.62 / 1.67. 5 %–95 % duration: 0.92–14.7 s.
**Consequences:** (a) test set = our own sentence-disjoint hash split of the 23,648 unique sentences (hold out whole sentences for *both* voices) plus 500 written sentences with numbers/loanwords; (b) shared text lets us train a speaker-disentangled duration/prosody model and compare F/M on identical text; (c) 8 % OOV means the lexicon/G2P generalisation (not memorisation) decides quality; (d) numbers, units, English words need a front-end + synthetic text list we write — the data will not teach them; (e) exclude PROPER NOUN from the base stage (or give it its own style id).

## 3. EXP8 — Rasa audio audit (`exp8_rasa_audio_audit.py`, 320 random clips from 4 train shards)
All 320 clips are 48 kHz, 0 % clipped. Median (p05 / p95):
| | Female | Male |
|---|---|---|
| noise floor p5 (dBFS) | −71.2 (−79.5 / −44.6) | −72.2 (−76.5 / −64.5) |
| dynamic range p95−p5 (dB) | 51 | 55 |
| peak | 0.33 (0.19–0.72) | 0.52 (0.34–0.81) |
| leading / trailing silence | ≈ 0.01 s / 0 s | 0.01 / 0.03 s (p95 0.31 / 0.32) |
| median F0 | **251 Hz** (neutral 227, conv 264, anger 292) | **154 Hz** (neutral 147, anger 209) |
| F0 IQR (semitones) | 4.9 | 6.8 |
| net speaking rate (chars/s) | 11.0 | 11.9 |
| in-utterance gap fraction | 14 % | 19 % |
Reading: exceptionally clean; **clips are trimmed to ~10 ms edges**, so a model trained on them will start speech abruptly unless we add a controlled 80–150 ms silence token at the edges. Female F0 is high (227–264 Hz) → harmonic spacing is wide, which favours an explicit-F0/harmonic source (the tiny-voca design) over a mel→wave inverter that guesses phase. Female noise-floor p95 −44.6 dBFS shows a tail of noisier clips → needs the SNR filter. **Caveat:** the `roll99` column (99 % energy roll-off) is dominated by speech energy and says nothing about true bandwidth — whether the 48 kHz content is real above 12 kHz is still unmeasured (todo for CPU prep).

## 4. Verification & peer review of the Gemini lane
* **G2P list reproduced** (`word_to_phones`): `ज्ञान → dz ɲ a n`, `विद्यालय → w i d̪ j …`, `जानकारी → dz a n ʌ k a r i`, `गडबड → g ʌ ɖ ʌ b ʌ ɖ`, `संविधान → s ʌ̃ w i d̪ʱ a n`, `बाह्र → b a h r ʌ`, `खर्च → kʰ ʌ r ts`, `सन्देह → s ʌ n d̪ e h` … all as Gemini listed. Confidence: **certain** — `ज्ञ → g j` (EXP6 data), homorganic `संव- → [sʌmb-]`, `बाह्र/तेह्र/सोह्र → …[r]ʌ` (numerals), medial schwa syncope in `जानकारी`-type compounds. **Needs native adjudication before coding** — final `-ह` schwa (`सन्देह`, `समूह`), `खर्च` final schwa (Nepali often keeps a final ʌ after clusters), `व→b/w` per word (etymology/position: `विश्व` = [bisʋa]/[biswa] in the same word). The 300-word native review sheet stays the gate (G1); do not hand-patch the 6 categories wholesale.
* **`Sakhiya-Voca-4.5M` prototype** (`gemini/experiments/exp4_*`): parameters/size/RTF are measured, but the weights are random-init; "ASR/quality" claims about it are projections from tiny-voca. Treat it as a *shape-checked skeleton*; the first real evidence must be a CPU 200-step overfit on 32 aligned Rasa clips (`06 §A5`).
* **Harmonic-blur proof (Gemini EXP2)** is a signal-processing argument on synthetic data; it is consistent with our probes (default tier: 28 % less F0 movement, +41 % jitter) but is not yet confirmed on the shipped model's actual mels — keep as hypothesis H-F0 with a test (compare HPVR of `cand_b_v2` vs V1 on the same sentences).
* **Converged with Gemini:** reset recipe (tiny-voca topology, real data only, explicit prosody, GAN end-to-end, pretrained aligner), two voices, stop polishing `cand_b_v2`.
* **Corrections to my own docs:** (i) Rasa test split is contaminated (§2); (ii) accent of Rasa-F differs from Nepal on `वि-` (§1) — `04 §2.3` lexicon mining must exclude `व`; (iii) MMS-FA/espeak CTC label bias for `ʌ` (Gemini's LogSumExp mapping `ʌ ← {ʌ,ə,ɐ,ɔ,o,ɑ}`) is adopted for alignment.
* **Other data seen this session:** `IndicVoices-R` Nepali (104 h, 65.8 GB, CC-BY-4.0, multi-speaker, recorded in India) — a possible stage-1 *volume* source, **lower priority than Rasa and no accent advantage**. Edge TTS stays reference-only.

## 5. Tooling notes (so the next session does not lose an hour)
* The MoLab kernel is **shared with the Gemini agent**; `molab_exec.py` stdout can be replaced by the other agent's stale cell output and is **truncated to ~6 KB**. Write results to files, then read in a second call; never pipe large base64 through stdout.
* The remote repo clone is a sparse/partial checkout (`git add` of new dirs fails). Move results with the GitHub contents API from the box (token regex must take the quoted value from `/tmp/gitenv.sh`), then `git pull --rebase` locally.
* `create_file`/native tools take paths relative to `/app` (project root = `/app/Sakhiya/...`), not the repo root.
* OpenSLR mirror `openslr.elda.org` ≈ 10 MB/s vs `openslr.org` ≈ 0.16 MB/s; SLR43 has only a female zip; its index is inside the zip (`ne_np_female/line_index.tsv`).
* Rasa parquet is **sorted by gender**: train shards 0–28 female, 29 mixed, 30–56 male; `test-*` shards are 447 utts each (test-00000 is female only).
