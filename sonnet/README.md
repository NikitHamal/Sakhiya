# Sonnet lane — analysis, root causes and plan for a tiny, clean, two-voice Nepali TTS

Author: Claude (Sonnet lane). Scope: research + measurement + documents only. **No model, runtime, `docs/`, `cpp/` or `src/` file was changed.**
Environment: MoLab CPU box `sb-6afcbf0219f478ba` (20 cores, no GPU) for every experiment; all scripts re-runnable from `sonnet/experiments/`.
Honest limit: I cannot listen to audio. Everything below is measured with objective probes on real vs synthetic speech; the final word belongs to a blind human test (defined in `05`).

## One-paragraph verdict
We are behind because we kept polishing a **student of a synthetic teacher** (OmniVoice-600M fine-tuned on 1.94 h → distilled 4 M student → ~1–2 h real-speech polish, Seed-VC haze in between) while **55 h of native studio Nepali in both genders (Rasa: F 28.7 h, M 26.4 h)** sat unused as a backbone, and because **every gate we promoted on was blind to what makes speech native** (the pCER metric deletes schwa, geminates and /w/–/b/ before scoring; ASR CER of real humans, V1 and the default are all ≈0.08–0.10; UTMOS for the default in `tiers.json` is a hard-coded `3.94`, re-measured **2.64–2.81** vs V1 **4.07** vs real **3.79**). The current default talks 12 % too fast, almost never pauses (0.20 vs 1.73 per utterance), has 28 % less pitch movement and 41 % more jitter than real speech — worse than the older V1 on every probe. Meanwhile **tiny-voca (4.5 M, 2 voices, English) and tiny-voca-hindi (9.7 M, trained on Rasa Hindi, #1 of 16 Hindi models)** show the winning recipe: explicit duration/pause/F0/voicing/energy, differentiable source-filter output trained end-to-end with GANs, consistent real data, pretrained-recogniser forced alignment, tiny cost. Nepali has the same data (Rasa) and no such model.

## Documents
| File | Content |
|---|---|
| `01_codebase_audit.md` | What exists, lineage of the shipped weights, keep/archive/rewrite map |
| `02_root_causes.md` | Ranked causes with measured evidence (tables) + Edge TTS "bar" addendum |
| `03_landscape_research.md` | tiny-voca, tiny-voca-hindi, Alpa, Indic Tiny/Parler, Kokoro, Kitten… design rules |
| `04_plan_sakhiya_tiny_ne.md` | Data, front-end, model, training budget, gates, risks |
| `05_evaluation_protocol.md` | Frozen test set, blind human tests, automatic tripwires with numeric bands |
| `06_next_session_runbook.md` | CPU prep tasks now, hour-by-hour 12 h GPU plan, stop/go rules |
| `experiments/exp1..5` | Reproducible measurement scripts (real Rasa vs shipped vs Edge) |
| `results/*.json` | Raw outputs (`exp1` prosody/ASR, `exp2` G2P-vs-real phones, `exp3` Rasa inventory + UTMOS, `exp4` UTMOS on bench-200, `exp5` Edge) |

## Top findings (all measured here, N=40 Rasa test sentences unless stated)
1. UTMOS22: real 3.79 · V1 4.07 · Edge-F 3.42 · Compact 3.43–3.53 · Nano 3.75 · **default 2.64 (bench-200: 2.80)**; claim in `docs/tiers.json` = 3.94 (hard-coded literal, `scripts/promote_and_sync_cand_b_v2.py:92`).
2. Default vs real speech: pauses 0.20 vs 1.73 · rate +12 % · |ΔF0| −28 % · jitter +41 % · treble tilt 3 dB darker. V1 is within ±10 % of real on all of them.
3. Strict ASR CER (no phoneme fuzzing): real 0.090, V1 0.083, default 0.098 → intelligibility is saturated; **it cannot guide naturalness/accent work**.
4. The repo's `ne_pcer` removes `ʌ`, geminates, final halant and merges `व/ब` before scoring (`scripts/eval_v3.py`) — it is structurally blind to accent.
5. Rasa Nepali (EXP3): 31,255 utterances / 55.1 h, F 28.74 h + M 26.36 h, ≈20 h neutral+conv+news per voice, 6 emotions + command styles labelled. Our shipped AM's final stage used 998 real utterances.
6. G2P (EXP2) is *not* shown to be the dominant error at coarse resolution (PER vs real audio 0.335 vs 0.354 vs synth; vowel-count ratio 1.02) — but the probe cannot see schwa quality; a native review sheet is in the plan.
7. The best commercial Nepali reference (Edge Hemkala/Sagar) also barely pauses (0.03/utt) and is 16–26 % faster than the human reader — human-like phrasing is an opening.

## Recommendation (details in `04`)
**Reset to *Sakhiya-Tiny-NE***: tiny-voca topology (T4.5 ≈ 4.5 M and T10 ≈ 9.7 M), two voices (Rasa F/M) + style ids, real-data-only training, pretrained-aligner durations with explicit pauses, data-driven Nepali pronunciation lexicon, end-to-end source-filter + GAN, ≤ 11 MB INT8. Promotion only by blind human test; automatic metrics as tripwires with *real-speech-relative* bands. Run the CPU prep list (`06 §A`) before the GPU session; keep V1 as the incumbent until the new model wins CMOS.

## Session 2 (pickup) — new documents and measurements
| File | Content |
|---|---|
| `07_session2_findings.md` | Word-level accent probe **with Nepal controls** (Rasa-F `[v/w]` 73 % vs Nepal 7–20 % on `वि-`; Rasa-M 37 %), Rasa text audit (test split 43.5 % contaminated, F/M share 48 % of sentences, no digits/Latin), Rasa audio audit, verification/peer review of the Gemini lane |
| `08_revised_plan_v2_and_next_session_gates.md` | Final ranked causes, decisions taken (use Rasa, accent as controllable variant, male-first, own held-out split), unproven items with cheapest tests, CPU prep list v2, GPU gates |
| `experiments/exp6..exp8` + `exp6_aggregate.py`, `exp6_stats.py` | Reproducible scripts; raw outputs in `results/s2/` |

## Caveats & open questions for the owner
* **Gate G0 (accent)**: listen to raw Rasa F/M clips first. If the accent is not the target, we still use Rasa as the phonetic backbone and adapt to SLR43-6834.
* Rasa is CC-BY-4.0 (commercial use with attribution OK); Edge TTS outputs are reference-only, never training data.
* Credentials: tokens supplied for this session were written only to `/tmp` (local and MoLab), never into tracked files, logs or commits.
