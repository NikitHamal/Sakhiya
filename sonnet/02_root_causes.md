# 02 · Why we are behind: ranked root causes, with evidence

Test bed: MoLab CPU box (20 cores). Sentences: 40 random neutral-style **test** sentences from `ai4bharat/Rasa` Nepali (female, studio),
same text synthesized by each system; plus the repo's own `eval/bench_general_200.json` (ne) for robustness.
Scripts: `sonnet/experiments/exp1..exp5`, raw results: `sonnet/results/`.
Limits: one speaker, N=40, objective proxies only, **no human listening**. Treat deltas as strong hints, not verdicts.

## RC1 — The shipped default is *worse* than older tiers on every probe I ran, and the project's metrics could not see it

| Probe (same 40 sentences) | Real Rasa female | **Default `cand_b_v2` (FP16)** | Older V1 (INT8, 16 MB) |
|---|---|---|---|
| UTMOS22 (English-trained proxy) | 3.79 | **2.64** | **4.08** |
| UTMOS22 on repo bench-200 (first 40 ne): INT8 / FP16 | – | **2.80 / 2.81** | 4.07 |
| `tiers.json` claim for the default | – | 3.94 (**hard-coded literal**, `promote_and_sync_cand_b_v2.py:92`) | 3.957 (reproduces: 4.07) |
| Pauses ≥150 ms per utterance | 1.73 | **0.20** | 1.50 |
| Mean pause length | 0.30 s | 0.05 s | 0.34 s |
| Speaking rate (graphemes/s, pauses excluded) | 6.51 | **7.26 (+12 %)** | 6.35 |
| Utterance duration (same text) | 6.82 s | 5.81 s (−15 %) | 7.14 s |
| Mean abs ΔF0 (st / 10 ms) – pitch movement | 0.596 | **0.427 (−28 %)** | 0.622 |
| F0 5–95 % range (st) | 11.4 | 9.7 | 11.7 |
| Jitter (local) | 0.0142 | **0.0201 (+41 %)** | 0.0163 |
| Shimmer (local) | 0.070 | 0.081 | 0.076 |
| HNR (dB) | 16.7 | 15.9 | 14.2 |
| Tilt 4–8 kHz vs 0–2 kHz (dB) | −23.2 | **−26.4 (duller by 3 dB)** | −24.0 |
| Strict ASR CER / WER (MMS-1b-all `npi`, no phoneme fuzzing) | 0.090 / 0.323 | 0.098 / 0.369 | 0.083 / 0.288 |

Reading: the default talks too fast, almost never pauses, has a flatter and less stable pitch line, and is darker. It reads like a
*compressed average* of its teacher: exactly what a 4-step deterministic student with a **frozen duration predictor** on
≈1–2 h of real speech produces. Each "breakthrough" session optimised upper-band sibilant/comb metrics
(`grit`, `sibSFM`, `HF_LSD`) while prosody silently regressed. Median F0 differs (219 vs 247 Hz) because the anchor speaker is a different person; that is not a defect.

(The strict-CER row is itself an important result: **a real human scores 0.090 and every shipped model scores 0.083–0.098 — ASR intelligibility is saturated and cannot rank naturalness or accent.** The project spent ~90 sessions steering by metrics of this kind.)

## RC2 — Training signal: teacher-of-a-teacher, VC haze, and ~2 h of real speech in the final AM

* Final AM polish: 2,500 steps, lr 3.5e-5, **998 native utterances** (cache of 1,425). Before that: distillation from a synthetic teacher (OmniVoice-600M fine-tuned on 1.94 h) and Seed-VC-converted audio (84.6 % of the S76 corpus). A student cannot exceed its teacher; the teacher's non-native prosody/accent leaks (the S66 render pipeline even needs a "Hindi-leak" retry gate).
* Meanwhile **Rasa Nepali = 55.1 h, 31,255 utterances, 48 kHz studio, F+M, labelled by style** (EXP3, no audio downloaded):

| Style (female) | h | Style (female) | h |
|---|---|---|---|
| WIKI (neutral reading) | 10.39 | PROPER NOUN (short names) | 2.01 |
| CONV | 3.20 | BOOK | 2.06 |
| NEWS | 2.22 | 6 emotions | 7.53 |
| ALEXA/BB/DIGI/UMANG commands | 1.31 | **Female total** | **28.74** (16,016 utts) |

  Male: 26.36 h, 15,239 utts, same structure (neutral ≈ 18.9 h). ≈ 20 h/voice of neutral+conversational+news speech — more than the 17.6 h on which tiny-voca (English, 4.5 M) reached UTMOS 4.43, and the same corpus on which **tiny-voca-hindi (9.7 M) ranks 1st of 16 Hindi models** on the Open TTS Leaderboard.
* The repo's own oracle (STATUS S81) already proved the vocoder was not the bottleneck (copy-synthesis ≈ real). The remaining error is in *text→prosody/duration/mel*, which is a **data and objective** problem, not a vocoder-polish problem.

## RC3 — The accuracy metric deletes the accent

`scripts/eval_v3.py` scores `ne_pcer` after `clean_asr_norm` + `ph_ne_mms`:
`व→ब` (merges /w/ and /b/), `([क-ह])्\1 → \1` (**collapses geminates**), strips final halant, `ph_ne(...).replace("ʌ","")` (**deletes schwa**), collapses repeated characters, merges compound words, then `min(1.0, ·)` and `0.05` for empty strings; the "anchor" floor (0.032) is the teacher itself; the 50-sentence set is reused across sessions (tuning on the test set). Schwa, gemination and /w/–/b/ are precisely the contrasts that make Nepali sound native. A model could drop every schwa and every geminate and lose **nothing** on this score. No MOS, no MUSHRA, no human A/B is in any promotion gate (STATUS repeatedly: "No human listening test was done").

## RC4 — Architecture/size mismatch and churn

* The "tiny" claim is misleading: shipped = 4.18 M AM **+ 13.5 M borrowed Vocos + adapter = 23 MB INT8 / 35.5 MB FP16**. tiny-voca is **4.5 M for everything, 18 MB FP32**, with a 26 K-parameter differentiable source-filter synthesizer and explicit duration/F0/voicing/energy — same bytes, far more controllable.
* A 4-step stochastic-trained flow whose output is evaluated deterministically (`mu_y`-start) is a classic recipe for *over-smoothed prosody* — consistent with RC1's ΔF0/pause numbers.
* Duration: `am.dur` frozen from an in-house mel-CTC aligner; tiny-voca documents that its *own learned aligner collapsed three times* and fixed durations only by forced alignment with a pretrained phone recogniser + empirical 19 ms boundary calibration + explicit pause tokens (≥75 ms silence). We have no equivalent timing validation.
* 8 architecture families in 8 days; INT8/Monarch/Hadamard compression applied *before* a quality baseline existed.

## RC5 — G2P: real work, but not (yet) shown to be the dominant error (EXP2)

Differential test with a language-agnostic phone recogniser on 60 real Rasa clips (coarse vowel/consonant classes):

| | value |
|---|---|
| PER(G2P target, **real** audio) | 0.335 |
| PER(G2P target, **synth** audio) | 0.354 |
| Vowel-class PER real / synth | 0.499 / 0.539 |
| Vowel count G2P / real-recognised / synth-recognised | 29.3 / 28.6 / 26.4 |
| Seconds per grapheme real / synth | 0.0910 / 0.0796 (synth −12.6 %) |

At this resolution the G2P target is about as far from the human audio as the synthesized audio is from the G2P (recogniser noise ≈ 0.33 dominates), and total vowel count agrees (ratio 1.02) — so there is **no evidence of gross systematic schwa over/under-retention**; the synthesizer realises ~8 % fewer vowels than humans (consistent with "too fast/reduced"). Honest limit: this recogniser never emitted a schwa-class symbol on Nepali speech, so it cannot adjudicate schwa quality; a native-speaker word-list review plus acoustic scoring of alternative pronunciations (see plan) is required. G2P remains a *risk* (hand-patched, 36-word owner checklist, Python↔JS duplicated), not a proven culprit.

## RC6 — Process and reproducibility

91 STATUS sessions / 550 commits in 8 days; training state lives in `/work/...` on ephemeral boxes; many "PROMOTED" decisions made on metrics in RC3. No frozen, hashed, never-trained-on test set; no blind listening protocol; claims in `tiers.json` not generated from measurement scripts (RC1).

## RC7 — Product mismatch

Goal: tiny, clean, **two voices (F/M)**, clearest Nepali accent. Repo: one VC-unified female voice, male retired, 23–35 MB default, accent never measured against real native speech.

## What the evidence does *not* say

* It does not say the Rasa speakers have the accent the owner wants (Rasa is an AI4Bharat/India collection — **listen first**, see plan Gate 0). If not, pretrain on Rasa and adapt to a Kathmandu speaker (SLR43 6834) — the acoustic/phonetic backbone still comes from the 55 h.
* It does not say V1 sounds better to humans; it says V1 matches real-speech prosody statistics and a MOS proxy far better than the default. Confirm by ear before relying on it.
* It does not say a 4.5 M model will match a 100 M model; it says tiny-voca-class models, trained on consistent data with explicit prosody, are the current efficiency frontier and Hindi/Rasa proves the recipe transfers to Devanagari.

## Addendum — the "bar" in Nepal, measured with the same battery (EXP5, same 40 sentences)

| System | UTMOS22 | strict CER / WER | pauses ≥150 ms per utt | graphemes/s | mean abs ΔF0 (st/10 ms) |
|---|---|---|---|---|---|
| Real Rasa female | 3.79 | 0.090 / 0.323 | 1.73 | 6.51 | 0.596 |
| Edge `ne-NP-HemkalaNeural` (F) | 3.42 | 0.084 / 0.346 | **0.03** | 7.55 | 0.478 |
| Edge `ne-NP-SagarNeural` (M) | 3.63 | 0.093 / 0.371 | **0.03** | 8.23 | 0.724 |
| Sakhiya-V1 INT8 | 4.08 | 0.083 / 0.288 | 1.50 | 6.35 | 0.622 |
| Sakhiya default `cand_b_v2` | 2.64 | 0.098 / 0.369 | 0.20 | 7.26 | 0.427 |

Two lessons: (1) UTMOS ranks V1 above Edge and above *real speech*, so it measures "clean and uniform", not "native" — it must not decide anything; (2) the commercial reference itself almost never pauses and speaks 16–26 % faster than the human reader. **Explicitly modelled, human-like phrasing/pausing is a measurable place where a small model trained on real speech can beat the best Nepali TTS available** (and V1 already shows the statistics are reachable). Whether that is *preferred* by listeners is exactly what the blind test (05) must establish.
