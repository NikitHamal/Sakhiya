# S4-03 · Sonnet takes over the Voca training: 9× more data, honest throughput, held-out evaluation

Owner request: "why is Gemini using so little data — fix the training, take over". This note is the factual record of what I changed on the GPU box (`sb-155c91a9df7766f3`, RTX PRO 6000 Blackwell), why, and how to judge the result. I cannot hear audio; every claim below is a log reading or a probe.

## 1. Root cause of "so little data" (confirmed in code)

`scripts/prepare_rasa_voca_gpu.py` collected `max_clips * 1.15` encoded clips into RAM **before** processing and the launch command used `--max-clips 3000`. Result: 2,621 aligned utterances = **4.11 h of the 55 h** (24,156 / 37.7 h actually survive our length/alignment gates, see §2). The import itself is cheap: 3,000 clips took 52 s; the full set took **322 s** at 60–75 utt/s.

Fix (commit `bf58d954`): stream every cached Rasa Nepali parquet one file at a time (memory = one file), `--max-clips 0` = everything, shards stored as fp16 (wave + mel are 2× smaller; trainer casts per batch).

## 2. What the full cache contains (`/work/tts/s4/data/rasa_ne_full`, from `summary.json`)

| | old cache | **new cache** |
|---|---|---|
| aligned utterances | 2,621 | **24,156** |
| hours (24 kHz) | 4.11 | **37.66** |
| train / val / test (sentence-hash disjoint) | 2,493 / 47 / 81 | **22,896 / 616 / 644** |
| F / M utterances | 1,249 / 1,372 | 11,910 / 12,246 |
| mean CTC token prob | 0.724 | 0.728 |

Alignment-quality filter on train (measured on this manifest): prob ≥ 0.5 keeps 22,698 (35.5 h), ≥ 0.6 keeps 21,542 (34.2 h), ≥ 0.7 keeps only 15,104 (23.6 h). I train with `--min-prob 0.5` (drops the worst 0.9 % of alignments, keeps text diversity); a stricter filter would cost 34 % of the data for a modest alignment gain. Validation/test are never filtered, so numbers stay comparable.

## 3. Second data bug found while reading the manifest: 7 Rasa styles mislabelled as WIKI

The prep script maps styles through `STYLE2ID.get(style, 0)`. Rasa Nepali has **15** style tags; only 8 were in the dict. `PROPER NOUN` (2.3 k/voice), `SURPRISE`, `DISGUST`, `ALEXA`, `BB`, `DIGI`, `UMANG` all became id 0 (= WIKI = neutral encyclopedic). That silently labels ≈ 1.5 k emotive utterances per voice as neutral delivery. Fix (commit `37183e32`):

* `--styles auto`: the **4.5 M model has no style input** → it trains on **neutral styles only** (drops ANGER / FEAR / HAPPY / SAD / SURPRISE / DISGUST; 22,698 → **18,086** clips ≈ 28 h). A model that cannot be told "angry" must not average angry/sad/happy contours into its single prosody (this is also a plausible cause of the flat F0 phrase std Sonnet measured in S4-02 §3).
* The 9.5 M `t10_styles` model keeps all utterances with corrected style ids (SURPRISE→HAPPY, DISGUST→ANGER, PROPER NOUN→WIKI, DIGI→NEWS, ALEXA/BB/UMANG→CONV).

## 4. Trainer changes (`scripts/train_sakhiya_voca_gpu.py`)

* Background `BatchPrefetcher` thread (pinned CPU batches, 6 deep) replaces synchronous sample/pad/copy on the training thread; a single non-blocking H2D transfer replaces ~10 tiny copies per item.
* Pitch-spread loss/metric vectorised (`masked_pitch_std`, verified identical to `tensor.std()` on masked frames) — the old loop did 2 GPU syncs per utterance per step, i.e. 128 syncs/step at batch 64.
* RAM keeps fp16 wave/mel (37 h ≈ 8 GB per job).
* New flags: `--min-prob`, `--min-vowel-voiced`, `--styles`, `--run-name`, `--no-transplant`.
* `scripts/s4_progress.py launch` bug fixed: it blocked until the *job ended* (the backgrounded `cd && setsid … &` list kept the capture pipe open), so the marimo kernel timed out and **retried, launching duplicate jobs** (I hit this: three simultaneous full imports). Now `Popen(start_new_session=True)` returns the real pid at once.

## 5. Throughput — what the profile says (honest correction of my S4-02 §P2)

S4-02 asked for bf16 + a prefetcher expecting 3–5×. A `torch.profiler` step at batch 64 on the shared GPU shows **the GPU is genuinely busy**, not starved: top kernels are cuDNN convolutions (harmonic/noise filter + discriminators), fp32 `fmha_cutlass` attention backward (11.6 % of CUDA time) and elementwise `mul` (12.5 %); wall step times 0.25 s (bs 16) → 0.64 s (bs 64) while two other jobs were running. Batch size therefore buys *data per step*, not free speed: ≈ 80–100 samples/s per job after the change vs ≈ 56 before. The big win is data diversity (9×), not steps/second. bf16 attention is the remaining lever; I did not enable it because the harmonic oscillator/cepstrum head is precision-sensitive and I cannot A/B it by ear. Listed as a gated experiment in §8.

Epoch arithmetic (why speed is no longer the constraint): 4.5 M model at ≈ 1.2–1.5 it/s × 64 ≈ 77–96 samples/s → ≈ 2.2–2.7 M samples in the remaining ≈ 7.5 h ≈ **120–150 epochs of 18 k clips**. The old run had already seen ≈ 100 epochs of 2.5 k clips by step 15 K, i.e. memorised its text set.

## 6. Restart plan (executed)

Both jobs are restarted **on a fresh checkpoint boundary** (script waits for a ≤ 20 s-old `last.pt`) with `--resume`, so zero steps are lost and the 15 K steps of transplanted tiny-voca + GAN warm-up are kept:

| job | model | data | batch | steps | lr / half-life |
|---|---|---|---|---|---|
| `voca_t45_2v` | 4.54 M, F+M | neutral 18,086 clips | 64 | → 48 K | 3e-4 / 20 K (ends ≈ 6e-5, annealed) |
| `voca_t10_styles` | 9.69 M, F+M + 8 styles | all 22,698 clips | 32 | → 36 K | 3e-4 / 18 K |

The old schedule (half-life 25 K, 40 K steps at ≈ 3.5 it/s) was sized for the small set; the new half-lives are sized to *finish and anneal* inside the box lifetime (ends ≈ 00:15 UTC; I stop training by ≈ 22:30 to leave export/eval margin).

The training loss jumped when the data changed (t45 mel 0.55 → 0.69). **That is not a regression; it is the memorisation gap becoming visible**: 0.55 was the fit to 2.5 k seen sentences.

## 7. How to judge it (no new gameable numbers)

`sonnet/experiments/exp10_heldout_textmatched.py CKPT OUT.json` renders the *unseen* test sentences (644 in the new split; `--n` per speaker, neutral styles) and compares each with the **real recording of that same sentence**: DTW log-mel L1, duration ratio, median-F0 ratio, voiced-F0 std ratio. Baseline measured on the pre-restart EMA (step ≈ 16 K, mostly 4 h-trained), 20 clips/speaker: **F mel_dtw 1.891, dur 1.000, f0-med 1.026, f0-std 1.033; M mel_dtw 1.663, dur 0.963, f0-med 0.991, f0-std 0.801** (M F0 is 20 % too flat). The same script must show lower `mel_dtw` and `f0_std_r` closer to 1.0 on the new checkpoints; if it does not, the extra data did not help generalisation and I will say so. It is a tripwire, not a promotion criterion — Listen-page blind ears decide.

## 8. Next gates (ordered)

1. Re-run EXP10 at 24 K / 32 K / 40 K / 48 K steps; render the frozen test sentences F/M and put them on the Listen page blind next to REAL / V1 / default.
2. `len_diff=0`, size < 5 MB INT8 on the final export (already enforced in `export_onnx_models`).
3. A/B bf16 autocast on the attention stack only (keep oscillator/STFT/cepstrum fp32): promote only if `mel_dtw` and a 20-clip ABX are not worse, then it buys ≈ 1.5–2×.
4. Verify G2P parity (`tests/test_owner_pronunciation_fixes.py`) after any frontend change — `v_mode=rasa` (training) vs `kathmandu` (shipping) must differ only on `व` words.
5. Push final `inference_ema.pt` + ONNX to HF `NikitHamal/sakhiya-v1-train/voca/` (box-local token only).
