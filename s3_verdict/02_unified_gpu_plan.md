# S3-02 · Unified 12h GPU Plan — Gemini Build + Sonnet Gates (RTX 6000 Pro Blackwell)

This is the **single executable plan** for the next session. It keeps every proven idea from both lanes and drops every refuted one. No architecture churn, no metric gaming, no SLR poisoning, no test leakage.

## 0. Locked decisions (do not re-debate on GPU day)

1. **Backbone:** 100% pure `ai4bharat/Rasa` Nepali Female (28.74h/16,016) + Male (26.36h/15,239). No SLR43/54 in pretraining. Owner approved Rasa; EXP6 proves why mixing destroys pauses/air.
2. **Architecture:** Gemini `Sakhiya-Voca v2` (T4.5 2-voice + T10 2-voice + T10 2-voice+styles, all three launched; kill losers at 40K/80K). No new families mid-session.
3. **Front-end:** Kathmandu `g2p_v4` (Gemini EXP3) as single source → generated `docs/g2p.js` (Sonnet A4). Training transcripts use orthographic `ब→b`/`व→w` + per-clip acoustic ʋ/b tag; inference defaults `v_mode=kathmandu` (`वि-→b`, `वालेट→w`, `संविधान→sʌmb…`, `ज्ञ→gj`).
4. **Split:** Ignore Rasa's split. `sha256(norm_phones)%1000`, val 25‰ + test 25‰, F+M same split (0% leakage). Freeze `eval/frozen_v1` (500 + 100 contrasts, SHA-256, CI refuses edits).
5. **Promotion = blind ears only.** CMOS ≥+0.5 vs V1 *and* default, MUSHRA-lite ≥75% of hidden ref, accent ≥90%, WER ≤ real+2pt. UTMOS/pCER/grit are tripwires. No hand literals in `tiers.json`.

## 1. File map (where each piece lives)

| Piece | Source lane | Destination in repo |
|---|---|---|
| `Sakhiya-Voca v2` model + 7 fixes + transplant | `gemini/experiments/exp4.py` + `exp7.py` + `exp8.py` + `gemini/05` runbook | `sakhiya/tiny/` (fork tiny-voca Apache-2.0; hook Devanagari front-end, 2 voices + style ids, ne token table) + `sakhiya/frontend/` (single-source G2P → lexicon JSON → generated JS) |
| LogSumExp aligner fix + 19 ms calibration + pause tokens + 64/96 ms edge tone | `gemini/exp5` + `exp8` + `sonnet/07 §2-3` | `scripts/rasa_import.py` + `scripts/align_rasa.py` (MMS_FA + espeak recogniser, voicing-flip probe, ASR-agreement filter) |
| Frozen set + tripwire battery + listening page | `sonnet/experiments/exp1_real_vs_shipped.py` + `sonnet/05` | `eval/frozen_v1/sentences.jsonl` (+hash) + `eval/human/` ratings + `docs/listen_g0.html` / `docs/listen.html` extension |
| Baselines (V1, default, Edge F/M ref-only, MMS/Parler research) | `sonnet/06 A7` | `eval/baselines.json` + audio (rendered on CPU box while GPU trains) |
| INT8/FP16 export + parity + RTF | `gemini/exp4` + `sonnet/06 B` | `scripts/export_voca.py` (per-channel INT8 on transformers only; predictors/synth FP32; `len_diff=0`, RTF ≤0.05 1-thread) |

## 2. CPU pre-work (this MoLab box, no GPU — de-risks all 12h)

| # | Task | Exit gate |
|---|---|---|
| A1 | G0 listen desk: 12 blind clips × (Rasa-F, Rasa-M, SLR43-6834, FLEURS ne_np) | Owner picks primary demo voice (expect M or Rasa-F+kathmandu flag) |
| A2 | `rasa_import.py`: 48→24 kHz, LogSumExp alignment, per-clip ʋ/b tags, ASR-agreement + voicing + confidence filters, 64/96 ms room-tone pads, sentence-hash split | `manifest.jsonl` + `rejected.jsonl` + alignment report (mean_token_prob, vowel_voiced, ±frames) |
| A3 | `frontend/`: `g2p_v4` + lexicon JSON → generated JS; 300-word native review sheet | 300/300 signed; `pytest -q tests/` + `node --check docs/*.js` + `tests/test_web_studio.py` green, `BAD:0` cells |
| A4 | `sakhiya/tiny/` fork + 32-clip CPU overfit (200 steps) + ONNX parity ≤0.6% peak | Shapes/parity/loss-curve smoke green |
| A5 | Freeze `eval/frozen_v1` + bake baselines on CPU | `eval/baselines.json` ready before GPU hour 0 |

## 3. GPU hour plan (12h, single RTX 6000 Pro Blackwell)

| Hours | Work | Kill / pass |
|---|---|---|
| 0:00–0:45 | NVMe pull, cache 24 kHz wav + mel + F0 + energy + alignments (mmap tensors) | dataloader ≥40 it/s |
| 0:45–1:30 | 3K smoke on T4.5 (transplant vs scratch A/B) | Audible source-filter output; transplant must beat scratch as in EXP8 or debug aligner |
| 1:30–2:00 | Launch 3 runs (T4.5 2v / T10 2v / T10 2v+styles), 8K spectral-only → GAN; ckpt every 10K | All past 20K |
| 2:00–8:00 | Train; CPU box pulls ckpts → tripwire battery + 10K-interval blind renders | Kill loser at 40K and 80K; 4th warm-start run only if scratch loss poor at 10K |
| 8:00–9:30 | Best-config fine-tune (low LR + EMA); **conditional** SLR43-female accent adapt (2h max, pause/bandwidth guards) only if G0 failed Rasa-F | EMA vs raw A/B; EXP6 probe ≤20% [v/w] after adapt |
| 9:30–10:30 | Export FP16 + INT8 (transformers only), parity + RTF | `len_diff=0`, RTF ≤0.05, STFT-LSD + pitch + dur identical |
| 10:30–11:30 | G2 renders (frozen set), blind pages, HF + release upload | Materials committed with script SHA + test hash |
| 11:30–12:00 | Human mini-listen (5 quick + owner) + go/no-go vs V1/default | Promote only on §4 gates; otherwise keep incumbent frozen |

## 4. Gates (Sonnet 05, enforced)

Tripwires vs same-voice real (never "cleaner is better"): pauses ≥150 ms ±25%, rate/duration ±7%, |ΔF0| + F0-range ±12%, jitter/shimmer/HNR ±20% (lower-jitter = fail), tilt ±1.5 dB, strict ASR CER/WER (no ʌ/geminate/व→ब norm, 2 ASRs) ≤ real+1.5pt, UTMOS floor only (≥ real−0.4), alignment confidence ≥ real−0.05, geminate ratio ±15%, old forensic gate as tripwire. Human: CMOS ≥+0.5 (≥8 listeners incl. owner + ≥3 Kathmandu natives, 24 sents, no class <−0.3), MUSHRA-lite ≥75%, accent ≥90% (0 wrong on schwa/geminate/च-छ), WER ≤ real+2pt on 40 SUS sentences.

## 5. Risks + mitigations

| Risk | Mitigation |
|---|---|
| Rasa-F accent rejected in G0 | Male-first demo; `v_mode` flag; conditional SLR43 adapt with guards (never pretrain-mix) |
| Prototype underperforms T10 | All three run concurrently; kill-gates decide, not opinions |
| GAN collapse / aligner drift | Transplant warm-start, pitch-spread loss, dynamic GAN 0.7, 19 ms calibration, voicing-flip probe; scratch-vs-transplant A/B at 3K |
| Numerals/loans hallucinated | They come from `frontend/`, not data (Rasa: 2 digit-utts, 0 Latin); 80 number + 70 loan frozen items must pass checklist |
| Another "breakthrough" overwrite | `pre-reset` tag before session; archive zoo; collapse 14 tiers to incumbent + V1 until blind-beaten; `tiers.json` generator-only |

## 6. What success looks like tomorrow

A ≤11 MB FP16 / ≤7 MB INT8 ONNX that (a) fits the budget smaller than `cand_b_v2` FP16 (35.5 MB) with 2 voices, (b) matches real Rasa prosody statistics (pauses/rate/pitch/jitter) instead of gaming grit, (c) passes Kathmandu `ज्ञ/व/schwa/ं/h` contrasts by ear, (d) beats V1 *and* default blind — with every number reproducible from `eval/frozen_v1` + committed ratings JSON. Anything less stays in staging.
