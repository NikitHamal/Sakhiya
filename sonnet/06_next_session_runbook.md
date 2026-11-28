# 06 · Runbook

## A. Before the GPU session (CPU / this MoLab box, no GPU needed) — do these first, they de-risk the 12 h

| # | Task | Output | Time |
|---|---|---|---|
| A1 | **Owner listening desk (Gate G0)**: 12 raw clips per speaker for Rasa F, Rasa M, SLR43-6834, FLEURS ne_np; blind-labelled; static page | `docs/listen_g0.html` + ratings JSON | 1 h build, 15 min owner |
| A2 | Download Rasa Nepali (≈ 18 GB parquet; token OK) to persistent storage + HF mirror; build `curated_24k` manifests (steps 1–5 of plan §1) with ASR-agreement filter (MMS-1b-all on CPU: ≈ 2.5 s/clip → use 16 procs; ≈ 3–5 h for 31 k clips, run detached) | `manifest.jsonl`, `rejected.jsonl` | overnight CPU |
| A3 | Forced-alignment pipeline (MMS_FA + espeak recogniser) dry run on 500 clips; voicing-flip probe; label coverage check for ʌ | alignment report | 2 h |
| A4 | **Single-source front-end**: normaliser tests + G2P → lexicon JSON → generated `docs/g2p.js`; candidate-pronunciation scorer; native-speaker review sheet (300 words) | `sakhiya/frontend/`, `eval/g2p_review.csv` | 1 day |
| A5 | Fork tiny-voca (Apache-2.0) into `sakhiya/tiny/`: Devanagari front-end hook, 2 voices + style ids, Nepali token table; unit tests (shapes, ONNX parity, 200-step overfit on 32 clips on CPU) | passing tests, CPU smoke loss curve | 1 day |
| A6 | Freeze test set + eval harness (`exp1` as library) + listening page for Gate G2 | `eval/frozen_v1/` | 0.5 day |
| A7 | Baseline bake-off on the frozen set: V1, default, Edge F/M, MMS-TTS (research), Indic Parler-TTS ne (GPU-free but slow: 3–5 s/sentence on 20 cores) | `eval/baselines.json` + audio | 3 h |

## B. GPU session hour plan (RTX PRO 6000 Blackwell, 12 h)

| Hours | Work | Exit criterion |
|---|---|---|
| 0:00–0:45 | Env, pull curated data to local NVMe, cache 24 kHz wav + mel + F0 + energy + alignments in RAM-mapped tensors | dataloader ≥ 40 it/s |
| 0:45–1:30 | 3 K-step smoke on (a) T4.5; verify: loss curves, teacher-forcing anneal, source-filter output audible, ONNX export parity ≤ 0.6 % peak | listenable (even if rough) audio |
| 1:30–2:00 | Launch **three concurrent runs**: (a) T4.5 2-voice; (b) T10 2-voice; (c) T10 2-voice + styles; 8 K spectral-only → GAN; checkpoints every 10 K | all three past 20 K |
| 2:00–8:00 | Monitor via EXP1 battery (CPU box pulls checkpoints) + frozen-listening renders each 10 K; kill the loser at 40 K and 80 K; try warm-start (tiny-voca/Alpa) as a 4th small run only if scratch loss at 10 K is poor | best config by tripwires + 5-person quick listen |
| 8:00–9:30 | Fine-tune best config: lower LR + EMA; optional accent adaptation (Gate G0 fail case) 2 h of SLR43-6834 on female | EMA vs raw A/B |
| 9:30–10:30 | Export ONNX (FP16 and INT8 w/ per-channel quantisation of transformer matrices only; keep predictors/synth FP32); measure parity (STFT-LSD, pitch, duration identical) and RTF 1-thread | `len_diff = 0`, RTF ≤ 0.05 |
| 10:30–11:30 | Gate G2 materials: render frozen set, build blind page, upload everything (HF + GitHub release); do **not** promote | page live |
| 11:30–12:00 | Write `sonnet/RESULTS_SESSION_N.md`; archive checkpoints + logs to HF; commit manifests/configs (no weights in git) | reproducible from git + HF |

Everything is launched detached (`setsid nohup … > logs/<job>.log`), state in git + HF, **never only in `/work`**.

## C. First 10 commands for next session
```bash
# 1. data (token in /tmp/hf_token): hf download ai4bharat/Rasa --repo-type dataset --include "Nepali/*" --local-dir /work/rasa_ne
# 2. python sakhiya/tiny/prepare.py --rasa /work/rasa_ne --out /work/ne_curated --asr mms-1b-all --max-cer 0.30
# 3. python sakhiya/tiny/align.py --manifest /work/ne_curated/manifest.jsonl --aligner mms_fa --probe voicing
# 4. python sakhiya/frontend/lexicon.py --score-variants --out eval/g2p_review.csv
# 5. python sakhiya/tiny/train.py --config configs/t45_2voice.yaml   (and t10_2voice, t10_styles)
# 6. python sonnet/experiments/exp1_real_vs_shipped.py REPO OUT 100   # tripwire battery per checkpoint
# 7. python scripts/make_listen_page.py --systems real,new,v1,default,edge_f,edge_m --n 24 --blind
# 8. python sakhiya/tiny/export_onnx.py --ckpt ... --fp16 --int8-matrices
# 9. node tests/test_tts_node.mjs   # ORT-web parity
# 10. python scripts/gen_tiers_json.py   # tiers.json generated from eval, no literals
```
(Scripts in items 2–5, 7–8, 10 do not exist yet; A4–A6 create them. Names are the proposed layout.)

## D. Stop/go rules
* If at 40 K steps no config is within the tripwire bands of `05_evaluation_protocol.md`, stop and diagnose (data/alignment), do **not** add architecture.
* If the blind test (G2) fails, the incumbent stays; the failure analysis goes into `sonnet/RESULTS_*`.
* No new architecture family without a written hypothesis linked to a measured failure of the previous one.
