# 01 · Codebase audit (Sonnet lane, read-only review of `main` @ `7e760e03`)

Everything below was read from the repo, then checked with measurements (`sonnet/experiments/`, results in `sonnet/results/`).
Claims that I could **not** verify by measurement are marked *(unverified)*. I cannot listen to audio in this environment,
so nothing here is a listening verdict; the owner's ears remain the final judge.

## 1. What the repo is, in numbers

| Fact | Value | Source |
|---|---|---|
| Commits / calendar span | 550 commits, 2026-11-21 → 2026-11-28 (170 on 11-23 alone) | `git log` |
| Session entries in `STATUS.md` | 91 (1,147 lines) | `grep "^## Session"` |
| Python | 244 scripts + 35 modules in `sakhiya/`; ≈35k lines | `wc -l` |
| Model tiers kept in `docs/` | 14 `models_*` dirs, 40 sample-version dirs, 178 MB of samples | `du` |
| Shipped default | `cand_b_v2` = 4.18 M-param AM + 92.7 K adapter + **frozen 13.5 M pretrained Vocos**, 23.0 MB INT8 / 35.5 MB FP16 | `docs/tiers.json` |
| Architectures tried (each replaced the last) | FastPitch+HiFiGAN → CFM (`cfm.py`) → NSF-iSTFTNet (`v2_vocoder.py`) → source-filter wavehax (`v15_*`) → Walsh-Hadamard/Monarch MeanFlow (`sakhiya_one.py`) → 10-block iMF (`news_am_100mel.py`) → 83 M RoPE-DiT (`studio_dit_clean.py`) → Vocos+adapter (`vocos_24k_adapter.py`) | `sakhiya/` |
| Teachers/data generators used | Microsoft Edge TTS (37 h, early), Alpa (Sanskrit), OmniVoice-600M fine-tuned on **1.94 h** (Pro-v3), Seed-VC voice conversion, kNN-VC | `PROVENANCE.md`, STATUS S66/S75/S76 |

## 2. Pipeline of the shipped default, as actually built

```
text ──► sakhiya/g2p.py (rules + ~hand lexicon, Python)  ◄─ duplicated by hand in docs/g2p.js (parity test)
     └─► SakhiyaOneAM (4.18M; r_enc=104 SVD, WH+Monarch MLPs, 4-step deterministic MeanFlow, FROZEN am.dur)
            └─► 80-mel ──► Mel80To100Adapter (92.7K) ──► Vocos-24kHz (13.5M, pretrained on other data)
                                                                 └─► wav ──► (video only) broadcast DSP "clean()"
```

Lineage of the *acoustic model weights* (this is the critical part):

1. `Pro-v3` teacher = **OmniVoice-600M** (multilingual zero-shot TTS) fine-tuned on a **1.94 h** single-voice set (SLR43 speaker 6834 + Seed-VC-converted clips) — STATUS S66.
2. `Sakhiya-One` student distilled from Pro-v3 "rectified trajectories" (15.85 h of *teacher output*) — S75.
3. Student polished on 15–16 h mostly Seed-VC-converted Nepali (84.6 % VC) — S76.
4. `cand_b_v2` = steps 1–3 warm-started, then three more short stages: adapter 1,200 steps → AM **2,500 steps at lr 3.5e-5 on 998 native + anchor utterances** → joint B2 4,000 steps — `scripts/train_cand_b_v2_breakthrough.py` (`--b1_steps 1200 --am_steps 2500 --b2_steps 4000`, `real_gt_aligned_cache.pt` = **1,425 utterances**).

So the shipped voice's text→prosody→mel mapping is "a 4 M student of a synthetic teacher, polished on ≈1–2 h of real speech".

## 3. What is good and should be kept

* **Browser/CPU runtime**: ONNX + `docs/tts.js` + WASM/ggml engine + Android JNI exist and run (23 MB, ≈30× real-time 1-thread).
* **Real-speech corpora identified and licensed** (Rasa, SLR43, FLEURS, CV) and a data-provenance habit (`PROVENANCE.md`).
* **Honesty culture in some places**: the Session 76 audit fixed 4 bugs in the forensic gate (e.g. `ne_pcer` gap math); several sessions say "No human listening test was done".
* **Useful diagnostics** (comb-grit, sibilant flatness, HNR, CTC-based geminate ratio) — good as *regression tripwires*, not as quality targets.
* **Nepali text knowledge** in `g2p.py`/`numbers.py` (schwa retention lists, loanword clusters, acronym lexicon, numeral/`छ` cliticisation). This is real domain work; keep as the seed of the new front-end.
* **Video/news pipeline** (`video/`), web studio, listening desk (`docs/listen.html`) — keep, they are consumers.

## 4. What is structurally wrong (details and evidence in `02_root_causes.md`)

1. **Data lineage**: the voice is built from teacher output and voice-converted audio; 55 h of native studio Nepali (Rasa F 28.74 h / M 26.36 h, 31,255 utterances, gender- and style-labelled) are not the training backbone of the shipped model.
2. **Metric blind spots**: the primary accuracy metric deletes the very features that make an accent (schwa, geminates, `व/ब`, final halant) before scoring; UTMOS in `tiers.json` for the default is a **hard-coded constant** (`scripts/promote_and_sync_cand_b_v2.py:92`, `t_item["utmos"] = 3.94`), and my re-measurement gives 2.64–2.81.
3. **Regression hidden by those metrics**: the default has fewer pauses (0.20 vs 1.73 per utterance in real speech), 12 % faster speech, 28 % less pitch dynamics and 41 % higher jitter than real speech, and is *worse than the older V1 on every one of those probes*.
4. **Process**: 91 sessions in 8 days, each replacing the architecture before the previous one was understood; "polish on polish" warm-start chains; ephemeral boxes (`/work/...` paths) so results are not reproducible from git.
5. **Duplicated front-end** (Python + JS) maintained by hand, patched word-by-word from owner clips (36-word checklist) — maintenance burden and overfitting risk, but *not* the dominant cause according to EXP2.
6. **Single voice via voice conversion**; male voice removed; contradicts the product goal (2 voices) and adds VC artefacts (16 kHz roll-off, HNR 17.8 dB, STATUS S79 T1).

## 5. Keep / archive / rewrite map

| Area | Decision |
|---|---|
| `sakhiya/g2p.py`, `numbers.py`, `docs/g2p.js` | **Keep as seed; rewrite into one data-driven front-end** (single source → generated JS) |
| `docs/` web UI, `video/`, `cpp/`, `android/` | Keep (consumers; swap model later) |
| `sakhiya/sakhiya_one.py`, `news_*`, `studio_dit_clean.py`, `awp_vocoder.py`, Monarch/WH, MeanFlow, `v6_*`, `v15_*` | **Archive** (tag `pre-reset`); do not extend |
| Seed-VC/kNN-VC data pipelines | **Stop using** |
| `scripts/forensic_audio_gate.py` | Keep as tripwire; remove pCER/UTMOS from promotion gates |
| 14 tiers in `docs/` | Collapse to: current default + V1 (as measured baseline) until the new model beats both in a blind listening test |
