# Session 3 Verdict — Gemini vs Sonnet: Are Both Good? Which Is Best?

**Judge:** Session-3 independent review (read-only audit of `main` @ `172424f9`, reproduced key claims locally + CPU checks, no weights/`docs/`/`cpp/`/`src/` touched, no secrets in repo)
**Question asked:** *Independently read both Gemini and Sonnet things. Tell me which is the best one or are both good?*
**One-line answer:** **Both are good — and neither alone is enough. They converged on the same 5 truths, then specialised in opposite directions that the GPU session needs together.**

- **Gemini wins on *what to build*** (physics, G2P, architecture, code-level training fixes, working ONNX prototype, transplant convergence).
- **Sonnet wins on *how to know it worked*** (honest measurement, eval protocol, de-risking, accent statistics, runbook discipline).
- **Ship the union:** Gemini's `Sakhiya-Voca v2` + 7 fixes + pure-Rasa backbone, gated by Sonnet's frozen test set + blind listening + sentence-disjoint split + per-utterance accent tagging.

---

## 1. Scorecard (independent, evidence-linked)

| Dimension | Gemini (`gemini/01-06`, EXP1-EXP8) | Sonnet (`sonnet/01-08`, EXP1-EXP8) | Judge |
|---|---|---|---|
| Why we sound behind (root causes) | 7 causes + math proofs, HPVR −31.45 dB blur law, param autopsy (76.65% in Vocos) | 7 causes + measured prosody regression table (0.20 vs 1.73 pauses, +12% rate, −28% ΔF0, +41% jitter), hard-coded `utmos=3.94` catch | **Tie — both proven, I reproduced both** |
| G2P / accent phonology | 70-word audit: current G2P **8.57% word-acc / 22.12% PER → v4 100%/0%** across ज्ञ/व/syncope/ह/ं/h-clusters; LogSumExp fix for aligner schwa blindness | Coarse probe (PER 0.335, collapsed contrasts, 0 schwas detected) → concluded "G2P not proven dominant" | **Gemini wins — I reproduced every bug (`ज्ञान→dz ɲ`, `जानकारी→…nʌk…`, `सिंह→s ĩ h`, `संविधान→s ʌ̃ w…`, `बाह्र→b a h r ʌ`, `खर्च→kʰ ʌ r ts`)** |
| व→w/b accent forensics | EXP5: Rasa says [v/ʋ/w] 89.5% (17/19, clip-level, no control) → warned mining lexicon from Rasa bakes Indian accent | EXP6 word-cropped + Nepal controls: Rasa-F 73.1% [v/w] vs SLR43 20% vs SLR54 7.1% (Fisher p=1.7e-8/4.2e-13), Rasa-M 36.7% (closer to Nepal); ज्ञ never dzɲ (0/24) | **Sonnet's number is the trustworthy one; Gemini's trap warning stands.** Adopt Sonnet's method + Gemini's token trick |
| Data strategy | Session-2: pure Rasa F (28.74h) + M (26.36h), never touch SLR43/54 (0.15 pauses/utt, −90.7 dB brickwall) | Rasa backbone + Gate-G0 listen first, male-first demo, stage-2 adapt female to SLR43 if needed; found 43.5% test-train sentence overlap + 48.2% F/M shared sentences | **Both right; union is correct** (owner already approved Rasa; keep Sonnet's split/tagging/adapt-gate as safety) |
| Architecture | `Sakhiya-Voca-4.5M` built+exported: 4,534,795 params, 2 voices + ne/sa, 6.556 MB INT8 / 10.849 MB FP16, 96–161× RTF on CPU (I verified JSON; `py_compile` OK) | Correctly identified tiny-voca family + design rules, but built no prototype | **Gemini wins — only buildable blueprint** |
| Training physics | 7 code-level fixes from tiny-voca-hindi + alpa (adaptive lifter −10.13 dB, split-band −65.3 dB buzz, −38 dB frame hum, R-coloured transplant 38.9→9.9 loss in 30 steps) | Missed pitch-spread loss, GAN balancing, lifter, split-band-train (Gemini §4 Trap S4 correct) | **Gemini wins** |
| Eval / process | Tripwires noted, but no frozen set / blind test / leakage control | Frozen 500+100 set + CMOS≥+0.5 + MUSHRA-lite + accent checklist + tripwire table + "real next to model" + no-hand-literals rules; 12h hour plan with kill-gates | **Sonnet wins — this is what stops session-84 regressing again** |
| Honesty / limits | Strong proofs, but EXP7 dB numbers are signal-processing demos (not trained-model gains) and EXP8 is 12-clip/30-step smoke; 100% G2P gold needs native-speaker sign-off | Explicit N=40, one-speaker, proxy-metric, no-listening limits on every claim; "does not say" sections | **Sonnet wins on caution; Gemini needs Sonnet's gates** |

**Convergence (both independently found):** (1) default regressed prosody chasing grit metrics, (2) `ne_pcer` deletes the accent (I verified `replace("ʌ","")` + `व→ब`), (3) 55h Rasa unused while AM polished on ~1–2h real, (4) tiny-voca/alpa source-filter family is the target, (5) single VC female + frozen dur + unconditioned-F0 is the mechanism. I verified the hard-code (`promote_and_sync_cand_b_v2.py:92`), `p_enc=zeros` + frozen `dur./f0_head./enc.f0_pred` lines, and eval normalisation myself.

## 2. What each lane got wrong (and the fix)

1. **Gemini 89.5% → use Sonnet 73.1%/36.7%.** Clip-counting any b/v before i/j without word crops or Nepal controls overstates. Word-level + controls is the citable number. Gemini's conclusion (don't mine lexicon from Rasa audio; tag per-utterance, infer `b` by default) survives — only the number changes.
2. **Sonnet "G2P not dominant" → retracted by its own S2.** Coarse `cls()` collapsed dental/retroflex, nasals, sibilants, affricates and mapped `ɑ→a`, guaranteeing 0 schwas and a 0.33 noise floor. Gemini's 70-word audit + my reproduction prove systematic bugs. Fix: adopt `g2p_v4` + native-speaker 300-word sheet (Sonnet A4) before training.
3. **Gemini "never touch SLR43/54" vs Sonnet "adapt female to SLR43".** EXP6 proves SLR43/54 must never be *mixed into pretraining* (pause/bandwidth poisoning). Sonnet proves Rasa-F is accent-shifted on वि- (73% vs 7–20% Nepal). Union: pretrain pure-Rasa F+M with per-clip ʋ/b tags → inference flag `v_mode=rasa|kathmandu` (default kathmandu=`b`) → **only if** blind G0 prefers Nepal form, do a light stage-2 accent adaptation on SLR43-female with pause/bandwidth guards, re-measured by EXP6 probe (target ≤20% [v/w]).
4. **Untrained prototype ≠ quality proof (Sonnet 07 §0.6 stands).** EXP4 size/speed valid, quality evidence zero. Don't cite EXP4 as "sounds better". Cite it as "fits 6.5 MB + 96× RTF, ready to train".
5. **EXP7/EXP8 are de-risking, not victory.** −65 dB / −38 dB / 38→9 loss are optimisation/physics smoke on synthetic probes + 12 clips. Required before GPU, not a substitute for the blind test.

## 3. Unified decision for the 12h RTX 6000 Pro Blackwell session

**Architecture (Gemini):** `Sakhiya-Voca v2` — 4L RoPE text enc + dur/pause-BCE + 2L prosody transformer (log F0 + voicing + energy + pitch-spread loss w=2.0) + 5L acoustic transformer + Nepali 5-class split-band homomorphic source-filter head (adaptive lifter 0.65, 2.67 ms sub-frame bursts, breath-pulse + formant-sharpen, R-coloured articulatory transplant warm-start, dynamic GAN balancing 0.7, 19 ms alignment calibration, 64/96 ms edge room-tone). 2 voices (F_sakhiya + M_sagar) + style ids + ne/sa svara channel. Target ≤11 MB FP16 / ≤7 MB INT8, RTF ≤0.05 1-thread.

**Data (both):** Pure Rasa F+M backbone (48→24 kHz) with Sonnet's sentence-disjoint hash split (0% leakage, F+M same split), ASR-agreement + voicing + alignment-confidence filters, per-utterance ʋ/b variant tags (never a mined lexicon), Kathmandu `g2p_v4` at inference (default `b`), numerals/Latin/dates from front-end (Rasa teaches none — Sonnet EXP7).

**Gates (Sonnet, mandatory):** Frozen `eval/frozen_v1` (500 + 100 contrasts, hashed) → tripwires within ±tolerances of real → blind CMOS ≥+0.5 vs V1 *and* default + MUSHRA-lite ≥75% of hidden ref + accent checklist ≥90% + WER ≤ real+2pt. No promotion on UTMOS/pCER/grit. Every `tiers.json` number generated with script SHA + test hash. Three concurrent runs (T4.5 2-voice / T10 2-voice / T10+styles), kill losers at 40K/80K, EMA + INT8 parity check, listening page per 10K.

Details: `01_independent_verification.md` (what I re-ran) + `02_unified_gpu_plan.md` (hour-by-hour + file map + risks).

**Bottom line for the owner:** Keep both lanes. Fund Gemini's build with Sonnet's brakes. If you must rank: **Gemini = best builder, Sonnet = best judge. The GPU session needs the builder's hands and the judge's gates.**
