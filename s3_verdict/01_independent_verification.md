# S3-01 · Independent Verification (what the judge re-ran on `main` @ `172424f9`)

All checks below were executed locally on the checked-out `main` (read-only, no training, no audio listened — text/phone/prose audit only). Commands and file:line pointers are given so either lane can reproduce in one shell.

## V1. Both lanes compile + JS parses

- `python3 -m py_compile gemini/experiments/*.py sonnet/experiments/*.py` → **OK** (8 + 9 files).
- `node --check docs/g2p.js`, `docs/tts.js` → OK (parity bridge intact).

## V2. Gemini EXP3 G2P bugs — REPRODUCED exactly (so Sonnet EXP2 "not dominant" is refuted)

Entry point is `word_to_phones` (`sakhiya/g2p.py:435`), not `g2p`:

| Word | I measured (`word_to_phones`) | Gemini claimed | Correct Kathmandu | Verdict |
|---|---|---|---|---|
| ज्ञान | `dz ɲ a n` | `dz ɲ a n` | `g j ã n` | **Bug confirmed (Cat1)** |
| विज्ञान | `w i dz ɲ a n` | `w i dz ɲ a n` | `b i g j a n` | **Bug confirmed** |
| जानकारी | `dz a n ʌ k a r i` | `dz a n ʌ k a r i` | syncope `dz a n k a r i` | **Phantom schwa confirmed (Cat3)** |
| सिंह | `s ĩ h` | `s ĩ h` | `s ĩ h ʌ` | **Dropped tatsama schwa (Cat4)** |
| संविधान | `s ʌ̃ w i d̪ʱ a n` | `s ʌ̃ w…` | `s ʌ m b i d̪ʱ a n` | **Missing homorganic closure (Cat5)** |
| बाह्र | `b a h r ʌ` | `b a h r ʌ` | `b a r ʌ` | **Epenthetic h (Cat6)** |
| खर्च | `kʰ ʌ r ts` | `kʰ ʌ r ts` | `kʰ ʌ r ts ʌ` | **Stripped native schwa (Cat4/LOAN collision)** |
| विद्यालय/विभाग | `w i …` both | identical [ʋ] realisation in Rasa | inference should default `b` | **Supports per-utterance tagging, not lexicon mining** |

Why Sonnet missed it: `sonnet/experiments/exp2_g2p_vs_real_phones.py:47-60 cls()` strips `~/ʰ/ʱ`, collapses `ʈ→t, ɖ→d, ɳ/ŋ/ɲ→n, ʃ/s→s, affricates→c`, maps `ɑ→a` — guaranteeing `nA_real=0` and a ~0.33 PER noise floor. Gemini EXP5's LogSumExp mapping (`ʌ→logsumexp(ʌ,ə,ɐ,ɔ,o,ɑ)`) is the correct fix; use it in `rasa_import.py`.

Note: current `G2P_LEX` is 246 entries and `विद्यालय ∉ LEX` today, so the live bug is the reverse of Gemini §1.2's snapshot (everything-`w` instead of split `b`/`w`) — but the mechanism (label-audio contradiction polluting the `b` embedding) is identical. Fix stands: orthographic `ब→b` / `व→w` + acoustic per-clip tag in training, `v_mode` flag at inference.

## V3. Sonnet RC1/RC3 — REPRODUCED (so Gemini's mechanism has a measured victim)

- `scripts/promote_and_sync_cand_b_v2.py:92` → `t_item["utmos"] = 3.94` **hard-coded literal confirmed**. Sonnet's re-measure (2.64 Rasa-40 / 2.80-2.81 bench-200) is therefore the citable number, not 3.94.
- `scripts/train_cand_b_v2_breakthrough.py:590,717` freeze `dur./f0_head./enc.f0_pred/enc.vuv_pred` + `:623,772` `p_enc=zeros` **confirmed** — exactly Gemini's unconditioned-F0 setup.
- `scripts/eval_v3.py:27` `व→ब`, `:58,63` `.replace("ʌ","")` **confirmed** — schwa/geminate/halant deleted before `ne_pcer`, so accent is unscored. Sonnet RC3 stands; Gemini RC2-blur is the acoustic consequence.
- `sonnet/results/exp1_real_vs_shipped.json` N=40 means match Sonnet's table (real 1.725 pauses vs synth 0.20-class, 6.51 vs 7.26 graphemes/s, ΔF0 0.596 vs 0.427, jitter 0.0142 vs 0.0201) — directionally identical to Gemini EXP2's blur/warble/flatness. **Two independent probes, same corpse.**

## V4. Rasa audits — BOTH confirmed, Sonnet's number cites

- `sonnet/results/s2/exp7.json`: test 3,027 unique sentences, **1,317 (43.5%) also in train**; F∩M shared 7,282 (48.2% of M). **Rasa's official split is unusable — sentence-hash split mandatory.** Both lanes now agree; lock `sha256(norm_phones)%1000` with F+M same split.
- `sonnet/results/s2/exp6_stats.json` + `exp6_summary.txt`: Rasa-F `वि-` 38v/9b/52 (73.1% [v/w]), Rasa-M 22v/31b/60 (36.7%), SLR43 12v/42b/60 (20%), SLR54 4v/43b/56 (7.1%); Fisher Rasa-F vs SLR43 1.7e-8, vs SLR54 4.2e-13; controls `ब-`→[b] 78-98%. **This replaces Gemini's 89.5% (17/19 clip-count, no control) as the citable accent number.** Trap S2 warning (never mine lexicon from Rasa) survives; framing changes from "Indian defect" to "controllable variant".
- `gemini/results/exp6…json`: Rasa F/M 48 kHz, −25/−26 dB 8-12 kHz air, 49/55 dB SNR, 2.15 pauses/utt vs SLR43 0.15 (med 0.0) vs SLR54 −90.7 dB air + 39.6 dB SNR. **Never pretrain-mix SLR43/54 confirmed.** (SLR43 line in that JSON says 48 kHz — treat as resampled; the pause/bandwidth conclusions are unaffected.)
- `ज्ञ` never `dʒ/ɲ` (0/24 across corpora) — both lanes agree G2P `ज्ञ→dz ɲ` is wrong everywhere.

## V5. Build readiness — Gemini prototype real, but untrained

- `gemini/results/exp4…json`: 4,534,795 params (emb 13K + enc 1,588,864 + dur/pause 123,906 + prosody 796,355 + acoustic 1,986,752 + head 25,862), 2 voices + ne/sa, ONNX 21.283 MB FP32 / 10.849 MB FP16 / 6.556 MB mixed-INT8, 95-161× RTF. Shapes + `py_compile` + JSON self-consistency check pass. **Size/speed citable; quality evidence zero (Sonnet 07 §0.6 stands).**
- `gemini/results/exp7…json` (−10.13 dB lifter, −65.3 dB buzz, −38 dB hum) and `exp8…json` (38.9→9.9 loss, 30 steps, 12 clips, exact-frame-sum 1.0, ±1.58-frame alignment) are **physics/optimisation smoke**, not listening proof. Require them as pre-GPU exit gates, not as quality claims.
- `tests/test_owner_pronunciation_fixes.py` + `node --check` still pass; no lane broke the runtime bridge.

## V6. What remains unproven (needs GPU + ears, not more CPU prose)

1. Native-speaker sign-off on `g2p_v4` golds (70 words → 300-word sheet, Sonnet A4).
2. Whether adaptive lifter + split-band-train survive full Rasa training (only synthetic probes so far).
3. Whether Rasa-F with `v_mode=kathmandu` satisfies the owner without SLR43 adaptation (Gate G0 blind listen).
4. Whether T4.5 vs T10 vs T10+styles wins on the frozen blind test (don't pre-pick).
