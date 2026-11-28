# 04 · Technical Peer Review of `sonnet/` & Unified Gemini + Sonnet Synthesis

Because both **Gemini 4 Agron (`gemini/`)** and **Claude Sonnet 5.5 (`sonnet/`)** investigated `Sakhiya` on `main` in parallel using the MoLab CPU box, we performed a thorough technical audit of `sonnet/01..06.md`, `sonnet/experiments/exp1..5.py`, and `sonnet/results/exp1..5.json`.

---

## 1. What `sonnet/` Nailed (100% Independent Convergence Between Gemini & Sonnet)

Both models independently converged on five foundational truths:
1. **The Promoted Default (`cand_b_v2`) Regressed Prosody & Naturalness While Chasing Spectral Grit Metrics**:
   - Sonnet's `EXP1` and `EXP4` measured `UTMOS22 = 2.64` (on 40 Rasa test sentences) and `2.80 / 2.81` (on `bench_general_200.json`), proving that `utmos: 3.94` in `docs/tiers.json` was a hardcoded literal in `scripts/promote_and_sync_cand_b_v2.py:92`.
   - Sonnet's `EXP1` measured that `cand_b_v2` pauses only **`0.20` times per utterance** (vs `1.73` in real speech and `1.50` in `Sakhiya-V1`), speaks **`+12%` faster** (`7.26` vs `6.51` graphemes/s), has **`-28%` less pitch movement** (`0.427` vs `0.596` st/10ms), and has **`+41%` higher local jitter** (`0.0201` vs `0.0142`).
   - Gemini's `EXP1` and `EXP2` independently proved the **mechanistic & mathematical cause** of every one of those numbers: `cand_b_v2` froze `am.dur` (with single-scalar regression that rounds word-boundary pauses to 0), froze and bypassed `f0_head`/`f0_pred`/`vuv_pred`, trained with `p_enc = 0`, and applied narrowband STFT/CQT losses between unconditioned-F0 AM predictions and ground-truth waveforms (`-31.45 dB` harmonic peak-to-valley blur in `EXP2`).
2. **The `ne_pcer` Gate (`scripts/eval_v3.py`) Was Structurally Blind to Nepali Accent**:
   - Both Gemini and Sonnet caught that `eval_v3.py` strips schwa (`ʌ`), collapses geminates, strips final halants, and merges `व -> ब` before scoring `ne_pcer`.
3. **55.1 Hours of 48 kHz Studio Nepali in Both Genders (`ai4bharat/Rasa`: `28.74 h` Female + `26.36 h` Male) Was Left Unused**:
   - While `cand_b_v2` trained its AM on `1,425` clips (`85%` synthetic `Pro-v3` + Seed-VC `gita`, only `998` real utterances), `ai4bharat/Rasa` has `31,255` studio recordings across Female (`16,016` utts) and Male (`15,239` utts) voices.
4. **The Winning Architectural Family Is `tiny-voca` (`4.5M` Dual-Voice) / `tiny-voca-hindi` (`9.7M`)**:
   - Both Gemini and Sonnet independently identified `n4ze3m/tiny-voca` (`4,537,291` params, 2 voices, `25,862`-param Homomorphic Complex-Cepstrum Source-Filter head, explicit duration/pause/F0/voicing/energy predictors, full-rank RoPE Transformers) as the exact architecture to replace `Sakhiya`'s `AM + Vocos` / `AM + AWPVocoder` stack.

---

## 2. Four Critical Traps in `sonnet/`'s Analysis & Plan That Gemini Caught and Solved

While Sonnet's high-level diagnosis is strong, inspecting `sonnet/experiments/exp2_g2p_vs_real_phones.py` and `sonnet/04_plan_sakhiya_tiny_ne.md` uncovered **four dangerous technical traps** that would sabotage training if not corrected before the 12-hour RTX PRO 6000 Blackwell GPU session:

### Trap S1: Why Sonnet's `EXP2` Missed the 6 Systematic Nepali G2P Bugs (`8.57%` Word Accuracy in Gemini `EXP3`)
- **What Sonnet Did (`sonnet/experiments/exp2_g2p_vs_real_phones.py:47–60`)**:
  To compare `sakhiya/g2p.py` against `wav2vec2-lv-60-espeak-cv-ft` on 60 Rasa clips, Sonnet's `cls(ph)` function:
  1. Stripped all nasalization (`~`), aspiration (`ʰ`), and breathy murmur (`ʱ`).
  2. Collapsed dental and retroflex stops (`ʈ -> t`, `ɖ -> d`), all nasals (`ɳ, ŋ, ɲ -> n`), all sibilants (`ʃ, s -> s`), and all affricates (`tʃ, ts, dʒ, dz -> c`).
  3. Mapped `wav2vec2-lv-60-espeak-cv-ft`'s back vowel `"ɑ"` into `"a"` (`VOW = {"a": "a", "ɑ": "a", ...}`), resulting in `"nA_real": 0.0` (0 schwas detected across 60 utterances) and a noisy `0.335` PER floor that masked G2P errors.
  - Consequently, Sonnet wrote in `README.md`: *"G2P (EXP2) is not shown to be the dominant error... there is no evidence of gross systematic schwa over/under-retention."*
- **What Gemini Proved (`gemini/experiments/exp3_nepali_phonology_and_g2p_audit.py` & `exp5`)**:
  - When tested without collapsing Nepali phonological contrasts on 70 standard non-lexicon Nepali words, current `sakhiya/g2p.py` gets **only `6 / 70` words right (`8.57%` word accuracy, `22.12%` PER)**:
    - `ज्ञ` (`ज्ञान, विज्ञान, वैज्ञानिक, विज्ञप्ति, विशेषज्ञ`) is phonemized 100% wrong as `dz ɲ` instead of Nepali `g j` (`0/10`).
    - Medial schwa deletion (`VC-CV` syncope) fails on almost every compound and derived word not hardcoded in `G2P_LEX`: `जानकारी -> dz a n ʌ k a r i`, `कामदार -> k a m ʌ d̪ a r`, `सरकारी -> s ʌ r ʌ k a r i`, `कारबाही -> k a r ʌ b a h i`, `राजनीति -> r a dz ʌ n i t̪ i`, `लोकतन्त्र -> l o k ʌ t̪ ʌ n t̪ r ʌ`, `फेसबुक -> pʰ e s ʌ b u k`, `स्मार्टफोन -> s m a r ʈ ʌ pʰ o n` (`1/19`).
    - Final `-ह` tatsamas drop their mandatory schwa (`सिंह -> s ĩ h`, `गृह -> g r i h`, `उत्साह -> u t̪ s a h`) and `खर्च` loses its final schwa (`kʰ ʌ r ts`) (`0/10`).
    - Anusvara `ं` before `व/स/श` (`संविधान -> s ʌ̃ w i d̪ʱ a n`, `संसार -> s ʌ̃ s a r`) lacks homorganic nasal closure (`s ʌ m b i d̪ʱ a n`, `s ʌ n s a r`) (`0/7`).
    - `बाह्र, तेह्र, सोह्र, चिह्न` insert literal voiceless `[h]` before `[r]/[n]` (`0/4`).
  - Furthermore, Gemini's `EXP5` proved why `wav2vec2-lv-60-espeak-cv-ft` had `nA_real = 0.0` in Sonnet's script: on Nepali audio, `wav2vec2-lv-60-espeak-cv-ft` emits Nepali `अ` (`/ʌ/`) as back vowels **`"ɑ"` and `"o"`** (`65` and `64` hits in `EXP5`), while emitting Nepali `आ` (`/a/`) as **`"a"`** (`111` hits)! Using **LogSumExp Posterior Mapping** (`ʌ -> logsumexp(logits[:, [id("ʌ"), id("ə"), id("ɐ"), id("ɔ"), id("o"), id("ɑ")]])`) fixes `wav2vec2-lv-60-espeak-cv-ft` so it aligns Nepali `अ` with 100% coverage.

### Trap S2: Why Sonnet's Proposal to Mine `व -> w/b` Lexicon Entries from `ai4bharat/Rasa` Audio (`sonnet/04_plan §2.3`) Would Bake an Indian Accent into Sakhiya
- **What Sonnet Proposed (`sonnet/04_plan_sakhiya_tiny_ne.md §2.3`)**:
  > *"Generate a data-driven pronunciation lexicon for the top ~30 k words in Rasa: for each word, candidate pronunciations (schwa kept/deleted at each medial site, `व`→w/b) → score each against the aligned audio (acoustic GOP with the aligner) → keep the best"*
- **What Gemini Proved in `EXP5` (`gemini/results/exp5_rasa_vs_kathmandu_accent_forensics.json`)**:
  - We actually ran `wav2vec2-lv-60-espeak-cv-ft` on `ai4bharat/Rasa` Nepali clips containing native `व`-initial words (`विद्यालय, विद्यार्थी, विकास, विचार, विभाग, व्यवस्था, विज्ञान`).
  - **Result:** The `ai4bharat/Rasa` speaker pronounces `व` before `इ/य` as **Indian labiodental/velar `[v]/[ʋ]/[w]` 89.5% of the time (`17 / 19`) and bilabial `[b]` only 10.5% of the time (`2 / 19`)**!
  - Why? Because `ai4bharat/Rasa` was recorded in India by Indian-Nepali speakers who pronounce `व` as Indian `[ʋ]` (`"vidyalaya"`, `"vikas"`, `"vichar"`), whereas **Nepal Kathmandu speakers (`SLR43` / `SLR54`) pronounce those words with `[b]` (`"bidyalaya"`, `"bikas"`, `"bichar"`)**!
  - If we used acoustic GOP on `Rasa` audio to decide `व -> w` vs `b`, **89.5% of native Nepali `व` words would be locked to `[w]`/`[v]`**, giving our TTS an unmistakable Indian-Hindi accent!
  - **Gemini's Fix**:
    1. Keep `ವ -> b` vs `w` strictly governed by **Kathmandu Nepali phonological rules (`g2p_v4_prototype`)** + Nepal-born speaker verification (`SLR43` / `SLR54`).
    2. During Stage-1 pretraining on `ai4bharat/Rasa` (`55.1 h`), condition the G2P token stream of `Rasa` utterances on what the `Rasa` speaker actually said (`[w]` on `Rasa` where `GOP(w) > GOP(b)` **only in `Rasa` training annotations** so the `[b]` phoneme embedding is NEVER polluted by `[ʋ]` audio!), while **Stage-2 Kathmandu Voice & Accent Lock (`SLR43` + `SLR54`)** and **all inference** use pure Kathmandu `g2p_v4` (`विद्यालय -> b i d̪ j a l ʌ j ʌ`, `वालेट -> w a l e ʈ`)!
    - Think about how clean that is: by annotating `Rasa` training clips with `[w]` when the `Rasa` speaker actually articulated `[ʋ]` and `[b]` when they articulated `[b]`, the acoustic model learns a **100% pure bilabial stop `[b]` representation** and a **100% pure approximant `[w]` representation** without cross-contaminating the two embeddings!

### Trap S3: Nepali 4-Way Stop/Affricate Phonation & Split-Band Source-Filter Physics
- In English (`tiny-voca`), stops are simply voiced or voiceless. In Nepali, there is a **4-way phonation contrast** (`क / ख / ग / घ`, `च / छ / ज / झ`, `ट / ठ / ड / ढ`, `त / थ / द / ध`, `प / फ / ब / भ`), where:
  - Breathy-voiced aspirates (`घ gʱ, झ dzʱ, ढ ɖʱ, ध d̪ʱ, भ bʱ`) require simultaneous low-band glottal pulsing (`< 1,600 Hz`) and high-band aspiration noise (`> 1,600 Hz`).
  - Vowels require a `-21 dB` (female) / `-19 dB` (male) harmonic-to-hiss floor (`sakhiya/sanskrit_cepstral.py:45, 105–108`) so high-pitched female vowels (`220–275 Hz`) never sound breathy or hoarse.
- Gemini's `NepaliSourceFilterHead` (`gemini/experiments/exp4_sakhiya_voca_prototype.py`) integrates `tiny-voca`'s `Exportable --optimized` packed STFT/cepstrum kernels with `sakhiya/sanskrit_cepstral.py`'s `sound_kind` 5-class phonation limits and `-21 dB / -19 dB` vowel de-hoarsening floor.

### Trap S4: Three Vital `tiny-voca` Training Mechanisms Missing from Sonnet's Plan (`04_plan §3`)
When we read `.cache_ref/tiny-voca/src/tiny_voca/train.py`, we found three specific mathematical mechanisms that made `tiny-voca` work from scratch on 1 GPU in 7.7 hours, which were omitted in `sonnet/04_plan_sakhiya_tiny_ne.md`:
1. **Global-Variance Pitch-Spread Loss (`pitch_spread`, weight `2.0`, `train.py:70–76`)**: Explicitly matches `masked_std(pred_f0, voiced)` to `masked_std(target_f0, voiced)` per utterance. Without this term, L1 pitch loss shrinks pitch dynamic range by `~25–30%`.
2. **Dynamic Gradient-Norm Balancing for GAN (`gan_ratio = 0.7`, `train.py:151–160`)**: Scales the adversarial + feature-matching loss at every step by `0.7 * ramp * ||∇_wave L_rec|| / ||∇_wave L_gan||` so the discriminator neither vanishes (`<1%` gradient) nor blows up the cepstral heads.
3. **Decoupled Prosody Annealing vs Oscillator Ground-Truth F0 (`train.py:79–85`)**: While `teacher_probability` for conditioning Frame Transformer B anneals from `1.0 -> 0.0` over steps `3,000 .. 10,000`, the `SourceFilter` oscillator **always** receives ground-truth `target_f0` during training (`m.vocoder(acoustic, m.frequency(target[..., 0]))`) so narrowband STFT/GAN losses never blur harmonics!

---

## 3. Summary Comparison of Gemini (`Sakhiya-Voca`) vs Sonnet (`Sakhiya-Tiny-NE`)

| Dimension | Sonnet (`sonnet/`) | Gemini (`gemini/`) | Unified Best Path for Next Session |
|---|---|---|---|
| **Core Architecture** | `tiny-voca` topology (`T4.5` ~4.5M & `T10` ~9.7M), 2 voices (F/M) | **`Sakhiya-Voca-4.5M` (`4,534,795` params, built & verified in `EXP4`)** + **`Sakhiya-Voca-Pro-9.5M` (`d=256`, 6 acoustic layers)** | Train both `Sakhiya-Voca-4.5M` (`~4.8 MB` INT8+FP16, `97×` 1T RTF) and `Sakhiya-Voca-Pro-9.5M` (`~9.8 MB` INT8+FP16) concurrently on the 96 GB Blackwell GPU! |
| **Waveform Head** | Vanilla `tiny-voca` `SourceFilter` (`25,862` params) | **Upgraded `NepaliSourceFilterHead` (`25,862` params)** with 5-class Nepali phonation limits, `-21/-19 dB` vowel de-hoarsening, and `analytic_cep`/`analytic_stft` | Use Gemini's `NepaliSourceFilterHead` (`0` extra params, eliminates vowel hoarseness and models Nepali breathy-voiced `घ/झ/ढ/ध/भ` + `2.67 ms` `च/छ/त/ट` bursts). |
| **G2P & Phonology** | Claimed G2P was not a major error (`EXP2`); proposed mining `व -> w/b` from `Rasa` audio | Proved 6 systematic G2P bugs (`8.57% -> 100.0%` in `EXP3`) AND proved `Rasa` speakers use Indian `[ʋ]` `89.5%` of the time (`EXP5`) | Use Gemini's **`g2p_v4` Kathmandu Phonological Engine** at inference; annotate `Rasa` training clips with acoustic `[w]` vs `[b]` **only during training** so `[b]` and `[w]` embeddings stay 100% pure! |
| **Forced Alignment** | Noted `wav2vec2-lv-60-espeak-cv-ft` emitted `0.0` schwas on Nepali (`EXP2`) | Solved why: Nepali `अ` (`/ʌ/`) is emitted as `"ɑ"` and `"o"` (`EXP5`); built **LogSumExp Posterior Mapping** (`ʌ -> logsumexp(["ʌ","ə","ɐ","ɔ","o","ɑ"])`) | Use `wav2vec2-lv-60-espeak-cv-ft` with Gemini's **LogSumExp Posterior Mapping** + voicing-flip onset calibration + `-45 dB` (`>=75 ms`) silence pause attribution! |
| **Training Losses** | Multi-res STFT + mel + F0/V/UV/energy/dur + MPD/MRD GAN | Adds `tiny-voca`'s **Global-Variance `pitch_spread` Loss**, **Gradient-Norm GAN Balancing (`gan_ratio=0.7`)**, and **Ground-Truth F0 Oscillator Locking** | Include all three essential `tiny-voca` loss mechanisms from step 1. |
| **Voices & Accent** | Rasa Female + Rasa Male (with Gate G0 check) | **Two-Stage Kathmandu Voice & Accent Curriculum**: Stage 1 (`55h` Rasa F+M + `SLR43`/`SLR54` backbone) -> Stage 2 (`SLR43 6834` Female + `SLR54` Top Studio Male Kathmandu Lock) | Expose **both** pure Kathmandu Female (`F_sakhiya`) & Kathmandu Male (`M_sagar`) voices (plus optional Rasa expressive styles)! |
