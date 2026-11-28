# 02 · The Seven Root Causes of Poor Voice & Unclear Nepali Accent (With Mathematical & Empirical Proof)

Every claim in this document is backed by reproducible scripts in `gemini/experiments/exp1..exp5.py` and measured JSON outputs in `gemini/results/`.

---

## Root Cause 1: The Inverted Parameter Budget & Zero-Attention Frame Decoder (`EXP1`)

In `cand_b_v2` (`Sakhiya-One Studio Vocos v2`, `23.01 MB` INT8 / `35.53 MB` FP16), **76.65% of all parameters (`13,531,650 / 17,652,875`)** live inside `PretrainedVocos24kHz`, leaving only **23.35% (`4,121,225` parameters)** to model all of Nepali and Sanskrit phonetics, timing, prosody, coarticulation, and spectrogram synthesis.

Even worse, look at how those `4.12M` Acoustic Model parameters in `SakhiyaOneAM` (`sakhiya/sakhiya_one.py`) are spent:
1. **SVD Rank Truncation in the Text Encoder (`student.py:239`)**: Every encoder Conv1d layer (`d_enc=144, ff_enc=224, k=5`) is factorized via truncated SVD to rank `r_enc = 104`.
2. **Zero Frame-Level Self-Attention (`sakhiya_one.py:178`)**: Once phoneme states are expanded to acoustic frames (`256` samples = `10.67 ms` per frame), `SakhiyaOneMeanFlowDecoder` (`d_vel=160, nb_vel=8`) processes frames using **only kernel-7 depthwise 1D convolutions** (`self.dw = nn.Conv1d(d, d, k=7, padding=3, groups=d)`).
   - **Total Receptive Field:** $8 \times (7 - 1) = 48\text{ frames} = 512\text{ ms}$ (`0.25 s` left and `0.25 s` right).
   - A frame decoder that can only see `256 ms` into the future **cannot form a natural sentence-level pitch contour, cannot anticipate clause-final cadence until the last two syllables (`phase >= 0.80`), and cannot maintain consistent vocal-tract formant trajectories across words**.
3. **Frozen Walsh-Hadamard + Monarch Block-Diagonal Approximation (`sakhiya_one.py:44`)**: Inside each velocity block, channel mixing is forced through a frozen $16 \times 16$ Sylvester Walsh-Hadamard rotation (`had_rot`) and grouped $1\times 1$ Monarch convolutions (`groups=15` and `groups=16`). Because mel bins `0..100` represent coupled vocal-tract formants ($F_1, F_2, F_3$) and harmonics ($k \cdot F_0$) that shift across non-adjacent channels as pitch moves, restricting channel mixing with block-diagonal Monarch factors smears inter-formant covariance.

---

## Root Cause 2: The Unconditioned-F0 Harmonic Blur Law (`EXP2`)

### 2.1 What `scripts/train_cand_b_v2_breakthrough.py` Actually Did in Code
Inspecting `scripts/train_cand_b_v2_breakthrough.py` reveals a fatal mathematical flaw in both Stage AM and Stage B2:
- **Line 590 & Line 717**:
  ```python
  if n.endswith("had_rot.weight") or n.startswith(("dur.", "f0_head.", "enc.f0_pred", "enc.vuv_pred")):
      continue
  ```
  All F0 and voicing heads (`f0_head`, `enc.f0_pred`, `enc.vuv_pred`) were **frozen during training** and **completely omitted from `SakhiyaOneVocosV2.infer()` (lines 162–260)**!
- **Line 623 & Line 772**:
  ```python
  p_enc = torch.zeros((B_c, L_c), device=dev)  # ZERO forced 0.14*sin hump!
  ```
  During all 2,500 steps of Stage AM and all 4,000 steps of Stage B2, the pitch input `p_enc` to the encoder was set to **all zeros (`0.0`)**!
- **Lines 241–249 (`mu_y`-Start Deterministic Iteration)**:
  ```python
  v_cur = self.am.vel(mu_y, t0, mu_y, g, m_t, prog=prog_4d, n_blocks=8)
  for s_i in range(1, 4):
      alpha_s = float(s_i) / 4.0
      x_s = mu_y + alpha_s * v_cur
      v_s = self.am.vel(x_s, t_s, mu_y, g, m_t, prog=prog_4d, r=r_one, n_blocks=8)
      v_cur = (1.0 - w2) * v_cur + w2 * v_s
  xt = mu_y + v_cur
  ```
  Because `x_0` is initialized deterministically to `mu_y` with **zero random noise** and **zero ground-truth $f_0(t)$ conditioning**, `G.am` is a **purely deterministic regression network mapping `(ids, dur) -> 100-band log-mel`**.

### 2.2 Mathematical Proof of Why Unconditioned-F0 Mel Regression Destroys Harmonics
In a 24 kHz, `n_fft=1024`, 100-band log-mel spectrogram, the lower 35 mel bins (`0 – 1,800 Hz`) have narrow bandwidths (`25 – 60 Hz`) that resolve individual harmonics $k \cdot f_0(t)$ of a human voice ($f_0 \approx 220\text{ Hz}$).
- When the true pitch of a training utterance is $f_0(t) = 215\text{ Hz}$, harmonics sit at $215, 430, 645, 860, 1075\text{ Hz}$.
- When another utterance has $f_0(t) = 235\text{ Hz}$ (`+1.5 st`), harmonics sit at $235, 470, 705, 940, 1175\text{ Hz}$ — **the peaks of one utterance fall directly into the inter-harmonic valleys of the other**!
- Because `G.am` is NOT given the target utterance's ground-truth $f_0(t)$ during Stage AM or Stage B2, `G.am` minimizes $\mathcal{L}_{L1}(\hat{Y}, Y_{\text{GT}})$ by predicting the conditional median/expectation $\mathbb{E}[Y_{\text{GT}} \mid \text{ids}, \text{dur}]$ across all pitches, **averaging harmonic peaks and inter-harmonic valleys into a flat, blurred formant smear**.
- Worse still: in Stage B2 (`lines 804–847`), `train_cand_b_v2_breakthrough.py` took those unconditioned-F0 AM predictions `mp_b`, passed them through `yp_all = G.vocos(G.adapter(mp_b))`, and computed narrowband `n_fft=1024, 2048` STFT magnitude + `<4 kHz` phase + Multi-Scale CQT discriminator losses against the **real ground-truth waveform `w_tgt_all`** (whose pitch $f_0^{\text{GT}}(t)$ does not match `yp_all`!). That gradient actively taught `Vocos` to inject broadband phase jitter and harmonic smearing so its output spectrum would overlap with mismatched-pitch ground truth.

### 2.3 Empirical Verification from `EXP2` (`gemini/results/exp2_acoustic_and_signal_forensics.json`)

| Condition / Model | Harmonic Peak-to-Valley Clarity `HPVR (0.2–2.6 kHz)` ↑ | Mid-Vowel Pitch Micro-Warble `(st)` ↓ | F0 Dynamic Spread `IQR (st)` ↑ | `3–5.5 kHz` Presence (rel. Formant) | `5.5–8.5 kHz` Sibilant (rel. Formant) |
|---|---:|---:|---:|---:|---:|
| **Controlled Proof: Homomorphic Source-Filter (`TinyVoca` `pulse_train` + `cepstrum`)** | **`46.717 dB` ✅** | **`0.0201 st` ✅** | — | — | — |
| **Controlled Proof: Unconditioned-F0 ($\pm 1.4\text{ st}$ Expected Spectrum -> Vocoder)** | **`15.266 dB` (`-31.45 dB` blur!) ❌** | **`0.0377 st` (`+87.6%` warble!) ❌** | — | — | — |
| **Staged `Sakhiya-Studio-80M + BigVGAN-v2` (32-step ODE, FP32)** | `22.634 dB` | `0.1115 st` | `2.879 st` | `-13.31 dB` | `-26.49 dB` |
| **Shipped `Sakhiya-V1` (`v15_v1_chandas`, 8-step Midpoint ODE)** | **`23.066 dB`** | `0.1015 st` | **`4.588 st`** | `-12.43 dB` | `-28.25 dB` |
| **Shipped Default `cand_b_v2` (Raw INT8 ONNX Output)** | **`20.636 dB` ❌** | **`0.1196 st` ❌** | **`2.900 st` (flat!) ❌** | **`-12.79 dB` (glare)** | **`-23.23 dB` (harsh)** |
| **Shipped Default `cand_b_v2` After `video/gen_audio.py` `clean()` Band-Aid** | `20.731 dB` | `0.1293 st` | `3.050 st` | **`-17.60 dB` (`-4.81 dB` cut!)** | **`-30.69 dB` (`-7.46 dB` cut!)** |

Look at the bottom two rows of `EXP2`: to make `cand_b_v2` listenable in the NEBians News video (`video/gen_audio.py`), the post-processing script had to slash `3–5.5 kHz` presence by **`-4.81 dB`**, slash `5.5–8.5 kHz` sibilants by **`-7.46 dB`**, slash `8.5–11.5 kHz` air by **`-6.84 dB`**, and boost `150–450 Hz` chest by **`+6.33 dB`** — which made the voice sound dark, muffled, and still left the `20.6 dB` blurred harmonics and `0.129 st` mid-vowel warble untouched!

---

## Root Cause 3: Frozen Single-Scalar Duration Predictor & Missing Pause-Presence Logit

1. **`am.dur` Was Frozen in `cand_b_v2`**: In `train_cand_b_v2_breakthrough.py` (line 590), `dur.` parameters were frozen while the minimum consonant floor `min_ph` was changed from `3 -> 1` frame (`10.7 ms`). Because `am.dur` was originally trained when `min_ph = 3` clamped short consonants upward, un-clamping `min_ph = 1` without retraining `am.dur` caused rapid consonant clusters to collapse to 1 frame (`+12%` faster speaking rate than human speech).
2. **Single-Scalar Duration Regression Destroys Interior Pauses**:
   - In `SakhiyaOneAM`, `am.dur` outputs a **single scalar `log_dur`** per token.
   - Word boundaries (`|`) and optional clause breaks have `0` frames of silence `~97%` of the time and `8–20` frames (`85–215 ms`) of silence `~3%` of the time (at prosodic phrase breaks).
   - A single-head regression network predicts the expected value $0.97 \times 0 + 0.03 \times 12 = 0.36\text{ frames}$, which rounds to `0` (`0.20` pauses $\ge 150\text{ ms}$ per utterance vs `1.73` in real human speech!).
   - Compare this to `n4ze3m/tiny-voca` (`src/tiny_voca/model.py:260` & `train.py:58–61`):
     ```python
     # Outputs per token: log(1 + frames) and, for pause-capable tokens, a "has frames at all" logit.
     # A pause is absent more often than present, so one regression would average both cases into a pause that is too short.
     present = batch["durations"] > 0
     duration = masked_mean((output[..., 0] - torch.log1p(batch["durations"].float())).square(), token_mask & present)
     pause = masked_mean(F.binary_cross_entropy_with_logits(output[..., 1], present.float(), reduction="none"), pausable)
     ```
     Separating **pause presence classification (BCE logit)** from **conditional pause length regression (MSE on `log1p(dur)` given `dur > 0`)** is why `tiny-voca` produces natural human-length breath pauses when `logit > 0` and crisp zero-gap word transitions when `logit <= 0`.

---

## Root Cause 4: Six Systematic Nepali Phonological Bugs in `sakhiya/g2p.py` (`EXP3`)

Previous sessions patched `sakhiya/g2p.py` (`docs/g2p.js`) one word at a time whenever the owner reported a mispronounced word in the 4 demo clips (`G2P_LEX` grew to 148 hardcoded entries). However, when we tested `sakhiya/g2p.py` in `EXP3` (`gemini/experiments/exp3_nepali_phonology_and_g2p_audit.py`) on **70 standard Nepali words across 6 core phonological rules**, current `sakhiya/g2p.py` passed only **6 / 70 words (`8.57%` word accuracy, `22.12%` Phoneme Error Rate)**:

| Category | Linguistic Rule in Standard Kathmandu Nepali (Khatiwada 2009; Pokharel 1989) | What Current `sakhiya/g2p.py` Outputs (Bug) | What Correct Kathmandu Nepali IPA Is (`g2p_v4_prototype`) | Current `g2p.py` Pass Rate | `g2p_v4` Prototype Pass Rate |
|---|---|---|---|---:|---:|
| **Cat 1: `ज्ञ` (`ज+्+ञ`)** | In Nepali, `ज्ञ` is ALWAYS velar-palatal **`[g j]`** (with nasalization on monosyllable `ज्ञान -> g j ã n`), NEVER palatal `[dz ɲ]`. | `ज्ञान -> dz ɲ a n`<br>`विज्ञान -> w i dz ɲ a n`<br>`वैज्ञानिक -> w ʌ i dz ɲ a n i k`<br>`विज्ञप्ति -> w i dz ɲ ʌ p t̪ i` | `ज्ञान -> g j ã n`<br>`विज्ञान -> b i g j a n`<br>`वैज्ञानिक -> b ʌ i g j a n i k`<br>`विज्ञप्ति -> b i g j ʌ p t̪ i` | **`0 / 10` (`0%`) ❌** | **`10 / 10` (`100%`) ✅** |
| **Cat 2: `व` (`[b]` vs `[w]`)** | Native Nepali/tatsama initial & stem `व` (`विद्यालय, विद्यार्थी, विचार, विभाग, विराटनगर, वीरगन्ज, वडा, वन, वर्षा, वरिष्ठ, व्यवस्था, व्यवहार, विवाद, विवाह, विमान`) is pronounced **`[b]`**; `[w]` occurs in loanwords (`वालेट, वेबसाइट`), `C+व` glides (`स्वागत, द्वारा`), and intervocalic `-आवा/-एवा` (`हावा, सेवा`). | Maps 100% of non-lexicon `व` to **`w`**, AND Session 80 hacked `w_e[id_w] = 0.78*w + 0.22*u - 0.16*b`, turning `-व` into `-उ` (`उतारचढाव -> उततार छढाउ`)! | Distinguishes native `[b]` (`विद्यालय -> b i d̪ j a l ʌ j ʌ`) from loan/glide `[w]` (`वालेट -> w a l e ʈ`, `स्वागत -> s w a g ʌ t̪`) in G2P with zero embedding hacks! | **`5 / 20` (`25%`) ❌** | **`20 / 20` (`100%`) ✅** |
| **Cat 3: Medial Schwa Syncope (`VC-CV`)** | Compound stems, derivational suffixes (`-कारी, -दार, -बाही, -नीति, -तन्त्र, -गन्ज, -घाट, -दान, -वार`), and `CVC-CVC` reduplicatives delete the stem-final medial schwa. | `_resolve` line 414 (`if not last: res.append("ʌ")`) retains 100% of medial schwas:<br>`जानकारी -> dz a n ʌ k a r i`<br>`कामदार -> k a m ʌ d̪ a r`<br>`सरकारी -> s ʌ r ʌ k a r i`<br>`राजनीति -> r a dz ʌ n i t̪ i`<br>`फेसबुक -> pʰ e s ʌ b u k` | `जानकारी -> dz a n k a r i`<br>`कामदार -> k a m d̪ a r`<br>`सरकारी -> s ʌ r k a r i`<br>`राजनीति -> r a dz n i t̪ i`<br>`फेसबुक -> pʰ e s b u k` | **`1 / 19` (`5.3%`) ❌** | **`19 / 19` (`100%`) ✅** |
| **Cat 4: Final `-ह` Tatsamas & Native `-र्च`** | Tatsama nouns in `-ह` (`सिंह, गृह, ग्रह, सन्देह, आग्रह, उत्साह, समूह, प्रवाह, तह`) and native `खर्च` retain final schwa `ʌ`. | Drops final `ʌ`:<br>`सिंह -> s ĩ h`<br>`गृह -> g r i h`<br>`उत्साह -> u t̪ s a h`<br>`खर्च -> kʰ ʌ r ts` (broken by `LOAN_FINAL_CL`) | `सिंह -> s i ŋ h ʌ`<br>`गृह -> g r i h ʌ`<br>`उत्साह -> u t̪ s a h ʌ`<br>`खर्च -> kʰ ʌ r ts ʌ` | **`0 / 10` (`0%`) ❌** | **`10 / 10` (`100%`) ✅** |
| **Cat 5: Anusvara (`ं`) Before `व, स, श, र`** | In Nepali, `सं+व` assimilates to bilabial **`[m b]`** (`संविधान -> s ʌ m b i d̪ʱ a n`, `संवाद -> s ʌ m b a d̪`), and `सं+स/श/र` has dental nasal closure **`[n]`** (`संसार -> s ʌ n s a r`). | Emits bare nasalized vowel with no nasal stop closure:<br>`संविधान -> s ʌ̃ w i d̪ʱ a n`<br>`संसार -> s ʌ̃ s a r` | `संविधान -> s ʌ m b i d̪ʱ a n`<br>`संवाद -> s ʌ m b a d̪`<br>`संसार -> s ʌ n s a r`<br>`संस्था -> s ʌ n s t̪ʰ a` | **`0 / 7` (`0%`) ❌** | **`7 / 7` (`100%`) ✅** |
| **Cat 6: `ह` + Sonorants (`बाह्र, तेह्र, सोह्र, चिह्न`)** | In Nepali numerals `बाह्र (12), तेह्र (13), सोह्र (16)`, `ह` marks historical breathiness (`b a r ʌ`, `t̪ e r ʌ`, `s o r ʌ`), not a full voiceless `[h]` segment before `[r]`. | `बाह्र -> b a h r ʌ`<br>`तेह्र -> t̪ e h r ʌ`<br>`सोह्र -> s o h r ʌ`<br>`चिह्न -> ts i h n ʌ` | `बाह्र -> b a r ʌ`<br>`तेह्र -> t̪ e r ʌ`<br>`सोह्र -> s o r ʌ`<br>`चिह्न -> ts i n h ʌ` | **`0 / 4` (`0%`) ❌** | **`4 / 4` (`100%`) ✅** |
| **Total (`70` Gold Words)** | — | **Word Acc: `8.57%` · PER: `22.12%` ❌** | **Word Acc: `100.0%` · PER: `0.00%` ✅** | **`6 / 70` ❌** | **`70 / 70` ✅** |

---

## Root Cause 5: Direct ASR Evidence of Acoustic/Phonetic Failures Inside `eval/forensic_cand_b_v2.json`

Inspecting `eval/forensic_cand_b_v2.json` lines 84–90 (the ASR transcripts of `cand_b_v2` on the Owner's own evaluation clips) exposes the exact acoustic consequences of Root Causes 1–4:
1. **`"म सखिया"` $\rightarrow$ heard by ASR as `"हसखिया"`** (`owner_clip3_intro_bulletin`): Initial bilabial nasal `म` collapsed into breathy `ह` because `onnx_tts.py` only prepended `[5, 1]` leading silence before unvoiced stops (`k, ts, t, p, s`), leavingvoiced nasal onset `म` to get clipped by the window ramp!
2. **`"त चलौँ ... लागौँ"` $\rightarrow$ heard by ASR as `"त चनौं ... लागुँ"`** (`owner_clip3_intro_bulletin`): Alveolar lateral `ल` (`l`) before nasalized diphthong `औँ` (`ʌ̃ ũ`) was blurred by the Monarch/Hadamard decoder into nasal `न` (`चनौं`), and `औँ` collapsed into monophthong `उँ` (`लागुँ`).
3. **`"अर्ब डलरको"` $\rightarrow$ heard by ASR as `"अर्म डलरको"`** (`owner_clip2_share_market`): Because `hop=256` (`10.67 ms`) with a 1-frame consonant floor cannot resolve the `r -> b` stop closure before `ɖ`, `ब` (`b`) lost its oral burst and sounded like nasal `म` (`अर्म`).
4. **`"उतारचढाव"` $\rightarrow$ heard by ASR as `"उततार छढाउ"`** (`owner_clip2_share_market`): Unaspirated `च` (`ts`) sounded like aspirated `छ` (`tsʰ`) due to coarse `10.67 ms` noise framing, and final `-व` (`w`) sounded like vowel `-उ` (`u`) due to Session 80's `+0.22 * w_e[id_u]` embedding hack!
5. **`"दार्चुला जिल्लामा ... चलिरहेछ"` $\rightarrow$ heard by ASR as `"दारचुला झिल्लामा ... छलिरहेछ"`** (`probe_chandra_chausir`): Unaspirated `ज` (`dz`) and `च` (`ts`) were heard as aspirated `झ` (`dzʱ`) and `छ` (`tsʰ`) because `Vocos-24kHz` lacked sub-frame voicing/noise boundary control.
6. **Palatal `श` (`ʃ`) vs Dental/Alveolar `स` (`s`) Erasure (`train_cand_b_v2_breakthrough.py:255`)**:
   ```python
   is_sib_fr = ((tok_t == id_s) | (tok_t == id_sh)).to(mel_vocos.dtype)[:, None, :]
   mel_vocos = torch.cat([mel_vocos[:, :50, :], mel_vocos[:, 50:100, :] + 0.90 * prof * is_sib_fr], dim=1)
   ```
   Adding the exact same static `+0.90 * sib_cavity_prof` vector (`~7.5 kHz` dental/alveolar peak, `-2.07` log-mel notch at `2.5–4.0 kHz`) onto **both** `s` (`स`) and `ʃ` (`श`) erased the acoustic distinction between palatal `श` (`विशेष, विश्व, शेयर`, which needs a `2.8–4.2 kHz` resonance) and dental `स` (`संसार, सुरक्षित`, which peaks at `6.5–8.5 kHz`)!

---

## Root Cause 6: The `ai4bharat/Rasa` Indian-Nepali Accent Trap vs Kathmandu Standard (`EXP5`)

In `EXP5` (`gemini/experiments/exp5_rasa_vs_kathmandu_accent_forensics.py`), we analyzed the actual human audio of `ai4bharat/Rasa` Nepali (`55.1 h`: `28.74 h` Female + `26.36 h` Male) using `facebook/wav2vec2-lv-60-espeak-cv-ft` and acoustic pitch/centroid probes:
- **Finding 6A (`व` Realization in `Rasa`)**: Across 20 `Rasa` clips containing native `व`-initial words (`विद्यालय, विद्यार्थी, विकास, विचार, विभाग, व्यवस्था, विज्ञान`), the `Rasa` speaker realized `व` before `इ/य` as **labiodental/velar `[v]/[ʋ]/[w]` 89.5% of the time (`17 / 19` instances)** and bilabial **`[b]` only 10.5% of the time (`2 / 19`)**, with a high female median F0 of **`273.1 Hz`** (vs `~218 Hz` for Kathmandu `SLR43 6834`).
- **Why?** Because `ai4bharat/Rasa` was recorded in India with Indian-Nepali speakers whose phonology reflects Indian `[ʋ]` (`"vidyalaya"`, `"vikas"`, `"vichar"`) rather than Nepal's Kathmandu standard `[b]` (`"bidyalaya"`, `"bikas"`, `"bichar"`).
- **Strategic Implication**:
  - `ai4bharat/Rasa` (`55.1 h` studio 48 kHz F & M) is a **tremendous multi-speaker studio pretraining backbone** for learning clean cepstral vocal-tract envelopes, consonant transitions, and expressive prosody.
  - **HOWEVER**, we must **never** use `Rasa` audio to mine `[w]` vs `[b]` lexicon pronunciations (which would lock in an 89.5% Indian `[ʋ]` accent!), and we **must** run Stage-2 Kathmandu Accent & Voice Lock on **Nepal-born Kathmandu speakers (`SLR43` Female `nep_6834`/`s0546`/`s3614` + `OpenSLR SLR54` high-SNR Nepal Male & Female speakers)** using our Kathmandu `g2p_v4` rules!

---

## Root Cause 7: Self-Deceiving Evaluation Metrics & Hardcoded Claims

1. **`scripts/eval_v3.py` Deleted the Accent Before Scoring**: `clean_asr_norm` and `ph_ne_mms` replaced all `व -> ब`, collapsed all geminates (`([क-ह])्\1 -> \1`), stripped all halants, and deleted every schwa (`.replace("ʌ", "")`). Consequently, a model could mispronounce every schwa, every geminate, and every `व/ब` contrast and still score `ne_pcer = 0.036`.
2. **Hardcoded `utmos: 3.94` in `docs/tiers.json`**: `scripts/promote_and_sync_cand_b_v2.py` (line 92) wrote the literal float `3.94` into `docs/tiers.json` without running `UTMOS22`. When actually measured on unseen sentences, `cand_b_v2` scores **`2.64 – 2.81`**, lower than `Sakhiya-One Compact` (`3.43 – 3.53`), `Nano` (`3.75`), and `Sakhiya-V1` (`4.07`).
