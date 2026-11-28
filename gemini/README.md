# Gemini Research Lane (`gemini/`) — Forensic Autopsy, Mathematical Proofs & `Sakhiya-Voca` 4.5M Dual-Voice Breakthrough Blueprint

**Author:** Gemini 4 Agron (`gemini/` lane on `main`)  
**Compute Used for Empirical Proofs:** MoLab 20-Core CPU Sandbox (`sb-6afcbf0219f478ba.sb.molab.run`, `/work/gemini/`)  
**Zero-Touch Contract Verified:** No model weights, `docs/`, `cpp/`, or `src/` files were overwritten; all secrets remained strictly in `/tmp/`.

---

## Executive Verdict: Why Sakhiya Sounds Behind, Unclear & Non-Native — And How We Fix It Permanently

Over 83 sessions, `Sakhiya` chased incremental spectral/comb-grit patches on top of a **fundamentally mismatched architecture and training pipeline**, while the true root causes of poor voice quality and unclear Nepali accent remained unaddressed:

1. **The Inverted Parameter Budget & Zero-Attention Frame Decoder (`EXP1`)**:
   - Our current Studio Default (`cand_b_v2`, `23.01 MB` INT8 / `35.53 MB` FP16) has **17,652,875 parameters**, but **76.65% (`13,531,650` params)** sit inside the `PretrainedVocos24kHz` mel-to-wav inverter while only **23.35% (`4,121,225` params)** model language, timing, prosody, and acoustics.
   - Worse, that `4.12M` acoustic model has **ZERO frame-level self-attention** (`SakhiyaOneMeanFlowDecoder` uses 8 depthwise `k=7` Conv1d blocks with a total receptive field of only `48 frames = 512 ms`), its encoder is rank-truncated (`d_enc=144, r_enc=104` SVD), and its channel mixing is restricted by frozen Walsh-Hadamard + Monarch block-diagonal approximations.
   - By contrast, **`n4ze3m/tiny-voca` (`4,537,291` params total, 2 voices F2 & M2, `0.74%` WER, `4.43` UTMOS22)** and **`77ethers/alpa` (`4.5M` params, 2 voices)** put **99.43% (`4,511,429` params)** into **11 full-rank RoPE Self-Attention Transformer blocks** (`4` text encoder + `2` prosody frame transformer + `5` acoustic frame transformer, `d=192, heads=4, ff=640`) and only **0.57% (`25,862` params)** into a **Homomorphic Complex-Cepstrum Source-Filter Waveform Head** with an analytical Dirichlet harmonic pulse train (`0` neural vocoder phase-guessing buzz!) and `2.67 ms` (`noise_hop=64`) sub-frame consonant burst control.

2. **The Unconditioned-F0 Harmonic Blur Law (`EXP2`)**:
   - In `cand_b_v2` (`scripts/train_cand_b_v2_breakthrough.py`, line 590), `dur.`, `f0_head.`, `enc.f0_pred`, and `enc.vuv_pred` were **completely frozen and bypassed in `infer()`**, and training ran for 6,500 steps with `p_enc = 0` (zero pitch conditioning).
   - When a deterministic (`mu_y`-start) acoustic model predicts 100-band log-mels from phonemes + durations **without conditioning on the target recording's ground-truth $f_0(t)$**, L1 regression converges to the conditional expectation $\mathbb{E}[\text{log-mel} \mid \text{phonemes}]$ averaged over natural pitch variation ($\pm 1.4\text{ semitones}$).
   - Our controlled signal-processing proof in `EXP2` demonstrates that averaging log-spectra over $\pm 1.4\text{ st}$ of unconditioned F0 variation **collapses Harmonic Peak-to-Valley Ratio (`HPVR`) by `-31.45 dB` (`46.72 dB` $\rightarrow$ `15.27 dB`)** andnearly doubles mid-vowel pitch warble (`0.0201 st` $\rightarrow$ `0.0377 st`).
   - Then, in Stage B2, `train_cand_b_v2_breakthrough.py` applied narrowband `n_fft=1024, 2048` STFT + `<4 kHz` phase + CQT losses between unconditioned-F0 AM predictions and ground-truth waveforms—actively training `Vocos` to smear harmonic ridges into noisy formant bands. To make `cand_b_v2` broadcast-tolerable for NEBians News Ep.1, `video/gen_audio.py` (`clean()`) had to apply a massive post-hoc DSP band-aid (`-4.81 dB` cut in `3–5.5 kHz`, `-7.46 dB` cut in `5.5–8.5 kHz`, `-6.84 dB` cut in `8.5–11.5 kHz`, plus `+6.33 dB` chest boost!).

3. **Six Systematic Nepali Phonological G2P Bugs (`EXP3`: Current G2P Scores `8.57%` Word Accuracy / `22.12%` PER on 70 Non-Lexicon Nepali Test Words -> Fixed to `100.0%` / `0.00%` PER in `g2p_v4_prototype`)**:
   - **Cat 1 (`ज्ञ` = `0/10` pass)**: `sakhiya/g2p.py` parses `ज्ञ` (`ज+्+ञ`) character-by-character into **`dz ɲ`** (`ज्ञान -> dz ɲ a n`, `विज्ञान -> w i dz ɲ a n`, `वैज्ञानिक -> w ʌ i dz ɲ a n i k`) instead of native Nepali **`g j`** (`g j ã n`, `b i g j a n`)!
   - **Cat 2 (`व` `[b]` vs `[w]` Conflation = `5/20` pass)**: `sakhiya/g2p.py` maps every Devanagari `व` to `"w"` (`विद्यालय -> w i d̪ j a l ʌ j ʌ`, `विद्यार्थी -> w i d̪ j a r t̪ʰ i`, `विचार -> w i ts a r`, `वडा -> w ʌ ɖ a`), and Session 80's embedding hack (`0.78*w + 0.22*u - 0.16*b`) turned word-final `-व` into `-उ` (causing ASR on Owner Clip 2 to hear `उतारचढाव` as **`उततार छढाउ`**!).
   - **Cat 3 (Medial Schwa Syncope `VC-CV` = `1/19` pass)**: Because `_resolve` blindly retains 100% of medial inherent schwas (`if not last: res.append("ʌ")`), every compound, suffixed, or 4-consonant stem outside the 120-word hardcoded `G2P_LEX` inserts robotic phantom schwas: `जानकारी -> dz a n ʌ k a r i`, `कामदार -> k a m ʌ d̪ a r`, `सरकारी -> s ʌ r ʌ k a r i`, `कारबाही -> k a r ʌ b a h i`, `राजनीति -> r a dz ʌ n i t̪ i`, `लोकतन्त्र -> l o k ʌ t̪ ʌ n t̪ r ʌ`, `नेपालगन्ज -> n e p a l ʌ g ʌ n dz`, `फेसबुक -> pʰ e s ʌ b u k`, `स्मार्टफोन -> s m a r ʈ ʌ pʰ o n`, `गडबड -> g ʌ ɖ ʌ b ʌ ɖ`!
   - **Cat 4 (Final `-ह` Tatsama Schwa Drop & `-र्च` Collision = `0/10` pass)**: `सिंह -> s ĩ h`, `गृह -> g r i h`, `ग्रह -> g r ʌ h`, `सन्देह -> s ʌ n d̪ e h`, `उत्साह -> u t̪ s a h`, `समूह -> s ʌ m u h` all drop their mandatory word-final Nepali schwa `ʌ`; meanwhile, adding `र्च` to `LOAN_FINAL_CL` in Session 83 stripped the final schwa off native Nepali `खर्च` (`kʰ ʌ r ts` instead of `kʰ ʌ r ts ʌ`)!
   - **Cat 5 (Anusvara `ं` Homorganic Nasal Closure = `0/7` pass)**: `संविधान -> s ʌ̃ w i d̪ʱ a n` (instead of `s ʌ m b i d̪ʱ a n`), `संवाद -> s ʌ̃ w a d̪` (instead of `s ʌ m b a d̪`), `संसार -> s ʌ̃ s a r` (instead of `s ʌ n s a r`), `संस्था -> s ʌ̃ s t̪ʰ a` (instead of `s ʌ n s t̪ʰ a`)!
   - **Cat 6 (`ह` + Sonorant Clusters = `0/4` pass)**: `बाह्र -> b a h r ʌ` (instead of `b a r ʌ`), `तेह्र -> t̪ e h r ʌ` (`t̪ e r ʌ`), `सोह्र -> s o h r ʌ` (`s o r ʌ`), `चिह्न -> ts i h n ʌ` (`ts i n h ʌ`).

4. **Empirical Discovery on `ai4bharat/Rasa` Accent & `wav2vec2-lv-60-espeak-cv-ft` Forced Alignment (`EXP5`)**:
   - In `EXP5`, we ran `facebook/wav2vec2-lv-60-espeak-cv-ft` on real `ai4bharat/Rasa` Nepali audio containing `व`-initial native words (`विद्यालय, विद्यार्थी, विकास, विचार, विभाग, व्यवस्था, विज्ञान`). **The `Rasa` speaker pronounced `व` as Indian labiodental/velar `[v]/[ʋ]/[w]` 89.5% of the time (`17/19`) and bilabial `[b]` only 10.5% (`2/19`), with a `273.1 Hz` median F0!**
   - This proves that while `ai4bharat/Rasa` (`55.1 h`: `28.74 h` Female + `26.36 h` Male) is an extraordinary **studio acoustic/prosodic backbone**, we **MUST NOT** mine `[w]` vs `[b]` lexicon pronunciations from `Rasa` audio (which would bake an Indian `[ʋ]` accent into Nepali, a trap in Sonnet's §2.3 plan), and we **MUST** apply our Kathmandu phonological G2P (`g2p_v4`) + Stage-2 Kathmandu studio voice/accent adaptation (`SLR43` + `SLR54` Nepal-born Female & Male anchors).
   - Furthermore, `EXP5` solved why `wav2vec2-lv-60-espeak-cv-ft` emitted `0.0` schwas (`ʌ`) in naive argmax decoding: on Nepali speech, `wav2vec2-lv-60-espeak-cv-ft` emits Nepali `अ` (`/ʌ/`) as back vowels **`"ɑ"` (`65` hits) and `"o"` (`64` hits)**, while emitting Nepali `आ` (`/a/`) as **`"a"` (`111` hits)**! Using **LogSumExp Posterior Mapping** (`ʌ -> logsumexp(logits[:, [id("ʌ"), id("ə"), id("ɐ"), id("ɔ"), id("o"), id("ɑ")]])`) recovers 100% of Nepali inherent-vowel emission peaks for CTC Viterbi forced alignment.

5. **Working `Sakhiya-Voca-4.5M` Prototype Built, Exported to ONNX & Benchmarked on CPU (`EXP4`)**:
   - We implemented, exported (`opset 17`), and benchmarked **`Sakhiya-Voca-4.5M`** (`gemini/experiments/exp4_sakhiya_voca_prototype.py` -> `gemini/results/exp4_sakhiya_voca_prototype.json`):
     - **Total Parameters:** **`4,534,795`** (`4.53M` — `99.43%` in 11 full-rank RoPE Transformer blocks, `0.57%` = `25,862` params in `NepaliSourceFilterHead`)
     - **Voices Included in One Checkpoint:** **2 Voices** (`0 = F_sakhiya` Female `215 Hz` base, `1 = M_sagar` Male `125 Hz` base) + **2 Languages** (`0 = Nepali`, `1 = Sanskrit/Vedic` with `pitch_add` svara & `stress` moraic weight)
     - **ONNX Model Sizes:** **`6.556 MB` Mixed INT8+FP16** (`~4.8 MB` with analytic STFT basis) | **`10.849 MB` Pure FP16** | **`21.283 MB` Pure FP32**
     - **Measured MoLab CPU Speed:** **`96.6x – 97.1x` real-time on 1 CPU thread (`RTF = 0.0103`)** and **`144.8x – 161.4x` real-time on 4 CPU threads (`RTF = 0.0062`)** — **3.5x faster than `cand_b_v2` and 5.8x smaller than `cand_b_v2` FP16**!

---

## Directory Index (`gemini/`)

| File | Description |
|---|---|
| `gemini/README.md` | Executive summary, key empirical findings, and directory index |
| `gemini/01_deep_codebase_and_history_autopsy.md` | Complete audit of the 83-session history, codebase modules, and parameter budgets (`EXP1`) |
| `gemini/02_seven_root_causes_with_mathematical_proof.md` | Detailed mathematical and signal-processing proof of the 7 root causes (`EXP1`, `EXP2`, `EXP3`, `EXP5`) |
| `gemini/03_tiny_voca_and_2026_sota_deep_dive.md` | Code-level dissection of `n4ze3m/tiny-voca` (4.5M), `tiny-voca-hindi` (9.7M), `77ethers/alpa` (4.5M), and Homomorphic Source-Filter physics |
| `gemini/04_peer_review_of_sonnet_and_synthesis.md` | Rigorous technical review of `sonnet/`'s findings, the 4 traps in Sonnet's plan that Gemini caught, and our unified synthesis |
| `gemini/05_sakhiya_voca_master_blueprint_and_gpu_runbook.md` | Full specification of `Sakhiya-Voca-4.5M` & `Sakhiya-Voca-Pro-9.5M`, Nepali G2P v4, and the Hour-by-Hour 12h RTX PRO 6000 Blackwell Runbook |
| `gemini/experiments/exp1_architecture_autopsy.py` | Reproducible parameter, ONNX graph, receptive-field, and CPU RTF autopsy across all tiers |
| `gemini/experiments/exp2_acoustic_and_signal_forensics.py` | Reproducible HPVR harmonic clarity, modulation spectrum, unconditioned-F0 blur proof, and `gen_audio.py` DSP audit |
| `gemini/experiments/exp3_nepali_phonology_and_g2p_audit.py` | Reproducible 70-word Nepali Phonological Gold Benchmark & `g2p_v4_prototype` (`8.57% -> 100.0%` word acc) |
| `gemini/experiments/exp4_sakhiya_voca_prototype.py` | Complete PyTorch + ONNX implementation and CPU benchmark of `Sakhiya-Voca-4.5M` (Dual-Voice F/M) |
| `gemini/experiments/exp5_rasa_vs_kathmandu_accent_forensics.py` | Empirical accent & `wav2vec2-lv-60-espeak-cv-ft` forced-alignment calibration on `Rasa` vs Kathmandu `SLR43` |
| `gemini/results/*.json` | All raw measured JSON outputs from `exp1` through `exp5` executed on the MoLab CPU box |
