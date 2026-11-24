# Sakhiya-Pro v1 evaluation (synthesized audio, 8 hard sentences)

Selected: **ft3_s200**

| tag | CER | tech CER | Vedic CER | UTMOS22 | spk-sim | Hindi leaks | clicks |
|---|---|---|---|---|---|---|---|
| ft3_s200 | 0.118 | 0.139 | 0.095 | 4.00 | 0.74 | 0 | 31 |
| ft3_s400 | 0.111 | 0.132 | 0.068 | 3.72 | 0.71 | 0 | 14 |
| ft3_s1000 | 0.114 | 0.137 | 0.068 | 3.70 | 0.72 | 0 | 14 |
| ft3_s600 | 0.113 | 0.134 | 0.068 | 3.66 | 0.72 | 0 | 11 |
| ft3_s800 | 0.114 | 0.137 | 0.068 | 3.65 | 0.71 | 0 | 23 |
| ft3_s1200 | 0.126 | 0.150 | 0.095 | 3.77 | 0.71 | 0 | 10 |
| ft2_s900 | 0.118 | 0.136 | 0.122 | 3.78 | 0.71 | 0 | 16 |
| ft2_s1500 | 0.124 | 0.149 | 0.068 | 3.67 | 0.69 | 0 | 19 |
| base_zeroshot | 0.102 | 0.120 | 0.081 | 4.03 | 0.55 | 0 | 51 |
| ft2_s1800 | 0.122 | 0.141 | 0.108 | 3.71 | 0.69 | 0 | 21 |
| ft2_s600 | 0.130 | 0.150 | 0.135 | 3.71 | 0.70 | 0 | 23 |
| ft2_s1200 | 0.141 | 0.162 | 0.149 | 3.67 | 0.69 | 0 | 16 |
| ft2_avg | 0.970 | 0.967 | 1.000 | 1.23 | -0.06 | 0 | 3 |


## Sakhiya-Pro v2 (native-Nepali two-stage adaptation) — selected **pron_s2_300 (stage-1 pron_s1_1200)**

| tag | native CER | word recall | tech CER | Vedic CER | UTMOS22 | spk-sim | Hindi leaks | clicks |
|---|---|---|---|---|---|---|---|---|
| pron_s1_1200 | 0.061 | 0.745 | 0.702 | 0.157 | 4.01 | 0.69 | 0 | 239 |
| pron_s1_400 | 0.062 | 0.755 | 0.654 | 0.125 | 4.04 | 0.68 | 1 | 176 |
| pron_s1_800 | 0.068 | 0.708 | 0.689 | 0.168 | 4.07 | 0.69 | 1 | 166 |
| pron_s2_100 | 0.066 | 0.724 | 0.764 | 0.132 | 3.87 | 0.69 | 1 | 48 |
| pron_s2_200 | 0.066 | 0.734 | 0.753 | 0.110 | 3.81 | 0.66 | 1 | 62 |
| pron_s2_300 | 0.066 | 0.740 | 0.711 | 0.147 | 3.80 | 0.67 | 1 | 44 |
| pron_v1 | 0.064 | 0.740 | 0.736 | 0.137 | 3.97 | 0.67 | 1 | 35 |

Automatic metrics only (ASR round trip, UTMOS22, ECAPA similarity); no human MOS.


## Sakhiya-Pro v3 (native-Nepal accent lock) — selected **v3b_checkpoint-600**

| metric | v2 | v3 |
|---|---|---|
| end_slope | -0.577 | -0.416 |
| rise_frac | 0.240 | 0.233 |
| spk_sim | 0.780 | 0.748 |
| cer | 0.123 | 0.120 |
| utmos | 3.888 | 3.918 |
| clicks | 28.833 | 25.792 (0 after declick) |
| f0_med_std | 8.481 | 6.920 |
| rms_std_db | 1.072 | 0.645 |

Episode 01: **33/33** lines pass the forensic gate (-14.0 LUFS broadcast master, 6-pass declick + phonetic loanword CER). Key lines:

| line | gate | CER | clicks |
|---|---|---|---|
| c3 | True | 0.000 | 0 |
| n1a | True | 0.000 | 0 |
| n2a | True | 0.000 | 0 |
| n4a | True | 0.000 | 0 |
| f1 | True | 0.000 | 0 |
| d2 | True | 0.053 | 0 |
| d5 | True | 0.043 | 0 |
| o1 | True | 0.000 | 0 |

end_slope < 0 = clause endings close downward (Nepal-native); > 0 = floating English-style rise. All numbers automatic (F0 contour, Kriti ASR, UTMOS22, ECAPA); no human MOS.


## Sakhiya-Pro v4 (Studio-Denoised Native-Nepal Lock + De-Karkar) — selected **v4b_checkpoint-100**

| metric | v3 (v3b_s600) | v4 (Studio-Denoised + De-Karkar) |
|---|---|---|
| end_slope | -0.208 | -1.288 |
| rise_frac | 0.201 | 0.031 |
| clicks | 4.500 | 0.125 |
| hf_karkar | nan | 0.002 |
| spk_sim | 0.852 | 0.770 |
| cer | 0.032 | 0.039 |
| utmos | 3.752 | 3.970 |

Episode 01 v4 Broadcast Master: **33/33** lines pass the forensic gate (-14.0 LUFS, -56 dB studio room-tone bed, zero clicks/karkar).


## Sakhiya v5 Clean Recovery — Locked Nepali `v3b-600` Master + Isolated Vedic LoRA-800 (`adapter-800`) + 16.69h Clean Distillation Corpus

- **Nepali Master (`sakhiya_v5_nepali` / `ep01_v5`)**: Locked on `v3b_checkpoint-600` with the de-karkar renderer (`11/11` Nepali pack pass, `33/33` Episode 01 pass at `-14.0 LUFS`, `0` clicks, `end_slope = -1.787`, `rise_frac = 0.0`, `CER = 0.025`, `UTMOS = 3.922`).
- **Isolated Vedic LoRA (`sakhiya_v5_vedic`, `adapter-800`)**: Trained on `3.74h` (`1,291` clips) pure Vagdhenu + Gita Vedic chants with base weights strictly frozen (`base_sha_before == base_sha_after == 5517602c2d24c470`).

| Vedic Checkpoint | Leak/Loop Gate (`scripts/leak_loop_gate.py`) | Rigveda 1.1.1 | Shanti Mantra (`ॐ`) | Gayatri Mantra (`ॐ`) | Bhagavad Gita 1.1 | Unseen Shloka |
|---|---|---|---|---|---|---|
| `base_pure` | `0/8` | 0/2 | 0/2 | 0/2 | 0/2 | — |
| `vl1_1600` | `9/12` | 3/3 | 1/3 | 2/3 | 3/3 | — |
| `vl1_2400` | `10/12` | 3/3 | 2/3 | 2/3 | 3/3 | — |
| **`vl1_800` (`adapter-800`, selected)** | **`11/12` (`5/5` Best-of-3, `14/15` takes)** | **PASS (`0` clicks)** | **PASS (`ॐ 1.46s`, `0` clicks)** | **PASS (`ॐ 2.44s`, `0` clicks)** | **PASS (`0` clicks)** | **PASS (`ॐ 1.92s`, `0` clicks)** |

- **16.69-Hour Single-Voice 24 kHz Distillation Corpus (`docs/sakhiya_pro/v5_distill_corpus_report.json`)**: `8,997` gate-verified utterances (`16.69h`, `164` rejected by `scripts/leak_loop_gate.py`, `0` leaks, `0` loops, `mean_clicks = 0.02`, `mean_cer = 0.014`).
