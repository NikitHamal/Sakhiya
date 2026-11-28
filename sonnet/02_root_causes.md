
## Addendum — the "bar" in Nepal, measured with the same battery (EXP5, same 40 sentences)

| System | UTMOS22 | strict CER / WER | pauses ≥150 ms per utt | graphemes/s | mean abs ΔF0 (st/10 ms) |
|---|---|---|---|---|---|
| Real Rasa female | 3.79 | 0.090 / 0.323 | 1.73 | 6.51 | 0.596 |
| Edge `ne-NP-HemkalaNeural` (F) | 3.42 | 0.084 / 0.346 | **0.03** | 7.55 | 0.478 |
| Edge `ne-NP-SagarNeural` (M) | 3.63 | 0.093 / 0.371 | **0.03** | 8.23 | 0.724 |
| Sakhiya-V1 INT8 | 4.08 | 0.083 / 0.288 | 1.50 | 6.35 | 0.622 |
| Sakhiya default `cand_b_v2` | 2.64 | 0.098 / 0.369 | 0.20 | 7.26 | 0.427 |

Two lessons: (1) UTMOS ranks V1 above Edge and above *real speech*, so it measures "clean and uniform", not "native" — it must not decide anything; (2) the commercial reference itself almost never pauses and speaks 16–26 % faster than the human reader. **Explicitly modelled, human-like phrasing/pausing is a measurable place where a small model trained on real speech can beat the best Nepali TTS available** (and V1 already shows the statistics are reachable). Whether that is *preferred* by listeners is exactly what the blind test (05) must establish.
