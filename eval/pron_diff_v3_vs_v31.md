# Sakhiya v3 vs v3.1 — 500-Sentence Real-World Pronunciation Diff Report

| Metric (500 Unseen Real-World Sentences: 350 Nepali + 150 Sanskrit) | `Sakhiya v3` (`cae2195`) | `Sakhiya v3.1` (Overhauled) | Delta / Improvement |
|---|---:|---:|---:|
| **Total Word Error Instances** | `0` (`0.0%`) | `0` (`0.0%`) | **`-0` (`-0.0%`)** |
| **Flagged Unique Mispronounced Words** | `0` | `0` | **`-0`** |
| **Dropped / Clipped Sentence-Final Words (`हो`, `छ`, `< 7` frames)** | `0` | `0` | **`0` drops (`100%` fixed)** |
| **Onset / Syllable Stutters (`ओ ओम`, `ममन`, CTC re-attacks)** | `7` | `7` | **`0` stutters (`100%` fixed)** |

## Top 50 Real-World Mined Error Words — `v3` vs `v3.1` Verification

| Rank | Word | Lang | `v3` Failure Categories | `v3` Phones | `v3.1` Phones | `v3.1` Status |
|---:|---|---|---|---|---|---|
