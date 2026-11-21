# Sakhiya v3 — 500-Sentence Real-World Pronunciation Error Report (Top 50 Worst Words)

- **Corpus:** `500` strictly unseen real-world sentences (`350` Nepali + `150` Sanskrit, `3639` total word tokens, `2622` unique words)
- **Flagged Real-World Word Error Instances:** `659` (`18.11%` of all words)
- **Dropped / Clipped Sentence-Final Words (`< 7` frames):** `3`
- **Onset / Syllable Stutter Instances (`ओ ओम`, `ममन`, CTC re-attacks):** `14`

## Top 50 Real-World Mispronounced Words in `v3`

| Rank | Word | Lang | Count | Failure Categories | `v3` Phones | Target Clean Phones | Mean Frames |
|---:|---|---|---:|---|---|---|---:|
| 1 | **ॐ** | `sa` | 10 | `sanskrit_pranava_om_double_onset_split` | `o m` | `o m` | 12.0 |
| 2 | **हो** | `ne` | 45 | `dropped_or_clipped_sentence_final` | `h o` | `h o` | 12.0 |
| 3 | **काठमाडौँ** | `ne` | 3 | `nepali_compound_medial_schwa, nepali_nasal_diphthong_split` | `k a ʈʰ ʌ m a ɖ ʌ ũ` | `k a ʈʰ m a ɖ ʌ̃ ũ` | 12.0 |
| 4 | **अग्निमीळे** | `sa` | 1 | `sanskrit_vedic_retroflex_la_as_stop_da` | `ʌ g n i m i ɽ e` | `ʌ g n i m i l e` | 12.0 |
| 5 | **गाउँ** | `ne` | 6 | `nepali_nasal_diphthong_split` | `g a ũ` | `g ã ũ` | 12.0 |
| 6 | **मन** | `sa` | 5 | `sonorant_onset_stutter_maman` | `m ʌ n ʌ` | `m ʌ n ʌ` | 12.0 |
| 7 | **आउँछ** | `ne` | 4 | `nepali_nasal_diphthong_split` | `a ũ tsʰ ʌ` | `a u n tsʰ ʌ` | 12.0 |
| 8 | **छैन** | `ne` | 7 | `nepali_missing_final_schwa` | `tsʰ ʌ i n` | `tsʰ ʌ i n ʌ` | 12.0 |
| 9 | **गाउँमा** | `ne` | 4 | `nepali_nasal_diphthong_split` | `g a ũ m a` | `g ã ũ m a` | 12.0 |
| 10 | **पूर्णमदः** | `sa` | 1 | `sanskrit_heavy_visarga_echo` | `p u r ɳ ʌ m ʌ d̪ ʌ h` | `p u r ɳ ʌ m ʌ d̪ ʌ h` | 12.0 |
| 11 | **मामकाः** | `sa` | 1 | `sanskrit_heavy_visarga_echo` | `m a m ʌ k a h` | `m a m ʌ k a h` | 12.0 |
| 12 | **स्वः** | `sa` | 1 | `sanskrit_heavy_visarga_echo` | `s w ʌ h` | `s w ʌ h` | 12.0 |
| 13 | **ईळे** | `sa` | 1 | `sanskrit_vedic_retroflex_la_as_stop_da` | `i ɽ e` | `i l e` | 12.0 |
| 14 | **युयुत्सवः** | `sa` | 1 | `sanskrit_heavy_visarga_echo` | `j u j u t̪ s ʌ w ʌ h` | `j u j u t̪ s ʌ w ʌ h` | 12.0 |
| 15 | **राजधानी** | `ne` | 3 | `nepali_compound_medial_schwa` | `r a dz ʌ d̪ʱ a n i` | `r a dz d̪ʱ a n i` | 12.0 |
| 16 | **मीळ्हुषे** | `sa` | 1 | `sanskrit_vedic_retroflex_la_as_stop_da` | `m i ɽ h u ʃ e` | `m i l h u ʃ e` | 12.0 |
| 17 | **मृळया** | `sa` | 1 | `sanskrit_vedic_retroflex_la_as_stop_da` | `m r i ɽ ʌ j a` | `m r i l ʌ j a` | 12.0 |
| 18 | **सुदूरपश्चिम** | `ne` | 4 | `nepali_compound_medial_schwa` | `s u d̪ u r ʌ p ʌ ʃ ts i m` | `s u d̪ u r p ʌ s ts i m` | 12.0 |
| 19 | **लगभग** | `ne` | 4 | `nepali_compound_medial_schwa` | `l ʌ g ʌ bʱ ʌ g` | `l ʌ g bʱ ʌ g` | 12.0 |
| 20 | **भैँसी** | `ne` | 1 | `nepali_compound_medial_schwa, nepali_nasal_diphthong_split` | `bʱ ʌ ĩ s i` | `bʱ ʌ̃ ĩ s i` | 12.0 |
| 21 | **ठाउँ** | `ne` | 2 | `nepali_nasal_diphthong_split` | `ʈʰ a ũ` | `ʈʰ ã ũ` | 12.0 |
| 22 | **दशैँ** | `ne` | 2 | `nepali_compound_medial_schwa, nepali_nasal_diphthong_split` | `d̪ ʌ ʃ ʌ ĩ` | `d̪ ʌ s ʌ̃ ĩ` | 12.0 |
| 23 | **सगरमाथा** | `ne` | 1 | `nepali_compound_medial_schwa` | `s ʌ g ʌ r ʌ m a t̪ʰ a` | `s ʌ g ʌ r m a t̪ʰ a` | 12.0 |
| 24 | **होइन** | `ne` | 3 | `nepali_missing_final_schwa` | `h o i n` | `h o i n ʌ` | 12.0 |
| 25 | **सय** | `ne` | 3 | `nepali_missing_final_schwa` | `s ʌ j` | `s ʌ j ʌ` | 12.0 |
| 26 | **समय** | `ne` | 2 | `nepali_missing_final_schwa` | `s ʌ m ʌ j` | `s ʌ m ʌ j ʌ` | 12.0 |
| 27 | **फिल्म** | `ne` | 3 | `nepali_loan_extra_final_schwa` | `pʰ i l m ʌ` | `pʰ i l m` | 12.0 |
| 28 | **इन्टेलिजेन्स** | `ne` | 1 | `nepali_loan_extra_final_schwa` | `i n ʈ e l i dz e n s ʌ` | `i n ʈ e l i dz e n s` | 12.0 |
| 29 | **एम्बुलेन्स** | `ne` | 1 | `nepali_loan_extra_final_schwa` | `e m b u l e n s ʌ` | `e m b u l e n s` | 12.0 |
| 30 | **राष्ट्रिय** | `ne` | 3 | `phoneme_mismatch` | `r a ʃ ʈ r i j` | `r a s ʈ r i j ʌ` | 12.0 |
| 31 | **अन्तर्राष्ट्रिय** | `ne` | 4 | `phoneme_mismatch` | `ʌ n t̪ ʌ r r a ʃ ʈ r i j` | `ʌ n t̪ ʌ r r a s ʈ r i j ʌ` | 12.0 |
| 32 | **तपाईंलाई** | `ne` | 2 | `phoneme_mismatch` | `t̪ ʌ p a ĩ l a i` | `t̪ ʌ p ã ĩ l a i` | 12.0 |
| 33 | **अली** | `ne` | 1 | `nepali_compound_medial_schwa` | `ʌ l i` | `` | 12.0 |
| 34 | **नः** | `sa` | 5 | `sanskrit_heavy_visarga_echo` | `n ʌ h` | `n ʌ h` | 12.0 |
| 35 | **आउँदा** | `ne` | 2 | `nepali_nasal_diphthong_split` | `a ũ d̪ a` | `a u n d̪ a` | 12.0 |
| 36 | **जनाः** | `sa` | 3 | `sanskrit_heavy_visarga_echo` | `dz ʌ n a h` | `dz ʌ n a h` | 12.0 |
| 37 | **नै** | `ne` | 2 | `nepali_compound_medial_schwa` | `n ʌ i` | `n a i` | 12.0 |
| 38 | **फुटबल** | `ne` | 2 | `nepali_compound_medial_schwa` | `pʰ u ʈ ʌ b ʌ l` | `pʰ u ʈ b ʌ l` | 12.0 |
| 39 | **फ्रान्सको** | `ne` | 2 | `nepali_compound_medial_schwa` | `pʰ r a n s ʌ k o` | `pʰ r a n s k o` | 12.0 |
| 40 | **बस** | `ne` | 2 | `nepali_compound_medial_schwa` | `b ʌ s ʌ` | `b ʌ s` | 12.0 |
| 41 | **अक्षरबाट** | `ne` | 1 | `nepali_missing_final_schwa_baata` | `ʌ k ʃ ʌ r b a ʈ` | `ʌ k tsʰ j ʌ r b a ʈ` | 12.0 |
| 42 | **इन्टरनेटबाट** | `ne` | 1 | `nepali_compound_medial_schwa, nepali_missing_final_schwa_baata` | `i n ʈ ʌ r ʌ n e ʈ b a ʈ` | `i n ʈ ʌ r n e ʈ b a ʈ` | 12.0 |
| 43 | **कुल्लोबाट** | `ne` | 1 | `nepali_missing_final_schwa_baata` | `k u l l o b a ʈ` | `k u l l o b a ʈ` | 12.0 |
| 44 | **जराबाट** | `ne` | 1 | `nepali_missing_final_schwa_baata` | `dz ʌ r a b a ʈ` | `dz ʌ r a b a ʈ` | 12.0 |
| 45 | **झ्यालबाट** | `ne` | 1 | `nepali_missing_final_schwa_baata` | `dzʱ j a l b a ʈ` | `dzʱ j a l b a ʈ` | 12.0 |
| 46 | **ढोकाबाट** | `ne` | 1 | `nepali_missing_final_schwa_baata` | `ɖʱ o k a b a ʈ` | `ɖʱ o k a b a ʈ` | 12.0 |
| 47 | **पदार्थबाट** | `ne` | 1 | `nepali_missing_final_schwa_baata` | `p ʌ d̪ a r t̪ʰ ʌ b a ʈ` | `p ʌ d̪ a r t̪ʰ ʌ b a ʈ` | 12.0 |
| 48 | **पोखराबाट** | `ne` | 1 | `nepali_missing_final_schwa, nepali_missing_final_schwa_baata` | `p o kʰ ʌ r a b a ʈ` | `p o kʰ ʌ r a b a ʈ ʌ` | 12.0 |
| 49 | **बसपार्कबाट** | `ne` | 1 | `nepali_compound_medial_schwa, nepali_missing_final_schwa_baata` | `b ʌ s ʌ p a r k ʌ b a ʈ` | `b ʌ s p a r k b a ʈ` | 12.0 |
| 50 | **बागबनाबाट** | `ne` | 1 | `nepali_missing_final_schwa_baata` | `b a g ʌ b ʌ n a b a ʈ` | `b a g ʌ b ʌ n a b a ʈ` | 12.0 |
