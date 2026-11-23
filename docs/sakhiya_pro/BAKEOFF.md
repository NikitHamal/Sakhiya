# Sakhiya-Pro zero-shot backbone bake-off (23 Nov 2026, RTX PRO 6000 Blackwell)

Prompt: one 5 s clip of the Nepal-native female anchor (SLR43 speaker 6834 – studio, UTMOS 3.89, Kriti CER 0.04,
no Indian-accented Rasa timbre). Eight hard sentences (NEBians Ep.01 news with Anthropic / Gemini 4 Neon / Jensen Huang /
GPT-6.2 / Liquid Glass Dock, conversational Nepali, conjunct clusters, Vedic ॐ पूर्णमदः), Devanagari and Latin-term variants.
All raw WAVs are in `NikitHamal/sakhiya-v1-train/sakhiya_pro/bakeoff/`.

| candidate | CER (Kriti) | tech CER | Vedic CER | UTMOS22 | ECAPA sim to anchor | Hindi leaks |
|---|---|---|---|---|---|---|
| **k2-fsa/OmniVoice (600M, `npi`)** | 0.110 | 0.130 | **0.081** | **4.03** | 0.54 | 0 |
| Aarjan/swar v6 EMA (F5-Base 335M) | 0.119 | 0.133 | 0.162 | 3.85 | 0.69 | 0 |
| Aarjan podvoices EMA (F5-Base) | 0.103 | 0.127 | 0.135 | 3.89 | 0.71 | 0 |
| OmniVoice, `hi` language tag (control) | 0.116 | 0.137 | 0.081 | 4.02 | 0.74 | 0 |

* `ai4bharat/IndicF5` is gated and the project HF token is not authorised (403). Swar is an IndicF5/F5-Base fine-tune with the
  identical architecture, so it stands in as the F5 representative.
* Verdict: **OmniVoice-600M** – best naturalness (UTMOS), clearly best Vedic/Sanskrit CER, handles Devanagari + Latin tech
  terms in one pass and ships a full training stack (so the speaker-similarity gap is closed by fine-tuning on the single anchor voice).
* Caveat: these are objective, automatic scores (ASR CER, UTMOS, ECAPA, pause forensics); no human listening was done in this run.
