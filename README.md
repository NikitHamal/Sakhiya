# सखिया · Sakhiya — Nepali & Sanskrit text-to-speech that runs in the browser

One female studio voice (`sakhiya`) for Nepali and Sanskrit, 6 delivery styles, Chandas/Vedic chant timing, and
in-browser ONNX synthesis. Live studio: https://nikithamal.github.io/Sakhiya/

## The three models (this is all the website offers)

| Name | What it is | Size (int8, what the browser downloads) |
|---|---|---|
| **Sakhiya-V1** | Best quality (the old "V5 Final"). | ~16 MB |
| **Sakhiya-V1-Nano** | Small model: small acoustic student + V1 vocoder. Measured UTMOS 3.08 (old 2.47), 0 clicks (old 37). | ~8.9 MB |
| **Sakhiya-V1-Atom** | Smallest model: tiny acoustic student + V1 vocoder. Measured UTMOS 2.95 (old 2.03). | ~6.5 MB |

Nano and Atom are distilled from V1 (a small acoustic model + a small vocoder). Numbers for every shipped model come from
`scripts/forensic_audio_gate.py` and are stored in `eval/forensic_*.json`; `STATUS.md` has the honest log of what worked.

## Why this repo is small now
The repo only keeps what the website and the code need. Everything heavy or legacy was moved to the GitHub Release
**`v8-archive`** (nothing was deleted):

| Release asset | Contents | Restore with |
|---|---|---|
| `legacy-models.tar.gz` | fp16/fp32 ONNX graphs, old tiers (`models`, `models_nano`, `models_atom`, `models_v6_teacher`, ...) | `tar xzf legacy-models.tar.gz` at the repo root |
| `checkpoints-and-ab-pack.tar.gz` | `ckpt/` PyTorch checkpoints + `eval/ab_pack` | same |
| `clip-library.tar.gz` | the old Clip Library page + mp3s | same |
| `old-sample-packs.tar.gz` | older sample versions (v3 ... v6) | same |
| `sakhiya-full-history.bundle.part*` | the complete pre-cleanup git history | `cat *.part* > h.bundle && git clone h.bundle` |

Training checkpoints for the new Nano/Atom are in Release `v8-ckpt`.

## Quickstart (Python)
```python
from sakhiya.onnx_tts import SakhiyaONNX      # numpy + onnxruntime only
tts = SakhiyaONNX("docs/models_v2/", tiny=True)   # V1 int8;  use docs/models_voca (Nano) or docs/models_sano (Atom)
wav, sr = tts.say("नेपालको राजधानी काठमाडौँ हो।", speaker="sakhiya", style="news")
```
Controls: `lang` (`auto`/`ne`/`sa`), `style` (`news`, `storytelling`, `conversational`, `warm_soft`, `expressive`, `vedic_chant`), `rate`, `pitch_shift`.

## How Nano / Atom were fixed (v8)
1. **Measured where the robotic sound comes from** (`scripts/v8_voc_gta.py`, "DIAG" line, real text -> model -> audio path):
   the small acoustic model's mel is fine (UTMOS 3.39 Nano / 3.16 Atom when played through the big V1 vocoder) but the tiny vocoder
   turned it into 2.69 / 2.27. So the tiny vocoder was the problem, not the acoustic model.
2. **Cause:** the tiny vocoders were trained on sharp ground-truth mels with their pitch detector frozen, but at run time they receive the
   smoother mel of the small acoustic model plus the export-time mel post-processing. Pitch was mis-read -> robotic/glitchy voice.
3. **Fix that shipped:** keep the small acoustic students, replace the tiny vocoder with V1's vocoder (`tier nanoplus/atomplus`, `scripts/v8_ship.py`). Gate result: Nano UTMOS 3.08, Atom 2.95, 0 clicks. This costs size (Nano 8.9 MB, Atom 6.5 MB). Honest note: both still show 1 word drop (आआफ्नो) on the gate.
4. **Fix tried and NOT shipped:** retraining the *tiny* vocoder on student mels with F0 distillation (`scripts/v8_voc_gta.py`). Real-path UTMOS fell (2.69 -> 2.17 in 3k steps), so it was stopped. A tiny (<3 MB) clean vocoder still needs a bigger vocoder budget or a different architecture.
   (the original plan text follows)
   **Plan:** train the vocoder on the mel the small acoustic model really produces, through the exact export interface, against the V1
   vocoder's waveform, with the pitch detector unfrozen and distilled from V1 (F0 + voicing loss), GAN + WavLM discriminator, and
   checkpoint selection by UTMOS on the real inference path (`scripts/v8_voc_gta.py`). Training runs under a hard `--max-minutes` cap
   with a step-50 throughput gate (`scripts/v8_am_distill.py`, `scripts/v8_voc_gta.py`). Shipping is automatic only if the real forensic
   gate beats the previous model (`scripts/v8_ship.py`).

## Data & licences
Trained on OpenSLR SLR43 (CC BY-SA 4.0), FLEURS ne_np (CC BY 4.0), Common Voice ne-NP (CC0). OpenSLR SLR143 (CC BY-NC-SA) is research-only.
Lexicon derived from `NikitHamal/transliterationlm`. See `PROVENANCE.md`.
