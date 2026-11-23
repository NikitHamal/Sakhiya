# सखिया · Sakhiya — Nepali & Sanskrit text-to-speech that runs in the browser

One female studio voice (`sakhiya`) for Nepali and Sanskrit, 6 delivery styles, Chandas/Vedic chant timing, and
in-browser ONNX synthesis. Live studio: https://nikithamal.github.io/Sakhiya/

## The three models (this is all the website offers)

| Name | What it is | Size (int8, what the browser downloads) |
|---|---|---|
| **Sakhiya-V1** | Best quality (the old "V5 Final"). | ~16 MB |
| **Sakhiya-V1-Nano** | Small model: fully trained small acoustic student + V1 vocoder. Measured numbers live in `docs/tiers.json` / `eval/forensic_v9_nano.json`. | ~8.9 MB |
| **Sakhiya-V1-Atom** | Smallest model: tiny acoustic student + V1 vocoder. Measured numbers live in `docs/tiers.json` / `eval/forensic_v9_atom.json`. | ~6.5 MB |

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


## GGUF tiers, WebAssembly, Python, Android (sakhiya.cpp)
```bash
# models: https://huggingface.co/NikitHamal/sakhiya-gguf  (atom_lite 3.0 MB, atom 7.4 MB, nano 10.1 MB, v1 37.5 MB; *_q8_0_exact = waveform-exact)
python convert_sakhiya_gguf.py --tier atom_lite --am am.pt --voc voc.pt --dtype q8_0 --out sakhiya_atom_lite_q8_0.gguf   # --profile exact for waveform parity
cmake -S cpp -B build-lib -DCMAKE_BUILD_TYPE=Release && cmake --build build-lib -j   # libsakhiya + sakhiya-cli-lite, no ggml needed
source emsdk_env.sh && scripts/build_wasm.sh                                          # docs/wasm/sakhiya.wasm (single-thread SIMD128)
python tests/test_ggml_tiers.py && node tests/test_wasm_node.mjs                      # parity + size + RTF gates, WASM vs native
```
```python
from sakhiya.ggml_tts import SakhiyaGGML
wav, sr = SakhiyaGGML.from_pretrained("atom-lite").say("नेपालको राजधानी काठमाडौँ हो।", style="news", speed=1.0)
```
Android: `cpp/android_jni.cpp` + `android/SakhiyaGGML.kt` (target `sakhiya_jni` in `cpp/CMakeLists.txt`). Browser: `docs/wasm/sakhiya_engine.js`. Measured numbers and caveats are in STATUS.md (Session 68).

## Quickstart (GGML C/C++ Standalone CLI: `sakhiya-cli`)
```bash
# 1. Convert PyTorch v5_c1 checkpoints (am.pt + voc.pt) into a single GGUF v3 file (f32, f16, or q8_0)
python3 convert_sakhiya_gguf.py --am /work/tts/runs/v5_c1/am.pt --voc /work/tts/runs/v5_c1/voc.pt --out /work/tts/models/sakhiya_f32.gguf --dtype f32

# 2. Build standalone sakhiya-cli linked against ggml (CMake or Makefile; optional -DGGML_CUDA=ON / -DGGML_VULKAN=ON / -DGGML_METAL=ON)
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# 3. Synthesize 24 kHz 16-bit PCM WAV (~12x real-time on CPU, cosine_sim = 1.000000 vs PyTorch) & run the stage-by-stage diff gate
./build/sakhiya-cli -m /work/tts/models/sakhiya_f32.gguf -t "नेपालको राजधानी काठमाडौँ हिमालको काखमा रहेको एउटा ऐतिहासिक सहर हो।" -l ne -s news -o sakhiya_cpp.wav -j 16
python3 tests/test_ggml_parity.py --threads 16
```

## How Nano / Atom were improved (v9)
v8 swapped the tiny vocoder for V1's vocoder, but the small acoustic students had only ~4.6k training steps because the teacher flow-ODE was recomputed every step (2.5 steps/s). v9 (`scripts/v9_am_cached.py`) runs the teacher once, keeps its targets in GPU memory and trains the students ~3x faster at 2x the batch size, under a hard 100-minute cap with a step-50 throughput gate. `scripts/v9_chain.sh` exports, runs the real forensic gate and publishes only if the result beats v8. The website selector lists only V1, Nano and Atom; older packs stay in the repo but are not listed.

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
