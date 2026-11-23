---
license: other
language: [ne, sa]
tags: [tts, gguf, ggml, webassembly, nepali, sanskrit]
---
# Sakhiya GGUF (GGML C++ / WASM engine)

Single-file GGUF v3 models for every shipped Sakhiya tier, run by the zero-dependency C++17 engine
(`cpp/sakhiya_ggml.hpp`, https://github.com/NikitHamal/Sakhiya) as `sakhiya-cli`, `libsakhiya.so`, Android JNI or `sakhiya.wasm`.

| file | tier | notes |
|---|---|---|
| `sakhiya_<tier>_q8_0.gguf` | v1 / nano / atom / atom_lite | compact Q8_0 (default; V1's Q8_0 is already waveform-exact) |
| `sakhiya_<tier>_q8_0_exact.gguf` | nano / atom / atom_lite | acoustic model FP32 + Q8_0 vocoder: waveform cosine >= 0.999 vs PyTorch |
| `sakhiya_<tier>_f16.gguf` | all | FP16 |

`SHA256SUMS.txt` lists checksums. Prebuilt: `sakhiya-linux-x86_64.tar.gz` (CLI + libsakhiya.so + sakhiya.h), `sakhiya-wasm.tar.gz` (sakhiya.wasm + JS).
```python
from sakhiya.ggml_tts import SakhiyaGGML
SakhiyaGGML.from_pretrained("atom-lite").say("नमस्ते", style="news")
```
