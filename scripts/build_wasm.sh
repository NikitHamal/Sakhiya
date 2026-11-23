#!/usr/bin/env bash
# Build docs/wasm/sakhiya.wasm + sakhiya_wasm.js (single-threaded WASM SIMD128, ES module, web+worker).
# Usage: source /path/to/emsdk/emsdk_env.sh && scripts/build_wasm.sh
# NOTE: deliberately NO -pthread: GitHub Pages cannot send COOP/COEP headers, so SharedArrayBuffer/pthread WASM would not boot.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/docs/wasm}"
mkdir -p "$OUT"
COMMON=(-std=c++17 -O3 -flto -msimd128 -fno-math-errno -fno-trapping-math -fexceptions -Wno-unknown-pragmas
        -DSAKHIYA_STANDALONE_GGUF -I"$ROOT/cpp"
        -sSTRICT -sALLOW_MEMORY_GROWTH -sINITIAL_MEMORY=33554432 -sSTACK_SIZE=1048576 -sMODULARIZE -sEXPORT_NAME=createSakhiya
        -sEXPORTED_FUNCTIONS=_malloc,_free,_init_model_from_buffer,_synthesize_text,_synthesize_ids,_get_pcm_ptr,_get_pcm_len,_get_sample_rate,_get_last_error,_get_model_tier,_get_model_voc_channels,_get_model_cfm_steps,_free_model
        -sEXPORTED_RUNTIME_METHODS=UTF8ToString,stringToUTF8,lengthBytesUTF8,HEAPU8,HEAPF32,HEAP32
        -sFILESYSTEM=0 -sASSERTIONS=0)
SRC=("$ROOT/cpp/sakhiya_wasm.cpp" "$ROOT/cpp/sakhiya_capi.cpp")
# 1) the shipped browser build
em++ "${COMMON[@]}" -sEXPORT_ES6 -sENVIRONMENT=web,worker "${SRC[@]}" -o "$OUT/sakhiya_wasm.js"
mv -f "$OUT/sakhiya_wasm.wasm" "$OUT/sakhiya.wasm"
# the loader resolves the binary via locateFile (docs/wasm/sakhiya_engine.js); also patch the glue default name
sed -i 's/sakhiya_wasm\.wasm/sakhiya.wasm/g' "$OUT/sakhiya_wasm.js"
# 2) node test glue (identical wasm module, node-capable JS glue) -> $OUT/../../build/wasm_node
NODE_OUT="$ROOT/build/wasm_node"; mkdir -p "$NODE_OUT"
em++ "${COMMON[@]}" -sEXPORT_ES6 -sENVIRONMENT=web,worker,node "${SRC[@]}" -o "$NODE_OUT/sakhiya_wasm.js"
cmp "$NODE_OUT/sakhiya_wasm.wasm" "$OUT/sakhiya.wasm" && echo "wasm binaries identical"
ls -l "$OUT"
