#pragma once
// C ABI for the Sakhiya GGML/GGUF inference engine (embeddable in C/C++, Rust, Go, Swift, Android JNI, WebAssembly, Python ctypes).
// One implementation (cpp/sakhiya_capi.cpp) backs sakhiya-cli, libsakhiya.so, sakhiya.wasm and the Android JNI wrapper.
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SAKHIYA_SAMPLE_RATE 24000

typedef struct sakhiya_ctx sakhiya_ctx_t;

typedef struct sakhiya_params {
    const char * lang;       // "ne", "sa", "auto" (mixed-script routing) or Tharu variants "thr_*"
    const char * style;      // "news", "storytelling", "conversational", "warm_soft", "expressive", "vedic_chant", ...
    float rate;              // <= 0 uses style default
    float pitch_shift;       // <= -900 uses style default
    float pitch_scale;       // <= 0 uses style default
    float pause_scale;       // <= 0 uses style default
    float energy_scale;      // <= 0 uses style default
    float svara;             // Vedic Svara depth, default 1.0f
} sakhiya_params_t;

sakhiya_params_t sakhiya_default_params(void);

// Load a Sakhiya .gguf (F32 / F16 / Q8_0; V1, Nano, Atom, Atom-Lite) from disk or from memory (the buffer is copied/dequantized,
// the caller may free it right after the call). Returns NULL on error; see sakhiya_last_error().
sakhiya_ctx_t *  sakhiya_init_from_file(const char * model_gguf_path, int n_threads);
sakhiya_ctx_t *  sakhiya_init_from_buffer(const void * gguf_data, size_t gguf_size, int n_threads);
void             sakhiya_free(sakhiya_ctx_t * ctx);

// Model metadata read from the GGUF header (sakhiya.tier / sakhiya.arch / sakhiya.voc_ch / sakhiya.cfm_steps ...).
const char *     sakhiya_model_tier(const sakhiya_ctx_t * ctx);
const char *     sakhiya_model_arch(const sakhiya_ctx_t * ctx);
int              sakhiya_model_voc_channels(const sakhiya_ctx_t * ctx);
int              sakhiya_model_cfm_steps(const sakhiya_ctx_t * ctx);

// Synthesize UTF-8 text into a newly allocated float32 mono PCM buffer at 24000 Hz.
// Caller frees *out_pcm with sakhiya_free_pcm(). Returns number of samples or <0 on error.
int64_t sakhiya_synthesize(
    sakhiya_ctx_t * ctx,
    const char * utf8_text,
    sakhiya_params_t params,
    float ** out_pcm,
    int * out_sample_rate);

// Synthesize from already-phonemized Sakhiya symbol ids (e.g. produced by docs/g2p.js). pitch_add / dm may be NULL
// (zeros / ones). Same output convention as sakhiya_synthesize().
int64_t sakhiya_synthesize_ids(
    sakhiya_ctx_t * ctx,
    const int64_t * ids, int n_ids,
    const float * pitch_add,
    const float * dm,
    sakhiya_params_t params,
    float ** out_pcm,
    int * out_sample_rate);

void        sakhiya_free_pcm(float * pcm);
const char* sakhiya_last_error(void);

#ifdef __cplusplus
}
#endif
