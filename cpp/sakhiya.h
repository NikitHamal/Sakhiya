#pragma once
// C ABI for Sakhiya GGML Inference Engine (embeddable in C/C++, Rust, Go, Swift, Android/iOS, Python ctypes)
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sakhiya_ctx sakhiya_ctx_t;

typedef struct sakhiya_params {
    const char * lang;       // "ne", "sa", or "auto"
    const char * style;      // "news", "storytelling", "conversational", "warm_soft", "expressive", "vedic_chant"
    float rate;              // <= 0 uses style default
    float pitch_shift;       // <= -900 uses style default
    float pitch_scale;       // <= 0 uses style default
    float pause_scale;       // <= 0 uses style default
    float energy_scale;      // <= 0 uses style default
    float svara;             // default 1.0f
} sakhiya_params_t;

sakhiya_params_t sakhiya_default_params(void);
sakhiya_ctx_t *  sakhiya_init_from_file(const char * model_gguf_path, int n_threads);
void             sakhiya_free(sakhiya_ctx_t * ctx);

// Synthesize UTF-8 text into newly allocated float32 PCM buffer at 24000 Hz.
// Caller frees *out_pcm with sakhiya_free_pcm(). Returns number of samples or <0 on error.
int64_t sakhiya_synthesize(
    sakhiya_ctx_t * ctx,
    const char * utf8_text,
    sakhiya_params_t params,
    float ** out_pcm,
    int * out_sample_rate);

void sakhiya_free_pcm(float * pcm);

#ifdef __cplusplus
}
#endif
