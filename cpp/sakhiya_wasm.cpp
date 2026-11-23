// WebAssembly entry points for the Sakhiya GGUF engine (single-threaded, no pthreads / SharedArrayBuffer needed,
// so it boots on GitHub Pages without COOP/COEP headers).
//
//   em++ -O3 -flto -msimd128 -sSTRICT -sEXPORT_ES6 -sENVIRONMENT=web,worker -sALLOW_MEMORY_GROWTH ... (see scripts/build_wasm.sh)
//
// JS API (all names are exported with a leading underscore by emscripten, e.g. Module._init_model_from_buffer):
//   init_model_from_buffer(ptr, len)        -> 0 on success, <0 on error (GGUF bytes were copied into float32 weights)
//   synthesize_text(text, lang, style, rate, pitch_shift, pitch_scale, pause_scale, energy_scale, svara) -> n samples or <0
//   synthesize_ids(ids_ptr_i64, n, pitch_add_ptr_f32|0, dm_ptr_f32|0, lang, style, rate, ... svara)      -> n samples or <0
//   get_pcm_ptr() / get_pcm_len()           -> float32 mono PCM @ 24 kHz owned by the module (valid until next synth / free)
//   free_model()                            -> releases the model and the last PCM buffer
#include "sakhiya.h"

#include <emscripten/emscripten.h>

#include <cstdlib>
#include <cstring>

static sakhiya_ctx_t * g_ctx = nullptr;
static float * g_pcm = nullptr;
static int64_t g_pcm_len = 0;

static void drop_pcm() {
    if (g_pcm) { sakhiya_free_pcm(g_pcm); g_pcm = nullptr; }
    g_pcm_len = 0;
}

static sakhiya_params_t mk(const char * lang, const char * style, float rate, float ps, float psc, float pause, float eng, float svara) {
    sakhiya_params_t p = sakhiya_default_params();
    if (lang && *lang) p.lang = lang;
    if (style && *style) p.style = style;
    p.rate = rate; p.pitch_shift = ps; p.pitch_scale = psc; p.pause_scale = pause; p.energy_scale = eng;
    p.svara = svara > 0.0f ? svara : 1.0f;
    return p;
}

extern "C" {

EMSCRIPTEN_KEEPALIVE int init_model_from_buffer(const uint8_t * data, size_t len) {
    if (g_ctx) { sakhiya_free(g_ctx); g_ctx = nullptr; }
    drop_pcm();
    g_ctx = sakhiya_init_from_buffer(data, len, 1);
    return g_ctx ? 0 : -1;
}

EMSCRIPTEN_KEEPALIVE int synthesize_text(const char * text, const char * lang, const char * style, float rate, float pitch_shift,
                                         float pitch_scale, float pause_scale, float energy_scale, float svara) {
    if (!g_ctx) return -10;
    drop_pcm();
    int sr = 0;
    int64_t n = sakhiya_synthesize(g_ctx, text, mk(lang, style, rate, pitch_shift, pitch_scale, pause_scale, energy_scale, svara), &g_pcm, &sr);
    if (n < 0) { g_pcm = nullptr; return static_cast<int>(n); }
    g_pcm_len = n;
    return static_cast<int>(n);
}

EMSCRIPTEN_KEEPALIVE int synthesize_ids(const int64_t * ids, int n_ids, const float * pitch_add, const float * dm, const char * lang,
                                        const char * style, float rate, float pitch_shift, float pitch_scale, float pause_scale,
                                        float energy_scale, float svara) {
    if (!g_ctx) return -10;
    drop_pcm();
    int sr = 0;
    int64_t n = sakhiya_synthesize_ids(g_ctx, ids, n_ids, pitch_add, dm,
                                       mk(lang, style, rate, pitch_shift, pitch_scale, pause_scale, energy_scale, svara), &g_pcm, &sr);
    if (n < 0) { g_pcm = nullptr; return static_cast<int>(n); }
    g_pcm_len = n;
    return static_cast<int>(n);
}

EMSCRIPTEN_KEEPALIVE const float * get_pcm_ptr(void) { return g_pcm; }
EMSCRIPTEN_KEEPALIVE int get_pcm_len(void) { return static_cast<int>(g_pcm_len); }
EMSCRIPTEN_KEEPALIVE int get_sample_rate(void) { return SAKHIYA_SAMPLE_RATE; }
EMSCRIPTEN_KEEPALIVE const char * get_last_error(void) { return sakhiya_last_error(); }
EMSCRIPTEN_KEEPALIVE const char * get_model_tier(void) { return g_ctx ? sakhiya_model_tier(g_ctx) : ""; }
EMSCRIPTEN_KEEPALIVE int get_model_voc_channels(void) { return g_ctx ? sakhiya_model_voc_channels(g_ctx) : 0; }
EMSCRIPTEN_KEEPALIVE int get_model_cfm_steps(void) { return g_ctx ? sakhiya_model_cfm_steps(g_ctx) : 0; }

EMSCRIPTEN_KEEPALIVE void free_model(void) {
    drop_pcm();
    if (g_ctx) { sakhiya_free(g_ctx); g_ctx = nullptr; }
}

} // extern "C"
