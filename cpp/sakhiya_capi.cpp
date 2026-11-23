// C ABI implementation of cpp/sakhiya.h on top of the zero-dependency GGUF engine in sakhiya_ggml.hpp.
// Compiled with -DSAKHIYA_STANDALONE_GGUF so that no ggml library is required (native .so, Android NDK, WebAssembly).
#ifndef SAKHIYA_STANDALONE_GGUF
#define SAKHIYA_STANDALONE_GGUF 1
#endif
#include "sakhiya.h"
#include "sakhiya_ggml.hpp"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct sakhiya_ctx {
    sakhiya::SakhiyaGGML engine;
};

static thread_local std::string g_last_error;

extern "C" {

sakhiya_params_t sakhiya_default_params(void) {
    sakhiya_params_t p;
    p.lang = "ne";
    p.style = "news";
    p.rate = -1.0f;
    p.pitch_shift = -999.0f;
    p.pitch_scale = -1.0f;
    p.pause_scale = -1.0f;
    p.energy_scale = -1.0f;
    p.svara = 1.0f;
    return p;
}

const char * sakhiya_last_error(void) { return g_last_error.c_str(); }

sakhiya_ctx_t * sakhiya_init_from_file(const char * path, int n_threads) {
    try {
        std::unique_ptr<sakhiya_ctx> c(new sakhiya_ctx());
        c->engine.load_model(path ? path : "", n_threads);
        return c.release();
    } catch (const std::exception & e) {
        g_last_error = e.what();
        return nullptr;
    }
}

sakhiya_ctx_t * sakhiya_init_from_buffer(const void * data, size_t size, int n_threads) {
    try {
        std::unique_ptr<sakhiya_ctx> c(new sakhiya_ctx());
        c->engine.load_model_from_buffer(static_cast<const uint8_t *>(data), size, n_threads);
        return c.release();
    } catch (const std::exception & e) {
        g_last_error = e.what();
        return nullptr;
    }
}

void sakhiya_free(sakhiya_ctx_t * ctx) { delete ctx; }

const char * sakhiya_model_tier(const sakhiya_ctx_t * ctx) { return ctx ? ctx->engine.cfg_tier.c_str() : ""; }
const char * sakhiya_model_arch(const sakhiya_ctx_t * ctx) { return ctx ? ctx->engine.cfg_arch.c_str() : ""; }
int sakhiya_model_voc_channels(const sakhiya_ctx_t * ctx) { return ctx ? ctx->engine.cfg_voc_ch : 0; }
int sakhiya_model_cfm_steps(const sakhiya_ctx_t * ctx) { return ctx ? ctx->engine.cfg_cfm_steps : 0; }

static bool build_override(const sakhiya_params_t & p, sakhiya::StyleConfig & ctl) {
    const std::string style = p.style ? p.style : "news";
    ctl = sakhiya::get_style_config(style);
    bool has = false;
    if (p.rate > 0.0f)          { ctl.rate = p.rate; has = true; }
    if (p.pitch_shift > -900.f) { ctl.pitch_shift = p.pitch_shift; has = true; }
    if (p.pitch_scale > 0.0f)   { ctl.pitch_scale = p.pitch_scale; has = true; }
    if (p.pause_scale > 0.0f)   { ctl.pause_scale = p.pause_scale; has = true; }
    if (p.energy_scale > 0.0f)  { ctl.energy_scale = p.energy_scale; has = true; }
    return has;
}

static int64_t emit_pcm(const std::vector<float> & w, float ** out_pcm, int * out_sr) {
    if (out_sr) *out_sr = SAKHIYA_SAMPLE_RATE;
    float * buf = static_cast<float *>(std::malloc(std::max<size_t>(w.size(), 1) * sizeof(float)));
    if (!buf) { g_last_error = "out of memory"; return -2; }
    if (!w.empty()) std::memcpy(buf, w.data(), w.size() * sizeof(float));
    *out_pcm = buf;
    return static_cast<int64_t>(w.size());
}

int64_t sakhiya_synthesize(sakhiya_ctx_t * ctx, const char * text, sakhiya_params_t p, float ** out_pcm, int * out_sr) {
    if (!ctx || !text || !out_pcm) { g_last_error = "null argument"; return -1; }
    try {
        sakhiya::StyleConfig ctl;
        const bool has = build_override(p, ctl);
        const std::string lang = p.lang ? p.lang : "ne";
        const std::string style = p.style ? p.style : "news";
        auto w = ctx->engine.say(text, lang, style, p.svara > 0.0f ? p.svara : 1.0f, has ? &ctl : nullptr, nullptr);
        return emit_pcm(w, out_pcm, out_sr);
    } catch (const std::exception & e) {
        g_last_error = e.what();
        return -1;
    }
}

int64_t sakhiya_synthesize_ids(sakhiya_ctx_t * ctx, const int64_t * ids, int n_ids, const float * pitch_add, const float * dm,
                               sakhiya_params_t p, float ** out_pcm, int * out_sr) {
    if (!ctx || !ids || n_ids <= 0 || !out_pcm) { g_last_error = "null/empty argument"; return -1; }
    try {
        sakhiya::StyleConfig ctl;
        build_override(p, ctl);
        const std::string lang = p.lang ? p.lang : "ne";
        std::vector<int64_t> v_ids(ids, ids + n_ids);
        std::vector<float> v_pa(n_ids, 0.0f), v_dm(n_ids, 1.0f);
        if (pitch_add) v_pa.assign(pitch_add, pitch_add + n_ids);
        if (dm) v_dm.assign(dm, dm + n_ids);
        auto ph = ctx->engine.ids_to_phones(v_ids);
        auto w = ctx->engine.synth_from_ids(ph, v_ids, v_dm, v_pa, lang, ctl, true, nullptr);
        return emit_pcm(w, out_pcm, out_sr);
    } catch (const std::exception & e) {
        g_last_error = e.what();
        return -1;
    }
}

void sakhiya_free_pcm(float * pcm) { std::free(pcm); }

} // extern "C"
