// Android JNI wrapper around cpp/sakhiya.h (same engine as sakhiya-cli / sakhiya.wasm).
// Kotlin side: android/SakhiyaGGML.kt (package com.sakhiya.tts). PCM is returned as a *direct* ByteBuffer that wraps the
// engine's malloc'd float32 buffer (zero-copy); release it with nativeFreePcm().
#include <jni.h>

#include <string>

#include "sakhiya.h"

namespace {
struct Utf {
    JNIEnv * env; jstring js; const char * c;
    Utf(JNIEnv * e, jstring s) : env(e), js(s), c(s ? e->GetStringUTFChars(s, nullptr) : nullptr) {}
    ~Utf() { if (c) env->ReleaseStringUTFChars(js, c); }
};
void throw_rt(JNIEnv * env, const char * msg) {
    jclass cls = env->FindClass("java/lang/RuntimeException");
    if (cls) env->ThrowNew(cls, msg && *msg ? msg : "sakhiya error");
}
} // namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_com_sakhiya_tts_SakhiyaGGML_nativeInitFile(JNIEnv * env, jclass, jstring path, jint threads) {
    Utf p(env, path);
    sakhiya_ctx_t * c = sakhiya_init_from_file(p.c, threads);
    if (!c) { throw_rt(env, sakhiya_last_error()); return 0; }
    return reinterpret_cast<jlong>(c);
}

// `buf` must be a direct ByteBuffer holding the .gguf bytes (e.g. memory-mapped from assets).
JNIEXPORT jlong JNICALL Java_com_sakhiya_tts_SakhiyaGGML_nativeInitBuffer(JNIEnv * env, jclass, jobject buf, jint threads) {
    void * ptr = env->GetDirectBufferAddress(buf);
    jlong len = env->GetDirectBufferCapacity(buf);
    if (!ptr || len <= 0) { throw_rt(env, "nativeInitBuffer needs a direct ByteBuffer"); return 0; }
    sakhiya_ctx_t * c = sakhiya_init_from_buffer(ptr, static_cast<size_t>(len), threads);
    if (!c) { throw_rt(env, sakhiya_last_error()); return 0; }
    return reinterpret_cast<jlong>(c);
}

JNIEXPORT void JNICALL Java_com_sakhiya_tts_SakhiyaGGML_nativeFree(JNIEnv *, jclass, jlong h) {
    sakhiya_free(reinterpret_cast<sakhiya_ctx_t *>(h));
}

JNIEXPORT jstring JNICALL Java_com_sakhiya_tts_SakhiyaGGML_nativeTier(JNIEnv * env, jclass, jlong h) {
    return env->NewStringUTF(sakhiya_model_tier(reinterpret_cast<sakhiya_ctx_t *>(h)));
}

JNIEXPORT jobject JNICALL Java_com_sakhiya_tts_SakhiyaGGML_nativeSynthesize(
    JNIEnv * env, jclass, jlong h, jstring text, jstring lang, jstring style,
    jfloat rate, jfloat pitchShift, jfloat pitchScale, jfloat pauseScale, jfloat energyScale, jfloat svara) {
    Utf t(env, text), l(env, lang), s(env, style);
    sakhiya_params_t p = sakhiya_default_params();
    if (l.c && *l.c) p.lang = l.c;
    if (s.c && *s.c) p.style = s.c;
    p.rate = rate; p.pitch_shift = pitchShift; p.pitch_scale = pitchScale; p.pause_scale = pauseScale;
    p.energy_scale = energyScale; p.svara = svara;
    float * pcm = nullptr; int sr = 0;
    int64_t n = sakhiya_synthesize(reinterpret_cast<sakhiya_ctx_t *>(h), t.c, p, &pcm, &sr);
    if (n < 0) { throw_rt(env, sakhiya_last_error()); return nullptr; }
    return env->NewDirectByteBuffer(pcm, static_cast<jlong>(n) * static_cast<jlong>(sizeof(float)));
}

JNIEXPORT void JNICALL Java_com_sakhiya_tts_SakhiyaGGML_nativeFreePcm(JNIEnv * env, jclass, jobject pcmBuf) {
    if (pcmBuf) sakhiya_free_pcm(static_cast<float *>(env->GetDirectBufferAddress(pcmBuf)));
}

} // extern "C"
