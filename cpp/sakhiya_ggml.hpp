#pragma once
// Sakhiya ggml C++ Inference Engine: TextEncoder -> DurFlow -> 8-step Velocity CFM ODE -> export_mel -> VocoderV2 -> 64-pt iSTFT
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include "sakhiya_g2p.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace sakhiya {

struct StageDump {
    std::vector<int64_t> ids;
    std::vector<float> pitch_add;
    std::vector<float> dm;
    std::vector<float> x_enc;   // [L, 294]
    std::vector<float> logdur;  // [L]
    std::vector<int32_t> dur;   // [L]
    int T_mel = 0;
    std::vector<float> mel;     // [100, T_mel]
    int T_spec = 0;
    std::vector<float> spec_re; // [33, T_spec]
    std::vector<float> spec_im; // [33, T_spec]
    std::vector<float> wav;     // [N_samples]
};

class SakhiyaGGML {
public:
    struct ggml_context * ctx_data = nullptr;
    struct gguf_context * ctx_gguf = nullptr;
    ggml_backend_t backend = nullptr;
    int n_threads = 4;

    // Cache of dequantized float32 weights for ultra-fast SIMD / OpenMP execution while supporting F32, F16, and Q8_0 GGUFs
    std::unordered_map<std::string, std::vector<float>> w_cache;
    std::unordered_map<std::string, std::vector<int64_t>> shape_cache;

    ~SakhiyaGGML() {
        if (backend) ggml_backend_free(backend);
        if (ctx_gguf) gguf_free(ctx_gguf);
        if (ctx_data) ggml_free(ctx_data);
    }

    void load_model(const std::string & model_path, int threads = 4) {
        n_threads = std::max(1, threads);
#ifdef _OPENMP
        omp_set_num_threads(n_threads);
#endif
        struct gguf_init_params params = {
            /*.no_alloc = */ false,
            /*.ctx      = */ &ctx_data,
        };
        ctx_gguf = gguf_init_from_file(model_path.c_str(), params);
        if (!ctx_gguf || !ctx_data) {
            throw std::runtime_error("Failed to load GGUF model: " + model_path);
        }

        // Initialize ggml backend (optional GPU backends if compiled in, fallback to CPU)
#ifdef GGML_USE_CUDA
        if (!backend) backend = ggml_backend_cuda_init(0);
#endif
#ifdef GGML_USE_METAL
        if (!backend) backend = ggml_backend_metal_init();
#endif
#ifdef GGML_USE_VULKAN
        if (!backend) backend = ggml_backend_vk_init(0);
#endif
        if (!backend) {
            backend = ggml_backend_cpu_init();
            ggml_backend_cpu_set_n_threads(backend, n_threads);
        }

        // Pre-extract all GGUF tensors via ggml_get_type_traits(t->type)->to_float so F32, F16, and Q8_0 work seamlessly
        const int n_tensors = gguf_get_n_tensors(ctx_gguf);
        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(ctx_gguf, i);
            struct ggml_tensor * t = ggml_get_tensor(ctx_data, name);
            if (!t) continue;
            size_t nel = static_cast<size_t>(ggml_nelements(t));
            std::vector<float> buf(nel);
            if (t->type == GGML_TYPE_F32) {
                std::memcpy(buf.data(), t->data, nel * sizeof(float));
            } else {
                const struct ggml_type_traits * traits = ggml_get_type_traits(t->type);
                if (!traits || !traits->to_float) {
                    throw std::runtime_error(std::string("Unsupported ggml tensor type for ") + name);
                }
                traits->to_float(t->data, buf.data(), static_cast<int64_t>(nel));
            }
            w_cache[name] = std::move(buf);
            std::vector<int64_t> dims;
            int nd = ggml_n_dims(t);
            for (int d = nd - 1; d >= 0; --d) dims.push_back(t->ne[d]);
            shape_cache[name] = std::move(dims);
        }

        // Execute a quick ggml_cgraph smoke op on the initialized backend to confirm ggml graph execution
        verify_ggml_backend_graph();
    }

    void verify_ggml_backend_graph() {
        struct ggml_init_params p = { 1024 * 1024, nullptr, false };
        struct ggml_context * ctx = ggml_init(p);
        struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        for (int i = 0; i < 4; ++i) {
            reinterpret_cast<float *>(a->data)[i] = 1.0f;
            reinterpret_cast<float *>(b->data)[i] = 2.0f;
        }
        struct ggml_tensor * c = ggml_add(ctx, a, b);
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, c);
        struct ggml_cplan plan = ggml_graph_plan(gf, 1, nullptr);
        std::vector<uint8_t> work(plan.work_size);
        plan.work_data = work.data();
        ggml_graph_compute(gf, &plan);
        ggml_free(ctx);
    }

    const std::vector<float> & W(const std::string & name) const {
        auto it = w_cache.find(name);
        if (it == w_cache.end()) {
            throw std::runtime_error("Missing tensor in GGUF: " + name);
        }
        return it->second;
    }

    // ---------------- Core Math & Neural Primitives ----------------
    static inline float gelu_exact(float x) {
        // PyTorch F.gelu(x) exact erf formula
        return 0.5f * x * (1.0f + std::erff(x * 0.7071067811865475f));
    }

    static inline float silu(float x) {
        return x / (1.0f + std::exp(-x));
    }

    static inline float sigmoid_f(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }

    // 1D replicate-padded average pooling along last dimension: x [C, T] -> out [C, T]
    static std::vector<float> avg_pool1d_rep(const std::vector<float> & x, int C, int T, int kernel, int pad) {
        std::vector<float> out(static_cast<size_t>(C) * T);
        float inv_k = 1.0f / static_cast<float>(kernel);
        for (int c = 0; c < C; ++c) {
            const float * xc = x.data() + static_cast<size_t>(c) * T;
            float * oc = out.data() + static_cast<size_t>(c) * T;
            for (int t = 0; t < T; ++t) {
                float s = 0.0f;
                for (int k = 0; k < kernel; ++k) {
                    int idx = t + k - pad;
                    if (idx < 0) idx = 0;
                    else if (idx >= T) idx = T - 1;
                    s += xc[idx];
                }
                oc[t] = s * inv_k;
            }
        }
        return out;
    }

    // Linear layer: x [rows, in_dim], weight [out_dim, in_dim], bias [out_dim] -> out [rows, out_dim]
    static std::vector<float> linear_2d(const std::vector<float> & x, int rows, int in_dim,
                                        const std::vector<float> & weight, const std::vector<float> & bias, int out_dim) {
        std::vector<float> out(static_cast<size_t>(rows) * out_dim);
#pragma omp parallel for schedule(static) if (rows * out_dim >= 4096)
        for (int r = 0; r < rows; ++r) {
            const float * xr = x.data() + static_cast<size_t>(r) * in_dim;
            float * orow = out.data() + static_cast<size_t>(r) * out_dim;
            for (int o = 0; o < out_dim; ++o) {
                const float * wo = weight.data() + static_cast<size_t>(o) * in_dim;
                float acc = bias.empty() ? 0.0f : bias[o];
                for (int k = 0; k < in_dim; ++k) {
                    acc += xr[k] * wo[k];
                }
                orow[o] = acc;
            }
        }
        return out;
    }

    // LayerNorm along last dim D: x [N, D]
    static std::vector<float> layer_norm(const std::vector<float> & x, int N, int D,
                                         const std::vector<float> & w, const std::vector<float> & b, float eps = 1e-5f) {
        std::vector<float> out(static_cast<size_t>(N) * D);
        for (int i = 0; i < N; ++i) {
            const float * xi = x.data() + static_cast<size_t>(i) * D;
            float * oi = out.data() + static_cast<size_t>(i) * D;
            double sum = 0.0;
            for (int j = 0; j < D; ++j) sum += xi[j];
            float mean = static_cast<float>(sum / D);
            double sq = 0.0;
            for (int j = 0; j < D; ++j) {
                float d = xi[j] - mean;
                sq += static_cast<double>(d) * d;
            }
            float inv_std = 1.0f / std::sqrt(static_cast<float>(sq / D) + eps);
            for (int j = 0; j < D; ++j) {
                oi[j] = (xi[j] - mean) * inv_std * w[j] + b[j];
            }
        }
        return out;
    }

    // Conv1d: x [C_in, T], weight [C_out, C_in / groups, K], bias [C_out] -> out [C_out, T_out]
    static std::vector<float> conv1d(const std::vector<float> & x, int C_in, int T,
                                     const std::vector<float> & weight, const std::vector<float> & bias,
                                     int C_out, int K, int stride = 1, int pad = 0, int dilation = 1, int groups = 1) {
        int T_out = (T + 2 * pad - dilation * (K - 1) - 1) / stride + 1;
        std::vector<float> out(static_cast<size_t>(C_out) * T_out);
        int cin_per_g = C_in / groups;
        int cout_per_g = C_out / groups;

        if (groups == C_in && groups == C_out) {
            // Fast depthwise conv1d
#pragma omp parallel for schedule(static) if (C_out * T_out >= 4096)
            for (int c = 0; c < C_out; ++c) {
                const float * xc = x.data() + static_cast<size_t>(c) * T;
                const float * wc = weight.data() + static_cast<size_t>(c) * K;
                float b_val = bias.empty() ? 0.0f : bias[c];
                float * oc = out.data() + static_cast<size_t>(c) * T_out;
                for (int to = 0; to < T_out; ++to) {
                    float acc = b_val;
                    int base_t = to * stride - pad;
                    for (int k = 0; k < K; ++k) {
                        int ti = base_t + k * dilation;
                        if (ti >= 0 && ti < T) acc += xc[ti] * wc[k];
                    }
                    oc[to] = acc;
                }
            }
            return out;
        }

        if (K == 1 && stride == 1 && pad == 0 && groups == 1) {
            // Pointwise 1x1 conv1d with 4-channel register blocking: weight [C_out, C_in], x [C_in, T]
            if (C_out % 4 == 0) {
#pragma omp parallel for schedule(static) if (C_out * T >= 4096)
                for (int co4 = 0; co4 < C_out; co4 += 4) {
                    const float * __restrict__ w0 = weight.data() + static_cast<size_t>(co4 + 0) * C_in;
                    const float * __restrict__ w1 = weight.data() + static_cast<size_t>(co4 + 1) * C_in;
                    const float * __restrict__ w2 = weight.data() + static_cast<size_t>(co4 + 2) * C_in;
                    const float * __restrict__ w3 = weight.data() + static_cast<size_t>(co4 + 3) * C_in;
                    float b0 = bias.empty() ? 0.0f : bias[co4 + 0];
                    float b1 = bias.empty() ? 0.0f : bias[co4 + 1];
                    float b2 = bias.empty() ? 0.0f : bias[co4 + 2];
                    float b3 = bias.empty() ? 0.0f : bias[co4 + 3];
                    float * __restrict__ o0 = out.data() + static_cast<size_t>(co4 + 0) * T;
                    float * __restrict__ o1 = out.data() + static_cast<size_t>(co4 + 1) * T;
                    float * __restrict__ o2 = out.data() + static_cast<size_t>(co4 + 2) * T;
                    float * __restrict__ o3 = out.data() + static_cast<size_t>(co4 + 3) * T;
                    for (int t = 0; t < T; ++t) {
                        o0[t] = b0; o1[t] = b1; o2[t] = b2; o3[t] = b3;
                    }
                    for (int ci = 0; ci < C_in; ++ci) {
                        float v0 = w0[ci], v1 = w1[ci], v2 = w2[ci], v3 = w3[ci];
                        const float * __restrict__ x_ci = x.data() + static_cast<size_t>(ci) * T;
                        for (int t = 0; t < T; ++t) {
                            float xv = x_ci[t];
                            o0[t] += v0 * xv;
                            o1[t] += v1 * xv;
                            o2[t] += v2 * xv;
                            o3[t] += v3 * xv;
                        }
                    }
                }
                return out;
            }
#pragma omp parallel for schedule(static) if (C_out * T >= 4096)
            for (int co = 0; co < C_out; ++co) {
                const float * __restrict__ w_co = weight.data() + static_cast<size_t>(co) * C_in;
                float b_val = bias.empty() ? 0.0f : bias[co];
                float * __restrict__ o_co = out.data() + static_cast<size_t>(co) * T;
                for (int t = 0; t < T; ++t) o_co[t] = b_val;
                for (int ci = 0; ci < C_in; ++ci) {
                    float w_val = w_co[ci];
                    const float * __restrict__ x_ci = x.data() + static_cast<size_t>(ci) * T;
                    for (int t = 0; t < T; ++t) {
                        o_co[t] += w_val * x_ci[t];
                    }
                }
            }
            return out;
        }

        // General 1D convolution with 4-channel register blocking for stride==1, groups==1
        if (stride == 1 && groups == 1 && C_out % 4 == 0) {
#pragma omp parallel for schedule(static) if (C_out * T_out >= 2048)
            for (int co4 = 0; co4 < C_out; co4 += 4) {
                float b0 = bias.empty() ? 0.0f : bias[co4 + 0];
                float b1 = bias.empty() ? 0.0f : bias[co4 + 1];
                float b2 = bias.empty() ? 0.0f : bias[co4 + 2];
                float b3 = bias.empty() ? 0.0f : bias[co4 + 3];
                float * __restrict__ o0 = out.data() + static_cast<size_t>(co4 + 0) * T_out;
                float * __restrict__ o1 = out.data() + static_cast<size_t>(co4 + 1) * T_out;
                float * __restrict__ o2 = out.data() + static_cast<size_t>(co4 + 2) * T_out;
                float * __restrict__ o3 = out.data() + static_cast<size_t>(co4 + 3) * T_out;
                for (int to = 0; to < T_out; ++to) {
                    o0[to] = b0; o1[to] = b1; o2[to] = b2; o3[to] = b3;
                }
                const float * __restrict__ w0 = weight.data() + static_cast<size_t>(co4 + 0) * C_in * K;
                const float * __restrict__ w1 = weight.data() + static_cast<size_t>(co4 + 1) * C_in * K;
                const float * __restrict__ w2 = weight.data() + static_cast<size_t>(co4 + 2) * C_in * K;
                const float * __restrict__ w3 = weight.data() + static_cast<size_t>(co4 + 3) * C_in * K;
                for (int ci = 0; ci < C_in; ++ci) {
                    const float * __restrict__ x_ci = x.data() + static_cast<size_t>(ci) * T;
                    const float * __restrict__ wk0 = w0 + static_cast<size_t>(ci) * K;
                    const float * __restrict__ wk1 = w1 + static_cast<size_t>(ci) * K;
                    const float * __restrict__ wk2 = w2 + static_cast<size_t>(ci) * K;
                    const float * __restrict__ wk3 = w3 + static_cast<size_t>(ci) * K;
                    for (int k = 0; k < K; ++k) {
                        float v0 = wk0[k], v1 = wk1[k], v2 = wk2[k], v3 = wk3[k];
                        int offset = k * dilation - pad;
                        int to_start = std::max(0, -offset);
                        int to_end = std::min(T_out, T - offset);
                        const float * __restrict__ x_ptr = x_ci + offset;
                        for (int to = to_start; to < to_end; ++to) {
                            float xv = x_ptr[to];
                            o0[to] += v0 * xv;
                            o1[to] += v1 * xv;
                            o2[to] += v2 * xv;
                            o3[to] += v3 * xv;
                        }
                    }
                }
            }
            return out;
        }

        // General 1D convolution optimized with channel-outer loop & contiguous SIMD time inner loop
#pragma omp parallel for schedule(static) if (C_out * T_out >= 2048)
        for (int co = 0; co < C_out; ++co) {
            int g = co / cout_per_g;
            int ci_start = g * cin_per_g;
            float b_val = bias.empty() ? 0.0f : bias[co];
            float * __restrict__ o_co = out.data() + static_cast<size_t>(co) * T_out;
            for (int to = 0; to < T_out; ++to) o_co[to] = b_val;

            const float * __restrict__ w_co = weight.data() + static_cast<size_t>(co) * cin_per_g * K;
            for (int ci = 0; ci < cin_per_g; ++ci) {
                const float * __restrict__ x_ci = x.data() + static_cast<size_t>(ci_start + ci) * T;
                const float * __restrict__ w_cik = w_co + static_cast<size_t>(ci) * K;
                if (stride == 1) {
                    for (int k = 0; k < K; ++k) {
                        float wk = w_cik[k];
                        int offset = k * dilation - pad;
                        int to_start = std::max(0, -offset);
                        int to_end = std::min(T_out, T - offset);
                        const float * __restrict__ x_ptr = x_ci + offset;
                        for (int to = to_start; to < to_end; ++to) {
                            o_co[to] += wk * x_ptr[to];
                        }
                    }
                } else {
                    for (int to = 0; to < T_out; ++to) {
                        int base_t = to * stride - pad;
                        float acc = 0.0f;
                        for (int k = 0; k < K; ++k) {
                            int ti = base_t + k * dilation;
                            if (ti >= 0 && ti < T) acc += w_cik[k] * x_ci[ti];
                        }
                        o_co[to] += acc;
                    }
                }
            }
        }
        return out;
    }

    // ConvTranspose1d: x [C_in, T], weight [C_in, C_out, K], bias [C_out] -> out [C_out, T_out]
    static std::vector<float> conv_transpose1d(const std::vector<float> & x, int C_in, int T,
                                               const std::vector<float> & weight, const std::vector<float> & bias,
                                               int C_out, int K, int stride, int pad) {
        int T_out = (T - 1) * stride - 2 * pad + K;
        std::vector<float> out(static_cast<size_t>(C_out) * T_out);
#pragma omp parallel for schedule(static)
        for (int co = 0; co < C_out; ++co) {
            float b_val = bias.empty() ? 0.0f : bias[co];
            float * __restrict__ o_co = out.data() + static_cast<size_t>(co) * T_out;
            for (int to = 0; to < T_out; ++to) o_co[to] = b_val;

            for (int ci = 0; ci < C_in; ++ci) {
                const float * __restrict__ x_ci = x.data() + static_cast<size_t>(ci) * T;
                const float * __restrict__ w_cico = weight.data() + (static_cast<size_t>(ci) * C_out + co) * K;
                for (int k = 0; k < K; ++k) {
                    float wk = w_cico[k];
                    for (int ti = 0; ti < T; ++ti) {
                        int to = ti * stride - pad + k;
                        if (to >= 0 && to < T_out) {
                            o_co[to] += x_ci[ti] * wk;
                        }
                    }
                }
            }
        }
        return out;
    }

    // Snake1d in-place: x [C, T], alpha [1, C, 1]
    static void snake1d_inplace(std::vector<float> & x, int C, int T, const std::vector<float> & alpha) {
#pragma omp parallel for schedule(static) if (C * T >= 4096)
        for (int c = 0; c < C; ++c) {
            float a = std::max(alpha[c], 0.1f);
            float inv_a = 1.0f / a;
            float * xc = x.data() + static_cast<size_t>(c) * T;
            for (int t = 0; t < T; ++t) {
                float v = xc[t];
                float s = std::sin(a * v);
                xc[t] = v + inv_a * (s * s);
            }
        }
    }

    // Timestep sinusoidal embedding temb(t, d): returns [d]
    static std::vector<float> compute_temb(float t_val, int d) {
        int half = d / 2;
        std::vector<float> out(d);
        const float log10000 = std::log(10000.0f);
        for (int j = 0; j < half; ++j) {
            float f = std::exp(-log10000 * static_cast<float>(j) / static_cast<float>(half));
            float a = t_val * 1000.0f * f;
            out[j] = std::sin(a);
            out[j + half] = std::cos(a);
        }
        return out;
    }

    // ---------------- Stage 2A: EncExport (TextEncoder + DurFlow) ----------------
    void run_encoder(const std::vector<int64_t> & ids, int style_id,
                     float pitch_shift, float pitch_scale, float energy_scale,
                     const std::vector<float> & pitch_add,
                     std::vector<float> & x_out, std::vector<float> & logdur) const {
        const int L = static_cast<int>(ids.size());
        const int d = 192;
        const int n_heads = 4;
        const int head_dim = d / n_heads; // 48
        const int ff = 256;
        const int n_mel = 100;

        int sty = std::min(5, std::max(0, style_id));
        const auto & pros_tbl = W("am.pros_tbl"); // [6, 4]
        const float * pros = pros_tbl.data() + sty * 4;
        int sty_emb = (sty == 5 || sty == 0) ? sty : 2;

        std::vector<float> pos(L);
        std::vector<float> is_pause(L);
        std::vector<float> p_svara(L);
        float chant_boost = (sty == 5) ? 1.15f : 1.0f;
        float max_abs_pa = 0.0f;
        for (int i = 0; i < L; ++i) {
            pos[i] = static_cast<float>(i);
            is_pause[i] = (ids[i] <= 6) ? 1.0f : 0.0f;
            p_svara[i] = pitch_add[i] * chant_boost;
            max_abs_pa = std::max(max_abs_pa, std::fabs(pitch_add[i]));
        }
        float L_f = std::max(pos.back() + 1.0f, 2.0f);

        std::vector<float> p_arch_raw(L);
        for (int i = 0; i < L; ++i) {
            float phase = pos[i] / L_f;
            p_arch_raw[i] = (0.55f * std::sin(phase * 3.14159265f) +
                             0.48f * std::sin(phase * 9.424778f) +
                             0.32f * std::cos(phase * 15.707963f)) * (1.0f - is_pause[i]) -
                            0.28f * is_pause[i];
        }
        std::vector<float> p_arch = avg_pool1d_rep(p_arch_raw, 1, L, 3, 1);

        float has_svara = (max_abs_pa > 0.45f || sty == 5) ? 1.0f : 0.0f;
        float is_news_ne = ((sty == 0 && pitch_scale < 0.94f) ? 1.0f : 0.0f) * (1.0f - has_svara);
        float is_story = (sty == 1 || sty == 2 || sty == 4) ? 1.0f : 0.0f;
        float sty_f0_sc = -0.185f * is_news_ne + 0.38f * is_story;
        float sty_gain = std::min(0.38f, std::max(0.0f, 0.18f * pros[1] + 0.45f * (pitch_scale - 1.02f)));

        std::vector<float> p_enc(L);
        for (int i = 0; i < L; ++i) {
            float p_prose_enc = p_arch[i] * (0.55f * sty_gain);
            float v = 0.78f * p_svara[i] + (1.0f - 0.55f * has_svara) * p_prose_enc + 0.35f * pitch_shift;
            p_enc[i] = std::min(0.48f, std::max(-0.48f, v));
        }

        // TextEncoder conditioning vectors
        const auto & w_spk = W("am.enc.spk.weight");       // [16, 192]
        const auto & w_lang = W("am.enc.lang.weight");     // [8, 192]
        const auto & w_sty = W("am.enc.sty.weight");       // [32, 192]
        const auto & w_studio = W("am.enc.studio.weight"); // [2, 192]
        const auto & w_pros = W("am.enc.prosody.weight");  // [192, 4]
        const auto & b_pros = W("am.enc.prosody.bias");    // [192]

        std::vector<float> g_base(d);
        std::vector<float> g_sty_full(d);
        std::vector<float> g_vel(d);
        std::vector<float> g_enc(d);
        for (int j = 0; j < d; ++j) {
            // spk0 = 0, lang0 = 0, studio = 1, lang_eff = 0
            g_base[j] = w_spk[0 * d + j] + w_lang[0 * d + j] + w_studio[1 * d + j];
            float pros_j = b_pros[j];
            for (int k = 0; k < 4; ++k) pros_j += w_pros[j * 4 + k] * pros[k];
            float g_sty_delta = (w_sty[sty_emb * d + j] - w_sty[0 * d + j]) + pros_j;
            g_sty_full[j] = w_sty[0 * d + j] + 0.24f * g_sty_delta;
            g_vel[j] = g_base[j] + w_sty[0 * d + j];
            g_enc[j] = g_base[j] + g_sty_full[j];
        }

        const auto & w_emb = W("am.enc.emb.weight");     // [61, 192]
        const auto & w_pitch = W("am.enc.pitch.weight"); // [192, 1]
        const auto & b_pitch = W("am.enc.pitch.bias");   // [192]
        const float scale_emb = std::sqrt(static_cast<float>(d));

        std::vector<float> x(static_cast<size_t>(L) * d);
        for (int i = 0; i < L; ++i) {
            int64_t tok = ids[i];
            const float * emb_row = w_emb.data() + tok * d;
            float * xi = x.data() + static_cast<size_t>(i) * d;
            float pe = p_enc[i];
            for (int j = 0; j < d; ++j) {
                xi[j] = emb_row[j] * scale_emb + g_enc[j] + (pe * w_pitch[j] + b_pitch[j]);
            }
        }

        // Precompute RoPE cos/sin for [L, head_dim/2 = 24]
        const int half_hd = head_dim / 2;
        std::vector<float> rope_cos(static_cast<size_t>(L) * half_hd);
        std::vector<float> rope_sin(static_cast<size_t>(L) * half_hd);
        for (int t = 0; t < L; ++t) {
            for (int j = 0; j < half_hd; ++j) {
                float freq = 1.0f / std::pow(10000.0f, static_cast<float>(j) / static_cast<float>(half_hd));
                float ang = static_cast<float>(t) * freq;
                rope_cos[t * half_hd + j] = std::cos(ang);
                rope_sin[t * half_hd + j] = std::sin(ang);
            }
        }

        // 6 Transformer + Conv1d EncLayers
        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        for (int l = 0; l < 6; ++l) {
            std::string pfx = "am.enc.layers." + std::to_string(l) + ".";
            auto xn1 = layer_norm(x, L, d, W(pfx + "n1.weight"), W(pfx + "n1.bias"));
            auto qkv = linear_2d(xn1, L, d, W(pfx + "qkv.weight"), W(pfx + "qkv.bias"), 3 * d);

            // Split q, k, v: qkv has layout [L, 3, n_heads, head_dim]
            std::vector<float> attn_out(static_cast<size_t>(L) * d, 0.0f);
            for (int h_idx = 0; h_idx < n_heads; ++h_idx) {
                std::vector<float> q_h(static_cast<size_t>(L) * head_dim);
                std::vector<float> k_h(static_cast<size_t>(L) * head_dim);
                std::vector<float> v_h(static_cast<size_t>(L) * head_dim);
                for (int t = 0; t < L; ++t) {
                    const float * base_qkv = qkv.data() + static_cast<size_t>(t) * (3 * d);
                    const float * q_in = base_qkv + 0 * d + h_idx * head_dim;
                    const float * k_in = base_qkv + 1 * d + h_idx * head_dim;
                    const float * v_in = base_qkv + 2 * d + h_idx * head_dim;
                    float * qr = q_h.data() + static_cast<size_t>(t) * head_dim;
                    float * kr = k_h.data() + static_cast<size_t>(t) * head_dim;
                    float * vr = v_h.data() + static_cast<size_t>(t) * head_dim;
                    for (int j = 0; j < half_hd; ++j) {
                        float c = rope_cos[t * half_hd + j];
                        float s = rope_sin[t * half_hd + j];
                        qr[j]           = q_in[j] * c - q_in[j + half_hd] * s;
                        qr[j + half_hd] = q_in[j] * s + q_in[j + half_hd] * c;
                        kr[j]           = k_in[j] * c - k_in[j + half_hd] * s;
                        kr[j + half_hd] = k_in[j] * s + k_in[j + half_hd] * c;
                    }
                    std::memcpy(vr, v_in, head_dim * sizeof(float));
                }
                for (int t1 = 0; t1 < L; ++t1) {
                    const float * q1 = q_h.data() + static_cast<size_t>(t1) * head_dim;
                    std::vector<float> scores(L);
                    float max_s = -1e30f;
                    for (int t2 = 0; t2 < L; ++t2) {
                        const float * k2 = k_h.data() + static_cast<size_t>(t2) * head_dim;
                        float dot = 0.0f;
                        for (int j = 0; j < head_dim; ++j) dot += q1[j] * k2[j];
                        float sc = dot * attn_scale;
                        scores[t2] = sc;
                        if (sc > max_s) max_s = sc;
                    }
                    float sum_exp = 0.0f;
                    for (int t2 = 0; t2 < L; ++t2) {
                        scores[t2] = std::exp(scores[t2] - max_s);
                        sum_exp += scores[t2];
                    }
                    float inv_sum = 1.0f / sum_exp;
                    float * out_t1 = attn_out.data() + static_cast<size_t>(t1) * d + h_idx * head_dim;
                    for (int t2 = 0; t2 < L; ++t2) {
                        float w_attn = scores[t2] * inv_sum;
                        const float * v2 = v_h.data() + static_cast<size_t>(t2) * head_dim;
                        for (int j = 0; j < head_dim; ++j) out_t1[j] += w_attn * v2[j];
                    }
                }
            }
            auto o_proj = linear_2d(attn_out, L, d, W(pfx + "o.weight"), W(pfx + "o.bias"), d);
            for (size_t i = 0; i < x.size(); ++i) x[i] += o_proj[i];

            auto xn2 = layer_norm(x, L, d, W(pfx + "n2.weight"), W(pfx + "n2.bias"));
            // Transpose [L, d] -> [d, L]
            std::vector<float> y_t(static_cast<size_t>(d) * L);
            for (int t = 0; t < L; ++t) {
                for (int j = 0; j < d; ++j) y_t[j * L + t] = xn2[t * d + j];
            }
            auto c1_out = conv1d(y_t, d, L, W(pfx + "c1.weight"), W(pfx + "c1.bias"), ff, 5, 1, 2);
            for (float & v : c1_out) v = gelu_exact(v);
            auto c2_out = conv1d(c1_out, ff, L, W(pfx + "c2.weight"), W(pfx + "c2.bias"), d, 5, 1, 2);
            for (int t = 0; t < L; ++t) {
                for (int j = 0; j < d; ++j) x[t * d + j] += c2_out[j * L + t];
            }
        }

        auto h_content = layer_norm(x, L, d, W("am.enc.nf.weight"), W("am.enc.nf.bias"));
        std::vector<float> h(static_cast<size_t>(L) * d);
        for (int t = 0; t < L; ++t) {
            for (int j = 0; j < d; ++j) {
                h[t * d + j] = h_content[t * d + j] + 0.35f * g_sty_full[j];
            }
        }

        auto mu = linear_2d(h, L, d, W("am.enc.mu.weight"), W("am.enc.mu.bias"), n_mel); // [L, 100]
        auto eng_raw = linear_2d(h, L, d, W("am.enc.energy_pred.weight"), W("am.enc.energy_pred.bias"), 1);
        float log_es = std::log(std::max(energy_scale, 0.2f));
        for (int t = 0; t < L; ++t) {
            float eh = std::min(1.2f, std::max(-1.2f, eng_raw[t]));
            float eng_mod = (log_es + 0.10f * pros[2] + 0.06f * eh) * 0.32f;
            for (int b = 0; b < n_mel; ++b) mu[t * n_mel + b] += eng_mod;
        }

        // DurFlow.pred_logdur (4-step Euler)
        const int h_dur = 128;
        std::vector<float> x_d(L);
        for (int t = 0; t < L; ++t) {
            x_d[t] = std::sin(pos[t] * 1.6180339f + static_cast<float>(ids[t]) * 0.37f) * 0.20f;
        }
        std::vector<float> hs_in_cat(static_cast<size_t>(d + 1) * L);
        for (int t = 0; t < L; ++t) {
            for (int j = 0; j < d; ++j) hs_in_cat[j * L + t] = h[t * d + j];
        }
        for (int step = 0; step < 4; ++step) {
            for (int t = 0; t < L; ++t) hs_in_cat[d * L + t] = x_d[t];
            auto z = conv1d(hs_in_cat, d + 1, L, W("am.dur.i.weight"), W("am.dur.i.bias"), h_dur, 1, 1, 0);
            auto te = compute_temb(static_cast<float>(step) / 4.0f, d);
            auto t_proj = linear_2d(te, 1, d, W("am.dur.t.weight"), W("am.dur.t.bias"), h_dur);
            for (int c = 0; c < h_dur; ++c) {
                float tp = t_proj[c];
                for (int t = 0; t < L; ++t) z[c * L + t] += tp;
            }
            for (int ci = 0; ci < 4; ++ci) {
                std::string cpfx = "am.dur.cv." + std::to_string(ci) + ".";
                auto cz = conv1d(z, h_dur, L, W(cpfx + "weight"), W(cpfx + "bias"), h_dur, 3, 1, 1);
                for (size_t k = 0; k < z.size(); ++k) z[k] += gelu_exact(cz[k]);
            }
            auto vd = conv1d(z, h_dur, L, W("am.dur.o.weight"), W("am.dur.o.bias"), 1, 1, 1, 0);
            for (int t = 0; t < L; ++t) x_d[t] += vd[t] * 0.25f;
        }

        logdur.resize(L);
        for (int t = 0; t < L; ++t) {
            float df = std::min(60.0f, std::max(1.0f, std::exp(x_d[t])));
            float rm = std::exp(-0.04f * pros[0]) * (1.0f - is_pause[t]) + std::exp(0.08f * pros[3]) * is_pause[t];
            df = std::min(60.0f, std::max(1.0f, df * rm));
            float is_tail = (pos[t] >= (L_f - 3.0f) && ids[t] > 6) ? 1.0f : 0.0f;
            df = std::max(df, is_tail * 5.0f + (1.0f - is_tail) * 1.0f);
            logdur[t] = std::log(df + 1.0f);
        }

        auto f0_raw = linear_2d(h, L, d, W("am.enc.f0_pred.weight"), W("am.enc.f0_pred.bias"), 1);
        auto f0_pool = avg_pool1d_rep(f0_raw, 1, L, 3, 1);

        const int out_dim = n_mel + d + 2; // 294
        x_out.resize(static_cast<size_t>(L) * out_dim);
        for (int t = 0; t < L; ++t) {
            float f0_hat = std::min(1.2f, std::max(-1.2f, f0_pool[t]));
            float phase = pos[t] / L_f;
            float p_story_arch = is_story * 0.42f *
                                 (0.60f * std::sin(phase * 6.2831853f) + 0.40f * std::cos(phase * 12.56637f)) *
                                 (1.0f - is_pause[t]);
            float p_prose = (0.35f * f0_hat + 0.65f * p_arch[t]) * sty_gain + p_story_arch;
            float p_out = has_svara * (0.88f * p_svara[t] + 0.35f * p_prose + pitch_shift) +
                          (1.0f - has_svara) * (p_prose + 0.50f * p_svara[t] + 0.35f * pitch_shift);

            float * row = x_out.data() + static_cast<size_t>(t) * out_dim;
            std::memcpy(row, mu.data() + static_cast<size_t>(t) * n_mel, n_mel * sizeof(float));
            std::memcpy(row + n_mel, g_vel.data(), d * sizeof(float));
            row[n_mel + d] = p_out;
            row[n_mel + d + 1] = sty_f0_sc;
        }
    }

    // ---------------- Stage 2B: Duration Rules + Repeat + DecExport (8-step CFM ODE + export_mel) ----------------
    std::vector<int32_t> compute_durations_and_adjust_x(
        const std::vector<std::string> & ph_valid,
        const std::vector<int64_t> & ids,
        const std::vector<float> & dm_arr,
        const std::vector<float> & logdur,
        std::vector<float> & x_enc,
        const std::string & lang,
        float rate,
        float pause_scale,
        bool is_last) const {
        const int L = static_cast<int>(ids.size());
        const int out_dim = 294;
        std::vector<int32_t> dur(L);
        for (int i = 0; i < L; ++i) {
            double d_val = std::max(std::exp(static_cast<double>(logdur[i])) - 1.0, 1.0);
            if (static_cast<size_t>(i) < dm_arr.size()) d_val *= static_cast<double>(dm_arr[i]);
            if (is_pause_id(ids[i])) d_val *= static_cast<double>(pause_scale);
            int32_t di = std::max(1, static_cast<int32_t>(std::nearbyint(d_val / static_cast<double>(rate))));
            dur[i] = di;
        }
        for (int idx_p = 0; idx_p < L; ++idx_p) {
            const std::string & p = ph_valid[idx_p];
            if (!is_punct_phone(p)) {
                dur[idx_p] = std::max<int32_t>(dur[idx_p], 3);
            }
            if (p == "h" && (idx_p + 1 == L || is_punct_phone(ph_valid[idx_p + 1]))) {
                dur[idx_p] = std::min<int32_t>(dur[idx_p], 6);
            }
            if (idx_p >= 1 && idx_p + 1 < L && p == "|" && ph_valid[idx_p - 1] == ph_valid[idx_p + 1] &&
                (ph_valid[idx_p - 1] == "a" || ph_valid[idx_p - 1] == "i" || ph_valid[idx_p - 1] == "u" ||
                 ph_valid[idx_p - 1] == "e" || ph_valid[idx_p - 1] == "o" || ph_valid[idx_p - 1] == "ʌ")) {
                if (idx_p >= 2 && ph_valid[idx_p - 2] == "|") {
                    dur[idx_p - 2] = 12;
                    for (int b = 0; b < 100; ++b) {
                        x_enc[(idx_p - 2) * out_dim + b] = x_enc[(L - 1) * out_dim + b];
                    }
                }
                dur[idx_p - 1] = std::max<int32_t>(dur[idx_p - 1], 19);
                dur[idx_p] = 7;
                dur[idx_p + 1] = std::max<int32_t>(dur[idx_p + 1], 18);
                for (int b = 0; b < 100; ++b) {
                    float v_next = x_enc[(idx_p + 1) * out_dim + b];
                    float v_sil = x_enc[(L - 1) * out_dim + b];
                    x_enc[(idx_p - 1) * out_dim + b] = v_next;
                    x_enc[idx_p * out_dim + b] = 0.45f * v_next + 0.55f * v_sil;
                }
            }
            if (lang == "sa" && p == "o" && idx_p + 1 < L && ph_valid[idx_p + 1] == "m" &&
                (idx_p == 0 || ph_valid[idx_p - 1] == "|" || ph_valid[idx_p - 1] == "," ||
                 ph_valid[idx_p - 1] == ";" || ph_valid[idx_p - 1] == ".")) {
                dur[idx_p] = std::min<int32_t>(44, std::max<int32_t>(34, dur[idx_p]));
                dur[idx_p + 1] = std::min<int32_t>(22, std::max<int32_t>(16, dur[idx_p + 1]));
                if (idx_p + 2 < L && ph_valid[idx_p + 2] == ",") {
                    dur[idx_p + 2] = std::min<int32_t>(22, std::max<int32_t>(14, dur[idx_p + 2]));
                }
            }
            if (idx_p >= 1 && idx_p + 2 < L &&
                (ph_valid[idx_p - 1] == "|" || ph_valid[idx_p - 1] == ",") &&
                (ph_valid[idx_p + 2] == "|" || ph_valid[idx_p + 2] == ",") &&
                (p == "r" || p == "t̪" || p == "n" || p == "m" || p == "h" || p == "tsʰ") &&
                (ph_valid[idx_p + 1] == "ʌ" || ph_valid[idx_p + 1] == "o")) {
                dur[idx_p] = std::min<int32_t>(9, std::max<int32_t>(5, dur[idx_p]));
                dur[idx_p + 1] = std::min<int32_t>(11, std::max<int32_t>(6, dur[idx_p + 1]));
            }
        }
        if (!is_last && L > 0) {
            dur[L - 1] = std::max<int32_t>(1, static_cast<int32_t>(std::nearbyint(7.0 * pause_scale)));
        }
        return dur;
    }

    // Velocity v_theta(xt, t | mu_y, g): xt [100, T], mu_y [100, T], g [192] -> [100, T]
    std::vector<float> run_velocity(const std::vector<float> & xt, float t_val,
                                    const std::vector<float> & mu_y, const std::vector<float> & g, int T) const {
        const int C = 100;
        const int d_vel = 256;
        const int c_dim = 192;

        // c = tm(temb(t, 192)) + g
        auto te = compute_temb(t_val, c_dim);
        auto tm0 = linear_2d(te, 1, c_dim, W("am.vel.tm.0.weight"), W("am.vel.tm.0.bias"), c_dim);
        for (float & v : tm0) v = silu(v);
        auto c_vec = linear_2d(tm0, 1, c_dim, W("am.vel.tm.2.weight"), W("am.vel.tm.2.bias"), c_dim);
        for (int i = 0; i < c_dim; ++i) c_vec[i] += g[i];

        // cat([xt, mu_y], dim=1) -> [200, T]
        std::vector<float> cat_in(static_cast<size_t>(2 * C) * T);
        std::memcpy(cat_in.data(), xt.data(), static_cast<size_t>(C) * T * sizeof(float));
        std::memcpy(cat_in.data() + static_cast<size_t>(C) * T, mu_y.data(), static_cast<size_t>(C) * T * sizeof(float));

        auto h = conv1d(cat_in, 2 * C, T, W("am.vel.inp.weight"), W("am.vel.inp.bias"), d_vel, 1, 1, 0);

        for (int b = 0; b < 12; ++b) {
            std::string bpfx = "am.vel.blocks." + std::to_string(b) + ".";
            auto film = linear_2d(c_vec, 1, c_dim, W(bpfx + "film.weight"), W(bpfx + "film.bias"), 2 * d_vel);
            const float * sc = film.data();
            const float * sh = film.data() + d_vel;

            auto y = conv1d(h, d_vel, T, W(bpfx + "dw.weight"), W(bpfx + "dw.bias"), d_vel, 7, 1, 3, 1, d_vel);

            // GroupNorm(1, d_vel) across all d_vel * T elements
            size_t tot = static_cast<size_t>(d_vel) * T;
            double sum = 0.0;
            for (size_t i = 0; i < tot; ++i) sum += y[i];
            float mean = static_cast<float>(sum / tot);
            double sq = 0.0;
            for (size_t i = 0; i < tot; ++i) {
                float diff = y[i] - mean;
                sq += static_cast<double>(diff) * diff;
            }
            float inv_std = 1.0f / std::sqrt(static_cast<float>(sq / tot) + 1e-5f);

            const auto & nw = W(bpfx + "n.weight");
            const auto & nb = W(bpfx + "n.bias");
#pragma omp parallel for schedule(static) if (d_vel * T >= 4096)
            for (int ch = 0; ch < d_vel; ++ch) {
                float scale = nw[ch] * inv_std * (1.0f + sc[ch]);
                float shift = nb[ch] * (1.0f + sc[ch]) + sh[ch];
                float * y_ch = y.data() + static_cast<size_t>(ch) * T;
                for (int t = 0; t < T; ++t) {
                    y_ch[t] = (y_ch[t] - mean) * scale + shift;
                }
            }

            auto p1 = conv1d(y, d_vel, T, W(bpfx + "p1.weight"), W(bpfx + "p1.bias"), d_vel * 3, 1, 1, 0);
#pragma omp parallel for schedule(static) if (p1.size() >= 4096)
            for (size_t i = 0; i < p1.size(); ++i) p1[i] = gelu_exact(p1[i]);
            auto p2 = conv1d(p1, d_vel * 3, T, W(bpfx + "p2.weight"), W(bpfx + "p2.bias"), d_vel, 1, 1, 0);
            for (size_t i = 0; i < h.size(); ++i) h[i] += p2[i];
        }

        return conv1d(h, d_vel, T, W("am.vel.out.weight"), W("am.vel.out.bias"), C, 1, 1, 0);
    }

    // DecExport: xe [T, 294] -> mel [100, T]
    std::vector<float> run_decoder(const std::vector<float> & xe, int T, int steps = 8, float temp = 0.667f) const {
        const int C = 100;
        const int c_dim = 192;
        const int in_dim = 294;

        std::vector<float> mu_y(static_cast<size_t>(C) * T);
        std::vector<float> g(c_dim);
        std::vector<float> p_xe(T);
        std::vector<float> sty_gain(T);

        for (int j = 0; j < c_dim; ++j) g[j] = xe[C + j];
        for (int t = 0; t < T; ++t) {
            const float * row = xe.data() + static_cast<size_t>(t) * in_dim;
            for (int c = 0; c < C; ++c) mu_y[c * T + t] = row[c];
            p_xe[t] = row[C + c_dim];
            sty_gain[t] = row[C + c_dim + 1];
        }

        std::vector<float> xt(static_cast<size_t>(C) * T);
        if (temp == 0.0f && steps == 1) {
            auto v0 = run_velocity(mu_y, 0.0f, mu_y, g, T);
            for (size_t i = 0; i < xt.size(); ++i) xt[i] = mu_y[i] + v0[i];
        } else {
            const float scale_z = static_cast<float>(static_cast<double>(temp) * 1.4142);
            for (int c = 0; c < C; ++c) {
                volatile float fc = static_cast<float>(c);
                volatile float c1_val = fc * 12.9898f;
                volatile float c2_val = fc * 39.346f;
                float * xtc = xt.data() + static_cast<size_t>(c) * T;
                for (int t = 0; t < T; ++t) {
                    volatile float ft = static_cast<float>(t);
                    volatile float t1_val = ft * 78.233f;
                    volatile float t2_val = ft * 11.135f;
                    float a1 = c1_val + t1_val;
                    float a2 = c2_val + t2_val;
                    xtc[t] = (std::sin(a1) * std::cos(a2)) * scale_z;
                }
            }
            float dt = 1.0f / static_cast<float>(steps);
            std::vector<float> xt_mid(xt.size());
            for (int s = 0; s < steps; ++s) {
                float t_val = static_cast<float>(s) * dt;
                auto k1 = run_velocity(xt, t_val, mu_y, g, T);
                for (size_t i = 0; i < xt.size(); ++i) xt_mid[i] = xt[i] + 0.5f * dt * k1[i];
                auto k2 = run_velocity(xt_mid, t_val + 0.5f * dt, mu_y, g, T);
                for (size_t i = 0; i < xt.size(); ++i) xt[i] += dt * k2[i];
            }
        }

        const auto & stats = W("am.stats");
        float mel_mean = stats[0];
        float mel_std = stats[1];
        std::vector<float> mel(xt.size());
        for (size_t i = 0; i < xt.size(); ++i) mel[i] = xt[i] * mel_std + mel_mean;

        // p_zero = p_xe - mean(p_xe)
        double p_sum = 0.0;
        for (int t = 0; t < T; ++t) p_sum += p_xe[t];
        float p_mean = static_cast<float>(p_sum / T);
        std::vector<float> p_zero(T);
        for (int t = 0; t < T; ++t) p_zero[t] = p_xe[t] - p_mean;

        auto p_s1 = avg_pool1d_rep(p_zero, 1, T, 9, 4);
        auto p_sm = avg_pool1d_rep(p_s1, 1, T, 5, 2);

        // env = avg_pool1d across the 100 mel bins (mel.transpose(1,2), pad=(3,3) replicate, k=7)
        std::vector<float> mel_w(mel.size());
        float comb_gain = (steps == 1) ? 0.12f : 0.08f;
        for (int kb = 0; kb < C; ++kb) {
            float wk = std::min(1.0f, std::max(0.0f, (64.0f - static_cast<float>(kb)) / 18.0f));
            float gain = 1.0f + comb_gain * wk;
            for (int t = 0; t < T; ++t) {
                float env_sum = 0.0f;
                for (int dk = -3; dk <= 3; ++dk) {
                    int k_clamped = std::min(C - 1, std::max(0, kb + dk));
                    env_sum += mel[k_clamped * T + t];
                }
                float env_val = env_sum / 7.0f;
                float m_val = mel[kb * T + t];
                mel_w[kb * T + t] = env_val + (m_val - env_val) * gain;
            }
        }

        // Side channels bins 98 and 99
        for (int t = 0; t < T; ++t) {
            float st_shift = std::min(1.80f, std::max(-1.80f, p_sm[t] * 1.45f));
            float m97 = mel_w[97 * T + t];
            mel_w[98 * T + t] = m97 + 0.01f * sty_gain[t];
            mel_w[99 * T + t] = m97 + 0.01f * st_shift;
        }
        return mel_w;
    }

    // ---------------- Stage 3: VocSpecExport (VocoderV2) + _ola64 + _declick_np ----------------
    void run_vocoder_spectrum(const std::vector<float> & mel, int T,
                              std::vector<float> & re_out, std::vector<float> & im_out) const {
        const int n_mel = 100;
        const int ch = 320;
        const int c1 = 160;
        const int c2 = 80;

        // Extract st_shift from bin 99 - bin 97, and build mel_clean (bins 98/99 replaced with bin 97)
        std::vector<float> st_shift_scaled(T);
        std::vector<float> mel_clean = mel;
        for (int t = 0; t < T; ++t) {
            float m97 = mel[97 * T + t];
            float m99 = mel[99 * T + t];
            float st = std::min(1.80f, std::max(-1.80f, (m99 - m97) * 100.0f));
            st_shift_scaled[t] = std::min(0.85f, std::max(-0.85f, 0.35f * st));
            mel_clean[98 * T + t] = m97;
            mel_clean[99 * T + t] = m97;
        }
        auto tot_shift = avg_pool1d_rep(st_shift_scaled, 1, T, 5, 2);

        // HarmonicSource: 3-layer f0_net
        auto h0 = conv1d(mel_clean, n_mel, T, W("voc.src.f0_net.0.weight"), W("voc.src.f0_net.0.bias"), 128, 5, 1, 2);
        for (float & v : h0) v = silu(v);
        auto h1 = conv1d(h0, 128, T, W("voc.src.f0_net.2.weight"), W("voc.src.f0_net.2.bias"), 128, 5, 1, 2);
        for (float & v : h1) v = silu(v);
        auto rf = conv1d(h1, 128, T, W("voc.src.f0_net.4.weight"), W("voc.src.f0_net.4.bias"), 2, 3, 1, 1);

        std::vector<float> f0(T), uv(T), w_v(T), lf0_raw(T), lf0_wv(T);
        for (int t = 0; t < T; ++t) {
            f0[t] = 60.0f + 440.0f * sigmoid_f(rf[0 * T + t]);
            uv[t] = sigmoid_f(rf[1 * T + t]);
            w_v[t] = (uv[t] > 0.35f ? 1.0f : 0.0f) * uv[t] + 0.04f;
            lf0_raw[t] = std::log2(std::min(380.0f, std::max(130.0f, f0[t])));
            lf0_wv[t] = lf0_raw[t] * w_v[t];
        }
        if (T >= 7) {
            auto num = avg_pool1d_rep(lf0_wv, 1, T, 7, 3);
            auto den = avg_pool1d_rep(w_v, 1, T, 7, 3);
            std::vector<float> lf0_mix(T);
            for (int t = 0; t < T; ++t) {
                float lf0_loc = num[t] / std::max(den[t], 1e-3f);
                lf0_mix[t] = 0.50f * lf0_raw[t] + 0.50f * lf0_loc;
            }
            auto lf0_sm = avg_pool1d_rep(lf0_mix, 1, T, 5, 2);
            for (int t = 0; t < T; ++t) f0[t] = std::pow(2.0f, lf0_sm[t]);
            uv = avg_pool1d_rep(uv, 1, T, 3, 1);
        }
        for (int t = 0; t < T; ++t) {
            f0[t] *= std::pow(2.0f, tot_shift[t] / 12.0f);
        }

        // 16x linear interpolation (align_corners=False)
        const int T4 = T * 4;
        const int T16 = T * 16;
        std::vector<float> f0_16(T16), uv_16(T16);
        for (int i = 0; i < T16; ++i) {
            float src_x = (static_cast<float>(i) + 0.5f) / 16.0f - 0.5f;
            if (src_x <= 0.0f) {
                f0_16[i] = f0[0];
                uv_16[i] = uv[0];
            } else if (src_x >= static_cast<float>(T - 1)) {
                f0_16[i] = f0[T - 1];
                uv_16[i] = uv[T - 1];
            } else {
                int i0 = static_cast<int>(std::floor(src_x));
                float w = src_x - static_cast<float>(i0);
                f0_16[i] = (1.0f - w) * f0[i0] + w * f0[i0 + 1];
                uv_16[i] = (1.0f - w) * uv[i0] + w * uv[i0 + 1];
            }
        }

        // Cumulative cycles cyc in [0, 1) + 8 harmonics + analytic unvoiced noise
        const float inv_fps16 = 1.0f / (24000.0f / 16.0f); // 1/1500
        const float two_pi = 2.0f * std::acos(-1.0f);
        static const float ap_freqs[8] = {
            1.41421f, 1.73205f, 2.23607f, 2.64575f, 3.14159f, 3.60555f, 4.12311f, 4.58258f
        };
        std::vector<float> har(static_cast<size_t>(16) * T16);
        double cyc_acc = 0.0;
        std::vector<float> ph_arr(T16);
        for (int t = 0; t < T16; ++t) {
            float step_cyc = f0_16[t] * inv_fps16;
            cyc_acc += static_cast<double>(step_cyc);
            float cyc_f = static_cast<float>(cyc_acc);
            float cyc_wrapped = cyc_f - std::floor(cyc_f);
            ph_arr[t] = two_pi * cyc_wrapped;
        }

        for (int k = 0; k < 8; ++k) {
            float fk = static_cast<float>(k + 1);
            float inv_sqrt_k = 1.0f / std::sqrt(fk);
            float ap_f = ap_freqs[k];
            float ap_f2 = ap_f * 1.17f;
            float * har_sin = har.data() + static_cast<size_t>(k) * T16;
            float * har_cos = har.data() + static_cast<size_t>(k + 8) * T16;
            for (int t = 0; t < T16; ++t) {
                float amp = (fk * f0_16[t] < 720.0f) ? inv_sqrt_k : 0.0f;
                float k_ph = fk * ph_arr[t];
                float uvt = uv_16[t];
                float n_scale = (1.0f - uvt) * 0.14f;
                float ft = static_cast<float>(t);
                har_sin[t] = std::sin(k_ph) * amp * uvt + n_scale * std::sin(ft * ap_f);
                har_cos[t] = std::cos(k_ph) * amp * uvt + n_scale * std::cos(ft * ap_f2);
            }
        }

        auto s4 = conv1d(har, 16, T16, W("voc.src.proj4.weight"), W("voc.src.proj4.bias"), c1, 4, 4, 0);
        auto s16 = conv1d(har, 16, T16, W("voc.src.proj16.weight"), W("voc.src.proj16.bias"), c2, 3, 1, 1);

        // VocoderV2 main stack
        auto x = conv1d(mel_clean, n_mel, T, W("voc.pre.weight"), W("voc.pre.bias"), ch, 7, 1, 3);

        const int res_kernels[3] = {3, 7, 11};
        const int dilations[3] = {1, 3, 5};

        // Stage 0: 320 -> 160, T -> T4
        snake1d_inplace(x, ch, T, W("voc.up_acts.0.alpha"));
        x = conv_transpose1d(x, ch, T, W("voc.ups.0.weight"), W("voc.ups.0.bias"), c1, 8, 4, 2);
        for (size_t i = 0; i < x.size(); ++i) x[i] += s4[i];
        {
            std::vector<float> mrf_sum(x.size(), 0.0f);
            for (int kb_i = 0; kb_i < 3; ++kb_i) {
                int rk = res_kernels[kb_i];
                std::vector<float> xb = x;
                for (int di = 0; di < 3; ++di) {
                    int d_val = dilations[di];
                    std::string pfx = "voc.mrfs.0." + std::to_string(kb_i) + ".";
                    std::vector<float> y = xb;
                    snake1d_inplace(y, c1, T4, W(pfx + "a1." + std::to_string(di) + ".alpha"));
                    y = conv1d(y, c1, T4, W(pfx + "c1." + std::to_string(di) + ".weight"),
                               W(pfx + "c1." + std::to_string(di) + ".bias"),
                               c1, rk, 1, (rk * d_val - d_val) / 2, d_val);
                    snake1d_inplace(y, c1, T4, W(pfx + "a2." + std::to_string(di) + ".alpha"));
                    y = conv1d(y, c1, T4, W(pfx + "c2." + std::to_string(di) + ".weight"),
                               W(pfx + "c2." + std::to_string(di) + ".bias"),
                               c1, rk, 1, (rk - 1) / 2, 1);
                    for (size_t i = 0; i < xb.size(); ++i) xb[i] += y[i];
                }
                for (size_t i = 0; i < mrf_sum.size(); ++i) mrf_sum[i] += xb[i];
            }
            for (size_t i = 0; i < x.size(); ++i) x[i] = mrf_sum[i] / 3.0f;
        }

        // Stage 1: 160 -> 80, T4 -> T16
        snake1d_inplace(x, c1, T4, W("voc.up_acts.1.alpha"));
        x = conv_transpose1d(x, c1, T4, W("voc.ups.1.weight"), W("voc.ups.1.bias"), c2, 8, 4, 2);
        for (size_t i = 0; i < x.size(); ++i) x[i] += s16[i];
        {
            std::vector<float> mrf_sum(x.size(), 0.0f);
            for (int kb_i = 0; kb_i < 3; ++kb_i) {
                int rk = res_kernels[kb_i];
                std::vector<float> xb = x;
                for (int di = 0; di < 3; ++di) {
                    int d_val = dilations[di];
                    std::string pfx = "voc.mrfs.1." + std::to_string(kb_i) + ".";
                    std::vector<float> y = xb;
                    snake1d_inplace(y, c2, T16, W(pfx + "a1." + std::to_string(di) + ".alpha"));
                    y = conv1d(y, c2, T16, W(pfx + "c1." + std::to_string(di) + ".weight"),
                               W(pfx + "c1." + std::to_string(di) + ".bias"),
                               c2, rk, 1, (rk * d_val - d_val) / 2, d_val);
                    snake1d_inplace(y, c2, T16, W(pfx + "a2." + std::to_string(di) + ".alpha"));
                    y = conv1d(y, c2, T16, W(pfx + "c2." + std::to_string(di) + ".weight"),
                               W(pfx + "c2." + std::to_string(di) + ".bias"),
                               c2, rk, 1, (rk - 1) / 2, 1);
                    for (size_t i = 0; i < xb.size(); ++i) xb[i] += y[i];
                }
                for (size_t i = 0; i < mrf_sum.size(); ++i) mrf_sum[i] += xb[i];
            }
            for (size_t i = 0; i < x.size(); ++i) x[i] = mrf_sum[i] / 3.0f;
        }

        // Post-conv -> 66 channels (33 log_mag + 33 phase)
        snake1d_inplace(x, c2, T16, W("voc.post_act.alpha"));
        auto h_post = conv1d(x, c2, T16, W("voc.post.weight"), W("voc.post.bias"), 66, 7, 1, 3);

        re_out.resize(static_cast<size_t>(33) * T16);
        im_out.resize(static_cast<size_t>(33) * T16);
#pragma omp parallel for schedule(static)
        for (int k = 0; k < 33; ++k) {
            const float * lm_row = h_post.data() + static_cast<size_t>(k) * T16;
            const float * ph_row = h_post.data() + static_cast<size_t>(k + 33) * T16;
            float * re_row = re_out.data() + static_cast<size_t>(k) * T16;
            float * im_row = im_out.data() + static_cast<size_t>(k) * T16;
            for (int t = 0; t < T16; ++t) {
                float lm = std::min(5.0f, std::max(-11.0f, lm_row[t]));
                float mag = std::exp(lm);
                float ph_v = ph_row[t];
                re_row[t] = mag * std::cos(ph_v);
                im_row[t] = mag * std::sin(ph_v);
            }
        }
    }

    // 64-pt iSTFT overlap-add (_ola64 in sakhiya/onnx_tts.py)
    std::vector<float> ola64(const std::vector<float> & re, const std::vector<float> & im, int T16) const {
        const int n_fft = 64;
        const int hop = 16;
        const auto & win = W("voc.win"); // 64-pt Hann window

        // Precompute 64-pt iRFFT basis: cosb[k, n], sinb[k, n]
        static float cosb[33][64];
        static float sinb[33][64];
        static bool basis_init = false;
        if (!basis_init) {
            const double two_pi = 2.0 * std::acos(-1.0);
            for (int k = 0; k <= 32; ++k) {
                double c_scale = (k == 0 || k == 32) ? (1.0 / 64.0) : (2.0 / 64.0);
                for (int n = 0; n < 64; ++n) {
                    double ang = two_pi * k * n / 64.0;
                    cosb[k][n] = static_cast<float>(c_scale * std::cos(ang));
                    sinb[k][n] = static_cast<float>(c_scale * std::sin(ang));
                }
            }
            basis_init = true;
        }

        int L = (T16 - 1) * hop + n_fft;
        std::vector<float> y(L, 0.0f);
        std::vector<float> env(L, 0.0f);
        float w2[64];
        for (int n = 0; n < 64; ++n) w2[n] = win[n] * win[n];

        for (int t = 0; t < T16; ++t) {
            float re_t[33], im_t[33];
            for (int k = 0; k <= 32; ++k) {
                re_t[k] = re[static_cast<size_t>(k) * T16 + t];
                im_t[k] = (k == 0 || k == 32) ? 0.0f : im[static_cast<size_t>(k) * T16 + t];
            }
            int base = t * hop;
            for (int n = 0; n < 64; ++n) {
                float acc = 0.0f;
                for (int k = 0; k <= 32; ++k) {
                    acc += re_t[k] * cosb[k][n] - im_t[k] * sinb[k][n];
                }
                y[base + n] += acc * win[n];
                env[base + n] += w2[n];
            }
        }

        int p = (n_fft - hop) / 2; // 24
        int out_len = L - 2 * p;
        std::vector<float> out(out_len);
        float pk = 0.0f;
        for (int i = 0; i < out_len; ++i) {
            float v = y[i + p] / std::max(env[i + p], 1e-8f);
            out[i] = v;
            float av = std::fabs(v);
            if (av > pk) pk = av;
        }
        if (pk > 1e-4f) {
            float sc = 0.90f / pk;
            for (int i = 0; i < out_len; ++i) {
                out[i] = std::min(1.0f, std::max(-1.0f, out[i] * sc));
            }
        }
        return out;
    }

    // Radix-2 complex FFT for _declick_np (n_fft=512)
    static void fft_radix2(std::vector<std::complex<float>> & a, bool invert) {
        int n = static_cast<int>(a.size());
        for (int i = 1, j = 0; i < n; ++i) {
            int bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) std::swap(a[i], a[j]);
        }
        const float pi = std::acos(-1.0f);
        for (int len = 2; len <= n; len <<= 1) {
            float ang = 2.0f * pi / static_cast<float>(len) * (invert ? 1.0f : -1.0f);
            std::complex<float> wlen(std::cos(ang), std::sin(ang));
            for (int i = 0; i < n; i += len) {
                std::complex<float> w(1.0f, 0.0f);
                for (int j = 0; j < len / 2; ++j) {
                    std::complex<float> u = a[i + j];
                    std::complex<float> v = a[i + j + len / 2] * w;
                    a[i + j] = u + v;
                    a[i + j + len / 2] = u - v;
                    w *= wlen;
                }
            }
        }
        if (invert) {
            float inv_n = 1.0f / static_cast<float>(n);
            for (auto & x : a) x *= inv_n;
        }
    }

    // Localized STFT transient smoother (_declick_np in sakhiya/onnx_tts.py)
    static std::vector<float> declick_np(std::vector<float> w, int n_fft = 512, int hop = 256) {
        if (static_cast<int>(w.size()) < n_fft * 2) return w;
        int pad = n_fft / 2;
        int n_bins = n_fft / 2 + 1;
        const float pi = std::acos(-1.0f);
        std::vector<float> win(n_fft), w2(n_fft);
        for (int i = 0; i < n_fft; ++i) {
            win[i] = 0.5f - 0.5f * std::cos(2.0f * pi * static_cast<float>(i) / static_cast<float>(n_fft));
            w2[i] = win[i] * win[i];
        }

        for (int iter = 0; iter < 3; ++iter) {
            int orig_len = static_cast<int>(w.size());
            int wp_len = orig_len + 2 * pad;
            std::vector<float> wp(wp_len);
            for (int i = 0; i < wp_len; ++i) {
                int idx = i - pad;
                if (idx < 0) idx = -idx;
                else if (idx >= orig_len) idx = 2 * orig_len - 2 - idx;
                wp[i] = w[std::min(orig_len - 1, std::max(0, idx))];
            }
            int n_frames = 1 + (wp_len - n_fft) / hop;
            // S [n_bins, n_frames]
            std::vector<std::complex<float>> S(static_cast<size_t>(n_bins) * n_frames);
            std::vector<float> log_spec(static_cast<size_t>(n_bins) * n_frames);
            std::vector<std::complex<float>> buf(n_fft);
            for (int t = 0; t < n_frames; ++t) {
                for (int n = 0; n < n_fft; ++n) {
                    buf[n] = std::complex<float>(wp[t * hop + n] * win[n], 0.0f);
                }
                fft_radix2(buf, false);
                for (int k = 0; k < n_bins; ++k) {
                    S[k * n_frames + t] = buf[k];
                    log_spec[k * n_frames + t] = std::log(std::abs(buf[k]) + 1e-4f);
                }
            }
            int n_diff = n_frames - 1;
            if (n_diff <= 0) break;
            std::vector<float> flux(n_diff);
            for (int t = 0; t < n_diff; ++t) {
                double sq_sum = 0.0;
                for (int k = 0; k < n_bins; ++k) {
                    float d = std::max(0.0f, log_spec[k * n_frames + (t + 1)] - log_spec[k * n_frames + t]);
                    sq_sum += static_cast<double>(d) * d;
                }
                flux[t] = std::sqrt(static_cast<float>(sq_sum / n_bins));
            }
            std::vector<float> flux_sorted = flux;
            std::sort(flux_sorted.begin(), flux_sorted.end());
            float med = (n_diff % 2 == 1)
                            ? flux_sorted[n_diff / 2]
                            : 0.5f * (flux_sorted[n_diff / 2 - 1] + flux_sorted[n_diff / 2]);
            double mean_f = 0.0;
            for (float v : flux) mean_f += v;
            mean_f /= n_diff;
            double var_f = 0.0;
            for (float v : flux) {
                double d = v - mean_f;
                var_f += d * d;
            }
            float std_f = std::sqrt(static_cast<float>(var_f / n_diff));

            std::vector<int> bad;
            for (int t = 0; t < n_diff; ++t) {
                float fz = (flux[t] - med) / (std_f + 1e-5f);
                if (fz > 7.5f) bad.push_back(t);
            }
            if (bad.empty()) break;

            for (int t : bad) {
                for (int k = 0; k < n_bins; ++k) {
                    S[k * n_frames + t] = 0.65f * S[k * n_frames + t] + 0.35f * S[k * n_frames + (t + 1)];
                }
            }
            int L = (n_frames - 1) * hop + n_fft;
            std::vector<float> y(L, 0.0f), env(L, 0.0f);
            for (int t = 0; t < n_frames; ++t) {
                buf[0] = std::complex<float>(S[0 * n_frames + t].real(), 0.0f);
                buf[n_fft / 2] = std::complex<float>(S[(n_fft / 2) * n_frames + t].real(), 0.0f);
                for (int k = 1; k < n_fft / 2; ++k) {
                    auto v = S[k * n_frames + t];
                    buf[k] = v;
                    buf[n_fft - k] = std::conj(v);
                }
                fft_radix2(buf, true);
                for (int n = 0; n < n_fft; ++n) {
                    y[t * hop + n] += buf[n].real() * win[n];
                    env[t * hop + n] += w2[n];
                }
            }
            for (int i = 0; i < orig_len; ++i) {
                w[i] = y[pad + i] / std::max(env[pad + i], 1e-8f);
            }
        }
        return w;
    }

    // Equal-power crossfade stitch (sakhiya/onnx_tts.py stitch)
    static std::vector<float> stitch_wavs(const std::vector<std::vector<float>> & wavs, int sr = 24000, int xf_ms = 25, int edge_ms = 90) {
        std::vector<std::vector<float>> clips;
        for (const auto & w : wavs) {
            if (w.empty()) continue;
            float mx = 0.0f;
            for (float v : w) mx = std::max(mx, std::fabs(v));
            float thr = std::max(mx * 0.006f, 1e-4f);
            int first = -1, last = -1;
            for (int i = 0; i < static_cast<int>(w.size()); ++i) {
                if (std::fabs(w[i]) > thr) {
                    if (first < 0) first = i;
                    last = i;
                }
            }
            if (first >= 0) {
                int keep = edge_ms * sr / 1000;
                int a = std::max(0, first - keep);
                int b = std::min(static_cast<int>(w.size()), last + 1 + keep);
                clips.emplace_back(w.begin() + a, w.begin() + b);
            } else {
                clips.push_back(w);
            }
        }
        if (clips.empty()) return {};
        int n = xf_ms * sr / 1000;
        std::vector<float> out = clips[0];
        const float half_pi = 0.5f * std::acos(-1.0f);
        for (size_t c = 1; c < clips.size(); ++c) {
            const auto & w = clips[c];
            int m = std::min({n, static_cast<int>(out.size()), static_cast<int>(w.size())});
            if (m < 8) {
                out.insert(out.end(), w.begin(), w.end());
                continue;
            }
            size_t base = out.size() - m;
            for (int i = 0; i < m; ++i) {
                float t = half_pi * static_cast<float>(i) / static_cast<float>(m - 1);
                out[base + i] = out[base + i] * std::cos(t) + w[i] * std::sin(t);
            }
            out.insert(out.end(), w.begin() + m, w.end());
        }
        return declick_np(std::move(out));
    }

    // Synthesize a single chunk (`_say_one` in sakhiya/onnx_tts.py)
    std::vector<float> say_one(const std::string & text, const std::string & lang, const StyleConfig & k,
                               float svara, bool is_last, StageDump * dump = nullptr) const {
        std::vector<std::string> ph;
        std::vector<double> off;
        std::vector<double> dm;
        bool has_vm = false;

        if (lang == "sa") {
            auto ch = compile_chandas(text, 75.0, 215.0, true);
            ph = std::move(ch.ph);
            off = std::move(ch.off);
            dm = std::move(ch.dm);
            has_vm = ch.has_vedic_marks;
        } else {
            auto acc = text_to_phones_acc(text, lang, true);
            ph = std::move(acc.first);
            off = std::move(acc.second);
            dm = phone_dm(ph);
        }

        const auto & sym2id = get_sym2id();
        std::vector<int64_t> ids;
        std::vector<float> dm_arr;
        std::vector<float> pitch_add;
        std::vector<std::string> ph_valid;
        float sv_sc = (lang == "sa") ? (has_vm ? 0.52f : 0.48f) : 1.15f;

        for (size_t i = 0; i < ph.size(); ++i) {
            auto it = sym2id.find(ph[i]);
            if (it != sym2id.end() && it->second >= 0) {
                ids.push_back(it->second);
                dm_arr.push_back(static_cast<float>(dm[i]));
                pitch_add.push_back(static_cast<float>(off[i]) * sv_sc * svara);
                ph_valid.push_back(ph[i]);
            }
        }

        std::vector<float> x_enc, logdur;
        run_encoder(ids, k.style_id, k.pitch_shift, k.pitch_scale, k.energy_scale, pitch_add, x_enc, logdur);

        auto dur = compute_durations_and_adjust_x(ph_valid, ids, dm_arr, logdur, x_enc, lang, k.rate, k.pause_scale, is_last);

        int T_mel = 0;
        for (int32_t d_i : dur) T_mel += d_i;

        const int out_dim = 294;
        std::vector<float> xe(static_cast<size_t>(T_mel) * out_dim);
        int f_idx = 0;
        for (size_t i = 0; i < dur.size(); ++i) {
            const float * src_row = x_enc.data() + i * out_dim;
            for (int r = 0; r < dur[i]; ++r) {
                std::memcpy(xe.data() + static_cast<size_t>(f_idx++) * out_dim, src_row, out_dim * sizeof(float));
            }
        }

        auto mel = run_decoder(xe, T_mel, 8, 0.667f);

        std::vector<float> re_spec, im_spec;
        run_vocoder_spectrum(mel, T_mel, re_spec, im_spec);

        int T_spec = T_mel * 16;
        auto w = ola64(re_spec, im_spec, T_spec);

        if (w.size() > 256) {
            const float pi = std::acos(-1.0f);
            for (int i = 0; i < 128; ++i) {
                float ramp = 0.5f - 0.5f * std::cos(pi * static_cast<float>(i) / 128.0f);
                w[i] *= ramp;
                w[w.size() - 1 - i] *= ramp;
            }
            int pad_n = static_cast<int>(0.055 * 24000); // 1320
            std::vector<float> w_pad(pad_n + w.size() + pad_n / 2, 0.0f);
            std::memcpy(w_pad.data() + pad_n, w.data(), w.size() * sizeof(float));
            w.swap(w_pad);
        }

        if (dump) {
            dump->ids = ids;
            dump->pitch_add = pitch_add;
            dump->dm = dm_arr;
            dump->x_enc = x_enc;
            dump->logdur = logdur;
            dump->dur = dur;
            dump->T_mel = T_mel;
            dump->mel = mel;
            dump->T_spec = T_spec;
            dump->spec_re = re_spec;
            dump->spec_im = im_spec;
        }
        return w;
    }

    // Top-level say() matching SakhiyaONNX.say() in sakhiya/onnx_tts.py
    std::vector<float> say(const std::string & text, const std::string & lang = "ne",
                           const std::string & style = "neutral",
                           float svara = 1.0f,
                           const StyleConfig * override_ctl = nullptr,
                           StageDump * dump = nullptr) const {
        if (lang == "auto") {
            auto routes = route_mixed(text, "thr_dangaura");
            std::vector<std::vector<float>> wavs;
            for (const auto & kv : routes) {
                wavs.push_back(say(kv.second, kv.first, style, svara, override_ctl, nullptr));
            }
            return stitch_wavs(wavs, 24000);
        }

        StyleConfig k = override_ctl ? *override_ctl : get_style_config(style);
        if (lang == "sa" && style != "chant" && style != "vedic_chant" && style != "storytelling" && !override_ctl) {
            k.rate *= 0.96f;
            k.pause_scale *= 1.05f;
            k.pitch_scale = 1.00f;
        }
        if (str_starts_with(lang, "thr") && !override_ctl) {
            if (lang == "thr_rana" || lang == "thr_chitwania") {
                k.rate *= 0.90f; k.pitch_scale = 1.00f; k.pause_scale *= 1.05f;
            } else {
                k.rate *= 0.90f; k.pitch_scale = 1.00f; k.pause_scale *= 1.00f;
            }
        }

        auto sa_segs = segment_text(text);
        std::istringstream iss(trim_str(text));
        size_t word_cnt = 0;
        std::string tmp_w;
        while (iss >> tmp_w) ++word_cnt;

        std::vector<std::string> chunks;
        if (dump || (lang == "sa" && word_cnt <= 45 && sa_segs.size() <= 2)) {
            chunks = {trim_str(text)};
        } else {
            chunks = sa_segs;
        }

        if (chunks.size() == 1) {
            auto w = declick_np(say_one(chunks[0], lang, k, svara, true, dump));
            if (dump) dump->wav = w;
            return w;
        }

        std::vector<float> concat_wav;
        for (size_t i = 0; i < chunks.size(); ++i) {
            auto wi = say_one(chunks[i], lang, k, svara, i + 1 == chunks.size(), (i == 0) ? dump : nullptr);
            concat_wav.insert(concat_wav.end(), wi.begin(), wi.end());
        }
        auto w = declick_np(std::move(concat_wav));
        if (dump) dump->wav = w;
        return w;
    }
};

inline void write_wav_pcm16(const std::string & path, const std::vector<float> & samples, int sr = 24000) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open WAV for writing: " + path);
    uint32_t n_samples = static_cast<uint32_t>(samples.size());
    uint32_t data_bytes = n_samples * 2;
    uint32_t riff_size = 36 + data_bytes;
    uint16_t audio_fmt = 1; // PCM
    uint16_t channels = 1;
    uint32_t byte_rate = sr * 2;
    uint16_t block_align = 2;
    uint16_t bits_per_sample = 16;
    uint32_t fmt_size = 16;

    f.write("RIFF", 4);
    f.write(reinterpret_cast<const char *>(&riff_size), 4);
    f.write("WAVEfmt ", 8);
    f.write(reinterpret_cast<const char *>(&fmt_size), 4);
    f.write(reinterpret_cast<const char *>(&audio_fmt), 2);
    f.write(reinterpret_cast<const char *>(&channels), 2);
    f.write(reinterpret_cast<const char *>(&sr), 4);
    f.write(reinterpret_cast<const char *>(&byte_rate), 4);
    f.write(reinterpret_cast<const char *>(&block_align), 2);
    f.write(reinterpret_cast<const char *>(&bits_per_sample), 2);
    f.write("data", 4);
    f.write(reinterpret_cast<const char *>(&data_bytes), 4);

    std::vector<int16_t> pcm(n_samples);
    for (size_t i = 0; i < n_samples; ++i) {
        float v = std::min(1.0f, std::max(-1.0f, samples[i]));
        pcm[i] = static_cast<int16_t>(std::nearbyint(v * 32767.0f));
    }
    f.write(reinterpret_cast<const char *>(pcm.data()), data_bytes);
}

inline void write_raw_bin(const std::string & path, const void * data, size_t n_bytes) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open binary file for writing: " + path);
    f.write(reinterpret_cast<const char *>(data), n_bytes);
}

} // namespace sakhiya
