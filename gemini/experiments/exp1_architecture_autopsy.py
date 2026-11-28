#!/usr/bin/env python3
"""Experiment 1 (Gemini Research): Complete Architectural, Parameter-Budget, Receptive-Field & CPU RTF Autopsy.

Compares:
1. All shipped Sakhiya ONNX tiers (`docs/models_*`)
2. Staged `Sakhiya-Studio-80M` (`sakhiya/studio_dit_clean.py`)
3. `Sakhiya-One Studio Vocos v2` (`cand_b_v2`: `SakhiyaOneAM` + `Mel80To100Adapter` + `PretrainedVocos24kHz`)
4. `Sakhiya-One Compact` (`SakhiyaOneAM` + `AWPVocoder`)
5. `n4ze3m/tiny-voca` (`4,537,291` params, 2 voices F2 & M2, 25.8K source-filter waveform head)
"""
import glob
import json
import os
import sys
import time
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto
import onnxruntime as ort
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

TV_SRC = "/work/gemini/tiny-voca/src"
if os.path.exists(TV_SRC) and TV_SRC not in sys.path:
    sys.path.insert(0, TV_SRC)


def count_pt_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def inspect_onnx_file(path: str) -> dict:
    if not os.path.exists(path):
        return {"exists": False}
    sz_mb = round(os.path.getsize(path) / 1e6, 3)
    m = onnx.load(path)
    total_elems = 0
    int8_elems = 0
    fp16_elems = 0
    fp32_elems = 0
    op_counts = {}
    for n in m.graph.node:
        op_counts[n.op_type] = op_counts.get(n.op_type, 0) + 1
    for init in m.graph.initializer:
        arr = numpy_helper.to_array(init)
        n_el = int(arr.size)
        total_elems += n_el
        if init.data_type in (TensorProto.INT8, TensorProto.UINT8):
            int8_elems += n_el
        elif init.data_type == TensorProto.FLOAT16:
            fp16_elems += n_el
        elif init.data_type == TensorProto.FLOAT:
            fp32_elems += n_el
    return {
        "exists": True,
        "file_mb": sz_mb,
        "total_initializer_params": total_elems,
        "int8_pct": round(100.0 * int8_elems / max(1, total_elems), 2),
        "fp16_pct": round(100.0 * fp16_elems / max(1, total_elems), 2),
        "fp32_pct": round(100.0 * fp32_elems / max(1, total_elems), 2),
        "total_nodes": len(m.graph.node),
        "top_ops": dict(sorted(op_counts.items(), key=lambda kv: -kv[1])[:8]),
    }


def analyze_pytorch_architectures() -> dict:
    from scripts.sakhiya_pro.distill.student import Student
    from sakhiya.vocos_24k_adapter import Mel80To100Adapter, PretrainedVocos24kHz
    from sakhiya.studio_dit_clean import StudioCleanTextEncoder, StudioCleanProsodyPredictor, StudioCleanDiTDecoder

    out = {}

    # 1. Sakhiya-One Compact (8.49 MB INT8)
    s1 = Student(tier="sakhiya_one")
    s1_enc = count_pt_params(s1.am.enc)
    s1_dur = count_pt_params(s1.am.dur)
    s1_vel = count_pt_params(s1.am.vel)
    s1_f0 = count_pt_params(s1.am.f0_head)
    s1_voc = count_pt_params(s1.voc)
    s1_tot = s1_enc + s1_dur + s1_vel + s1_f0 + s1_voc
    out["sakhiya_one_compact"] = {
        "total_params": s1_tot,
        "breakdown": {
            "text_encoder": s1_enc,
            "duration_predictor": s1_dur,
            "frame_prosody_f0_head": s1_f0,
            "frame_acoustic_decoder_vel": s1_vel,
            "waveform_vocoder": s1_voc,
        },
        "pct_in_acoustic_and_text_vs_vocoder": {
            "linguistic_prosodic_acoustic_pct": round(100.0 * (s1_tot - s1_voc) / s1_tot, 2),
            "waveform_vocoder_pct": round(100.0 * s1_voc / s1_tot, 2),
        },
        "architectural_Bottlenecks": [
            "d_enc=144 with SVD rank r_enc=104 truncation on ConvNeXt FFN",
            "Frame decoder (SakhiyaOneMeanFlowDecoder, d=160, 8 blocks) has ZERO self-attention; receptive field is only 8 * (7-1) = 48 frames (~512 ms)",
            "Walsh-Hadamard + Monarch block-diagonal MLPs restrict cross-channel formant coupling",
            "AWPVocoder operates on 33-bin n_fft=64 (375 Hz bin width), forcing neural MRF blocks to synthesize pitch harmonics inside coarse 375 Hz bins",
        ],
    }

    # 2. Sakhiya-One Studio Vocos v2 (cand_b_v2: 23.01 MB INT8 / 35.53 MB FP16)
    ad = Mel80To100Adapter()
    vc = PretrainedVocos24kHz()
    cb_ad = count_pt_params(ad)
    cb_vc = count_pt_params(vc)
    cb_am = s1_enc + s1_dur + s1_vel  # note: f0_head is unused in SakhiyaOneVocosV2.infer!
    cb_tot = cb_am + cb_ad + cb_vc
    out["cand_b_v2_vocos_flagship"] = {
        "total_params": cb_tot,
        "breakdown": {
            "text_encoder": s1_enc,
            "duration_predictor_frozen": s1_dur,
            "frame_prosody_f0_head_active": 0,
            "frame_acoustic_decoder_vel": s1_vel,
            "mel_80_to_100_adapter": cb_ad,
            "waveform_vocoder_vocos24k": cb_vc,
        },
        "pct_in_acoustic_and_text_vs_vocoder": {
            "linguistic_prosodic_acoustic_pct": round(100.0 * (cb_am + cb_ad) / cb_tot, 2),
            "waveform_vocoder_pct": round(100.0 * cb_vc / cb_tot, 2),
        },
        "architectural_Bottlenecks": [
            "76.0% of all parameters (13.49M / 17.76M) sit in the Vocos mel-to-wav inverter while only 23.5% (4.18M) model text, timing, prosody, and acoustics",
            "ZERO explicit F0 or Voicing predictor in SakhiyaOneVocosV2.infer(): f0_head, f0_pred, and vuv_pred are bypassed and frozen",
            "Stage AM and Stage B2 trained with p_enc = 0 (zero F0 conditioning), forcing the deterministic mu_y-start decoder to average harmonic peaks across pitches",
            "Stage B2 applied narrowband STFT (n_fft=1024, 2048) + <4kHz phase + CQT losses between unconditioned-F0 AM predictions and real GT waveforms, teaching the model to blur harmonics",
            "No frame-level self-attention in acoustic decoder (48-frame = 0.51s local Conv1d receptive field only)",
            "Static sib_cavity_prof step-function added identically to both dental/alveolar 's' (स) and palatal 'ʃ' (श/ष)",
        ],
    }

    # 3. Sakhiya-Studio-80M Clean (Dense Reference)
    st_enc = StudioCleanTextEncoder()
    st_pros = StudioCleanProsodyPredictor()
    st_dit = StudioCleanDiTDecoder()
    p_st_enc = count_pt_params(st_enc)
    p_st_pros = count_pt_params(st_pros)
    p_st_dit = count_pt_params(st_dit)
    out["sakhiya_studio_80m_clean"] = {
        "total_am_params": p_st_enc + p_st_pros + p_st_dit,
        "vocoder_bigvgan_params": 112_400_000,
        "breakdown": {
            "text_encoder_8L_rope": p_st_enc,
            "prosody_and_prior_predictor": p_st_pros,
            "acoustic_dit_16L_rope": p_st_dit,
        },
        "notes": "Clean 19.18 dB HNR and 1.379 comb grit, but 195.6M total params (AM+BigVGAN) and 32 ODE steps (~780 MB FP32, not browser/edge deployable), and ne_pcer=0.0716 due to 6,164-utterance corpus limit.",
    }

    # 4. TinyVoca (n4ze3m/tiny-voca, 4,537,291 params, 2 voices F2 & M2)
    if os.path.exists(TV_SRC):
        from tiny_voca.model import TinyVoca
        from tiny_voca.config import Config
        tv = TinyVoca(Config())
        tv_emb = count_pt_params(tv.tokens) + count_pt_params(tv.stress) + count_pt_params(tv.voices)
        tv_enc = count_pt_params(tv.encoder)
        tv_dur = count_pt_params(tv.duration)
        tv_pros = (
            count_pt_params(tv.frame_voices)
            + count_pt_params(tv.frame_position)
            + count_pt_params(tv.prosody_stack)
            + count_pt_params(tv.prosody_head)
        )
        tv_ac = count_pt_params(tv.prosody_projection) + count_pt_params(tv.acoustic_stack)
        tv_voc = count_pt_params(tv.vocoder)
        tv_tot = count_pt_params(tv)
        out["tiny_voca_4p5m_reference"] = {
            "total_params": tv_tot,
            "voices": ["F2 (Female)", "M2 (Male)"],
            "breakdown": {
                "embeddings_phoneme_stress_voice": tv_emb,
                "text_encoder_4L_rope_d192": tv_enc,
                "duration_and_pause_predictor": tv_dur,
                "frame_transformer_A_prosody_2L_rope": tv_pros,
                "frame_transformer_B_acoustic_5L_rope": tv_ac,
                "source_filter_cepstral_waveform_head": tv_voc,
            },
            "pct_in_acoustic_and_text_vs_vocoder": {
                "linguistic_prosodic_acoustic_pct": round(100.0 * (tv_tot - tv_voc) / tv_tot, 2),
                "waveform_vocoder_pct": round(100.0 * tv_voc / tv_tot, 2),
            },
            "why_it_wins": [
                "99.43% of parameters (4.511M / 4.537M) are dedicated to full-rank RoPE Transformers (d=192, ff=640, k=7 conv + dense FFN, ZERO SVD/Monarch/Hadamard)",
                "7 full-utterance Frame-Level RoPE Self-Attention Transformer blocks (2 prosody + 5 acoustic) give global sentence-level coarticulation and prosody context",
                "Explicit per-phoneme Stress embedding (0/1/2) + dual-output Duration Predictor (log_dur + BCE pause-presence logit so word boundaries only pause when truly silent)",
                "Explicit Frame Prosody Head (log_F0, voiced_logit, log_energy) trained with Global-Variance Pitch-Spread Loss so contours never flatten",
                "Oscillator follows ground-truth F0 during training while Frame Transformer B predicts pitch-invariant complex cepstra (80 causal + 16 anti-causal + 32 noise cepstra)",
                "25,862-param Homomorphic Cepstral Source-Filter head with closed-form band-limited Dirichlet pulse train (0 neural phase-guessing buzz) and 4 noise sub-frame gains per frame (2.67 ms stop-burst precision)",
            ],
        }
    return out


def benchmark_shipped_tiers() -> dict:
    from sakhiya.onnx_tts import SakhiyaONNX
    test_sents = [
        ("ne", "नमस्ते सबै जनालाई, म सखिया हुँ, तपाईंको नेपाली समाचार वाचक।"),
        ("ne", "प्रसिद्ध एआई अनुसन्धान कम्पनीको शेयर बजारमा कारोबार खुलेको पहिलो दिनमै १७ प्रतिशतले उल्लेखनीय वृद्धि भएको छ।"),
        ("sa", "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः। मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय॥"),
    ]
    tiers_json = json.load(open(os.path.join(REPO, "docs", "tiers.json")))
    results = {}
    for t in tiers_json["tiers"]:
        tid = t["id"]
        mdir = os.path.join(REPO, "docs", t["dir"].rstrip("/"))
        tiny_flag = (t.get("variant") == "tiny")
        onnx_file = os.path.join(mdir, "model.int8.onnx" if tiny_flag else "model.fp16.onnx")
        if not os.path.exists(onnx_file):
            onnx_file = os.path.join(mdir, "encoder.int8.onnx")
        onnx_meta = inspect_onnx_file(onnx_file)
        try:
            tts_1t = SakhiyaONNX(mdir, tiny=tiny_flag, threads=1)
            tts_4t = SakhiyaONNX(mdir, tiny=tiny_flag, threads=4)
            # Warmup
            tts_1t.say(test_sents[0][1], lang="ne", style="news")
            tts_4t.say(test_sents[0][1], lang="ne", style="news")
            rtfs_1t, rtfs_4t, durs = [], [], []
            for lg, txt in test_sents:
                t0 = time.perf_counter()
                w1, sr = tts_1t.say(txt, lang=lg, style="news")
                dt1 = time.perf_counter() - t0
                dur_s = len(w1) / float(sr)
                rtfs_1t.append(dt1 / max(dur_s, 1e-4))
                durs.append(round(dur_s, 2))

                t0 = time.perf_counter()
                w4, _ = tts_4t.say(txt, lang=lg, style="news")
                dt4 = time.perf_counter() - t0
                rtfs_4t.append(dt4 / max(dur_s, 1e-4))
            results[tid] = {
                "label": t["label"],
                "mb": t["mb"],
                "onnx_inspection": onnx_meta,
                "rtf_1t_mean": round(float(np.mean(rtfs_1t)), 4),
                "speed_1t_x": round(1.0 / max(float(np.mean(rtfs_1t)), 1e-5), 1),
                "rtf_4t_mean": round(float(np.mean(rtfs_4t)), 4),
                "speed_4t_x": round(1.0 / max(float(np.mean(rtfs_4t)), 1e-5), 1),
                "clip_durs_s": durs,
            }
        except Exception as e:
            results[tid] = {"label": t["label"], "error": str(e), "onnx_inspection": onnx_meta}
    return results


def main():
    t0 = time.time()
    arch_comp = analyze_pytorch_architectures()
    tier_bench = benchmark_shipped_tiers()
    report = {
        "experiment": "exp1_architecture_autopsy",
        "elapsed_sec": round(time.time() - t0, 2),
        "architecture_comparison": arch_comp,
        "shipped_tiers_cpu_benchmark": tier_bench,
    }
    out_p = os.path.join(HERE, "..", "results", "exp1_architecture_autopsy.json")
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[EXP1 DONE] Saved {out_p} in {report['elapsed_sec']}s")


if __name__ == "__main__":
    main()
