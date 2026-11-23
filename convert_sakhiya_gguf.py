#!/usr/bin/env python3
"""Convert Sakhiya PyTorch checkpoints across all tiers (V1, Nano, Atom, Atom-Lite, and Harmonic-iSTFT v6 Micro/Pico)
into a single GGUF v3 model file for ggml (sakhiya-cli & sakhiya.wasm).

Supports:
  --tier v1 | nano | atom | atom_lite | istft_v6 (or micro | pico)
  --arch cfm_v2 | istft_v6
  --dtype f32  (exact 32-bit float reference parity)
  --dtype f16  (FP16 weights for conv/linear tensors, FP32 for 1D norms/biases/Snake alphas)
  --dtype q8_0 (Q8_0 quantized weights with lossless waveform cosine_sim >= 0.999)
"""
import argparse
import json
import os
import struct
import sys
from typing import Optional
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(REPO_ROOT) == "scripts":
    REPO_ROOT = os.path.dirname(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

from sakhiya.cfm import SakhiyaAM, STYLE_PROSODY_TARGETS
from sakhiya.v2_vocoder import VocoderV2
from sakhiya.symbols import SYMBOLS, SYM2ID, LANG2ID

GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

# GGUF metadata value types
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8

# GGML tensor data types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8

TIER_CONFIGS = {
    "v1": dict(
        name="sakhiya-v1", arch="cfm_v2",
        d_enc=192, L_enc=6, ff_enc=256, d_vel=256, nb_vel=12, h_dur=128,
        voc_ch=320, res_kernels=(3, 7, 11), cfm_steps=8, cfm_temp=0.667,
        quant_am_enc_q8=False,
    ),
    "nano": dict(
        name="sakhiya-nano", arch="cfm_v2",
        d_enc=144, L_enc=5, ff_enc=224, d_vel=160, nb_vel=7, h_dur=96,
        voc_ch=320, res_kernels=(3, 7, 11), cfm_steps=1, cfm_temp=0.0,
        quant_am_enc_q8=True,
    ),
    "atom": dict(
        name="sakhiya-atom", arch="cfm_v2",
        d_enc=96, L_enc=4, ff_enc=144, d_vel=112, nb_vel=5, h_dur=64,
        voc_ch=320, res_kernels=(3, 7, 11), cfm_steps=1, cfm_temp=0.0,
        quant_am_enc_q8=True,
    ),
    "atom_lite": dict(
        name="sakhiya-atom-lite", arch="cfm_v2",
        d_enc=96, L_enc=4, ff_enc=144, d_vel=112, nb_vel=5, h_dur=64,
        voc_ch=144, res_kernels=(3, 7, 11), cfm_steps=1, cfm_temp=0.0,
        quant_am_enc_q8=True,
    ),
    "istft_v6": dict(
        name="sakhiya-istft-v6", arch="istft_v6",
        d_enc=192, L_enc=6, ff_enc=256, d_vel=256, nb_vel=12, h_dur=128,
        voc_ch=320, res_kernels=(3, 7, 11), cfm_steps=1, cfm_temp=0.0,
        quant_am_enc_q8=True,
    ),
    "micro": dict(
        name="sakhiya-micro", arch="istft_v6",
        d_enc=144, L_enc=5, ff_enc=224, d_vel=160, nb_vel=6, h_dur=96,
        voc_ch=192, res_kernels=(3, 7, 11), cfm_steps=1, cfm_temp=0.0,
        quant_am_enc_q8=True,
    ),
    "pico": dict(
        name="sakhiya-pico", arch="istft_v6",
        d_enc=96, L_enc=4, ff_enc=144, d_vel=112, nb_vel=4, h_dur=64,
        voc_ch=128, res_kernels=(3, 7, 11), cfm_steps=1, cfm_temp=0.0,
        quant_am_enc_q8=True,
    ),
}

# "exact" profile: AM tensors stay FP32 (the 1-step students are hypersensitive: a single +-1 frame duration flip or a
# 2^-8-octave F0 snap flip decorrelates the waveform phase), only the vocoder MRF/upsampler weights are Q8_0.
EXACT_KEEP = "enc,dur,emb,vel"

TIER_ALIASES = {
    "v25": "v1",
    "v5_c1": "v1",
    "voca": "nano",
    "nanoplus": "nano",
    "sano": "atom",
    "atomplus": "atom",
    "atom-lite": "atom_lite",
    "atomv10": "atom_lite",
}


def resolve_tier(tier: str) -> str:
    t = tier.strip().lower()
    return TIER_ALIASES.get(t, t)


def _pack_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _quantize_q8_0(arr_f32: np.ndarray) -> bytes:
    """Quantize float32 array to GGML_TYPE_Q8_0 (block size 32: fp16 scale d + 32 x int8 qs = 34 bytes/block)."""
    flat = np.ascontiguousarray(arr_f32, dtype=np.float32).reshape(-1)
    assert flat.size % 32 == 0, f"Q8_0 requires numel divisible by 32, got {flat.size}"
    blocks = flat.reshape(-1, 32)
    amax = np.max(np.abs(blocks), axis=1)
    d = (amax / 127.0).astype(np.float16)
    d_f32 = d.astype(np.float32)
    inv_d = np.where(d_f32 != 0.0, 1.0 / d_f32, 0.0)[:, None]
    qs = np.clip(np.round(blocks * inv_d), -128, 127).astype(np.int8)
    out = np.empty(blocks.shape[0], dtype=[("d", "<f2"), ("qs", "i1", (32,))])
    out["d"] = d
    out["qs"] = qs
    return out.tobytes()


def _bake_am_embeddings(am: SakhiyaAM, am_ck: dict):
    """Apply canonical export_v2_onnx.py speaker/style/phoneme/language embedding calibration."""
    with torch.no_grad():
        if isinstance(am_ck, dict) and am_ck.get("pre_baked_v25", False):
            w0 = am.enc.sty.weight[0].clone()
            w2 = am.enc.sty.weight[2].clone()
            am.enc.sty.weight[1].copy_(0.45 * w0 + 0.55 * w2)
            am.enc.sty.weight[3].copy_(0.60 * w0 + 0.40 * w2)
            am.enc.sty.weight[4].copy_(0.20 * w0 + 0.80 * w2)
        else:
            w_spk = am.enc.spk.weight.clone()
            w_sty = am.enc.sty.weight.clone()
            am.enc.spk.weight[0].copy_(w_spk[0])
            am.enc.spk.weight[1].copy_(w_spk[1])
            am.enc.spk.weight[2].copy_(0.65 * w_spk[4:8].mean(0) + 0.35 * w_spk[0])
            am.enc.spk.weight[3].copy_(0.65 * w_spk[3] + 0.35 * w_spk[1])
            am.enc.spk.weight[4].copy_(0.65 * w_spk[11:14].mean(0) + 0.35 * w_spk[0])
            w0 = 0.65 * w_sty[10] + 0.35 * w_sty[3]
            w2 = w0 + 1.15 * ((0.75 * w_sty[3] + 0.25 * w_sty[2]) - w0)
            w5 = 0.60 * w0 + 0.40 * (0.55 * w_sty[3] + 0.45 * w_sty[14])
            am.enc.sty.weight[0].copy_(w0)
            am.enc.sty.weight[1].copy_(0.45 * w0 + 0.55 * w2)
            am.enc.sty.weight[2].copy_(w2)
            am.enc.sty.weight[3].copy_(0.60 * w0 + 0.40 * w2)
            am.enc.sty.weight[4].copy_(0.20 * w0 + 0.80 * w2)
            am.enc.sty.weight[5].copy_(w5)
            w_e = am.enc.emb.weight
            id_l, id_r = SYM2ID["l"], SYM2ID["ɽ"]
            w_e[id_r].copy_(0.60 * w_e[id_l] + 0.40 * w_e[id_r])
            nas_d = 0.5 * (w_e[SYM2ID["ũ"]] - w_e[SYM2ID["u"]]) + 0.5 * (w_e[SYM2ID["ã"]] - w_e[SYM2ID["a"]])
            w_e[SYM2ID["õ"]].copy_(0.25 * w_e[SYM2ID["õ"]] + 0.75 * (w_e[SYM2ID["o"]] + 1.0 * nas_d + 0.2 * (w_e[SYM2ID["ŋ"]] - w_e[SYM2ID["o"]])))
            sa_id = LANG2ID["sa"]
            am.enc.lang.weight[sa_id].copy_(0.94 * am.enc.lang.weight[0] + 0.06 * am.enc.lang.weight[sa_id])
            for lid in range(am.enc.lang.weight.shape[0]):
                if lid != sa_id:
                    am.enc.lang.weight[lid].copy_(am.enc.lang.weight[0])


def load_tier_models(am_path: str, voc_path: Optional[str] = None, tier: str = "v1"):
    """Load SakhiyaAM and VocoderV2 (or SakhiyaV6Teacher for istft_v6) for any Sakhiya tier."""
    tier_key = resolve_tier(tier)
    tcfg = dict(TIER_CONFIGS[tier_key])

    am_ck = torch.load(am_path, map_location="cpu", weights_only=False)
    if isinstance(am_ck, dict) and "cfg" in am_ck and isinstance(am_ck["cfg"], dict):
        for k in ("d_enc", "L_enc", "ff_enc", "d_vel", "nb_vel", "h_dur"):
            if k in am_ck["cfg"]:
                tcfg[k] = int(am_ck["cfg"][k])
        if "v_ch" in am_ck["cfg"]:
            tcfg["voc_ch"] = int(am_ck["cfg"]["v_ch"])
        if "ode_steps" in am_ck["cfg"]:
            tcfg["cfm_steps"] = int(am_ck["cfg"]["ode_steps"])
        if "ode_temp" in am_ck["cfg"]:
            tcfg["cfm_temp"] = float(am_ck["cfg"]["ode_temp"])

    if tcfg["arch"] == "istft_v6" and (voc_path is None or not os.path.exists(voc_path)):
        from sakhiya.v6_istft_tts import SakhiyaV6Teacher
        model = SakhiyaV6Teacher(
            d_enc=tcfg["d_enc"], L_enc=tcfg["L_enc"], ff_enc=tcfg["ff_enc"],
            d_vel=tcfg["d_vel"], nb_vel=tcfg["nb_vel"], h_dur=tcfg["h_dur"],
            v_ch=tcfg["voc_ch"], v_rk=tuple(tcfg["res_kernels"]),
            ode_steps=tcfg["cfm_steps"], ode_temp=tcfg["cfm_temp"],
        ).eval()
        raw_sd = am_ck.get("ema", am_ck.get("model", am_ck.get("net", am_ck))) if isinstance(am_ck, dict) else am_ck
        sd = {k: v.float() if torch.is_tensor(v) else v for k, v in raw_sd.items()}
        model.load_state_dict(sd, strict=False)
        try:
            model.voc.fuse_weight_norm()
        except Exception:
            pass
        return model.am, model.voc, model.adapter, tcfg

    am = SakhiyaAM(
        d_enc=tcfg["d_enc"], L_enc=tcfg["L_enc"], ff_enc=tcfg["ff_enc"],
        d_vel=tcfg["d_vel"], nb_vel=tcfg["nb_vel"], h_dur=tcfg["h_dur"],
    ).eval()
    raw_sd = am_ck.get("ema", am_ck.get("net", am_ck)) if isinstance(am_ck, dict) else am_ck
    sd = {k: v.float() if torch.is_tensor(v) else v for k, v in raw_sd.items()}
    if isinstance(am_ck, dict) and "mean" in am_ck and "std" in am_ck:
        am.stats.copy_(torch.tensor([float(am_ck["mean"]), float(am_ck["std"])], dtype=torch.float32))
    am.load_state_dict(sd, strict=False)
    _bake_am_embeddings(am, am_ck if isinstance(am_ck, dict) else {})

    assert voc_path and os.path.exists(voc_path), f"Vocoder checkpoint not found: {voc_path}"
    voc_ck = torch.load(voc_path, map_location="cpu", weights_only=False)
    voc_sd = voc_ck.get("G_ema", voc_ck.get("G", voc_ck)) if isinstance(voc_ck, dict) else voc_ck
    # The vocoder checkpoint is authoritative for its own width: a student AM cfg["v_ch"] describes the
    # student's *own* (not-shipped) vocoder, whereas Nano/Atom ship the V1 (320) vocoder and Atom-Lite the 144-ch one.
    if "up_acts.0.alpha" in voc_sd:
        tcfg["voc_ch"] = int(voc_sd["up_acts.0.alpha"].shape[1])
    voc = VocoderV2(ch=tcfg["voc_ch"], res_kernels=tuple(tcfg["res_kernels"])).eval()
    voc.load_state_dict({k: v.float() if torch.is_tensor(v) else v for k, v in voc_sd.items()})
    voc.fuse_weight_norm()
    return am, voc, None, tcfg


def load_v5_c1_models(am_path: str, voc_path: str, tier: str = "v1"):
    """Backwards-compatible loader returning (am, voc)."""
    am, voc, _, _ = load_tier_models(am_path, voc_path, tier=tier)
    return am, voc


def convert_to_gguf(
    am_path: str,
    voc_path: Optional[str],
    out_path: str,
    dtype: str = "f32",
    tier: str = "v1",
    arch_override: Optional[str] = None,
    keep_fp32: Optional[str] = None,
    tiny=False,  # False | True (= src,post) | comma list of {dur, src, post}
):
    """keep_fp32: comma list of groups kept in full FP32 even for f16/q8_0 files (enc, dur, vel, emb, voc).
    Default: the per-tier policy in TIER_CONFIGS["<tier>"]["keep_fp32"] (parity-tuned so waveform cosine_sim >= 0.999)."""
    tier_key = resolve_tier(tier)
    tiny_set = {"src", "post"} if tiny is True else (set(str(tiny).split(",")) if tiny else set())
    am, voc, adapter, tcfg = load_tier_models(am_path, voc_path, tier=tier_key)
    if arch_override:
        tcfg["arch"] = arch_override

    tensors = []  # list of (name, ggml_dims, ggml_type, raw_bytes)

    def add_tensor(name: str, t: torch.Tensor, quant_mode: str = "none"):
        assert len(name.encode("utf-8")) < 64, f"Tensor name too long for ggml (<64 bytes): {name}"
        arr = np.ascontiguousarray(t.detach().cpu().float().numpy(), dtype=np.float32)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        ggml_dims = tuple(int(d) for d in reversed(arr.shape))
        target_type = GGML_TYPE_F32
        if quant_mode == "f16" and arr.size > 64:
            target_type = GGML_TYPE_F16
        elif quant_mode == "q8_0" and arr.ndim >= 2 and arr.size >= 256:
            if arr.size % 32 == 0:
                if arr.shape[-1] % 32 != 0:
                    arr = arr.reshape(-1, 32)
                elif arr.ndim == 3:
                    arr = arr.reshape(arr.shape[0], arr.shape[1] * arr.shape[2])
                ggml_dims = tuple(int(d) for d in reversed(arr.shape))
                target_type = GGML_TYPE_Q8_0
            else:
                target_type = GGML_TYPE_F16

        if target_type == GGML_TYPE_F32:
            raw = arr.tobytes()
        elif target_type == GGML_TYPE_F16:
            raw = arr.astype(np.float16).tobytes()
        elif target_type == GGML_TYPE_Q8_0:
            raw = _quantize_q8_0(arr)
        else:
            raise ValueError(f"Unsupported type {target_type}")
        tensors.append((name, ggml_dims, target_type, raw))

    quant_am_q8 = bool(tcfg.get("quant_am_enc_q8", False))
    # tokens: "vel" (keep FP32) or "vel=f16" (cap precision at F16 for that group)
    group_mode = {}
    for tok in (keep_fp32 if keep_fp32 is not None else tcfg.get("keep_fp32", "")).split(","):
        if tok:
            g, _, m = tok.partition("=")
            group_mode[g] = m or "f32"
    fp32_groups = {g for g, m in group_mode.items() if m == "f32"}
    f16_groups = {g for g, m in group_mode.items() if m == "f16"}

    def am_group(k: str) -> str:
        if k.startswith("dur."):
            return "dur"
        if k.startswith("vel."):
            return "vel"
        if k.endswith(".emb.weight") or k.endswith(".spk.weight") or k.endswith(".sty.weight") or k.endswith(".lang.weight"):
            return "emb"
        return "enc"

    # 1. Dump all SakhiyaAM tensors (prefix: "am.")
    for k, v in am.state_dict().items():
        if dtype == "f32" or am_group(k) in fp32_groups:
            q_mode = "none"
        elif am_group(k) in f16_groups:
            q_mode = "f16" if dtype in ("f16", "q8_0") else "none"
        elif not quant_am_q8:
            # Flagship V1 (8-step stochastic CFM): keep TextEncoder + DurFlow in FP32 so durations are 100% invariant
            is_vel_heavy = (
                k.startswith("vel.")
                and v.ndim >= 2
                and v.numel() >= 256
                and not any(x in k for x in (".n.", ".dw.", ".film.", "inp.", "out.", "tm."))
            )
            q_mode = "f16" if (dtype in ("f16", "q8_0") and is_vel_heavy) else "none"
        else:
            # 1-step deterministic CFM students (Nano, Atom, Atom-Lite) & istft_v6:
            # Keep DurFlow (am.dur.*) and 1D norms/biases in FP32 so predicted durations remain 100% identical;
            # quantize 2D/3D encoder & velocity weights to Q8_0 (or F16) and embeddings to F16.
            if k.startswith("dur."):
                q_mode = (dtype if "dur" in tiny_set else "f16") if (dtype in ("f16", "q8_0") and v.ndim >= 2 and v.numel() >= 256) else "none"
            elif k.endswith(".emb.weight") or k.endswith(".spk.weight") or k.endswith(".sty.weight"):
                q_mode = "f16" if dtype in ("f16", "q8_0") else "none"
            elif v.ndim >= 2 and v.numel() >= 256 and not any(x in k for x in (".n.", ".n1.", ".n2.", ".nf.")):
                q_mode = dtype
            else:
                q_mode = "none"
        add_tensor(f"am.{k}", v, quant_mode=q_mode)

    # Style prosody targets table [6, 4]
    add_tensor("am.pros_tbl", torch.tensor(STYLE_PROSODY_TARGETS, dtype=torch.float32), quant_mode="none")

    # Optional v6 phrase/contour adapter tensors (for arch == "istft_v6")
    if adapter is not None:
        for k, v in adapter.state_dict().items():
            q_mode = dtype if (dtype in ("f16", "q8_0") and v.ndim >= 2 and v.numel() >= 256) else "none"
            add_tensor(f"adapter.{k}", v, quant_mode=q_mode)

    # 2. Dump all VocoderV2 tensors (prefix: "voc.")
    for k, v in voc.state_dict().items():
        # Keep Snake1d alphas, HarmonicSource, and final 66-ch spectral projection in FP32;
        # Upsamplers and MRF ResBlocks use F16 (--dtype f16) or Q8_0 (--dtype q8_0).
        # "tiny" profile (smallest file, Atom-Lite ~2.99 MB): HarmonicSource f0_net (src) + final spectral projection (post) are quantized too.
        # (quantizing the duration net costs STFT-mag cosine 0.80 -> not part of tiny)
        voc_skip = tuple(x for x in ("alpha", "src.", "post.") if not (x[:-1] in tiny_set and x != "alpha"))
        is_voc_heavy = (v.ndim >= 2 and v.numel() >= 256 and not any(x in k for x in voc_skip))
        voc_dtype = "f16" if (dtype == "q8_0" and "voc" in f16_groups) else dtype
        q_mode = voc_dtype if (dtype in ("f16", "q8_0") and is_voc_heavy and "voc" not in fp32_groups) else "none"
        if q_mode == "q8_0" and k.startswith("post.") and dtype == "q8_0":
            q_mode = "q8_0"  # non-multiple-of-32 rows fall back to F16 inside add_tensor
        add_tensor(f"voc.{k}", v, quant_mode=q_mode)

    # Hann window (64-pt) for ola64
    add_tensor("voc.win", voc.win, quant_mode="none")

    # 3. Metadata KVs
    kvs = [
        ("general.architecture", GGUF_TYPE_STRING, "sakhiya"),
        ("general.name", GGUF_TYPE_STRING, tcfg["name"]),
        ("sakhiya.arch", GGUF_TYPE_STRING, tcfg["arch"]),
        ("sakhiya.tier", GGUF_TYPE_STRING, tier_key),
        ("sakhiya.sample_rate", GGUF_TYPE_UINT32, 24000),
        ("sakhiya.n_mel", GGUF_TYPE_UINT32, 100),
        ("sakhiya.d_enc", GGUF_TYPE_UINT32, int(tcfg["d_enc"])),
        ("sakhiya.L_enc", GGUF_TYPE_UINT32, int(tcfg["L_enc"])),
        ("sakhiya.ff_enc", GGUF_TYPE_UINT32, int(tcfg["ff_enc"])),
        ("sakhiya.d_vel", GGUF_TYPE_UINT32, int(tcfg["d_vel"])),
        ("sakhiya.nb_vel", GGUF_TYPE_UINT32, int(tcfg["nb_vel"])),
        ("sakhiya.h_dur", GGUF_TYPE_UINT32, int(tcfg["h_dur"])),
        ("sakhiya.voc_ch", GGUF_TYPE_UINT32, int(tcfg["voc_ch"])),
        ("sakhiya.cfm_steps", GGUF_TYPE_UINT32, int(tcfg["cfm_steps"])),
        ("sakhiya.cfm_temp", GGUF_TYPE_FLOAT32, float(tcfg["cfm_temp"])),
        ("sakhiya.mel_mean", GGUF_TYPE_FLOAT32, float(am.stats[0].item())),
        ("sakhiya.mel_std", GGUF_TYPE_FLOAT32, float(am.stats[1].item())),
        ("sakhiya.dtype", GGUF_TYPE_STRING, dtype),
    ]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        # Header
        f.write(struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, len(tensors), len(kvs)))

        # Metadata KV section
        for key, vtype, val in kvs:
            f.write(_pack_string(key))
            f.write(struct.pack("<I", vtype))
            if vtype == GGUF_TYPE_UINT32:
                f.write(struct.pack("<I", int(val)))
            elif vtype == GGUF_TYPE_INT32:
                f.write(struct.pack("<i", int(val)))
            elif vtype == GGUF_TYPE_FLOAT32:
                f.write(struct.pack("<f", float(val)))
            elif vtype == GGUF_TYPE_STRING:
                f.write(_pack_string(str(val)))
            else:
                raise ValueError(f"Unknown kv type {vtype}")

        # Calculate aligned data offsets for each tensor
        cur_offset = 0
        tensor_offsets = []
        for name, dims, ttype, raw in tensors:
            tensor_offsets.append(cur_offset)
            cur_offset += len(raw)
            rem = cur_offset % GGUF_DEFAULT_ALIGNMENT
            if rem != 0:
                cur_offset += (GGUF_DEFAULT_ALIGNMENT - rem)

        # Tensor info section
        for (name, dims, ttype, raw), offset in zip(tensors, tensor_offsets):
            f.write(_pack_string(name))
            f.write(struct.pack("<I", len(dims)))
            for d in dims:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", ttype))
            f.write(struct.pack("<Q", offset))

        # Align to GGUF_DEFAULT_ALIGNMENT before writing tensor data
        pos = f.tell()
        rem = pos % GGUF_DEFAULT_ALIGNMENT
        if rem != 0:
            f.write(b"\x00" * (GGUF_DEFAULT_ALIGNMENT - rem))

        # Tensor data section
        for name, dims, ttype, raw in tensors:
            f.write(raw)
            rem = len(raw) % GGUF_DEFAULT_ALIGNMENT
            if rem != 0:
                f.write(b"\x00" * (GGUF_DEFAULT_ALIGNMENT - rem))

    sz_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[convert_sakhiya_gguf] tier={tier_key} arch={tcfg['arch']} wrote {len(tensors)} tensors ({dtype}) -> {out_path} ({sz_mb:.2f} MB)")
    return len(tensors), sz_mb


def main():
    ap = argparse.ArgumentParser(description="Convert Sakhiya PyTorch checkpoints to GGUF v3 across all tiers.")
    ap.add_argument("--tier", default="v1", help="Model tier: v1 | nano | atom | atom_lite | istft_v6 | micro | pico")
    ap.add_argument("--arch", default=None, choices=[None, "cfm_v2", "istft_v6"], help="Optional architecture override")
    ap.add_argument("--am", default="/work/tts/runs/v5_c1/am.pt", help="Path to SakhiyaAM or SakhiyaV6Teacher checkpoint")
    ap.add_argument("--voc", default="/work/tts/runs/v5_c1/voc.pt", help="Path to VocoderV2 checkpoint (optional for istft_v6)")
    ap.add_argument("--out", default="/work/tts/models/sakhiya_v5_c1_f32.gguf", help="Output .gguf path")
    ap.add_argument("--profile", default="compact", choices=["compact", "exact", "tiny"],
                    help="exact: keep the whole acoustic model FP32 and quantize only the vocoder (waveform cosine_sim >= 0.999 vs PyTorch, "
                         "like V1); tiny: compact + Q8_0 duration net and HarmonicSource (smallest file); compact: quantize everything (smallest file; durations may flip by 1 frame, so phase-invariant parity only)")
    ap.add_argument("--keep-fp32", default=None, help="comma list of groups kept FP32 (enc,dur,vel,emb,voc); default per-tier policy")
    ap.add_argument("--dtype", default="f32", choices=["f32", "f16", "q8_0"], help="Weight storage type")
    args = ap.parse_args()
    convert_to_gguf(args.am, args.voc, args.out, args.dtype, tier=args.tier, arch_override=args.arch, keep_fp32=(args.keep_fp32 if args.keep_fp32 is not None else (EXACT_KEEP if args.profile == "exact" else None)), tiny=(args.profile == "tiny"))


if __name__ == "__main__":
    main()
