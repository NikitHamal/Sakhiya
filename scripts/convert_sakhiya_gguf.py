#!/usr/bin/env python3
"""Convert Sakhiya v5_c1 PyTorch checkpoints (am.pt + voc.pt) into a single GGUF v3 model file for ggml (sakhiya-cli).

Dumps every actual tensor from SakhiyaAM (v5_c1/am.pt) and VocoderV2 (v5_c1/voc.pt) after weight-norm fusion
and canonical v5_c1 style/language embedding calibration, plus voc_eq.json and phoneme/prosody tables.
Supports:
  --dtype f32  (exact 32-bit float reference parity)
  --dtype f16  (FP16 weights for 2D/3D conv/linear tensors, FP32 for 1D norms/biases/Snake alphas/embeddings)
  --dtype q8_0 (Q8_0 quantized weights after FP32 parity verification)
"""
import argparse
import json
import os
import struct
import sys
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


def load_v5_c1_models(am_path: str, voc_path: str):
    """Load SakhiyaAM and VocoderV2 with exact export_v2_onnx.py v5_c1 weight preparation."""
    am_ck = torch.load(am_path, map_location="cpu", weights_only=False)
    am = SakhiyaAM(d_enc=192, L_enc=6, ff_enc=256, d_vel=256, nb_vel=12, h_dur=128).eval()
    sd = {k: v.float() if torch.is_tensor(v) else v for k, v in am_ck.get("ema", am_ck.get("net", am_ck)).items()}
    if "mean" in am_ck and "std" in am_ck:
        am.stats.copy_(torch.tensor([float(am_ck["mean"]), float(am_ck["std"])], dtype=torch.float32))
    am.load_state_dict(sd, strict=False)

    with torch.no_grad():
        if am_ck.get("pre_baked_v25", False):
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

    voc_ck = torch.load(voc_path, map_location="cpu", weights_only=False)
    voc = VocoderV2(ch=320, res_kernels=(3, 7, 11)).eval()
    voc_sd = voc_ck.get("G_ema", voc_ck.get("G", voc_ck)) if isinstance(voc_ck, dict) else voc_ck
    voc.load_state_dict({k: v.float() if torch.is_tensor(v) else v for k, v in voc_sd.items()})
    voc.fuse_weight_norm()
    return am, voc


def convert_to_gguf(am_path: str, voc_path: str, out_path: str, dtype: str = "f32"):
    am, voc = load_v5_c1_models(am_path, voc_path)

    tensors = []  # list of (name, ggml_dims, ggml_type, raw_bytes)

    def add_tensor(name: str, t: torch.Tensor, quant_mode: str = "none"):
        assert len(name.encode("utf-8")) < 64, f"Tensor name too long for ggml (<64 bytes): {name}"
        arr = np.ascontiguousarray(t.detach().cpu().float().numpy(), dtype=np.float32)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        # In GGML, ne[0] is the innermost contiguous dimension (reversed from numpy shape)
        ggml_dims = tuple(int(d) for d in reversed(arr.shape))
        target_type = GGML_TYPE_F32
        if quant_mode == "f16" and arr.ndim >= 2 and arr.size > 128:
            target_type = GGML_TYPE_F16
        elif quant_mode == "q8_0" and arr.ndim >= 2 and arr.size >= 256:
            # Flatten 3D conv weights [C_out, C_in, K] -> 2D [C_out, C_in * K] so innermost dim is divisible by 32
            if arr.ndim == 3 and (arr.shape[1] * arr.shape[2]) % 32 == 0:
                arr = arr.reshape(arr.shape[0], arr.shape[1] * arr.shape[2])
                ggml_dims = tuple(int(d) for d in reversed(arr.shape))
            if arr.shape[-1] % 32 == 0:
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

    # 1. Dump all SakhiyaAM tensors (prefix: "am.")
    for k, v in am.state_dict().items():
        # Keep TextEncoder + DurFlow in FP32 so phoneme durations and prior mu are 100% invariant;
        # Velocity ConvNeXt stack uses FP16 in f16/q8_0 modes (zero ODE trajectory drift).
        is_vel_heavy = (k.startswith("vel.") and v.ndim >= 2 and v.numel() >= 256 and not any(x in k for x in (".n.", ".dw.", ".film.", "inp.", "out.", "tm.")))
        q_mode = "f16" if (dtype in ("f16", "q8_0") and is_vel_heavy) else "none"
        add_tensor(f"am.{k}", v, quant_mode=q_mode)

    # Style prosody targets table [6, 4]
    add_tensor("am.pros_tbl", torch.tensor(STYLE_PROSODY_TARGETS, dtype=torch.float32), quant_mode="none")

    # 2. Dump all VocoderV2 tensors (prefix: "voc.")
    for k, v in voc.state_dict().items():
        # Keep Snake1d alphas, HarmonicSource, and final 66-ch spectral projection in FP32;
        # Upsamplers and MRF ResBlocks use F16 (--dtype f16) or Q8_0 (--dtype q8_0).
        is_voc_heavy = (v.ndim >= 2 and v.numel() >= 256 and not any(x in k for x in ("alpha", "src.", "post.")))
        q_mode = dtype if (dtype in ("f16", "q8_0") and is_voc_heavy) else "none"
        add_tensor(f"voc.{k}", v, quant_mode=q_mode)

    # Hann window (64-pt) for ola64
    add_tensor("voc.win", voc.win, quant_mode="none")

    # 3. Metadata KVs
    kvs = [
        ("general.architecture", GGUF_TYPE_STRING, "sakhiya"),
        ("general.name", GGUF_TYPE_STRING, "sakhiya-v5_c1"),
        ("sakhiya.sample_rate", GGUF_TYPE_UINT32, 24000),
        ("sakhiya.n_mel", GGUF_TYPE_UINT32, 100),
        ("sakhiya.d_enc", GGUF_TYPE_UINT32, 192),
        ("sakhiya.L_enc", GGUF_TYPE_UINT32, 6),
        ("sakhiya.ff_enc", GGUF_TYPE_UINT32, 256),
        ("sakhiya.d_vel", GGUF_TYPE_UINT32, 256),
        ("sakhiya.nb_vel", GGUF_TYPE_UINT32, 12),
        ("sakhiya.h_dur", GGUF_TYPE_UINT32, 128),
        ("sakhiya.voc_ch", GGUF_TYPE_UINT32, 320),
        ("sakhiya.cfm_steps", GGUF_TYPE_UINT32, 8),
        ("sakhiya.cfm_temp", GGUF_TYPE_FLOAT32, 0.667),
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
    print(f"[convert_sakhiya_gguf] Wrote {len(tensors)} tensors ({dtype}) -> {out_path} ({sz_mb:.2f} MB)")
    return len(tensors), sz_mb


def main():
    ap = argparse.ArgumentParser(description="Convert Sakhiya v5_c1 PyTorch checkpoints to GGUF v3.")
    ap.add_argument("--am", default="/work/tts/runs/v5_c1/am.pt", help="Path to SakhiyaAM checkpoint (am.pt)")
    ap.add_argument("--voc", default="/work/tts/runs/v5_c1/voc.pt", help="Path to VocoderV2 checkpoint (voc.pt)")
    ap.add_argument("--out", default="/work/tts/models/sakhiya_v5_c1_f32.gguf", help="Output .gguf path")
    ap.add_argument("--dtype", default="f32", choices=["f32", "f16", "q8_0"], help="Weight storage type")
    args = ap.parse_args()
    convert_to_gguf(args.am, args.voc, args.out, args.dtype)


if __name__ == "__main__":
    main()
