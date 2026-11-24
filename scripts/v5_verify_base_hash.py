#!/usr/bin/env python3
"""Fixed base-weight hash verification for the Vedic LoRA adapter (no training, inference-load only).
Checks: (1) base sha (rename-robust, LoRA keys excluded, ".base_layer." normalised) is identical before and after injecting the
adapter into a fresh model instance and equals the sha recorded at training start; (2) the adapter file holds LoRA tensors only;
(3) the on-disk base model.safetensors sha256 matches the Hugging Face LFS oid.   usage: v5_verify_base_hash.py [adapter_dir]"""
import hashlib, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "sakhiya_pro"))
import torch
from v5_vedic_adapter import load_base, load_vedic, BASE
AD = sys.argv[1] if len(sys.argv) > 1 else "/work/tts/v5_vedic/exp/vl1/adapter-800"


def base_sha(m):
    h = hashlib.sha256(); items = {}
    for n, p in m.named_parameters():
        if "lora_" in n: continue
        items[n.replace(".base_layer.", ".")] = p
    for n in sorted(items):
        h.update(n.encode()); h.update(items[n].detach().float().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()[:16], len(items)


out = {"adapter": AD}
cfg = json.load(open(f"{AD}/adapter_config.json")); out["recorded_base_sha"] = cfg["base_sha"]; out["step"] = cfg["step"]
from safetensors import safe_open
with safe_open(f"{AD}/vedic_lora.safetensors", "pt") as f:
    keys = list(f.keys())
out["adapter_tensors"] = len(keys); out["adapter_lora_only"] = all("lora_" in k for k in keys)
m = load_base(dtype=torch.float32)       # fp32 load: matches training-time hash dtype path (hash casts to float anyway)
s0, n0 = base_sha(m); out["sha_fresh_base"] = s0; out["n_base_tensors"] = n0
load_vedic(m, AD)
s1, n1 = base_sha(m); out["sha_after_adapter_injection"] = s1
out["base_frozen_ok"] = bool(s0 == s1)
out["matches_recorded"] = bool(s0 == cfg["base_sha"])
fp = f"{BASE}/model.safetensors"; h = hashlib.sha256()
with open(fp, "rb") as f:
    for ch in iter(lambda: f.read(1 << 24), b""): h.update(ch)
out["disk_sha256"] = h.hexdigest()
try:
    from huggingface_hub import HfApi
    ti = [x for x in HfApi(token=open("/tmp/hf_token").read().strip()).list_repo_tree("NikitHamal/sakhiya-v1-train", path_in_repo="sakhiya_pro_v3/exp/v3b/latest", repo_type="model") if x.path.endswith("model.safetensors")][0]
    out["hf_lfs_sha256"] = ti.lfs.sha256; out["disk_matches_hf"] = bool(ti.lfs.sha256 == out["disk_sha256"])
except Exception as e:
    out["hf_check_error"] = repr(e)[:160]
out["PASS"] = bool(out["base_frozen_ok"] and out["adapter_lora_only"] and out.get("disk_matches_hf", False))
json.dump(out, open("/work/tts/v5_recovery/state/base_hash_verify.json", "w"), indent=1); print(json.dumps(out, indent=1))
