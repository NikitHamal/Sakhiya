#!/usr/bin/env python3
"""Isolated Vedic-only LoRA for OmniVoice on top of the FROZEN Sakhiya-Pro v5 Nepali weights (v3b step-600).

Isolation guarantees:
  * base weights are never optimised or written; only 'lora_' params train; adapter saved to its own file
  * sha of base tensors recorded before + after training (must be identical) -> Nepali cannot regress
  * adapter is injected ONLY for style == 'vedic_chant' (see scripts/v5_vedic_adapter.py); Nepali path never loads it
Guard: hard wall-clock cap + step-50 throughput watchdog (like guarded_train.py).
usage: v5_vedic_lora_train.py --name vl1 --base <v3b-600 dir> --data_cfg <data_cfg.json> [--minutes 40]
"""
import argparse, hashlib, json, os, sys, time, logging
import torch
from peft import LoraConfig, inject_adapter_in_model, get_peft_model_state_dict
from safetensors.torch import save_file
from omnivoice.training.config import TrainingConfig
from omnivoice.training.builder import build_model_and_tokenizer, build_dataloaders

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="vl1"); ap.add_argument("--base", required=True); ap.add_argument("--data_cfg", required=True)
ap.add_argument("--minutes", type=float, default=40); ap.add_argument("--steps", type=int, default=2400)
ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--batch_tokens", type=int, default=8192); ap.add_argument("--save_steps", type=int, default=200)
ap.add_argument("--prompt_ratio", type=float, default=0.10)
a = ap.parse_args()
logging.basicConfig(level=logging.INFO)
OUT = f"/work/tts/v5_vedic/exp/{a.name}"; os.makedirs(OUT, exist_ok=True)

cfg = TrainingConfig(output_dir=OUT, data_config=a.data_cfg, init_from_checkpoint=a.base, learning_rate=a.lr, steps=a.steps,
                     batch_tokens=a.batch_tokens, attn_implementation="sdpa", num_workers=4, language_ratio=1.0,
                     use_pinyin_ratio=0.0, instruct_ratio=0.0, only_instruct_ratio=0.0,
                     prompt_ratio_range=(a.prompt_ratio, a.prompt_ratio), drop_cond_ratio=0.1)
model, tok = build_model_and_tokenizer(cfg)
for p in model.parameters(): p.requires_grad_(False)


def base_sha(m):
    h = hashlib.sha256()
    for n, p in sorted(m.named_parameters(), key=lambda x: x[0]):
        if "lora_" in n: continue
        h.update(n.encode()); h.update(p.detach().float().flatten()[:4096].cpu().numpy().tobytes())
    return h.hexdigest()[:16]


sha0 = base_sha(model)
lcfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, bias="none",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
inject_adapter_in_model(lcfg, model.llm)
params = [p for n, p in model.named_parameters() if "lora_" in n]
for p in params: p.requires_grad_(True)
print(f"BASE_SHA_BEFORE {sha0}  lora_params {sum(p.numel() for p in params)/1e6:.2f}M", flush=True)
dev = torch.device("cuda"); model.to(dev).train()
train_loader, eval_loader = build_dataloaders(cfg, tok)
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
warm = max(20, int(0.05 * a.steps))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + __import__("math").cos(3.14159 * min(1, s / a.steps)))))


def to_dev(b): return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}


def save(step):
    sd = {k: v.detach().cpu().contiguous() for k, v in get_peft_model_state_dict(model.llm).items()}
    d = f"{OUT}/adapter-{step}"; os.makedirs(d, exist_ok=True)
    save_file(sd, f"{d}/vedic_lora.safetensors")
    json.dump(dict(r=a.rank, alpha=2 * a.rank, targets=sorted(lcfg.target_modules), base=a.base, step=step, base_sha=sha0, style="vedic_chant"),
              open(f"{d}/adapter_config.json", "w"))


@torch.no_grad()
def dev_loss(n=15):
    model.eval(); L = []
    for i, b in enumerate(eval_loader):
        if i >= n: break
        with torch.autocast("cuda", dtype=torch.bfloat16): L.append(model(**to_dev(b)).loss.item())
    model.train(); return sum(L) / max(1, len(L))


t0 = time.time(); step = 0; it = iter(train_loader); run = []; t50 = None
while step < a.steps:
    try: b = next(it)
    except StopIteration: it = iter(train_loader); b = next(it)
    with torch.autocast("cuda", dtype=torch.bfloat16): loss = model(**to_dev(b)).loss
    loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    step += 1; run.append(loss.item())
    if step == 5: t50 = time.time()
    if step == 50:
        sps = 45 / (time.time() - t50); eta = (a.steps - step) / sps / 60
        print(f"STEP50 CHECK {sps:.2f} steps/s projected {eta:.1f} min (cap {a.minutes})", flush=True)
        if eta > a.minutes * 1.6: a.steps = int(a.minutes * 60 * sps * 0.95); print("steps reduced to", a.steps, flush=True)
    if step % 20 == 0: print(f"Step {step} | loss {sum(run[-20:])/20:.4f} | {time.time()-t0:.0f}s", flush=True)
    if step % a.save_steps == 0:
        save(step); print(f"CKPT {step} dev_loss {dev_loss():.4f}", flush=True)
    if time.time() - t0 > a.minutes * 60: print("hard cap reached", flush=True); break
save(step); sha1 = base_sha(model)
rep = dict(steps=step, minutes=round((time.time() - t0) / 60, 1), base_sha_before=sha0, base_sha_after=sha1, base_frozen=(sha0 == sha1))
json.dump(rep, open(f"{OUT}/train_report.json", "w")); print("DONE", rep, flush=True)
