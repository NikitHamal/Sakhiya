"""Naturalness-first promote gate: python3 scripts/promote.py TAG. Promote if UTMOS rises >0.02 and pCER <= max(old+0.012, 0.085); else restore best<TAG>."""
import json, os, shutil, sys, torch
R = "/work/tts"; T = sys.argv[1]
g = lambda p: next(s for s in json.load(open(p))["systems"] if s["system"] == "Sakhiya-fp16")["sentence"]
new, old = g(f"{R}/repo/docs/metrics.json"), g(f"{R}/best{T}/metrics.json")
better = new["mos"] > old["mos"] + 0.02 and new["pcer"] <= max(old["pcer"] + 0.012, 0.085)
print("new", new["pcer"], new["mos"], "old", old["pcer"], old["mos"], "PROMOTE" if better else "REVERT", flush=True)
if better:
    os.makedirs(f"{R}/repo/ckpt", exist_ok=True); c = torch.load(f"{R}/runs/am{T}/last.pt", weights_only=False)
    c["M"] = {k: v.half() for k, v in c["M"].items()}; torch.save(c, f"{R}/repo/ckpt/am.fp16.pt")
    torch.save({k: v.half() for k, v in torch.load(f"{R}/runs/voc{T}/G.pt", weights_only=False).items()}, f"{R}/repo/ckpt/G.fp16.pt")
    shutil.copy(f"{R}/repo/docs/metrics.json", f"{R}/repo/eval/metrics.json")
else:
    for d in ("models", "samples"): shutil.rmtree(f"{R}/repo/docs/{d}"); shutil.copytree(f"{R}/best{T}/{d}", f"{R}/repo/docs/{d}")
    for d in ("docs", "eval"): shutil.copy(f"{R}/best{T}/metrics.json", f"{R}/repo/{d}/metrics.json")
