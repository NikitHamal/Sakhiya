"""WiSE-FT weight soup between the shipped checkpoints and a fine-tuned run: python3 scripts/soup.py ALPHA TAG.
AM speaker tables differ between runs (shipped 32 speakers vs mix 130): rows are matched by speaker name, the fine-tuned table is the base."""
import os, sys, torch
R = "/work/tts"; a = float(sys.argv[1]); T = sys.argv[2]; KEEP = os.environ.get("KEEP", "")   # KEEP=am|voc -> ablation: that component stays the shipped one
am0 = torch.load(f"{R}/ckpt0/am.pt", weights_only=False, map_location="cpu"); am1 = torch.load(f"{R}/runs/am{T}/last.pt", weights_only=False, map_location="cpu")
M0, M1 = am0["M"], am1["M"]; out = {}
for k, v in M1.items():
    if k.startswith("sa_"):
        out[k] = v.float() if v.is_floating_point() else v; continue
    u = M0.get(k)
    if u is None or not v.is_floating_point() or u.shape != v.shape: out[k] = v.float() if v.is_floating_point() else v; continue
    out[k] = (1 - a) * u.float() + a * v.float()
if "lang.weight" in M1 and "lang.weight" in M0:
    lw = out["lang.weight"].clone()
    lw[4] = M1["lang.weight"][4].float()
    out["lang.weight"] = lw
if "spk.weight" in M1 and "spk.weight" in M0:
    w = M1["spk.weight"].float().clone(); s0, s1 = am0["spk2id"], am1["spk2id"]; hit = 0
    for n, i in s0.items():
        if n in s1 and not n.startswith("sa_"):
            w[s1[n]] = (1 - a) * M0["spk.weight"][i].float() + a * w[s1[n]]; hit += 1
    out["spk.weight"] = w; print("speaker rows blended", hit, "of", len(s0), flush=True)
am1["has_sa"] = True
am1["M"] = out; os.makedirs(f"{R}/runs/am{T}s", exist_ok=True); torch.save(am1, f"{R}/runs/am{T}s/last.pt")
g0 = torch.load(f"{R}/ckpt0/G.pt", weights_only=False, map_location="cpu"); g1 = torch.load(f"{R}/runs/voc{T}/G.pt", weights_only=False, map_location="cpu")
os.makedirs(f"{R}/runs/voc{T}s", exist_ok=True)
g1 = g0 if KEEP == "voc" else g1
torch.save({k: ((1 - a) * g0[k].float() + a * g1[k].float()) if g0[k].is_floating_point() and k in g1 else g0[k] for k in g0}, f"{R}/runs/voc{T}s/G.pt"); print("soup alpha", a, flush=True)
