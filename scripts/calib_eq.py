"""Calibrate the vocoder inference EQ by long-term-average-spectrum (LTAS) matching to teacher speech, then compare UTMOS of candidates on predicted (GTA) mels.
usage: python3 scripts/calib_eq.py G.pt [n]  -> writes /work/tts/eq_candidates.json and prints UTMOS per candidate (existing / ltas / blend)."""
import sys, os, json, types, numpy as np, torch, scipy.signal as ss
sys.path.insert(0, "/work/tts/repo")
from sakhiya import audio, vocoder as V
src = open("/work/tts/repo/scripts/evaluate.py").read()
ns = dict(torch=torch, os=os, sys=sys, types=types); exec(src[src.index("def load_utmos"): src.index("try: utmos = ")], ns)
dev = "cuda"; ut = ns["load_utmos"]().to(dev).eval()
U = lambda w: float(ut(torch.from_numpy(ss.resample_poly(w, 16000, 22050).astype(np.float32))[None].to(dev), 16000))
G = V.Vocoder().to(dev); ck = torch.load(sys.argv[1], weights_only=False); V.load_compat(G, ck["G"] if "G" in ck else ck); G.eval()
n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
data = [r for r in torch.load("/work/tts/feats/edge.pt", weights_only=False) if r["split"] == "val"][:n]
gta = torch.load("/work/tts/feats/gta_edge.pt", weights_only=False); data = [r for r in data if r["id"] in gta]
win = torch.hann_window(1024, device=dev)
def ltas(w):
    S = torch.stft(w, 1024, 256, 1024, win, return_complex=True).abs() ** 2; return S.mean(-1)
old = {"pts": V._EQ["pts"].clone(), "pow": V._EQ["pow"], "contrast": V._EQ["contrast"]}
def run(eq):
    V._EQ.update(eq); out = []
    with torch.no_grad():
        for r in data: out.append(G(torch.from_numpy(gta[r["id"]]).float().to(dev)[None])[0])
    return out
V._EQ["pts"] = torch.ones_like(old["pts"]); base = run(V._EQ)
T = sum(ltas(torch.from_numpy(r["wav"]).float().to(dev) / 32767) / (ltas(torch.from_numpy(r["wav"]).float().to(dev) / 32767).sum()) for r in data) / len(data)
B = sum(ltas(w) / ltas(w).sum() for w in base) / len(base)
ratio = (T / B).sqrt().cpu().numpy()      # magnitude ratio per FFT bin (energy-normalised)
ratio = np.convolve(np.pad(ratio, 8, mode="edge"), np.ones(17) / 17, mode="valid")
pts_ltas = np.clip(np.interp(np.linspace(0, 512, 129), np.arange(513), ratio) ** (1 / old["pow"]), 0.6, 4.0)
cands = {"existing": old["pts"].flatten().numpy(), "ltas": pts_ltas, "blend": np.sqrt(old["pts"].flatten().numpy() * pts_ltas)}
res = {}
for k, p in cands.items():
    outs = run({"pts": torch.tensor(p, dtype=torch.float32).view(1, 1, -1), "pow": old["pow"], "contrast": old["contrast"]})
    res[k] = round(float(np.mean([U(o.float().cpu().numpy()) for o in outs])), 3); print(k, res[k], flush=True)
json.dump({"utmos": res, **{k: [float(x) for x in v] for k, v in cands.items()}}, open("/work/tts/eq_candidates.json", "w"))
