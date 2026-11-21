"""Where is the naturalness gap? UTMOS of teacher wav vs vocoder(GT mel) vs vocoder(GTA mel) on validation clips. python3 scripts/diag_voc.py G.pt [n]"""
import sys, types, os, json, numpy as np, torch
sys.path.insert(0, "/work/tts/repo"); sys.argv_ = sys.argv
from sakhiya import audio
from sakhiya.vocoder import Vocoder, load_compat
import importlib.util
src = open("/work/tts/repo/scripts/evaluate.py").read()
i0 = src.index("def load_utmos"); i1 = src.index("try: utmos = ")
ns = dict(torch=torch, os=os, sys=sys, types=types); exec(src[i0:i1], ns)
dev = "cuda"; ut = ns["load_utmos"]().to(dev).eval()
def U(w, sr=22050):
    import scipy.signal as ss
    w = ss.resample_poly(w, 16000, sr).astype(np.float32); return float(ut(torch.from_numpy(w)[None].to(dev), 16000))
G = Vocoder().to(dev); ck = torch.load(sys.argv[1], weights_only=False); load_compat(G, ck["G"] if "G" in ck else ck); G.eval()
if os.environ.get("NOHACK"): G.train()
n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
vp = "/work/tts/feats/edge_val40.pt" if os.path.exists("/work/tts/feats/edge_val40.pt") and n <= 40 else "/work/tts/feats/edge.pt"
data = [r for r in torch.load(vp, weights_only=False) if r.get("split", "val") == "val"][:n]
gta = torch.load("/work/tts/feats/gta_edge.pt", weights_only=False); res = {"teacher": [], "gt": [], "gta": [], "gta_ids": 0}
for r in data:
    w = torch.from_numpy(r["wav"]).float().to(dev) / 32767; w = w[: w.shape[0] // 256 * 256]
    res["teacher"].append(U(w.cpu().numpy()))
    with torch.no_grad():
        y = G(audio.mel_spec(w[None], dev)); res["gt"].append(U(y[0].float().cpu().numpy()))
        if r["id"] in gta: y = G(torch.from_numpy(gta[r["id"]]).float().to(dev)[None]); res["gta"].append(U(y[0].float().cpu().numpy()))
print(json.dumps({k: (round(float(np.mean(v)), 3) if isinstance(v, list) and v else v) for k, v in res.items()}), flush=True)
