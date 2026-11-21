"""Wave 43 single-female-voice fold with a REAL-HUMAN-ONLY identity (no Edge-TTS embedding, unlike wave 42's `0.55*edge_f`).

Acoustic.text_emb adds spk(0) + lang(l) linearly, so one speaker row v_ne plus a Sanskrit language row (lang[4] += v_sa - v_ne)
gives Nepali/Tharu one timbre and Sanskrit another inside the single `sakhiya` id.  Here v_ne / v_sa are blends of REAL speakers.
usage: python3 scripts/fold43.py RAW_AM.pt OUT_AM.pt [--ne s3614:0.7,s0546:0.3] [--sa sa_f0:0.45,s0546:0.25,fl_f1:0.30]"""
import sys, argparse, torch

ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--ne", default="s3614:0.7,s0546:0.3"); ap.add_argument("--sa", default="sa_f0:0.45,s0546:0.25,fl_f1:0.30")
a = ap.parse_args()
ck = torch.load(a.src, map_location="cpu", weights_only=False); M = {k: v.float() for k, v in ck["M"].items()}; s2 = ck["spk2id"]
mix = lambda spec: sum(float(w) * M["spk.weight"][s2[n]] for n, w in (p.split(":") for p in spec.split(",")))
v_ne, v_sa = mix(a.ne), mix(a.sa)
M["lang.weight"] = M["lang.weight"].clone(); M["lang.weight"][4] += v_sa - v_ne
M["spk.weight"] = v_ne[None, :].repeat(M["spk.weight"].shape[0], 1)
ps = ck["pitch_stats"]; first = a.ne.split(",")[0].split(":")[0]
torch.save({"M": M, "spk2id": {"sakhiya": 0}, "pitch_stats": {"sakhiya": ps.get(first, [5.437, 0.2])}, "en": ck["en"], "step": ck.get("step", 0), "has_sa": True}, a.out)
print("folded", a.src, "->", a.out, "ne", a.ne, "sa", a.sa, flush=True)
