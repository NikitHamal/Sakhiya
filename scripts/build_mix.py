"""Merge the three data tiers into feats/mix.pt (train on hours, not minutes):
  tier A  real studio speech   : SLR43 female + FLEURS + CommonVoice   (feats/v2.pt, also holds the held-out val/test splits)
  tier B  real crowd speech    : OpenSLR-54, speaker-first mined, gender-tagged (feats/slr54.pt)
  tier C  clean teacher corpus : edge_f / edge_m (feats/edge.pt, research-only, optional)
Also picks the best real female / male speaker for the 'human' voices (real_voices.json)."""
import json, collections, torch
F = "/work/tts/feats"
v2 = torch.load(f"{F}/v2.pt", weights_only=False); s54 = torch.load(f"{F}/slr54.pt", weights_only=False)
try: ed = torch.load(f"{F}/edge.pt", weights_only=False)
except Exception as e: print("no edge tier", e); ed = []
for r in s54: r["split"] = "train"
import os
sa = torch.load(f"{F}/sa.pt", weights_only=False) if os.path.exists(f"{F}/sa.pt") else []
mix = v2 + s54 + ed + sa
hrs = lambda rs: round(sum(len(r["wav"]) for r in rs) / 22050 / 3600, 2)
tr = [r for r in mix if r["split"] == "train"]
print("hours train: v2", hrs([r for r in v2 if r["split"] == "train"]), "slr54", hrs(s54), "edge", hrs([r for r in ed if r["split"] == "train"]), "total", hrs(tr), flush=True)
by = collections.defaultdict(list)
for r in s54: by[r["spk"]].append(r)
info = json.load(open(f"{F}/slr54_speakers.json"))
best = {g: max((s for s in by if by[s][0].get("gender") == g), key=lambda s: len(by[s]) * (info[s]["snr"] - 40 * info[s]["cer"]), default=None) for g in "fm"}
fem_s43 = [r for r in v2 if r["src"] == "slr43" and r["split"] == "train"]
if fem_s43: best["f"] = collections.Counter(r["spk"] for r in fem_s43).most_common(1)[0][0]
json.dump(dict(best=best, hours=dict(v2=hrs(v2), slr54=hrs(s54), edge=hrs(ed))), open(f"{F}/real_voices.json", "w"), indent=1)
print("human voices", best, flush=True); torch.save(mix, f"{F}/mix.pt"); print("MIX_DONE", len(mix), flush=True)
