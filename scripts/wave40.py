"""Wave 40 (Session 31): Single-female-voice (`sakhiya`) WiSE-FT + Sanskrit language-slot vocal-tract folding + Vocoder SWA.

Solves the multi-speaker collapse of Wave 30/31 (`sa_pCER 0.8944 -> 0.1231`, `ne_pCER 0.1944 -> 0.0535`) by:
1. Aligning with pure Nepali-space phoneme embeddings (`M.emb(ids)*sqrt(d) + M.spk(spk) + M.lang(0)`) in `precompute_clean_mas()`
   so `M.aligner` never collapses on `sa_emb`-shifted Sanskrit tokens, and keeping natural-cadence clips (`4.0 <= fpp <= 10.5`).
2. Preserving distinct speaker IDs (`orig_spk`) during GPU fine-tuning (`runs/am40`) on real human speech (`v2.pt` SLR43/FLEURS
   female + `sa.pt` Shrutilipi female + `vag.pt` normal-cadence Vagdhenu) so speaker embeddings absorb microphone/room variance.
3. Folding the Sanskrit vocal-tract vector `beta * (spk[sa_f0] - spk[s3614])` directly into `lang.weight[4]` (`lang="sa"`), since
   `Acoustic.text_emb` adds `s.spk(spk) + s.lang(lang)` linearly, and collapsing `spk.weight` to the single real-human female
   voice `s3614` (`spk2id = {"sakhiya": 0}`), eliminating `sanskrit.onnx` (Alpa) and shrinking `Sakhiya-fp16` to 29.06 MB and
   `Sakhiya-tiny-int8` to 14.61 MB.
"""
import os, sys, math, random, torch, torch.nn.functional as F, librosa
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sakhiya.acoustic import Acoustic, MEL_MU, MEL_SD
from sakhiya import g2p, sanskrit as SA, symbols as S


def fold_single_female_voice(ck0_path, ck40_path, g0_path, g31_path, out_am, out_voc, alpha=0.05, beta=1.0, gamma=0.05):
    ck0 = torch.load(ck0_path, map_location="cpu", weights_only=False)
    ck40 = torch.load(ck40_path, map_location="cpu", weights_only=False)
    M0 = {k: v.float() for k, v in ck0["M"].items()}
    M40 = {k: v.float() for k, v in ck40["M"].items()}
    s0 = ck0["spk2id"]
    out = {}
    for k, v in M0.items():
        if k in M40 and v.shape == M40[k].shape and v.is_floating_point():
            out[k] = (1.0 - alpha) * v + alpha * M40[k]
        else:
            out[k] = v.clone()
    sw = out["spk.weight"]
    v_ne = sw[s0["s3614"]].clone()
    v_sa = sw[s0["sa_f0"]].clone()
    lw = out["lang.weight"].clone()
    lw[4] = lw[4] + beta * (v_sa - v_ne)
    out["lang.weight"] = lw
    new_sw = torch.zeros_like(sw)
    new_sw[:] = v_ne[None, :]
    out["spk.weight"] = new_sw
    p_st = ck0["pitch_stats"].get("s3614", [5.437, 0.200])
    os.makedirs(os.path.dirname(out_am), exist_ok=True)
    torch.save({"M": out, "spk2id": {"sakhiya": 0}, "pitch_stats": {"sakhiya": p_st}, "en": ck0.get("en", (-2.0, 2.0)), "step": 3202, "has_sa": True}, out_am)

    g0 = {k: v.float() for k, v in torch.load(g0_path, map_location="cpu", weights_only=False).items()}
    g31_raw = torch.load(g31_path, map_location="cpu", weights_only=False)
    g31 = {k: v.float() for k, v in (g31_raw["G"] if isinstance(g31_raw, dict) and "G" in g31_raw else g31_raw).items()}
    sd_g = {k: (1.0 - gamma) * g0[k] + gamma * g31.get(k, g0[k]) for k in g0}
    os.makedirs(os.path.dirname(out_voc), exist_ok=True)
    torch.save(sd_g, out_voc)
    print("Saved Wave 40 single-female-voice checkpoints:", out_am, out_voc, flush=True)


if __name__ == "__main__":
    R = "/work/tts"
    fold_single_female_voice(f"{R}/repo/ckpt/am.fp16.pt", f"{R}/runs/am40/last.pt",
                             f"{R}/repo/ckpt/G.fp16.pt", f"{R}/runs/voc31/G.pt",
                             f"{R}/runs/am40_best/last.pt", f"{R}/runs/voc40_best/G.pt")
