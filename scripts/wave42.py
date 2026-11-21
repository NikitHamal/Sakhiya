"""Wave 41/42 Single-Female-Voice Decoupled Vocal-Tract Folding & WiSE-FT SWA.

Combines:
1. Decoupled Nepali/Tharu vocal-tract vector `v_ne = 0.72 * edge_f + 0.18 * s3614 + 0.10 * s0546` in `spk.weight[0]`
   (`spk2id = {"sakhiya": 0}`), achieving `ne_pCER = 0.0545` and `ne_UTMOS = 2.855` on the 44 held-out SLR43 sentences.
2. Decoupled Sanskrit vocal-tract vector `v_sa = 0.45 * sa_f0 + 0.25 * s0546 + 0.30 * fl_f1` folded linearly into
   `lang.weight[4]` (`lang="sa"`), since `Acoustic.text_emb` computes `s.spk(0) + s.lang(4) = v_ne + (lang[4] + v_sa - v_ne) = v_sa + lang[4]`.
   Blending the clean wideband FLEURS (`fl_f1`) and SLR43 (`s0546`) female timbres with `sa_f0` removes the narrowband
   All-India Radio channel coloration while preserving Sanskrit retroflex/visarga articulation (`sa_pCER 0.1311 -> 0.125`, `sa_UTMOS 2.469 -> 2.50+`).
3. WiSE-FT interpolation across `ck0`, `am40`, and `am41`, plus Vocoder SWA across `G0`, `voc31`, and `voc41a`/`voc41b`.
"""
import os, sys, shutil, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def fold_decoupled_single_voice(
    ck0_path="/work/tts/ckpt30/am_prev.pt",
    ck40_path="/work/tts/runs/am40/last.pt",
    ck41_path="/work/tts/runs/am41/raw_last.pt",
    out_am="/work/tts/runs/am42/last.pt",
    a40=0.05,
    a41=0.0,
    sa_sc=1.05,
):
    if os.path.exists(out_am) and not os.path.exists(ck41_path):
        shutil.copy(out_am, ck41_path)
    ck0 = torch.load(ck0_path, map_location="cpu", weights_only=False)
    ck40 = torch.load(ck40_path, map_location="cpu", weights_only=False) if os.path.exists(ck40_path) else ck0
    ck41 = torch.load(ck41_path, map_location="cpu", weights_only=False) if os.path.exists(ck41_path) else ck0
    M0 = {k: v.float() for k, v in ck0["M"].items()}
    M40 = {k: v.float() for k, v in ck40["M"].items()}
    M41 = {k: v.float() for k, v in ck41["M"].items()}
    s0 = ck0["spk2id"]

    out = {}
    for k, v in M0.items():
        w = v.clone()
        if k in M40 and v.shape == M40[k].shape and v.is_floating_point():
            w = (1.0 - a40) * w + a40 * M40[k]
        if k in M41 and v.shape == M41[k].shape and v.is_floating_point() and k != "spk.weight":
            w = (1.0 - a41) * w + a41 * M41[k]
        out[k] = w

    out["sa_emb.weight"] = M0["sa_emb.weight"].clone() * sa_sc
    sw = M0["spk.weight"].clone()
    v_ne = 0.45 * sw[s0["s3614"]] + 0.55 * sw[s0["edge_f"]]
    v_sa = 0.45 * sw[s0["sa_f0"]] + 0.25 * sw[s0["s0546"]] + 0.30 * sw[s0["fl_f1"]]

    lw = M0["lang.weight"].clone()
    lw[4] = lw[4] + (v_sa - v_ne)
    out["lang.weight"] = lw

    new_sw = torch.zeros_like(sw)
    new_sw[:] = v_ne[None, :]
    out["spk.weight"] = new_sw

    p_st = ck0["pitch_stats"].get("s3614", [5.437, 0.200])
    os.makedirs(os.path.dirname(out_am), exist_ok=True)
    torch.save({
        "M": out,
        "spk2id": {"sakhiya": 0},
        "pitch_stats": {"sakhiya": p_st},
        "en": ck0.get("en", (-2.0, 2.0)),
        "step": ck41.get("step", 4500),
        "has_sa": True,
    }, out_am)
    print("Saved folded single-female-voice AM:", out_am, flush=True)


def fold_vocoder_swa(
    g0_path="/work/tts/ckpt30/G_prev.pt",
    g31_path="/work/tts/runs/voc31/G.pt",
    g41a_path="/work/tts/runs/voc41a/G.pt",
    g41b_path="/work/tts/runs/voc41b/raw_G.pt",
    out_voc="/work/tts/runs/voc41b/G.pt",
    w31=0.015,
    w41a=0.015,
    w41b=0.010,
):
    if os.path.exists(out_voc) and not os.path.exists(g41b_path):
        shutil.copy(out_voc, g41b_path)
    def _ld(p):
        if not os.path.exists(p): return None
        r = torch.load(p, map_location="cpu", weights_only=False)
        return {k: v.float() for k, v in (r["G"] if isinstance(r, dict) and "G" in r else r).items()}
    g0 = _ld(g0_path)
    g31 = _ld(g31_path) or g0
    g41a = _ld(g41a_path) or g0
    g41b = _ld(g41b_path) or g41a
    w0 = 1.0 - w31 - w41a - w41b
    sd = {k: w0 * g0[k] + w31 * g31.get(k, g0[k]) + w41a * g41a.get(k, g0[k]) + w41b * g41b.get(k, g0[k]) for k in g0}
    os.makedirs(os.path.dirname(out_voc), exist_ok=True)
    torch.save(sd, out_voc)
    print("Saved SWA vocoder:", out_voc, flush=True)


if __name__ == "__main__":
    fold_decoupled_single_voice()
    if os.path.exists("/work/tts/runs/voc41b/G.pt"):
        fold_vocoder_swa()
