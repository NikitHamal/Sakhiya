"""Sakhiya v2 Stage 2C (Track A): Dump 24 kHz Ground-Truth-Aligned (GTA) mels from converged SakhiyaAM EMA checkpoint.
Uses MAS alignment on ground-truth mels + 6-step midpoint ODE velocity integration so GTA mels are frame-aligned with VocoderV2 wav shards.
Saves /work/tts/feats/v2_gta_24k.pt (dict mapping clip id -> float16 [100, T] mel) and sets gta_ready=True in /work/tts/COORD.json.
usage: python3 scripts/v2_dump_gta.py [--ckpt /work/tts/runs/v2_am/last.pt] [--out /work/tts/feats/v2_gta_24k.pt]"""
import os, sys, glob, json, time, argparse
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sakhiya.cfm import SakhiyaAM, _mel_f0_z
from sakhiya.mas import maximum_path
from sakhiya.symbols import SYM2ID, LANG2ID


@torch.no_grad()
def dump_batch(net, xs, mean, std, steps=8, temp=0.667, dev="cuda"):
    L = max(len(x["ids"]) for x in xs); T = max(x["mel"].shape[1] for x in xs)
    B = len(xs)
    ids = torch.zeros(B, L, dtype=torch.long, device=dev)
    y = torch.zeros(B, 100, T, device=dev)
    il = torch.tensor([len(x["ids"]) for x in xs], device=dev)
    yl = torch.tensor([x["mel"].shape[1] for x in xs], device=dev)
    for k, x in enumerate(xs):
        ids[k, :len(x["ids"])] = x["ids"].to(dev)
        y[k, :, :x["mel"].shape[1]] = (x["mel"].to(dev).float() - mean) / std
    im = torch.arange(L, device=dev)[None] < il[:, None]
    ym = torch.arange(T, device=dev)[None] < yl[:, None]
    spk = torch.tensor([x["spk"] for x in xs], device=dev)
    lang = torch.tensor([LANG2ID[x["lang"]] for x in xs], device=dev)
    sty = torch.tensor([x.get("style_id", 4) for x in xs], device=dev)
    studio = torch.tensor([x.get("is_studio", 1) for x in xs], device=dev)
    fz = _mel_f0_z(y, ym)
    pos = (torch.arange(L, device=dev, dtype=torch.float32)[None] + 0.5) / il[:, None].float().clamp(min=1.0)
    t_idx = (pos * (yl[:, None].float() - 1.0)).clamp(min=0, max=T - 1).long()
    p_cond = torch.gather(fz, 1, t_idx) * im.float()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, mu, g = net.enc(ids, spk, lang, sty, studio, im, pitch_add=p_cond)
        mu_f = mu.float(); yf = y.float()
        lp = -0.5 * (mu_f.pow(2).sum(-1)[:, :, None] - 2 * torch.einsum("bld,bdt->blt", mu_f, yf) + yf.pow(2).sum(1)[:, None])
        attn = maximum_path(lp, im[:, :, None] & ym[:, None, :])
        mu_y = torch.einsum("blt,bld->bdt", attn.to(mu.dtype), mu)
        m = ym[:, None].to(mu.dtype)
        x_t = torch.randn_like(mu_y) * 0.667
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((B,), i * dt, device=dev)
            k1 = net.vel(x_t, t, mu_y, g, m)
            k2 = net.vel(x_t + 0.5 * dt * k1, t + 0.5 * dt, mu_y, g, m)
            x_t = x_t + dt * k2
    mel_pred = (x_t.float() * std + mean).cpu().to(torch.float16)
    return {xs[k]["id"]: mel_pred[k, :, :xs[k]["mel"].shape[1]].numpy() for k in range(B)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/work/tts/runs/v2_am/last.pt")
    ap.add_argument("--data", default="/work/tts/feats/v2_studio_24k.pt")
    ap.add_argument("--native", default="/work/tts/feats/v2_native_24k.pt")
    ap.add_argument("--out", default="/work/tts/feats/v2_gta_24k.pt")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--stage_a_soup", type=float, default=0.80)
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    c = torch.load(a.ckpt, map_location=dev, weights_only=False)
    sd = c.get("ema", c["net"])
    mean, std = float(c.get("mean", -6.177)), float(c.get("std", 2.492))
    p_2a = "/work/tts/runs/v2_am/last.pt"
    if a.stage_a_soup > 0 and os.path.exists(p_2a) and os.path.abspath(a.ckpt) != os.path.abspath(p_2a):
        c_a = torch.load(p_2a, map_location=dev, weights_only=False)
        sd_a = c_a.get("ema", c_a.get("net", c_a))
        sd = {k: a.stage_a_soup * sd_a[k].float() + (1.0 - a.stage_a_soup) * sd[k].float() for k in sd_a}
        mean, std = float(c_a.get("mean", mean)), float(c_a.get("std", std))
    net = SakhiyaAM().to(dev).eval()
    net.load_state_dict(sd)
    net.stats.copy_(torch.tensor([mean, std], device=dev))
    # Only dump GTA mels for clips present in v2_voc_shards so Stage 3B VocoderV2 uses them directly
    voc_ids = set()
    for p in glob.glob("/work/tts/feats/v2_voc_shards/*.pt"):
        for r in torch.load(p, weights_only=False):
            voc_ids.add(r["id"])
    D = torch.load(a.data, weights_only=False)["items"]
    if os.path.exists(a.native):
        D = D + torch.load(a.native, weights_only=False).get("items", [])
    items = []
    for x in D:
        if voc_ids and x["id"] not in voc_ids:
            continue
        ph = [q for p in x["phones"] for q in (["o", "m"] if p == "<om>" else [p])]
        ids = torch.tensor([SYM2ID[p] for p in ph if p in SYM2ID], dtype=torch.long)
        if 3 <= len(ids) and x["mel"].shape[1] <= 1400 and x["mel"].shape[1] > len(ids):
            x["ids"] = ids
            items.append(x)
    items.sort(key=lambda x: x["mel"].shape[1])
    out = {}
    for i in range(0, len(items), a.bs):
        out.update(dump_batch(net, items[i:i + a.bs], mean, std, dev=dev))
        if (i // a.bs) % 25 == 0:
            print(f"GTA dumped {len(out)}/{len(items)} ({int(time.time()-t0)}s)", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(out, a.out + ".tmp"); os.replace(a.out + ".tmp", a.out)
    try:
        cp = "/work/tts/COORD.json"; cj = json.load(open(cp))
        cj["gta_ready"] = True; cj["track_a"]["stage"] = f"Stage2C GTA ready ({len(out)} mels)"; cj["track_a"]["updated"] = time.time()
        json.dump(cj, open(cp, "w"), indent=1)
    except Exception as e:
        print("coord err", e)
    print(f"V2_GTA_DONE {len(out)} clips in {time.time()-t0:.1f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
