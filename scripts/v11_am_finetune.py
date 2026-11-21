#!/usr/bin/env python3
"""v11: fine-tune Sakhiya-V1's acoustic model (v5_c1) on the mined data with SPEAKER SLOTS, keeping the Sakhiya voice.

  * every batch: (1 - p_mined) Sakhiya clips (speaker 0, studio flag 1) + p_mined mined clips (speaker slots 1..15, studio 0);
    the speaker-adversarial head (lambda_grl) pushes the text encoder to carry expression/pronunciation, not identity, so
    what is learned from other voices transfers to Sakhiya (slot 0) at inference;
  * style ids are trained on the SAME ids the runtime uses (news 0, conversational 1, storytelling 2, expressive 4,
    vedic 5). V1's corpus only ever used 0 / 2 / 5, so runtime styles 1 and 4 ran on extrapolated vectors.
    Sakhiya clips are relabelled from their corpus tag (CONV->1, BOOK/BB->2, HAPPY/SURPRISE/...->4, NEWS/WIKI/...->0, VAG->5);
  * decoder (vel) trains at vel_lr_mult x lr; CTC pronunciation loss on for the first ctc_frac of the run;
  * hard --max-minutes cap (<= 110), step-50 throughput gate, EMA checkpoints, best checkpoint kept by Sakhiya-val CFM
    (the voice we ship), plus the last checkpoint."""
import argparse, copy, json, math, os, random, sys, time
from collections import Counter
import torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, REPO)
from sakhiya.cfm import SakhiyaAM  # noqa: E402
from sakhiya.ctc_aligner import MelCTCAligner  # noqa: E402
from sakhiya.symbols import LANG2ID  # noqa: E402
from scripts.train_v3_am import _batch_prosody  # noqa: E402
R = "/work/tts"
TAG2STY = {"CONV": 1, "BOOK": 2, "BB": 2, "HAPPY": 4, "SURPRISE": 4, "FEAR": 4, "ANGER": 4, "SAD": 4, "DISGUST": 4, "UMANG": 4,
           "NEWS": 0, "WIKI": 0, "NAMES": 0, "ALEXA": 0, "DIGI": 0}


def sak_style(x):
    t = str(x.get("style", ""))
    if t.startswith("VAG") or x.get("lang") == "sa": return 5
    return TAG2STY.get(t, int(x.get("v3_style_id", 0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--am_init", default=f"{R}/v11/in/v5_c1/am.pt")
    ap.add_argument("--sak", default=f"{R}/v11/in/feats/v6_expressive_vedic_24k.pt")
    ap.add_argument("--mined", default=f"{R}/v11/v11_mined_24k.pt")
    ap.add_argument("--ctc", default=f"{R}/v11/in/feats/v31_ctc_recognizer.pt")
    ap.add_argument("--out", default=f"{R}/v11/am_ft")
    ap.add_argument("--max-minutes", type=float, default=60.0)
    ap.add_argument("--bs", type=int, default=24); ap.add_argument("--p_mined", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1.2e-5); ap.add_argument("--vel_lr_mult", type=float, default=0.3)
    ap.add_argument("--lambda_grl", type=float, default=0.1); ap.add_argument("--ctc_frac", type=float, default=0.6)
    ap.add_argument("--ema", type=float, default=0.002); ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args(); assert a.max_minutes <= 110, "stages are capped under 2 h"
    dev = "cuda"; os.makedirs(a.out, exist_ok=True); random.seed(0); torch.manual_seed(0)
    ck = torch.load(a.am_init, map_location=dev, weights_only=False); mean, std = float(ck["mean"]), float(ck["std"])
    S = torch.load(a.sak, weights_only=False); M = torch.load(a.mined, weights_only=False)
    sak = [x for x in S["items"] if len(x["ids"]) >= 3]; mined = list(M["items"])
    for x in sak: x["spk"], x["studio"], x["sty_t"] = 0, 1, sak_style(x)
    for x in mined: x["studio"], x["sty_t"] = 0, int(x["v3_style_id"])
    for x in sak + mined: x["mel"] = x["mel"].to(dev)
    tr_s = [x for x in sak if x.get("split") == "train"]; va_s = [x for x in sak if x.get("split") == "val"][:96]
    tr_m = [x for x in mined if x.get("split") == "train"]; va_m = [x for x in mined if x.get("split") == "val"][:96]
    tr_m_ne = [x for x in tr_m if x["lang"] == "ne"]; tr_m_sa = [x for x in tr_m if x["lang"] == "sa"]
    print(f"[v11-ft] sakhiya train={len(tr_s)} val={len(va_s)} styles={dict(Counter(x['sty_t'] for x in tr_s))} | mined train={len(tr_m)} (ne {len(tr_m_ne)}, sa {len(tr_m_sa)}) val={len(va_m)} styles={dict(Counter(x['sty_t'] for x in tr_m))}", flush=True)
    net = SakhiyaAM().to(dev); net.stats.copy_(torch.tensor([mean, std]))
    net.load_state_dict({k: v.float() if torch.is_tensor(v) else v for k, v in ck.get("ema", ck).items()}, strict=False)
    ctc = MelCTCAligner().to(dev).train(False); cc = torch.load(a.ctc, map_location=dev, weights_only=False); ctc.load_state_dict(cc.get("model", cc))
    for p in ctc.parameters(): p.requires_grad_(False)
    net.ctc_rec = ctc
    ema = copy.deepcopy(net).train(False)
    vel_p = [p for n, p in net.named_parameters() if n.startswith("vel.") and p.requires_grad]
    oth_p = [p for n, p in net.named_parameters() if not n.startswith("vel.") and not n.startswith("ctc_rec.") and p.requires_grad]
    opt = torch.optim.AdamW([dict(params=oth_p, lr=a.lr), dict(params=vel_p, lr=a.lr * a.vel_lr_mult)], betas=(0.9, 0.99), weight_decay=1e-2)
    base = [a.lr, a.lr * a.vel_lr_mult]

    def batch(xs):
        L = max(len(x["ids"]) for x in xs); T = max(x["mel"].shape[1] for x in xs); B = len(xs)
        ids = torch.zeros(B, L, dtype=torch.long, device=dev); dur = torch.zeros(B, L, dtype=torch.long, device=dev); y = torch.zeros(B, 100, T, device=dev)
        for k, x in enumerate(xs):
            Lk, Tm = len(x["ids"]), x["mel"].shape[1]
            ids[k, :Lk] = x["ids"].to(dev); dur[k, :Lk] = x["dur_ctc"].to(dev).long(); y[k, :, :Tm] = (x["mel"].float() - mean) / std
        il = torch.tensor([len(x["ids"]) for x in xs], device=dev); yl = torch.tensor([x["mel"].shape[1] for x in xs], device=dev)
        spk = torch.tensor([int(x["spk"]) for x in xs], device=dev); studio = torch.tensor([int(x["studio"]) for x in xs], device=dev)
        lang = torch.tensor([0 if (x["lang"] == "sa" and random.random() < 0.85) else LANG2ID[x["lang"]] for x in xs], device=dev)
        sty = torch.tensor([int(x["sty_t"]) for x in xs], device=dev)
        return ids, il, y, yl, spk, lang, sty, studio, _batch_prosody(y, yl, il), dur

    def score_val(model, xs):
        tot, n = {}, 0
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(xs), 24):
                ids, il, y, yl, spk, lang, sty, studio, pros, dur = batch(xs[i:i + 24])
                torch.manual_seed(1234 + i)
                _, parts = model.loss(ids, il, y, yl, spk, lang, sty, studio, None, None, pros, dur, spk_true=spk, lambda_ctc=0.0, lambda_grl=0.0)
                for k, v in parts.items(): tot[k] = tot.get(k, 0.0) + float(v)
                n += 1
        return {k: round(v / max(1, n), 4) for k, v in tot.items()}

    v0s, v0m = score_val(ema, va_s), score_val(ema, va_m)
    print(f"[v11-ft] V1 start  val_sakhiya={v0s}  val_mined={v0m}", flush=True)
    best = v0s.get("cfm", 9.9); t0, budget, step = time.time(), a.max_minutes * 60.0, 0
    n_m = int(round(a.bs * a.p_mined)); hist = []
    while time.time() - t0 < budget:
        frac = (time.time() - t0) / budget; step += 1
        sc = min(1.0, step / 200.0) * (0.5 * (1 + math.cos(math.pi * frac)) * 0.85 + 0.15)
        for g_, b in zip(opt.param_groups, base): g_["lr"] = b * sc
        if step == 50:
            sps = 50.0 / (time.time() - t0)
            print(f"[GATE] step-50 throughput {sps:.2f} steps/s -> ~{sps * budget / 1000:.1f}k steps inside the {a.max_minutes:.0f} min cap", flush=True)
            if sps < 1.5: print("[GATE] too slow, aborting", flush=True); sys.exit(3)
        n_sa = min(len(tr_m_sa), max(1, n_m // 4)) if tr_m_sa else 0
        xs = random.sample(tr_s, a.bs - n_m) + random.sample(tr_m_ne, n_m - n_sa) + (random.sample(tr_m_sa, n_sa) if n_sa else [])
        ids, il, y, yl, spk, lang, sty, studio, pros, dur = batch(xs)
        lam_ctc = 0.24 if frac < a.ctc_frac else 0.0
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, parts = net.loss(ids, il, y, yl, spk, lang, sty, studio, None, None, pros, dur, spk_true=spk, lambda_ctc=lam_ctc, lambda_grl=a.lambda_grl)
        opt.zero_grad(set_to_none=True); loss.backward()
        if torch.isfinite(torch.nn.utils.clip_grad_norm_(oth_p + vel_p, 0.8)): opt.step()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), net.parameters()): pe.lerp_(p, a.ema)
        if step % 100 == 0:
            print(f"[v11-ft] step={step} {step / (time.time() - t0):.1f} st/s t={(time.time() - t0) / 60:.1f}m lr={opt.param_groups[0]['lr']:.2e} ctc={lam_ctc} " + " ".join(f"{k}={float(v):.3f}" for k, v in parts.items()), flush=True)
        done_smoke = a.smoke and step >= a.smoke
        if step % 1000 == 0 or done_smoke:
            vs, vm = score_val(ema, va_s), score_val(ema, va_m); hist.append(dict(step=step, sak=vs, mined=vm))
            sd = {k: v for k, v in ema.state_dict().items() if not k.startswith("ctc_rec.")}
            ckd = dict(ema=sd, step=step, mean=mean, std=std, pre_baked_v25=True, version="11.0", val_sakhiya=vs, val_mined=vm)
            torch.save(ckd, f"{a.out}/last.pt.tmp"); os.replace(f"{a.out}/last.pt.tmp", f"{a.out}/last.pt")
            if vs.get("cfm", 9.9) <= best:
                best = vs.get("cfm", 9.9); torch.save(ckd, f"{a.out}/best.pt.tmp"); os.replace(f"{a.out}/best.pt.tmp", f"{a.out}/best.pt")
            print(f"[v11-ft] SCORE step={step} val_sakhiya={vs} val_mined={vm} best_sak_cfm={best:.4f} (V1 {v0s.get('cfm')})", flush=True)
            if done_smoke: break
    json.dump(dict(v1_val_sakhiya=v0s, v1_val_mined=v0m, hist=hist, steps=step, best_sak_cfm=best, args=vars(a)), open(f"{a.out}/summary.json", "w"), indent=1)
    print(f"[v11-ft] DONE steps={step} best_sak_cfm={best:.4f} V1 cfm={v0s.get('cfm')}", flush=True)


if __name__ == "__main__":
    main()
