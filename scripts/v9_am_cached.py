#!/usr/bin/env python3
"""v9 acoustic-student distillation with CACHED teacher targets (fixes the 2.5 steps/s bottleneck of v8).

The teacher flow-ODE input is a fixed deterministic pattern, so the teacher output for an item is a pure function of
(ids, pitch offsets, durations, style, lang). We therefore run the teacher ONCE over the whole training pool, keep the
targets (mel, per-token prior mu, F0 head, energy head) in GPU memory (98 GB card), and the student loop only runs the
small student. Hard wall-clock cap (--max-minutes <= 150), step-50 throughput gate, time-based cosine LR, plateau-safe.
"""
import argparse, copy, json, math, os, random, sys, time
import torch, torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

R = "/work/tts"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from sakhiya.ctc_aligner import MelCTCAligner  # noqa: E402
from sakhiya.cfm import SakhiyaAM, STYLE_PROSODY_TARGETS  # noqa: E402
from sakhiya.symbols import LANG2ID  # noqa: E402
from sakhiya import v6_distill as V  # noqa: E402
from scripts.export_v2_onnx import TIER_CFG  # noqa: E402
from scripts.train_nano_atom import warm_slice, _bake  # noqa: E402

dev = "cuda"


@torch.no_grad()
def build_cache(items, t_am, ctc_unused, mean, std):
    c_idx = torch.arange(100, device=dev, dtype=torch.float32)[None, :, None]
    pros_tbl = torch.tensor(STYLE_PROSODY_TARGETS, dtype=torch.float32, device=dev)
    order = sorted(range(len(items)), key=lambda i: int(items[i]["dur_ctc"].sum()))
    out = [None] * len(items)
    t0 = time.time()
    for bi in range(0, len(order), 64):
        idx = order[bi:bi + 64]; xs = [items[i] for i in idx]; B = len(xs)
        L = max(len(x["ids"]) for x in xs); T = min(max(int(x["dur_ctc"].sum()) for x in xs), 850)
        ids = torch.zeros(B, L, dtype=torch.long, device=dev); durs = torch.zeros(B, L, dtype=torch.long, device=dev)
        po = torch.zeros(B, L, device=dev)
        for k, x in enumerate(xs):
            Lk = len(x["ids"]); ids[k, :Lk] = x["ids"].to(dev); durs[k, :Lk] = x["dur_ctc"].to(dev).long()
            if x.get("po") is not None: po[k, :Lk] = x["po"][:Lk].to(dev)
        il = torch.tensor([len(x["ids"]) for x in xs], device=dev); yl = durs.sum(1).clamp(max=T)
        im = torch.arange(L, device=dev)[None] < il[:, None]; ym = torch.arange(T, device=dev)[None] < yl[:, None]; ymf = ym[:, None].float()
        langs = torch.tensor([0 if (x["lang"] == "sa" and random.random() < 0.85) else LANG2ID.get(x["lang"], 0) for x in xs], device=dev)
        stys = torch.tensor([int(x.get("v3_style_id", 0)) for x in xs], device=dev).clamp(0, 5)
        spks = torch.zeros(B, dtype=torch.long, device=dev); stud = torch.ones(B, dtype=torch.long, device=dev)
        ones_m = torch.ones((B, 1), device=dev)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            h_t, mu_t, g_t = t_am.enc(ids, spks, langs, stys, stud, im, pitch_add=po, prosody=pros_tbl[stys], sty_mask=ones_m)
            f0_t = t_am.enc.f0_pred(h_t); eng_t = t_am.enc.energy_pred(h_t)
            mu_y = V._expand_batch_durs(mu_t, durs, T) * ymf
            t_idx = torch.arange(T, device=dev, dtype=torch.float32)[None, None, :]
            xt = (torch.sin(c_idx * 12.9898 + t_idx * 78.233) * torch.cos(c_idx * 39.346 + t_idx * 11.135) * 0.7071).expand(B, 100, T).to(mu_y.dtype)
            for i_o in range(4):
                to = torch.full((B,), i_o * 0.25, device=dev)
                k1 = t_am.vel(xt, to, mu_y, g_t, ymf)
                xt = xt + 0.25 * t_am.vel(xt + 0.125 * k1, to + 0.125, mu_y, g_t, ymf)
            y_clean = xt * ymf
        for k, i in enumerate(idx):
            Lk = len(xs[k]["ids"]); Tk = int(yl[k])
            out[i] = dict(ids=ids[k, :Lk].clone(), dur=durs[k, :Lk].clone(), po=po[k, :Lk].clone(), lang=int(langs[k]), sty=int(stys[k]),
                          mu=mu_t[k, :Lk].half().clone(), f0=f0_t[k, :Lk].half().clone(), eng=eng_t[k, :Lk].half().clone(),
                          y=y_clean[k, :, :Tk].T.half().contiguous(), w=1.0)
        if (bi // 64) % 40 == 0: print(f"[cache] {bi}/{len(order)} {time.time()-t0:.0f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, choices=["voca", "sano"])
    ap.add_argument("--max-minutes", type=float, default=90.0)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--init", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--teacher_am", default=f"{R}/runs/v6_teacher/am.pt")
    a = ap.parse_args()
    assert a.max_minutes <= 150, "hard cap is 150 minutes"
    cap = a.max_minutes * 60.0
    out_dir = a.out or f"{R}/runs/v9_{a.tier}_am"; os.makedirs(out_dir, exist_ok=True)
    c = TIER_CFG[a.tier]
    D = torch.load(f"{R}/feats/v6_expressive_vedic_24k.pt", weights_only=False)
    mean, std = float(D.get("mean", -5.5028)), float(D.get("std", 2.1869))
    ctc = MelCTCAligner().to(dev).train()
    ck = torch.load(f"{R}/feats/v31_ctc_recognizer.pt", map_location=dev, weights_only=False)
    ctc.load_state_dict(ck.get("model", ck)); [p.requires_grad_(False) for p in ctc.parameters()]
    t_ck = torch.load(a.teacher_am, map_location=dev, weights_only=False)
    t_sd = {k: v.float() for k, v in t_ck.get("ema", t_ck.get("net", t_ck)).items() if not k.startswith("ctc_rec.")}
    t_am = SakhiyaAM().to(dev).eval(); t_am.load_state_dict(t_sd, strict=False); t_am.stats.copy_(torch.tensor([mean, std], device=dev)); _bake(t_am)

    anchors = V.build_teacher_anchor_bank(t_am, mean, std, ctc, dev)
    base = [x for x in D["items"] if x.get("split") == "train" and len(x["ids"]) >= 3 and 24 <= x["mel"].shape[1] <= 850 and int(x["dur_ctc"].sum()) <= 850]
    print(f"[v9-{a.tier}] items={len(base)} anchors={len(anchors)}", flush=True)
    cache_path = f"{R}/runs/v9_teacher_cache.pt"
    if os.path.exists(cache_path):
        cache = torch.load(cache_path, map_location=dev, weights_only=False)
        print("[cache] loaded", len(cache), flush=True)
    else:
        cache = build_cache(base + anchors, t_am, ctc, mean, std)
        for i, x in enumerate(cache):
            x["w"] = 35.0 if i >= len(base) else (3.0 if (base[i].get("v3_style_id", 0) in (2, 5)) else 1.0)
        torch.save(cache, cache_path)
        print("[cache] built", len(cache), flush=True)
    del t_ck, t_sd
    pool_w = [x["w"] for x in cache]
    anchor_idx = list(range(len(base), len(cache)))

    s_am = SakhiyaAM(**{k: c[k] for k in ("d_enc", "L_enc", "ff_enc", "d_vel", "nb_vel", "h_dur")}).to(dev)
    s_am.stats.copy_(torch.tensor([mean, std], device=dev))
    warm_slice(s_am, t_am.state_dict()); _bake(s_am)
    init = a.init or (f"{out_dir}/am.pt" if os.path.exists(f"{out_dir}/am.pt") else f"{R}/runs/v8_{a.tier}_am/am.pt")
    start_step = 1
    if os.path.exists(init):
        ck_s = torch.load(init, map_location=dev, weights_only=False)
        s_am.load_state_dict(ck_s.get("ema", ck_s.get("net", ck_s)), strict=False); start_step = int(ck_s.get("step", 0)) + 1
        print(f"[v9-{a.tier}] warm start from {init} step {start_step}", flush=True)
    ema = copy.deepcopy(s_am).eval()
    opt = torch.optim.AdamW(s_am.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=1e-2)

    t0 = time.time(); step = start_step - 1; ema_loss = None; best = 1e9; flat = 0
    nb = a.batch; na = nb // 4
    while True:
        step += 1; el = time.time() - t0
        if step == start_step + 50:
            sps = 50.0 / max(el, 1e-6)
            print(f"[GATE] step-50 throughput {sps:.2f} steps/s -> {sps*cap/1000:.1f}k steps fit in cap {cap/60:.0f} min", flush=True)
            if sps < 3.0: print("[GATE] ABORT: too slow", flush=True); sys.exit(3)
        if el > cap: print("[v9] wall-clock cap hit, saving", flush=True); break
        frac = el / cap
        lr = max(a.lr * 0.5 * (1 + math.cos(math.pi * frac)), 3e-5)
        for g in opt.param_groups: g["lr"] = lr
        sel = random.choices(range(len(cache)), weights=pool_w, k=nb - na) + random.choices(anchor_idx, k=na)
        xs = [cache[i] for i in sel]; B = len(xs)
        ids = pad_sequence([x["ids"] for x in xs], batch_first=True); L = ids.shape[1]
        durs = pad_sequence([x["dur"] for x in xs], batch_first=True); po = pad_sequence([x["po"] for x in xs], batch_first=True)
        mu_t = pad_sequence([x["mu"] for x in xs], batch_first=True).float()
        f0_t = pad_sequence([x["f0"] for x in xs], batch_first=True).float(); eng_t = pad_sequence([x["eng"] for x in xs], batch_first=True).float()
        y_clean = pad_sequence([x["y"] for x in xs], batch_first=True).float().transpose(1, 2).contiguous(); T = y_clean.shape[2]
        il = torch.tensor([len(x["ids"]) for x in xs], device=dev); yl = torch.tensor([x["y"].shape[0] for x in xs], device=dev)
        im = torch.arange(L, device=dev)[None] < il[:, None]; ym = torch.arange(T, device=dev)[None] < yl[:, None]; ymf = ym[:, None].float()
        langs = torch.tensor([x["lang"] for x in xs], device=dev); stys = torch.tensor([x["sty"] for x in xs], device=dev)
        spks = torch.zeros(B, dtype=torch.long, device=dev); stud = torch.ones(B, dtype=torch.long, device=dev)
        pros_b = torch.tensor(STYLE_PROSODY_TARGETS, dtype=torch.float32, device=dev)[stys]; ones_m = torch.ones((B, 1), device=dev)
        ld_tgt = torch.log(durs.float().clamp(min=1.0)) * im.float()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad(): log_p_t = ctc(y_clean.float())
            h_s, mu_s, g_s = s_am.enc(ids, spks, langs, stys, stud, im, pitch_add=po, prosody=pros_b, sty_mask=ones_m)
            l_heads = 0.5 * (((s_am.enc.f0_pred(h_s) - f0_t) ** 2 + (s_am.enc.energy_pred(h_s) - eng_t) ** 2) * im[:, :, None]).sum() / im.sum().clamp(min=1)
            mu_y_s = V._expand_batch_durs(mu_s, durs, T) * ymf
            l_prior = 2.2 * (((mu_s - mu_t).abs() + (mu_s - mu_t) ** 2) * im[:, :, None]).sum() / (im.sum() * 100.0)
            ld_pred = s_am.dur.pred_logdur(ids, h_s, im)
            l_dur = F.smooth_l1_loss(ld_pred * im.float(), ld_tgt, reduction="sum") / im.sum().clamp(min=1)
            t0_z = torch.zeros((B,), device=dev)
            y_pred = (mu_y_s + s_am.vel(mu_y_s, t0_z, mu_y_s, g_s, ymf)) * ymf
            denom = (ymf.sum() * 100.0).clamp(min=1.0)
            l_mel = ((1.6 * (y_pred - y_clean).abs() + (y_pred - y_clean) ** 2) * ymf).sum() / denom
            df_p = y_pred[:, 1:, :] - y_pred[:, :-1, :]; df_t = y_clean[:, 1:, :] - y_clean[:, :-1, :]
            l_df = ((df_p - df_t).abs() * ymf).sum() / denom
            l_std = ((y_pred.float().std(dim=1, keepdim=True) - y_clean.float().std(dim=1, keepdim=True)).abs() * ymf).sum() / ymf.sum().clamp(min=1.0)
            ym_t = ymf[:, :, 1:] * ymf[:, :, :-1]
            l_dt = (((y_pred[:, :, 1:] - y_pred[:, :, :-1]) - (y_clean[:, :, 1:] - y_clean[:, :, :-1])).abs() * ym_t).sum() / (ym_t.sum() * 100.0 + 1.0)
            cep_p = torch.fft.rfft(y_pred.float(), dim=1).abs()[:, 4:48, :]; cep_t = torch.fft.rfft(y_clean.float(), dim=1).abs()[:, 4:48, :]
            l_cep = ((cep_p - cep_t).abs() * ymf).sum() / (ymf.sum() * 44.0).clamp(min=1.0)
            log_p_s = ctc(y_pred.float())
            l_kl = F.kl_div(log_p_s, log_p_t.exp().detach(), reduction="none").sum(-1)
            l_kl = ((l_kl.transpose(0, 1) if l_kl.shape[0] != ym.shape[0] else l_kl) * ym.float()).sum() / ym.sum().clamp(min=1.0)
            l_ctc = F.relu(ctc.ctc_loss(0.3 * mu_y_s + 0.7 * y_pred, yl, ids, il) - 0.05)
            loss = l_mel + 2.4 * l_df + 0.8 * l_std + 1.2 * l_dt + 0.65 * l_cep + l_prior + l_heads + 0.6 * l_dur + 0.45 * l_kl + 0.25 * l_ctc
        opt.zero_grad(set_to_none=True); loss.backward()
        if torch.isfinite(torch.nn.utils.clip_grad_norm_(s_am.parameters(), 1.0)): opt.step()
        with torch.no_grad():
            d_ema = min(0.999, (1 + (step - start_step)) / (10 + (step - start_step)))
            for pe, p in zip(ema.parameters(), s_am.parameters()): pe.lerp_(p, 1.0 - d_ema)
        lv = float(loss.item()) if step % 25 == 0 else None
        if lv is not None: ema_loss = lv if ema_loss is None else 0.9 * ema_loss + 0.1 * lv
        if step % 500 == 0:
            torch.save(dict(ema=ema.state_dict(), net=s_am.state_dict(), step=step, tier=a.tier, cfg=c, mean=mean, std=std, pre_baked_v25=True, seqkd_1pass=True), f"{out_dir}/am.pt.tmp")
            os.replace(f"{out_dir}/am.pt.tmp", f"{out_dir}/am.pt")
            sps = (step - start_step + 1) / max(time.time() - t0, 1e-6)
            print(f"[{a.tier}] step={step} {sps:.1f} st/s elapsed={el/60:.1f}m lr={lr:.2e} loss={ema_loss:.4f} mel={l_mel.item():.3f} dur={l_dur.item():.3f} ctc={l_ctc.item():.3f} kl={l_kl.item():.3f}", flush=True)
            if ema_loss is not None:
                if ema_loss < best - 1e-3: best = ema_loss; flat = 0
                else: flat += 1
                if flat >= 40 and frac > 0.5: print("[v9] validation flat, stopping early", flush=True); break
    torch.save(dict(ema=ema.state_dict(), net=s_am.state_dict(), step=step, tier=a.tier, cfg=c, mean=mean, std=std, pre_baked_v25=True, seqkd_1pass=True), f"{out_dir}/am.pt")
    print(f"[v9-{a.tier}] DONE step={step}", flush=True)


if __name__ == "__main__":
    main()
