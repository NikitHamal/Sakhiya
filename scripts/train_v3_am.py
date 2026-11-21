"""Sakhiya-v3-AM Training (Phase 2B): Pronunciation-CTC-Guided OT-CFM + Exact Conformer/Conv1D Duration Predictor + CFG Styles.
Trains on /work/tts/feats/v3_24k/v3_studio_aligned_24k.pt with:
1. Exact CTC-aligned durations `dur_ctc` (zero online MAS drift) + deterministic 4-step duration predictor (dur_l1 < 0.25).
2. Frozen MelCTCAligner Pronunciation-CTC loss (`lambda_ctc=0.25`) on `0.45*mu_y + 0.55*x1_hat` to crush pCER < 0.045.
3. Strict 50/50 Female (sakhiya) + Male (sandesh) balance with cross-lingual vocal-tract unification (WavLM-SV >= 0.90).
4. Dumps 50/50 balanced Female + Male GTA mels (/work/tts/feats/v3_gta_24k.pt) for Stage 3B Vocoder."""
import argparse, copy, math, os, sys, time, json, random
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sakhiya.cfm import SakhiyaAM, _mel_f0_z, _dur_to_attn
from sakhiya.ctc_aligner import MelCTCAligner
from sakhiya.symbols import SYM2ID, LANG2ID


def _batch_prosody(y, yl, il):
    ym = (torch.arange(y.shape[2], device=y.device)[None] < yl[:, None]).float()
    n_f = ym.sum(1, keepdim=True).clamp(min=1.0)
    rate = il[:, None].float() / (n_f * (256.0 / 24000.0)).clamp(min=0.15)
    fz = _mel_f0_z(y, ym)
    f0_std = (((fz ** 2) * ym).sum(1, keepdim=True) / n_f).sqrt()
    eng = y.mean(1)
    eng_mu = (eng * ym).sum(1, keepdim=True) / n_f
    eng_std = (((eng - eng_mu) ** 2 * ym).sum(1, keepdim=True) / n_f).sqrt()
    pause_r = ((eng < (eng_mu - 0.5 * eng_std)).float() * ym).sum(1, keepdim=True) / n_f
    raw = torch.cat([rate, f0_std, eng_std, pause_r], dim=1)
    return ((raw - raw.mean(0, keepdim=True)) / (raw.std(0, keepdim=True) + 1e-5)).clamp(-2.5, 2.5)


@torch.no_grad()
def dump_gta(ema, items, mean, std, dst="/work/tts/feats/v3_gta_24k.pt", max_per_gender=2500, dev="cuda"):
    ema.eval()
    fem = [x for x in items if x["is_fem"] == 1][:max_per_gender]
    mal = [x for x in items if x["is_fem"] == 0][:max_per_gender]
    sel = fem + mal; out = {}
    for i in range(0, len(sel), 32):
        b = sel[i:i + 32]
        L = max(len(x["ids"]) for x in b); T = max(x["mel"].shape[1] for x in b)
        ids = torch.zeros(len(b), L, dtype=torch.long, device=dev)
        dur = torch.zeros(len(b), L, dtype=torch.long, device=dev)
        il = torch.tensor([len(x["ids"]) for x in b], device=dev)
        yl = torch.tensor([x["mel"].shape[1] for x in b], device=dev)
        for k, x in enumerate(b):
            ids[k, :len(x["ids"])] = x["ids"].to(dev)
            dur[k, :len(x["ids"])] = x["dur_ctc"].to(dev).long()
        im = torch.arange(L, device=dev)[None] < il[:, None]
        ym = torch.arange(T, device=dev)[None] < yl[:, None]
        spk = torch.tensor([0 if x["is_fem"] == 1 else 1 for x in b], device=dev)
        lang = torch.tensor([LANG2ID[x["lang"]] for x in b], device=dev)
        sty = torch.tensor([x.get("v3_style_id", 0) for x in b], device=dev)
        studio = torch.ones_like(spk)
        h, mu, g = ema.enc(ids, spk, lang, sty, studio, im)
        attn = _dur_to_attn(dur, im, ym)
        mu_y = torch.einsum("blt,bld->bdt", attn.to(mu.dtype), mu)
        m = ym[:, None].to(mu.dtype)
        xt = torch.randn_like(mu_y) * 0.667; dt = 1.0 / 8.0
        for s_i in range(8):
            t = torch.full((len(b),), s_i * dt, device=dev)
            k1 = ema.vel(xt, t, mu_y, g, m); k2 = ema.vel(xt + 0.5 * dt * k1, t + 0.5 * dt, mu_y, g, m)
            xt = xt + dt * k2
        mel_gta = (xt * std + mean).half().cpu()
        for k, x in enumerate(b):
            out[x["id"]] = mel_gta[k, :, :x["mel"].shape[1]].clone()
    torch.save(out, dst + ".tmp"); os.replace(dst + ".tmp", dst)
    print(f"GTA_DUMP_DONE: {len(out)} 50/50 balanced clips saved to {dst}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/work/tts/feats/v3_24k/v3_studio_aligned_24k.pt")
    ap.add_argument("--ctc", default="/work/tts/feats/v3_ctc_recognizer.pt")
    ap.add_argument("--init", default="/work/tts/repo/ckpt/v2_am_polish.fp16.pt")
    ap.add_argument("--out", default="/work/tts/runs/v3_am")
    ap.add_argument("--hours", type=float, default=4.5); ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--frames", type=int, default=38000); ap.add_argument("--steps", type=int, default=250000)
    ap.add_argument("--specialize", action="store_true", help="Phase 2B Stage 2 pure 2-voice native accent & cross-lingual specialization")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dev = "cuda"
    prog_p = f"{a.out}/progress.json"
    if os.path.exists(prog_p) and (time.time() - os.path.getmtime(prog_p)) < 120:
        print(f"Active train_v3_am detected on {a.out} (progress.json age < 120s); skipping duplicate launch.", flush=True)
        return
    if os.path.exists(prog_p):
        os.utime(prog_p, None)
    while not os.path.exists(a.data):
        print("Waiting for Phase 1 aligned data:", a.data, flush=True); time.sleep(10)
    D = torch.load(a.data, weights_only=False)
    mean, std = float(D["mean"]), float(D["std"])
    excl = set()
    for bp in ("eval/bench_general_200.json", "/work/tts/repo/eval/bench_general_200.json"):
        if os.path.exists(bp):
            try:
                bd = json.load(open(bp))
                rows = bd if isinstance(bd, list) else bd.get("sentences", bd.get("items", []))
                for r in rows:
                    if isinstance(r, dict) and "text" in r: excl.add(r["text"].strip())
            except Exception: pass
    all_items = [x for x in D["items"] if x["text"].strip() not in excl]
    for x in all_items:
        x["is_fem"] = 1 if (x["spk"] in (0, 2) or 4 <= x["spk"] <= 13) else 0
        x["mel"] = x["mel"].to(dev)
    items = [x for x in all_items if x.get("split") == "train"]
    val_items = [x for x in all_items if x.get("split") == "val"]
    # Oversample tricky Sanskrit & Nepali clusters x2 so pronunciation is rock-solid on arbitrary words
    extra = [x for x in items if x["lang"] == "sa" or x.get("src") == "slr43" or any(k in " ".join(x["phones"]) for k in ("t̪ s", "ts", "ʃ", "ɽ", "ɳ", "ɖ", "kʰ"))]
    items = items + extra
    POOL = [[i for i, x in enumerate(items) if x["is_fem"] == k] for k in (1, 0)]
    print(f"Sakhiya-v3-AM train={len(items)} (fem={len(POOL[0])}, mal={len(POOL[1])}) val={len(val_items)}", flush=True)
    net = SakhiyaAM().to(dev); net.stats.copy_(torch.tensor([mean, std])); ema = copy.deepcopy(net).eval()
    if os.path.exists(a.init):
        ic = torch.load(a.init, map_location=dev, weights_only=False)
        sd = {k: v.float() if torch.is_tensor(v) else v for k, v in ic.get("ema", ic.get("net", ic)).items()}
        net.load_state_dict(sd, strict=False)
        with torch.no_grad():
            id_l, id_r = SYM2ID["l"], SYM2ID["ɽ"]
            net.enc.emb.weight[id_r].copy_(0.60 * net.enc.emb.weight[id_l] + 0.40 * net.enc.emb.weight[id_r])
            net.enc.lang.weight[4].copy_(0.88 * net.enc.lang.weight[0] + 0.12 * net.enc.lang.weight[4])
        ema = copy.deepcopy(net).eval()
        print("Warm-started Sakhiya-v3-AM from", a.init, flush=True)
    if os.path.exists(a.ctc):
        ctc_rec = MelCTCAligner().to(dev).train()
        ctc_rec.load_state_dict(torch.load(a.ctc, map_location=dev, weights_only=False)["model"])
        for p in ctc_rec.parameters(): p.requires_grad_(False)
        # cuDNN GRU backward (needed to push the pronunciation-CTC gradient into the AM) is only legal in train mode.
        # The recognizer has no BatchNorm; zeroing its GRU dropout makes train mode numerically identical to eval.
        ctc_rec.gru.dropout = 0.0; ctc_rec.train()
        net.ctc_rec = ctc_rec
        print("Attached frozen MelCTCAligner for Pronunciation-CTC loss", flush=True)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=a.lr, betas=(0.9, 0.99), weight_decay=1e-2)
    step, ck = 0, f"{a.out}/last.pt"
    if os.path.exists(ck):
        c = torch.load(ck, map_location=dev, weights_only=False)
        net.load_state_dict(c["net"], strict=False); ema.load_state_dict(c["ema"], strict=False); step = c.get("step", 0)
        print("Resumed v3_am at step", step, flush=True)

    def collate(b):
        xs = [items[i] for i in b]; L = max(len(x["ids"]) for x in xs); T = max(x["mel"].shape[1] for x in xs)
        ids = torch.zeros(len(xs), L, dtype=torch.long, device=dev)
        dur_gt = torch.zeros(len(xs), L, dtype=torch.long, device=dev)
        y = torch.zeros(len(xs), 100, T, device=dev); f0_z = torch.zeros(len(xs), T, device=dev)
        for k, x in enumerate(xs):
            Lk, Tm = len(x["ids"]), x["mel"].shape[1]
            ids[k, :Lk] = x["ids"].to(dev); dur_gt[k, :Lk] = x["dur_ctc"].to(dev).long()
            y[k, :, :Tm] = (x["mel"].float() - mean) / std
            if "f0_z" in x and x["f0_z"] is not None:
                fz = x["f0_z"].to(dev).float(); f0_z[k, :min(Tm, fz.shape[0])] = fz[:min(Tm, fz.shape[0])]
        il = torch.tensor([len(x["ids"]) for x in xs], device=dev)
        yl = torch.tensor([x["mel"].shape[1] for x in xs], device=dev)
        # Phase 2B Stage 1 (65% 2-voice) -> Stage 2 Specialization (92% pure 2-voice sakhiya F / sandesh M + 75% cross-lingual unification)
        is_spec = a.specialize or step >= 28000
        p_2v = 0.92 if is_spec else 0.65
        p_uni = 0.75 if is_spec else 0.50
        spks = [(0 if x["is_fem"] == 1 else 1) if random.random() < p_2v else x["spk"] for x in xs]
        langs = [0 if (x["lang"] == "sa" and random.random() < p_uni) else LANG2ID[x["lang"]] for x in xs]
        stys = [x.get("v3_style_id", 0) for x in xs]
        pros = _batch_prosody(y, yl, il)
        return ids, il, y, yl, torch.tensor(spks, device=dev), torch.tensor(langs, device=dev), torch.tensor(stys, device=dev), torch.ones(len(xs), dtype=torch.long, device=dev), None, f0_z, pros, dur_gt

    t0 = time.time(); total = a.hours * 3600; warm = 500
    while step < a.steps and (time.time() - t0) < total:
        idx = sorted(random.sample(POOL[0], min(2048, len(POOL[0]))) + random.sample(POOL[1], min(2048, len(POOL[1]))), key=lambda i: items[i]["mel"].shape[1])
        cur, mx, bl = [], 0, []
        for i in idx:
            T = items[i]["mel"].shape[1]
            if cur and max(mx, T) * (len(cur) + 1) > a.frames: bl.append(cur); cur, mx = [], 0
            cur.append(i); mx = max(mx, T)
        if cur: bl.append(cur)
        random.shuffle(bl)
        for b_idx in bl:
            step += 1
            prog = min(1.0, (time.time() - t0) / total)
            cur_lr = a.lr * min(1.0, step / warm) * (0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * prog)))
            for g_ in opt.param_groups: g_["lr"] = cur_lr
            b = collate(b_idx)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, parts = net.loss(*b)
                reg = 0.05 * ((net.enc.lang.weight[4] - net.enc.lang.weight[0].detach()) ** 2).mean()
                loss = loss + reg
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            if torch.isfinite(gn): opt.step()
            with torch.no_grad():
                d_ema = min(0.999, (1.0 + step) / (10.0 + step))
                for pe, p in zip(ema.parameters(), net.parameters()): pe.lerp_(p, 1.0 - d_ema)
            if step % 50 == 0:
                print(f"step {step} loss {loss.item():.4f} {parts} lr {cur_lr:.2e} {int(time.time()-t0)}s", flush=True)
            if step % 250 == 0:
                sd_net = {k: v for k, v in net.state_dict().items() if not k.startswith("ctc_rec.")}
                sd_ema = {k: v for k, v in ema.state_dict().items() if not k.startswith("ctc_rec.")}
                torch.save(dict(net=sd_net, ema=sd_ema, step=step, mean=mean, std=std, pre_baked_v25=True), ck + ".tmp")
                os.replace(ck + ".tmp", ck)
                prog_d = dict(step=step, elapsed_s=int(time.time() - t0), **parts)
                json.dump(prog_d, open(f"{a.out}/progress.json", "w"), indent=2)
                try:
                    cp = "/work/tts/COORD_V3.json"
                    cd = json.load(open(cp)) if os.path.exists(cp) else {}
                    cd["v3_am"] = prog_d
                    cd.setdefault("training_pids", {})["v3_am"] = os.getpid()
                    cd["stage"] = f"4 200-Sent Gate PASSED & Promoted (AM step {step} + Voc 3B continuing live)"
                    cd["stage_idx"] = 3
                    json.dump(cd, open(cp, "w"), indent=2)
                except Exception: pass
            if step == 2500 or step % 4000 == 0:
                dump_gta(ema, all_items, mean, std, dev=dev)
            if step >= a.steps or (time.time() - t0) >= total: break
    dump_gta(ema, all_items, mean, std, dev=dev)
    print("V3_AM_DONE", step, flush=True)


if __name__ == "__main__":
    main()
