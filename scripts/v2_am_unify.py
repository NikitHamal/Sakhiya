"""Track A: Targeted Cross-Lingual Single-Voice Unification Fine-Tune on finished Sakhiya-v2-AM (bf16, EMA 0.999).
Decouples vocal-tract timbre from language (`g_vel` in Velocity has zero lang=sa offset; TextEncoder uses bounded cadence residual),
anchors `sakhiya` (spk=0) to native Nepal female prosody (SLR43 + FLEURS ne_np + Rasa Nepali Female), and preserves clean studio
Sanskrit articulation & Vedic svara (`pitch_add`) strictly through the `sakhiya` Nepali female vocal tract.
After fine-tuning, dumps unified GTA mels (`/work/tts/feats/v2_gta_unified.pt`) and sets `gta_unified_ready=True` in `/work/tts/COORD.json`.
usage: python3 scripts/v2_am_unify.py --init /work/tts/runs/v2_am_D/last.pt --out /work/tts/runs/v2_am_uni --minutes 32"""
import argparse, copy, math, os, sys, time, json, random, subprocess
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sakhiya.cfm import SakhiyaAM
from sakhiya.symbols import SYM2ID, LANG2ID


def coord(**kw):
    try:
        p = "/work/tts/COORD.json"; c = json.load(open(p))
        c.setdefault("track_a_uni", {}).update(kw, updated=time.time())
        if "gta_unified_ready" in kw:
            c["gta_unified_ready"] = kw["gta_unified_ready"]
        json.dump(c, open(p, "w"), indent=1)
    except Exception as e:
        print("coord", e, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/work/tts/feats/v2_studio_24k.pt")
    ap.add_argument("--native", default="/work/tts/feats/v2_native_24k.pt")
    ap.add_argument("--init", default="/work/tts/runs/v2_am_D/last.pt")
    ap.add_argument("--out", default="/work/tts/runs/v2_am_uni")
    ap.add_argument("--minutes", type=float, default=32.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lr_min", type=float, default=5e-6)
    ap.add_argument("--frames", type=int, default=48000)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--gta_out", default="/work/tts/feats/v2_gta_unified.pt")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); dev = "cuda"

    D = torch.load(a.data, weights_only=False)
    items = [x for x in D["items"] if x["split"] == "train" and (x["spk"] in (0, 2) or (4 <= x["spk"] <= 13))]
    if os.path.exists(a.native):
        N_D = torch.load(a.native, weights_only=False)
        n_tr = [x for x in N_D.get("items", []) if x.get("split") == "train" and (x["spk"] == 0 or (4 <= x["spk"] <= 13))]
        items.extend(n_tr * 4)
        print(f"Loaded {len(n_tr)} pure Nepal-native 24kHz female clips (x4) from {a.native}", flush=True)

    for x in items:
        ph = [q for p in x["phones"] for q in (["o", "m"] if p == "<om>" else [p])]
        val_idx = [i for i, p in enumerate(ph) if p in SYM2ID]
        x["ids"] = torch.tensor([SYM2ID[ph[i]] for i in val_idx], dtype=torch.long)
        po = x.get("pitch_off")
        if po is not None and len(po) >= len(ph):
            x["po"] = torch.tensor([float(po[i]) for i in val_idx], dtype=torch.float32)
    items = [x for x in items if 3 <= len(x["ids"]) and x["mel"].shape[1] <= 1200 and x["mel"].shape[1] > len(x["ids"])]

    ck_path = f"{a.out}/last.pt" if os.path.exists(f"{a.out}/last.pt") else a.init
    c = torch.load(ck_path, map_location=dev, weights_only=False)
    mean = float(c.get("mean", -6.177)); std = float(c.get("std", 2.492))
    net = SakhiyaAM().to(dev); net.stats.copy_(torch.tensor([mean, std], device=dev))
    sd = c.get("ema", c.get("net", c))
    net.load_state_dict(sd)
    if ck_path == a.init:
        # Strip memorized Rasa Sanskrit speaker offset from lang[4] at warm-start so lang[4] starts aligned with lang[0]
        with torch.no_grad():
            w_l = net.enc.lang.weight
            w_l[4].copy_(0.85 * w_l[0] + 0.15 * w_l[4])
    ema = copy.deepcopy(net).eval()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=1e-2)
    step = int(c.get("uni_step", 0)) if ck_path != a.init else 0
    init_step = int(c.get("step", 0))
    print(f"Warm-started Sakhiya-v2 Unification from {ck_path} (base step {init_step}, uni_step {step}) on {len(items)} clips", flush=True)

    def batches():
        while True:
            idx = sorted(random.sample(range(len(items)), min(len(items), 3072)), key=lambda i: items[i]["mel"].shape[1])
            cur, mx, bl = [], 0, []
            for i in idx:
                T = items[i]["mel"].shape[1]
                if cur and max(mx, T) * (len(cur) + 1) > a.frames:
                    bl.append(cur); cur, mx = [], 0
                cur.append(i); mx = max(mx, T)
            if cur: bl.append(cur)
            random.shuffle(bl)
            for b in bl: yield b

    def collate(b):
        xs = [items[i] for i in b]; L = max(len(x["ids"]) for x in xs); T = max(x["mel"].shape[1] for x in xs)
        ids = torch.zeros(len(xs), L, dtype=torch.long, device=dev)
        y = torch.zeros(len(xs), 100, T, device=dev)
        f0_z = torch.zeros(len(xs), T, device=dev); po = torch.zeros(len(xs), L, device=dev); has_po = False
        for k, x in enumerate(xs):
            ids[k, :len(x["ids"])] = x["ids"].to(dev)
            Tm = x["mel"].shape[1]
            y[k, :, :Tm] = (x["mel"].to(dev).float() - mean) / std
            if "f0_z" in x and x["f0_z"] is not None:
                fz = x["f0_z"].float().to(dev)
                f0_z[k, :min(Tm, fz.shape[0])] = fz[:min(Tm, fz.shape[0])]
            if "po" in x and x["po"] is not None:
                p = x["po"].to(dev); po[k, :min(L, p.shape[0])] = p[:min(L, p.shape[0])]; has_po = True
        t = lambda f: torch.tensor([f(x) for x in xs], device=dev)
        # Route 60% of native Nepal SLR43/FLEURS female clips through spk=0 (`sakhiya`), keep true spk=2 on Sanskrit audio so spk_emb (not lang_emb) explains actress timbre
        spks = [0 if (4 <= x["spk"] <= 13 and random.random() < 0.6) else x["spk"] for x in xs]
        # Random language-id dropout (50% of Sanskrit clips use lang=0 in TextEncoder) forces phoneme embeddings to share the unified acoustic space cleanly
        langs = [0 if (x["lang"] == "sa" and random.random() < 0.5) else LANG2ID[x["lang"]] for x in xs]
        return (ids, t(lambda x: len(x["ids"])), y, t(lambda x: x["mel"].shape[1]), torch.tensor(spks, device=dev),
                torch.tensor(langs, device=dev), t(lambda x: x["style_id"]), t(lambda x: x["is_studio"]),
                po if has_po else None, f0_z if any("f0_z" in x for x in xs) else None)

    def save(tag="last"):
        torch.save(dict(net=net.state_dict(), ema=ema.state_dict(), step=init_step + step, uni_step=step,
                        stage="U", pitch_trained=True, mean=mean, std=std, styles=D["styles"]), f"{a.out}/{tag}.pt.tmp")
        os.replace(f"{a.out}/{tag}.pt.tmp", f"{a.out}/{tag}.pt")

    t0 = time.time(); tl = time.time(); tsave = time.time(); gen = batches(); warm = 150
    total = a.minutes * 60.0; hist = []
    coord(pids=[os.getpid()], stage=f"Stage2U: Single-Voice Unification Fine-Tune (base {init_step})", gta_unified_ready=False)
    while step < a.steps and (time.time() - t0) < total:
        prog = min(1.0, (time.time() - t0) / max(total, 1.0))
        cur_lr = a.lr_min + (a.lr - a.lr_min) * 0.5 * (1.0 + math.cos(math.pi * prog))
        cur_lr *= min(1.0, (step + 1) / warm)
        for g in opt.param_groups: g["lr"] = cur_lr
        b = collate(next(gen))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, parts = net.loss(*b)
            reg = 0.05 * ((net.enc.lang.weight[4] - net.enc.lang.weight[0].detach()) ** 2).mean()
            loss = loss + reg
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        if torch.isfinite(gn): opt.step()
        step += 1
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), net.parameters()): pe.lerp_(p, 1.0 - 0.999)
        hist.append(loss.item())
        if step % 50 == 0:
            vr = torch.cuda.max_memory_allocated() / 2**30
            print(f"uni_step {step} loss {sum(hist)/len(hist):.4f} {parts} lr {cur_lr:.2e} vram {vr:.1f}G {int(time.time()-tl)}s", flush=True)
            hist = []; tl = time.time()
            coord(vram_gb=round(torch.cuda.memory_reserved() / 2**30, 1), step=init_step + step, uni_step=step)
        if step % 250 == 0 or (time.time() - tsave) > 180:
            save(); tsave = time.time(); print("saved unified ckpt at uni_step", step, flush=True)

    save(); save("unified_final")
    print(f"UNIFY_DONE uni_step={step}, dumping unified GTA mels -> {a.gta_out}", flush=True)
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "v2_dump_gta.py"),
                    "--ckpt", f"{a.out}/last.pt", "--out", a.gta_out, "--stage_a_soup", "0.25"], check=True)
    coord(stage=f"Stage2U done (uni_step {step}) + feats/v2_gta_unified.pt ready", pids=[], gta_unified_ready=True)
    print("GTA_UNIFIED_READY", a.gta_out, flush=True)


if __name__ == "__main__":
    main()
