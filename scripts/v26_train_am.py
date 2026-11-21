"""v2.6 Stage 2 (derived from v2_polish.py): 2-VOICE balanced AM on RE-PHONEMIZED data. Changes vs v2.5: (1) every item re-phonemized in memory with the exact
v2.6 front-end (sakhiya.g2p / sanskrit.phonemize_v26: no stale phones, no inference duration hacks); (2) voices unified to 2 ids: sakhiya=spk0 (Rasa ne-F + sa-F + SLR43 F),
sandesh=spk1 (Rasa ne-M + sa-M) with STRICT 50/50 female/male batches; (3) noisy crowd clips (Common Voice / FLEURS) dropped; (4) writes /work/tts/COORD_V26.json.
Original doc: v2.5 Stage 1: pronunciation-precision polish of the converged v2 AM (warm start from the fp16 EMA export, bf16, EMA 0.999).
Oversamples the owner-reported failure clusters: ts / t̪s / ʃts (युयुत्सवः, पाण्डवाश्चैव), ळ->ɽ, ॐ, -er converb, kʰm, visarga echo (ʌ h), retroflex ɳɖ + all Sanskrit x2.
usage: python3 scripts/v2_polish.py --init /work/tts/runs/v2_am/last.pt --out /work/tts/runs/v2_am_polish --hours 0.4"""
import argparse, copy, math, os, sys, time, json, random
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sakhiya.cfm import SakhiyaAM, _mel_f0_z
from sakhiya.symbols import SYM2ID, LANG2ID

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/work/tts/feats/v2_studio_24k.pt"); ap.add_argument("--native", default="/work/tts/feats/v2_native_24k.pt")
ap.add_argument("--out", default="/work/tts/runs/v26_am")
ap.add_argument("--hours", type=float, default=2.5); ap.add_argument("--stage", default="A"); ap.add_argument("--lr", type=float, default=0)
ap.add_argument("--frames", type=int, default=36000, help="max mel frames per batch"); ap.add_argument("--max-vram", type=float, default=40)
ap.add_argument("--steps", type=int, default=120000); ap.add_argument("--init", default="/work/tts/runs/v2_am/last.pt"); ap.add_argument("--boost", type=int, default=4); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); dev = "cuda"
D = torch.load(a.data, weights_only=False); items = [x for x in D["items"] if x["split"] == "train"]
if os.path.exists(a.native):
    N_D = torch.load(a.native, weights_only=False)
    n_tr = [x for x in N_D.get("items", []) if x.get("split") == "train"]
    items.extend([x for x in n_tr if x.get("src") == "slr43"])   # v2.6: drop noisy Common Voice / FLEURS crowd clips
    print(f"Loaded {len(n_tr)} pure Nepal-native 24kHz clips from {a.native}", flush=True)
if False:
    # Stage 2B: Female Native-Nepali (Rasa ne-F spk 0 + SLR43/FLEURS female spk 4..13) + Studio Sanskrit Female (Rasa sa-F spk 2)
    items = [x for x in items if x["spk"] in (0, 2) or (4 <= x["spk"] <= 13)]
from sakhiya import g2p as _g2p, sanskrit as _sa
FEM = lambda x: x["spk"] in (0, 2) or 4 <= x["spk"] <= 13
_chg = 0
for x in items:
    try: ph = _sa.phonemize_v26(x["text"]) if x["lang"] == "sa" else _g2p.text_to_phones(x["text"], "ne")
    except Exception: ph = x["phones"]
    _chg += int(ph != x["phones"]); x["phones"] = ph; x["pitch_off"] = None
    x["is_fem"] = 1 if FEM(x) else 0                                          # Keep spk 0,1,2,3 distinct so TextEncoder never bakes actor timbre into Sanskrit phones!
print(f"v2.6 re-phonemized {len(items)} clips ({_chg} differ from cached v2.5 phones); fem: {sum(x['is_fem']==1 for x in items)} / mal: {sum(x['is_fem']==0 for x in items)}", flush=True)
def hard(p):
    j = " ".join(p)
    return any(k in j for k in ("t̪ s", "ts ", "ʃ ts", "ɽ", "<om>", "e r ʌ", "kʰ m", "ʌ h", "ɳ ɖ", "ʃ ts ʌ i")) or j.endswith("ʌ h")
hardn = 0; extra = []
for x in items:
    if hard(x["phones"]): extra += [x] * (a.boost - 1); hardn += 1
    elif x["lang"] == "sa": extra += [x]
items = items + extra; print(f"polish: {hardn} hard-cluster clips x{a.boost}, total {len(items)} (incl. dup refs)", flush=True)
for x in items:
    ph = [q for p in x["phones"] for q in (["o", "m"] if p == "<om>" else [p])]            # pranava expands to single-syllable o-m
    val_idx = [i for i, p in enumerate(ph) if p in SYM2ID]
    x["ids"] = torch.tensor([SYM2ID[ph[i]] for i in val_idx], dtype=torch.long)
    po = x.get("pitch_off")
    if po is not None and len(po) >= len(ph):
        x["po"] = torch.tensor([float(po[i]) for i in val_idx], dtype=torch.float32)
items = [x for x in items if 3 <= len(x["ids"]) and x["mel"].shape[1] <= 1400 and x["mel"].shape[1] > len(x["ids"])]
allm = torch.cat([x["mel"][:, ::8].float() for x in items[::10]], 1); mean, std = allm.mean().item(), allm.std().item()
for x in items: x["mel"] = x["mel"].to(dev)
print(f"{len(items)} utts, mel mean {mean:.3f} std {std:.3f}", flush=True)
net = SakhiyaAM().to(dev); net.stats.copy_(torch.tensor([mean, std])); ema = copy.deepcopy(net).eval()
print(f"params: enc {sum(p.numel() for p in net.enc.parameters())/1e6:.2f}M dur {sum(p.numel() for p in net.dur.parameters())/1e6:.2f}M vel {sum(p.numel() for p in net.vel.parameters())/1e6:.2f}M", flush=True)
lr = a.lr or 6e-5
ic = torch.load(a.init, map_location=dev, weights_only=False); sd = {k: v.float() for k, v in ic["ema"].items()}
net.load_state_dict(sd); mean, std = ic["mean"], ic["std"]; net.stats.copy_(torch.tensor([mean, std]))
with torch.no_grad():
    # Root-cause fix for Vedic ळ (SYM2ID["ɽ"]): blend lateral 'l' (60%) + retroflex 'ɽ' (40%) so अग्निमीळे is lateral [ɭ], never stop ड
    id_l, id_r = SYM2ID["l"], SYM2ID["ɽ"]
    lat_target = (0.60 * net.enc.emb.weight[id_l] + 0.40 * net.enc.emb.weight[id_r]).detach().clone()
    net.enc.emb.weight[id_r].copy_(lat_target)
    # Anchor Sanskrit lang[4] to Nepali lang[0] so every voice (female & male) preserves its exact vocal tract across ne/sa/thr
    net.enc.lang.weight[4].copy_(0.85 * net.enc.lang.weight[0] + 0.15 * net.enc.lang.weight[4])
ema = copy.deepcopy(net).eval()
print("INIT from", a.init, "step", ic["step"], "(with Vedic ळ lateral-retroflex init & cross-lingual voice lock)", flush=True)
opt = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-2); step = 0; ck = f"{a.out}/last.pt"
if os.path.exists(ck):
    c = torch.load(ck, map_location=dev); net.load_state_dict(c["net"], strict=False); ema.load_state_dict(c["ema"], strict=False); step = c["step"]
    if "opt" in c:
        try: opt.load_state_dict(c["opt"])
        except Exception: pass
    print("RESUMED at step", step, flush=True)
if os.path.exists(f"{a.out}/stage"): pass
open(f"{a.out}/stage", "w").write(a.stage)


POOL = [[i for i, x in enumerate(items) if x["is_fem"] == k] for k in (1, 0)]; print("pools fem/mal", len(POOL[0]), len(POOL[1]), flush=True)


def batches():
    while True:
        idx = sorted(random.sample(POOL[0], 2048) + random.sample(POOL[1], 2048), key=lambda i: items[i]["mel"].shape[1]); cur, mx = [], 0; bl = []   # STRICT 50/50 sakhiya/sandesh
        for i in idx:
            T = items[i]["mel"].shape[1]
            if cur and max(mx, T) * (len(cur) + 1) > a.frames: bl.append(cur); cur, mx = [], 0
            cur.append(i); mx = max(mx, T)
        if cur: bl.append(cur)
        random.shuffle(bl)
        for b in bl: yield b


def _batch_prosody(y, yl, il):
    # Compute batch 4-dim prosody z-scores [rate_z, f0_std_z, energy_std_z, pause_z] on GPU
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


def collate(b):
    xs = [items[i] for i in b]; L = max(len(x["ids"]) for x in xs); T = max(x["mel"].shape[1] for x in xs)
    ids = torch.zeros(len(xs), L, dtype=torch.long); y = torch.zeros(len(xs), 100, T, device=dev)
    f0_z = torch.zeros(len(xs), T, device=dev); po = torch.zeros(len(xs), L, device=dev); has_po = False
    for k, x in enumerate(xs):
        ids[k, :len(x["ids"])] = x["ids"]
        Tm = x["mel"].shape[1]
        y[k, :, :Tm] = (x["mel"].float() - mean) / std
        if "f0_z" in x and x["f0_z"] is not None:
            fz = x["f0_z"].float().to(dev)
            f0_z[k, :min(Tm, fz.shape[0])] = fz[:min(Tm, fz.shape[0])]
        if "po" in x and x["po"] is not None:
            p = x["po"].to(dev); po[k, :min(L, p.shape[0])] = p[:min(L, p.shape[0])]; has_po = True
    t = lambda f: torch.tensor([f(x) for x in xs], device=dev)
    il = t(lambda x: len(x["ids"])); yl = t(lambda x: x["mel"].shape[1])
    spks = [x["spk"] for x in xs]
    langs = [0 if (x["lang"] == "sa" and random.random() < 0.50) else LANG2ID[x["lang"]] for x in xs]
    pros = _batch_prosody(y, yl, il)
    return (ids.to(dev), il, y, yl, torch.tensor(spks, device=dev),
            torch.tensor(langs, device=dev), t(lambda x: x["style_id"]), t(lambda x: x["is_studio"]),
            po if has_po else None, f0_z if any("f0_z" in x for x in xs) else None, pros)


def coord(**kw):
    try:
        p = "/work/tts/COORD_V26.json"; c = json.load(open(p)); top = {k: kw.pop(k) for k in ("stage", "stage_idx", "right_now") if k in kw}
        c.update(top); c.setdefault("am", {}).update(kw, updated=time.time()); c["updated"] = time.time(); json.dump(c, open(p, "w"), indent=1)
    except Exception as e: print("coord", e)


def save(tag="last"):
    torch.save(dict(net=net.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), step=step, stage="polish", pitch_trained=True, mean=mean, std=std, styles=D["styles"]), f"{a.out}/{tag}.pt.tmp")
    os.replace(f"{a.out}/{tag}.pt.tmp", f"{a.out}/{tag}.pt")


t0 = time.time(); tl = time.time(); tsave = time.time(); gen = batches(); warm = 200; coord(pids=[os.getpid()], stage="Stage 2: v2.6 2-voice AM training (re-phonemized, 50/50 balanced)", stage_idx=1, right_now="Stage 2 running: v2.6 AM, 50/50 sakhiya/sandesh batches on re-phonemized data")
total = a.hours * 3600; hist = []
while step < a.steps and time.time() - t0 < total:
    for g in opt.param_groups:
        prog = min(1.0, (time.time() - t0) / total); g["lr"] = lr * min(1, (step + 1) / warm) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog)))
    b = collate(next(gen))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, parts = net.loss(*b)
        reg = 0.05 * ((net.enc.lang.weight[4] - net.enc.lang.weight[0].detach()) ** 2).mean() + 0.05 * ((net.enc.emb.weight[id_r] - lat_target) ** 2).mean()
        loss = loss + reg
    opt.zero_grad(set_to_none=True); loss.backward(); gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    if torch.isfinite(gn): opt.step()
    step += 1
    with torch.no_grad():
        d = min(0.999, (1 + step) / (10 + step))
        for pe, p in zip(ema.parameters(), net.parameters()): pe.lerp_(p, 1 - d)
    hist.append(loss.item())
    if step % 100 == 0:
        vr = torch.cuda.max_memory_allocated() / 2**30
        print(f"step {step} loss {sum(hist)/len(hist):.4f} {parts} lr {opt.param_groups[0]['lr']:.2e} vram {vr:.1f}G {time.time()-tl:.0f}s", flush=True); hist = []; tl = time.time()
        coord(vram_gb=round(torch.cuda.memory_reserved() / 2**30, 1), step=step)
    if step % 500 == 0 or time.time() - tsave > 300: save(); tsave = time.time(); print("saved", step, flush=True)
save(); save("polish_final"); coord(stage=f"Stage 2 done at step {step}", stage_idx=2, pids=[]); print("AM_DONE", step, flush=True)
