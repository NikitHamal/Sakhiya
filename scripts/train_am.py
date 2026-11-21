"""Train Sakhiya-AM (vectorised GPU collate + batched prior + cached MAS + contour/delta losses)."""
import os, sys, time, json, argparse, random, math
import numpy as np, torch, torch.nn.functional as F
from numba import njit, prange
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sakhiya.acoustic import Acoustic, MEL_MU, MEL_SD
from sakhiya.symbols import PAD, SYM2ID, LANG2ID

ap = argparse.ArgumentParser()
ap.add_argument("--set", default="v2"); ap.add_argument("--minutes", type=float, default=100)
ap.add_argument("--bs", type=int, default=64); ap.add_argument("--init", default="")
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--align_warm", type=int, default=0)
ap.add_argument("--w_pit", type=float, default=0.15); ap.add_argument("--ema", type=float, default=0.998); ap.add_argument("--log_every", type=int, default=200); ap.add_argument("--spks", default=""); ap.add_argument("--sa_w", type=int, default=5); ap.add_argument("--vag_w", type=int, default=1); ap.add_argument("--spk_init", default=""); ap.add_argument("--sa_only", action="store_true"); ap.add_argument("--out", default="/work/tts/runs/am2")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(4 if dev == "cuda" else max(4, os.cpu_count() or 8))
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True


@njit
def mas_one(lp, m_len, t_len):
    q = np.full((m_len, t_len), -1e9, dtype=np.float32); ind = np.zeros((m_len, t_len), dtype=np.int32)
    q[0, 0] = lp[0, 0]
    for i in range(1, m_len):
        hi = i + 1 if i + 1 < t_len else t_len
        for j in range(hi):
            s = q[i - 1, j]; m = q[i - 1, j - 1] if j > 0 else -1e9
            if m > s: q[i, j] = m + lp[i, j]; ind[i, j] = j - 1
            else: q[i, j] = s + lp[i, j]; ind[i, j] = j
    path = np.zeros(m_len, dtype=np.int32); j = t_len - 1
    for i in range(m_len - 1, -1, -1):
        path[i] = j; j = ind[i, j]
    return path


@njit(parallel=True)
def mas_batch(lsoft, ml, tl):
    B, Tm, Tt = lsoft.shape
    hard = np.zeros((B, Tm, Tt), dtype=np.float32)
    for b in prange(B):
        m_len, t_len = int(ml[b]), int(tl[b])
        p = mas_one(lsoft[b], m_len, t_len)
        for i in range(m_len):
            hard[b, i, p[i]] = 1.0
    return hard


def prior_batch(tl, ml, Tt, Tm, sc=0.05):
    i = torch.arange(1, Tm + 1, device=dev, dtype=torch.float32)[None, :, None]
    k = torch.arange(Tt, device=dev, dtype=torch.float32)[None, None, :]
    m_f = ml[:, None, None].float(); n_f = (tl - 1).clamp(min=1)[:, None, None].float()
    al = (sc * i).clamp(min=1e-4); be = (sc * (m_f + 1.0 - i)).clamp(min=1e-4)
    nk = (n_f - k).clamp(min=0.0)
    lb = lambda x, y: torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y)
    lg = torch.lgamma(n_f + 1.0) - torch.lgamma(k + 1.0) - torch.lgamma(nk + 1.0) + lb(k + al, nk + be) - lb(al, be)
    valid = (i <= m_f) & (k <= n_f)
    return torch.where(valid, lg.exp(), torch.zeros_like(lg))


data = torch.load(f"/work/tts/feats/{a.set}.pt", weights_only=False)
if a.spks:
    keep_s = set(a.spks.split(","))
    data = [r for r in data if r["spk"] in keep_s or r["split"] != "train"]
import librosa
BIN_OK = torch.tensor(librosa.mel_frequencies(n_mels=80, fmin=0.0, fmax=11025.0) < 7400, device=dev).float()
BIN_ONES = torch.ones(80, device=dev)
HAS_SA = any(r.get("lang") == "sa" for r in data)
spks = sorted({r["spk"] for r in data}); spk2id = {s: i for i, s in enumerate(spks)}
st = {}
for s in spks:
    vs = [np.log(r["f0"][r["f0"] > 0].astype(np.float32)) for r in data if r["spk"] == s and r["split"] == "train" and (r["f0"] > 0).any()]
    if not vs: vs = [np.log(r["f0"][r["f0"] > 0].astype(np.float32)) for r in data if r["spk"] == s and (r["f0"] > 0).any()]
    v = np.concatenate(vs); st[s] = (float(v.mean()), float(v.std() + 1e-5))
for r in data:
    m = torch.from_numpy(r["mel"].astype(np.float32)).to(dev)
    f = torch.from_numpy(r["f0"].astype(np.float32)).to(dev); mu, sd = st[r["spk"]]
    r["_ph"] = torch.tensor(r["ph"], dtype=torch.long, device=dev)
    r["_mel"], r["_voiced"] = m, (f > 0).float(); r["_wide"] = bool(r.get("wide", True))
    r["_pz"] = torch.where(f > 0, (torch.log(f.clamp(min=1)) - mu) / sd, torch.zeros_like(f))
    r["_en_raw"] = torch.logsumexp(m, 0)
en_all = torch.cat([r["_en_raw"] for r in data]); EN_MU, EN_SD = en_all.mean().item(), en_all.std().item()
for r in data:
    r["_en"] = (r["_en_raw"] - EN_MU) / EN_SD; del r["_en_raw"]
tr = [r for r in data if r["split"] == "train"]; va = [r for r in data if r["split"] == "val"]
M = Acoustic(n_spk=max(32, len(spks))).to(dev)
print("params", sum(p.numel() for n, p in M.named_parameters() if not n.startswith("aligner")), "train clips", len(tr), flush=True)
if a.init:
    ck0 = torch.load(a.init, weights_only=False); sd0 = ck0["M"]; old = ck0.get("spk2id")
    if old and old != spk2id:
        w = M.spk.weight.data.clone(); hit = 0
        base = sd0["spk.weight"][old.get("sakhiya", old.get("s3614", next(iter(old.values()))))].to(w.device)
        for s_, i in spk2id.items():   # unseen real speakers start from the shipped single-voice embedding (+ small noise), not from random init
            if s_ in old: w[i] = sd0["spk.weight"][old[s_]].to(w.device); hit += 1
            else: w[i] = base + 0.02 * torch.randn_like(base)
        for kv in filter(None, a.spk_init.split(",")):   # e.g. edge_f=s3614,edge_m=cv_13b830fe : new voice starts from a similar old embedding
            nk, ok_ = kv.split("=")
            if nk in spk2id and ok_ in old: w[spk2id[nk]] = sd0["spk.weight"][old[ok_]].to(w.device); hit += 1
        sd0 = dict(sd0); sd0["spk.weight"] = w.cpu(); print("speaker remap", hit, "of", len(spk2id), flush=True)
    M.load_state_dict(sd0, strict=False); print("resumed from", a.init, flush=True)
    if not ck0.get("has_sa"):
        with torch.no_grad():
            M.lang.weight[LANG2ID["sa"]] = M.lang.weight[0]   # new Sanskrit language slot starts from Nepali
            # Alpa Section 3.3 r-coloured retroflex stops/nasal prior on sa_emb (never 'ʃ', which is also palatal श)
            r_vec = M.emb.weight[SYM2ID["r"]]
            for sym in ("ʈ", "ʈʰ", "ɖ", "ɖʱ", "ɳ"):
                M.sa_emb.weight[SYM2ID[sym]] = 0.20 * (r_vec - M.emb.weight[SYM2ID[sym]])
            M.sa_emb.weight[SYM2ID["w"]] = 0.18 * (M.emb.weight[SYM2ID["u"]] - M.emb.weight[SYM2ID["b"]])
if a.sa_only:
    for n, p in M.named_parameters():
        p.requires_grad_(n in ("sa_emb.weight", "spk.weight", "lang.weight"))
    saved_spk = M.spk.weight.detach().clone()
    saved_lang = M.lang.weight.detach().clone()
    sa_rows = [i for s_, i in spk2id.items() if s_.startswith("sa_")]
    non_sa_rows = [i for i in range(M.spk.weight.shape[0]) if i not in sa_rows]
opt = torch.optim.AdamW([p for p in M.parameters() if p.requires_grad], a.lr, betas=(0.9, 0.98), weight_decay=1e-6)


def collate(rs):
    B = len(rs); Tt = max(r["_ph"].numel() for r in rs); Tm = max(r["_mel"].shape[1] for r in rs)
    ids = torch.zeros((B, Tt), dtype=torch.long, device=dev)
    mel = torch.full((B, 80, Tm), MEL_MU, device=dev)
    pz = torch.zeros((B, Tm), device=dev); vo = torch.zeros((B, Tm), device=dev); en = torch.zeros((B, Tm), device=dev)
    tl = torch.tensor([r["_ph"].numel() for r in rs], device=dev, dtype=torch.long)
    ml = torch.tensor([r["_mel"].shape[1] for r in rs], device=dev, dtype=torch.long)
    for i, r in enumerate(rs):
        t_i, m_i = r["_ph"].numel(), r["_mel"].shape[1]
        ids[i, :t_i] = r["_ph"]; mel[i, :, :m_i] = r["_mel"]
        pz[i, :m_i] = r["_pz"]; vo[i, :m_i] = r["_voiced"]; en[i, :m_i] = r["_en"]
    spk = torch.tensor([spk2id[r["spk"]] for r in rs], device=dev, dtype=torch.long)
    lang = torch.tensor([LANG2ID.get(r.get("lang", "ne"), 0) for r in rs], device=dev, dtype=torch.long)
    bw = torch.stack([BIN_ONES if r["_wide"] else BIN_OK for r in rs])
    return ids, mel, pz, vo, en, tl, ml, spk, lang, bw


def step_loss(rs, step, ret_pred=False):
    ids, mel, pz, vo, en, tl, ml, spk, lang, bw = collate(rs)
    B, Tt, Tm = ids.shape[0], ids.shape[1], mel.shape[2]
    tm = torch.arange(Tt, device=dev)[None] < tl[:, None]; mm = torch.arange(Tm, device=dev)[None] < ml[:, None]
    warm_only = step < a.align_warm and not ret_pred
    use_cache = (step >= a.align_warm) and (bool(a.init) or step % 8 != 0 or ret_pred) and all("_cdur" in r for r in rs)
    if use_cache:
        dur = torch.zeros((B, Tt), device=dev); pt = torch.zeros((B, Tt), device=dev); et = torch.zeros((B, Tt), device=dev)
        for i, r in enumerate(rs):
            t_i = r["_ph"].numel()
            dur[i, :t_i] = r["_cdur"]; pt[i, :t_i] = r["_cpt"]; et[i, :t_i] = r["_cet"]
        lctc = torch.zeros((), device=dev); lbin = torch.zeros((), device=dev)
    else:
        mel_n = (mel - MEL_MU) / MEL_SD
        with torch.autocast("cuda", enabled=False):
            emb = M.text_emb(ids, spk, lang)
            if warm_only or a.init: emb = emb.detach()
            lg = M.aligner(emb, mel_n).masked_fill(~tm[:, None, :], -1e4)
        pr = prior_batch(tl, ml, Tt, Tm)
        alp = F.log_softmax(lg, -1) + torch.log(pr + 1e-8)
        lp = torch.cat([torch.full((B, Tm, 1), -1.0, device=dev), alp], -1).permute(1, 0, 2).float()
        tg = torch.arange(1, Tt + 1, device=dev)[None].expand(B, -1)
        lctc = (F.ctc_loss(lp, tg, ml, tl, blank=0, reduction="none", zero_infinity=True) / tl).mean()
        if warm_only and step < a.align_warm // 2:
            return 2.0 * lctc, dict(mel=0.0, dur=0.0, pit=0.0, eng=0.0, ctc=lctc.item(), bin=0.0)
        soft = F.softmax(alp, -1); lsoft = torch.log(soft.clamp(min=1e-8)).detach().cpu().numpy()
        hard = torch.from_numpy(mas_batch(lsoft, ml.cpu().numpy(), tl.cpu().numpy())).to(dev)
        lbin = -(torch.log(soft.clamp(min=1e-8)) * hard).sum() / hard.sum().clamp(min=1.0)
        dur = hard.sum(1)
        A = hard.transpose(1, 2)
        vt = torch.bmm(A, vo[..., None])[..., 0]
        pt = torch.bmm(A, (pz * vo)[..., None])[..., 0] / vt.clamp(min=1)
        et = torch.bmm(A, (en * mm)[..., None])[..., 0] / dur.clamp(min=1)
        if step >= max(0, a.align_warm - 50) and not ret_pred:
            for i, r in enumerate(rs):
                t_i = r["_ph"].numel()
                r["_cdur"] = dur[i, :t_i].detach(); r["_cpt"] = pt[i, :t_i].detach(); r["_cet"] = et[i, :t_i].detach()
        if warm_only:
            return 2.0 * lctc + min(1.0, (step + 1) / max(1, a.align_warm)) * lbin, dict(mel=0.0, dur=0.0, pit=0.0, eng=0.0, ctc=lctc.item(), bin=lbin.item())
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h, ld, p, e = M.encode(ids, spk, lang, tm)
        x = M.regulate(M.condition(h, pt, et), dur.long())
        pred = M.decode(x, mm)
    pred, ld, p, e = pred.float(), ld.float(), p.float(), e.float()
    if ret_pred: return pred, ml
    wm = mm[:, None] * bw[:, :, None]
    lmel = (torch.abs(pred - mel) * wm).sum() / wm.sum().clamp(min=1.0) / MEL_SD
    dt_w = wm[:, :, 1:] * wm[:, :, :-1]
    ldelta_t = (torch.abs((pred[:, :, 1:] - pred[:, :, :-1]) - (mel[:, :, 1:] - mel[:, :, :-1])) * dt_w).sum() / dt_w.sum().clamp(min=1.0) / MEL_SD
    df_w = wm[:, 1:, :] * wm[:, :-1, :]
    ldelta_f = (torch.abs((pred[:, 1:, :] - pred[:, :-1, :]) - (mel[:, 1:, :] - mel[:, :-1, :])) * df_w).sum() / df_w.sum().clamp(min=1.0) / MEL_SD
    f = lambda y, t: (((y - t) ** 2) * tm).sum() / tm.sum().clamp(min=1.0)
    tm2 = tm[:, 1:] & tm[:, :-1]
    fd = lambda y, t: ((((y[:, 1:] - y[:, :-1]) - (t[:, 1:] - t[:, :-1])) ** 2) * tm2).sum() / tm2.sum().clamp(min=1.0)
    ldur = f(ld, torch.log(dur + 1))
    lpit = f(p, pt) + 0.5 * fd(p, pt)
    leng = f(e, et) + 0.5 * fd(e, et)
    w_bin = 1.0 if a.init else min(1.0, step / 4000)
    loss = lmel + 0.15 * ldelta_t + 0.12 * ldur + a.w_pit * lpit + 0.12 * leng + 1.5 * lctc + w_bin * lbin
    return loss, dict(mel=lmel.item(), dur=ldur.item(), pit=lpit.item(), eng=leng.item(), ctc=lctc.item(), bin=lbin.item())


def lr_at(s): return a.lr * min(1, (s + 1) / (250 if a.init else 1000)) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1, (time.time() - t0) / (a.minutes * 60)))))


@torch.no_grad()
def precompute_mas():
    M.eval(); ordered = sorted(tr + va, key=lambda r: r["_mel"].shape[1])
    for i in range(0, len(ordered), 64):
        rs = ordered[i:i + 64]
        ids, mel, pz, vo, en, tl, ml, spk, lang, _ = collate(rs)
        B, Tt, Tm = ids.shape[0], ids.shape[1], mel.shape[2]
        tm = torch.arange(Tt, device=dev)[None] < tl[:, None]; mm = torch.arange(Tm, device=dev)[None] < ml[:, None]
        pure_emb = M.emb(ids) * math.sqrt(M.d) + M.spk(spk)[:, None] + M.lang(torch.zeros_like(lang))[:, None]
        lg = M.aligner(pure_emb, (mel - MEL_MU) / MEL_SD).masked_fill(~tm[:, None, :], -1e4)
        alp = F.log_softmax(lg, -1) + torch.log(prior_batch(tl, ml, Tt, Tm) + 1e-8)
        soft = F.softmax(alp, -1); lsoft = torch.log(soft.clamp(min=1e-8)).cpu().numpy()
        hard = torch.from_numpy(mas_batch(lsoft, ml.cpu().numpy(), tl.cpu().numpy())).to(dev)
        dur = hard.sum(1); A = hard.transpose(1, 2)
        vt = torch.bmm(A, vo[..., None])[..., 0]
        pt = torch.bmm(A, (pz * vo)[..., None])[..., 0] / vt.clamp(min=1)
        et = torch.bmm(A, (en * mm)[..., None])[..., 0] / dur.clamp(min=1)
        for j, r in enumerate(rs):
            t_j = r["_ph"].numel()
            r["_cdur"] = dur[j, :t_j]; r["_cpt"] = pt[j, :t_j]; r["_cet"] = et[j, :t_j]
    M.train(); print("MAS precomputed for", len(ordered), "clips", flush=True)


# Oversample flagship voices (edge_f/s3614 Female, edge_m/cv_b0487694/cv_13b830fe Male) and wideband native speakers for sharper acoustics
tr_pool = []
for r in tr:
    if a.sa_only and r.get("lang") != "sa": continue
    w_mult = a.vag_w if r["spk"] == "vag" else a.sa_w if r.get("lang") == "sa" else 6 if r["spk"] in ("edge_f", "edge_m", "s3614", "cv_b0487694", "cv_13b830fe", "s7957") else (2 if r.get("wide", True) else 1)
    tr_pool.extend([r] * w_mult)

step, t0, ck = 0, time.time(), f"{a.out}/last.pt"
if a.align_warm == 0: precompute_mas()
ema_m = {k: v.detach().clone().float() for k, v in M.state_dict().items()}
M.train(); logs = []
while time.time() - t0 < a.minutes * 60:
    if a.align_warm > 0 and step == a.align_warm: precompute_mas()
    for g in opt.param_groups: g["lr"] = (1.5e-3 if step < a.align_warm else lr_at(step))
    loss, d = step_loss(random.sample(tr_pool, min(a.bs, len(tr_pool))), step); opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(M.parameters(), 1.0); opt.step(); step += 1; logs.append(d)
    with torch.no_grad():
        if a.sa_only:
            M.spk.weight[non_sa_rows] = saved_spk[non_sa_rows]
            for li in range(M.lang.weight.shape[0]):
                if li != LANG2ID["sa"]: M.lang.weight[li] = saved_lang[li]
            M.sa_emb.weight[SYM2ID[PAD]].zero_()
        for k, v in M.state_dict().items():
            ema_m[k].mul_(a.ema).add_(v.detach().float(), alpha=1.0 - a.ema)
    if step % a.log_every == 0:
        print(step, {k: round(sum(x[k] for x in logs) / len(logs), 3) for k in d}, f"{(time.time()-t0)/60:.1f}m", flush=True); logs = []
    if step % 1500 == 0:
        M.eval()
        with torch.no_grad(): vs = [step_loss(va[i:i + 8], 10**9)[1] for i in range(0, min(24, len(va)), 8)]
        M.train(); v = {k: sum(x[k] for x in vs) / len(vs) for k in vs[0]}
        json.dump({"step": step, "val": v, "t": time.time()}, open(f"{a.out}/progress.json", "w")); print("VAL", step, v, flush=True)
        torch.save({"M": ema_m, "spk2id": spk2id, "pitch_stats": st, "en": (EN_MU, EN_SD), "step": step, "has_sa": HAS_SA or bool(a.init and ck0.get("has_sa"))}, ck)
M.load_state_dict(ema_m)
torch.save({"M": ema_m, "spk2id": spk2id, "pitch_stats": st, "en": (EN_MU, EN_SD), "step": step, "has_sa": HAS_SA or bool(a.init and ck0.get("has_sa"))}, ck); print("DONE", step, flush=True)

M.eval(); gta = {}
alls = [r for r in data if r["split"] != "test"]; alls.sort(key=lambda r: r["_mel"].shape[1])
with torch.no_grad():
    for i in range(0, len(alls), 24):
        rs = alls[i:i + 24]; pred, ml = step_loss(rs, 10**9, ret_pred=True)
        for j, r in enumerate(rs): gta[r["id"]] = pred[j, :, :int(ml[j])].cpu().numpy().astype(np.float16)
torch.save(gta, f"/work/tts/feats/gta_{a.set}.pt"); print("GTA_DONE", len(gta), flush=True)
