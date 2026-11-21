"""GAN training for Sakhiya-Voc (MPD+MRD, bf16, optional GTA predicted-mel fine-tune via --gta)."""
import os, sys, time, json, argparse, random, math
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sakhiya import audio
from sakhiya.vocoder import Vocoder, n_params, load_compat


class PD(nn.Module):
    def __init__(s, p):
        super().__init__(); s.p = p; ch = [1, 16, 64, 256, 512]
        s.c = nn.ModuleList([nn.utils.parametrizations.weight_norm(nn.Conv2d(ch[i], ch[i + 1], (5, 1), (3, 1), (2, 0))) for i in range(4)])
        s.c.append(nn.utils.parametrizations.weight_norm(nn.Conv2d(512, 512, (5, 1), 1, (2, 0))))
        s.o = nn.utils.parametrizations.weight_norm(nn.Conv2d(512, 1, (3, 1), 1, (1, 0)))

    def forward(s, x):
        B, _, T = x.shape
        if T % s.p: x = F.pad(x, (0, s.p - T % s.p), mode="reflect")
        x = x.view(B, 1, -1, s.p); fm = []
        for c in s.c:
            x = F.leaky_relu(c(x), 0.1); fm.append(x)
        return s.o(x).flatten(1), fm


class RD(nn.Module):
    def __init__(s, n):
        super().__init__(); s.n = n; w = nn.utils.parametrizations.weight_norm
        s.c = nn.ModuleList([w(nn.Conv2d(1, 32, (3, 9), padding=(1, 4)))] +
                            [w(nn.Conv2d(32, 32, (3, 9), (1, 2), (1, 4))) for _ in range(3)] + [w(nn.Conv2d(32, 32, 3, padding=1))])
        s.o = w(nn.Conv2d(32, 1, 3, padding=1)); s.register_buffer("win", torch.hann_window(n))

    def forward(s, x):
        x = torch.stft(x[:, 0], s.n, s.n // 4, window=s.win, return_complex=True).abs()[:, None]
        fm = []
        for c in s.c:
            x = F.leaky_relu(c(torch.log(x + 1e-5) if not fm else x), 0.1); fm.append(x)
        return s.o(x).flatten(1), fm


class Disc(nn.Module):
    def __init__(s):
        super().__init__(); s.ds = nn.ModuleList([PD(p) for p in (2, 3, 5, 7, 11)] + [RD(n) for n in (512, 1024, 2048)])

    def forward(s, x):
        r = [d(x) for d in s.ds]; return [o for o, _ in r], [f for _, f in r]


def phase_loss(y, x, n=1024, dev="cuda"):
    """Magnitude-guarded anti-wrapped group-delay + instantaneous-frequency derivative loss."""
    w = torch.hann_window(n, device=dev)
    Y = torch.stft(y, n, n // 4, window=w, return_complex=True)[:, :200]
    X = torch.stft(x, n, n // 4, window=w, return_complex=True)[:, :200]
    aw = lambda d: torch.abs(d - 2 * math.pi * torch.round(d / (2 * math.pi)))
    Ym = Y.abs(); Xm = X.abs()
    Y_safe = torch.where(Ym > 1e-3, Y, X.detach() + 1e-4)
    py, px = Y_safe.angle(), X.angle()
    wt = ((Xm * Ym.detach().clamp(min=1e-3)).sqrt() / (Xm.mean() + 1e-5)).clamp(max=4) / 4
    gd = (aw(py.diff(dim=1) - px.diff(dim=1)) * wt[:, 1:]).mean()
    ifr = (aw(py.diff(dim=2) - px.diff(dim=2)) * wt[:, :, 1:]).mean()
    return 0.5 * (gd + ifr)


def mrstft(y, x, dev="cuda"):
    l = 0
    for n in (512, 1024, 2048):
        w = torch.hann_window(n, device=dev)
        Y = torch.stft(y, n, n // 4, window=w, return_complex=True).abs()
        X = torch.stft(x, n, n // 4, window=w, return_complex=True).abs()
        l = l + (torch.norm(X - Y) / torch.norm(X).clamp(min=1e-5)) + F.l1_loss(torch.log(Y + 1e-5), torch.log(X + 1e-5))
    return l / 3


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="v2"); ap.add_argument("--minutes", type=float, default=150)
    ap.add_argument("--bs", type=int, default=24); ap.add_argument("--seg", type=int, default=16384)
    ap.add_argument("--out", default="/work/tts/runs/voc2"); ap.add_argument("--warm", type=int, default=2000)
    ap.add_argument("--gta_p", type=float, default=0.2); ap.add_argument("--gta", default=""); ap.add_argument("--init", default=""); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--focus", default=""); ap.add_argument("--focus_p", type=float, default=0.5)
    ap.add_argument("--wide", type=int, default=1); ap.add_argument("--phase", type=float, default=0.0); ap.add_argument("--no_gan", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu": torch.set_num_threads(max(4, os.cpu_count() or 8))
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

    data = torch.load(f"/work/tts/feats/{a.set}.pt", weights_only=False)
    if a.wide: data = [r for r in data if r.get("wide", True)]
    gta = torch.load(a.gta, weights_only=False) if a.gta else {}
    fs = set(a.focus.split(",")) if a.focus else set()
    tr, trf = [], []
    for r in data:
        if r["split"] == "train":
            item = (torch.from_numpy(r["wav"]).to(dev).float() / 32767, (torch.from_numpy(gta[r["id"]]).to(dev).float() if r["id"] in gta else None))
            tr.append(item)
            if r["spk"] in fs: trf.append(item)
    va = [(torch.from_numpy(r["wav"]).to(dev).float() / 32767, (torch.from_numpy(gta[r["id"]]).to(dev).float() if r["id"] in gta else None)) for r in data if r["split"] == "val"]
    print("focus clips", len(trf), flush=True)
    print("train clips", len(tr), "gta", sum(g is not None for _, g in tr), flush=True)
    G = Vocoder().to(dev); D = Disc().to(dev); print("G params", n_params(G), flush=True)
    has_d = False
    if a.init:
        ck_init = torch.load(a.init, weights_only=False)
        if isinstance(ck_init, dict) and "G" in ck_init:
            load_compat(G, ck_init["G"])
            if "D" in ck_init: D.load_state_dict(ck_init["D"]); has_d = True
        else:
            load_compat(G, ck_init)
            sib = os.path.join(os.path.dirname(a.init), "last.pt")
            if os.path.exists(sib):
                s_ck = torch.load(sib, weights_only=False)
                if isinstance(s_ck, dict) and "D" in s_ck: D.load_state_dict(s_ck["D"]); has_d = True
        print("init from", a.init, "loaded_D=", has_d, flush=True)
    oG = torch.optim.AdamW(G.parameters(), a.lr, betas=(0.8, 0.99)); oD = torch.optim.AdamW(D.parameters(), a.lr, betas=(0.8, 0.99))
    step, t0, ck = 0, time.time(), f"{a.out}/last.pt"
    if os.path.exists(ck) and not a.init:
        s = torch.load(ck, weights_only=False); G.load_state_dict(s["G"]); D.load_state_dict(s["D"]); oG.load_state_dict(s["oG"]); oD.load_state_dict(s["oD"]); step = s["step"]; has_d = True
    SF = a.seg // 256
    win1024 = torch.hann_window(1024, device=dev); pad1024 = (1024 - 256) // 2

    def batch():
        xs, gs = [], []
        for _ in range(a.bs):
            w, g = random.choice(trf if (trf and random.random() < a.focus_p) else tr)
            n = w.shape[0] // 256
            if n <= SF + 1: w = F.pad(w, (0, (SF + 2) * 256 - w.shape[0])); n = SF + 2
            f = random.randint(0, n - SF - 1); x = w[f * 256:(f + SF) * 256]; xs.append(x)
            gs.append(g[:, f:f + SF] if (g is not None and random.random() < a.gta_p and g.shape[1] >= f + SF) else None)
        x_b = torch.stack(xs); xm = audio.mel_spec(x_b, dev)
        mel = torch.stack([xm[i] if gs[i] is None else gs[i] for i in range(a.bs)])
        return x_b, mel, xm

    @torch.no_grad()
    def val():
        from sakhiya.vocoder import _ola
        G.eval(); l = []
        for w, _ in va[:40]:
            w = w[: w.shape[0] // 256 * 256][None]; m = audio.mel_spec(w, dev)
            re_v, im_v = G.spectrum(m); y = _ola(torch.complex(re_v.float(), im_v.float()), G.win)
            T = min(y.shape[1], w.shape[1])
            l.append(F.l1_loss(audio.mel_spec(y[:, :T], dev), m[:, :, :T // 256]).item())
        G.train(); return sum(l) / len(l)

    def ac(): return torch.autocast("cuda", dtype=torch.bfloat16)

    while time.time() - t0 < a.minutes * 60:
        x, mel, xm = batch(); lr = a.lr * (0.04 + 0.96 * 0.5 * (1 + math.cos(math.pi * min(1, (time.time() - t0) / (a.minutes * 60)))))
        for o in (oG, oD):
            for g_ in o.param_groups: g_["lr"] = lr
        with ac(): re_p, im_p = G.spectrum(mel)
        sp_p = torch.complex(re_p.float(), im_p.float()); mag0 = sp_p.abs()
        from sakhiya.vocoder import _ola
        y = _ola(sp_p, G.win)[:, : x.shape[1]].float()
        ym = audio.mel_spec(y, dev); lm = 0.5 * F.l1_loss(ym, xm) + 0.5 * F.l1_loss(ym, mel[:, :, :ym.shape[-1]])
        st_y = torch.stft(F.pad(y[:, None], (pad1024, pad1024), mode="reflect")[:, 0], 1024, 256, 1024, win1024, center=False, return_complex=True)
        st_x = torch.stft(F.pad(x[:, None], (pad1024, pad1024), mode="reflect")[:, 0], 1024, 256, 1024, win1024, center=False, return_complex=True)
        T_st = min(st_y.shape[-1], mag0.shape[-1], st_x.shape[-1])
        log_mag0 = torch.log(mag0[:, :, :T_st].clamp(min=1e-5))
        lcons = F.l1_loss(torch.log(st_y.abs()[:, :, :T_st].clamp(min=1e-5)), log_mag0.detach()) + 0.5 * F.l1_loss(log_mag0, torch.log(st_x.abs()[:, :, :T_st].clamp(min=1e-5)))
        gan = (not a.no_gan) and (step >= (0 if has_d else (800 if a.init else a.warm)))
        if (not a.no_gan) and a.init and not has_d and step < 800:
            with ac():
                ro, _ = D(x[:, None]); fo, _ = D(y.detach()[:, None])
                ld = sum(torch.mean(F.relu(1 - r.float()) + F.relu(1 + f.float())) for r, f in zip(ro, fo)) / len(ro)
            oD.zero_grad(set_to_none=True); ld.backward(); torch.nn.utils.clip_grad_norm_(D.parameters(), 10); oD.step()
        if gan:
            with ac():
                ro, _ = D(x[:, None]); fo, _ = D(y.detach()[:, None])
                ld = sum(torch.mean(F.relu(1 - r.float()) + F.relu(1 + f.float())) for r, f in zip(ro, fo)) / len(ro)
            oD.zero_grad(set_to_none=True); ld.backward(); torch.nn.utils.clip_grad_norm_(D.parameters(), 10); oD.step()
            with torch.no_grad(), ac(): _, rf = D(x[:, None])
            with ac():
                fo, ff = D(y[:, None])
                lg = sum(torch.mean(F.relu(1 - f.float())) for f in fo) / len(fo)
                lfm = sum(F.l1_loss(a_.float(), b_.detach().float()) for fa, fb in zip(ff, rf) for a_, b_ in zip(fa, fb)) / sum(len(f) for f in ff)
            loss = 45 * lm + 2 * mrstft(y, x, dev) + 2.5 * lcons + lg + 4 * lfm + (a.phase * phase_loss(y, x, dev=dev) if a.phase else 0)
        else:
            loss = 45 * lm + 2 * mrstft(y, x, dev) + 2.5 * lcons + (a.phase * phase_loss(y, x, dev=dev) if a.phase else 0); ld = torch.zeros(()); lg = torch.zeros(())
        oG.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(G.parameters(), 10); oG.step(); step += 1
        if step == 1: ema_g = {k: v.detach().clone().float() for k, v in G.state_dict().items()}
        with torch.no_grad():
            for k, v in G.state_dict().items():
                ema_g[k].mul_(0.998).add_(v.detach().float(), alpha=0.002)
        if step % 200 == 0:
            print(f"step {step} mel {lm.item():.3f} D {float(ld.detach()):.2f} G {float(lg.detach()):.2f} t {(time.time()-t0)/60:.1f}m", flush=True)
        if step % 1500 == 0:
            v = val()
            json.dump({"step": step, "train_mel_l1": lm.item(), "val_mel_l1": v, "t": time.time(), "mode": "gta" if a.gta else "gt"}, open(f"{a.out}/progress.json", "w"))
            print("VAL", step, v, flush=True)
            torch.save({"G": ema_g, "D": D.state_dict(), "oG": oG.state_dict(), "oD": oD.state_dict(), "step": step}, ck)
            torch.save(ema_g, f"{a.out}/G.pt")
    if step > 0: G.load_state_dict(ema_g)
    torch.save(G.state_dict(), f"{a.out}/G.pt")
    torch.save({"G": G.state_dict(), "D": D.state_dict(), "oG": oG.state_dict(), "oD": oD.state_dict(), "step": step}, ck)
    print("DONE", step, val(), flush=True)
