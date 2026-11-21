"""Sakhiya-v2-Voc Training (Track B): 24 kHz Source-Filter Hybrid iSTFTNet/HiFTNet Vocoder.
Stage 3A: 24 kHz studio GT mels (Rasa Nepali/Sanskrit F/M + SLR43 studio).
Stage 3B: GTA fine-tuning (--gta /work/tts/feats/v2_gta_24k.pt --gta_p 0.7 --lr 5e-5)."""
import os, sys, io, glob, time, json, math, random, argparse, copy
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sakhiya.v2_vocoder import VocoderV2, mel_spec_24k, ola64, n_params, SR, HOP


class PD(nn.Module):
    def __init__(s, p):
        super().__init__(); s.p = p; ch = [1, 16, 64, 256, 512]
        s.c = nn.ModuleList([weight_norm(nn.Conv2d(ch[i], ch[i + 1], (5, 1), (3, 1), (2, 0))) for i in range(4)])
        s.c.append(weight_norm(nn.Conv2d(512, 512, (5, 1), 1, (2, 0))))
        s.o = weight_norm(nn.Conv2d(512, 1, (3, 1), 1, (1, 0)))

    def forward(s, x):
        B, _, T = x.shape
        if T % s.p: x = F.pad(x, (0, s.p - T % s.p), mode="reflect")
        x = x.view(B, 1, -1, s.p); fm = []
        for c in s.c: x = F.leaky_relu(c(x), 0.1); fm.append(x)
        return s.o(x).flatten(1), fm


class RD(nn.Module):
    def __init__(s, n):
        super().__init__(); s.n = n
        s.c = nn.ModuleList([weight_norm(nn.Conv2d(1, 32, (3, 9), padding=(1, 4)))] +
                            [weight_norm(nn.Conv2d(32, 32, (3, 9), (1, 2), (1, 4))) for _ in range(3)] +
                            [weight_norm(nn.Conv2d(32, 32, 3, padding=1))])
        s.o = weight_norm(nn.Conv2d(32, 1, 3, padding=1)); s.register_buffer("win", torch.hann_window(n))

    def forward(s, x):
        x = torch.stft(x[:, 0], s.n, s.n // 4, window=s.win, return_complex=True).abs()[:, None]; fm = []
        for c in s.c: x = F.leaky_relu(c(torch.log(x + 1e-5) if not fm else x), 0.1); fm.append(x)
        return s.o(x).flatten(1), fm


class DiscV2(nn.Module):
    def __init__(s):
        super().__init__(); s.ds = nn.ModuleList([PD(p) for p in (2, 3, 5, 7, 11)] + [RD(n) for n in (512, 1024, 2048)])

    def forward(s, x):
        r = [d(x) for d in s.ds]; return [o for o, _ in r], [f for _, f in r]


_win_cache = {}
def _hann(n, dev):
    k = (n, str(dev))
    if k not in _win_cache:
        _win_cache[k] = torch.hann_window(n, device=dev)
    return _win_cache[k]


def phase_loss(y, x, n=1024, dev="cuda"):
    """Magnitude-guarded anti-wrapped group-delay + instantaneous-frequency derivative loss."""
    w = _hann(n, dev)
    Y = torch.stft(y.float(), n, n // 4, window=w, return_complex=True)[:, :200]
    X = torch.stft(x.float(), n, n // 4, window=w, return_complex=True)[:, :200]
    aw = lambda d: torch.abs(d - 2 * math.pi * torch.round(d / (2 * math.pi)))
    Ym, Xm = Y.abs(), X.abs()
    Y_safe = torch.where(Ym > 1e-3, Y, X.detach() + 1e-4)
    py, px = Y_safe.angle(), X.angle()
    wt = ((Xm * Ym.detach().clamp(min=1e-3)).sqrt() / (Xm.mean() + 1e-5)).clamp(max=4.0) / 4.0
    gd = (aw(py.diff(dim=1) - px.diff(dim=1)) * wt[:, 1:]).mean()
    ifr = (aw(py.diff(dim=2) - px.diff(dim=2)) * wt[:, :, 1:]).mean()
    return 0.5 * (gd + ifr)


def mr_complex_stft_loss(y, x, dev="cuda"):
    y, x = y.float(), x.float()
    l = 0.0
    for n in (256, 512, 1024, 2048):
        w = _hann(n, dev)
        Ym = torch.stft(y, n, n // 4, window=w, return_complex=True).abs()
        Xm = torch.stft(x, n, n // 4, window=w, return_complex=True).abs()
        sc = torch.norm(Xm - Ym) / torch.norm(Xm).clamp(min=1e-5)
        mag = F.l1_loss(torch.log(Ym + 1e-5), torch.log(Xm + 1e-5))
        l = l + sc + mag
    return l / 4.0 + 0.75 * phase_loss(y, x, 1024, dev)


@torch.no_grad()
def gpu_f0_uv(wav, n_frames):
    """Fast vectorized GPU autocorrelation F0 + voicing extractor at 24 kHz (60-500 Hz, hop=256)."""
    pad = (1024 - 256) // 2
    x = F.pad(wav[:, None], (pad, pad), mode="reflect")[:, 0]
    fr = x.unfold(1, 1024, 256)[:, :n_frames]                         # (B, T, 1024)
    fr = fr - fr.mean(dim=-1, keepdim=True)
    eng = (fr ** 2).mean(dim=-1, keepdim=True).clamp(min=1e-8)
    spec = torch.fft.rfft(fr, n=2048, dim=-1)
    acf = torch.fft.irfft(spec * spec.conj(), n=2048, dim=-1)[..., :401] / (eng * 1024.0)
    lags = acf[..., 48:400]                                           # 24000/500=48 to 24000/60=400
    pk_val, pk_idx = lags.max(dim=-1)
    f0 = (24000.0 / (pk_idx.float() + 48.0)).clamp(60.0, 500.0)
    uv = ((pk_val > 0.32) & (eng[..., 0] > 1e-5)).float()
    return f0[:, None], uv[:, None]


def extract_rasa_clip(r, sub, meter):
    import soundfile as sf, soxr, librosa, pyloudnorm as pyln
    is_f = r["gender"] == "Female"
    y, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
    if y.ndim > 1: y = y.mean(1)
    y = soxr.resample(y, sr, SR, quality="VHQ") if sr != 48000 else y[::2].copy()
    _, (a, b) = librosa.effects.trim(y, top_db=45, frame_length=1024, hop_length=256)
    m = int(0.10 * SR); y = y[max(0, a - m): min(len(y), b + m)]
    if len(y) < SR * 0.7 or len(y) > SR * 18: return None
    L = meter.integrated_loudness(y); y = pyln.normalize.loudness(y, L, -23.0)
    pk = np.abs(y).max()
    if pk > 0.97: y = y * (0.97 / pk)
    spk = (0 if is_f else 1) if sub == "Nepali" else (2 if is_f else 3)
    return dict(id=r["filename"], wav=(y * 32767.0).astype(np.int16), spk=spk, lang="ne" if sub == "Nepali" else "sa", split="val" if hash(r["filename"]) % 50 == 0 else "train")


def build_or_load_wavs(cache_path, max_per_subset=250, tr_list=None, trf_list=None, gta_dict=None):
    if os.path.exists(cache_path):
        d = torch.load(cache_path, weights_only=False)
        if len(d) >= 400: return d
    import pyarrow.parquet as pq, pyloudnorm as pyln, threading
    meter = pyln.Meter(SR); out = []; seen = set()
    for sub in ("Nepali", "Sanskrit"):
        pqs = sorted(glob.glob(f"/work/tts/data/rasa/{sub}/*.parquet"))
        cnt_f = 0
        for f in pqs:
            pf = pq.ParquetFile(f)
            for g in range(pf.num_row_groups):
                for r in pf.read_row_group(g).to_pylist():
                    if r["gender"] != "Female" or r["filename"] in seen: continue
                    it = extract_rasa_clip(r, sub, meter)
                    if it:
                        out.append(it); seen.add(it["id"]); cnt_f += 1
                        if cnt_f >= max_per_subset: break
                if cnt_f >= max_per_subset: break
            if cnt_f >= max_per_subset: break
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(out, cache_path + ".tmp"); os.replace(cache_path + ".tmp", cache_path)

    def _bg():
        m2 = pyln.Meter(SR)
        for sub in ("Nepali", "Sanskrit"):
            cnt = 0
            for f in sorted(glob.glob(f"/work/tts/data/rasa/{sub}/*.parquet")):
                pf = pq.ParquetFile(f)
                for g in range(pf.num_row_groups):
                    for r in pf.read_row_group(g).to_pylist():
                        if r["filename"] in seen: continue
                        if r["gender"] != "Female" and cnt % 4 != 0: continue
                        it = extract_rasa_clip(r, sub, m2)
                        if it:
                            seen.add(it["id"]); out.append(it); cnt += 1
                            if tr_list is not None and it["split"] == "train":
                                wt = torch.from_numpy(it["wav"]).float() / 32767.0
                                gt = (gta_dict or {}).get(it["id"])
                                tup = (wt, torch.from_numpy(gt).float() if (gt is not None and not isinstance(gt, torch.Tensor)) else gt, it["spk"])
                                tr_list.append(tup)
                                if it["spk"] in (0, 2) and trf_list is not None: trf_list.append(tup)
                        if cnt >= 2500: break
                    if cnt >= 2500: break
                if cnt >= 2500: break
        torch.save(out, cache_path + ".tmp"); os.replace(cache_path + ".tmp", cache_path)
        print(f"BG cache expanded to {len(out)} clips", flush=True)

    if tr_list is not None:
        threading.Thread(target=_bg, daemon=True).start()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wavs", default="/work/tts/feats/v2_voc_wavs_24k.pt")
    ap.add_argument("--out", default="/work/tts/runs/v2_voc")
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--seg", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1.5e-4); ap.add_argument("--warm", type=int, default=1500)
    ap.add_argument("--minutes", type=float, default=210); ap.add_argument("--ema", type=float, default=0.995)
    ap.add_argument("--gta", default=""); ap.add_argument("--gta_p", type=float, default=0.7)
    ap.add_argument("--init", default=""); ap.add_argument("--compile", type=int, default=0); ap.add_argument("--gan_every", type=int, default=1); ap.add_argument("--val_every", type=int, default=250); ap.add_argument("--max_per_sub", type=int, default=2500)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    tr, va = [], []
    gta = torch.load(a.gta, weights_only=False) if (a.gta and os.path.exists(a.gta)) else {}
    SF = a.seg // HOP
    for f in sorted(glob.glob("/work/tts/feats/v2_voc_shards/*.pt")):
        try: d = torch.load(f, weights_only=False)
        except Exception: continue
        for r in d:
            g_t = gta.get(r["id"]) if gta else None
            if g_t is not None and not isinstance(g_t, torch.Tensor): g_t = torch.from_numpy(g_t)
            item = (r["wav"].to(torch.int16), g_t.half() if g_t is not None else None, r["spk"])
            if r["split"] == "val": va.append(item)
            elif item[0].shape[0] // HOP > SF + 1: tr.append(item)
    print(f"Sakhiya-v2-Voc: train={len(tr)} val={len(va)} gta={len(gta)}", flush=True)
    # ---- all training audio / GTA mels resident on GPU: batches are built with one gather, zero CPU work per step ----
    WAV = torch.cat([w for w, _, _ in tr]).to(dev); lens = torch.tensor([w.shape[0] // HOP for w, _, _ in tr], device=dev)
    starts = torch.zeros(len(tr), dtype=torch.long); starts[1:] = torch.cumsum(torch.tensor([w.shape[0] for w, _, _ in tr]), 0)[:-1]; starts = starts.to(dev)
    fem = torch.tensor([i for i, t in enumerate(tr) if t[2] in (0, 2)], device=dev)
    has_g = torch.tensor([t[1] is not None for t in tr], device=dev); gl = torch.tensor([t[1].shape[1] if t[1] is not None else 0 for t in tr], device=dev)
    gst = torch.zeros(len(tr), dtype=torch.long); gcat = [t[1] for t in tr if t[1] is not None]
    GM = torch.cat(gcat, 1).to(dev) if gcat else torch.zeros(100, 1, dtype=torch.half, device=dev)
    c = 0
    for i, t in enumerate(tr):
        gst[i] = c
        if t[1] is not None: c += t[1].shape[1]
    gst = gst.to(dev); ar_w = torch.arange(SF * HOP, device=dev); ar_f = torch.arange(SF, device=dev)
    print(f"GPU pools: wav {WAV.numel()*2/1e9:.2f} GB gta {GM.numel()*2/1e9:.2f} GB", flush=True)
    G = VocoderV2().to(dev); G_ema = copy.deepcopy(G).to(dev).eval(); D = DiscV2().to(dev)
    print("VocoderV2 params:", n_params(G), "DiscV2 params:", n_params(D), flush=True)
    oG = torch.optim.AdamW(G.parameters(), a.lr, betas=(0.8, 0.99))
    oD = torch.optim.AdamW(D.parameters(), a.lr, betas=(0.8, 0.99))
    step, t0, ck = 0, time.time(), f"{a.out}/last.pt"
    if os.path.exists(ck):
        s = torch.load(ck, map_location=dev, weights_only=False)
        G.load_state_dict(s["G"]); G_ema.load_state_dict(s.get("G_ema", s["G"]))
        if "D" in s: D.load_state_dict(s["D"])
        if "oG" in s and not a.init: oG.load_state_dict(s["oG"]); oD.load_state_dict(s["oD"])
        step = s.get("step", 0)
        print(f"Resumed Sakhiya-v2-Voc at step {step}", flush=True)
    elif a.init and os.path.exists(a.init):
        s = torch.load(a.init, map_location=dev, weights_only=False)
        G.load_state_dict(s.get("G_ema", s.get("G", s))); G_ema.load_state_dict(G.state_dict())
        if isinstance(s, dict) and "D" in s: D.load_state_dict(s["D"])
        print("Initialized from", a.init, flush=True)
    for g_ in oG.param_groups + oD.param_groups: g_["lr"] = a.lr
    spec_f0 = G.spectrum_and_f0
    if a.compile:
        spec_f0 = torch.compile(G.spectrum_and_f0); D_fwd = torch.compile(D)
    else: D_fwd = D

    def sample_batch():
        pick_f = torch.rand(a.bs, device=dev) < 0.75
        i = torch.where(pick_f, fem[torch.randint(0, len(fem), (a.bs,), device=dev)], torch.randint(0, len(tr), (a.bs,), device=dev))
        f = (torch.rand(a.bs, device=dev) * (lens[i] - SF - 1).float()).long()
        x_b = WAV[(starts[i] + f * HOP)[:, None] + ar_w[None]].float() / 32767.0
        xm = mel_spec_24k(x_b, dev)
        use_g = has_g[i] & (torch.rand(a.bs, device=dev) < a.gta_p) & (gl[i] >= f + SF)
        if bool(use_g.any()):
            gi = (gst[i] + f).clamp(max=GM.shape[1] - SF)[:, None] + ar_f[None]
            mg = GM[:, gi].permute(1, 0, 2).float()
            mel = torch.where(use_g[:, None, None], mg, xm)
        else: mel = xm
        return x_b, mel, xm

    @torch.no_grad()
    def run_val():
        G_ema.eval(); ls = []
        for w, _, _ in va[:60]:
            w_d = (w.float() / 32767.0)[: w.shape[0] // HOP * HOP][None].to(dev)
            m = mel_spec_24k(w_d, dev)
            y = G_ema(m)
            T = min(y.shape[1], w_d.shape[1])
            ls.append(F.l1_loss(mel_spec_24k(y[:, :T], dev), m[:, :, :T // HOP]).item())
        G.train(); return sum(ls) / max(len(ls), 1)

    best_val = run_val(); torch.save(G_ema.state_dict(), f"{a.out}/G_best.pt"); G.train(); D.train()
    while (time.time() - t0) < a.minutes * 60:
        step += 1
        prog = min(1.0, (time.time() - t0) / (a.minutes * 60)); cur_lr = a.lr * (0.08 + 0.92 * 0.5 * (1 + math.cos(math.pi * prog)))
        for g_ in oG.param_groups + oD.param_groups: g_["lr"] = cur_lr
        x, mel, xm = sample_batch()
        f0_gt, uv_gt = gpu_f0_uv(x, SF)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            re_v, im_v, f0_p, uv_p = spec_f0(mel)
        y = ola64(torch.complex(re_v.float(), im_v.float()), G.win)
        T = min(y.shape[1], x.shape[1]); y, x = y[:, :T], x[:, :T]
        use_gan = (step > a.warm) and (step % max(1, a.gan_every) == 0)
        d_loss_val = 0.0; fr_d = None
        if use_gan:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                dr, fr_d = D_fwd(x[:, None]); df, _ = D_fwd(y.detach()[:, None])
                ld = sum(F.relu(1.0 - r.float()).mean() + F.relu(1.0 + f.float()).mean() for r, f in zip(dr, df)) / len(dr)
            oD.zero_grad(set_to_none=True); ld.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), 10.0); oD.step(); d_loss_val = ld.item()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            ym = mel_spec_24k(y, dev)
            l_mel = 0.5 * F.l1_loss(ym, xm[:, :, :ym.shape[2]]) + 0.5 * F.l1_loss(ym, mel[:, :, :ym.shape[2]])
            l_stft = mr_complex_stft_loss(y, x, dev)
            l_f0 = F.l1_loss( torch.log(f0_p[:, :, :SF] + 1e-5) * uv_gt, torch.log(f0_gt + 1e-5) * uv_gt ) + (-(uv_gt.float() * torch.log(uv_p[:, :, :SF].float() + 1e-6) + (1 - uv_gt.float()) * torch.log(1 - uv_p[:, :, :SF].float() + 1e-6)).mean()) * 0.25
            lg = 45.0 * l_mel + 2.0 * l_stft + 2.0 * l_f0
            if use_gan:
                fr = [[t.detach() for t in fs] for fs in fr_d] if use_gan else None; df, ff = D_fwd(y[:, None])
                l_adv = sum(F.relu(1.0 - f.float()).mean() for f in df) / len(df)
                n_fm = sum(len(A) for A in fr)
                l_fm = sum(F.l1_loss(b.float(), a.float()) for A, B in zip(fr, ff) for a, b in zip(A, B)) / n_fm
                lg = lg + l_adv + 4.0 * l_fm
        oG.zero_grad(set_to_none=True); lg.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0); oG.step()
        with torch.no_grad():
            for pe, p in zip(G_ema.parameters(), G.parameters()): pe.lerp_(p, 1.0 - a.ema)
        if step % 50 == 0:
            vram = torch.cuda.memory_allocated() / 1e9 if dev == "cuda" else 0.0
            print(f"step {step} mel {l_mel.item():.4f} stft {l_stft.item():.3f} f0 {l_f0.item():.3f} D {d_loss_val:.3f} vram {vram:.1f}G {int(time.time()-t0)}s", flush=True)
        if step % a.val_every == 0:
            vl = run_val()
            print(f"VAL step {step} ema_mel_l1 {vl:.4f}", flush=True)
            torch.save(dict(G=G.state_dict(), G_ema=G_ema.state_dict(), D=D.state_dict(), oG=oG.state_dict(), oD=oD.state_dict(), step=step, val_mel_l1=vl), ck + ".tmp")
            os.replace(ck + ".tmp", ck)
            torch.save(G_ema.state_dict(), f"{a.out}/G.pt.tmp"); os.replace(f"{a.out}/G.pt.tmp", f"{a.out}/G.pt")
            if vl < best_val: best_val = vl; torch.save(G_ema.state_dict(), f"{a.out}/G_best.pt.tmp"); os.replace(f"{a.out}/G_best.pt.tmp", f"{a.out}/G_best.pt")
            json.dump(dict(step=step, mel_l1=float(l_mel.item()), val_mel_l1=float(vl), best_val=float(best_val), elapsed_s=int(time.time()-t0)), open(f"{a.out}/progress.json", "w"))

