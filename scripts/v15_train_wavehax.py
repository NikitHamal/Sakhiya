#!/usr/bin/env python3
"""Sakhiya v15 Phase 2: Train MS-RND-Wavehax-Lite (<= 450 KB int8) & MS-RND-Wavehax-Pro (<= 900 KB int8).
Features:
1. 100% Pure Studio Sakhiya Timbre (v5_c1/voc.pt on v6_expressive_vedic_24k.pt — ZERO noisy IA audio!).
2. Pre-cached exact export_mel() triplets (30% V1 studio mel + 35% Nano v9 mel + 35% Atom v9 mel) with shift_st variation.
3. Dense supervision on 33-bin complex spectrum (re, im), log-magnitude, instantaneous phase/group-delay, F0/UV,
   100-band Mel L1, Multi-Resolution Complex STFT, Frozen WavLM-base-plus perceptual feature matching (every 2nd step),
   and MPD+MRD GAN + feature matching.
4. Straight-Through Estimator (STE) Per-Channel INT8 QAT polish in the final 20% of the budget.
5. Hard --max-minutes cap + Step-50 throughput gate + real text->AM->vocoder UTMOS checkpointing."""
import argparse, copy, json, math, os, random, sys, time
import torch, torch.nn.functional as F

R = "/work/tts"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from sakhiya.v15_rnd_wavehax import MSRNDWavehax
from sakhiya.v2_vocoder import VocoderV2, mel_spec_24k, ola64
from sakhiya.v4_slm import WavLMSLMHelper
from scripts.train_v2_voc import DiscV2, mr_complex_stft_loss
from scripts.train_v3_voc import resamp16
from scripts.v2_full_eval import load_utmos
from scripts.v8_voc_gta import load_am, export_mel, val_mels, cache_student


def spec_phase_gd_loss(re_s, im_s, re_t, im_t, mag_s, mag_t):
    """Wrap-free unit-phasor Phase + Instantaneous-Frequency (time diff) + Group-Delay (freq diff) loss."""
    w = (mag_t / (mag_t.mean(dim=(1, 2), keepdim=True) + 1e-5)).clamp(max=3.0).detach()
    nr = re_s / (mag_s + 1e-5); ni = im_s / (mag_s + 1e-5)
    tr = re_t / (mag_t + 1e-5); ti = im_t / (mag_t + 1e-5)
    l_cos = ((1.0 - (nr * tr + ni * ti)) * w).mean()
    if_sr = nr[:, :, 1:] * nr[:, :, :-1] + ni[:, :, 1:] * ni[:, :, :-1]
    if_si = ni[:, :, 1:] * nr[:, :, :-1] - nr[:, :, 1:] * ni[:, :, :-1]
    if_tr = tr[:, :, 1:] * tr[:, :, :-1] + ti[:, :, 1:] * ti[:, :, :-1]
    if_ti = ti[:, :, 1:] * tr[:, :, :-1] - tr[:, :, 1:] * ti[:, :, :-1]
    l_if = ((1.0 - (if_sr * if_tr + if_si * if_ti)) * w[:, :, 1:]).mean()
    gd_sr = nr[:, 1:, :] * nr[:, :-1, :] + ni[:, 1:, :] * ni[:, :-1, :]
    gd_si = ni[:, 1:, :] * nr[:, :-1, :] - nr[:, 1:, :] * ni[:, :-1, :]
    gd_tr = tr[:, 1:, :] * tr[:, :-1, :] + ti[:, 1:, :] * ti[:, :-1, :]
    gd_ti = ti[:, 1:, :] * tr[:, :-1, :] - tr[:, 1:, :] * ti[:, :-1, :]
    l_gd = ((1.0 - (gd_sr * gd_tr + gd_si * gd_ti)) * w[:, 1:, :]).mean()
    return (l_cos + l_if + l_gd) * 0.3333


def warm_init_f0(s_voc, t_voc, pools, dev):
    """Copy (if c_f0==128) or warm-slice + 250-step fast distill (if c_f0<128) f0_net from V1 teacher vocoder."""
    with torch.no_grad():
        for (ns, ps), (nt, pt) in zip(s_voc.src.f0_net.named_parameters(), t_voc.src.f0_net.named_parameters()):
            if ps.shape == pt.shape:
                ps.copy_(pt)
            elif ps.ndim == pt.ndim:
                sl = tuple(slice(0, b) for b in ps.shape)
                ps.copy_(pt[sl])
    if s_voc.src.c_f0 == 128:
        print("[Wavehax-F0] Exact 128-ch f0_net copied 1:1 from V1 teacher.", flush=True)
        return
    opt_f0 = torch.optim.AdamW(s_voc.src.f0_net.parameters(), lr=2e-3)
    gt = pools[0]
    for st in range(250):
        mi = []
        while len(mi) < 64:
            m = gt[random.randrange(len(gt))]
            if m is not None and m.shape[1] >= 64:
                s = random.randint(0, m.shape[1] - 64)
                mi.append(m[:, s:s + 64].float())
        x = export_mel(torch.stack(mi))
        with torch.no_grad():
            rt = t_voc.src.f0_net(x).float()
        rs = s_voc.src.f0_net(x).float()
        loss_f0 = F.l1_loss(rs, rt)
        opt_f0.zero_grad(set_to_none=True)
        loss_f0.backward()
        opt_f0.step()
    print(f"[Wavehax-F0] Warmed {s_voc.src.c_f0}-ch f0_net in 250 steps (l1={loss_f0.item():.4f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="lite", choices=["lite", "pro"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-minutes", type=float, default=45.0)
    ap.add_argument("--teacher_voc", default=f"{R}/runs/v5_c1/voc.pt")
    ap.add_argument("--am_voca", default=f"{R}/ck/v9_voca_am.pt")
    ap.add_argument("--am_sano", default=f"{R}/ck/v9_sano_am.pt")
    ap.add_argument("--init", default=None)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seg", type=int, default=64)
    ap.add_argument("--gan_after", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    assert a.max_minutes <= 115.0, "hard stage cap <= 115 min"

    out = a.out or f"{R}/v15/wavehax_{a.variant}"
    os.makedirs(out, exist_ok=True)
    dev = "cuda"
    tag = f"v15-wavehax-{a.variant}"
    torch.backends.cudnn.benchmark = True

    D = torch.load(f"{R}/feats/v6_expressive_vedic_24k.pt", weights_only=False)
    mean, std = float(D.get("mean", -5.5028)), float(D.get("std", 2.1869))
    items = [x for x in D["items"] if x.get("split") == "train" and len(x["ids"]) >= 3 and 24 <= x["mel"].shape[1] <= 850]
    gt = [x["mel"].half().to(dev) for x in items]

    t_voc = VocoderV2().to(dev).train(False)
    vck = torch.load(a.teacher_voc, map_location=dev, weights_only=False)
    t_voc.load_state_dict({k: v.float() for k, v in vck.get("G_ema", vck.get("G", vck)).items()}, strict=False)
    for p in t_voc.parameters():
        p.requires_grad_(False)

    cache_p = f"{R}/v15/student_mel_cache.pt"
    if os.path.exists(cache_p):
        print(f"[{tag}] Loading cached student mels from {cache_p}", flush=True)
        c_obj = torch.load(cache_p, weights_only=False)
        pools = [gt, [m.to(dev) if m is not None else None for m in c_obj["nano"]], [m.to(dev) if m is not None else None for m in c_obj["atom"]]]
        vms = {k: [m.to(dev) for m in v] for k, v in c_obj["vms"].items()}
        am_steps = c_obj["am_steps"]
    else:
        pools, vms, am_steps = [gt], {}, {}
        for name, path in (("nano", a.am_voca), ("atom", a.am_sano)):
            s_am, st = load_am("voca" if name == "nano" else "sano", path, dev, mean, std)
            pools.append(cache_student(s_am, items, mean, std, dev))
            vms[name] = val_mels(s_am, dev, mean, std)
            am_steps[name] = st
            del s_am
        os.makedirs(f"{R}/v15", exist_ok=True)
        torch.save(dict(
            nano=[m.cpu() if m is not None else None for m in pools[1]],
            atom=[m.cpu() if m is not None else None for m in pools[2]],
            vms={k: [m.cpu() for m in v] for k, v in vms.items()},
            am_steps=am_steps,
        ), cache_p)
        print(f"[{tag}] Saved student mel cache to {cache_p}", flush=True)
    torch.cuda.empty_cache()

    s_voc = MSRNDWavehax(variant=a.variant).to(dev)
    step0, ik = 0, {}
    if a.init and os.path.exists(a.init):
        ik = torch.load(a.init, map_location=dev, weights_only=False)
        s_voc.load_state_dict(ik.get("G", ik.get("G_ema", ik)), strict=False)
        step0 = int(ik.get("step", 0))
    else:
        warm_init_f0(s_voc, t_voc, pools, dev)

    n_par = sum(p.numel() for p in s_voc.parameters())
    print(f"[{tag}] items={len(items)} params={n_par:,} am_steps={am_steps}", flush=True)
    ema = copy.deepcopy(s_voc).train(False)
    if "G_ema" in ik:
        ema.load_state_dict(ik["G_ema"], strict=False)

    utmos = load_utmos(dev)
    slm = WavLMSLMHelper(dev)
    D_gan = DiscV2().to(dev)

    def score_utmos(voc):
        r = {}
        with torch.no_grad():
            for name, vm in vms.items():
                sc = []
                for m in vm:
                    w = voc(export_mel(m[None]))
                    sc.append(float(utmos(resamp16(w.float()), torch.full((1,), 16000, device=dev)).mean()))
                r[name] = round(sum(sc) / len(sc), 4)
        r["mean"] = round((r["nano"] + r["atom"]) / 2.0, 4)
        return r

    base = score_utmos(t_voc)
    print(f"[{tag}] TEACHER(V1 5.04MB) on student mels utmos={json.dumps(base)}", flush=True)

    opt = torch.optim.AdamW(s_voc.parameters(), lr=a.lr, betas=(0.8, 0.99), weight_decay=1e-5)
    oD = torch.optim.AdamW(D_gan.parameters(), lr=2e-4, betas=(0.8, 0.99))
    best, hist = -1.0, []
    t0, budget, step = time.time(), a.max_minutes * 60.0, step0
    L = a.seg * 256
    qat_active = False

    while (time.time() - t0) < budget:
        frac = (time.time() - t0) / budget
        if frac >= 0.80 and not qat_active:
            s_voc.set_qat(True)
            qat_active = True
            print(f"[{tag}] step={step} Enabled Per-Channel INT8 STE QAT for final 20% polish!", flush=True)

        lr = a.lr * min(1.0, (step - step0 + 1) / 400.0) * (0.5 * (1.0 + math.cos(math.pi * frac)) * 0.9 + 0.1)
        for g_ in opt.param_groups:
            g_["lr"] = lr
        step += 1

        if step == step0 + 50:
            sps = 50.0 / max(time.time() - t0, 1e-6)
            print(f"[GATE] step-50 throughput {sps:.2f} steps/s -> ~{sps * budget / 1000:.1f}k steps inside {a.max_minutes:.0f} min cap", flush=True)
            if sps < 4.0:
                print("[GATE] throughput too low, aborting", flush=True)
                sys.exit(3)

        mi = []
        while len(mi) < a.bs:
            i = random.randrange(len(items))
            u_r = random.random()
            src = pools[0] if u_r < 0.30 else (pools[1] if u_r < 0.65 else pools[2])
            m = src[i]
            if m is None or m.shape[1] < a.seg:
                continue
            s = random.randint(0, m.shape[1] - a.seg)
            mi.append(m[:, s:s + a.seg].float())
        x = export_mel(torch.stack(mi))
        # Random continuous shift_st in [-0.75, 0.75] semitones on 40% of batch items for Vedic Svara / Chandas robustness
        if random.random() < 0.40:
            sh_st = (torch.rand((a.bs, 1, 1), device=dev) - 0.5) * 1.50
        else:
            sh_st = None

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            re_t, im_t, f0_t, uv_t = t_voc.spectrum_and_f0(x, shift_st=sh_st)
        re_t, im_t = re_t.float(), im_t.float()
        wav_t = s_voc.spec_to_wav(re_t, im_t)[:, :L]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            re_s, im_s, f0_s, uv_s = s_voc.spectrum_and_f0(x, shift_st=sh_st)
        re_s, im_s = re_s.float(), im_s.float()
        wav_s = s_voc.spec_to_wav(re_s, im_s)[:, :L]

        use_gan = (step - step0) > a.gan_after
        b_gan = min(16, a.bs)
        if use_gan:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                dr, _ = D_gan(wav_t[:b_gan, None])
                df, _ = D_gan(wav_s[:b_gan].detach()[:, None])
                l_d = sum(F.relu(1.0 - r.float()).mean() + F.relu(1.0 + f.float()).mean() for r, f in zip(dr, df)) / len(dr)
            oD.zero_grad(set_to_none=True)
            l_d.backward()
            oD.step()

        mag_t = torch.sqrt(re_t ** 2 + im_t ** 2 + 1e-9)
        mag_s = torch.sqrt(re_s ** 2 + im_s ** 2 + 1e-9)
        l_lmag = F.l1_loss(torch.log(mag_s + 1e-5), torch.log(mag_t + 1e-5))
        l_cplx = ((re_s - re_t).abs() + (im_s - im_t).abs()).mean() / (mag_t.mean() + 1e-6)
        l_ph = spec_phase_gd_loss(re_s, im_s, re_t, im_t, mag_s, mag_t)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            l_mel = F.l1_loss(mel_spec_24k(wav_s, dev)[:, :, :a.seg].float(), mel_spec_24k(wav_t, dev)[:, :, :a.seg].float())
            l_stft = mr_complex_stft_loss(wav_s, wav_t, dev)
            # Frozen WavLM-base-plus perceptual feature + SV timbre matching every 2nd step on 8-clip sub-batch
            if step % 2 == 0:
                b_slm = min(8, a.bs)
                with torch.no_grad():
                    ft_t, emb_t = slm.extract(resamp16(wav_t[:b_slm].float()))
                ft_s, emb_s = slm.extract(resamp16(wav_s[:b_slm].float()))
                l_wavlm = F.l1_loss(ft_s.float(), ft_t.float()) + 0.5 * (1.0 - F.cosine_similarity(emb_s.float(), emb_t.float(), dim=-1).mean())
            else:
                l_wavlm = torch.tensor(0.0, device=dev)

        w_v = uv_t.detach().float()
        l_f0 = ((torch.log2(f0_s.float().clamp(min=60.0)) - torch.log2(f0_t.float().clamp(min=60.0))).abs() * w_v).sum() / w_v.sum().clamp(min=1.0)
        l_uv = F.l1_loss(uv_s.float(), w_v)

        loss = 45.0 * l_mel + 2.5 * l_stft + 3.5 * l_lmag + 4.5 * l_cplx + 1.5 * l_ph + 65.0 * l_f0 + 20.0 * l_uv + 4.0 * l_wavlm
        if use_gan:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    _, fr_r = D_gan(wav_t[:b_gan, None])
                df_g, fr_g = D_gan(wav_s[:b_gan, None])
                l_adv = sum(F.relu(1.0 - f.float()).mean() for f in df_g) / len(df_g)
                l_fm = sum(F.l1_loss(b_.float(), a_.float()) for A, B in zip(fr_r, fr_g) for a_, b_ in zip(A, B)) / len(fr_r)
            loss = loss + 1.0 * l_adv + 2.0 * l_fm

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if torch.isfinite(torch.nn.utils.clip_grad_norm_(s_voc.parameters(), 1.0)):
            opt.step()

        with torch.no_grad():
            k = 1.0 - min(0.998, (1.0 + step - step0) / (10.0 + step - step0))
            for pe, p in zip(ema.parameters(), s_voc.parameters()):
                pe.lerp_(p, k)

        if (step - step0) % 250 == 0:
            print(f"[{tag}] step={step} {(step - step0) / (time.time() - t0):.1f} st/s t={(time.time() - t0) / 60:.1f}m lr={lr:.2e} mel={l_mel.item():.3f} stft={l_stft.item():.3f} lmag={l_lmag.item():.3f} cplx={l_cplx.item():.3f} ph={l_ph.item():.3f} f0={l_f0.item():.4f} wavlm={l_wavlm.item():.3f} qat={int(qat_active)} gan={int(use_gan)}", flush=True)

        done_smoke = a.smoke and (step - step0) >= a.smoke
        if (step - step0) % a.eval_every == 0 or done_smoke:
            cur = score_utmos(ema)
            hist.append(cur["mean"])
            ck = dict(
                arch="ms_rnd_wavehax", variant=a.variant,
                G_ema=ema.state_dict(), G=s_voc.state_dict(),
                step=step, utmos=cur, teacher=base, am_steps=am_steps, params=n_par,
            )
            torch.save(ck, f"{out}/last.pt.tmp")
            os.replace(f"{out}/last.pt.tmp", f"{out}/last.pt")
            if cur["mean"] > best:
                best = cur["mean"]
                torch.save(ck, f"{out}/voc.pt.tmp")
                os.replace(f"{out}/voc.pt.tmp", f"{out}/voc.pt")
            print(f"[{tag}] SCORE step={step} utmos={json.dumps(cur)} best_mean={best:.3f} teacher_mean={base['mean']:.3f}", flush=True)
            if done_smoke:
                break

    json.dump(dict(variant=a.variant, best_mean=best, hist=hist, teacher=base, steps=step - step0, params=n_par, am_steps=am_steps), open(f"{out}/summary.json", "w"), indent=2)
    print(f"[{tag}] DONE best_mean={best:.3f} teacher_mean={base['mean']:.3f} steps={step - step0}", flush=True)


if __name__ == "__main__":
    main()
