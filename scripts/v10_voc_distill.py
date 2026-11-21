#!/usr/bin/env python3
"""v10: dense teacher->student VOCODER distillation -- attacks the real size wall.

Measured root cause of the >3 MB floor: every small tier ships the 5.04 MB V1 vocoder (Atom's acoustic model is only
1.46 MB int8). The earlier tiny vocoders (v6-v8: 0.30-0.45 M params, GAN-first, 13-20 min) sounded robotic (UTMOS 1.9-2.6).
What is different here:
  * one shared 1.2 M-param student (ch 144, the SAME iSTFTNet graph -> same ONNX + browser path), 2.6x the old tiny
    vocoders, 4x smaller than V1;
  * DENSE supervision: the student regresses the teacher's own complex iSTFT spectrum (re/im at hop 16), its log-magnitude
    and teacher F0/voicing on the SAME input mel (a deterministic function -> nothing to average -> no blur/buzz), plus
    waveform mel-L1 and multi-resolution complex STFT;
  * inputs = ground-truth mels + the mels the Nano and Atom acoustic students really produce (export_mel interface);
  * light MPD/MRD hinge GAN + feature matching only after --gan_after steps (crispness); no SLM disc (speed);
  * hard --max-minutes cap (<=110), step-50 throughput gate, checkpoint by UTMOS on the real text->AM->vocoder path.
"""
import argparse, copy, json, math, os, random, sys, time
import torch, torch.nn.functional as F
R = "/work/tts"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from sakhiya.v2_vocoder import VocoderV2, mel_spec_24k, ola64  # noqa: E402
from scripts.train_v2_voc import DiscV2, mr_complex_stft_loss  # noqa: E402
from scripts.train_v3_voc import resamp16  # noqa: E402
from scripts.v2_full_eval import load_utmos  # noqa: E402
from scripts.v8_voc_gta import load_am, export_mel, val_mels, cache_student  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ch", type=int, default=144)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-minutes", type=float, default=80.0)
    ap.add_argument("--teacher_voc", default=f"{R}/runs/v5_c1/voc.pt")
    ap.add_argument("--am_voca", default=f"{R}/runs/v9_voca_am/am.pt")
    ap.add_argument("--am_sano", default=f"{R}/runs/v9_sano_am/am.pt")
    ap.add_argument("--init", default=None)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--seg", type=int, default=64)
    ap.add_argument("--p_student", type=float, default=0.55)
    ap.add_argument("--gan_after", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--eval_every", type=int, default=1500)
    ap.add_argument("--smoke", type=int, default=0, help="run N steps + one score pass, then exit")
    a = ap.parse_args()
    assert a.max_minutes <= 110, "stages are capped under 2 h"
    out = a.out or f"{R}/runs/v10_voc{a.ch}"; os.makedirs(out, exist_ok=True)
    dev = "cuda"; tag = f"v10-voc{a.ch}"
    torch.backends.cudnn.benchmark = True
    D = torch.load(f"{R}/feats/v6_expressive_vedic_24k.pt", weights_only=False)
    mean, std = float(D.get("mean", -5.5028)), float(D.get("std", 2.1869))
    items = [x for x in D["items"] if x.get("split") == "train" and len(x["ids"]) >= 3 and 24 <= x["mel"].shape[1] <= 850]
    gt = [x["mel"].half().to(dev) for x in items]
    t_voc = VocoderV2().to(dev).train(False)
    vck = torch.load(a.teacher_voc, map_location=dev, weights_only=False)
    t_voc.load_state_dict({k: v.float() for k, v in vck.get("G_ema", vck.get("G", vck)).items()}, strict=False)
    for p in t_voc.parameters(): p.requires_grad_(False)
    pools, vms, am_steps = [gt], {}, {}
    for name, path in (("nano", a.am_voca), ("atom", a.am_sano)):
        s_am, st = load_am("voca" if name == "nano" else "sano", path, dev, mean, std)
        pools.append(cache_student(s_am, items, mean, std, dev)); vms[name] = val_mels(s_am, dev, mean, std); am_steps[name] = st
        del s_am
    torch.cuda.empty_cache()
    s_voc = VocoderV2(ch=a.ch, res_kernels=(3, 7, 11)).to(dev)
    step0, ik = 0, {}
    if a.init and os.path.exists(a.init):
        ik = torch.load(a.init, map_location=dev, weights_only=False); s_voc.load_state_dict(ik.get("G", ik)); step0 = int(ik.get("step", 0))
    if not ik:   # start the pitch tracker from V1 exactly (same 128-ch f0_net in teacher and student): no F0 jitter at step 0
        s_voc.src.f0_net.load_state_dict(t_voc.src.f0_net.state_dict())
    n_par = sum(p.numel() for p in s_voc.parameters())
    print(f"[{tag}] items={len(items)} params={n_par} am_steps={am_steps} init={a.init}", flush=True)
    ema = copy.deepcopy(s_voc).train(False)
    if "G_ema" in ik: ema.load_state_dict(ik["G_ema"])
    utmos = load_utmos(dev)

    def score_utmos(voc):
        r = {}
        with torch.no_grad():
            for name, vm in vms.items():
                sc = []
                for m in vm:
                    w = voc(export_mel(m[None]))
                    sc.append(float(utmos(resamp16(w.float()), torch.full((1,), 16000, device=dev)).mean()))
                r[name] = round(sum(sc) / len(sc), 4)
        r["mean"] = round((r["nano"] + r["atom"]) / 2, 4)
        return r

    base = score_utmos(t_voc)
    print(f"[{tag}] TEACHER(V1 5.0MB) on student mels utmos={json.dumps(base)}", flush=True)
    D_gan = DiscV2().to(dev)
    opt = torch.optim.AdamW(s_voc.parameters(), lr=a.lr, betas=(0.8, 0.99), weight_decay=1e-5)
    oD = torch.optim.AdamW(D_gan.parameters(), lr=2e-4, betas=(0.8, 0.99))
    best, hist = -1.0, []
    t0, budget, step = time.time(), a.max_minutes * 60.0, step0
    L = a.seg * 256
    while (time.time() - t0) < budget:
        frac = (time.time() - t0) / budget
        lr = a.lr * min(1.0, (step - step0 + 1) / 500.0) * (0.5 * (1 + math.cos(math.pi * frac)) * 0.9 + 0.1)
        for g_ in opt.param_groups: g_["lr"] = lr
        step += 1
        if step == step0 + 50:
            sps = 50.0 / (time.time() - t0)
            print(f"[GATE] step-50 throughput {sps:.2f} steps/s -> ~{sps * budget / 1000:.0f}k steps inside the {a.max_minutes:.0f} min cap", flush=True)
            if sps < 2.0: print("[GATE] too slow, aborting", flush=True); sys.exit(3)
        mi = []
        while len(mi) < a.bs:
            i = random.randrange(len(items))
            src = pools[0] if random.random() > a.p_student else pools[random.choice((1, 2))]
            m = src[i]
            if m is None or m.shape[1] < a.seg: continue
            s = random.randint(0, m.shape[1] - a.seg); mi.append(m[:, s:s + a.seg].float())
        x = export_mel(torch.stack(mi))
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            re_t, im_t, f0_t, uv_t = t_voc.spectrum_and_f0(x)
        re_t, im_t = re_t.float(), im_t.float()
        wav_t = ola64(torch.complex(re_t, im_t), t_voc.win)[:, :L]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            re_s, im_s, f0_s, uv_s = s_voc.spectrum_and_f0(x)
        re_s, im_s = re_s.float(), im_s.float()
        wav_s = ola64(torch.complex(re_s, im_s), s_voc.win)[:, :L]
        use_gan = (step - step0) > a.gan_after
        if use_gan:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                dr, _ = D_gan(wav_t[:, None]); df, _ = D_gan(wav_s.detach()[:, None])
                l_d = sum(F.relu(1.0 - r.float()).mean() + F.relu(1.0 + f.float()).mean() for r, f in zip(dr, df)) / len(dr)
            oD.zero_grad(set_to_none=True); l_d.backward(); oD.step()
        mag_t = torch.sqrt(re_t ** 2 + im_t ** 2 + 1e-9); mag_s = torch.sqrt(re_s ** 2 + im_s ** 2 + 1e-9)
        l_lmag = F.l1_loss(torch.log(mag_s + 1e-5), torch.log(mag_t + 1e-5))
        l_cplx = ((re_s - re_t).abs() + (im_s - im_t).abs()).mean() / (mag_t.mean() + 1e-6)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            l_mel = F.l1_loss(mel_spec_24k(wav_s, dev)[:, :, :a.seg].float(), mel_spec_24k(wav_t, dev)[:, :, :a.seg].float())
            l_stft = mr_complex_stft_loss(wav_s, wav_t, dev)
        w_v = uv_t.detach().float()
        l_f0 = ((torch.log2(f0_s.float().clamp(min=60)) - torch.log2(f0_t.float().clamp(min=60))).abs() * w_v).sum() / w_v.sum().clamp(min=1.0)
        l_uv = F.l1_loss(uv_s.float(), w_v)
        loss = 45.0 * l_mel + 2.5 * l_stft + 3.0 * l_lmag + 4.0 * l_cplx + 72.0 * l_f0 + 24.0 * l_uv
        if use_gan:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad(): _, fr_r = D_gan(wav_t[:, None])
                df_g, fr_g = D_gan(wav_s[:, None])
                l_adv = sum(F.relu(1.0 - f.float()).mean() for f in df_g) / len(df_g)
                l_fm = sum(F.l1_loss(b_.float(), a_.float()) for A, B in zip(fr_r, fr_g) for a_, b_ in zip(A, B)) / len(fr_r)
            loss = loss + 1.0 * l_adv + 2.0 * l_fm
        opt.zero_grad(set_to_none=True); loss.backward()
        if torch.isfinite(torch.nn.utils.clip_grad_norm_(s_voc.parameters(), 1.0)): opt.step()
        with torch.no_grad():
            k = 1.0 - min(0.998, (1.0 + step - step0) / (10.0 + step - step0))
            for pe, p in zip(ema.parameters(), s_voc.parameters()): pe.lerp_(p, k)
        if (step - step0) % 250 == 0:
            print(f"[{tag}] step={step} {(step - step0) / (time.time() - t0):.1f} st/s t={(time.time() - t0) / 60:.1f}m lr={lr:.2e} mel={l_mel.item():.3f} stft={l_stft.item():.3f} lmag={l_lmag.item():.3f} cplx={l_cplx.item():.3f} f0={l_f0.item():.4f} uv={l_uv.item():.3f} gan={int(use_gan)}", flush=True)
        done_smoke = a.smoke and (step - step0) >= a.smoke
        if (step - step0) % a.eval_every == 0 or done_smoke:
            cur = score_utmos(ema); hist.append(cur["mean"])
            ck = dict(G_ema=ema.state_dict(), G=s_voc.state_dict(), step=step, ch=a.ch, res_kernels=(3, 7, 11), utmos=cur, teacher=base, am_steps=am_steps)
            torch.save(ck, f"{out}/last.pt.tmp"); os.replace(f"{out}/last.pt.tmp", f"{out}/last.pt")
            if cur["mean"] > best:
                best = cur["mean"]; torch.save(ck, f"{out}/voc.pt.tmp"); os.replace(f"{out}/voc.pt.tmp", f"{out}/voc.pt")
            print(f"[{tag}] SCORE step={step} utmos={json.dumps(cur)} best_mean={best:.3f} teacher_mean={base['mean']:.3f}", flush=True)
            if done_smoke: break
    json.dump(dict(best_mean=best, hist=hist, teacher=base, steps=step - step0, params=n_par, am_steps=am_steps), open(f"{out}/summary.json", "w"))
    print(f"[{tag}] DONE best_mean={best:.3f} teacher_mean={base['mean']:.3f} steps={step - step0}", flush=True)


if __name__ == "__main__":
    main()
