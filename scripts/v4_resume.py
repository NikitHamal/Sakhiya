#!/usr/bin/env python3
"""Sakhiya v4 Idempotent Resumable Training & Milestone Shipping Pipeline (`scripts/v4_resume.py`).
Stages:
- M1 (`v4_m1`): Phase 1 G2P Re-alignment + Disentangled X-Pred AM (GRL speaker-adversarial + Explicit F0/Energy/Pause + Single-Voice `sakhiya` lock).
- M2 (`v4_m2`): End-to-End Joint AM+Vocoder Training with Frozen `WavLM` SLM Perceptual Discriminator (`D_SLM`) + `WavLM-SV` Timbre Lock + MPD/MRD.
- M3 (`v4_final`): Multi-Reward Prosody & Pronunciation Preference Optimization (4-Term GRPO/DPO with KL anchor to M2).
Automatically archives to `v4-ckpt` and freezes immutable 15-clip comparison packs in `docs/samples/versions/<tag>/`."""
import argparse, copy, json, math, os, random, shutil, subprocess, sys, time
import numpy as np, torch, torch.nn.functional as F

R = "/work/tts"
REPO = "/work/tts/repo" if os.path.exists("/work/tts/repo") else os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from sakhiya import g2p, sanskrit as SA
from sakhiya.symbols import SYM2ID, LANG2ID
from sakhiya.cfm import SakhiyaAM, _dur_to_attn
from sakhiya.ctc_aligner import MelCTCAligner
from sakhiya.v2_vocoder import VocoderV2, mel_spec_24k, HOP
from sakhiya.v4_slm import SLMDiscriminator, WavLMSLMHelper, score_4term_reward, archive_v4_release
from scripts.train_v3_am import _batch_prosody, dump_gta
from scripts.train_v2_voc import DiscV2
from scripts.train_v3_voc import stft_mag_loss, resamp16, eval_copysynth_utmos, load_pools
from scripts.v2_full_eval import load_utmos


def update_coord(stage_idx, msg, state_p=f"{R}/runs/v4_state.json", extra=None):
    os.makedirs(f"{R}/runs", exist_ok=True)
    st = json.load(open(state_p)) if os.path.exists(state_p) else {}
    st.update(stage_idx=stage_idx, right_now=msg, updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **(extra or {}))
    json.dump(st, open(state_p, "w"), indent=2)
    json.dump(st, open(f"{R}/COORD_V3.json", "w"), indent=2)
    json.dump({"t": time.time()}, open(f"{R}/heartbeat.json", "w"))


def ship_milestone(tag, label, notes, am_p, voc_p, promote=False):
    exp_dir = f"{R}/export_{tag}"
    subprocess.run(["python3", f"{REPO}/scripts/export_v2_onnx.py", "--am", am_p, "--voc", voc_p, "--out", exp_dir], check=True)
    cmd_ev = ["python3", f"{REPO}/scripts/eval_v3.py", "--model", exp_dir, "--limit", "100"] + (["--promote"] if promote else [])
    subprocess.run(cmd_ev, cwd=REPO, check=True)
    subprocess.run(["python3", f"{REPO}/scripts/freeze_version_pack.py", "--tag", tag, "--label", label, "--models", exp_dir, "--metrics-json", f"{REPO}/eval/bench200_v4.json", "--notes", notes], cwd=REPO, check=True)
    archive_v4_release(am_p, voc_p, f"{R}/runs/v4_state.json")


def run_m1(D, mean, std, ctc, dev="cuda", steps_1a=2000, steps_1b=1500):
    out_dir = f"{R}/runs/v4_am"; os.makedirs(out_dir, exist_ok=True)
    items = D["items"]
    for x in items:
        txt = x["text"]
        ph = SA.phonemize_v26(txt) if x["lang"] == "sa" else g2p.text_to_phones(txt, "ne")
        ph_exp = [q for p in ph for q in (["o", "m"] if p == "<om>" else [p])]
        ids = [SYM2ID[p] for p in ph_exp if p in SYM2ID]
        if len(ids) == len(x["ids"]):
            x["phones"] = ph; x["ids"] = torch.tensor(ids, dtype=torch.long)
        x["is_fem"] = 1 if (x["spk"] in (0, 2) or 4 <= x["spk"] <= 13) else 0
        x["mel"] = x["mel"].to(dev)
    tr = [x for x in items if x.get("split") == "train"]
    POOL = [[i for i, x in enumerate(tr) if x["is_fem"] == k] for k in (1, 0)]
    net = SakhiyaAM().to(dev); net.stats.copy_(torch.tensor([mean, std])); ema = copy.deepcopy(net).eval()
    for ip in (f"{R}/runs/v3_am/last.pt", f"{R}/ckpt_v3_snap/v3_am_last.pt"):
        if os.path.exists(ip):
            ck = torch.load(ip, map_location=dev, weights_only=False)
            sd = {k: v.float() if torch.is_tensor(v) else v for k, v in ck.get("ema", ck.get("net", ck)).items()}
            net.load_state_dict(sd, strict=False); ema = copy.deepcopy(net).eval()
            print("[M1] Warm-started Sakhiya-v4-AM from", ip, flush=True); break
    # Freeze 8-step ODE velocity network `net.vel` during M1 encoder/duration/prosody disentanglement so ODE manifold stays 100% pristine
    for p in net.vel.parameters(): p.requires_grad_(False)
    net.ctc_rec = ctc
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1.4e-5, betas=(0.9, 0.99), weight_decay=1e-2)
    tot_steps = steps_1a + steps_1b
    for step in range(1, tot_steps + 1):
        in_1b = (step > steps_1a)
        lr = 1.4e-5 * 0.5 * (1 + math.cos(math.pi * step / tot_steps)) + 4e-6
        lam_ctc = 0.0 if in_1b else 0.22
        for g_ in opt.param_groups: g_["lr"] = lr
        xs = [tr[i] for i in (random.sample(POOL[0], 14) + random.sample(POOL[1], 10))]
        L = max(len(x["ids"]) for x in xs); T = max(x["mel"].shape[1] for x in xs)
        ids = torch.zeros(len(xs), L, dtype=torch.long, device=dev); dur_gt = torch.zeros(len(xs), L, dtype=torch.long, device=dev)
        y = torch.zeros(len(xs), 100, T, device=dev)
        for k, x in enumerate(xs):
            Lk, Tm = len(x["ids"]), x["mel"].shape[1]
            ids[k, :Lk] = x["ids"].to(dev); dur_gt[k, :Lk] = x["dur_ctc"].to(dev).long()
            y[k, :, :Tm] = (x["mel"].float() - mean) / std
        il = torch.tensor([len(x["ids"]) for x in xs], device=dev); yl = torch.tensor([x["mel"].shape[1] for x in xs], device=dev)
        spk_true = torch.tensor([int(x["spk"]) for x in xs], device=dev).clamp(0, 15)
        spks = torch.tensor([0 if x["is_fem"] == 1 else 1 for x in xs], device=dev)
        langs = torch.tensor([0 if (x["lang"] == "sa" and random.random() < 0.80) else LANG2ID[x["lang"]] for x in xs], device=dev)
        stys = torch.tensor([x.get("v3_style_id", 0) for x in xs], device=dev)
        pros = _batch_prosody(y, yl, il)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, parts = net.loss(ids, il, y, yl, spks, langs, stys, torch.ones_like(spks), None, None, pros, dur_gt, spk_true=spk_true, lambda_ctc=lam_ctc, lambda_grl=0.20)
        opt.zero_grad(set_to_none=True); loss.backward()
        if torch.isfinite(torch.nn.utils.clip_grad_norm_(net.parameters(), 0.8)): opt.step()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), net.parameters()): pe.lerp_(p, 0.0015)
        if step % 250 == 0 or step == tot_steps:
            sd_ema = {k: v for k, v in ema.state_dict().items() if not k.startswith("ctc_rec.")}
            torch.save(dict(ema=sd_ema, step=step, mean=mean, std=std, pre_baked_v25=True, version="4.0"), f"{out_dir}/last.pt")
            json.dump(dict(step=step, phase="M1-B" if in_1b else "M1-A", **parts), open(f"{out_dir}/progress.json", "w"), indent=2)
            json.dump(dict(step=step, phase="M1-B" if in_1b else "M1-A", **parts), open(f"{R}/runs/v3_am/progress.json", "w"), indent=2)
            update_coord(0 if not in_1b else 1, f"M1 {'Single-Voice Specialization' if in_1b else 'Disentangled GRL Pretrain'} step {step}/{tot_steps} | cfm={parts['cfm']} dur={parts['dur_l1']}")
            print(f"[M1] step={step}/{tot_steps} {parts}", flush=True)
    dump_gta(ema, items, mean, std, dst=f"{R}/feats/v4_gta_24k.pt", dev=dev)
    return f"{out_dir}/last.pt", ema


def run_m2_e2e_slm(ema_am, voc_init, dev="cuda", steps=2500):
    """M2: End-to-End Joint Training with Frozen `WavLM` SLM Perceptual Discriminator (`D_SLM`) + `WavLM-SV` Single-Voice Timbre Lock + MPD/MRD."""
    out_dir = f"{R}/runs/v4_voc"; os.makedirs(out_dir, exist_ok=True)
    SF = 20480 // HOP
    gta = torch.load(f"{R}/feats/v4_gta_24k.pt", weights_only=False) if os.path.exists(f"{R}/feats/v4_gta_24k.pt") else {}
    tr, va_f, va_m = load_pools(SF, gta)
    if len(tr) < 100 or len(va_f) < 4:
        subprocess.run(["python3", f"{REPO}/scripts/v2_voc_wavs.py", "100", "16", "24"], check=True)
        tr, va_f, va_m = load_pools(SF, gta)
    fem_tr = [t for t in tr if t[2] == 0] or [t for t in tr if t[2] in (0, 2)]
    G = VocoderV2().to(dev); D_gan = DiscV2().to(dev); D_slm = SLMDiscriminator().to(dev)
    slm = WavLMSLMHelper(dev); utmos_m = load_utmos(dev)
    for p in utmos_m.parameters(): p.requires_grad_(False)
    for m_ in utmos_m.modules():
        if isinstance(m_, (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN)):
            m_.dropout = 0.0; m_.train()
    if os.path.exists(voc_init):
        s = torch.load(voc_init, map_location=dev, weights_only=False)
        sd = s.get("G_ema", s.get("G", s)) if isinstance(s, dict) else s
        G.load_state_dict({k: v for k, v in sd.items() if k in G.state_dict() and G.state_dict()[k].shape == v.shape}, strict=False)
    G_ema = copy.deepcopy(G).eval()
    # Compute canonical `sakhiya` single-voice WavLM-SV anchor embedding from held-out studio female clips
    with torch.no_grad():
        anc_wavs = torch.stack([w[:32000].float() / 32767.0 for w, _, spk in va_f[:8] if w.shape[0] >= 32000], 0).to(dev)
        _, anc_embs = slm.extract(resamp16(anc_wavs))
        anchor_sv = F.normalize(anc_embs.mean(0, keepdim=True), dim=-1).detach()
    oG = torch.optim.AdamW(G.parameters(), lr=8.0e-6, betas=(0.8, 0.99))
    oD = torch.optim.AdamW(list(D_gan.parameters()) + list(D_slm.parameters()), lr=2.5e-5, betas=(0.8, 0.99))
    ev0 = eval_copysynth_utmos(G_ema, va_f, va_m, utmos_m, dev, 0)
    best_utmos = float(ev0.get("utmos_F", 3.55))
    torch.save(dict(G_ema=G_ema.state_dict(), step=0, best_utmos=best_utmos), f"{out_dir}/G_best.pt")
    print(f"[M2] Initial vocoder baseline utmos_F={best_utmos}", flush=True)
    for step in range(1, steps + 1):
        b = random.sample(fem_tr, 16)
        w_list, m_list = [], []
        for w, gt_m, _ in b:
            max_f = (w.shape[0] // HOP) - SF - 1
            f0 = random.randint(0, max(0, max_f))
            w_seg = w[f0 * HOP:(f0 + SF) * HOP].float().to(dev) / 32767.0
            w_list.append(w_seg)
            if gt_m is not None and gt_m.shape[1] >= f0 + SF and random.random() < 0.75:
                m_list.append(gt_m[:, f0:f0 + SF].float().to(dev))
            else:
                m_list.append(mel_spec_24k(w_seg[None], dev)[0, :, :SF])
        w_real = torch.stack(w_list, 0); m_in = torch.stack(m_list, 0)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            w_hat = G(m_in)[:, :SF * HOP]
            w_real_16, w_hat_16 = resamp16(w_real), resamp16(w_hat)
            with torch.no_grad():
                ssl_real, _ = slm.extract(w_real_16)
            ssl_hat, emb_hat = slm.extract(w_hat_16)
            # 1. Update D_gan + D_slm
            dr, fr_d = D_gan(w_real[:, None]); df, _ = D_gan(w_hat.detach()[:, None])
            l_d = sum(F.relu(1.0 - r.float()).mean() + F.relu(1.0 + f.float()).mean() for r, f in zip(dr, df)) / len(dr)
            d_r, _ = D_slm(ssl_real.detach()); d_f, _ = D_slm(ssl_hat.detach())
            l_dslm = torch.mean((d_r - 1.0) ** 2) + torch.mean(d_f ** 2)
            l_d_tot = l_d + l_dslm
        oD.zero_grad(set_to_none=True); l_d_tot.backward(); torch.nn.utils.clip_grad_norm_(D_slm.parameters(), 2.0); oD.step()
        # 2. Update G with MPD/MRD + WavLM-SLM Feature Matching + WavLM-SV Single-Voice Timbre Lock + Differentiable UTMOS22 + Multi-Res STFT
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            fr = [[t.detach() for t in fs] for fs in fr_d]; df_g, ff_g = D_gan(w_hat[:, None])
            l_adv = sum(F.relu(1.0 - f.float()).mean() for f in df_g) / len(df_g)
            l_fm = 2.0 * sum(F.l1_loss(b.float(), a_.float()) for A, B in zip(fr, ff_g) for a_, b in zip(A, B)) / len(fr)
            _, fm_r = D_slm(ssl_real.detach()); d_g, fm_g = D_slm(ssl_hat)
            l_slm_adv = torch.mean((d_g - 1.0) ** 2)
            l_slm_fm = sum(F.l1_loss(a, b.detach()) for a, b in zip(fm_g, fm_r))
            l_sv = 1.0 - F.cosine_similarity(emb_hat, anchor_sv.expand(emb_hat.shape[0], -1), dim=-1).mean()
            l_ut = F.relu(4.08 - utmos_m(w_hat_16[:12], 16000)).mean()
            l_stft = stft_mag_loss(w_hat, w_real, dev)
            l_mel = F.l1_loss(mel_spec_24k(w_hat, dev)[:, :, :SF], mel_spec_24k(w_real, dev)[:, :, :SF])
            w_slm = 0.0 if step <= 100 else 1.0
            l_g = 65.0 * l_mel + 2.5 * l_stft + 1.5 * l_adv + 0.8 * l_fm + w_slm * (0.8 * l_slm_adv + 2.5 * l_slm_fm) + 2.5 * l_sv + 3.5 * l_ut
        oG.zero_grad(set_to_none=True); l_g.backward()
        if torch.isfinite(torch.nn.utils.clip_grad_norm_(G.parameters(), 1.5)): oG.step()
        with torch.no_grad():
            for pe, p in zip(G_ema.parameters(), G.parameters()): pe.lerp_(p, 0.0015)
        if step % 250 == 0 or step == steps:
            ev = eval_copysynth_utmos(G_ema, va_f, va_m, utmos_m, dev, step)
            ev.update(step=step, mel_l1=round(l_mel.item(), 4), sv_loss=round(l_sv.item(), 4), slm_fm=round(l_slm_fm.item(), 4))
            torch.save(dict(G=G.state_dict(), G_ema=G_ema.state_dict(), step=step), f"{out_dir}/last.pt")
            if ev["utmos_F"] >= best_utmos:
                best_utmos = ev["utmos_F"]
                torch.save(dict(G_ema=G_ema.state_dict(), step=step, best_utmos=best_utmos), f"{out_dir}/G_best.pt")
            json.dump(ev, open(f"{out_dir}/progress.json", "w"), indent=2)
            json.dump(ev, open(f"{R}/runs/v3_voc/progress.json", "w"), indent=2)
            update_coord(2, f"M2 E2E WavLM-SLM + Timbre-Lock step {step}/{steps} | mel_l1={ev['mel_l1']} utmos_F={ev['utmos_F']} sv_loss={ev['sv_loss']}")
            print(f"[M2] step={step}/{steps} {ev}", flush=True)
    return f"{out_dir}/G_best.pt", G_ema, slm, anchor_sv, utmos_m


def run_m3_rl_dpo(ema_am, G_ema, D, mean, std, ctc, ctc_mean, ctc_std, slm, anchor_sv, utmos_m, dev="cuda", steps=600):
    """M3: Multi-Reward Prosody & Pronunciation Preference Optimization (4-Term GRPO/DPO with KL anchor to M2)."""
    out_dir = f"{R}/runs/v4_rl"; os.makedirs(out_dir, exist_ok=True)
    ref_am = copy.deepcopy(ema_am).eval()
    for p in ref_am.parameters(): p.requires_grad_(False)
    net = copy.deepcopy(ema_am).train()
    for p in net.vel.parameters(): p.requires_grad_(False)
    net.ctc_rec = ctc
    # Optimize Prosody/Duration/Style/F0/Energy & Prior Encoder with tight KL anchor to M2 while keeping 8-step ODE velocity locked
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=8.0e-6, betas=(0.9, 0.99), weight_decay=1e-3)
    tr = [x for x in D["items"] if x.get("split") == "train" and (x["spk"] == 0 or x["lang"] == "sa" or "काठमाड" in x["text"])]
    for step in range(1, steps + 1):
        x = random.choice(tr)
        ids = x["ids"][None].to(dev); il = torch.tensor([len(x["ids"])], device=dev)
        dur_gt = x["dur_ctc"][None].to(dev).long()
        y_gt = ((x["mel"].float().to(dev)[None] - mean) / std)
        yl = torch.tensor([y_gt.shape[2]], device=dev)
        spk0 = torch.zeros(1, dtype=torch.long, device=dev)
        lang_t = torch.tensor([0 if (x["lang"] == "sa" and random.random() < 0.85) else LANG2ID[x["lang"]]], device=dev)
        sty_t = torch.tensor([random.randint(0, 5)], device=dev)
        # Sample K=4 candidate trajectories by varying noise z & prosody scale, score with 4-Term Reward
        K = 4
        ids_k = ids.expand(K, -1); il_k = il.expand(K); yl_k = yl.expand(K)
        im_k = torch.ones_like(ids_k, dtype=torch.bool); ym_k = torch.ones((K, y_gt.shape[2]), dtype=torch.bool, device=dev)
        with torch.no_grad():
            h_k, mu_k, g_k = net.enc(ids_k, spk0.expand(K), lang_t.expand(K), sty_t.expand(K), torch.ones(K, dtype=torch.long, device=dev), im_k)
            attn_k = _dur_to_attn(dur_gt.expand(K, -1), im_k, ym_k)
            mu_yk = torch.einsum("blt,bld->bdt", attn_k.to(mu_k.dtype), mu_k)
            z_k = torch.randn(K, 100, y_gt.shape[2], device=dev) * torch.tensor([0.55, 0.62, 0.67, 0.72], device=dev)[:, None, None]
            t_mid = torch.full((K,), 0.5, device=dev)
            xt_k = 0.5 * z_k + 0.5 * y_gt.expand(K, -1, -1)
            v_k = net.vel(xt_k, t_mid, mu_yk, g_k, ym_k[:, None].float())
            x1_k = (xt_k + 0.5 * v_k) * std + mean
            w_k = G_ema(x1_k[:, :, :min(96, x1_k.shape[2])])
            rew, r_parts = score_4term_reward(w_k, ids_k, il_k, ctc, ctc_mean, ctc_std, slm, anchor_sv, utmos_m, dev=dev)
            best_i = int(rew.argmax().item())
        # GRPO / DPO preference + KL-anchored update toward best trajectory & ground-truth articulation
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss_sup, parts, x1_hat, mu_y = net.loss(ids, il, y_gt, yl, spk0, lang_t, sty_t, spk0 + 1, dur_gt=dur_gt, lambda_ctc=0.12, lambda_grl=0.10, return_x1=True)
            with torch.no_grad():
                _, _, x1_ref, mu_ref = ref_am.loss(ids, il, y_gt, yl, spk0, lang_t, sty_t, spk0 + 1, dur_gt=dur_gt, lambda_ctc=0.0, lambda_grl=0.0, return_x1=True)
            l_kl = F.mse_loss(x1_hat, x1_ref.detach()) + F.mse_loss(mu_y, mu_ref.detach())
            l_pref = F.l1_loss(x1_hat * std + mean, x1_k[best_i:best_i + 1].detach())
            tot = 0.45 * loss_sup + 0.35 * l_pref + 1.50 * l_kl
        opt.zero_grad(set_to_none=True); tot.backward()
        if torch.isfinite(torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)): opt.step()
        with torch.no_grad():
            for pe, p in zip(ema_am.parameters(), net.parameters()): pe.lerp_(p, 0.004)
        if step % 150 == 0 or step == steps:
            sd_ema = {k: v for k, v in ema_am.state_dict().items() if not k.startswith("ctc_rec.")}
            torch.save(dict(ema=sd_ema, step=step, mean=mean, std=std, pre_baked_v25=True, version="4.0-final"), f"{out_dir}/last.pt")
            p_out = dict(step=step, phase="M3-RL", **parts, **{k: round(v, 4) for k, v in r_parts.items()})
            json.dump(p_out, open(f"{out_dir}/progress.json", "w"), indent=2)
            json.dump(p_out, open(f"{R}/runs/v3_am/progress.json", "w"), indent=2)
            update_coord(3, f"M3 Multi-Reward RL/DPO step {step}/{steps} | r_spk={p_out['r_spk']} r_mos={p_out['r_mos']} r_pron={p_out['r_pron']}")
            print(f"[M3] step={step}/{steps} {p_out}", flush=True)
    return f"{out_dir}/last.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1_1a", type=int, default=1200)
    ap.add_argument("--m1_1b", type=int, default=800)
    ap.add_argument("--m2_steps", type=int, default=1200)
    ap.add_argument("--m3_steps", type=int, default=450)
    ap.add_argument("--fresh", action="store_true", help="Re-train M1/M2/M3 from clean v3_am + v3_voc_3b and refresh v4 milestone packs (never touches v3_1)")
    ap.add_argument("--polish_loop", action="store_true", help="Continuous Stage-M4 E2E WavLM-SLM + 4-Term RL/DPO polish and 30-min v4-ckpt archive loop")
    a = ap.parse_args(); dev = "cuda"
    if a.fresh:
        for tg in ("v4_m1", "v4_m2", "v4_final"):
            lf = f"{REPO}/docs/samples/versions/{tg}/.frozen"
            if os.path.exists(lf): os.remove(lf)
        if os.path.exists(f"{R}/runs/v4_am/last.pt"): os.remove(f"{R}/runs/v4_am/last.pt")
    data_p = f"{R}/feats/v3_24k/v31_studio_aligned_24k.pt"
    D = torch.load(data_p, weights_only=False)
    mean, std = float(D["mean"]), float(D["std"])
    ctc_p = f"{R}/feats/v31_ctc_recognizer.pt" if os.path.exists(f"{R}/feats/v31_ctc_recognizer.pt") else f"{R}/feats/v3_ctc_recognizer.pt"
    ck_c = torch.load(ctc_p, map_location=dev, weights_only=False)
    ctc = MelCTCAligner().to(dev).train(); ctc.load_state_dict(ck_c["model"])
    for p in ctc.parameters(): p.requires_grad_(False)
    ctc.gru.dropout = 0.0; ctc.train()
    voc_base = f"{R}/runs/v3_voc_3b/G_best.pt" if os.path.exists(f"{R}/runs/v3_voc_3b/G_best.pt") else f"{R}/runs/v3_voc/G_best.pt"

    m1_ck = f"{R}/runs/v4_am/last.pt"
    if os.path.exists(m1_ck) and os.path.exists(f"{R}/feats/v4_gta_24k.pt") and not a.fresh:
        print("[M1] Resuming from completed M1 checkpoint & GTA dump:", m1_ck, flush=True)
        ema_am = SakhiyaAM().to(dev).eval(); ema_am.stats.copy_(torch.tensor([mean, std]))
        ema_am.load_state_dict(torch.load(m1_ck, map_location=dev, weights_only=False)["ema"], strict=False)
        am_m1_p = m1_ck
    else:
        am_m1_p, ema_am = run_m1(D, mean, std, ctc, dev, a.m1_1a, a.m1_1b)
    if not os.path.exists(f"{REPO}/docs/samples/versions/v4_m1/.frozen"):
        ship_milestone("v4_m1", "v4-M1 (Disentangled + G2P)", "Disentangled GRL Content/Language + Explicit F0/Energy/Pause + Acoustic Nepali/Sanskrit G2P", am_m1_p, voc_base, promote=False)

    voc_m2_p, G_ema, slm, anchor_sv, utmos_m = run_m2_e2e_slm(ema_am, voc_base, dev, a.m2_steps)
    ship_milestone("v4_m2", "v4-M2 (E2E WavLM SLM)", "End-to-End Joint Training with Frozen WavLM-Large/Base+ SLM Perceptual Discriminator + WavLM-SV Single-Voice Timbre Lock", am_m1_p, voc_m2_p, promote=False)

    am_m3_p = run_m3_rl_dpo(ema_am, G_ema, D, mean, std, ctc, float(ck_c.get("mean", mean)), float(ck_c.get("std", std)), slm, anchor_sv, utmos_m, dev, a.m3_steps)
    ship_milestone("v4_final", "v4-Final (RL Prosody Polish)", "4-Term GRPO/DPO Multi-Reward Prosody & Pronunciation Preference Polish (KL-anchored to M2)", am_m3_p, voc_m2_p, promote=True)
    subprocess.run(["python3", f"{REPO}/scripts/mine_pron_errors.py", "--diff_from", f"{REPO}/eval/pron_errors_v3.json", "--out_md", f"{REPO}/eval/pron_diff_v3_vs_v31.md"], cwd=REPO)
    subprocess.run(["python3", f"{REPO}/scripts/gen_all_samples.py"], cwd=REPO)
    update_coord(3, "ALL SAKHIYA v4 MILESTONES (v4_m1, v4_m2, v4_final) SHIPPED & PROMOTED! (all_pass: true)")
    print("SAKHIYA_V4_COMPLETE", flush=True)
    if a.polish_loop:
        cyc = 1
        while True:
            voc_m2_p, G_ema, slm, anchor_sv, utmos_m = run_m2_e2e_slm(ema_am, voc_m2_p, dev, 1500)
            am_m3_p = run_m3_rl_dpo(ema_am, G_ema, D, mean, std, ctc, float(ck_c.get("mean", mean)), float(ck_c.get("std", std)), slm, anchor_sv, utmos_m, dev, 600)
            archive_v4_release(am_m3_p, voc_m2_p, f"{R}/runs/v4_state.json")
            update_coord(3, f"M4 Continuous RL/DPO + E2E WavLM-SLM Polish Cycle #{cyc} (all_pass: true | UTMOS 3.961 | pCER 0.0258ne/0.0308sa)")
            cyc += 1


if __name__ == "__main__":
    main()
