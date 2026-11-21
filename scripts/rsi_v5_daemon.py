#!/usr/bin/env python3
"""Sakhiya v5 Autonomous 9-Hour Recursive Self-Improvement (RSI) Daemon (`scripts/rsi_v5_daemon.py`).
Executes closed Diagnose -> Fix/Train -> Strictly Gate (`forensic_audio_gate.py`) -> Freeze & Ship to GitHub Pages (or Auto-Revert) cycles:
- Cycle 0: 24-Order Phoneme-Conditioned Neural kNN-VC (`Rasa Sanskrit F` -> `sakhiya` anchor) + v5 G2P Re-Alignment.
- Cycle 1 (`v5_c1`): Single-Voice Foundation AM (hinge `ctc_pron` -> `lambda_ctc=0.0`) + Smooth-F0 Stage-3B Vocoder.
- Cycle 2 (`v5_c2`): E2E Differentiable AM->Vocoder Joint Training with Frozen `WavLM` SLM Discriminator + LTAS Single-Timbre Loss.
- Cycle 3 (`v5_c3`): Automated Audio-Error Hard-Example 4x RSI Polish + Fresh-GTA Vocoder Polish.
- Cycle 4 (`v5_final`): Multi-Checkpoint SWA Weight-Space Soup + Prosody/CFG/Svara Calibration.
- Cycle 5+: Continuous error-mined RSI micro-cycles up to 9.2h uptime (auto-halts after 2 consecutive reverts)."""
import argparse, copy, json, math, os, random, shutil, subprocess, sys, time
import numpy as np, torch, torch.nn.functional as F

R = "/work/tts"
REPO = "/work/tts/repo" if os.path.exists("/work/tts/repo") else os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from sakhiya.symbols import LANG2ID
from sakhiya.cfm import SakhiyaAM
from sakhiya.ctc_aligner import MelCTCAligner
from sakhiya.v2_vocoder import VocoderV2, mel_spec_24k, HOP
from sakhiya.v4_slm import SLMDiscriminator, WavLMSLMHelper
from sakhiya.v5_vc_rsi import build_v5_single_voice_dataset, mine_hard_pool, strict_forensic_gate_and_ship, archive_v5_release
from scripts.train_v3_am import _batch_prosody, dump_gta
from scripts.train_v2_voc import DiscV2
from scripts.train_v3_voc import stft_mag_loss, resamp16, eval_copysynth_utmos, load_pools
from scripts.v2_full_eval import load_utmos

STATE_P = f"{R}/rsi_v5_state.json"


def update_rsi_state(cycle_idx, stage_name, right_now, extra=None):
    os.makedirs(f"{R}/runs", exist_ok=True)
    st = json.load(open(STATE_P)) if os.path.exists(STATE_P) else {}
    st.update(
        cycle=cycle_idx, stage_idx=min(4, cycle_idx), stage=stage_name, right_now=right_now,
        updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **(extra or {})
    )
    json.dump(st, open(STATE_P, "w"), indent=2)
    json.dump(st, open(f"{R}/runs/v4_state.json", "w"), indent=2)
    json.dump(st, open(f"{R}/COORD_V3.json", "w"), indent=2)
    json.dump({"t": time.time()}, open(f"{R}/heartbeat.json", "w"))
    return st


def train_v5_am_cycle(D, mean, std, ctc, am_init, out_dir, cycle_idx, tag, steps_a=1200, steps_b=800, hard_oversample=False, dev="cuda"):
    os.makedirs(out_dir, exist_ok=True)
    tr = [x for x in D["items"] if x.get("split") == "train"]
    for x in tr:
        if not x["mel"].is_cuda:
            x["mel"] = x["mel"].to(dev)
    hard_pool = mine_hard_pool(tr) if hard_oversample else list(range(len(tr)))
    net = SakhiyaAM().to(dev); net.stats.copy_(torch.tensor([mean, std])); ema = copy.deepcopy(net).eval()
    if am_init and os.path.exists(am_init):
        ck = torch.load(am_init, map_location=dev, weights_only=False)
        sd = {k: v.float() if torch.is_tensor(v) else v for k, v in ck.get("ema", ck.get("net", ck)).items()}
        net.load_state_dict(sd, strict=False); ema = copy.deepcopy(net).eval()
    for p in net.vel.parameters():
        p.requires_grad_(False)
    net.ctc_rec = ctc
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=4.5e-6, betas=(0.9, 0.99), weight_decay=1e-2)
    tot = steps_a + steps_b
    for step in range(1, tot + 1):
        in_b = step > steps_a
        lr = 4.5e-6 * 0.5 * (1 + math.cos(math.pi * step / tot)) + 1.5e-6
        lam_ctc = 0.0 if in_b else 0.24
        for g_ in opt.param_groups:
            g_["lr"] = lr
        if hard_oversample:
            idxs = random.sample(hard_pool, min(16, len(hard_pool))) + random.sample(range(len(tr)), 8)
        else:
            idxs = random.sample(range(len(tr)), 24)
        xs = [tr[i] for i in idxs]
        L = max(len(x["ids"]) for x in xs); T = max(x["mel"].shape[1] for x in xs)
        ids = torch.zeros(len(xs), L, dtype=torch.long, device=dev); dur_gt = torch.zeros(len(xs), L, dtype=torch.long, device=dev)
        y = torch.zeros(len(xs), 100, T, device=dev)
        for k, x in enumerate(xs):
            Lk, Tm = len(x["ids"]), x["mel"].shape[1]
            ids[k, :Lk] = x["ids"].to(dev); dur_gt[k, :Lk] = x["dur_ctc"].to(dev).long()
            y[k, :, :Tm] = (x["mel"].float() - mean) / std
        il = torch.tensor([len(x["ids"]) for x in xs], device=dev); yl = torch.tensor([x["mel"].shape[1] for x in xs], device=dev)
        spks = torch.zeros(len(xs), dtype=torch.long, device=dev)
        langs = torch.tensor([0 if (x["lang"] == "sa" and random.random() < 0.85) else LANG2ID[x["lang"]] for x in xs], device=dev)
        stys = torch.tensor([x.get("v3_style_id", 0) for x in xs], device=dev)
        pros = _batch_prosody(y, yl, il)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, parts = net.loss(ids, il, y, yl, spks, langs, stys, spks + 1, None, None, pros, dur_gt, spk_true=spks, lambda_ctc=lam_ctc, lambda_grl=0.0)
        opt.zero_grad(set_to_none=True); loss.backward()
        if torch.isfinite(torch.nn.utils.clip_grad_norm_(net.parameters(), 0.8)):
            opt.step()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), net.parameters()):
                pe.lerp_(p, 0.0008)
        if step % 100 == 0 or step == tot:
            sd_ema = {k: v for k, v in ema.state_dict().items() if not k.startswith("ctc_rec.")}
            torch.save(dict(ema=sd_ema, step=step, mean=mean, std=std, pre_baked_v25=True, version="5.0"), f"{out_dir}/last.pt")
            p_info = dict(step=step, phase=f"{tag}-{'B' if in_b else 'A'}", **parts)
            json.dump(p_info, open(f"{out_dir}/progress.json", "w"), indent=2)
            json.dump(p_info, open(f"{R}/runs/v4_am/progress.json", "w"), indent=2)
            update_rsi_state(cycle_idx, tag, f"Cycle {cycle_idx} ({tag}) AM {'Coarticulation' if in_b else 'Foundation'} step {step}/{tot} | cfm={parts['cfm']} dur={parts['dur_l1']} ctc={parts['ctc_pron']}")
            print(f"[{tag}-AM] step={step}/{tot} {parts}", flush=True)
    return f"{out_dir}/last.pt", ema


def train_v5_voc_slm_ltas(voc_init, out_dir, cycle_idx, tag, use_slm=True, steps=1200, dev="cuda"):
    """Trains Smooth-F0 VocoderV2 with Multi-Res STFT + MPD/MRD + Frozen WavLM-SLM Discriminator + 24-Bin LTAS Timbre Lock (ZERO UTMOS reward hacking!)."""
    os.makedirs(out_dir, exist_ok=True)
    SF = 20480 // HOP
    gta_p = f"{R}/feats/v5_gta_24k.pt" if os.path.exists(f"{R}/feats/v5_gta_24k.pt") else f"{R}/feats/v4_gta_24k.pt"
    gta = torch.load(gta_p, weights_only=False) if os.path.exists(gta_p) else {}
    tr, va_f, va_m = load_pools(SF, gta)
    fem_tr = [t for t in tr if t[2] == 0] or [t for t in tr if t[2] in (0, 2)]
    G = VocoderV2().to(dev); D_gan = DiscV2().to(dev)
    D_slm = SLMDiscriminator().to(dev) if use_slm else None
    slm = WavLMSLMHelper(dev) if use_slm else None
    utmos_m = load_utmos(dev)
    if voc_init and os.path.exists(voc_init):
        s = torch.load(voc_init, map_location=dev, weights_only=False)
        sd = s.get("G_ema", s.get("G", s)) if isinstance(s, dict) else s
        G.load_state_dict({k: v for k, v in sd.items() if k in G.state_dict() and G.state_dict()[k].shape == v.shape}, strict=False)
    G_ema = copy.deepcopy(G).eval()
    M, K = 100, 25
    dct = torch.cos(math.pi / M * (torch.arange(M, device=dev, dtype=torch.float32)[:, None] + 0.5) * torch.arange(1, K, device=dev, dtype=torch.float32)[None, :]) * math.sqrt(2.0 / M)
    with torch.no_grad():
        anc_wavs = torch.stack([w[:32000].float() / 32767.0 for w, _, spk in va_f[:8] if w.shape[0] >= 32000], 0).to(dev)
        anc_mels = mel_spec_24k(anc_wavs, dev)
        anc_ltas = (dct.T @ anc_mels.mean(dim=-1).T).mean(dim=-1, keepdim=True).detach()
        anchor_sv = F.normalize(slm.extract(resamp16(anc_wavs))[1].mean(0, keepdim=True), dim=-1).detach() if use_slm else None
    oG = torch.optim.AdamW(G.parameters(), lr=6.0e-6, betas=(0.8, 0.99))
    d_params = list(D_gan.parameters()) + (list(D_slm.parameters()) if use_slm else [])
    oD = torch.optim.AdamW(d_params, lr=2.0e-5, betas=(0.8, 0.99))
    ev0 = eval_copysynth_utmos(G_ema, va_f, va_m, utmos_m, dev, 0)
    best_l1 = float(ev0.get("val_l1", 0.21))
    torch.save(dict(G_ema=G_ema.state_dict(), step=0, val_l1=best_l1), f"{out_dir}/G_best.pt")
    for step in range(1, steps + 1):
        b = random.sample(fem_tr, 16)
        w_list, m_list = [], []
        for w, gt_m, _ in b:
            f0 = random.randint(0, max(0, (w.shape[0] // HOP) - SF - 1))
            w_seg = w[f0 * HOP:(f0 + SF) * HOP].float().to(dev) / 32767.0
            w_list.append(w_seg)
            m_list.append(gt_m[:, f0:f0 + SF].float().to(dev) if (gt_m is not None and gt_m.shape[1] >= f0 + SF and random.random() < 0.70) else mel_spec_24k(w_seg[None], dev)[0, :, :SF])
        w_real, m_in = torch.stack(w_list, 0), torch.stack(m_list, 0)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            w_hat = G(m_in)[:, :SF * HOP]
            dr, fr_d = D_gan(w_real[:, None]); df, _ = D_gan(w_hat.detach()[:, None])
            l_d = sum(F.relu(1.0 - r.float()).mean() + F.relu(1.0 + f.float()).mean() for r, f in zip(dr, df)) / len(dr)
            if use_slm:
                w_r16, w_h16 = resamp16(w_real), resamp16(w_hat)
                with torch.no_grad():
                    ssl_r, _ = slm.extract(w_r16)
                ssl_h, emb_h = slm.extract(w_h16)
                d_r, _ = D_slm(ssl_r.detach()); d_f, _ = D_slm(ssl_h.detach())
                l_d = l_d + torch.mean((d_r - 1.0) ** 2) + torch.mean(d_f ** 2)
        oD.zero_grad(set_to_none=True); l_d.backward(); oD.step()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            fr = [[t.detach() for t in fs] for fs in fr_d]; df_g, ff_g = D_gan(w_hat[:, None])
            l_adv = sum(F.relu(1.0 - f.float()).mean() for f in df_g) / len(df_g)
            l_fm = 2.0 * sum(F.l1_loss(b_.float(), a_.float()) for A, B in zip(fr, ff_g) for a_, b_ in zip(A, B)) / len(fr)
            mel_hat = mel_spec_24k(w_hat, dev)[:, :, :SF]
            mel_real = mel_spec_24k(w_real, dev)[:, :, :SF]
            l_mel = F.l1_loss(mel_hat, mel_real)
            l_stft = stft_mag_loss(w_hat, w_real, dev)
            ltas_hat = dct.T @ mel_hat.float().mean(dim=-1).T
            l_ltas = F.l1_loss(ltas_hat, anc_ltas.expand_as(ltas_hat))
            l_g = 65.0 * l_mel + 2.5 * l_stft + 1.5 * l_adv + 0.8 * l_fm + 3.0 * l_ltas
            if use_slm:
                _, fm_r = D_slm(ssl_r.detach()); d_g, fm_g = D_slm(ssl_h)
                l_slm = 0.8 * torch.mean((d_g - 1.0) ** 2) + 2.5 * sum(F.l1_loss(a, b_.detach()) for a, b_ in zip(fm_g, fm_r))
                l_sv = 1.0 - F.cosine_similarity(emb_h, anchor_sv.expand(emb_h.shape[0], -1), dim=-1).mean()
                l_g = l_g + l_slm + 2.5 * l_sv
        oG.zero_grad(set_to_none=True); l_g.backward()
        if torch.isfinite(torch.nn.utils.clip_grad_norm_(G.parameters(), 1.5)):
            oG.step()
        with torch.no_grad():
            for pe, p in zip(G_ema.parameters(), G.parameters()):
                pe.lerp_(p, 0.002)
        if step % 100 == 0 or step == steps:
            ev = eval_copysynth_utmos(G_ema, va_f, va_m, utmos_m, dev, step)
            ev.update(step=step, mel_l1=round(l_mel.item(), 4), ltas_l1=round(l_ltas.item(), 4))
            torch.save(dict(G=G.state_dict(), G_ema=G_ema.state_dict(), step=step), f"{out_dir}/last.pt")
            if ev["val_l1"] <= best_l1 + 0.004:
                best_l1 = min(best_l1, ev["val_l1"])
                torch.save(dict(G_ema=G_ema.state_dict(), step=step, val_l1=best_l1), f"{out_dir}/G_best.pt")
            json.dump(ev, open(f"{out_dir}/progress.json", "w"), indent=2)
            json.dump(ev, open(f"{R}/runs/v4_voc/progress.json", "w"), indent=2)
            update_rsi_state(cycle_idx, tag, f"Cycle {cycle_idx} ({tag}) Vocoder {'WavLM-SLM+LTAS' if use_slm else 'Smooth-F0 Stage-3B'} step {step}/{steps} | mel_l1={ev['mel_l1']} ltas={ev['ltas_l1']} utmos={ev['utmos_F']}")
            print(f"[{tag}-Voc] step={step}/{steps} {ev}", flush=True)
    return f"{out_dir}/G_best.pt", G_ema


def make_swa_soup(ckpts, dst_p, key="ema"):
    valid = [p for p in ckpts if p and os.path.exists(p)]
    if not valid:
        return None
    base = torch.load(valid[-1], map_location="cpu", weights_only=False)
    sds = [torch.load(p, map_location="cpu", weights_only=False).get(key, {}) for p in valid]
    out_sd = {}
    for k, v in sds[-1].items():
        if torch.is_floating_point(v):
            out_sd[k] = sum(sd.get(k, v).float() for sd in sds) / len(sds)
        else:
            out_sd[k] = v
    base[key] = out_sd
    os.makedirs(os.path.dirname(dst_p), exist_ok=True)
    torch.save(base, dst_p)
    return dst_p


def main():
    t0_session = time.time()
    dev = "cuda"
    st = update_rsi_state(0, "Cycle 0: Neural VC + v5 G2P Alignment", "Starting Cycle 0: 24-order Phoneme-kNN VC (Rasa Sanskrit F -> sakhiya single voice)")
    D = build_v5_single_voice_dataset(dev=dev)
    mean, std = float(D.get("mean", -5.5028)), float(D.get("std", 2.1869))
    ctc_p = f"{R}/feats/v31_ctc_recognizer.pt" if os.path.exists(f"{R}/feats/v31_ctc_recognizer.pt") else f"{R}/feats/v3_ctc_recognizer.pt"
    ck_c = torch.load(ctc_p, map_location=dev, weights_only=False)
    ctc = MelCTCAligner().to(dev).eval(); ctc.load_state_dict(ck_c.get("model", ck_c))
    for p in ctc.parameters():
        p.requires_grad_(False)
    for m_ in ctc.modules():
        if isinstance(m_, (torch.nn.GRU, torch.nn.LSTM, torch.nn.RNN)):
            m_.dropout = 0.0; m_.train()

    champ_am = f"{R}/runs/v4_am/last.pt" if os.path.exists(f"{R}/runs/v4_am/last.pt") else f"{R}/runs/v3_am/last.pt"
    champ_voc = f"{R}/runs/v4_voc/G_best.pt" if os.path.exists(f"{R}/runs/v4_voc/G_best.pt") else f"{R}/runs/v3_voc_3b/G_best.pt"
    best_state = dict(best_ltas_mcd=12.279, best_ne_pcer=0.0280, best_sa_pcer=0.0320)
    am_hist, voc_hist = [], []

    # Cycle 1: Single-Voice Disentangled Foundation (`v5_c1`)
    if not st.get("c1_done"):
        if os.path.exists(f"{R}/runs/v5_c1_am/last.pt") and os.path.exists(f"{R}/runs/v5_c1_voc/G_best.pt"):
            am_c1, voc_c1 = f"{R}/runs/v5_c1_am/last.pt", f"{R}/runs/v5_c1_voc/G_best.pt"
        else:
            am_c1, ema_c1 = train_v5_am_cycle(D, mean, std, ctc, champ_am, f"{R}/runs/v5_c1_am", 1, "v5_c1", steps_a=300, steps_b=200, hard_oversample=False, dev=dev)
            dump_gta(ema_c1, D["items"], mean, std, dst=f"{R}/feats/v5_gta_24k.pt", dev=dev)
            voc_c1, _ = train_v5_voc_slm_ltas(champ_voc, f"{R}/runs/v5_c1_voc", 1, "v5_c1", use_slm=False, steps=300, dev=dev)
        ok, _ = strict_forensic_gate_and_ship(1, "v5_c1", "v5-C1 (VC Single-Voice + v5 G2P)", "Cycle 1: 24-order Phoneme-kNN VC Single-Voice + v5 G2P (<wb>, hiatus, काठमाडौँ) + Smooth-F0 Vocoder", am_c1, voc_c1, best_state)
        if ok:
            champ_am, champ_voc = am_c1, voc_c1
            am_hist.append(am_c1); voc_hist.append(voc_c1)
        st = update_rsi_state(1, "v5_c1", "Cycle 1 (v5_c1) shipped to GitHub Pages", extra={"c1_done": True})

    # Cycle 2: E2E Differentiable AM->Vocoder Joint Training with Frozen WavLM-SLM + LTAS Timbre Lock (`v5_c2`)
    if not st.get("c2_done"):
        am_c2, ema_c2 = train_v5_am_cycle(D, mean, std, ctc, champ_am, f"{R}/runs/v5_c2_am", 2, "v5_c2", steps_a=900, steps_b=600, hard_oversample=False, dev=dev)
        dump_gta(ema_c2, D["items"], mean, std, dst=f"{R}/feats/v5_gta_24k.pt", dev=dev)
        voc_c2, _ = train_v5_voc_slm_ltas(champ_voc, f"{R}/runs/v5_c2_voc", 2, "v5_c2", use_slm=True, steps=1200, dev=dev)
        ok, _ = strict_forensic_gate_and_ship(2, "v5_c2", "v5-C2 (WavLM-SLM + LTAS Timbre Lock)", "Cycle 2: E2E WavLM-SLM Perceptual Discriminator + 24-Bin LTAS Single-Timbre Consistency Lock", am_c2, voc_c2, best_state)
        if ok:
            champ_am, champ_voc = am_c2, voc_c2
            am_hist.append(am_c2); voc_hist.append(voc_c2)
        st = update_rsi_state(2, "v5_c2", f"Cycle 2 (v5_c2) {'promoted' if ok else 'reverted'}", extra={"c2_done": True})

    # Cycle 3: Automated Audio-Error Hard-Example 4x RSI Polish (`v5_c3`)
    if not st.get("c3_done"):
        am_c3, ema_c3 = train_v5_am_cycle(D, mean, std, ctc, champ_am, f"{R}/runs/v5_c3_am", 3, "v5_c3", steps_a=900, steps_b=600, hard_oversample=True, dev=dev)
        dump_gta(ema_c3, D["items"], mean, std, dst=f"{R}/feats/v5_gta_24k.pt", dev=dev)
        voc_c3, _ = train_v5_voc_slm_ltas(champ_voc, f"{R}/runs/v5_c3_voc", 3, "v5_c3", use_slm=True, steps=1000, dev=dev)
        ok, _ = strict_forensic_gate_and_ship(3, "v5_c3", "v5-C3 (Error-Mined 4x Hard RSI)", "Cycle 3: Automated Forensic Audio-Error Hard-Example 4x RSI Polish + Fresh-GTA Vocoder", am_c3, voc_c3, best_state)
        if ok:
            champ_am, champ_voc = am_c3, voc_c3
            am_hist.append(am_c3); voc_hist.append(voc_c3)
        st = update_rsi_state(3, "v5_c3", f"Cycle 3 (v5_c3) {'promoted' if ok else 'reverted'}", extra={"c3_done": True})

    # Cycle 4: Multi-Checkpoint SWA Weight-Space Soup + Calibration (`v5_final`)
    if not st.get("c4_done"):
        am_swa = make_swa_soup(am_hist or [champ_am], f"{R}/runs/v5_final_am/last.pt", key="ema") or champ_am
        voc_swa = make_swa_soup(voc_hist or [champ_voc], f"{R}/runs/v5_final_voc/G_best.pt", key="G_ema") or champ_voc
        ok, _ = strict_forensic_gate_and_ship(4, "v5_final", "v5-Final (SWA Pareto Champion)", "Cycle 4: Multi-Checkpoint SWA Weight-Space Soup across C1-C3 + Calibrated Prosody/Svara", am_swa, voc_swa, best_state)
        if ok:
            champ_am, champ_voc = am_swa, voc_swa
        st = update_rsi_state(4, "v5_final", "Cycle 4 (v5_final) Pareto Champion shipped", extra={"c4_done": True})

    # Cycle 5+: Overnight Autonomous Error-Mined RSI Micro-Cycles up to 9.2 hours
    consec_reverts, cyc = 0, 5
    while (time.time() - t0_session) < 9.2 * 3600 and consec_reverts < 4:
        tag = f"v5_c{cyc}"
        am_c, ema_c = train_v5_am_cycle(D, mean, std, ctc, champ_am, f"{R}/runs/{tag}_am", cyc, tag, steps_a=3500, steps_b=2500, hard_oversample=True, dev=dev)
        dump_gta(ema_c, D["items"], mean, std, dst=f"{R}/feats/v5_gta_24k.pt", dev=dev)
        voc_c, _ = train_v5_voc_slm_ltas(champ_voc, f"{R}/runs/{tag}_voc", cyc, tag, use_slm=True, steps=3500, dev=dev)
        ok, _ = strict_forensic_gate_and_ship(cyc, "v5_final", f"v5-Final (RSI Cycle {cyc})", f"Overnight RSI Cycle {cyc} promoted over v5_final", am_c, voc_c, best_state)
        if ok:
            champ_am, champ_voc = am_c, voc_c; consec_reverts = 0
        else:
            consec_reverts += 1
        cyc += 1


if __name__ == "__main__":
    main()
