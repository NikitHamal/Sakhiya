#!/usr/bin/env python3
"""v8 student-vocoder fine-tune on the mels the STUDENT acoustic model really produces (GTA-style), with F0 distillation.

Root-cause being fixed (measured, see eval/v8_*.json): the v7 tiny vocoders were trained only on *sharp ground-truth
mels*, but at inference they receive the *smoothed* mel of the tiny acoustic student. The vocoder's F0 estimator (frozen,
copied from the big teacher) then mis-reads pitch from the blurred mel -> pitch jitter / buzz = "robotic, glitchy".

What this script does
  * caches the student AM's own mel for every training clip (teacher-forced CTC durations) and refreshes the cache from the
    newest AM checkpoint every --recache-min minutes (so it can train beside the AM job),
  * trains the vocoder on a 70/30 mix of student mels / ground-truth mels against the teacher-vocoder waveform of the
    ground-truth mel (the vocoder learns to de-blur),
  * UNFREEZES the F0 estimator and distills teacher F0/voicing (computed from the sharp mel) into it from the blurred mel,
  * keeps MPD+MRD GAN, feature matching and the WavLM SLM discriminator,
  * picks the best checkpoint by UTMOS measured on the REAL inference path (text -> student AM -> student vocoder),
  * enforces --max-minutes (<= 150) and prints a step-50 throughput gate.
"""
import argparse, copy, json, math, os, random, sys, time
import torch, torch.nn.functional as F

R = "/work/tts"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from sakhiya import g2p, sanskrit as SA  # noqa: E402
from sakhiya.cfm import SakhiyaAM, STYLE_PROSODY_TARGETS  # noqa: E402
from sakhiya.symbols import SYM2ID, LANG2ID  # noqa: E402
from sakhiya.v2_vocoder import VocoderV2, mel_spec_24k, ola64  # noqa: E402
from sakhiya.v4_slm import SLMDiscriminator, WavLMSLMHelper  # noqa: E402
from sakhiya.v6_distill import _expand_batch_durs  # noqa: E402
from scripts.export_v2_onnx import TIER_CFG  # noqa: E402
from scripts.freeze_version_pack import MASTER_15_CLIPS  # noqa: E402
from scripts.train_nano_atom import _bake  # noqa: E402
from scripts.train_v2_voc import DiscV2, mr_complex_stft_loss, phase_loss  # noqa: E402
from scripts.train_v3_voc import resamp16  # noqa: E402
from scripts.v2_full_eval import load_utmos  # noqa: E402

EXTRA = [
    ("ne", "storytelling", "वाह! के अद्भुत दृश्य हो यो! माछापुच्छ्रे हिमाल बिहानको घाममा सुनजस्तै चम्किरहेको छ!"),
    ("ne", "news", "सरकारले आज नयाँ बजेट सार्वजनिक गर्दै शिक्षा र स्वास्थ्य क्षेत्रमा थप लगानी गर्ने घोषणा गरेको छ।"),
    ("ne", "storytelling", "ओहो! पहाडको टुप्पोबाट झरेको झरना कस्तो सुन्दर देखिएको छ, हगि?"),
    ("sa", "vedic_chant", "ॐ असतो मा सद्गमय। तमसो मा ज्योतिर्गमय। मृत्योर्मा अमृतं गमय॥"),
    ("sa", "vedic_chant", "यदा यदा हि धर्मस्य ग्लानिर्भवति भारत। अभ्युत्थानमधर्मस्य तदात्मानं सृजाम्यहम्॥"),
]
ST_MAP = {"news": 0, "conversational": 2, "storytelling": 2, "warm_soft": 2, "expressive": 2, "vedic_chant": 5}


def load_am(tier, path, dev, mean, std):
    c = TIER_CFG[tier]
    m = SakhiyaAM(**{k: c[k] for k in ("d_enc", "L_enc", "ff_enc", "d_vel", "nb_vel", "h_dur")}).to(dev)
    _bake(m)
    ck = torch.load(path, map_location=dev, weights_only=False)
    m.load_state_dict({k: v.float() for k, v in ck.get("ema", ck.get("net", ck)).items()}, strict=False)
    m.stats.copy_(torch.tensor([mean, std], device=dev))
    return m.eval(), int(ck.get("step", 0))


def export_mel(mel):
    """Exactly the interface the shipped ONNX decoder+vocoder pair uses (scripts/export_v2_onnx.py DecExport/VocSpecExport):
    spectral-envelope comb sharpening on the low bins, and bins 98/99 are side channels that the vocoder replaces by bin 97."""
    env = F.avg_pool1d(F.pad(mel.transpose(1, 2), (3, 3), mode="replicate"), 7, stride=1).transpose(1, 2)
    kb = torch.arange(100, device=mel.device, dtype=mel.dtype)
    wk = ((64.0 - kb) / 18.0).clamp(0.0, 1.0)[None, :, None]
    mw = env + (mel - env) * (1.0 + 0.12 * wk)
    return torch.cat([mw[:, :98], mw[:, 97:98], mw[:, 97:98]], dim=1)


@torch.no_grad()
def val_mels(s_am, dev, mean, std):
    """Inference path: text -> g2p -> student encoder + duration predictor -> 1-step decoder -> raw log-mel (export_mel applied by caller)."""
    pros_tbl = torch.tensor(STYLE_PROSODY_TARGETS, dtype=torch.float32, device=dev)
    sents = [(lg, st, tx) for _, lg, st, tx in MASTER_15_CLIPS if lg != "auto"] + EXTRA
    out = []
    for lg, st_name, txt in sents:
        ph, off, dm = SA.laya_scales(txt, v2=True) if lg == "sa" else (*g2p.text_to_phones_acc(txt, "ne"), g2p.phone_dm(g2p.text_to_phones(txt, "ne")))
        vi = [i for i, p in enumerate(ph) if p in SYM2ID]
        ids = torch.tensor([[SYM2ID[ph[i]] for i in vi]], dtype=torch.long, device=dev)
        po = torch.tensor([[float(off[i]) * 0.48 for i in vi]], dtype=torch.float32, device=dev)
        dmt = torch.tensor([[float(dm[i]) for i in vi]], dtype=torch.float32, device=dev)
        sty = torch.tensor([ST_MAP.get(st_name, 0)], device=dev); z = torch.zeros(1, dtype=torch.long, device=dev)
        im = torch.ones_like(ids, dtype=torch.bool)
        h, mu, g = s_am.enc(ids, z, z, sty, torch.ones(1, dtype=torch.long, device=dev), im, pitch_add=po, prosody=pros_tbl[sty], sty_mask=torch.ones((1, 1), device=dev))
        dur = torch.round(torch.exp(s_am.dur.pred_logdur(ids, h, im)).clamp(1.0, 60.0) * dmt / (0.92 if lg == "sa" else 1.0)).clamp(min=1).long()
        T = int(dur.sum()); ymf = torch.ones(1, 1, T, device=dev)
        mu_y = _expand_batch_durs(mu, dur, T)
        y = mu_y + s_am.vel(mu_y, torch.zeros((1,), device=dev), mu_y, g, ymf)
        out.append((y.float() * std + mean)[0])
    return out


@torch.no_grad()
def cache_student(s_am, items, mean, std, dev):
    pros_tbl = torch.tensor(STYLE_PROSODY_TARGETS, dtype=torch.float32, device=dev)
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    res = [None] * len(items)
    for b0 in range(0, len(order), 96):
        idx = order[b0:b0 + 96]; xs = [items[i] for i in idx]
        B = len(xs); L = max(len(x["ids"]) for x in xs); T = min(max(int(x["dur_ctc"].sum()) for x in xs), 850)
        ids = torch.zeros(B, L, dtype=torch.long, device=dev); durs = torch.zeros(B, L, dtype=torch.long, device=dev); po = torch.zeros(B, L, device=dev)
        for k, x in enumerate(xs):
            Lk = len(x["ids"]); ids[k, :Lk] = x["ids"].to(dev); durs[k, :Lk] = x["dur_ctc"].to(dev).long()
            if x.get("po") is not None: po[k, :Lk] = x["po"][:Lk].to(dev)
        il = torch.tensor([len(x["ids"]) for x in xs], device=dev); yl = durs.sum(1).clamp(max=T)
        im = torch.arange(L, device=dev)[None] < il[:, None]; ymf = (torch.arange(T, device=dev)[None] < yl[:, None])[:, None].float()
        langs = torch.tensor([0 if (x["lang"] == "sa" and random.random() < 0.85) else LANG2ID.get(x["lang"], 0) for x in xs], device=dev)
        stys = torch.tensor([int(x.get("v3_style_id", 0)) for x in xs], device=dev).clamp(0, 5)
        z = torch.zeros(B, dtype=torch.long, device=dev)
        h, mu, g = s_am.enc(ids, z, langs, stys, torch.ones(B, dtype=torch.long, device=dev), im, pitch_add=po, prosody=pros_tbl[stys], sty_mask=torch.ones((B, 1), device=dev))
        mu_y = _expand_batch_durs(mu, durs, T) * ymf
        y = (mu_y + s_am.vel(mu_y, torch.zeros((B,), device=dev), mu_y, g, ymf)) * ymf
        y = (y.float() * std + mean)
        for k, i in enumerate(idx):
            n = int(min(yl[k].item(), items[i]["mel"].shape[1]))
            res[i] = y[k, :, :n].half()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, choices=["voca", "sano"])
    ap.add_argument("--am", required=True, help="student AM checkpoint, re-read every --recache-min")
    ap.add_argument("--init_voc", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-minutes", type=float, default=100.0)
    ap.add_argument("--recache-min", type=float, default=15.0)
    ap.add_argument("--teacher_voc", default=f"{R}/runs/v5_c1/voc.pt")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seg", type=int, default=72)
    ap.add_argument("--p_student", type=float, default=0.7)
    ap.add_argument("--f0_w", type=float, default=6.0)
    a = ap.parse_args()
    assert a.max_minutes <= 150
    dev = "cuda"; os.makedirs(a.out, exist_ok=True); c = TIER_CFG[a.tier]
    D = torch.load(f"{R}/feats/v6_expressive_vedic_24k.pt", weights_only=False)
    mean, std = float(D.get("mean", -5.5028)), float(D.get("std", 2.1869))
    items = [x for x in D["items"] if x.get("split") == "train" and len(x["ids"]) >= 3 and 24 <= x["mel"].shape[1] <= 850]
    gt = [x["mel"].half().to(dev) for x in items]
    print(f"[v8-voc-{a.tier}] items={len(items)} frames={sum(m.shape[1] for m in gt)}", flush=True)

    t_voc = VocoderV2().to(dev).eval()
    vck = torch.load(a.teacher_voc, map_location=dev, weights_only=False)
    t_voc.load_state_dict({k: v.float() for k, v in vck.get("G_ema", vck.get("G", vck)).items()}, strict=False)
    for p in t_voc.parameters(): p.requires_grad_(False)

    s_voc = VocoderV2(ch=c["v_ch"], res_kernels=c["v_rk"]).to(dev)
    init = a.init_voc or (f"{a.out}/voc.pt" if os.path.exists(f"{a.out}/voc.pt") else f"{R}/runs/v6_{a.tier}_voclong/voc.pt")
    ik = torch.load(init, map_location=dev, weights_only=False)
    s_voc.load_state_dict(ik.get("G_ema", ik.get("G", ik)), strict=False); step0 = int(ik.get("step", 0)) if init.startswith(a.out) else 0
    print(f"[v8-voc-{a.tier}] init from {init}", flush=True)
    for p in s_voc.parameters(): p.requires_grad_(True)   # F0 estimator is UNFROZEN (the fix)
    ema = copy.deepcopy(s_voc).eval()

    D_gan, D_slm, slm = DiscV2().to(dev), SLMDiscriminator().to(dev), WavLMSLMHelper(dev)
    utmos = load_utmos(dev)
    base_lr = 2.0e-4
    opt = torch.optim.AdamW(s_voc.parameters(), lr=base_lr, betas=(0.8, 0.99), weight_decay=1e-4)
    oD = torch.optim.AdamW(list(D_gan.parameters()) + list(D_slm.parameters()), lr=base_lr * 0.7, betas=(0.8, 0.99))

    def eval_utmos(voc, vm, teacher=False):
        sc = []
        with torch.no_grad():
            for m in vm:
                w = voc(m[None] if teacher else export_mel(m[None]))
                sc.append(float(utmos(resamp16(w.float()), torch.full((1,), 16000, device=dev)).mean()))
        return sum(sc) / len(sc)

    s_am, am_step = load_am(a.tier, a.am, dev, mean, std)
    vm = val_mels(s_am, dev, mean, std)
    st_cache = cache_student(s_am, items, mean, std, dev)
    diag = dict(am_step=am_step, utmos_studentmel_teachervoc=eval_utmos(t_voc, vm, True), utmos_studentmel_studentvoc_before=eval_utmos(ema, vm))
    print(f"[v8-voc-{a.tier}] DIAG {json.dumps(diag)}", flush=True)
    best = diag["utmos_studentmel_studentvoc_before"]; t_last_cache = time.time()
    t_start, budget, step = time.time(), a.max_minutes * 60.0, step0
    hist = []
    while (time.time() - t_start) < budget:
        frac = (time.time() - t_start) / budget
        lr = base_lr * (0.5 * (1 + math.cos(math.pi * frac)) * 0.9 + 0.1)
        for g_ in opt.param_groups: g_["lr"] = lr
        step += 1
        if step == step0 + 50:
            sps = 50.0 / (time.time() - t_start); print(f"[GATE] step-50 throughput {sps:.2f} steps/s -> {sps * budget:.0f} steps inside the {a.max_minutes:.0f} min cap", flush=True)
            if sps < 1.0: print("[GATE] too slow, aborting", flush=True); sys.exit(3)
        if time.time() - t_last_cache > a.recache_min * 60 and os.path.exists(a.am):
            try:
                s_am, am_step = load_am(a.tier, a.am, dev, mean, std); st_cache = cache_student(s_am, items, mean, std, dev); vm = val_mels(s_am, dev, mean, std)
                print(f"[v8-voc-{a.tier}] re-cached student mels from AM step {am_step}", flush=True)
            except Exception as e:
                print("recache failed", repr(e), flush=True)
            t_last_cache = time.time()
        ii = random.sample(range(len(items)), a.bs * 2); mg, mi, ok = [], [], 0
        for i in ii:
            n = min(gt[i].shape[1], st_cache[i].shape[1])
            if n < a.seg: continue
            s = random.randint(0, n - a.seg); g_m = gt[i][:, s:s + a.seg].float(); s_m = st_cache[i][:, s:s + a.seg].float()
            mg.append(g_m); mi.append(s_m if random.random() < a.p_student else g_m + 0.0)
            ok += 1
            if ok == a.bs: break
        mel_gt, mel_in = torch.stack(mg), export_mel(torch.stack(mi))
        with torch.no_grad():
            wav_t = t_voc(mel_gt)[:, :a.seg * 256]
            mel_t = mel_spec_24k(wav_t, dev)[:, :, :a.seg]
            _, _, f0_t, uv_t = t_voc.spectrum_and_f0(mel_gt)
        use_gan = (step - step0) > 300
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            re, im_, f0_s, uv_s = s_voc.spectrum_and_f0(mel_in)
            wav_s = ola64(torch.complex(re.float(), im_.float()), s_voc.win)[:, :a.seg * 256]
            if use_gan:
                dr, fr_d = D_gan(wav_t[:, None]); df, _ = D_gan(wav_s.detach()[:, None])
                l_d = sum(F.relu(1.0 - r.float()).mean() + F.relu(1.0 + f.float()).mean() for r, f in zip(dr, df)) / len(dr)
                if step % 2 == 0:
                    with torch.no_grad(): ssl_r, _ = slm.extract(resamp16(wav_t))
                    ssl_h, _ = slm.extract(resamp16(wav_s))
                    d_r, _ = D_slm(ssl_r.detach()); d_f, _ = D_slm(ssl_h.detach())
                    l_d = l_d + torch.mean((d_r - 1.0) ** 2) + torch.mean(d_f ** 2)
        if use_gan:
            oD.zero_grad(set_to_none=True); l_d.backward(); oD.step()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            mel_s = mel_spec_24k(wav_s, dev)[:, :, :a.seg]
            l_mel = F.l1_loss(mel_s, mel_t)
            l_stft = mr_complex_stft_loss(wav_s, wav_t, dev) + 0.8 * phase_loss(wav_s, wav_t, 512, dev)
            w_v = uv_t.detach().float()
            l_f0 = ((torch.log2(f0_s.float().clamp(min=60)) - torch.log2(f0_t.float().clamp(min=60))).abs() * w_v).sum() / w_v.sum().clamp(min=1.0)
            l_uv = F.l1_loss(uv_s.float(), uv_t.detach().float())
            loss = 45.0 * l_mel + 2.5 * l_stft + a.f0_w * (12.0 * l_f0 + 4.0 * l_uv)
            if use_gan and (step - step0) > 340:
                fr = [[t.detach() for t in fs] for fs in fr_d]; df_g, ff_g = D_gan(wav_s[:, None])
                l_adv = sum(F.relu(1.0 - f.float()).mean() for f in df_g) / len(df_g)
                l_fm = 2.0 * sum(F.l1_loss(b_.float(), a_.float()) for A, B in zip(fr, ff_g) for a_, b_ in zip(A, B)) / len(fr)
                loss = loss + 1.5 * l_adv + 1.2 * l_fm
                if step % 2 == 0:
                    _, fm_r = D_slm(ssl_r.detach()); d_g, fm_g = D_slm(ssl_h)
                    loss = loss + 0.8 * torch.mean((d_g - 1.0) ** 2) + 2.5 * sum(F.l1_loss(x_, y_.detach()) for x_, y_ in zip(fm_g, fm_r))
        opt.zero_grad(set_to_none=True); loss.backward()
        if torch.isfinite(torch.nn.utils.clip_grad_norm_(s_voc.parameters(), 1.0)): opt.step()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), s_voc.parameters()): pe.lerp_(p, 0.015)
        if (step - step0) % 400 == 0:
            cur = eval_utmos(ema, vm); hist.append(cur)
            if cur >= best:
                best = cur; tmp = f"{a.out}/voc.pt.tmp"
                torch.save(dict(G_ema=ema.state_dict(), G=s_voc.state_dict(), step=step, tier=a.tier, cfg=c, utmos=cur, am_step=am_step), tmp); os.replace(tmp, f"{a.out}/voc.pt")
            print(f"[v8-voc-{a.tier}] step={step} mel={l_mel.item():.3f} f0={l_f0.item():.4f} uv={l_uv.item():.3f} utmos(real path)={cur:.3f} best={best:.3f} t={time.time() - t_start:.0f}s", flush=True)
            if len(hist) >= 6 and max(hist[-4:]) <= max(hist[:-4]) - 0.0 and (time.time() - t_start) > 0.5 * budget and max(hist[-4:]) < best - 0.02:
                print("[v8-voc] validation flat for 4 checks, stopping early", flush=True); break
    json.dump(dict(diag=diag, best_utmos=best, steps=step - step0, am_step=am_step), open(f"{a.out}/summary.json", "w"))
    print(f"[v8-voc-{a.tier}] DONE best={best:.3f}", flush=True)


if __name__ == "__main__":
    main()
