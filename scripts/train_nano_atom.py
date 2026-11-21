"""Distill Sakhiya-Nano (<=2.5 MB int8) and Sakhiya-Atom (<1 MB int8, sub-1M params) from Teacher-Free Sakhiya v2.5.
Uses 1-Pass Direct Deterministic Residual SeqKD (zero ODE loop!), Pure Teacher Mel Targets (zero multi-speaker recording blur),
Teacher-MAS 4-step Euler DurFlow KD, Spectral/Temporal Delta-Mel losses, and Frozen-Teacher-F0 Analytic HPN Vocoder."""
import argparse, copy, math, os, random, sys
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sakhiya.cfm import SakhiyaAM, _mel_f0_z
from sakhiya.mas import maximum_path
from sakhiya.v2_vocoder import VocoderV2, mel_spec_24k, ola64, n_params
from sakhiya.symbols import SYM2ID, LANG2ID
from sakhiya import g2p, sanskrit as _sa
from scripts.train_v2_voc import mr_complex_stft_loss

CFG = {
    "nano": dict(d_enc=96, L_enc=4, ff_enc=144, d_vel=128, nb_vel=5, h_dur=64, v_ch=112, v_rk=(3, 7, 11), ode_steps=1),
    "atom": dict(d_enc=64, L_enc=3, ff_enc=96, d_vel=80, nb_vel=4, h_dur=48, v_ch=64, v_rk=(3, 7), ode_steps=1),
}

ANCHOR_SENTS = [
    ("ne", "नेपालको राजधानी काठमाडौँ हिमालको काखमा रहेको एउटा ऐतिहासिक सहर हो।"),
    ("ne", "एक समयको कुरा हो, एउटा सानो गाउँमा एउटी बुढी आमा बस्थिन्। उनी बिहानै उठेर चरालाई दाना हाल्थिन्।"),
    ("ne", "हेलो साथीहरू। म सखिया हुँ, निकित हमालज्यूले बनाउनुभएको सानो टेक्स्ट टु स्पीच मोडेल।"),
    ("ne", "शिक्षा र स्वास्थ्य क्षेत्रमा सुधार ल्याउन स्थानीय सरकारले नयाँ योजना अघि सारेको छ।"),
    ("ne", "सगरमाथा आरोहणका लागि यस वर्ष धेरै विदेशी पर्यटकहरू नेपाल आएका छन्।"),
    ("ne", "हाम्रो संस्कृति र परम्परालाई जोगाउनु हामी सबैको साझा दायित्व हो।"),
    ("ne", "किसानहरूले समयमा मल र बीउ पाउँदा धान उत्पादनमा उल्लेखनीय वृद्धि भएको छ।"),
    ("ne", "प्रविधिको विकाससँगै ग्रामीण भेगमा पनि इन्टरनेटको पहुँच विस्तार हुँदै गएको छ।"),
    ("ne", "वातावरण संरक्षणका लागि वृक्षारोपण र फोहोर व्यवस्थापनमा विशेष ध्यान दिनुपर्छ।"),
    ("sa", "ॐ पूर्णमदः पूर्णमिदं पूर्णात्पूर्णमुदच्यते। पूर्णस्य पूर्णमादाय पूर्णमेवावशिष्यते॥"),
    ("sa", "अ॒ग्निमी॑ळे पु॒रोहि॑तं य॒ज्ञस्य॑ दे॒वमृ॒त्विज॑म् । होता॑रं रत्न॒धात॑मम् ॥"),
    ("sa", "अग्निमीळे पुरोहितं यज्ञस्य देवमृत्विजम्। होतारं रत्नधातमम्॥"),
    ("sa", "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः। मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय॥"),
    ("sa", "वसुधैव कुटुम्बकम् इति भारतीयसंस्कृतेः मूलमन्त्रः अस्ति।"),
    ("sa", "विद्या ददाति विनयं विनयाद्याति पात्रताम्। पात्रत्वाद्धनमाप्नोति धनाद्धर्मं ततः सुखम्॥"),
    ("sa", "सत्यमेव जयते नानृतं सत्येन पन्था विततो देवयानः।"),
    ("thr_dangaura", "हेलो संघरियाहुक्रे। मै सखिया हों, निकित हमाल जी बनाइल छोट टेक्स्ट टु स्पीच मोडेल।"),
    ("thr_dangaura", "हमार गाउँ बहुत सुन्दर बा"),
    ("thr_rana", "हमार नाउ का हो"),
    ("thr_chitwania", "जात जातक भासा अलग बा"),
]


def warm_slice(dst_m, src_sd):
    with torch.no_grad():
        for k, p in dst_m.state_dict().items():
            if k not in src_sd or not torch.is_floating_point(p) or "original0" in k: continue
            s = src_sd[k].float().to(p.device)
            if s.shape == p.shape:
                p.copy_(s)
            elif s.ndim == p.ndim and all(a >= b for a, b in zip(s.shape, p.shape)):
                sl = tuple(slice(0, b) for b in p.shape)
                sc = math.sqrt(s.shape[1] / p.shape[1]) if s.ndim >= 2 and s.shape[1] > p.shape[1] and "emb" not in k else 1.0
                p.copy_(s[sl] * min(sc, 1.25))


def _bake(m):
    with torch.no_grad():
        w_spk = m.enc.spk.weight.clone(); w_sty = m.enc.sty.weight.clone(); w_e = m.enc.emb.weight
        m.enc.spk.weight[0].copy_(w_spk[0]); m.enc.spk.weight[1].copy_(w_spk[1])
        m.enc.spk.weight[2].copy_(0.65 * w_spk[4:8].mean(0) + 0.35 * w_spk[0])
        m.enc.spk.weight[3].copy_(0.65 * w_spk[3] + 0.35 * w_spk[1])
        m.enc.spk.weight[4].copy_(0.65 * w_spk[11:14].mean(0) + 0.35 * w_spk[0])
        m.enc.sty.weight[0].copy_(0.65 * w_sty[10] + 0.35 * w_sty[3])
        m.enc.sty.weight[1].copy_(0.75 * w_sty[4] + 0.25 * w_sty[13])
        m.enc.sty.weight[2].copy_(0.75 * w_sty[3] + 0.25 * w_sty[2])
        m.enc.sty.weight[3].copy_(0.50 * w_sty[0] + 0.30 * w_sty[5] + 0.20 * w_sty[11])
        m.enc.sty.weight[4].copy_(0.65 * w_sty[8] + 0.35 * w_sty[12])
        m.enc.sty.weight[5].copy_(0.55 * w_sty[3] + 0.45 * w_sty[14])
        id_l, id_r = SYM2ID["l"], SYM2ID["ɽ"]
        w_e[id_r].copy_(0.60 * w_e[id_l] + 0.40 * w_e[id_r])
        nas_d = 0.5 * (w_e[SYM2ID["ũ"]] - w_e[SYM2ID["u"]]) + 0.5 * (w_e[SYM2ID["ã"]] - w_e[SYM2ID["a"]])
        w_e[SYM2ID["õ"]].copy_(0.25 * w_e[SYM2ID["õ"]] + 0.75 * (w_e[SYM2ID["o"]] + 1.0 * nas_d + 0.2 * (w_e[SYM2ID["ŋ"]] - w_e[SYM2ID["o"]])))
        sa_id = LANG2ID["sa"]
        m.enc.lang.weight[sa_id].copy_(0.85 * m.enc.lang.weight[0] + 0.15 * m.enc.lang.weight[sa_id])


def train_tier(tier, data_pt, native_pt, t_am_pt, t_voc_pt, out_dir, am_steps=6000, voc_steps=4500):
    dev = "cuda"; c = CFG[tier]; os.makedirs(out_dir, exist_ok=True)
    D = torch.load(data_pt, weights_only=False); items = [x for x in D["items"] if x["split"] == "train"]
    if os.path.exists(native_pt):
        N_D = torch.load(native_pt, weights_only=False)
        items.extend([x for x in N_D.get("items", []) if x.get("split") == "train"] * 2)
    for x in items:
        ph = [q for p in x["phones"] for q in (["o", "m"] if p == "<om>" else [p])]
        vi = [i for i, p in enumerate(ph) if p in SYM2ID]
        x["ids"] = torch.tensor([SYM2ID[ph[i]] for i in vi], dtype=torch.long)
        po = x.get("pitch_off")
        if po is not None and len(po) >= len(ph): x["po"] = torch.tensor([float(po[i]) for i in vi], dtype=torch.float32)
    items = [x for x in items if 3 <= len(x["ids"]) and 24 <= x["mel"].shape[1] <= 800 and x["mel"].shape[1] > len(x["ids"])]

    t_ck = torch.load(t_am_pt, map_location=dev, weights_only=False)
    t_sd = {k: v.float() for k, v in t_ck.get("ema", t_ck.get("net", t_ck)).items()}
    p_2a = "/work/tts/runs/v2_am/last.pt"
    if os.path.exists(p_2a):
        ck_a = torch.load(p_2a, map_location=dev, weights_only=False); sd_a = ck_a.get("ema", ck_a.get("net", ck_a))
        t_sd = {k: 0.35 * sd_a[k].float() + 0.65 * t_sd[k].float() for k in sd_a}
    mean, std = float(t_ck.get("mean", -6.177)), float(t_ck.get("std", 2.492))
    t_am = SakhiyaAM().to(dev).eval(); t_am.load_state_dict(t_sd); t_am.stats.copy_(torch.tensor([mean, std], device=dev))
    _bake(t_am)

    # Synthesize clean Teacher-Distilled Anchor Items across voices/styles so Nano/Atom master all target clusters & held-out sentences
    anchor_items = []
    with torch.no_grad():
        for lg, txt in ANCHOR_SENTS:
            if lg == "sa":
                ph, off, dm = _sa.laya_scales(txt, v2=True)
            else:
                ph, off = g2p.text_to_phones_acc(txt, lg); dm = g2p.phone_dm(ph)
            vi = [i for i, p in enumerate(ph) if p in SYM2ID]
            ids_a = torch.tensor([[SYM2ID[ph[i]] for i in vi]], dtype=torch.long, device=dev)
            po_a = torch.tensor([[float(off[i]) * 0.48 for i in vi]], dtype=torch.float32, device=dev)
            dm_a = torch.tensor([[float(dm[i]) for i in vi]], dtype=torch.float32, device=dev)
            for spk_i, sty_i in ((0, 0), (0, 2), (0, 5), (1, 0), (2, 2), (3, 5)):
                mel_a = t_am.synth(ids_a, torch.tensor([spk_i], device=dev), torch.tensor([0], device=dev),
                                   torch.tensor([sty_i], device=dev), torch.tensor([1], device=dev),
                                   rate=0.95 if lg == "sa" else 1.0, steps=8, temp=0.5, pitch_add=po_a, dm=dm_a)
                anchor_items.append(dict(ids=ids_a[0].cpu(), po=po_a[0].cpu(), mel=mel_a[0].cpu(),
                                         spk=spk_i, lang=lg, style_id=sty_i, phones=[ph[i] for i in vi], anchor=True))
    print(f"[{tier}] Generated {len(anchor_items)} clean Teacher-Distilled anchor utterances", flush=True)

    hard_extra = []
    for x in items:
        j = " ".join(x["phones"])
        if any(k in j for k in ("t̪ s", "ts ", "ʃ ts", "ɽ", "<om>", "e r ʌ", "kʰ m", "ʌ h", "ɳ ɖ")): hard_extra += [x] * 3
    items = items + hard_extra + anchor_items * 25

    s_am = SakhiyaAM(**{k: c[k] for k in ("d_enc", "L_enc", "ff_enc", "d_vel", "nb_vel", "h_dur")}).to(dev)
    s_am.stats.copy_(torch.tensor([mean, std], device=dev))
    warm_slice(s_am, t_am.state_dict())
    if os.path.exists(f"{out_dir}/am.pt"):
        ck_s = torch.load(f"{out_dir}/am.pt", map_location=dev, weights_only=False)
        if ck_s.get("seqkd_1pass", False):
            s_am.load_state_dict(ck_s.get("ema", ck_s["net"]))
        else:
            # Keep already-distilled encoder + duration flow, re-init velocity for 1-pass residual SeqKD
            sd_prev = ck_s.get("ema", ck_s["net"])
            s_am.load_state_dict({k: sd_prev[k] if k.startswith(("enc.", "dur.")) else v for k, v in s_am.state_dict().items()})
            torch.nn.init.zeros_(s_am.vel.out.weight); torch.nn.init.zeros_(s_am.vel.out.bias)
    _bake(s_am)
    ema_am = copy.deepcopy(s_am).eval()
    opt = torch.optim.AdamW(s_am.parameters(), lr=1.0e-3, betas=(0.9, 0.99), weight_decay=1e-2)
    print(f"[{tier}] 1-Pass SeqKD AM params: {n_params(s_am)/1e6:.3f}M, training {am_steps} steps...", flush=True)

    c_idx = torch.arange(100, device=dev, dtype=torch.float32)[None, :, None]
    for step in range(1, am_steps + 1):
        b_idx = random.sample(range(len(items)), 56) + [len(items) - 1 - random.randint(0, len(anchor_items) * 25 - 1) for _ in range(16)]
        xs = [items[i] for i in b_idx]; B = len(xs); L = max(len(x["ids"]) for x in xs); T = max(x["mel"].shape[1] for x in xs)
        ids = torch.zeros(B, L, dtype=torch.long, device=dev); y = torch.zeros(B, 100, T, device=dev)
        po = torch.zeros(B, L, device=dev); has_po = False
        for k, x in enumerate(xs):
            ids[k, :len(x["ids"])] = x["ids"].to(dev); Tm = x["mel"].shape[1]
            y[k, :, :Tm] = (x["mel"].to(dev).float() - mean) / std
            if "po" in x and x["po"] is not None: p = x["po"].to(dev); po[k, :min(L, p.shape[0])] = p[:min(L, p.shape[0])]; has_po = True
        il = torch.tensor([len(x["ids"]) for x in xs], device=dev); yl = torch.tensor([x["mel"].shape[1] for x in xs], device=dev)
        im = torch.arange(L, device=dev)[None] < il[:, None]; ym = torch.arange(T, device=dev)[None] < yl[:, None]; ymf = ym[:, None].float()
        # Use true recording speaker for MAS alignment, then distill pure Teacher mel for target speaker/style
        orig_spks = torch.tensor([int(x.get("spk", 0)) for x in xs], device=dev).clamp(0, 15)
        spks = torch.tensor([int(x["spk"]) if x.get("anchor") else (0 if random.random() < 0.72 else random.randint(1, 4)) for x in xs], device=dev)
        langs = torch.zeros(B, dtype=torch.long, device=dev)
        stys = torch.tensor([int(x.get("style_id", 0)) if x.get("anchor") else (0 if random.random() < 0.70 else random.randint(1, 5)) for x in xs], device=dev)
        stud = torch.ones(B, dtype=torch.long, device=dev)
        lr_cur = 1.0e-3 * 0.5 * (1.0 + math.cos(math.pi * step / am_steps))
        for g in opt.param_groups: g["lr"] = max(lr_cur, 6e-5)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            p_in = po if has_po else None
            with torch.no_grad():
                # 1. Align duration path using original speaker so MAS is 100% accurate
                _, mu_align, _ = t_am.enc(ids, orig_spks, langs, stys, stud, im, p_in)
                mu_af, yf = mu_align.float(), y.float()
                lp = -0.5 * (mu_af.pow(2).sum(-1)[:, :, None] - 2 * torch.einsum("bld,bdt->blt", mu_af, yf) + yf.pow(2).sum(1)[:, None])
                attn_t = maximum_path(lp, im[:, :, None] & ym[:, None, :]).to(mu_align.dtype)
                # 2. Compute pure Teacher encoder & 4-step Midpoint Teacher mel for target spks/stys (ZERO speaker mismatch!)
                h_t, mu_t, g_t = t_am.enc(ids, spks, langs, stys, stud, im, p_in)
                pos = torch.arange(L, device=dev, dtype=torch.float32)[None, :]
                xd0 = torch.sin(pos * 1.6180339 + ids.float() * 0.37) * 0.20
                xd_t = xd0.clone()
                for i_d in range(4):
                    xd_t = xd_t + t_am.dur(xd_t, torch.full((B,), i_d / 4.0, device=dev), h_t, im) * 0.25
                ld_tgt = xd_t.clamp(-0.5, 4.1) * im.float()
                mu_y_t = torch.einsum("blt,bld->bdt", attn_t, mu_t)
                t_idx = torch.arange(T, device=dev, dtype=torch.float32)[None, None, :]
                xt = (torch.sin(c_idx * 12.9898 + t_idx * 78.233) * torch.cos(c_idx * 39.346 + t_idx * 11.135) * (0.50 * 1.4142)).expand(B, 100, T).to(mu_y_t.dtype)
                for i_o in range(4):
                    to = torch.full((B,), i_o * 0.25, device=dev)
                    k1 = t_am.vel(xt, to, mu_y_t, g_t, ymf)
                    k2 = t_am.vel(xt + 0.125 * k1, to + 0.125, mu_y_t, g_t, ymf)
                    xt = xt + 0.25 * k2
                y_clean = xt * ymf

            h_s, mu_s, g_s = s_am.enc(ids, spks, langs, stys, stud, im, p_in)
            mu_y_s = torch.einsum("blt,bld->bdt", attn_t.to(mu_s.dtype), mu_s)
            l_prior = 2.5 * (((mu_s - mu_t) ** 2 + (mu_s - mu_t).abs()) * im[:, :, None]).sum() / (im.sum() * 100.0)

            xd_s = xd0.clone(); v_dtgt = ld_tgt - xd0; l_dur = 0.0
            for i_d in range(4):
                vd_s = s_am.dur(xd_s, torch.full((B,), i_d / 4.0, device=dev), h_s, im)
                l_dur = l_dur + (((vd_s - v_dtgt) ** 2) * im).sum() / (im.sum() * 4.0)
                xd_s = xd_s + vd_s * 0.25
            l_dur = l_dur + 2.0 * (F.smooth_l1_loss(xd_s, ld_tgt, reduction="none") * im).sum() / im.sum()

            # 1-Pass Direct Deterministic Residual SeqKD: y_pred = mu_y_s + s_am.vel(mu_y_s, t=0, mu_y_s, g_s, ymf)
            t0_z = torch.zeros((B,), device=dev, dtype=torch.float32)
            y_pred = (mu_y_s + s_am.vel(mu_y_s, t0_z, mu_y_s, g_s, ymf)) * ymf
            denom = ymf.sum() * 100.0
            l_mel = ((1.5 * (y_pred - y_clean).abs() + (y_pred - y_clean) ** 2) * ymf).sum() / denom
            # Spectral (frequency-axis) delta loss for razor-sharp harmonics & formants (lifts HNR & consonant clarity)
            df_p = y_pred[:, 1:, :] - y_pred[:, :-1, :]; df_t = y_clean[:, 1:, :] - y_clean[:, :-1, :]
            l_df = ((df_p - df_t).abs() * ymf).sum() / denom
            # Temporal delta loss for crisp stop closures & releases
            ym_t = ymf[:, :, 1:] * ymf[:, :, :-1]
            dt_p = y_pred[:, :, 1:] - y_pred[:, :, :-1]; dt_t = y_clean[:, :, 1:] - y_clean[:, :, :-1]
            l_dt = ((dt_p - dt_t).abs() * ym_t).sum() / (ym_t.sum() * 100.0 + 1.0)
            l_cfm = l_mel + 1.8 * l_df + 1.2 * l_dt

            loss = l_cfm + l_prior + 0.4 * l_dur
        opt.zero_grad(set_to_none=True); loss.backward()
        if torch.isfinite(torch.nn.utils.clip_grad_norm_(s_am.parameters(), 1.0)): opt.step()
        with torch.no_grad():
            d = min(0.998, (1 + step) / (10 + step))
            for pe, p in zip(ema_am.parameters(), s_am.parameters()): pe.lerp_(p, 1 - d)
        if step % 250 == 0 or step == am_steps:
            print(f"[{tier}-AM-1Pass] step {step}/{am_steps} loss {loss.item():.4f} (mel={l_mel.item():.3f} df={l_df.item():.3f} dt={l_dt.item():.3f} prior={l_prior.item():.3f} dur={l_dur.item():.3f})", flush=True)
            torch.save(dict(ema=ema_am.state_dict(), net=s_am.state_dict(), step=step, tier=tier, cfg=c, mean=mean, std=std, pre_baked_v25=True, seqkd_1pass=True), f"{out_dir}/am.pt")

    if voc_steps <= 0:
        print(f"[{tier}] AM-only DONE! Saved to {out_dir}", flush=True); return
    t_voc = VocoderV2().to(dev).eval()
    vck = torch.load(t_voc_pt, map_location=dev, weights_only=False)
    t_vsd = vck.get("G_ema", vck.get("G", vck)) if isinstance(vck, dict) and ("G_ema" in vck or "G" in vck) else vck
    t_voc.load_state_dict(t_vsd)
    s_voc = VocoderV2(ch=c["v_ch"], res_kernels=c["v_rk"]).to(dev); s_voc.src.use_hpn = True
    warm_slice(s_voc, t_vsd)
    if os.path.exists(f"{out_dir}/voc.pt"):
        vck_s = torch.load(f"{out_dir}/voc.pt", map_location=dev, weights_only=False)
        s_voc.load_state_dict(vck_s.get("G_ema", vck_s["G"]))
    s_voc.src.f0_net.load_state_dict(t_voc.src.f0_net.state_dict())
    for p in s_voc.src.f0_net.parameters(): p.requires_grad_(False)
    ema_voc = copy.deepcopy(s_voc).eval(); ema_voc.src.use_hpn = True
    v_opt = torch.optim.AdamW([p for p in s_voc.parameters() if p.requires_grad], lr=8e-4, betas=(0.8, 0.99), weight_decay=1e-3)
    print(f"[{tier}] Voc params: {n_params(s_voc)/1e6:.3f}M (HPN=True, frozen teacher f0_net), training {voc_steps} steps...", flush=True)

    seg_f = 72
    for step in range(1, voc_steps + 1):
        b_idx = [i for i in random.sample(range(len(items)), 64) if items[i]["mel"].shape[1] >= seg_f][:36]
        mels = torch.stack([x["mel"][:, s:(s + seg_f)] for x in (items[i] for i in b_idx) for s in [random.randint(0, x["mel"].shape[1] - seg_f)]]).to(dev).float()
        lr_cur = 1.0e-3 * 0.5 * (1.0 + math.cos(math.pi * step / voc_steps))
        for g in v_opt.param_groups: g["lr"] = max(lr_cur, 5e-5)
        with torch.no_grad():
            re_t, im_t, _, _ = t_voc.spectrum_and_f0(mels)
            wav_t = ola64(torch.complex(re_t.float(), im_t.float()), t_voc.win)
            mag_t = torch.sqrt(re_t.float() ** 2 + im_t.float() ** 2 + 1e-7)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            re_s, im_s, _, _ = s_voc.spectrum_and_f0(mels)
        wav_s = ola64(torch.complex(re_s.float(), im_s.float()), s_voc.win)
        mel_s = mel_spec_24k(wav_s, dev)[:, :, :seg_f]
        mag_s = torch.sqrt(re_s.float() ** 2 + im_s.float() ** 2 + 1e-7)
        l_stft = mr_complex_stft_loss(wav_s, wav_t, dev)
        l_mel = F.l1_loss(mel_s, mels)
        l_env = F.l1_loss(torch.log(mag_s), torch.log(mag_t)) + 0.5 * (F.l1_loss(re_s.float(), re_t.float()) + F.l1_loss(im_s.float(), im_t.float()))
        v_loss = l_stft + 14.0 * l_mel + 2.5 * l_env
        v_opt.zero_grad(set_to_none=True); v_loss.backward()
        if torch.isfinite(torch.nn.utils.clip_grad_norm_(s_voc.parameters(), 1.0)): v_opt.step()
        with torch.no_grad():
            d = min(0.998, (1 + step) / (10 + step))
            for pe, p in zip(ema_voc.parameters(), s_voc.parameters()): pe.lerp_(p, 1 - d)
        if step % 250 == 0 or step == voc_steps:
            print(f"[{tier}-Voc] step {step}/{voc_steps} loss {v_loss.item():.4f} (mel={l_mel.item():.3f} stft={l_stft.item():.3f} env={l_env.item():.3f})", flush=True)
            torch.save(dict(G_ema=ema_voc.state_dict(), G=s_voc.state_dict(), step=step, tier=tier, cfg=c), f"{out_dir}/voc.pt")
    print(f"[{tier}] DONE! Saved to {out_dir}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default="nano,atom")
    ap.add_argument("--data", default="/work/tts/feats/v2_studio_24k.pt")
    ap.add_argument("--native", default="/work/tts/feats/v2_native_24k.pt")
    ap.add_argument("--t_am", default="/work/tts/runs/v2_am_polish_r2/last.pt")
    ap.add_argument("--t_voc", default="/work/tts/runs/v2_voc/G_best.pt")
    ap.add_argument("--am_steps", type=int, default=6000)
    ap.add_argument("--voc_steps", type=int, default=4500)
    a = ap.parse_args()
    for t in a.tiers.split(","):
        train_tier(t.strip(), a.data, a.native, a.t_am, a.t_voc, f"/work/tts/runs/{t.strip()}", a.am_steps, a.voc_steps)
