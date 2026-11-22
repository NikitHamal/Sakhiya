"""Sakhiya v6 Deterministic Single-Pass Teacher->Student Distillation (`sakhiya/v6_distill.py`).
Trains `Sakhiya-Voca` (~4.18M params -> ~4.34 MB QDQ int8) and `Sakhiya-Sano` (~1.64M params -> ~1.71 MB QDQ int8)
from `Sakhiya-Teacher` (`v6_teacher`) on `v6_expressive_vedic_24k.pt`:
1. Single-Voice (`sakhiya`, spk=0) & 3-Mode (`news=0`, `storytelling=2`, `vedic_chant=5`) explicit `pitch_add` conditioning.
2. Exact CTC-aligned durations `dur_ctc` + Frozen `MelCTCAligner` pronunciation loss (`ctc_pron`) so sub-2MB & 4MB models
   never drop words, stutter, or slur consonant clusters.
3. Joint Student-Predicted Mel (`y_pred`) -> Source-Filter Harmonic Vocoder distillation against Teacher 24 kHz audio."""
import copy, json, math, os, random, subprocess, sys, time
import numpy as np, torch, torch.nn.functional as F
from . import g2p, sanskrit as SA
from .cfm import SakhiyaAM, STYLE_PROSODY_TARGETS
from .ctc_aligner import MelCTCAligner
from .symbols import SYM2ID, LANG2ID
from .v2_vocoder import VocoderV2, mel_spec_24k, ola64, n_params
from scripts.export_v2_onnx import TIER_CFG
from scripts.freeze_version_pack import MASTER_15_CLIPS
from scripts.train_nano_atom import warm_slice, _bake
from scripts.train_v2_voc import mr_complex_stft_loss, phase_loss

R = "/work/tts"
REPO = "/work/tts/repo" if os.path.exists("/work/tts/repo") else os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _expand_batch_durs(mu, durs, T_max):
    B, L, D = mu.shape
    out = mu.new_zeros(B, D, T_max)
    for b in range(B):
        rep = torch.repeat_interleave(mu[b], durs[b].clamp(min=1), dim=0)[:T_max]
        out[b, :, :rep.shape[0]] = rep.T
    return out


def apply_hiatus_mu_rule(ids, mu):
    """Apply exact reduplicated vowel hiatus mu rule (matching onnx_tts.py / docs/tts.js 1:1) on batch (B, L, 100)."""
    VOW_IDS = {SYM2ID[p] for p in ("a", "i", "u", "e", "o", "ʌ") if p in SYM2ID}
    WB_ID = SYM2ID.get("|", 1)
    B, L, _ = mu.shape
    mu_out = mu.clone()
    ids_cpu = ids.detach().cpu().tolist()
    for b in range(B):
        row = ids_cpu[b]
        last_valid = max((i for i, v in enumerate(row) if v > 0), default=L - 1)
        for t in range(1, last_valid):
            if row[t] == WB_ID and row[t - 1] == row[t + 1] and row[t - 1] in VOW_IDS:
                if t >= 2 and row[t - 2] == WB_ID:
                    mu_out[b, t - 2] = mu[b, last_valid]
                mu_out[b, t - 1] = mu[b, t + 1]
                mu_out[b, t] = 0.45 * mu[b, t + 1] + 0.55 * mu[b, last_valid]
    return mu_out


def build_teacher_anchor_bank(t_am, mean, std, ctc, dev="cuda"):
    """Synthesize clean Teacher distillation anchors across all 15 master clips (clause-split matching SakhiyaONNX._segment)
    using exact EncExport + reduplicated hiatus / Pranava Om / function-word floor + DecExport."""
    from .onnx_tts import _segment, STYLES, PAUSE
    from . import chandas as _ch
    from scripts.export_v2_onnx import EncExport, DecExport
    extra_sents = [
        ("ne", "storytelling", "गाउँलेहरू आआफ्नो खेतबारीमा काम गर्न निस्किए।"),
        ("ne", "storytelling", "आआफ्नो जिम्मेवारी पूरा गर्न हामी सबै आआफ्नो ठाउँबाट लाग्नुपर्छ।"),
        ("ne", "storytelling", "वाह! के अद्भुत दृश्य हो यो! माछापुच्छ्रे हिमाल बिहानको घाममा सुनजस्तै चम्किरहेको छ!"),
        ("ne", "storytelling", "अरे! तिमी आज कति चाँडो आइपुग्यौ? आऊ, सँगै बसेर मिठो चिया खाऔँ!"),
        ("ne", "storytelling", "ओहो! पहाडको टुप्पोबाट झरेको झरना कस्तो सुन्दर देखिएको छ, हगि?"),
        ("sa", "vedic_chant", "ॐ सह नाववतु। सह नौ भुनक्तु। सह वीर्यं करवावहै। तेजस्वि नावधीतमस्तु मा विद्विषावहै॥"),
        ("sa", "vedic_chant", "ॐ असतो मा सद्गमय। तमसो मा ज्योतिर्गमय। मृत्योर्मा अमृतं गमय॥"),
        ("sa", "vedic_chant", "यदा यदा हि धर्मस्य ग्लानिर्भवति भारत। अभ्युत्थानमधर्मस्य तदात्मानं सृजाम्यहम्॥"),
    ]
    raw_s = [(lg, st, tx) for _, lg, st, tx in MASTER_15_CLIPS if lg != "auto"] + extra_sents
    all_s = []
    for lg, st_name, full_txt in raw_s:
        for seg_txt in _segment(full_txt):
            all_s.append((lg, st_name, seg_txt))
        if len(_segment(full_txt)) > 1:
            all_s.append((lg, st_name, full_txt))
    enc_w = EncExport(t_am).to(dev).eval()
    dec_w = DecExport(t_am, steps=8, temp=0.5).to(dev).eval()
    anchors = []
    with torch.no_grad():
        for lg, st_name, txt in all_s:
            k = STYLES.get(st_name, STYLES["neutral"])
            sty_i = int(k.get("style_id", 0))
            if lg == "sa":
                ci = _ch.compile_chandas(txt, bpm=75.0, tonic_hz=215.0)
                ph, off, dm = ci["ph"], ci["off"], ci["dm"]
                sv_sc = 0.52 if ci.get("has_vedic_marks") else 0.48
            else:
                ph, off = g2p.text_to_phones_acc(txt, "ne")
                dm = g2p.phone_dm(ph)
                sv_sc = 1.15
            vi = [i for i, p in enumerate(ph) if p in SYM2ID]
            if len(vi) < 3:
                continue
            ph_valid = [ph[i] for i in vi]
            ids = torch.tensor([[SYM2ID[p] for p in ph_valid]], dtype=torch.long, device=dev)
            po = torch.tensor([[float(off[i]) * sv_sc for i in vi]], dtype=torch.float32, device=dev)
            dm_arr = np.array([float(dm[i]) for i in vi], dtype=np.float32)
            z_long = torch.zeros((1,), dtype=torch.long, device=dev)
            sty_t = torch.tensor([sty_i], dtype=torch.long, device=dev)
            x_enc, ld = enc_w(
                ids, z_long, z_long,
                torch.tensor(float(k["pitch_shift"]), dtype=torch.float32, device=dev),
                torch.tensor(float(k["pitch_scale"]), dtype=torch.float32, device=dev),
                torch.tensor(float(k["energy_scale"]), dtype=torch.float32, device=dev),
                po, sty_t,
            )
            ld_np = ld[0].cpu().numpy()
            ids_np = ids[0].cpu().numpy()
            dur = np.clip(np.exp(ld_np) - 1.0, 1.0, None) * dm_arr
            dur = np.where(np.isin(ids_np, PAUSE), dur * float(k["pause_scale"]), dur)
            dur = np.maximum(np.round(dur / float(k["rate"])), 1).astype(int)
            for idx_p, p in enumerate(ph_valid):
                if p not in ("|", ",", ".", "?", "!", ";"):
                    dur[idx_p] = max(int(dur[idx_p]), 3)
                if p == "h" and (idx_p + 1 == len(ph_valid) or ph_valid[idx_p + 1] in ("|", ",", ".", "?", "!", ";")):
                    dur[idx_p] = min(int(dur[idx_p]), 6)
                if idx_p >= 1 and idx_p + 1 < len(ph_valid) and p == "|" and ph_valid[idx_p - 1] == ph_valid[idx_p + 1] and ph_valid[idx_p - 1] in ("a", "i", "u", "e", "o", "ʌ"):
                    if idx_p >= 2 and ph_valid[idx_p - 2] == "|":
                        dur[idx_p - 2] = 12
                        x_enc[0, idx_p - 2, :100] = x_enc[0, -1, :100]
                    dur[idx_p - 1] = max(int(dur[idx_p - 1]), 16)
                    dur[idx_p] = 7
                    dur[idx_p + 1] = max(int(dur[idx_p + 1]), 15)
                    x_enc[0, idx_p - 1, :100] = x_enc[0, idx_p + 1, :100]
                    x_enc[0, idx_p, :100] = 0.45 * x_enc[0, idx_p + 1, :100] + 0.55 * x_enc[0, -1, :100]
                if lg == "sa" and p == "o" and idx_p + 1 < len(ph_valid) and ph_valid[idx_p + 1] == "m" and (idx_p == 0 or ph_valid[idx_p - 1] in ("|", ",", ";", ".")):
                    dur[idx_p] = int(np.clip(dur[idx_p], 34, 44))
                    dur[idx_p + 1] = int(np.clip(dur[idx_p + 1], 16, 22))
                    if idx_p + 2 < len(ph_valid) and ph_valid[idx_p + 2] == ",":
                        dur[idx_p + 2] = int(np.clip(dur[idx_p + 2], 14, 22))
                if idx_p >= 1 and idx_p + 2 < len(ph_valid) and ph_valid[idx_p - 1] in ("|", ",") and ph_valid[idx_p + 2] in ("|", ",") and p in ("r", "t̪", "n", "m", "h", "tsʰ") and ph_valid[idx_p + 1] in ("ʌ", "o"):
                    dur[idx_p] = int(np.clip(dur[idx_p], 5, 9))
                    dur[idx_p + 1] = int(np.clip(dur[idx_p + 1], 6, 11))
            durs_t = torch.from_numpy(dur).to(dev).long()
            xe = torch.repeat_interleave(x_enc[0], durs_t, dim=0)[None]
            mel_a = dec_w(xe)[0]
            # Strip side-channel bins 98,99 back to clean bin 97 for ground-truth mel target y_gt
            mel_a[98:100] = mel_a[97:98]
            anchors.append(dict(ids=ids[0].cpu(), po=po[0].cpu(), dur_ctc=durs_t.cpu().long(), mel=mel_a.half().cpu(),
                                spk=0, lang=lg, v3_style_id=sty_i, phones=ph_valid, text=txt, split="train", anchor=True))
    return anchors


def distill_v6_student(tier, D, t_am_pt, t_voc_pt, ctc, out_dir, am_steps=2400, voc_steps=1600, update_cb=None, dev="cuda"):
    os.makedirs(out_dir, exist_ok=True)
    c = TIER_CFG[tier]
    mean, std = float(D.get("mean", -5.5028)), float(D.get("std", 2.1869))
    t_ck = torch.load(t_am_pt, map_location=dev, weights_only=False)
    t_sd = {k: v.float() for k, v in t_ck.get("ema", t_ck.get("net", t_ck)).items() if not k.startswith("ctc_rec.")}
    t_am = SakhiyaAM().to(dev).eval(); t_am.load_state_dict(t_sd, strict=False); t_am.stats.copy_(torch.tensor([mean, std], device=dev))
    _bake(t_am)

    anchors = build_teacher_anchor_bank(t_am, mean, std, ctc, dev)
    base_tr = [x for x in D["items"] if x.get("split") == "train" and 3 <= len(x["ids"]) and 24 <= x["mel"].shape[1] <= 850]
    exp_tr = [x for x in base_tr if x.get("v3_style_id", 0) in (2, 5)]
    pool = base_tr + exp_tr * 2 + anchors * 35

    s_am = SakhiyaAM(**{k: c[k] for k in ("d_enc", "L_enc", "ff_enc", "d_vel", "nb_vel", "h_dur")}).to(dev)
    s_am.stats.copy_(torch.tensor([mean, std], device=dev))
    warm_slice(s_am, t_am.state_dict()); _bake(s_am)
    torch.nn.init.zeros_(s_am.vel.out.weight); torch.nn.init.zeros_(s_am.vel.out.bias)
    start_step = 1
    if os.path.exists(f"{out_dir}/am.pt"):
        ck_s = torch.load(f"{out_dir}/am.pt", map_location=dev, weights_only=False)
        s_am.load_state_dict(ck_s.get("ema", ck_s.get("net", ck_s)), strict=False)
        start_step = int(ck_s.get("step", 0)) + 1
    ema_am = copy.deepcopy(s_am).eval()
    opt = torch.optim.AdamW(s_am.parameters(), lr=8e-4, betas=(0.9, 0.99), weight_decay=1e-2)
    c_idx = torch.arange(100, device=dev, dtype=torch.float32)[None, :, None]
    pros_tbl = torch.tensor(STYLE_PROSODY_TARGETS, dtype=torch.float32, device=dev)

    for step in range(start_step, am_steps + 1):
        xs = random.sample(pool, 36) + random.sample(anchors, min(12, len(anchors)))
        B = len(xs); L = max(len(x["ids"]) for x in xs); T = max(int(x["dur_ctc"].sum().item()) for x in xs)
        T = min(max(T, max(x["mel"].shape[1] for x in xs)), 850)
        ids = torch.zeros(B, L, dtype=torch.long, device=dev); durs = torch.zeros(B, L, dtype=torch.long, device=dev)
        po = torch.zeros(B, L, dtype=torch.float32, device=dev)
        for k, x in enumerate(xs):
            Lk = len(x["ids"]); ids[k, :Lk] = x["ids"].to(dev); durs[k, :Lk] = x["dur_ctc"].to(dev).long()
            if "po" in x and x["po"] is not None: po[k, :Lk] = x["po"][:Lk].to(dev)
        il = torch.tensor([len(x["ids"]) for x in xs], device=dev); yl = durs.sum(1).clamp(max=T)
        im = torch.arange(L, device=dev)[None] < il[:, None]; ym = torch.arange(T, device=dev)[None] < yl[:, None]; ymf = ym[:, None].float()
        spks = torch.zeros(B, dtype=torch.long, device=dev); stud = torch.ones(B, dtype=torch.long, device=dev)
        langs = torch.tensor([0 if (x["lang"] == "sa" and random.random() < 0.85) else LANG2ID.get(x["lang"], 0) for x in xs], device=dev)
        stys = torch.tensor([int(x.get("v3_style_id", 0)) for x in xs], device=dev).clamp(0, 5)
        pros_b = pros_tbl[stys]; ones_m = torch.ones((B, 1), dtype=torch.float32, device=dev)
        lr = max(8e-4 * 0.5 * (1.0 + math.cos(math.pi * (step - start_step) / max(1, am_steps - start_step + 1))), 5e-5)
        for g in opt.param_groups: g["lr"] = lr

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                h_t, mu_t, g_t = t_am.enc(ids, spks, langs, stys, stud, im, pitch_add=po, prosody=pros_b, sty_mask=ones_m)
                f0_t = t_am.enc.f0_pred(h_t); eng_t = t_am.enc.energy_pred(h_t)
                mu_y_t = _expand_batch_durs(mu_t, durs, T) * ymf
                t_idx = torch.arange(T, device=dev, dtype=torch.float32)[None, None, :]
                xt = (torch.sin(c_idx * 12.9898 + t_idx * 78.233) * torch.cos(c_idx * 39.346 + t_idx * 11.135) * 0.7071).expand(B, 100, T).to(mu_y_t.dtype)
                for i_o in range(4):
                    to = torch.full((B,), i_o * 0.25, device=dev)
                    k1 = t_am.vel(xt, to, mu_y_t, g_t, ymf)
                    xt = xt + 0.25 * t_am.vel(xt + 0.125 * k1, to + 0.125, mu_y_t, g_t, ymf)
                y_clean = xt * ymf
                ld_tgt = torch.log(durs.float().clamp(min=1.0)) * im.float()
                log_p_t = ctc(y_clean.float())

            h_s, mu_s, g_s = s_am.enc(ids, spks, langs, stys, stud, im, pitch_add=po, prosody=pros_b, sty_mask=ones_m)
            l_heads = 0.5 * (((s_am.enc.f0_pred(h_s) - f0_t) ** 2 + (s_am.enc.energy_pred(h_s) - eng_t) ** 2) * im[:, :, None]).sum() / im.sum().clamp(min=1)
            mu_y_s = _expand_batch_durs(mu_s, durs, T) * ymf
            l_prior = 2.2 * (((mu_s - mu_t).abs() + (mu_s - mu_t) ** 2) * im[:, :, None]).sum() / (im.sum() * 100.0)
            ld_pred = s_am.dur.pred_logdur(ids, h_s, im)
            l_dur = F.smooth_l1_loss(ld_pred * im.float(), ld_tgt, reduction="sum") / im.sum().clamp(min=1)
            t0_z = torch.zeros((B,), device=dev, dtype=torch.float32)
            y_pred = (mu_y_s + s_am.vel(mu_y_s, t0_z, mu_y_s, g_s, ymf)) * ymf
            denom = (ymf.sum() * 100.0).clamp(min=1.0)
            l_mel = ((1.6 * (y_pred - y_clean).abs() + (y_pred - y_clean) ** 2) * ymf).sum() / denom
            df_p = y_pred[:, 1:, :] - y_pred[:, :-1, :]; df_t = y_clean[:, 1:, :] - y_clean[:, :-1, :]
            l_df = ((df_p - df_t).abs() * ymf).sum() / denom
            l_std = ((y_pred.float().std(dim=1, keepdim=True) - y_clean.float().std(dim=1, keepdim=True)).abs() * ymf).sum() / ymf.sum().clamp(min=1.0)
            ym_t = ymf[:, :, 1:] * ymf[:, :, :-1]
            l_dt = (((y_pred[:, :, 1:] - y_pred[:, :, :-1]) - (y_clean[:, :, 1:] - y_clean[:, :, :-1])).abs() * ym_t).sum() / (ym_t.sum() * 100.0 + 1.0)
            cep_p = torch.fft.rfft(y_pred.float(), dim=1).abs()[:, 4:48, :]
            cep_t = torch.fft.rfft(y_clean.float(), dim=1).abs()[:, 4:48, :]
            l_cep = ((cep_p - cep_t).abs() * ymf).sum() / (ymf.sum() * 44.0).clamp(min=1.0)
            log_p_s = ctc(y_pred.float())
            l_kl = F.kl_div(log_p_s, log_p_t.exp().detach(), reduction="none").sum(-1)
            l_kl = ((l_kl.transpose(0, 1) if l_kl.shape[0] != ym.shape[0] else l_kl) * ym.float()).sum() / ym.sum().clamp(min=1.0)
            l_ctc = F.relu(ctc.ctc_loss(0.3 * mu_y_s + 0.7 * y_pred, yl, ids, il) - 0.05)
            loss = l_mel + 2.4 * l_df + 0.8 * l_std + 1.2 * l_dt + 0.65 * l_cep + l_prior + l_heads + 0.6 * l_dur + 0.45 * l_kl + 0.25 * l_ctc

        opt.zero_grad(set_to_none=True); loss.backward()
        if torch.isfinite(torch.nn.utils.clip_grad_norm_(s_am.parameters(), 1.0)): opt.step()
        with torch.no_grad():
            d_ema = min(0.996, (1 + (step - start_step)) / (10 + (step - start_step)))
            for pe, p in zip(ema_am.parameters(), s_am.parameters()): pe.lerp_(p, 1.0 - d_ema)
        if step % 150 == 0 or step == am_steps:
            tmp_am = f"{out_dir}/am.pt.tmp"
            torch.save(dict(ema=ema_am.state_dict(), net=s_am.state_dict(), step=step, tier=tier, cfg=c, mean=mean, std=std, pre_baked_v25=True, seqkd_1pass=True), tmp_am)
            os.replace(tmp_am, f"{out_dir}/am.pt")
            if update_cb: update_cb(step, am_steps, f"{tier}-AM step {step}/{am_steps} mel={l_mel.item():.3f} dur={l_dur.item():.3f} ctc={l_ctc.item():.3f}")
            print(f"[{tier}-AM] step={step}/{am_steps} mel={l_mel.item():.3f} dur={l_dur.item():.3f} ctc={l_ctc.item():.3f}", flush=True)

    t_voc = VocoderV2().to(dev).eval()
    vck = torch.load(t_voc_pt, map_location=dev, weights_only=False)
    t_vsd = vck.get("G_ema", vck.get("G", vck)) if isinstance(vck, dict) else vck
    t_voc.load_state_dict({k: v.float() for k, v in t_vsd.items()}, strict=False)
    vlong_p = f"{R}/runs/v6_{tier}_voclong/voc.pt"
    prev_vp = vlong_p if os.path.exists(vlong_p) else (f"{out_dir}/voc.pt" if os.path.exists(f"{out_dir}/voc.pt") else None)
    if voc_steps <= 0 and prev_vp:
        return f"{out_dir}/am.pt", prev_vp
    s_voc = VocoderV2(ch=c["v_ch"], res_kernels=c["v_rk"]).to(dev)
    if prev_vp:
        ck_v = torch.load(prev_vp, map_location=dev, weights_only=False)
        s_voc.load_state_dict(ck_v.get("G_ema", ck_v.get("G", ck_v)), strict=False)
    else:
        with torch.no_grad():
            for k, p in s_voc.state_dict().items():
                if k not in t_vsd or not torch.is_floating_point(p): continue
                s = t_vsd[k].float().to(p.device)
                if s.shape == p.shape:
                    p.copy_(s)
                elif s.ndim == p.ndim and all(a >= b for a, b in zip(s.shape, p.shape)):
                    sl = tuple(slice(0, b) for b in p.shape)
                    sc = math.sqrt(s.shape[1] / p.shape[1]) if s.ndim >= 2 and s.shape[1] > p.shape[1] else 1.0
                    p.copy_(s[sl] * min(sc, 1.25))
    s_voc.src.f0_net.load_state_dict(t_voc.src.f0_net.state_dict())
    for p in s_voc.src.f0_net.parameters(): p.requires_grad_(False)
    ema_voc = copy.deepcopy(s_voc).eval()
    v_opt = torch.optim.AdamW([p for p in s_voc.parameters() if p.requires_grad], lr=9e-4, betas=(0.8, 0.99), weight_decay=1e-3)
    seg_f = 72
    for step in range(1, voc_steps + 1):
        b_xs = [x for x in random.sample(pool, 64) if x["mel"].shape[1] >= seg_f][:32]
        mels = torch.stack([x["mel"][:, s:(s + seg_f)] for x in b_xs for s in [random.randint(0, x["mel"].shape[1] - seg_f)]]).to(dev).float()
        with torch.no_grad():
            re_t, im_t, _, _ = t_voc.spectrum_and_f0(mels)
            wav_t = ola64(torch.complex(re_t.float(), im_t.float()), t_voc.win)
            mag_t = torch.sqrt(re_t.float() ** 2 + im_t.float() ** 2 + 1e-7)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            re_s, im_s, _, _ = s_voc.spectrum_and_f0(mels)
        wav_s = ola64(torch.complex(re_s.float(), im_s.float()), s_voc.win)
        mel_s = mel_spec_24k(wav_s, dev)[:, :, :seg_f]
        mag_s = torch.sqrt(re_s.float() ** 2 + im_s.float() ** 2 + 1e-7)
        l_ph = phase_loss(wav_s, wav_t, 512, dev)
        v_loss = mr_complex_stft_loss(wav_s, wav_t, dev) + 14.0 * F.l1_loss(mel_s, mels) + 2.5 * F.l1_loss(torch.log(mag_s), torch.log(mag_t)) + 0.8 * l_ph
        v_opt.zero_grad(set_to_none=True); v_loss.backward()
        if torch.isfinite(torch.nn.utils.clip_grad_norm_(s_voc.parameters(), 1.0)): v_opt.step()
        with torch.no_grad():
            for pe, p in zip(ema_voc.parameters(), s_voc.parameters()): pe.lerp_(p, 0.008)
        if step % 200 == 0 or step == voc_steps:
            torch.save(dict(G_ema=ema_voc.state_dict(), G=s_voc.state_dict(), step=step, tier=tier, cfg=c), f"{out_dir}/voc.pt")
            if update_cb: update_cb(step, voc_steps, f"{tier}-Voc step {step}/{voc_steps} loss={v_loss.item():.3f}")
            print(f"[{tier}-Voc] step={step}/{voc_steps} loss={v_loss.item():.3f}", flush=True)
    return f"{out_dir}/am.pt", f"{out_dir}/voc.pt"
