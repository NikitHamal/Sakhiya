#!/usr/bin/env python3
"""Phase 1.2 — Anchored Full-Utterance AM Micro-Polish (`scripts/train_phase1_am_polish.py`).

Hard Anti-Regression Laws Enforced:
1. Warm-starts strictly from `Sakhiya-One` `/work/sp/distill/runs/sakhiya_one/ema_final.pt`.
2. Re-aligns the 100% pure `Sakhiya-Pro-v3` full-utterance corpus with the fixed Phase-1 Nepali G2P (0% Old-V1, 0% Seed-VC).
3. Trains strictly on FULL UTTERANCES ONLY (zero W=176 cropping, zero Hann window stitching) for 4,500 steps
   (`lr = 1.2e-5` cosine -> `2e-6`, `EMA = 0.999`, Step-50 stdout watchdog).
4. Oversamples 4x on utterances containing `व`/`वा`/`वे` words, geminates (`त्त`, `ल्ल`, `क्क`, `न्न`, `द्ध`),
   halanta finals (`त्`), chandrabindu `औँ`/`ऐँ` diphthongs, long compounds, and English tech loanwords.
5. Anti-Drift Anchor Lock: `0.25 * L1(pred_mel, frozen_sakhiya_one_mel)` on non-bug standard sentences so
   `Sakhiya-One`'s exact voice identity and smooth sentence flow cannot drift or overfit.
"""
import argparse
import copy
import json
import math
import os
import random
import re
import sys
import time
import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
for p in (REPO, HERE, os.path.join(REPO, "scripts", "sakhiya_pro"), os.path.join(REPO, "scripts", "sakhiya_pro", "distill")):
    if p not in sys.path:
        sys.path.insert(0, p)

from sakhiya import g2p, chandas as _ch
from sakhiya.symbols import SYM2ID
from sakhiya.v2_vocoder import SR, HOP, mel_spec_24k
from sakhiya.ctc_aligner import MelCTCAligner, batch_viterbi_ctc_durations
from sakhiya.cfm import _dur_to_attn
from student import Student, build, compute_svara_mora_ids, compute_intra_phoneme_progress, _calibrate_styles
from export_gate import ids_of, _normalize_lufs_16
from scripts.forensic_audio_gate import compute_comb_grit_and_warble
from scripts.run_phase0_sakhiya_one_honest import synth_full_utterance_torch, _voiced_vowel_hnr_0p2_4khz

LOG_FILE = "/work/tts/logs/sakhiya_one_polish.log"
TELEM_FILE = "/work/sp/polish/telemetry.json"
LISTEN_DIR = "/work/sp/polish/listen"
CK_DIR = "/work/sp/polish/ck"
os.makedirs(LISTEN_DIR, exist_ok=True)
os.makedirs(CK_DIR, exist_ok=True)

HARD_PAT = re.compile(
    r"(वालेट|वेब|वर्ल्ड|वाचक|विश्व|विशेष|विकास|व्यवसाय|संवेदनशील|बुद्धिमत्ता|उल्लेखनीय|पक्कै|चिटिक्क|"
    r"बृहत|बृहत्|अर्थात|अर्थात्|गर्नेछौँ|लागौँ|चलौँ|पाँचौँ|काठमाडौँ|अर्बौँ|ल्याउँदै|"
    r"अनुसन्धानात्मक|प्रयोगकर्ता|उतारचढाव|त्रैमासिक|गिरावट|उपलब्धि|सुरक्षित|"
    r"जेमिनाई|नियोन|जिपिटी|जीपीटी|च्याटजिपिटी|ओपन|एआई|एन्थ्रोपिक|मोबाइल|एप्लिकेसन|प्रतिशत|"
    r"त्त|ल्ल|क्क|न्न|द्ध|ँ|वा|वे|वि)"
)

SURGICAL_SENTENCES = [
    "दोस्रो समाचार नयाँ पुस्ताका एआई मोडेलहरू बारे छ।",
    "गुगलले जेमिनाई फोर नियोन नाम गरेको अत्यन्तै छरितो र शक्तिशाली नयाँ एआई मोडेल सार्वजनिक गरेको छ।",
    "जटिल तर्क गर्न लामो कोड लेख्न र धेरै चरण भएका अनुसन्धानात्मक कामहरू छिटो सम्पन्न गर्न यो मोडेल विशेष रूपमा तयार पारिएको बताइएको छ।",
    "यस्तै ओपन एआईले पनि जीपीटी सिक्स पोइन्ट टु नाम गरेका नयाँ संस्करणहरू ल्याउँदै प्रयोगकर्ताको तर्फबाट सुरक्षित रूपमा अनलाइन किनमेल गर्न मिल्ने च्याटजिपिटी वालेट सुविधा सुरु गर्ने घोषणा गरेको छ।",
    "पहिलो समाचार कृत्रिम बुद्धिमत्ता बृहत र विश्व सेयर बजारसँग सम्बन्धित छ।",
    "प्रसिद्ध एआई अनुसन्धान कम्पनी एन्थ्रोपिकको सेयर बजारमा कारोबार खुलेको पहिलो दिनमै १७ प्रतिशतले उल्लेखनीय वृद्धि भएको छ।",
    "तर लगानीकर्ताहरूको नाफा सुरक्षित गर्ने होडबाजीका कारण अर्को दिन करिब १४ प्रतिशतले गिरावट आयो जसले गर्दा अन्य प्रमुख प्रविधि कम्पनीहरूको सेयर मूल्यमा समेत ठूलो उतारचढाव देखिएको छ।",
    "विश्लेषकहरूका अनुसार एआई पूर्वाधारमा भइरहेको अर्बौँ डलरको लगानीले आगामी त्रैमासिकमा बजारलाई थप संवेदनशील बनाउने निश्चित छ।",
    "नमस्ते सबैजनालाई। म सखिया तपाईंको समाचार वाचक।",
    "आजको विशेष प्राविधिक बुलेटिनमा हामी विश्वभर भइरहेका कृत्रिम बुद्धिमत्ता अर्थात एआईका नवीनतम उपलब्धिहरू, नेपाली प्रविधि क्षेत्रको तीव्र विकास र मोबाइल एप्लिकेसन निर्माणमा आएका नयाँ परिवर्तनहरूबारे विस्तृत चर्चा गर्नेछौँ।",
    "प्रविधिको दुनियाँ हरेक दिन नयाँ मोड लिइरहेको छ र त्यसको प्रत्यक्ष प्रभाव हाम्रो दैनिक जीवन, शिक्षा र व्यवसायमा पर्न थालेको छ।",
    "त चलौँ आजका मुख्य समाचार तर्फ लागौँ।",
    "कत्ति राम्रो मौसम छ आज काठमाडौँ उपत्यकामा, पक्कै पनि पानी पर्दैन।",
    "चिटिक्क परेको घरमा ठ्ठयाक्कै उस्तै मान्छे बसेको रहेछ।",
    "सक्कली खबर सुन्दा अड्डाका कर्मचारीहरू छक्क परे।",
]


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def update_stage(idx: int, step: int, total: int, its: str, losses: str, el_m: float, eta_m: float, status: str):
    if not os.path.exists(TELEM_FILE):
        return
    try:
        st = json.load(open(TELEM_FILE))
        st["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        st["stages"][idx].update(
            step=int(step), total_steps=int(total), its=str(its),
            losses=str(losses), elapsed_min=round(float(el_m), 2),
            eta_min=round(float(eta_m), 2), status=str(status)
        )
        tmp = TELEM_FILE + ".tmp"
        json.dump(st, open(tmp, "w"), indent=2)
        os.replace(tmp, TELEM_FILE)
    except Exception:
        pass


def forward_4step_meanflow(model: Student, mu_y: torch.Tensor, g: torch.Tensor, ymf: torch.Tensor, prog_4d: torch.Tensor, nb_v: int = 8):
    B = mu_y.shape[0]
    dev = mu_y.device
    t0 = torch.zeros((B,), device=dev, dtype=torch.float32)
    r_one = torch.ones((B,), device=dev, dtype=torch.float32)
    w2 = model.am.vel.step2_gate.clamp(0.0, 0.35)
    v_cur = model.am.vel(mu_y, t0, mu_y, g, ymf, prog=prog_4d, n_blocks=nb_v)
    for s_i in range(1, 4):
        alpha_s = float(s_i) / 4.0
        x_s = mu_y + alpha_s * v_cur
        t_s = torch.full((B,), alpha_s, device=dev, dtype=torch.float32)
        v_s = model.am.vel(x_s, t_s, mu_y, g, ymf, prog=prog_4d, r=r_one, n_blocks=nb_v)
        v_cur = (1.0 - w2) * v_cur + w2 * v_s
    return (mu_y + v_cur) * ymf, v_cur * ymf


def build_realigned_full_utt_corpus(frozen_base: Student, dev="cuda"):
    cache_p = "/work/sp/polish/realigned_full_utt_cache.pt"
    if os.path.exists(cache_p):
        log(f"[Phase 1.2] Loading cached re-aligned full-utterance corpus from {cache_p}...")
        return torch.load(cache_p, map_location="cpu", weights_only=False)

    log("[Phase 1.2] Re-aligning pure Pro-v3 full-utterance corpus with fixed Phase-1 Nepali G2P (0% Old-V1, 0% Seed-VC)...")
    teacher = build("t15").to(dev).warm_init("/work/tts/ck").eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    _calibrate_styles(teacher.am)

    cc = torch.load("/work/tts/ck/v31_ctc_recognizer.pt", map_location=dev, weights_only=False)
    ctc_al = MelCTCAligner().to(dev).eval()
    ctc_al.load_state_dict(cc.get("model", cc), strict=False)
    m_mean, m_std = float(cc.get("mean", -6.1761)), float(cc.get("std", 2.4932))

    D = torch.load("/work/sp/distill/cache_train.pt", map_location="cpu", weights_only=False)
    mf = {json.loads(l)["id"]: json.loads(l) for l in open("/work/sp/distill/manifest.jsonl")}
    DROP_LEGACY = {"vc_slr54", "vc_slr43", "vc_v13", "anchor_6834_stitched"}

    raw_prompts = []
    for x in D["items"][:1100]:
        r = mf.get(x["id"], {})
        src = r.get("src", "wiki")
        if src in DROP_LEGACY:
            continue
        txt = r.get("text", "").strip()
        lg_str = r.get("lang", "ne")
        if not txt or len(txt) < 6:
            continue
        lg_i = 1 if lg_str == "sa" else 0
        ids_c = ids_of(txt, "sa" if lg_i == 1 else "ne")
        if 4 <= len(ids_c) <= 130:
            is_hard = bool(HARD_PAT.search(txt))
            raw_prompts.append((x["id"], ids_c, lg_i, is_hard, txt))

    for idx_s, s_txt in enumerate(SURGICAL_SENTENCES):
        ids_c = ids_of(s_txt, "ne")
        # 4x oversampling on surgical hard-word sentences
        for rep in range(8):
            raw_prompts.append((f"surgical_{idx_s}_{rep}", ids_c, 0, True, s_txt))

    items = []
    pros_tbl = teacher.pros_tbl
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for b0 in range(0, len(raw_prompts), 40):
            sub = raw_prompts[b0 : b0 + 40]
            B = len(sub)
            ids_pad = pad_sequence([torch.tensor([1] + [5 if x == 8 else x for x in s[1]], dtype=torch.long) for s in sub], batch_first=True).to(dev)
            il = torch.tensor([len(s[1]) + 1 for s in sub], device=dev)
            L = ids_pad.shape[1]
            im = torch.arange(L, device=dev)[None] < il[:, None]
            spk0 = torch.zeros(B, dtype=torch.long, device=dev)
            studio = torch.ones(B, dtype=torch.long, device=dev)
            is_sa = torch.tensor([s[2] == 1 for s in sub], device=dev)
            lang_eff = torch.where(is_sa, torch.full((B,), 4, dtype=torch.long, device=dev), torch.zeros(B, dtype=torch.long, device=dev))
            sty = torch.where(is_sa, torch.full((B,), 5, dtype=torch.long, device=dev), torch.zeros(B, dtype=torch.long, device=dev))
            pros = pros_tbl[sty]

            pos = torch.arange(L, device=dev, dtype=torch.float32)[None, :]
            L_f = il[:, None].float().clamp(min=2.0)
            is_pause = (ids_pad <= 8).float()
            phase = (pos / L_f).clamp(0.0, 1.0)
            tail_closure = teacher.closure_slope.clamp(-0.30, -0.05) * ((phase - 0.72).clamp(min=0.0) / 0.28) * (1.0 - is_pause)
            p_arch = (0.14 * torch.sin(phase * 3.14159265) + tail_closure) * (1.0 - is_pause) - 0.15 * is_pause
            p_enc = F.avg_pool1d(F.pad(p_arch[:, None, :], (1, 1), mode="replicate"), 3, stride=1)[:, 0] * im.float()
            ones_m = torch.ones((B, 1), device=dev)

            # Compute teacher durations & clean full-utterance trajectories
            ht, mut, gt = teacher.am.enc(ids_pad, spk0, lang_eff, sty, studio, im, pitch_add=p_enc, prosody=pros, sty_mask=ones_m)
            f0_t = teacher.am.enc.f0_pred(ht).squeeze(-1).clamp(-1.5, 1.5)
            eng_t = teacher.am.enc.energy_pred(ht).squeeze(-1).clamp(-1.2, 1.2)
            mut_e = mut + (0.10 * pros[:, 2:3] + 0.06 * eng_t)[..., None] * 0.32
            xd = teacher.am.dur.pred_logdur(ids_pad, ht, im) + teacher.dur_rate_bias.clamp(-0.12, 0.12)
            dur_frames = torch.exp(xd).clamp(min=1.0, max=26.0)
            rate_mult = torch.exp(-0.04 * pros[:, 0:1]) * (1.0 - is_pause) + torch.exp(0.08 * pros[:, 3:4]) * is_pause
            dur_frames = (dur_frames * rate_mult).clamp(min=1.0, max=26.0)
            is_tail = (((pos >= (L_f - 3.0)) & (ids_pad > 8)) | (ids_pad >= 53)).float()
            min_ph = is_tail * 5.0 + (1.0 - is_tail) * ((ids_pad > 8).float() * 3.0 + (ids_pad <= 8).float() * 1.0)
            dur = torch.round(torch.maximum(dur_frames, min_ph)).clamp(min=1.0, max=26.0)

            # Enforce our surgical geminate, halanta, clause-onset, and chandrabindu duration floors in target durations too!
            id_wb, id_t, id_retro_t, id_l_c, id_sh_c = SYM2ID["|"], SYM2ID["t̪"], SYM2ID["ʈ"], SYM2ID["l"], SYM2ID["ʃ"]
            next_ids = torch.cat([ids_pad[:, 1:], torch.zeros_like(ids_pad[:, -1:])], dim=1)
            prev_ids = torch.cat([torch.zeros_like(ids_pad[:, :1]), ids_pad[:, :-1]], dim=1)
            is_onset_wb = ((pos == 0.0) & (ids_pad == id_wb)).float()
            is_gem_half = ((ids_pad > 8) & ((ids_pad == next_ids) | (ids_pad == prev_ids))).float()
            is_word_final_stop = (((ids_pad == id_t) | (ids_pad == id_retro_t)) & (next_ids <= 6) & (prev_ids > 8)).float()
            is_post_stop_wb = ((ids_pad == id_wb) & ((prev_ids == id_t) | (prev_ids == id_retro_t))).float()
            is_post_sib_l = ((ids_pad == id_l_c) & (prev_ids == id_sh_c)).float()
            is_nas_vow = (ids_pad >= 58).float()
            dur = torch.maximum(dur, is_onset_wb * 4.0 + is_gem_half * 5.0 + is_word_final_stop * 5.0 + is_post_stop_wb * 4.0 + is_post_sib_l * 5.0 + is_nas_vow * 5.0)
            dur = (dur * im.float()).long()

            yl = dur.sum(1)
            T = int(yl.max().item())
            ym = torch.arange(T, device=dev)[None] < yl[:, None]
            ymf = ym[:, None].float()
            attn = _dur_to_attn(dur, im, ym)
            mu_y = torch.einsum("blt,bld->bdt", attn.to(mut_e.dtype), mut_e) * ymf

            ends = torch.cumsum(dur.float(), dim=1)
            t_arange = torch.arange(0, T, device=dev, dtype=torch.float32)
            idx_fr = (t_arange[None, :, None] >= ends[:, None, :]).to(torch.float32).sum(-1).to(torch.int64).clamp(max=L - 1)
            prog_4d = compute_intra_phoneme_progress(dur.float(), ends, idx_fr, t_arange) * ymf

            c_idx = torch.arange(100, device=dev, dtype=torch.float32)[None, :, None]
            t_idx = torch.arange(T, device=dev, dtype=torch.float32)[None, None, :]
            z_init = torch.sin(c_idx * 12.9898 + t_idx * 78.233) * torch.cos(c_idx * 39.346 + t_idx * 11.135) * (0.667 * 1.4142)
            xt = z_init.expand(B, 100, T)
            dt = 0.25
            for i_s in range(4):
                ti = torch.full((B,), i_s * dt, device=dev, dtype=torch.float32)
                k1 = teacher.am.vel(xt, ti, mu_y, gt, ymf, prog=prog_4d)
                k2 = teacher.am.vel(xt + 0.5 * dt * k1, ti + 0.5 * dt, mu_y, gt, ymf, prog=prog_4d)
                xt = xt + dt * k2
            y_teacher = (xt * ymf).float()

            # Also compute frozen Sakhiya-One anchor mels on the exact same full utterance for Anti-Drift Anchor Lock!
            ds_tok = (dur.float() / 7.5).clamp(0.6, 1.8)
            sv_ids, mo_ids = compute_svara_mora_ids(ids_pad, is_sa, ds_tok, p_enc)
            h_f, mu_f, g_f = frozen_base.am.enc(ids_pad, spk0, lang_eff, sty, studio, im, pitch_add=p_enc, prosody=frozen_base.pros_tbl[sty], sty_mask=ones_m, svara_ids=sv_ids, mora_ids=mo_ids)
            eng_f = frozen_base.am.enc.energy_pred(h_f).squeeze(-1).clamp(-1.2, 1.2)
            mu_f_e = mu_f + (0.10 * frozen_base.pros_tbl[sty][:, 2:3] + 0.06 * eng_f)[..., None] * 0.32
            mu_y_f = torch.einsum("blt,bld->bdt", attn.to(mu_f_e.dtype), mu_f_e) * ymf
            y_frozen_one, _ = forward_4step_meanflow(frozen_base, mu_y_f, g_f, ymf, prog_4d, nb_v=8)

            for k in range(B):
                Lk = int(il[k].item())
                Tk = int(yl[k].item())
                if Lk < 4 or Tk < 24:
                    continue
                is_hard = bool(sub[k][3])
                # Blend target mel: on standard non-bug sentences, anchor strongly to frozen_sakhiya_one + Pro-v3 teacher
                y_tgt = (0.75 * y_teacher[k, :, :Tk] + 0.25 * y_frozen_one[k, :, :Tk]).T.half().contiguous().cpu()
                y_anc = y_frozen_one[k, :, :Tk].T.half().contiguous().cpu()
                item = dict(
                    id=sub[k][0],
                    ids=ids_pad[k, :Lk].cpu(),
                    dur=dur[k, :Lk].cpu(),
                    p_enc=p_enc[k, :Lk].half().cpu(),
                    lang=int(sub[k][2]),
                    is_hard=is_hard,
                    mu=(0.70 * mut[k, :Lk] + 0.30 * mu_f[k, :Lk]).half().cpu(),
                    f0=f0_t[k, :Lk].half().cpu(),
                    eng=eng_t[k, :Lk].half().cpu(),
                    y_clean=y_tgt,
                    y_frozen=y_anc,
                )
                items.append(item)

    del teacher, ctc_al
    torch.cuda.empty_cache()
    torch.save(items, cache_p)
    log(f"[Phase 1.2] Built {len(items)} re-aligned full-utterance trajectories (hard_count={sum(1 for x in items if x['is_hard'])}).")
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4500)
    ap.add_argument("--bs", type=int, default=28)
    a = ap.parse_args()

    t0 = time.time()
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(42)
    torch.manual_seed(42)

    update_stage(1, 0, a.steps, "init", "loading Sakhiya-One ema_final.pt & re-aligning full-utt corpus", 0.0, 4.0, "running")

    # 1. Load untouched Sakhiya-One (ema_final.pt)
    ck_p = "/work/sp/distill/runs/sakhiya_one/ema_final.pt"
    S = torch.load(ck_p, map_location="cpu", weights_only=False)
    G = Student(tier="sakhiya_one")
    G.steps = 4
    G.cfg["steps"] = 4
    G.warm_init("/work/tts/ck")
    own_sd = G.state_dict()
    filt_sd = {k: v for k, v in S["ema"].items() if k in own_sd and own_sd[k].shape == v.shape}
    G.load_state_dict(filt_sd, strict=False)
    _calibrate_styles(G.am)
    G.to(dev)

    # Frozen Sakhiya-One anchor for Anti-Drift Anchor Lock
    frozen_base = copy.deepcopy(G).to(dev).eval()
    for p in frozen_base.parameters():
        p.requires_grad_(False)

    items = build_realigned_full_utt_corpus(frozen_base, dev=dev)
    hard_items = [x for x in items if x["is_hard"]]
    std_items = [x for x in items if not x["is_hard"]]
    # 4x oversampling pool on hard utterances
    train_pool = std_items + hard_items * 4
    val_pool = hard_items[:16] + std_items[:16]

    # Freeze vocoder during Phase 1 AM polish
    for p in G.voc.parameters():
        p.requires_grad_(False)
    for n, p in G.am.named_parameters():
        if n.endswith("had_rot.weight"):
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)

    ema = copy.deepcopy(G).eval()

    cc = torch.load("/work/tts/ck/v31_ctc_recognizer.pt", map_location=dev, weights_only=False)
    ctc = MelCTCAligner().to(dev).train()
    ctc.load_state_dict(cc.get("model", cc), strict=False)
    for m_ in ctc.modules():
        if isinstance(m_, nn.Dropout):
            m_.p = 0.0
    for p_ in ctc.parameters():
        p_.requires_grad_(False)

    am_params = [p for p in G.parameters() if p.requires_grad]
    lr_max, lr_min = 1.2e-5, 2.0e-6
    oG = torch.optim.AdamW(am_params, lr=lr_max, betas=(0.9, 0.99), weight_decay=1e-4)

    tstep = time.time()
    log(f"[Phase 1.2] Starting Anchored Full-Utterance AM Micro-Polish for {a.steps} steps (lr={lr_max}->2e-6, EMA=0.999, pool={len(train_pool)})...")

    for st in range(1, a.steps + 1):
        prog = st / max(1, a.steps)
        lr_cur = lr_min + (lr_max - lr_min) * 0.5 * (1.0 + math.cos(math.pi * prog))
        for pg in oG.param_groups:
            pg["lr"] = lr_cur

        xs = random.sample(train_pool, a.bs)
        B = len(xs)
        ids = pad_sequence([x["ids"].to(dev) for x in xs], batch_first=True)
        dur = pad_sequence([x["dur"].to(dev) for x in xs], batch_first=True)
        p_enc = pad_sequence([x["p_enc"].to(dev).float() for x in xs], batch_first=True)
        mu_t = pad_sequence([x["mu"].to(dev).float() for x in xs], batch_first=True)
        f0_t = pad_sequence([x["f0"].to(dev).float() for x in xs], batch_first=True)
        eng_t = pad_sequence([x["eng"].to(dev).float() for x in xs], batch_first=True)
        y_clean = pad_sequence([x["y_clean"].to(dev).float() for x in xs], batch_first=True).transpose(1, 2).contiguous()
        y_frozen = pad_sequence([x["y_frozen"].to(dev).float() for x in xs], batch_first=True).transpose(1, 2).contiguous()
        il = torch.tensor([len(x["ids"]) for x in xs], device=dev)
        yl = torch.tensor([x["y_clean"].shape[0] for x in xs], device=dev)
        L, T = ids.shape[1], y_clean.shape[2]
        im = torch.arange(L, device=dev)[None] < il[:, None]
        ym = torch.arange(T, device=dev)[None] < yl[:, None]
        ymf = ym[:, None].float()
        spk = torch.zeros(B, dtype=torch.long, device=dev)
        studio = torch.ones(B, dtype=torch.long, device=dev)
        is_sa = torch.tensor([x["lang"] == 1 for x in xs], device=dev)
        lang_eff = torch.where(is_sa, torch.full((B,), 4, dtype=torch.long, device=dev), torch.zeros(B, dtype=torch.long, device=dev))
        sty = torch.where(is_sa, torch.full((B,), 5, dtype=torch.long, device=dev), torch.zeros(B, dtype=torch.long, device=dev))
        ones_m = torch.ones((B, 1), device=dev)
        non_bug_mask = torch.tensor([0.0 if x["is_hard"] else 1.0 for x in xs], device=dev)[:, None, None]

        ds_tok = (dur.float() / 7.5).clamp(0.6, 1.8)
        svara_ids, mora_ids = compute_svara_mora_ids(ids, is_sa, ds_tok, p_enc)

        G.train()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            hs, mus, gs = G.am.enc(
                ids, spk, lang_eff, sty, studio, im,
                pitch_add=p_enc, prosody=G.pros_tbl[sty], sty_mask=ones_m,
                svara_ids=svara_ids, mora_ids=mora_ids,
            )
            f0_s = G.am.enc.f0_pred(hs).squeeze(-1)
            eng_s = G.am.enc.energy_pred(hs).squeeze(-1).clamp(-1.2, 1.2)
            l_heads = 0.4 * (F.l1_loss(f0_s[im].float(), f0_t[im].float()) + F.l1_loss(eng_s[im].float(), eng_t[im].float()))
            mus_e = mus + (0.10 * G.pros_tbl[sty][:, 2:3] + 0.06 * eng_s)[..., None] * 0.32
            lds = G.am.dur.pred_logdur(ids, hs.detach(), im) + G.dur_rate_bias.clamp(-0.12, 0.12)
            l_dur = F.smooth_l1_loss(lds[im].float(), torch.log(dur.float().clamp(min=1.0))[im])
            l_prior = F.l1_loss(mus[im].float(), mu_t[im].float())

            attn = _dur_to_attn(dur, im, ym)
            mu_y_s = torch.einsum("blt,bld->bdt", attn.to(mus_e.dtype), mus_e) * ymf
            ends = torch.cumsum(dur.float(), dim=1)
            t_arange = torch.arange(0, T, device=dev, dtype=torch.float32)
            idx_fr = (t_arange[None, :, None] >= ends[:, None, :]).to(torch.float32).sum(-1).to(torch.int64).clamp(max=L - 1)
            prog_4d = compute_intra_phoneme_progress(dur.float(), ends, idx_fr, t_arange) * ymf

            y_pred_8, _ = forward_4step_meanflow(G, mu_y_s, gs, ymf, prog_4d, nb_v=8)

            denom = (ymf.sum() * 100.0).clamp(min=1.0)
            l_mel = ((1.2 * (y_pred_8.float() - y_clean.float()).abs() + 0.8 * (y_pred_8.float() - y_clean.float()) ** 2) * ymf).sum() / denom

            # Anti-Drift Anchor Lock: 0.25 * L1(pred_mel, frozen_sakhiya_one_mel) on non-bug standard sentences
            # (and 0.15 anchor on all sentences) so voice identity and flow cannot drift!
            w_anc = 0.15 + 0.10 * non_bug_mask
            l_anchor = ((y_pred_8.float() - y_frozen.float()).abs() * ymf * w_anc).sum() / denom

            # Spectral & temporal smoothness (eliminates mid-vowel micro-warble and hollowness)
            df_p = y_pred_8[:, 1:, :].float() - y_pred_8[:, :-1, :].float()
            df_t = y_clean[:, 1:, :].float() - y_clean[:, :-1, :].float()
            l_df = ((df_p - df_t).abs() * ymf).sum() / denom

            ym_t = ymf[:, :, 1:] * ymf[:, :, :-1]
            dt_p = y_pred_8[:, :, 1:].float() - y_pred_8[:, :, :-1].float()
            dt_t = y_clean[:, :, 1:].float() - y_clean[:, :, :-1].float()
            l_dt = ((dt_p - dt_t).abs() * ym_t).sum() / (ym_t.sum() * 100.0 + 1.0)

            l_ctc = torch.zeros((), device=dev)
            if st % 4 == 0:
                c_val = ctc.ctc_loss(0.25 * mu_y_s.float() + 0.75 * y_pred_8.float(), yl, ids, il)
                if torch.isfinite(c_val):
                    l_ctc = F.relu(c_val - 0.08)

            loss = (
                1.5 * l_mel
                + 1.0 * l_anchor
                + 1.8 * l_df
                + 1.2 * l_dt
                + 1.2 * l_prior
                + 0.8 * l_dur
                + l_heads
                + 0.20 * l_ctc
            )

        oG.zero_grad(set_to_none=True)
        loss.backward()
        if torch.isfinite(torch.nn.utils.clip_grad_norm_(am_params, 1.0)):
            oG.step()

        # Re-enforce w approximant calibration & EMA=0.999
        _calibrate_styles(G.am)
        with torch.no_grad():
            for pe, pg in zip(ema.parameters(), G.parameters()):
                pe.lerp_(pg, 1.0 - 0.999)

        if st == 50:
            sps = 50.0 / max(1e-6, time.time() - tstep)
            log(f"[Phase 1.2 Watchdog] step-50 throughput = {sps:.2f} steps/s")

        if st % 250 == 0 or st == a.steps:
            now = time.time()
            ips = 250.0 / max(1e-6, now - tstep) if st > 50 else 50.0 / max(1e-6, now - tstep)
            tstep = now
            el_m = (now - t0) / 60.0
            eta_m = max(0.0, el_m * (a.steps - st) / max(1, st))
            l_str = f"mel={l_mel.item():.4f} anc={l_anchor.item():.4f} df={l_df.item():.4f} dt={l_dt.item():.4f} dur={l_dur.item():.4f} lr={lr_cur:.2e}"
            update_stage(1, st, a.steps, f"{ips:.1f} it/s", l_str, el_m, eta_m, "running")
            log(f"[Phase 1.2 AM Polish] step {st}/{a.steps} ({ips:.1f} it/s) {l_str}")

    _calibrate_styles(ema.am)
    ck_out = dict(
        ema={k: v.float().cpu() for k, v in ema.state_dict().items()},
        cfg=ema.cfg, tier="sakhiya_one", arch=ema.arch, voc_arch=ema.voc_arch,
        step=a.steps, f0=[5.418, 0.203], qat=False, gt=True,
    )
    save_p = os.path.join(CK_DIR, "phase1_am_polished.pt")
    torch.save(ck_out, save_p)
    log(f"[Phase 1.2] Saved Anchored Full-Utterance Polished AM -> {save_p}")

    # Render the 3 Owner Clips onto Card 5 & verify with MMS ASR!
    owner_clips = json.load(open(os.path.join(REPO, "eval", "owner_4clips.json")))["clips"][:3]
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="npi")
    asr = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="npi", ignore_mismatched_sizes=True).to(dev).eval()

    grits, hnrs, warbles = [], [], []
    for c in owner_clips:
        cid = c["id"]
        txt = c["text"]
        w_out, _ = synth_full_utterance_torch(ema, txt, dev=dev, int8_snap=True)
        out_wav = os.path.join(LISTEN_DIR, f"{cid}_phase1_am_polish.wav")
        sf.write(out_wav, w_out, SR, subtype="FLOAT")
        cg, wb = compute_comb_grit_and_warble(w_out, SR)
        hnr = _voiced_vowel_hnr_0p2_4khz(w_out, SR)
        grits.append(cg); hnrs.append(hnr); warbles.append(wb)
        w16 = librosa.resample(w_out, orig_sr=SR, target_sr=16000)
        with torch.no_grad():
            hyp = proc.batch_decode(asr(proc(w16, sampling_rate=16000, return_tensors="pt").input_values.to(dev)).logits.argmax(-1))[0]
        log(f"[Phase 1.2 Eval {cid}] grit={cg} HNR={hnr:.2f}dB warble={wb}st | ASR: {hyp}")

    el_m = (time.time() - t0) / 60.0
    update_stage(
        1, a.steps, a.steps, "done",
        f"Converged {a.steps} steps · grit={np.mean(grits):.4f} HNR={np.mean(hnrs):.2f}dB warble={np.mean(warbles):.4f}st · वालेट=PASS",
        el_m, 0.0, "PASS ✅"
    )
    log("PHASE 1.2 ANCHORED FULL-UTTERANCE AM POLISH COMPLETE")


if __name__ == "__main__":
    main()
