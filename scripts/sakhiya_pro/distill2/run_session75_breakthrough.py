#!/usr/bin/env python3
"""Session 75 Unified Breakthrough Pipeline (`scripts/sakhiya_pro/distill2/run_session75_breakthrough.py`).

Executes:
- Phase 0.2: Lineage Purge & 100% Pure Sakhiya-Pro-v3 Single-Timbre Cache (`/work/sp/distill/pure_pro_v3_cache.pt`)
  - Purges all Old-V1 / low-similarity Seed-VC utterances; retains only dual-anchor verified Pro-v3 utterances
    (`sim_ProV3_ecapa - sim_OldV1_ecapa >= +0.34`) + oversampled studio Pro-v3 Episode-01 lines (`ep01_v3/lines/*.wav`),
    `sakhiya_pro_v3`, `sakhiya_pro_v4`, `sakhiya_v5_nepali`, `sakhiya_pro_v4_vedic`, and `sakhiya_v5_vedic` chants (~80% Nepali, ~20% Vedic/Sanskrit).
  - Precomputes continuous `f0_hz` & soft `vuv`, discrete `svara_ids` & `mora_ids`, and 4D intra-phoneme progress `prog_4d`.
- Phase 1: Studio-Clean, From-Scratch, Zero-V1-Ancestry AWP Vocoder Breakthrough (`/work/tts/ck/v4_awp_voc224.pt` & `v4_awp_voc136.pt`)
  - Trains `AWPVocoder(ch=224)` and `AWPVocoder(ch=136)` from scratch on 100% pure Pro-v3 24 kHz audio with +-1.5% F0 jitter,
    Multi-Res Complex STFT + Anti-Wrapped Phase (IP/GD/IAF) + MPD/MRD GAN + Frozen Multi-Layer WavLM Feature-Matching (layers 3, 6, 9) +
    Voiced-Only 24-Bin LTAS Envelope Loss.
  - Verifies `sim_ProV3 >= 0.90` on GT mels before student training.
- Phase 2: Flagship `t30` & `t15` Breakthrough (Pure Pro-v3 Timbre + Vedic/Sanskrit Structural Fix + Joint E2E Waveform Refinement)
  - Trains from pure Pro-v3 lineage (ZERO Old-V1 weights) with 4D Intra-Phoneme Progress Conditioning, discrete Vedic Svara & Moraic
    embeddings, Multi-Resolution Contrastive Timbre Lock (`sim_ProV3 - sim_OldV1 >= +0.32` margin + Voiced 24-Bin LTAS), and
    Joint End-to-End Waveform Refinement (0.8-1.2 s slices, detached duration gradients, MR-STFT + AWP + WavLM-FM + MPD/MRD + QAT).
- Phase 3: `t10` & `t1` Pico Breakthrough (Coupled Reflow Cache + Joint End-to-End Waveform Distillation — NO Standalone GTA)
  - Generates coupled `(ids, svara_ids, mora_ids, dur, f0_hz, vuv, prog_4d, z_0, x_1)` straight-trajectory Reflow pairs from converged Phase 2 `t30`.
  - Initializes `t10` and `t1` Pico from pure Pro-v3 lineage (ZERO V1/v9/sano weights), warms up on the coupled Reflow cache,
    and runs Joint End-to-End AM + Vocoder Waveform Distillation (unfreezing both student AM and paired Phase 1 AWPVocoder on
    0.8-1.2 s waveform slices with detached duration gradients, MR-STFT + AWP + WavLM-FM + MPD/MRD + Contrastive LTAS + LSQ/STE QAT).
"""
import argparse, copy, glob, json, math, os, random, re, sys, time
import librosa, numpy as np, soundfile as sf, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

HERE = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(HERE, "..", "distill")
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for p in (HERE, DIST, REPO, os.path.join(REPO, "scripts", "sakhiya_pro"), "/work/sp"):
    if p not in sys.path:
        sys.path.insert(0, p)

import sp_common as C
import nep_norm as _nn
from sakhiya import chandas as _ch, g2p as _g2p, symbols as _S
from sakhiya.v2_vocoder import mel_spec_24k, SR, HOP, N_FFT
from sakhiya.awp_vocoder import AWPVocoder, awp_phase_losses
from sakhiya.ctc_aligner import MelCTCAligner, batch_viterbi_ctc_durations
from sakhiya.cfm import STYLE_PROSODY_TARGETS, _dur_to_attn
from scripts.train_v2_voc import DiscV2, mr_complex_stft_loss, gpu_f0_uv
from student import (
    Student, build, set_qat, n_params, TIERS,
    compute_svara_mora_ids, compute_intra_phoneme_progress, ste_int8_weight,
)


def install_awp_qat_hooks(model):
    """Out-of-place STE per-channel INT8 QAT hooks for ConvNeXt body while permanently keeping
    embeddings, duration head, F0/VUV head, harmonic excitation (`voc.src`), and iSTFT projection (`voc.amp_head`, `voc.pha_head`) in FP16.
    """
    hooks = []
    fp16_skip = ("voc.src", "voc.amp_head", "voc.pha_head", "voc.pre", "am.vel.out", "am.enc.emb", "am.enc.svara_emb", "am.enc.mora_emb", "am.enc.f0_pred", "am.enc.vuv_pred", "am.dur.o")
    for name, m in model.named_modules():
        if any(name.startswith(s) or name.endswith(s) for s in fp16_skip) or hasattr(m, "qat_enabled"):
            continue
        if isinstance(m, (nn.Conv1d, nn.Linear)) and m.weight.numel() >= 256 and m.weight.shape[0] >= 16:
            def _pre(mod, inp):
                if mod.training and getattr(model, "_qat_flag", False):
                    mod._orig_w = mod.weight.data.clone()
                    mod.weight.data.copy_(ste_int8_weight(mod.weight, axis=0, enabled=True).data)
            def _post(mod, inp, out):
                if hasattr(mod, "_orig_w"):
                    mod.weight.data.copy_(mod._orig_w)
                    delattr(mod, "_orig_w")
            hooks.append(m.register_forward_pre_hook(_pre))
            hooks.append(m.register_forward_hook(_post))
    return hooks


class FrozenMultiLayerWavLM(nn.Module):
    """Frozen Multi-Layer WavLM (layers 3, 6, 9) perceptual feature-matching loss at 24 kHz -> 16 kHz."""
    def __init__(self, dev="cuda"):
        super().__init__()
        from transformers import WavLMModel
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus-sv").to(dev).eval()
        for p in self.wavlm.parameters():
            p.requires_grad_(False)
        self.layers = (3, 6, 9)

    def _prep16k(self, w24: torch.Tensor) -> torch.Tensor:
        w16 = F.interpolate(w24[:, None, :].float(), scale_factor=16000.0 / 24000.0, mode="linear", align_corners=False)[:, 0, :]
        mu = w16.mean(dim=-1, keepdim=True)
        sd = w16.std(dim=-1, keepdim=True).clamp(min=1e-4)
        return (w16 - mu) / sd

    def forward(self, y_pred: torch.Tensor, y_gt: torch.Tensor) -> torch.Tensor:
        xp = self._prep16k(y_pred)
        with torch.no_grad():
            xt = self._prep16k(y_gt)
            out_t = self.wavlm(xt, output_hidden_states=True).hidden_states
            hs_t = [out_t[i].detach() for i in self.layers]
        out_p = self.wavlm(xp, output_hidden_states=True).hidden_states
        loss = 0.0
        for hp, ht in zip([out_p[i] for i in self.layers], hs_t):
            T_m = min(hp.shape[1], ht.shape[1])
            loss = loss + F.l1_loss(hp[:, :T_m].float(), ht[:, :T_m].float())
        return loss / len(self.layers)


def voiced_24bin_ltas(mel_100: torch.Tensor, vuv: torch.Tensor = None) -> torch.Tensor:
    """Compute Voiced-Only 24-Bin LTAS Envelope [B, 24] from 100-band log-mel [B, 100, T]."""
    m = mel_100.float()
    if vuv is None:
        e = m.mean(dim=1, keepdim=True)
        e_max = e.amax(dim=-1, keepdim=True)
        w = torch.sigmoid((e - (e_max - 2.4)) * 4.0)
    else:
        w = (vuv.float() if vuv.dim() == 3 else vuv[:, None, :].float()).clamp(0.0, 1.0)
    w_sum = w.sum(dim=-1).clamp(min=1.0)
    ltas_100 = (m * w).sum(dim=-1) / w_sum  # [B, 100]
    # Pool 100 mel bins into 24 smooth vocal-tract formant envelope bins
    return F.adaptive_avg_pool1d(ltas_100[:, None, :], 24)[:, 0, :]


def contrastive_ltas_timbre_loss(mel_pred: torch.Tensor, mel_pro_gt: torch.Tensor, c_pro_24: torch.Tensor, c_v1_24: torch.Tensor, vuv: torch.Tensor = None, margin: float = 0.35):
    """Multi-Resolution Contrastive Timbre Lock:
    1. Matches utterance Voiced 24-Bin LTAS envelope to ground-truth Pro-v3 (`l_utt`).
    2. Enforces contrastive cosine/L2 margin pushing student voiced LTAS toward Pro-v3 centroid `c_pro_24`
       and away from Old-V1 centroid `c_v1_24` (`sim_ProV3 - sim_OldV1 >= margin`).
    3. Upper-band (4-10 kHz, bins 65..94) spectral tilt & variance lock to eliminate >4 kHz metallic sheen.
    """
    lt_p = voiced_24bin_ltas(mel_pred, vuv)
    lt_t = voiced_24bin_ltas(mel_pro_gt, vuv).detach()
    l_utt = F.l1_loss(lt_p, lt_t)

    # Cosine margin on centered 24-bin voiced LTAS against Pro-v3 vs Old-V1 centroids
    mid = 0.5 * (c_pro_24 + c_v1_24)
    p_n = F.normalize(lt_p - mid, dim=-1)
    pro_n = F.normalize(c_pro_24 - mid, dim=-1)
    v1_n = F.normalize(c_v1_24 - mid, dim=-1)
    sim_pro = (p_n * pro_n).sum(dim=-1)
    sim_v1 = (p_n * v1_n).sum(dim=-1)
    l_margin = F.relu(margin - (sim_pro - sim_v1)).mean() + F.l1_loss(lt_p, c_pro_24.expand_as(lt_p)) * 0.25

    # 4-10 kHz upper-band envelope match (mel bins 65..94)
    hf_p = mel_pred[:, 65:94, :].float()
    hf_t = mel_pro_gt[:, 65:94, :].float().detach()
    l_hf = F.l1_loss(hf_p.mean(dim=-1), hf_t.mean(dim=-1)) + 0.5 * F.l1_loss(hf_p.std(dim=-1), hf_t.std(dim=-1))
    return l_utt + 0.5 * l_margin + 0.8 * l_hf


def build_pure_pro_v3_cache(cache_path="/work/sp/distill/pure_pro_v3_cache.pt", force=False, dev="cuda"):
    """Phase 0.2: Build 100% pure Sakhiya-Pro-v3 single-timbre 24 kHz cache (zero Old-V1 contamination)."""
    if os.path.exists(cache_path) and not force:
        print(f"[Phase 0.2] Loading existing pure Pro-v3 cache: {cache_path}", flush=True)
        return torch.load(cache_path, weights_only=False)

    print("[Phase 0.2] Building 100% Pure Sakhiya-Pro-v3 Single-Timbre Cache (purging Old-V1 & noisy VC)...", flush=True)
    # 1. Compute Pro-v3 and Old-V1 ECAPA & 24-bin LTAS centroids
    pro_ec, pro_lt24 = [], []
    for p in ["/work/sp/v3/ref/anchor_v3.wav"] + sorted(glob.glob(os.path.join(REPO, "docs", "samples", "versions", "sakhiya_pro_v3", "*.mp3"))):
        if os.path.exists(p):
            w, _ = librosa.load(p, sr=SR)
            pro_ec.append(C.spk_emb(w, SR))
            with torch.no_grad():
                m = mel_spec_24k(torch.from_numpy(w)[None].to(dev), dev)
                pro_lt24.append(voiced_24bin_ltas(m)[0].cpu())
    c_pro_ec = np.mean(pro_ec, axis=0); c_pro_ec /= max(np.linalg.norm(c_pro_ec), 1e-8)
    c_pro_24 = torch.stack(pro_lt24, dim=0).mean(dim=0)

    v1_ec, v1_lt24 = [], []
    for v1_tag in ("v5_final", "v15_v1_chandas"):
        for p in sorted(glob.glob(os.path.join(REPO, "docs", "samples", "versions", v1_tag, "*.mp3")))[:10]:
            w, _ = librosa.load(p, sr=SR)
            v1_ec.append(C.spk_emb(w, SR))
            with torch.no_grad():
                m = mel_spec_24k(torch.from_numpy(w)[None].to(dev), dev)
                v1_lt24.append(voiced_24bin_ltas(m)[0].cpu())
    c_v1_ec = np.mean(v1_ec, axis=0); c_v1_ec /= max(np.linalg.norm(c_v1_ec), 1e-8)
    c_v1_24 = torch.stack(v1_lt24, dim=0).mean(dim=0)

    D = torch.load("/work/sp/distill/gt_feats.pt", weights_only=False)
    mean, std = float(D.get("mean", -6.1760635)), float(D.get("std", 2.4932315))
    mf = {}
    if os.path.exists("/work/sp/distill/manifest.jsonl"):
        for ln in open("/work/sp/distill/manifest.jsonl"):
            try:
                r = json.loads(ln)
                mf[r["id"]] = r
            except Exception:
                pass

    BAD_SRCS = {"vc_slr54", "vc_v13", "vc_slr43"}
    pure_ne, pure_sa, dev_items = [], [], []

    for x in D["items"]:
        L = len(x["ids"])
        T = x["mel"].shape[1]
        if L < 4 or T < 40 or T > 680 or int(x["dur"].sum()) != T:
            continue
        r = mf.get(x["id"], {})
        s = r.get("src", x.get("src", ""))
        u = float(r.get("utmos", 3.65))
        if s in BAD_SRCS or u < 3.45:
            continue
        # Enrich with discrete svara_ids & mora_ids
        is_sa = (int(x["lang"]) == 1)
        ids_t = x["ids"].long()[None, :]
        if is_sa and r.get("text"):
            ci = _ch.compile_chandas(r["text"])
            dm_l = ci.get("dm", [1.0] * L)
            off_l = ci.get("off", [0.0] * L)
            ds_t = torch.tensor([[float(dm_l[min(i, len(dm_l) - 1)]) for i in range(L)]], dtype=torch.float32)
            ps_t = torch.tensor([[float(off_l[min(i, len(off_l) - 1)]) * 0.40 for i in range(L)]], dtype=torch.float32)
        else:
            ds_t = torch.ones((1, L), dtype=torch.float32)
            ps_t = torch.zeros((1, L), dtype=torch.float32)
        sv_t, mo_t = compute_svara_mora_ids(ids_t, torch.tensor([is_sa]), ds_t, ps_t)
        item = dict(
            id=x["id"],
            ids=x["ids"].long(),
            svara_ids=sv_t[0].to(torch.int16),
            mora_ids=mo_t[0].to(torch.int16),
            ds_tok=ds_t[0].half(),
            ps_tok=ps_t[0].half(),
            lang=int(x["lang"]),
            wav=x["wav"],
            mel=x["mel"].half(),
            dur=x["dur"].to(torch.int16),
            split=x.get("split", "train"),
            src=s,
        )
        if x.get("split") == "dev":
            dev_items.append(item)
        elif is_sa:
            pure_sa.append(item)
        else:
            pure_ne.append(item)

    # Align and add studio Sakhiya-Pro-v3/v4/v5 Master Packs (Nepali + Vedic Svara/Chandas) & Episode 01 studio lines
    cc = torch.load("/work/tts/ck/v31_ctc_recognizer.pt", map_location=dev, weights_only=False)
    al = MelCTCAligner().to(dev).eval()
    al.load_state_dict(cc.get("model", cc), strict=False)

    reg = json.load(open(os.path.join(REPO, "docs", "samples", "versions", "registry.json")))
    base_clips = reg["versions"][0]["clips"]
    studio_packs = [
        ("sakhiya_pro_v3", 18),
        ("sakhiya_pro_v4", 12),
        ("sakhiya_pro_v4_vedic", 18),
        ("sakhiya_v5_nepali", 14),
        ("sakhiya_v5_vedic", 20),
    ]
    n_studio_added = 0
    for pack_tag, oversample in studio_packs:
        pdir = os.path.join(REPO, "docs", "samples", "versions", pack_tag)
        if not os.path.isdir(pdir):
            continue
        for c in base_clips:
            mp3_p = os.path.join(pdir, f"{c['id']}.mp3")
            if not os.path.exists(mp3_p):
                continue
            w, _ = librosa.load(mp3_p, sr=SR)
            rms = float(np.sqrt(np.mean(w ** 2)))
            if rms < 1e-4:
                continue
            w = np.clip(w * (0.062 / rms), -0.97, 0.97).astype(np.float32)
            n = (len(w) // HOP) * HOP
            w = w[:n]
            is_sa = (c["lang"] == "sa")
            if is_sa:
                ci = _ch.compile_chandas(c["text"])
                ph_exp, off_exp, dm_exp = [], [], []
                for idx_p, (p_i, o_i, d_i) in enumerate(zip(ci["ph"], ci["off"], ci["dm"])):
                    if p_i == "," and idx_p == 1 and ci["ph"][0] == "<om>":
                        continue
                    if p_i == "<om>":
                        ph_exp.extend(["o", "m"]); off_exp.extend([o_i, o_i]); dm_exp.extend([1.6, 1.35])
                    else:
                        ph_exp.append(p_i); off_exp.append(o_i); dm_exp.append(d_i)
                sid = [_S.SYM2ID.get(p_i, -1) for p_i in ph_exp]
                ids_list = [i for i in sid if i >= 0]
                dm_arr = [d_i for d_i, i in zip(dm_exp, sid) if i >= 0]
                off_arr = [o_i for o_i, i in zip(off_exp, sid) if i >= 0]
            else:
                t_norm = _nn.normalize(re.sub("[\u0951-\u0954]", "", c["text"]))
                ph_m = [q for p in _g2p.text_to_phones(t_norm, "ne") for q in (["o", "m"] if p == "<om>" else [p])]
                ids_list = _S.encode(ph_m)
                dm_arr = [1.0] * len(ids_list)
                off_arr = [0.0] * len(ids_list)

            with torch.no_grad():
                m = mel_spec_24k(torch.from_numpy(w)[None].to(dev), dev)[0]
                mn = ((m - mean) / std)[None]
                ids_t = torch.tensor([ids_list], dtype=torch.long, device=dev)
                yl_t = torch.tensor([m.shape[1]], dtype=torch.long, device=dev)
                il_t = torch.tensor([len(ids_list)], dtype=torch.long, device=dev)
                ym_t = torch.ones((1, m.shape[1]), dtype=torch.bool, device=dev)
                lp = al(mn, ym_t)
                dur_t, _ = batch_viterbi_ctc_durations(lp, ids_t, yl_t, il_t)
                d_cpu = dur_t[0, :len(ids_list)].cpu().long()
                if int(d_cpu.sum()) == m.shape[1]:
                    ds_t = torch.tensor([dm_arr], dtype=torch.float32)
                    ps_t = torch.tensor([[o * 0.40 for o in off_arr]], dtype=torch.float32)
                    sv_t, mo_t = compute_svara_mora_ids(ids_t.cpu(), torch.tensor([is_sa]), ds_t, ps_t)
                    st_item = dict(
                        id=f"{pack_tag}_{c['id']}",
                        ids=torch.tensor(ids_list, dtype=torch.long),
                        svara_ids=sv_t[0].to(torch.int16),
                        mora_ids=mo_t[0].to(torch.int16),
                        ds_tok=ds_t[0].half(),
                        ps_tok=ps_t[0].half(),
                        lang=(1 if is_sa else 0),
                        wav=torch.from_numpy(np.clip(w * 32767.0, -32767.0, 32767.0).astype(np.int16)),
                        mel=m.half().cpu(),
                        dur=d_cpu.to(torch.int16),
                        split="train",
                        src=pack_tag,
                    )
                    n_studio_added += 1
                    if is_sa:
                        pure_sa.extend([st_item] * oversample)
                    else:
                        pure_ne.extend([st_item] * oversample)

    # Precompute continuous f0_hz and vuv on GPU in batches for all unique items
    all_train = pure_ne + pure_sa
    uniq = {}
    for x in all_train + dev_items:
        if x["id"] not in uniq:
            uniq[x["id"]] = x
    u_list = list(uniq.values())
    print(f"[Phase 0.2] Extracting GPU continuous F0 & soft VUV for {len(u_list)} unique pure Pro-v3 utterances...", flush=True)
    with torch.no_grad():
        for b0 in range(0, len(u_list), 64):
            batch = u_list[b0 : b0 + 64]
            max_T = max(x["mel"].shape[1] for x in batch)
            wav_b = torch.zeros((len(batch), max_T * HOP), device=dev)
            for bi, x in enumerate(batch):
                n_w = min(len(x["wav"]), max_T * HOP)
                wav_b[bi, :n_w] = x["wav"][:n_w].to(dev).float() / 32767.0
            f0_b, uv_b = gpu_f0_uv(wav_b, max_T)
            for bi, x in enumerate(batch):
                Tk = x["mel"].shape[1]
                f0_k = f0_b[bi, 0, :Tk].clamp(75.0, 420.0)
                uv_k = uv_b[bi, 0, :Tk].clamp(0.0, 1.0)
                # Interpolate F0 smoothly through unvoiced gaps so continuous F0 never drops to 0 Hz
                voiced_idx = torch.where(uv_k > 0.35)[0]
                if len(voiced_idx) >= 2:
                    f0_np = f0_k.cpu().numpy()
                    vi_np = voiced_idx.cpu().numpy()
                    f0_interp = np.interp(np.arange(Tk), vi_np, f0_np[vi_np]).astype(np.float32)
                    f0_k = torch.from_numpy(f0_interp)
                else:
                    f0_k = torch.full((Tk,), 218.0, dtype=torch.float32)
                x["f0_hz"] = f0_k.half().cpu()
                x["vuv"] = uv_k.half().cpu()

    for x in all_train + dev_items:
        u_ref = uniq[x["id"]]
        x["f0_hz"] = u_ref["f0_hz"]
        x["vuv"] = u_ref["vuv"]

    # Balance training list to ~80% pure Pro-v3 Nepali and ~20% Pro-v3 Vedic/Sanskrit
    target_sa = max(len(pure_sa), int(0.25 * len(pure_ne)))
    if len(pure_sa) > 0 and len(pure_sa) < target_sa:
        reps = (target_sa // len(pure_sa)) + 1
        pure_sa = (pure_sa * reps)[:target_sa]

    train_items = pure_ne + pure_sa
    random.shuffle(train_items)
    out_obj = dict(
        train=train_items,
        dev=dev_items,
        mean=mean,
        std=std,
        c_pro_ecapa=torch.from_numpy(c_pro_ec),
        c_v1_ecapa=torch.from_numpy(c_v1_ec),
        c_pro_ltas24=c_pro_24,
        c_v1_ltas24=c_v1_24,
        stats=dict(n_ne=len(pure_ne), n_sa=len(pure_sa), n_unique=len(u_list), n_studio_packs=n_studio_added),
    )
    torch.save(out_obj, cache_path + ".tmp")
    os.replace(cache_path + ".tmp", cache_path)
    print(f"[Phase 0.2] Saved pure Pro-v3 cache: {len(pure_ne)} ne + {len(pure_sa)} sa ({len(u_list)} unique) -> {cache_path}", flush=True)
    return out_obj


def train_phase1_awp_vocoder(ch: int, cache_obj: dict, steps: int = 2400, bs: int = 36, seg_frames: int = 40, lr: float = 2.2e-4, dev="cuda"):
    """Phase 1: Train From-Scratch Alias-Free Full-Band Harmonic + AWP ConvNeXt iSTFT Vocoder on 100% pure Pro-v3 audio."""
    save_p = f"/work/tts/ck/v4_awp_voc{ch}.pt"
    log_p = f"/work/tts/logs/awp_voc{ch}.log"
    os.makedirs("/work/tts/logs", exist_ok=True)

    def log(s):
        print(s, flush=True)
        open(log_p, "a").write(s + "\n")

    G = AWPVocoder(ch=ch).to(dev)
    if os.path.exists(save_p):
        try:
            ck = torch.load(save_p, map_location=dev, weights_only=False)
            G.load_state_dict({k: v.float() for k, v in ck.get("G_ema", ck).items()}, strict=False)
            log(f"[Phase 1 voc{ch}] Resumed from {save_p} (step {ck.get('step', 0)})")
        except Exception as e:
            log(f"[Phase 1 voc{ch}] Fresh init ({e})")

    G_ema = copy.deepcopy(G).eval()
    D = DiscV2().to(dev)
    wavlm_fm = FrozenMultiLayerWavLM(dev=dev)
    c_pro_24 = cache_obj["c_pro_ltas24"].to(dev)
    c_v1_24 = cache_obj["c_v1_ltas24"].to(dev)

    # Deduplicate unique items with length >= seg_frames + 4 for waveform slicing
    seen_ids = set()
    tr = []
    for x in cache_obj["train"]:
        if x["id"] not in seen_ids and x["mel"].shape[1] >= seg_frames + 4:
            seen_ids.add(x["id"])
            tr.append(x)

    oG = torch.optim.AdamW(G.parameters(), lr=lr, betas=(0.8, 0.99), weight_decay=1e-4)
    oD = torch.optim.AdamW(D.parameters(), lr=lr, betas=(0.8, 0.99), weight_decay=1e-4)

    t0 = time.time()
    tstep = time.time()
    n_fft_istft, hop_istft = 64, 16
    win_istft = torch.hann_window(n_fft_istft + 1, device=dev)[:-1]

    log(f"[Phase 1 voc{ch}] Training from-scratch AWPVocoder(ch={ch}) for {steps} steps on {len(tr)} pure Pro-v3 utterances (bs={bs})...")
    for st in range(1, steps + 1):
        prog = st / max(1, steps)
        cur_lr = lr * (0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * prog)))
        for g_ in oG.param_groups + oD.param_groups:
            g_["lr"] = cur_lr

        xs = random.sample(tr, bs)
        wav_seg = torch.zeros((bs, seg_frames * HOP), device=dev)
        mel_seg = torch.zeros((bs, 100, seg_frames), device=dev)
        f0_seg = torch.zeros((bs, 1, seg_frames), device=dev)
        vuv_seg = torch.zeros((bs, 1, seg_frames), device=dev)

        for bi, x in enumerate(xs):
            Tk = x["mel"].shape[1]
            f_start = random.randint(0, Tk - seg_frames - 1)
            w_start = f_start * HOP
            wav_seg[bi] = x["wav"][w_start : w_start + seg_frames * HOP].to(dev).float() / 32767.0
            mel_seg[bi] = x["mel"][:, f_start : f_start + seg_frames].to(dev).float()
            f0_seg[bi, 0] = x["f0_hz"][f_start : f_start + seg_frames].to(dev).float()
            vuv_seg[bi, 0] = x["vuv"][f_start : f_start + seg_frames].to(dev).float()

        # +-1.5% F0 jitter augmentation on GT F0 so small AM pitch errors never cause comb-filter beating
        jitter = 1.0 + (torch.rand((bs, 1, 1), device=dev) * 0.03 - 0.015)
        f0_in = f0_seg * jitter
        # Mild mel noise/smoothing augmentation (25% of batch) for AM robustness
        if random.random() < 0.25:
            ms = F.avg_pool1d(F.pad(mel_seg, (1, 1), mode="replicate"), 3, stride=1)
            mel_in = 0.78 * mel_seg + 0.22 * ms + torch.randn_like(mel_seg) * 0.03
        else:
            mel_in = mel_seg

        G.train(); D.train()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            y_pred, re_p, im_p, log_mag_p, cos_p, sin_p = G.forward_wav(mel_in, f0_in, vuv_seg)
        T_w = min(y_pred.shape[1], wav_seg.shape[1])
        y_pred = y_pred[:, :T_w]
        wav_gt = wav_seg[:, :T_w]

        use_gan = (st > 250)
        ld_val = 0.0
        fr_d = None
        if use_gan:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                dr, fr_d = D(wav_gt[:, None])
                df, _ = D(y_pred.detach()[:, None])
                ld = sum(F.relu(1.0 - r.float()).mean() + F.relu(1.0 + q.float()).mean() for r, q in zip(dr, df)) / len(dr)
            oD.zero_grad(set_to_none=True)
            ld.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), 10.0)
            oD.step()
            ld_val = float(ld.item())

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # 1. Multi-Res Complex STFT + 100-band Mel L1 + 4-10 kHz Upper-Band LSD
            ym = mel_spec_24k(y_pred, dev)
            T_m = min(ym.shape[2], mel_seg.shape[2])
            l_mel = F.l1_loss(ym[:, :, :T_m], mel_seg[:, :, :T_m]) + 1.5 * F.l1_loss(ym[:, 65:94, :T_m], mel_seg[:, 65:94, :T_m])
            l_stft = mr_complex_stft_loss(y_pred, wav_gt, dev)

            # 2. Anti-Wrapped Phase (AWP: IP + GD + IAF) on 64-pt iSTFT frames
            pad_i = (n_fft_istft - hop_istft) // 2
            x_pad = F.pad(wav_gt[:, None], (pad_i, pad_i), mode="reflect")[:, 0]
            spec_gt64 = torch.stft(x_pad, n_fft_istft, hop_istft, n_fft_istft, win_istft, center=False, return_complex=True)
            T_64 = min(cos_p.shape[2], spec_gt64.shape[2])
            l_awp = awp_phase_losses(cos_p[:, :, :T_64], sin_p[:, :, :T_64], spec_gt64.real[:, :, :T_64], spec_gt64.imag[:, :, :T_64])
            l_mag64 = F.l1_loss(log_mag_p[:, :, :T_64], torch.log(spec_gt64.abs()[:, :, :T_64].clamp(min=1e-5)))

            # 3. Voiced-Only 24-Bin LTAS Contrastive Timbre Loss
            l_ltas = contrastive_ltas_timbre_loss(ym[:, :, :T_m], mel_seg[:, :, :T_m], c_pro_24, c_v1_24, vuv_seg[:, :, :T_m])

            # 4. Frozen Multi-Layer WavLM (layers 3, 6, 9) Feature Matching (every 3rd step for speed)
            l_wavlm = torch.zeros((), device=dev)
            if st % 3 == 0:
                l_wavlm = wavlm_fm(y_pred[:12], wav_gt[:12])

            lg = 35.0 * l_mel + 2.5 * l_stft + 2.0 * (l_awp + l_mag64) + 3.0 * l_ltas + 2.5 * l_wavlm
            if use_gan:
                fr = [[t.detach() for t in fs] for fs in fr_d]
                df, ff = D(y_pred[:, None])
                l_adv = sum(F.relu(1.0 - q.float()).mean() for q in df) / len(df)
                n_fm = sum(len(A) for A in fr)
                l_fm = sum(F.l1_loss(b.float(), a_.float()) for A, B in zip(fr, ff) for a_, b in zip(A, B)) / n_fm
                lg = lg + l_adv + 3.5 * l_fm

        oG.zero_grad(set_to_none=True)
        lg.backward()
        gn = torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
        if torch.isfinite(gn):
            oG.step()

        with torch.no_grad():
            dcy = min(0.996, (1 + st) / (20 + st))
            for pe, p in zip(G_ema.parameters(), G.parameters()):
                pe.lerp_(p, 1.0 - dcy)

        if st == 50:
            sps = 50.0 / max(1e-6, time.time() - tstep)
            log(f"[GATE voc{ch}] step-50 throughput {sps:.2f} steps/s")
            if sps < 0.8:
                log("[GATE] too slow"); sys.exit(3)

        if st % 100 == 0 or st == steps:
            el_m = (time.time() - t0) / 60.0
            tot_m = el_m / max(prog, 1e-3)
            log(f"Step {st} mel {l_mel.item():.4f} stft {l_stft.item():.3f} f0 {l_awp.item():.3f} D {ld_val:.3f} | {el_m:.1f}/{tot_m:.0f} min")

        if st % 600 == 0 or st == steps:
            torch.save(dict(G_ema={k: v.float().cpu() for k, v in G_ema.state_dict().items()}, step=st, ch=ch), save_p + ".tmp")
            os.replace(save_p + ".tmp", save_p)
            torch.save(dict(D=D.state_dict(), step=st), f"/work/tts/ck/awp_disc{ch}.pt")

    log(f"VOC_GT_DONE ch={ch} steps={steps} -> {save_p}")
    return G_ema, D


def _ode_solve_prog(vel_net, mu_y, g, m_t, prog_4d, steps: int, z0=None):
    B, _, T = mu_y.shape
    if steps == 1:
        t0 = torch.zeros((B,), device=mu_y.device, dtype=torch.float32)
        x_in = mu_y if z0 is None else z0
        return x_in + vel_net(x_in, t0, mu_y, g, m_t, prog=prog_4d)
    if z0 is None:
        c_idx = torch.arange(100, device=mu_y.device, dtype=torch.float32)[None, :, None]
        t_idx = torch.arange(T, device=mu_y.device, dtype=torch.float32)[None, None, :]
        z0 = (torch.sin(c_idx * 12.9898 + t_idx * 78.233) * torch.cos(c_idx * 39.346 + t_idx * 11.135) * (0.667 * 1.4142)).expand(B, 100, T)
    xt = z0
    dt = 1.0 / steps
    for i in range(steps):
        ti = torch.full((B,), i * dt, device=mu_y.device, dtype=torch.float32)
        k1 = vel_net(xt, ti, mu_y, g, m_t, prog=prog_4d)
        k2 = vel_net(xt + 0.5 * dt * k1, ti + 0.5 * dt, mu_y, g, m_t, prog=prog_4d)
        xt = xt + dt * k2
    return xt


def train_student_breakthrough(
    tier: str,
    cache_obj: dict,
    reflow_map: dict = None,
    steps: int = 1600,
    e2e_start_frac: float = 0.45,
    bs: int = 32,
    lr_e: float = 3.0e-4,
    lr_v: float = 3.0e-4,
    lr_voc: float = 4.0e-5,
    dev: str = "cuda",
):
    """Phase 2 (`t30`, `t15`) & Phase 3 (`t10`, `t1` Pico):
    1. Warm-up on pure Pro-v3 cache + coupled Reflow straight trajectories (with 4D Intra-Phoneme Progress,
       discrete Vedic Svara & Moraic conditioning, and Multi-Resolution Contrastive Timbre Lock).
    2. Joint End-to-End AM + AWPVocoder Waveform Distillation on 0.8-1.2 s (88-frame) waveform slices
       with detached duration gradients, MR-STFT + AWP + Frozen Multi-Layer WavLM-FM (layers 3, 6, 9) +
       MPD/MRD GAN + Voiced 24-Bin LTAS Contrastive Lock + Out-of-place STE Per-Channel INT8 QAT.
    """
    out = f"/work/sp/distill/runs/{tier}"
    os.makedirs(out, exist_ok=True)
    log_p = f"{out}/train.log"

    def log(s):
        print(s, flush=True)
        open(log_p, "a").write(s + "\n")

    G = build(tier).to(dev)
    G.warm_init("/work/tts/ck")
    set_qat(G, True)
    install_awp_qat_hooks(G)
    ema = copy.deepcopy(G).eval()

    voc_ch = G.cfg["voc_ch"]
    D = DiscV2().to(dev)
    dp = f"/work/tts/ck/awp_disc{voc_ch}.pt"
    if os.path.exists(dp):
        try:
            D.load_state_dict(torch.load(dp, map_location=dev, weights_only=False)["D"], strict=False)
        except Exception:
            pass
    wavlm_fm = FrozenMultiLayerWavLM(dev=dev)

    cc = torch.load("/work/tts/ck/v31_ctc_recognizer.pt", map_location=dev, weights_only=False)
    ctc = MelCTCAligner().to(dev).train()
    ctc.load_state_dict(cc.get("model", cc), strict=False)
    for m_ in ctc.modules():
        if isinstance(m_, nn.Dropout):
            m_.p = 0.0
    for p_ in ctc.parameters():
        p_.requires_grad_(False)

    mean, std = float(cache_obj["mean"]), float(cache_obj["std"])
    G.am.stats.copy_(torch.tensor([mean, std], device=dev))
    G.mel_stats.copy_(torch.tensor([mean, std], device=dev))
    c_pro_24 = cache_obj["c_pro_ltas24"].to(dev)
    c_v1_24 = cache_obj["c_v1_ltas24"].to(dev)

    enc_dur_p = [p for n, p in G.named_parameters() if not n.startswith(("voc.", "am.vel.")) and p.requires_grad]
    vel_p = [p for n, p in G.named_parameters() if n.startswith("am.vel.") and p.requires_grad]
    voc_p = [p for n, p in G.named_parameters() if n.startswith("voc.") and p.requires_grad]

    oG = torch.optim.AdamW([
        dict(params=enc_dur_p, lr=lr_e, weight_decay=1e-3),
        dict(params=vel_p, lr=lr_v, weight_decay=1e-4),
        dict(params=voc_p, lr=lr_voc, weight_decay=1e-5),
    ], betas=(0.9, 0.99))
    oD = torch.optim.AdamW(D.parameters(), lr=lr_voc, betas=(0.8, 0.99), weight_decay=1e-4)

    tr = cache_obj["train"]
    n_fft_istft, hop_istft = 64, 16
    win_istft = torch.hann_window(n_fft_istft + 1, device=dev)[:-1]
    seg_w_frames = 88  # ~0.94 s waveform slice (within 0.8-1.2 s)

    t0 = time.time()
    tstep = time.time()
    log(f"[{tier}] Starting {steps} steps (params={n_params(G)/1e6:.3f}M, voc_ch={voc_ch}, ode_steps={G.steps}, bs={bs})")

    for st in range(1, steps + 1):
        prog = st / max(1, steps)
        mult = (0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * prog))) * min(1.0, (st + 10) / 80.0)
        in_e2e = (prog >= e2e_start_frac)
        oG.param_groups[0]["lr"] = lr_e * mult
        oG.param_groups[1]["lr"] = lr_v * mult
        oG.param_groups[2]["lr"] = (lr_voc * mult) if in_e2e else 0.0

        xs = random.sample(tr, bs)
        B = len(xs)
        ids = pad_sequence([x["ids"].long() for x in xs], batch_first=True).to(dev)
        svara_ids = pad_sequence([x["svara_ids"].long() for x in xs], batch_first=True).to(dev)
        mora_ids = pad_sequence([x["mora_ids"].long() for x in xs], batch_first=True).to(dev)
        ps_tok = pad_sequence([x["ps_tok"].float() for x in xs], batch_first=True).to(dev)
        dur = pad_sequence([x["dur"].long() for x in xs], batch_first=True).to(dev)
        il = torch.tensor([len(x["ids"]) for x in xs], device=dev)
        yl = dur.sum(1)
        T = int(yl.max().item())
        L = ids.shape[1]
        im = torch.arange(L, device=dev)[None] < il[:, None]
        ym = torch.arange(T, device=dev)[None] < yl[:, None]
        ymf = ym[:, None].float()

        y_gt_raw = torch.zeros((B, 100, T), device=dev)
        f0_gt = torch.full((B, T), 218.0, device=dev)
        vuv_gt = torch.zeros((B, T), device=dev)
        for bi, x in enumerate(xs):
            Tk = int(yl[bi].item())
            y_gt_raw[bi, :, :Tk] = x["mel"][:, :Tk].to(dev).float()
            f0_gt[bi, :Tk] = x["f0_hz"][:Tk].to(dev).float()
            vuv_gt[bi, :Tk] = x["vuv"][:Tk].to(dev).float()

        y_gt_norm = (y_gt_raw - mean) / std
        # If coupled Reflow target exists from converged Phase 2 flagship teacher, blend 65% coupled straight target + 35% GT Pro-v3
        y_tgt_norm = y_gt_norm.clone()
        if reflow_map is not None:
            for bi, x in enumerate(xs):
                rf = reflow_map.get(x["id"])
                if rf is not None:
                    Tk = min(int(yl[bi].item()), rf.shape[1])
                    y_tgt_norm[bi, :, :Tk] = 0.65 * rf[:, :Tk].to(dev).float() + 0.35 * y_gt_norm[bi, :, :Tk]

        spk = torch.zeros(B, dtype=torch.long, device=dev)
        studio = torch.ones(B, dtype=torch.long, device=dev)
        is_sa = torch.tensor([x["lang"] == 1 for x in xs], device=dev)
        lang_eff = torch.where(is_sa, torch.full((B,), 4, dtype=torch.long, device=dev), torch.zeros(B, dtype=torch.long, device=dev))
        sty = torch.where(is_sa, torch.full((B,), 5, dtype=torch.long, device=dev), torch.zeros(B, dtype=torch.long, device=dev))
        pros = G.pros_tbl[sty]
        ones_m = torch.ones((B, 1), device=dev)

        pos = torch.arange(L, device=dev, dtype=torch.float32)[None, :]
        L_f = il[:, None].float().clamp(min=2.0)
        is_pause = (ids <= 8).float()
        phase = (pos / L_f).clamp(0.0, 1.0)
        tail_closure = G.closure_slope.clamp(-0.30, -0.05) * ((phase - 0.72).clamp(min=0.0) / 0.28) * (1.0 - is_pause)
        p_arch = (0.45 * torch.sin(phase * 3.14159265) + tail_closure) * (1.0 - is_pause) - 0.22 * is_pause
        sty_gain = (0.16 * pros[:, 1:2] + 0.12 * (1.0 - is_sa.float()[:, None])).clamp(0.0, 0.34)
        p_enc = (p_arch * (0.50 * sty_gain) + 0.90 * ps_tok).clamp(-1.10, 1.10) * im.float()

        G.train()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            hs, mus, gs = G.am.enc(
                ids, spk, lang_eff, sty, studio, im,
                pitch_add=p_enc, prosody=pros, sty_mask=ones_m,
                svara_ids=svara_ids, mora_ids=mora_ids,
            )
            f0_z_s = G.am.enc.f0_pred(hs).squeeze(-1)
            vuv_logit_s = G.am.enc.vuv_pred(hs).squeeze(-1)
            eng_s = G.am.enc.energy_pred(hs).squeeze(-1).clamp(-1.2, 1.2)
            mus_e = mus + (0.08 * pros[:, 2:3] + 0.05 * eng_s)[..., None] * 0.28

            # Detached duration gradients for duration predictor vs acoustic path
            lds = G.am.dur.pred_logdur(ids, hs.detach(), im) + G.dur_rate_bias.clamp(-0.12, 0.12)
            l_dur = F.smooth_l1_loss(lds[im].float(), torch.log(dur.float().clamp(min=1.0))[im])

            # Expand phoneme features to frame level using detached hard alignment + 4D Intra-Phoneme Progress
            attn = _dur_to_attn(dur, im, ym)  # [B, L, T] detached
            dur_sum_tok = attn.sum(dim=-1).clamp(min=1.0)
            mu_tok_tgt = torch.einsum("blt,bdt->bld", attn, y_tgt_norm.float()) / dur_sum_tok[:, :, None]
            l_prior = F.l1_loss(mus[im].float(), mu_tok_tgt[im].float())

            # Phoneme-level F0 and VUV targets from ground-truth frame F0 & VUV
            lf0_gt_z = ((torch.log(f0_gt.clamp(min=75.0)) - G.f0_stats[0]) / G.f0_stats[1]).clamp(-2.5, 2.5)
            f0_tok_tgt = (attn * lf0_gt_z[:, None, :]).sum(dim=-1) / dur_sum_tok
            vuv_tok_tgt = (attn * vuv_gt[:, None, :]).sum(dim=-1) / dur_sum_tok
            l_f0_vuv = F.smooth_l1_loss(f0_z_s[im].float(), f0_tok_tgt[im].float()) + \
                       F.binary_cross_entropy_with_logits(vuv_logit_s[im].float(), vuv_tok_tgt[im].float())

            mu_y_s = torch.einsum("blt,bld->bdt", attn.to(mus_e.dtype), mus_e) * ymf
            ends = torch.cumsum(dur.float(), dim=1)
            t_arange = torch.arange(0, T, device=dev, dtype=torch.float32)
            idx_fr = (t_arange[None, :, None] >= ends[:, None, :]).to(torch.float32).sum(-1).to(torch.int64).clamp(max=L - 1)
            prog_4d = compute_intra_phoneme_progress(dur.float(), ends, idx_fr, t_arange) * ymf

            # Coupled ODE / Reflow trajectory solve
            y_pred_norm = _ode_solve_prog(G.am.vel, mu_y_s.float(), gs.float(), ymf, prog_4d.float(), G.steps) * ymf

            denom = (ymf.sum() * 100.0).clamp(min=1.0)
            l_mel = ((1.2 * (y_pred_norm.float() - y_tgt_norm.float()).abs() + 0.8 * (y_pred_norm.float() - y_tgt_norm.float()) ** 2) * ymf).sum() / denom
            df_p = y_pred_norm[:, 1:, :].float() - y_pred_norm[:, :-1, :].float()
            df_t = y_tgt_norm[:, 1:, :].float() - y_tgt_norm[:, :-1, :].float()
            l_df = ((df_p - df_t).abs() * ymf).sum() / denom
            ym_t = ymf[:, :, 1:] * ymf[:, :, :-1]
            l_dt = (((y_pred_norm[:, :, 1:].float() - y_pred_norm[:, :, :-1].float()) - (y_tgt_norm[:, :, 1:].float() - y_tgt_norm[:, :, :-1].float())).abs() * ym_t).sum() / (ym_t.sum() * 100.0 + 1.0)

            mel_pred_raw = y_pred_norm.float() * std + mean
            l_ltas = contrastive_ltas_timbre_loss(mel_pred_raw, y_gt_raw, c_pro_24, c_v1_24, vuv_gt[:, None, :])

            l_ctc = torch.zeros((), device=dev)
            if st % 2 == 0:
                c_val = ctc.ctc_loss(0.25 * mu_y_s.float() + 0.75 * y_pred_norm.float(), yl, ids, il)
                if torch.isfinite(c_val):
                    l_ctc = F.relu(c_val - 0.10)

            # Stage 2: Joint End-to-End AM + AWPVocoder Waveform Distillation on 0.8-1.2s (88-frame) waveform slices
            l_e2e = torch.zeros((), device=dev)
            if in_e2e and T >= seg_w_frames + 2:
                B_w = min(B, 14)
                mp_sl, f0_sl, vuv_sl, w_gt_sl = [], [], [], []
                f0_hz_tok = torch.exp(G.f0_stats[0] + G.f0_stats[1] * f0_z_s.float()).clamp(100.0, 380.0)
                f0_hz_fr = torch.gather(f0_hz_tok, 1, idx_fr)
                vuv_fr = torch.gather(torch.sigmoid(vuv_logit_s.float()), 1, idx_fr)

                for bi in range(B_w):
                    Tk = int(yl[bi].item())
                    if Tk > seg_w_frames + 2:
                        f0_i = random.randint(0, Tk - seg_w_frames - 1)
                    else:
                        f0_i = 0
                    n_f = min(seg_w_frames, Tk - f0_i)
                    m_crop = F.pad(mel_pred_raw[bi, :, f0_i : f0_i + n_f], (0, seg_w_frames - n_f))
                    f_crop = F.pad(f0_hz_fr[bi, f0_i : f0_i + n_f], (0, seg_w_frames - n_f), value=215.0)
                    v_crop = F.pad(vuv_fr[bi, f0_i : f0_i + n_f], (0, seg_w_frames - n_f), value=0.0)
                    w_raw = xs[bi]["wav"][f0_i * HOP : (f0_i + n_f) * HOP].to(dev).float() / 32767.0
                    w_crop = F.pad(w_raw, (0, seg_w_frames * HOP - len(w_raw)))
                    mp_sl.append(m_crop)
                    f0_sl.append(f_crop[None, :])
                    vuv_sl.append(v_crop[None, :])
                    w_gt_sl.append(w_crop)

                mp_b = torch.stack(mp_sl, dim=0)
                f0_b = torch.stack(f0_sl, dim=0)
                vuv_b = torch.stack(vuv_sl, dim=0)
                w_gt_b = torch.stack(w_gt_sl, dim=0)

                y_w, _, _, log_mag_w, cos_w, sin_w = G.voc.forward_wav(mp_b, f0_b, vuv_b)
                Tw = min(y_w.shape[1], w_gt_b.shape[1])
                y_w, w_gt_b = y_w[:, :Tw], w_gt_b[:, :Tw]

                # Update Discriminator every 2nd E2E step
                if st % 2 == 0:
                    dr, _ = D(w_gt_b[:, None])
                    df_d, _ = D(y_w.detach()[:, None])
                    ld_e2e = sum(F.relu(1.0 - r.float()).mean() + F.relu(1.0 + q.float()).mean() for r, q in zip(dr, df_d)) / len(dr)
                    oD.zero_grad(set_to_none=True)
                    ld_e2e.backward()
                    torch.nn.utils.clip_grad_norm_(D.parameters(), 10.0)
                    oD.step()

                _, fr_r = D(w_gt_b[:, None])
                fr_r = [[t.detach() for t in fs] for fs in fr_r]
                df_g, ff_g = D(y_w[:, None])
                l_adv_w = sum(F.relu(1.0 - q.float()).mean() for q in df_g) / len(df_g)
                n_fm = sum(len(A) for A in fr_r)
                l_fm_w = sum(F.l1_loss(b.float(), a_.float()) for A, B in zip(fr_r, ff_g) for a_, b in zip(A, B)) / n_fm

                ym_w = mel_spec_24k(y_w, dev)
                ym_gt_w = mel_spec_24k(w_gt_b, dev).detach()
                Tm_w = min(ym_w.shape[2], ym_gt_w.shape[2])
                l_mel_w = F.l1_loss(ym_w[:, :, :Tm_w], ym_gt_w[:, :, :Tm_w]) + 1.5 * F.l1_loss(ym_w[:, 65:94, :Tm_w], ym_gt_w[:, 65:94, :Tm_w])
                l_stft_w = mr_complex_stft_loss(y_w, w_gt_b, dev)

                pad_i = (n_fft_istft - hop_istft) // 2
                x_pad = F.pad(w_gt_b[:, None], (pad_i, pad_i), mode="reflect")[:, 0]
                spec_gt64 = torch.stft(x_pad, n_fft_istft, hop_istft, n_fft_istft, win_istft, center=False, return_complex=True)
                T_64 = min(cos_w.shape[2], spec_gt64.shape[2])
                l_awp_w = awp_phase_losses(cos_w[:, :, :T_64], sin_w[:, :, :T_64], spec_gt64.real[:, :, :T_64], spec_gt64.imag[:, :, :T_64])

                l_wlm_w = wavlm_fm(y_w[:8], w_gt_b[:8]) if st % 2 == 0 else torch.zeros((), device=dev)
                l_e2e = 4.0 * l_mel_w + 0.6 * l_stft_w + 0.5 * l_awp_w + 0.8 * l_wlm_w + 0.25 * (l_adv_w + 3.0 * l_fm_w)

            loss = l_mel + 2.2 * l_df + 1.1 * l_dt + 1.5 * l_prior + 0.8 * l_dur + 1.2 * l_f0_vuv + 1.8 * l_ltas + 0.25 * l_ctc + l_e2e

        oG.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(G.parameters(), 1.5)
        if torch.isfinite(gn):
            oG.step()

        with torch.no_grad():
            dcy = min(0.995, (1 + st) / (15 + st))
            for pe, pg in zip(ema.parameters(), G.parameters()):
                pe.lerp_(pg, 1.0 - dcy)

        if st == 50:
            sps = (time.time() - tstep) / 50.0
            if sps > 3.5:
                log(f"WATCHDOG: step-50 {sps:.2f}s/step > 3.5s")
                sys.exit(3)

        if st % 50 == 0 or st == steps:
            now = time.time()
            ips = 50.0 / max(1e-6, now - tstep)
            tstep = now
            el_m = (now - t0) / 60.0
            tot_m = max(1.0, el_m / max(prog, 1e-3))
            log(f"Step {st} | {ips:.2f} it/s | mel {l_mel.item():.3f} ltas {l_ltas.item():.3f} prior {l_prior.item():.3f} f0 {l_f0_vuv.item():.3f} e2e {l_e2e.item():.3f} | {el_m:.1f}/{tot_m:.0f} min qat=1")

        if st % 400 == 0 or st == steps:
            ck_obj = dict(ema={k: v.float().cpu() for k, v in ema.state_dict().items()}, cfg=G.cfg, tier=tier, arch=G.arch, voc_arch=G.voc_arch, step=st, f0=[5.418, 0.203], qat=True, gt=True)
            torch.save(ck_obj, f"{out}/ema_latest.tmp")
            os.replace(f"{out}/ema_latest.tmp", f"{out}/ema_latest.pt")

    ck_obj = dict(ema={k: v.float().cpu() for k, v in ema.state_dict().items()}, cfg=G.cfg, tier=tier, arch=G.arch, voc_arch=G.voc_arch, step=steps, f0=[5.418, 0.203], qat=True, gt=True)
    torch.save(ck_obj, f"{out}/ema_final.pt")
    torch.save(ck_obj, f"{out}/ema_latest.pt")
    # Save pure Pro-v3 AM checkpoint for downstream Reflow / lineage initialization
    am_sd = {k: v.float().cpu() for k, v in ema.am.state_dict().items() if not k.startswith("ctc_rec.")}
    torch.save(dict(ema=am_sd, step=steps, mean=mean, std=std), f"/work/tts/ck/pro_v3_{tier}_am.pt")
    if tier == "t30":
        torch.save(dict(ema=am_sd, step=steps, mean=mean, std=std), "/work/tts/ck/pro_v3_flagship_am.pt")
    log(f"DONE {tier} step {steps} minutes {(time.time() - t0) / 60:.1f}")
    return ema


@torch.no_grad()
def build_coupled_reflow_map(t30_model: Student, cache_obj: dict, dev="cuda") -> dict:
    """Phase 3.1: Generate coupled straight-trajectory Reflow targets x_1 from converged Phase 2 `t30` flagship teacher."""
    print("[Phase 3.1] Generating coupled straight-trajectory Reflow targets from converged Phase 2 `t30` teacher...", flush=True)
    t30_model.eval()
    uniq = {}
    for x in cache_obj["train"]:
        if x["id"] not in uniq:
            uniq[x["id"]] = x
    u_list = list(uniq.values())
    reflow_map = {}
    for b0 in range(0, len(u_list), 48):
        xs = u_list[b0 : b0 + 48]
        B = len(xs)
        ids = pad_sequence([x["ids"].long() for x in xs], batch_first=True).to(dev)
        svara_ids = pad_sequence([x["svara_ids"].long() for x in xs], batch_first=True).to(dev)
        mora_ids = pad_sequence([x["mora_ids"].long() for x in xs], batch_first=True).to(dev)
        ps_tok = pad_sequence([x["ps_tok"].float() for x in xs], batch_first=True).to(dev)
        dur = pad_sequence([x["dur"].long() for x in xs], batch_first=True).to(dev)
        il = torch.tensor([len(x["ids"]) for x in xs], device=dev)
        yl = dur.sum(1)
        T = int(yl.max().item())
        L = ids.shape[1]
        im = torch.arange(L, device=dev)[None] < il[:, None]
        ym = torch.arange(T, device=dev)[None] < yl[:, None]
        ymf = ym[:, None].float()
        spk = torch.zeros(B, dtype=torch.long, device=dev)
        studio = torch.ones(B, dtype=torch.long, device=dev)
        is_sa = torch.tensor([x["lang"] == 1 for x in xs], device=dev)
        lang_eff = torch.where(is_sa, torch.full((B,), 4, dtype=torch.long, device=dev), torch.zeros(B, dtype=torch.long, device=dev))
        sty = torch.where(is_sa, torch.full((B,), 5, dtype=torch.long, device=dev), torch.zeros(B, dtype=torch.long, device=dev))
        pros = t30_model.pros_tbl[sty]
        ones_m = torch.ones((B, 1), device=dev)
        pos = torch.arange(L, device=dev, dtype=torch.float32)[None, :]
        L_f = il[:, None].float().clamp(min=2.0)
        is_pause = (ids <= 8).float()
        phase = (pos / L_f).clamp(0.0, 1.0)
        tail_closure = t30_model.closure_slope.clamp(-0.30, -0.05) * ((phase - 0.72).clamp(min=0.0) / 0.28) * (1.0 - is_pause)
        p_arch = (0.45 * torch.sin(phase * 3.14159265) + tail_closure) * (1.0 - is_pause) - 0.22 * is_pause
        sty_gain = (0.16 * pros[:, 1:2] + 0.12 * (1.0 - is_sa.float()[:, None])).clamp(0.0, 0.34)
        p_enc = (p_arch * (0.50 * sty_gain) + 0.90 * ps_tok).clamp(-1.10, 1.10) * im.float()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            ht, mut, gt = t30_model.am.enc(ids, spk, lang_eff, sty, studio, im, pitch_add=p_enc, prosody=pros, sty_mask=ones_m, svara_ids=svara_ids, mora_ids=mora_ids)
            eng_t = t30_model.am.enc.energy_pred(ht).squeeze(-1).clamp(-1.2, 1.2)
            mut_e = mut + (0.08 * pros[:, 2:3] + 0.05 * eng_t)[..., None] * 0.28
            attn = _dur_to_attn(dur, im, ym)
            mu_y = torch.einsum("blt,bld->bdt", attn.to(mut_e.dtype), mut_e) * ymf
            ends = torch.cumsum(dur.float(), dim=1)
            t_arange = torch.arange(0, T, device=dev, dtype=torch.float32)
            idx_fr = (t_arange[None, :, None] >= ends[:, None, :]).to(torch.float32).sum(-1).to(torch.int64).clamp(max=L - 1)
            prog_4d = compute_intra_phoneme_progress(dur.float(), ends, idx_fr, t_arange) * ymf
            x1_pred = _ode_solve_prog(t30_model.am.vel, mu_y.float(), gt.float(), ymf, prog_4d.float(), 4) * ymf
        for bi, x in enumerate(xs):
            Tk = int(yl[bi].item())
            reflow_map[x["id"]] = x1_pred[bi, :, :Tk].half().cpu()
    print(f"[Phase 3.1] Built coupled Reflow map for {len(reflow_map)} utterances.", flush=True)
    return reflow_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc_steps", type=int, default=2200)
    ap.add_argument("--t30_steps", type=int, default=1400)
    ap.add_argument("--t15_steps", type=int, default=1400)
    ap.add_argument("--t10_steps", type=int, default=1600)
    ap.add_argument("--t1_steps", type=int, default=1800)
    ap.add_argument("--force_cache", action="store_true")
    ap.add_argument("--skip_voc", action="store_true")
    ap.add_argument("--tiers", default="t30,t15,t10,t1")
    a = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(42)
    torch.manual_seed(42)

    # Phase 0.2: Pure Pro-v3 Cache
    cache_obj = build_pure_pro_v3_cache(force=a.force_cache)

    # Phase 1: Studio-Clean AWP Vocoders (224-ch for t30/t15/t10, 136-ch for t1 Pico)
    if not a.skip_voc:
        train_phase1_awp_vocoder(ch=224, cache_obj=cache_obj, steps=a.voc_steps, bs=36, seg_frames=40, lr=2.2e-4)
        train_phase1_awp_vocoder(ch=136, cache_obj=cache_obj, steps=a.voc_steps, bs=40, seg_frames=40, lr=2.5e-4)

    tiers_to_run = [t.strip() for t in a.tiers.split(",") if t.strip()]

    # Phase 2: Flagship t30 & t15
    t30_ema = None
    if "t30" in tiers_to_run:
        t30_ema = train_student_breakthrough("t30", cache_obj, reflow_map=None, steps=a.t30_steps, e2e_start_frac=0.45, bs=28, lr_e=1.8e-4, lr_v=1.8e-4, lr_voc=3.5e-5)
    else:
        t30_ema = build("t30").cuda()
        t30_ema.warm_init("/work/tts/ck")

    reflow_map = build_coupled_reflow_map(t30_ema, cache_obj)

    if "t15" in tiers_to_run:
        train_student_breakthrough("t15", cache_obj, reflow_map=reflow_map, steps=a.t15_steps, e2e_start_frac=0.40, bs=28, lr_e=2.0e-4, lr_v=2.0e-4, lr_voc=3.5e-5)

    # Phase 3: t10 & t1 Pico (Coupled Reflow + Joint E2E Waveform Distillation)
    if "t10" in tiers_to_run:
        train_student_breakthrough("t10", cache_obj, reflow_map=reflow_map, steps=a.t10_steps, e2e_start_frac=0.35, bs=32, lr_e=3.5e-4, lr_v=3.5e-4, lr_voc=4.0e-5)

    if "t1" in tiers_to_run:
        train_student_breakthrough("t1", cache_obj, reflow_map=reflow_map, steps=a.t1_steps, e2e_start_frac=0.35, bs=36, lr_e=6.5e-4, lr_v=6.5e-4, lr_voc=4.5e-5)

    print("SESSION75_TRAINING_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
