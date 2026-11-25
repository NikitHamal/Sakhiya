"""Sakhiya-One Unified Single-Graph Alias-Free Harmonic + AWP Distilled Model (`scripts/sakhiya_pro/distill/student.py`).

Primary Sweet-Spot Flagship (Micro `t5` and fragmented multi-tier zoo retired):
- `sakhiya_one`:      Flagship Sakhiya-One Studio (8-block Laddered Hadamard/Monarch X-Pred MeanFlow AM + Mid-Layer Engram +
                      208-ch Alias-Free Full-Band Harmonic AWP-iSTFT Vocoder, 4 steps) -> ~7.3 MB INT8 ONNX (in 6.5-8.5 MB sweet spot)
- `sakhiya_one_fast`: Sakhiya-One Fast / Eco Mode (4-block prefix slice of the EXACT SAME `sakhiya_one` checkpoint, 1 step) -> ~5.8 MB INT8 ONNX
"""
import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sakhiya.cfm import SakhiyaAM, STYLE_PROSODY_TARGETS
from sakhiya.sakhiya_one import SakhiyaOneAM, apply_lsq_int8_weight
from sakhiya.v15_pico import SakhiyaPicoAM
from sakhiya.v2_vocoder import ISTFT_N_FFT, ISTFT_HOP, SR, N_FFT, HOP, NMEL as N_MEL
from sakhiya.awp_vocoder import AWPVocoder
from sakhiya.v15_rnd_wavehax import _build_cola_istft_kernel, ste_int8_weight, QATConv1d
from sakhiya.symbols import SYM2ID

NB = N_FFT // 2 + 1
N_SYM = 64
N_LANG = 2

TIERS = {
    "sakhiya_one":      dict(tier="sakhiya_one",      arch="sakhiya_one", voc_arch="awp", d_enc=192, L_enc=6, ff_enc=256, d_vel=256, nb_vel=8,  h_dur=128, voc_ch=208, steps=4, n_engram=352, r_enc=88,   r_voc_k7=None, r_voc_k11=None),
    "sakhiya_one_fast": dict(tier="sakhiya_one_fast", arch="sakhiya_one", voc_arch="awp", d_enc=192, L_enc=6, ff_enc=256, d_vel=256, nb_vel=4,  h_dur=128, voc_ch=208, steps=1, n_engram=352, r_enc=88,   r_voc_k7=None, r_voc_k11=None),
    "t30": dict(tier="t30", arch="sakhiya_am",   voc_arch="awp", d_enc=192, L_enc=6, ff_enc=256, d_vel=256, nb_vel=12, h_dur=128, voc_ch=320, steps=4, r_enc=None, r_voc_k7=None, r_voc_k11=None),
    "t15": dict(tier="t15", arch="sakhiya_am",   voc_arch="awp", d_enc=192, L_enc=6, ff_enc=256, d_vel=256, nb_vel=12, h_dur=128, voc_ch=320, steps=4, r_enc=72,   r_voc_k7=None, r_voc_k11=None),
    "t10": dict(tier="t10", arch="sakhiya_am",   voc_arch="awp", d_enc=144, L_enc=5, ff_enc=224, d_vel=160, nb_vel=7,  h_dur=96,  voc_ch=320, steps=1, r_enc=104,  r_voc_k7=None, r_voc_k11=None),
    "t1":  dict(tier="t1",  arch="sakhiya_pico", voc_arch="awp", d_enc=96,  L_enc=4, ff_enc=144, d_vel=112, nb_vel=5,  h_dur=64,  voc_ch=144, steps=1, r_enc=52,   r_voc_k7=44,   r_voc_k11=27),
}

VOWEL_IDS = tuple(SYM2ID[p] for p in ("ʌ", "a", "i", "u", "e", "o", "ʌ̃", "ã", "ĩ", "ũ", "ẽ", "õ") if p in SYM2ID)
LONG_VOWEL_IDS = tuple(SYM2ID[p] for p in ("a", "e", "o", "ã", "ĩ", "ũ", "ẽ", "õ") if p in SYM2ID)


class AMF0VUVHead(nn.Module):
    """Acoustic-Model Continuous F0 (in Hz) + Soft V/UV Predictor (kept in FP16, lives inside Student.am, ZERO vocoder f0_net, ZERO 9-frame pooling)."""
    def __init__(self, n_mels: int = N_MEL):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_mels, 128, 5, padding=2),
            nn.SiLU(),
            nn.Conv1d(128, 128, 5, padding=2),
            nn.SiLU(),
            nn.Conv1d(128, 2, 3, padding=1),
        )

    def forward(self, mel: torch.Tensor, svara_shift_st: torch.Tensor = None):
        rf = self.net(mel.float()).float()
        f0 = 60.0 + 440.0 * torch.sigmoid(rf[:, 0:1])
        vuv = torch.sigmoid(rf[:, 1:2]).clamp(0.0, 1.0)
        lf0 = torch.log2(f0.clamp(130.0, 380.0))
        if svara_shift_st is not None:
            lf0 = lf0 + svara_shift_st.to(lf0.dtype) * (1.0 / 12.0)
        f0_hz = torch.pow(2.0, lf0).clamp(130.0, 380.0)
        return f0_hz, vuv


def compute_svara_mora_ids(ids: torch.Tensor, is_sa: torch.Tensor, ds_tok: torch.Tensor, ps_tok: torch.Tensor):
    id_o, id_m = SYM2ID["o"], SYM2ID["m"]
    next_ids = torch.cat([ids[:, 1:], torch.zeros_like(ids[:, -1:])], dim=1)
    prev_ids = torch.cat([torch.zeros_like(ids[:, :1]), ids[:, :-1]], dim=1)
    is_sa_b = is_sa[:, None] if is_sa.dim() == 1 else is_sa

    is_vow = torch.zeros_like(ids, dtype=torch.bool)
    for vid in VOWEL_IDS:
        is_vow = is_vow | (ids == vid)
    is_long_v = torch.zeros_like(ids, dtype=torch.bool)
    for vid in LONG_VOWEL_IDS:
        is_long_v = is_long_v | (ids == vid)

    sv = torch.zeros_like(ids, dtype=torch.long)
    sv = torch.where(is_vow & is_sa_b, torch.ones_like(sv), sv)
    sv = torch.where(is_vow & is_sa_b & (ps_tok < -0.10), torch.full_like(sv, 2), sv)
    sv = torch.where(is_vow & is_sa_b & (ps_tok > 0.25), torch.full_like(sv, 3), sv)
    sv = torch.where(is_vow & is_sa_b & (ps_tok >= 0.05) & (ps_tok <= 0.25), torch.full_like(sv, 4), sv)

    is_om = (((ids == id_o) & (next_ids == id_m) & (prev_ids <= 8)) | ((ids == id_m) & (prev_ids == id_o))) & is_sa_b
    mo = torch.zeros_like(ids, dtype=torch.long)
    mo = torch.where(is_long_v | (ds_tok >= 1.02), torch.ones_like(mo), mo)
    mo = torch.where(ds_tok >= 1.22, torch.full_like(mo, 2), mo)
    mo = torch.where(is_om, torch.full_like(mo, 3), mo)
    return sv, mo


def compute_intra_phoneme_progress(dur: torch.Tensor, ends: torch.Tensor, idx: torch.Tensor, t_arange: torch.Tensor):
    starts = ends - dur
    st_f = torch.gather(starts, 1, idx)
    d_f = torch.gather(dur, 1, idx).clamp(min=1.0)
    k_f = (t_arange[None, :] - st_f).clamp(min=0.0)
    u = ((k_f + 0.5) / d_f).clamp(0.0, 1.0)
    p0 = u
    p1 = torch.sin(u * math.pi)
    p2 = torch.cos(u * math.pi)
    p3 = torch.log1p(d_f) * 0.25
    return torch.stack([p0, p1, p2, p3], dim=1)


def _make_svd_seq(conv: nn.Conv1d, rank: int, use_qat: bool = False):
    Co, Ci, K = conv.weight.shape
    r = min(int(rank), Co, Ci * K)
    cls = QATConv1d if use_qat else nn.Conv1d
    c1 = cls(Ci, r, K, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, bias=False)
    c2 = cls(r, Co, 1, bias=(conv.bias is not None))
    with torch.no_grad():
        W = conv.weight.data.float()
        U, S, Vh = torch.linalg.svd(W.reshape(Co, Ci * K), full_matrices=False)
        sq = torch.sqrt(S[:r])
        c1.weight.data.copy_((sq[:, None] * Vh[:r, :]).reshape(r, Ci, K))
        c2.weight.data.copy_((U[:, :r] * sq[None, :]).reshape(Co, r, 1))
        if conv.bias is not None:
            c2.bias.data.copy_(conv.bias.data.float())
    return nn.Sequential(c1, c2)


def _calibrate_styles(am, pre_baked=False):
    with torch.no_grad():
        if hasattr(am.enc, "emb") and am.enc.emb.weight.shape[0] > max(SYM2ID["ɽ"], SYM2ID["õ"]):
            w_e = am.enc.emb.weight
            id_l, id_r = SYM2ID["l"], SYM2ID["ɽ"]
            w_e[id_r].copy_(0.60 * w_e[id_l] + 0.40 * w_e[id_r])
        w_l = am.enc.lang.weight
        if w_l.shape[0] > 4:
            w_l[4].copy_(0.94 * w_l[0] + 0.06 * w_l[4])
        for lid in range(w_l.shape[0]):
            if lid != 4:
                w_l[lid].copy_(w_l[0])


def snap_f0_int8(voc):
    pass


def snap_am_int8(am):
    with torch.no_grad():
        if hasattr(am, "f0_head"):
            for idx in (0, 2):
                w = am.f0_head.net[idx].weight
                w.copy_(ste_int8_weight(w.float(), axis=0, enabled=True))
            if len(am.f0_head.net) > 4:
                am.f0_head.net[4].weight.copy_(am.f0_head.net[4].weight.half().float())
        for name, p in am.named_parameters():
            if name.startswith(("enc.", "dur.")):
                if any(k in name for k in ("emb", "svara_emb", "mora_emb", "engram", "f0_pred", "vuv_pred", "dur.o")):
                    p.copy_(p.half().float())
                elif (p.ndim == 3 and p.shape[1] == 1) or (p.ndim >= 2 and p.shape[0] <= 8) or p.shape == (66, 1, 64) or (p.ndim == 2 and p.shape[0] == 100):
                    if p.numel() >= 256:
                        p.copy_(p.half().float())
                elif p.ndim >= 2 and p.numel() >= 256 and p.shape[0] >= 16:
                    p.copy_(ste_int8_weight(p.float(), axis=0, enabled=True))


class Student(nn.Module):
    def __init__(self, tier="sakhiya_one", arch=None, voc_arch="awp", d_enc=192, L_enc=6, ff_enc=256, d_vel=240, nb_vel=8, h_dur=128, voc_ch=208, steps=4, n_engram=352, r_enc=None, r_voc_k7=None, r_voc_k11=None, **kw):
        super().__init__()
        if tier in TIERS:
            c = TIERS[tier]
            if arch is None:
                arch = c["arch"]
            d_enc, L_enc, ff_enc, d_vel, nb_vel, h_dur, voc_ch, steps = (
                c["d_enc"], c["L_enc"], c["ff_enc"], c["d_vel"], c["nb_vel"], c["h_dur"], c["voc_ch"], c["steps"]
            )
            n_engram = c.get("n_engram", n_engram)
            if r_enc is None:
                r_enc = c.get("r_enc")
            if r_voc_k7 is None:
                r_voc_k7 = c.get("r_voc_k7")
            if r_voc_k11 is None:
                r_voc_k11 = c.get("r_voc_k11")

        self.tier = tier
        self.arch = arch or ("sakhiya_one" if tier.startswith("sakhiya_one") else ("sakhiya_pico" if tier == "t1" else "sakhiya_am"))
        self.voc_arch = "awp"
        self.steps = int(steps)
        self.n_engram = int(n_engram)
        self.r_enc = r_enc
        self.r_voc_k7 = r_voc_k7
        self.r_voc_k11 = r_voc_k11
        self.cfg = dict(
            tier=self.tier, arch=self.arch, voc_arch=self.voc_arch, d_enc=d_enc, L_enc=L_enc, ff_enc=ff_enc,
            d_vel=d_vel, nb_vel=nb_vel, h_dur=h_dur, voc_ch=voc_ch, steps=self.steps, n_engram=self.n_engram,
            r_enc=self.r_enc, r_voc_k7=self.r_voc_k7, r_voc_k11=self.r_voc_k11,
        )
        self._init_raw_submodules(d_enc, L_enc, ff_enc, d_vel, nb_vel, h_dur, voc_ch, self.n_engram)
        self._apply_svd_structure()

        K_istft, _ = _build_cola_istft_kernel(ISTFT_N_FFT, ISTFT_HOP)
        self.register_buffer("K_istft", torch.from_numpy(K_istft), persistent=False)
        self.register_buffer("pros_tbl", torch.tensor(STYLE_PROSODY_TARGETS, dtype=torch.float32), persistent=False)
        self.register_buffer("f0_stats", torch.tensor([5.418, 0.203]))
        self.register_buffer("mel_stats", torch.tensor([-6.1761, 2.4932]))
        self.closure_slope = nn.Parameter(torch.tensor(-0.06))
        self.dur_rate_bias = nn.Parameter(torch.tensor(0.0))
        self._qat_flag = False

    def _init_raw_submodules(self, d_enc, L_enc, ff_enc, d_vel, nb_vel, h_dur, voc_ch, n_engram=352):
        if self.arch == "sakhiya_one":
            self.am = SakhiyaOneAM(d_enc=d_enc, L_enc=L_enc, ff_enc=ff_enc, d_vel=d_vel, nb_vel=nb_vel, h_dur=h_dur, n_engram=n_engram)
        elif self.arch == "sakhiya_pico":
            self.am = SakhiyaPicoAM(d_enc=d_enc, L_enc=L_enc, ff_enc=ff_enc, d_vel=d_vel, nb_vel=nb_vel, h_dur=h_dur)
        else:
            self.am = SakhiyaAM(d_enc=d_enc, L_enc=L_enc, ff_enc=ff_enc, d_vel=d_vel, nb_vel=nb_vel, h_dur=h_dur)
            if hasattr(self.am, "ctc_rec"):
                self.am.ctc_rec = nn.Identity()
            if hasattr(self.am.enc, "spk_adv"):
                self.am.enc.spk_adv = nn.Identity()
        self.am.f0_head = AMF0VUVHead(n_mels=N_MEL)
        self.voc = AWPVocoder(ch=voc_ch)

    def _apply_svd_structure(self):
        if self.arch == "sakhiya_pico":
            self.am.vel.ublock_late = self.am.vel.ublock_early
        if self.r_enc is not None:
            if self.arch == "sakhiya_pico":
                if isinstance(self.am.enc.ublock.c1, nn.Conv1d):
                    self.am.enc.ublock.c1 = _make_svd_seq(self.am.enc.ublock.c1, self.r_enc, use_qat=True)
                    self.am.enc.ublock.c2 = _make_svd_seq(self.am.enc.ublock.c2, self.r_enc, use_qat=True)
            else:
                for lyr in self.am.enc.layers:
                    if isinstance(lyr.c1, nn.Conv1d):
                        lyr.c1 = _make_svd_seq(lyr.c1, self.r_enc, use_qat=False)
                        lyr.c2 = _make_svd_seq(lyr.c2, self.r_enc, use_qat=False)

        for b_idx, r_k in ((1, self.r_voc_k7), (2, self.r_voc_k11)):
            if r_k is not None:
                blk = self.voc.mrfs[0][b_idx]
                for i in range(len(blk.c1)):
                    if isinstance(blk.c1[i], nn.Conv1d):
                        blk.c1[i] = _make_svd_seq(blk.c1[i], r_k, use_qat=False)
                    if isinstance(blk.c2[i], nn.Conv1d):
                        blk.c2[i] = _make_svd_seq(blk.c2[i], r_k, use_qat=False)

    def warm_init(self, ck_dir="/work/tts/ck"):
        """Initialize Student AM and Phase 1 AWPVocoder from pure Pro-v3 checkpoints ONLY (zero V1/v5/v9 weights)."""
        dev = next(self.parameters()).device
        c = self.cfg
        self._init_raw_submodules(c["d_enc"], c["L_enc"], c["ff_enc"], c["d_vel"], c["nb_vel"], c["h_dur"], c["voc_ch"], c.get("n_engram", 352))
        self._apply_svd_structure()
        self.to(dev)

        # 1. Load Sakhiya-One or Pro-v3 teacher checkpoint
        cand_cks = [
            f"/work/sp/distill/runs/{self.tier}/ema_final.pt",
            "/work/sp/distill/runs/sakhiya_one/ema_final.pt",
            "/work/sp/distill/runs/t30/ema_final.pt",
        ]
        run_ck_p = next((p for p in cand_cks if os.path.exists(p)), None)
        if run_ck_p:
            ck = torch.load(run_ck_p, map_location=dev, weights_only=False)
            ema_sd = {k: v.float() for k, v in ck.get("ema", ck).items()}
            own_sd = self.state_dict()
            with torch.no_grad():
                for k, v in own_sd.items():
                    if k.endswith("had_rot.weight"):
                        continue
                    if k in ema_sd:
                        src_t = ema_sd[k]
                        if src_t.shape == v.shape:
                            v.copy_(src_t)
                        elif src_t.ndim == v.ndim:
                            sl = tuple(slice(0, min(a, b)) for a, b in zip(v.shape, src_t.shape))
                            v[sl].copy_(src_t[sl])
                    elif k.startswith("am.f0_head.net."):
                        legacy_k = k.replace("am.f0_head.net.", "voc.src.f0_net.")
                        if legacy_k in ema_sd and ema_sd[legacy_k].shape == v.shape:
                            v.copy_(ema_sd[legacy_k])
            self.load_state_dict(own_sd, strict=False)

        # 2. Load Phase 1 Studio-Clean AWPVocoder checkpoint if available, else Pro-v3 filterbank
        awp_p = os.path.join(ck_dir, "sakhiya_one_voc.pt")
        if os.path.exists(awp_p):
            vck = torch.load(awp_p, map_location=dev, weights_only=False)
            vsd = {k: v.float() for k, v in vck.get("G_ema", vck.get("G", vck)).items()}
            self.voc.load_state_dict(vsd, strict=False)
            if "f0_head" in vck:
                self.am.f0_head.load_state_dict({k: v.float() for k, v in vck["f0_head"].items()}, strict=False)
        elif os.path.exists(os.path.join(ck_dir, "v3_voc320_fused.pt")):
            self.voc.load_pro_v3_filterbank(os.path.join(ck_dir, "v3_voc320_fused.pt"))
            vck = torch.load(os.path.join(ck_dir, "v3_voc320_fused.pt"), map_location=dev, weights_only=False)
            vsd = vck.get("G_ema", vck.get("G", vck))
            with torch.no_grad():
                for k, v in self.am.f0_head.net.state_dict().items():
                    if ("src.f0_net." + k) in vsd:
                        v.copy_(vsd["src.f0_net." + k].float())

        self.to(dev)
        snap_am_int8(self.am)
        return self

    def voc_wav(self, mel, f0_hz=None, vuv=None, shift_st=None):
        """Differentiable waveform synthesis from 100-band log-mel [B, 100, T] + AM continuous F0 & soft VUV."""
        if f0_hz is None or vuv is None:
            f0_hz, vuv = self.am.f0_head(mel, svara_shift_st=shift_st)
        elif shift_st is not None:
            f0_hz = (f0_hz * torch.pow(2.0, shift_st.to(f0_hz.dtype) * (1.0 / 12.0))).clamp(130.0, 380.0)
        y, re_s, im_s, _, _, _ = self.voc.forward_wav(mel, f0_hz, vuv)
        return y, re_s, im_s, f0_hz, vuv

    @torch.no_grad()
    def infer(self, ids, lang, dur_scale, pitch_shift):
        """Studio-clean single-graph ONNX inference with Intra-Phoneme Progress Conditioning,
        Mid-Layer Phoneme Engrams, Vedic Svara & Moraic embeddings, and direct AM F0/VUV -> AWPVocoder synthesis.
        """
        B, L = ids.shape
        im = torch.ones((B, L), dtype=torch.bool, device=ids.device)
        spk0 = torch.zeros((B,), dtype=torch.long, device=ids.device)
        studio = torch.ones((B,), dtype=torch.long, device=ids.device)
        is_sa = (lang == 1)
        lang_eff = torch.where(is_sa, torch.full_like(lang, 4), torch.zeros_like(lang))
        sty = torch.where(is_sa, torch.full_like(lang, 5), torch.zeros_like(lang))
        pros = self.pros_tbl[sty]

        z_tok = torch.zeros((B, L), dtype=torch.float32, device=ids.device)
        ds_tok = dur_scale.reshape(B, -1) + z_tok
        ps_tok = pitch_shift.reshape(B, -1) + z_tok

        svara_ids, mora_ids = compute_svara_mora_ids(ids, is_sa, ds_tok, ps_tok)

        pos = torch.arange(L, device=ids.device, dtype=torch.float32)[None, :]
        L_f = (pos[:, -1:] + 1.0).clamp(min=2.0)
        is_pause = (ids <= 8).float()
        phase = (pos / L_f).clamp(0.0, 1.0)
        tail_closure = self.closure_slope.clamp(-0.30, -0.05) * ((phase - 0.72).clamp(min=0.0) / 0.28) * (1.0 - is_pause)
        p_arch = (0.14 * torch.sin(phase * 3.14159265) + tail_closure) * (1.0 - is_pause) - 0.15 * is_pause
        p_arch = F.avg_pool1d(F.pad(p_arch[:, None, :], (1, 1), mode="replicate"), 3, stride=1)[:, 0]
        p_enc = (p_arch + 0.95 * ps_tok).clamp(-1.10, 1.10)

        ones_m = torch.ones((B, 1), dtype=torch.float32, device=ids.device)
        h, mu, g = self.am.enc(
            ids, spk0, lang_eff, sty, studio, im,
            pitch_add=p_enc, prosody=pros, sty_mask=ones_m,
            svara_ids=svara_ids, mora_ids=mora_ids,
        )
        if hasattr(self.am.enc, "energy_pred"):
            eng_hat = self.am.enc.energy_pred(h).squeeze(-1).clamp(-1.2, 1.2)
            mu = mu + (0.10 * pros[:, 2:3] + 0.06 * eng_hat)[..., None] * 0.32

        x_d = self.am.dur.pred_logdur(ids, h, im) + self.dur_rate_bias.clamp(-0.12, 0.12)
        dur_frames = torch.exp(x_d).clamp(min=1.0, max=60.0)
        rate_mult = torch.exp(-0.04 * pros[:, 0:1]) * (1.0 - is_pause) + torch.exp(0.08 * pros[:, 3:4]) * is_pause
        dur_frames = (dur_frames * rate_mult * ds_tok).clamp(min=1.0, max=60.0)

        is_tail = (((pos >= (L_f - 3.0)) & (ids > 8)) | (ids >= 53)).float()
        min_ph = is_tail * 5.0 + (1.0 - is_tail) * ((ids > 8).float() * 3.0 + (ids <= 8).float() * 1.0)
        dur = torch.round(torch.maximum(dur_frames, min_ph)).clamp(min=1.0, max=60.0)

        id_o, id_m, id_h, id_a, id_wb = SYM2ID["o"], SYM2ID["m"], SYM2ID["h"], SYM2ID["a"], SYM2ID["|"]
        next_ids = torch.cat([ids[:, 1:], torch.zeros_like(ids[:, -1:])], dim=1)
        prev_ids = torch.cat([torch.zeros_like(ids[:, :1]), ids[:, :-1]], dim=1)
        next2_ids = torch.cat([ids[:, 2:], torch.zeros_like(ids[:, -2:])], dim=1)
        prev2_ids = torch.cat([torch.zeros_like(ids[:, :2]), ids[:, :-2]], dim=1)
        is_sa_f = is_sa.float()[:, None]
        is_om_o = ((ids == id_o) & (next_ids == id_m) & (prev_ids <= 8)).float() * is_sa_f
        is_om_m = ((ids == id_m) & (prev_ids == id_o)).float() * is_sa_f
        dur = torch.maximum(dur, is_om_o * 12.0 + is_om_m * 7.0)
        is_final_h = ((ids == id_h) & (next_ids <= 8)).float() * is_sa_f
        dur = torch.where(is_final_h > 0.5, torch.clamp(dur, min=3.0, max=5.0), dur)
        next3_ids = torch.cat([ids[:, 3:], torch.zeros_like(ids[:, -3:])], dim=1)
        prev3_ids = torch.cat([torch.zeros_like(ids[:, :3]), ids[:, :-3]], dim=1)
        is_pre_hiatus_wb = ((ids == id_wb) & (next_ids == id_a) & (next2_ids == id_wb) & (next3_ids == id_a)).float()
        is_hiatus_wb = ((prev2_ids == id_wb) & (prev_ids == id_a) & (ids == id_wb) & (next_ids == id_a)).float()
        is_hiatus_a1 = ((prev_ids == id_wb) & (ids == id_a) & (next_ids == id_wb) & (next2_ids == id_a)).float()
        is_hiatus_a2 = ((prev3_ids == id_wb) & (prev2_ids == id_a) & (prev_ids == id_wb) & (ids == id_a)).float()
        dur = torch.where(is_pre_hiatus_wb > 0.5, torch.full_like(dur, 12.0), dur)
        dur = torch.maximum(dur, is_hiatus_a1 * 19.0 + is_hiatus_a2 * 18.0)
        dur = torch.where(is_hiatus_wb > 0.5, torch.full_like(dur, 7.0), dur)
        mu_next2 = torch.cat([mu[:, 2:, :], mu[:, -2:, :]], dim=1)
        mu_next1 = torch.cat([mu[:, 1:, :], mu[:, -1:, :]], dim=1)
        m_pre = is_pre_hiatus_wb[:, :, None]
        m_a1 = is_hiatus_a1[:, :, None]
        m_wb = is_hiatus_wb[:, :, None]
        mu = mu * (1.0 - m_pre) + mu[:, -1:, :] * m_pre
        mu = mu * (1.0 - m_a1) + mu_next2 * m_a1
        mu = mu * (1.0 - m_wb) + (0.45 * mu_next1 + 0.55 * mu[:, -1:, :]) * m_wb
        id_r, id_schwa = SYM2ID["r"], SYM2ID["ʌ"]
        is_func_c = ((prev_ids <= 6) & (next2_ids <= 6) & ((ids == id_r) | (ids == id_h)) & ((next_ids == id_schwa) | (next_ids == id_o))).float()
        is_func_v = ((prev2_ids <= 6) & (next_ids <= 6) & ((prev_ids == id_r) | (prev_ids == id_h)) & ((ids == id_schwa) | (ids == id_o))).float()
        dur = torch.maximum(dur, is_func_c * 7.0 + is_func_v * 9.0)

        ends = torch.cumsum(dur, 1)
        T = ends[0, -1].to(torch.int64)
        t_arange = torch.arange(0, T, device=ids.device, dtype=torch.float32)
        idx = (t_arange[None, :, None] >= ends[:, None, :]).to(torch.float32).sum(-1).to(torch.int64).clamp(max=L - 1)

        mu_t = mu.transpose(1, 2)
        mu_y = torch.gather(mu_t, 2, idx[:, None, :].expand(-1, mu_t.shape[1], -1))
        prog_4d = compute_intra_phoneme_progress(dur, ends, idx, t_arange)
        m_t = torch.ones((B, 1, T), dtype=mu_y.dtype, device=mu_y.device)

        if self.steps == 1:
            t0 = torch.zeros((B,), device=mu_y.device, dtype=torch.float32)
            xt = mu_y + self.am.vel(mu_y, t0, mu_y, g, m_t, prog=prog_4d)
        else:
            c_idx = torch.arange(100, device=mu_y.device, dtype=torch.float32)[None, :, None]
            t_idx = torch.arange(T, device=mu_y.device, dtype=torch.float32)[None, None, :]
            z_init = torch.sin(c_idx * 12.9898 + t_idx * 78.233) * torch.cos(c_idx * 39.346 + t_idx * 11.135) * (0.667 * 1.4142)
            xt = z_init.expand(B, 100, T)
            dt = 1.0 / self.steps
            for i in range(self.steps):
                ti = torch.full((B,), i * dt, device=mu_y.device, dtype=torch.float32)
                k1 = self.am.vel(xt, ti, mu_y, g, m_t, prog=prog_4d)
                k2 = self.am.vel(xt + 0.5 * dt * k1, ti + 0.5 * dt, mu_y, g, m_t, prog=prog_4d)
                xt = xt + dt * k2

        mel = xt * self.am.stats[1] + self.am.stats[0]
        mel_clean = torch.cat([mel[:, :98, :], mel[:, 97:98, :], mel[:, 97:98, :]], dim=1)
        ps_frame = torch.gather(ps_tok, 1, idx)[:, None, :] * 0.25 * is_sa_f[:, :, None]
        f0_hz, vuv = self.am.f0_head(mel_clean, svara_shift_st=ps_frame)

        y, _, _, _, _ = self.voc_wav(mel_clean, f0_hz=f0_hz, vuv=vuv)
        pk = y.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
        return (y * (0.88 / pk)).clamp(-0.98, 0.98)


def set_qat(model, flag):
    model._qat_flag = bool(flag)
    if hasattr(model.am, "set_qat"):
        model.am.set_qat(bool(flag))
    for name, m in model.named_modules():
        if name.startswith(("voc.src", "voc.amp_head", "voc.pha_head", "am.f0_head")):
            continue
        if hasattr(m, "qat_enabled"):
            m.qat_enabled = bool(flag)


def build(tier):
    return Student(tier=tier)


def n_params(m, deployed=True):
    tot = 0
    for name, p in m.named_parameters():
        if deployed and name.startswith(("am.ctc_rec.", "am.enc.spk_adv.")):
            continue
        tot += p.numel()
    return tot
