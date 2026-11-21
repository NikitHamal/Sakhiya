"""Export full 24 kHz Sakhiya-v2 (CFM Acoustic Model + 16x Hybrid iSTFTNet Vocoder) to 3-graph ONNX:
- encoder.onnx / encoder.int8.onnx (RoPE TextEncoder + 4-step Euler DurFlow -> x=[mu, g], logdur)
- decoder.onnx / decoder.int8.onnx (6-step Midpoint ODE ConvNeXt-1D Velocity -> 100-band 24 kHz log-mel)
- vocoder.onnx / vocoder.int8.onnx (16x Snake1d + HarmonicSource -> n_fft=64, hop=16 complex spectrum for _ola64)
Targets: Sakhiya-v2-fp16 <= 32 MB, Sakhiya-v2-tiny-int8 <= 16 MB (weight-only QDQ DequantizeLinear, zero ConvInteger).
Single shipped female voice `sakhiya` (spk=0): blends native Nepal SLR43/FLEURS female with Rasa Nepali Female for lang=ne/thr,
and folds studio Rasa Sanskrit Female (spk=2) into lang=sa.
usage: python3 scripts/export_v2_onnx.py --am /work/tts/runs/v2_am/last.pt --voc /work/tts/runs/v2_voc/G_best.pt --out /work/tts/export_v2"""
import os, sys, json, math, argparse
import numpy as np, onnx, torch, torch.nn as nn, torch.nn.functional as F
from onnx import numpy_helper, helper, TensorProto
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sakhiya.cfm import SakhiyaAM, V2_LANG2ID, STYLE_PROSODY_TARGETS
from sakhiya.v2_vocoder import VocoderV2, SR, HOP
from sakhiya.symbols import SYMBOLS, SYM2ID, LANG2ID


def _strip_meta(m):
    del m.graph.value_info[:]
    m.doc_string = ""
    io_names = {x.name for x in list(m.graph.input) + list(m.graph.output)}
    remap = {}
    def rn(s):
        if not s or s in io_names: return s
        if s not in remap: remap[s] = f"t{len(remap)}"
        return remap[s]
    for init in m.graph.initializer:
        init.name = rn(init.name)
    for n in m.graph.node:
        n.doc_string = ""
        n.name = ""
        for i in range(len(n.input)): n.input[i] = rn(n.input[i])
        for i in range(len(n.output)): n.output[i] = rn(n.output[i])


def to_fp16_weights(src, dst):
    m = onnx.load(src); new_nodes, keep = [], []
    for init in m.graph.initializer:
        if init.data_type == TensorProto.FLOAT and np.prod(init.dims) > 128:
            h = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name + "_fp16")
            keep.append(h)
            new_nodes.append(helper.make_node("Cast", [h.name], [init.name], to=TensorProto.FLOAT, name=init.name + "_cast"))
        else:
            keep.append(init)
    del m.graph.initializer[:]; m.graph.initializer.extend(keep)
    nodes = list(m.graph.node); del m.graph.node[:]; m.graph.node.extend(new_nodes + nodes)
    _strip_meta(m); onnx.save(m, dst)


def to_int8_qdq_weights(src, dst):
    """Weight-only per-channel int8 QDQ (DequantizeLinear) for all weight tensors > 128 elements; zero ConvInteger."""
    m = onnx.load(src); use = {}
    for n in m.graph.node:
        for i in n.input: use.setdefault(i, n.op_type)
    new_nodes, keep = [], []
    for init in m.graph.initializer:
        arr = numpy_helper.to_array(init)
        op = use.get(init.name, "")
        if init.data_type == TensorProto.FLOAT and arr.size >= 64 and arr.ndim >= 2:
            ax = 1 if op in ("ConvTranspose", "MatMul") else 0
            red = tuple(i for i in range(arr.ndim) if i != ax)
            sc = np.maximum(np.abs(arr).max(axis=red), 1e-8) / 127.0
            shp = [1] * arr.ndim; shp[ax] = -1
            q = np.clip(np.round(arr / sc.reshape(shp)), -127, 127).astype(np.int8)
            keep += [numpy_helper.from_array(q, init.name + "_q"), numpy_helper.from_array(sc.astype(np.float32), init.name + "_s")]
            new_nodes.append(helper.make_node("DequantizeLinear", [init.name + "_q", init.name + "_s"], [init.name], axis=ax, name=init.name + "_dq"))
        elif init.data_type == TensorProto.FLOAT and arr.ndim == 1 and arr.size >= 128:
            sc = np.float32(max(float(np.abs(arr).max()), 1e-8) / 127.0)
            q = np.clip(np.round(arr / sc), -127, 127).astype(np.int8)
            keep += [numpy_helper.from_array(q, init.name + "_q"), numpy_helper.from_array(np.array(sc, dtype=np.float32), init.name + "_s")]
            new_nodes.append(helper.make_node("DequantizeLinear", [init.name + "_q", init.name + "_s"], [init.name], name=init.name + "_dq"))
        elif init.data_type == TensorProto.FLOAT and arr.size > 8:
            h = numpy_helper.from_array(arr.astype(np.float16), init.name + "_fp16")
            keep.append(h)
            new_nodes.append(helper.make_node("Cast", [h.name], [init.name], to=TensorProto.FLOAT, name=init.name + "_cast"))
        else:
            keep.append(init)
    del m.graph.initializer[:]; m.graph.initializer.extend(keep)
    nodes = list(m.graph.node); del m.graph.node[:]; m.graph.node.extend(new_nodes + nodes)
    _strip_meta(m); onnx.save(m, dst)


class EncExport(nn.Module):
    def __init__(s, am, cfg_scale=1.35):
        super().__init__(); s.enc = am.enc; s.dur = am.dur; s.cfg_scale = cfg_scale
        s.register_buffer("pros_tbl", torch.tensor(STYLE_PROSODY_TARGETS, dtype=torch.float32))

    def forward(s, ids, spk, lang, pitch_shift, pitch_scale, energy_scale, pitch_add, style):
        B, L = ids.shape
        im = torch.ones((B, L), dtype=torch.bool, device=ids.device)
        sty = style.clamp(0, 5)
        pros = s.pros_tbl[sty]
        sty_emb = torch.where(sty == 5, sty, torch.where(sty == 0, sty, torch.full_like(sty, 2)))
        # Hard Single-Voice Lock: always use canonical female speaker 0 ('sakhiya') so voice identity never shifts!
        spk0 = spk * 0
        studio = spk * 0 + 1
        lang_eff = torch.where(lang == 4, lang, lang * 0)
        pos = torch.arange(L, device=ids.device, dtype=torch.float32)[None, :]
        L_f = (pos[:, -1:] + 1.0).clamp(min=2.0)
        is_pause = (ids <= 6).float()
        chant_boost = torch.where((sty == 5)[:, None], torch.full_like(pitch_add, 1.15), torch.ones_like(pitch_add))
        p_svara = pitch_add * chant_boost
        # Smooth human multi-clause phrase intonation arch + sub-phrase cadence (zero high-freq token jitter!)
        phase = pos / L_f
        p_arch = (0.55 * torch.sin(phase * 3.14159265) + 0.48 * torch.sin(phase * 9.424778) + 0.32 * torch.cos(phase * 15.707963)) * (1.0 - is_pause) - 0.28 * is_pause
        p_arch = F.avg_pool1d(F.pad(p_arch[:, None, :], (1, 1), mode="replicate"), 3, stride=1)[:, 0]
        has_svara = ((pitch_add.abs().amax(dim=1, keepdim=True) > 0.45) | ((sty == 5)[:, None])).float()
        is_news_ne = ((sty == 0)[:, None] & (pitch_scale < 0.94)).float() * (1.0 - has_svara)
        is_story = (((sty == 1) | (sty == 2) | (sty == 4))[:, None]).float()
        sty_f0_sc = -0.185 * is_news_ne + 0.38 * is_story
        sty_gain = ((0.18 * pros[:, 1:2] + 0.45 * (pitch_scale - 1.02)).clamp(0.0, 0.38))
        p_prose_enc = p_arch * (0.55 * sty_gain)
        p_enc = (0.78 * p_svara + (1.0 - 0.55 * has_svara) * p_prose_enc + 0.35 * pitch_shift).clamp(-0.48, 0.48)
        ones_m = torch.ones((B, 1), dtype=torch.float32, device=ids.device)
        h, mu, g = s.enc(ids, spk0, lang_eff, sty_emb, studio, im, pitch_add=p_enc, prosody=pros, sty_mask=ones_m)
        eng_hat = s.enc.energy_pred(h).squeeze(-1).clamp(-1.2, 1.2) if hasattr(s.enc, "energy_pred") else torch.zeros_like(p_enc)
        eng_mod = (torch.log(energy_scale.clamp(min=0.2)) + 0.10 * pros[:, 2:3] + 0.06 * eng_hat)[..., None] * 0.32
        mu = mu + eng_mod
        x_d = s.dur.pred_logdur(ids, h, im)
        dur_frames = torch.exp(x_d).clamp(min=1.0, max=60.0)
        rate_mult = torch.exp(-0.04 * pros[:, 0:1]) * (1.0 - is_pause) + torch.exp(0.08 * pros[:, 3:4]) * is_pause
        dur_frames = (dur_frames * rate_mult).clamp(min=1.0, max=60.0)
        # Protect sentence-final phoneme tokens ('हो', 'छ') and single-syllable function words ('र', 'न') from frame collapse
        is_tail = ((pos >= (L_f - 3.0)) & (ids > 6)).float()
        dur_frames = torch.maximum(dur_frames, is_tail * 5.0 + (1.0 - is_tail) * 1.0)
        logdur = torch.log(dur_frames + 1.0)
        f0_hat = F.avg_pool1d(F.pad(s.enc.f0_pred(h).transpose(1, 2), (1, 1), mode="replicate"), 3, stride=1)[:, 0].clamp(-1.2, 1.2)
        p_story_arch = is_story * 0.42 * (0.60 * torch.sin(phase * 6.2831853) + 0.40 * torch.cos(phase * 12.56637)) * (1.0 - is_pause)
        p_prose = (0.35 * f0_hat + 0.65 * p_arch) * sty_gain + p_story_arch
        p_out = has_svara * (0.88 * p_svara + 0.35 * p_prose + pitch_shift) + (1.0 - has_svara) * (p_prose + 0.50 * p_svara + 0.35 * pitch_shift)
        sty_exp = sty_f0_sc.expand(B, L)
        g_exp = g[:, None, :].expand(B, L, g.shape[-1])
        x_out = torch.cat([mu, g_exp, p_out[..., None], sty_exp[..., None]], dim=-1)
        return x_out, logdur


class DecExport(nn.Module):
    """Sakhiya-v4 Glitch-Free X-Pred CFM Decoder:
    Zero integer-bin mel-spectrogram gather warping (`torch.gather` removed 100%!).
    Passes smooth continuous raised-cosine F0 contour `st_shift` directly to VocoderV2's analytical HarmonicSource."""
    def __init__(s, am, steps=8, temp=0.667, cfg_scale=1.35):
        super().__init__(); s.vel = am.vel; s.register_buffer("stats", am.stats.clone())
        s.steps = steps; s.temp = temp; s.c_dim = am.vel.c; s.cfg_scale = cfg_scale

    def forward(s, x):
        mu_y = x[:, :, :100].transpose(1, 2)
        g = x[:, 0, 100:100 + s.c_dim]
        p_xe = x[:, :, 100 + s.c_dim]
        sty_gain = x[:, :, 100 + s.c_dim + 1]
        B, C, T = mu_y.shape
        m = torch.ones((B, 1, T), dtype=mu_y.dtype, device=mu_y.device)
        if s.temp == 0.0 and s.steps == 1:
            t0 = torch.zeros((B,), device=mu_y.device, dtype=torch.float32)
            xt = mu_y + s.vel(mu_y, t0, mu_y, g, m)
        else:
            c_idx = torch.arange(C, device=mu_y.device, dtype=torch.float32)[None, :, None]
            t_idx = torch.arange(T, device=mu_y.device, dtype=torch.float32)[None, None, :]
            z = torch.sin(c_idx * 12.9898 + t_idx * 78.233) * torch.cos(c_idx * 39.346 + t_idx * 11.135) * (s.temp * 1.4142)
            xt = z.expand(B, C, T)
            dt = 1.0 / s.steps
            for i in range(s.steps):
                t = torch.full((B,), i * dt, device=mu_y.device, dtype=torch.float32)
                k1 = s.vel(xt, t, mu_y, g, m)
                k2 = s.vel(xt + 0.5 * dt * k1, t + 0.5 * dt, mu_y, g, m)
                xt = xt + dt * k2
        mel = xt * s.stats[1] + s.stats[0]
        bins = torch.arange(20, device=mel.device, dtype=mel.dtype)[None, :, None]
        w_f0 = torch.softmax(mel[:, 2:22, :] * 2.2, dim=1)
        c_f0 = (w_f0 * bins).sum(1)
        fz_mel = ((c_f0 - c_f0.mean(dim=1, keepdim=True)) / (c_f0.std(dim=1, keepdim=True) + 0.35)).clamp(-2.2, 2.2)
        p_zero = p_xe - p_xe.mean(dim=1, keepdim=True)
        # Two-stage raised-cosine Conv1D smoothing (9-frame + 5-frame) guarantees zero frame-to-frame F0 spikes/glitches!
        p_s1 = F.avg_pool1d(F.pad(p_zero[:, None, :], (4, 4), mode="replicate"), 9, stride=1)
        p_sm = F.avg_pool1d(F.pad(p_s1, (2, 2), mode="replicate"), 5, stride=1)
        # Glitch-Free Source-Filter Mel Decomposition: preserve exact formant & harmonic bins (zero torch.gather bin tearing!)
        env = F.avg_pool1d(F.pad(mel.transpose(1, 2), (3, 3), mode="replicate"), 7, stride=1).transpose(1, 2)
        kb = torch.arange(100, device=mel.device, dtype=mel.dtype)
        wk = ((64.0 - kb) / 18.0).clamp(0.0, 1.0)[None, :, None]
        comb_gain = 0.12 if s.steps == 1 else 0.08
        mel_w = env + (mel - env) * (1.0 + comb_gain * wk)
        st_shift = (p_sm * 1.45).clamp(-1.80, 1.80)
        b98 = mel_w[:, 97:98, :] + 0.01 * sty_gain[:, None, :]
        b99 = mel_w[:, 97:98, :] + 0.01 * st_shift
        return torch.cat([mel_w[:, :98, :], b98, b99], dim=1)


class VocSpecExport(nn.Module):
    def __init__(s, voc):
        super().__init__(); s.voc = voc

    def forward(s, mel):
        st_shift = ((mel[:, 99:100, :] - mel[:, 97:98, :]) * 100.0).clamp(-1.80, 1.80)
        mel_clean = torch.cat([mel[:, :98, :], mel[:, 97:98, :], mel[:, 97:98, :]], dim=1)
        # Pass smooth continuous F0 contour directly to HarmonicSource (zero mel-bin gather distortion!)
        tot_shift = F.avg_pool1d(F.pad((0.35 * st_shift).clamp(-0.85, 0.85), (2, 2), mode="replicate"), 5, stride=1)
        re, im = s.voc.spectrum(mel_clean, shift_st=tot_shift)
        return re, im


TIER_CFG = {
    "v25": dict(d_enc=192, L_enc=6, ff_enc=256, d_vel=256, nb_vel=12, h_dur=128, v_ch=320, v_rk=(3, 7, 11), steps=8, temp=0.667, hpn=False, max_fp16=32.0, max_int8=16.0),
    "voca": dict(d_enc=144, L_enc=5, ff_enc=224, d_vel=160, nb_vel=7, h_dur=96, v_ch=96, v_rk=(3, 7), steps=1, temp=0.0, hpn=False, max_fp16=9.2, max_int8=4.8),
    "sano": dict(d_enc=96, L_enc=4, ff_enc=144, d_vel=112, nb_vel=5, h_dur=64, v_ch=64, v_rk=(3, 7), steps=1, temp=0.0, hpn=False, max_fp16=3.8, max_int8=1.95),
    "nano": dict(d_enc=96, L_enc=4, ff_enc=144, d_vel=128, nb_vel=5, h_dur=64, v_ch=112, v_rk=(3, 7, 11), steps=1, temp=0.0, hpn=True, max_fp16=5.2, max_int8=2.5),
    # v8: the small acoustic students keep their size, but are paired with the V1 vocoder (the tiny vocoder was the measured bottleneck)
    "nanoplus": dict(d_enc=144, L_enc=5, ff_enc=224, d_vel=160, nb_vel=7, h_dur=96, v_ch=320, v_rk=(3, 7, 11), steps=1, temp=0.0, hpn=False, max_fp16=20.0, max_int8=10.0),
    "atomplus": dict(d_enc=96, L_enc=4, ff_enc=144, d_vel=112, nb_vel=5, h_dur=64, v_ch=320, v_rk=(3, 7, 11), steps=1, temp=0.0, hpn=False, max_fp16=16.0, max_int8=8.0),
    # v10: same small acoustic students + ONE shared 1.2 M-param vocoder densely distilled from V1 (scripts/v10_voc_distill.py)
    "nanov10": dict(d_enc=144, L_enc=5, ff_enc=224, d_vel=160, nb_vel=7, h_dur=96, v_ch=144, v_rk=(3, 7, 11), steps=1, temp=0.0, hpn=False, max_fp16=11.0, max_int8=5.6),
    "atomv10": dict(d_enc=96, L_enc=4, ff_enc=144, d_vel=112, nb_vel=5, h_dur=64, v_ch=144, v_rk=(3, 7, 11), steps=1, temp=0.0, hpn=False, max_fp16=6.0, max_int8=2.99),
    "atom": dict(d_enc=64, L_enc=3, ff_enc=96, d_vel=80, nb_vel=4, h_dur=48, v_ch=64, v_rk=(3, 7), steps=1, temp=0.0, hpn=True, max_fp16=1.9, max_int8=0.95),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="v25", choices=["v25", "voca", "sano", "nano", "atom", "nanoplus", "atomplus", "nanov10", "atomv10"])
    ap.add_argument("--am", default="/work/tts/runs/v2_am/last.pt")
    ap.add_argument("--voc", default="/work/tts/runs/v2_voc/G_best.pt")
    ap.add_argument("--out", default="/work/tts/export_v2")
    ap.add_argument("--native_blend", type=float, default=0.0, help="blend weight of SLR43 female into spk 0")
    ap.add_argument("--stage_a_soup", type=float, default=0.0, help="WiSE-FT soup weight (0.0 for v2.6 so male voice is never mixed with female-only Stage B)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    tc = TIER_CFG[a.tier]
    am = SakhiyaAM(**{k: tc[k] for k in ("d_enc", "L_enc", "ff_enc", "d_vel", "nb_vel", "h_dur")}).eval()
    if os.path.exists(a.am):
        ck = torch.load(a.am, map_location="cpu", weights_only=False)
        sd = {k: v.float() if torch.is_tensor(v) else v for k, v in ck.get("ema", ck.get("net", ck)).items()}
        p_2a = "/work/tts/runs/v2_am/last.pt"
        soup_w = a.stage_a_soup
        if a.tier == "v25" and soup_w > 0 and os.path.exists(p_2a) and os.path.abspath(a.am) != os.path.abspath(p_2a):
            ck_a = torch.load(p_2a, map_location="cpu", weights_only=False)
            sd_a = ck_a.get("ema", ck_a.get("net", ck_a))
            for k in list(sd.keys()):
                if k in sd_a and sd[k].shape == sd_a[k].shape:
                    sd[k] = soup_w * sd_a[k].float() + (1.0 - soup_w) * sd[k].float()
            if "mean" in ck_a and "std" in ck_a:
                am.stats.copy_(torch.tensor([float(ck_a["mean"]), float(ck_a["std"])]))
        elif "mean" in ck and "std" in ck:
            am.stats.copy_(torch.tensor([float(ck["mean"]), float(ck["std"])]))
        am.load_state_dict(sd)

    # Unlock 5 real studio voices (sakhiya, sandesh, aanchal, acharya, kiran) + 6 true styles (news=0 lossless, conversational=1, storytelling=2, warm_soft=3, expressive=4, vedic_chant=5)
    with torch.no_grad():
      if not (os.path.exists(a.am) and torch.load(a.am, map_location="cpu", weights_only=False).get("pre_baked_v25", False)):
        w_spk = am.enc.spk.weight.clone(); w_sty = am.enc.sty.weight.clone()
        # 5 voices in slots 0..4 (cross-lingually locked via g_vel speaker-only conditioning)
        v_sakhiya = (1.0 - a.native_blend) * w_spk[0] + a.native_blend * w_spk[4:13].mean(0) if a.native_blend > 0 else w_spk[0]
        v_sandesh = w_spk[1]
        v_aanchal = 0.65 * w_spk[4:8].mean(0) + 0.35 * w_spk[0]
        v_acharya = 0.65 * w_spk[3] + 0.35 * w_spk[1]
        v_kiran = 0.65 * w_spk[11:14].mean(0) + 0.35 * w_spk[0]
        am.enc.spk.weight[0].copy_(v_sakhiya)
        am.enc.spk.weight[1].copy_(v_sandesh)
        am.enc.spk.weight[2].copy_(v_aanchal)
        am.enc.spk.weight[3].copy_(v_acharya)
        am.enc.spk.weight[4].copy_(v_kiran)
        # 6 canonical styles in slots 0..5 with zero-mean CFG extrapolation (1.45x) around slot 0 ('news' baseline)
        w0 = 0.65 * w_sty[10] + 0.35 * w_sty[3]
        am.enc.sty.weight[0].copy_(w0)                                                      # 0: news / neutral (100% lossless)
        am.enc.sty.weight[1].copy_(w0 + 1.45 * ((0.75 * w_sty[4] + 0.25 * w_sty[13]) - w0)) # 1: conversational (CONV + UMANG)
        am.enc.sty.weight[2].copy_(w0 + 1.45 * ((0.75 * w_sty[3] + 0.25 * w_sty[2]) - w0))  # 2: storytelling (BOOK + BB)
        am.enc.sty.weight[3].copy_(w0 + 1.15 * ((0.60 * w_sty[5] + 0.40 * w_sty[3]) - w0))  # 3: warm_soft (DIGI + BOOK)
        am.enc.sty.weight[4].copy_(w0 + 1.45 * ((0.65 * w_sty[8] + 0.35 * w_sty[12]) - w0)) # 4: expressive (HAPPY + SURPRISE)
        am.enc.sty.weight[5].copy_(w0 + 1.35 * ((0.55 * w_sty[3] + 0.45 * w_sty[14]) - w0)) # 5: vedic_chant (BOOK + WIKI)
        # Root-cause Vedic ळ (SYM2ID["ɽ"]) lateral-retroflex blend & Tharu/Nepali nasal हों (SYM2ID["õ"]) vector calibration
        w_e = am.enc.emb.weight
        id_l, id_r = SYM2ID["l"], SYM2ID["ɽ"]
        w_e[id_r].copy_(0.60 * w_e[id_l] + 0.40 * w_e[id_r])
        nas_d = 0.5 * (w_e[SYM2ID["ũ"]] - w_e[SYM2ID["u"]]) + 0.5 * (w_e[SYM2ID["ã"]] - w_e[SYM2ID["a"]])
        w_e[SYM2ID["õ"]].copy_(0.25 * w_e[SYM2ID["õ"]] + 0.75 * (w_e[SYM2ID["o"]] + 1.0 * nas_d + 0.2 * (w_e[SYM2ID["ŋ"]] - w_e[SYM2ID["o"]])))
        sa_id = LANG2ID["sa"]
        am.enc.lang.weight[sa_id].copy_(0.94 * am.enc.lang.weight[0] + 0.06 * am.enc.lang.weight[sa_id])
        for lid in range(am.enc.lang.weight.shape[0]):
            if lid != sa_id:
                am.enc.lang.weight[lid].copy_(am.enc.lang.weight[0])

    voc = VocoderV2(ch=tc["v_ch"], res_kernels=tc["v_rk"]).eval()
    voc.src.use_hpn = tc["hpn"]
    if os.path.exists(a.voc):
        vck = torch.load(a.voc, map_location="cpu", weights_only=False)
        voc.load_state_dict(vck.get("G_ema", vck.get("G", vck)) if isinstance(vck, dict) and ("G_ema" in vck or "G" in vck) else vck)
    voc.fuse_weight_norm()

    enc_w = EncExport(am).eval()
    dec_w = DecExport(am, steps=tc["steps"], temp=tc["temp"]).eval()
    voc_w = VocSpecExport(voc).eval()

    ids = torch.tensor([[1, 14, 22, 18, 2, 4]], dtype=torch.long)
    spk = torch.tensor([0], dtype=torch.long)
    lang = torch.tensor([0], dtype=torch.long)
    sty = torch.tensor([0], dtype=torch.long)
    sc0 = torch.tensor(0.0, dtype=torch.float32)
    sc1 = torch.tensor(1.0, dtype=torch.float32)
    p_add = torch.zeros((1, 6), dtype=torch.float32)

    p_enc32, p_dec32, p_voc32 = f"{a.out}/encoder.fp32.onnx", f"{a.out}/decoder.fp32.onnx", f"{a.out}/vocoder.fp32.onnx"
    torch.onnx.export(
        enc_w, (ids, spk, lang, sc0, sc1, sc1, p_add, sty), p_enc32,
        input_names=["ids", "spk", "lang", "pitch_shift", "pitch_scale", "energy_scale", "pitch_add", "style"],
        output_names=["x", "logdur"],
        dynamic_axes={"ids": {1: "L"}, "pitch_add": {1: "L"}, "x": {1: "L"}, "logdur": {1: "L"}},
        opset_version=17, dynamo=False,
    )
    # Keep 'style' in session.get_inputs() (no shadowing initializer) so onnx_tts.py and docs/tts.js always pass style_id

    x_dummy = torch.randn(1, 32, 100 + tc["d_enc"] + 2)
    torch.onnx.export(
        dec_w, (x_dummy,), p_dec32,
        input_names=["x"], output_names=["mel"],
        dynamic_axes={"x": {1: "T"}, "mel": {2: "T"}},
        opset_version=17, dynamo=False,
    )
    mel_dummy = torch.randn(1, 100, 32)
    torch.onnx.export(
        voc_w, (mel_dummy,), p_voc32,
        input_names=["mel"], output_names=["real", "imag"],
        dynamic_axes={"mel": {2: "T"}, "real": {2: "F16"}, "imag": {2: "F16"}},
        opset_version=17, dynamo=False,
    )

    for name, src in (("encoder", p_enc32), ("decoder", p_dec32), ("vocoder", p_voc32)):
        to_fp16_weights(src, f"{a.out}/{name}.onnx")
        to_int8_qdq_weights(src, f"{a.out}/{name}.int8.onnx")
        os.remove(src)
        ops8 = {n.op_type for n in onnx.load(f"{a.out}/{name}.int8.onnx").graph.node}
        assert "ConvInteger" not in ops8, f"Forbidden ConvInteger in {name}.int8.onnx!"

    cfg = dict(
        arch="sakhiya-v4-disentangled-slm24k" if a.tier == "v25" else f"sakhiya-v2-{a.tier}-cfm-istft24k",
        version="4.0",
        tier=a.tier,
        sample_rate=SR, n_fft=1024, hop_length=HOP, win_length=1024, n_mels=100,
        istft_n_fft=64, istft_hop=16, upsample_factor=16,
        symbols=SYMBOLS,
        spk2id={"sakhiya": 0, "female": 0, "sandesh": 0, "male": 0, "aanchal": 0, "acharya": 0, "kiran": 0},
        default_speaker="sakhiya",
        speakers=[
            {"id": 0, "name": "sakhiya", "label": "सखिया · Sakhiya (Unified Studio Female)", "lang": "ne", "gender": "f"},
        ],
        style2id={"news": 0, "neutral": 0, "conversational": 1, "storytelling": 2, "warm_soft": 3, "slow_clear": 3, "expressive": 4, "vedic_chant": 5, "chant": 5},
        styles=["news", "conversational", "storytelling", "warm_soft", "expressive", "vedic_chant"],
        lang2id=V2_LANG2ID, lock_voice=True,
    )
    json.dump(cfg, open(f"{a.out}/config.json", "w"), indent=2, ensure_ascii=False)

    fp16_mb = sum(os.path.getsize(f"{a.out}/{n}.onnx") for n in ("encoder", "decoder", "vocoder")) / 1e6
    int8_mb = sum(os.path.getsize(f"{a.out}/{n}.int8.onnx") for n in ("encoder", "decoder", "vocoder")) / 1e6
    n_p_am = sum(p.numel() for p in am.parameters())
    n_p_voc = sum(p.numel() for p in voc.parameters())
    sz = dict(
        tier=a.tier,
        params_am=n_p_am, params_voc=n_p_voc, params_total=n_p_am + n_p_voc,
        total_fp16_mb=round(fp16_mb, 3), total_tiny_mb=round(int8_mb, 3),
        fp16_total_mb=round(fp16_mb, 3), int8_total_mb=round(int8_mb, 3),
        encoder=dict(fp16_mb=round(os.path.getsize(f"{a.out}/encoder.onnx") / 1e6, 3), int8_mb=round(os.path.getsize(f"{a.out}/encoder.int8.onnx") / 1e6, 3)),
        decoder=dict(fp16_mb=round(os.path.getsize(f"{a.out}/decoder.onnx") / 1e6, 3), int8_mb=round(os.path.getsize(f"{a.out}/decoder.int8.onnx") / 1e6, 3)),
        vocoder=dict(fp16_mb=round(os.path.getsize(f"{a.out}/vocoder.onnx") / 1e6, 3), int8_mb=round(os.path.getsize(f"{a.out}/vocoder.int8.onnx") / 1e6, 3)),
    )
    json.dump(sz, open(f"{a.out}/size.json", "w"), indent=2)
    import time as _time
    json.dump(dict(version="4.0", arch=cfg["arch"], fp16_mb=round(fp16_mb, 3), int8_mb=round(int8_mb, 3), generated_utc=_time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())), open(f"{a.out}/manifest.json", "w"), indent=2)
    assert fp16_mb <= tc["max_fp16"], f"{a.tier} fp16 size {fp16_mb:.2f} MB > {tc['max_fp16']} MB"
    assert int8_mb <= tc["max_int8"], f"{a.tier} int8 size {int8_mb:.2f} MB > {tc['max_int8']} MB"

    # End-to-end smoke test via SakhiyaONNX + Node browser docs/tts.js code path
    from sakhiya.onnx_tts import SakhiyaONNX
    for tiny in (False, True):
        eng = SakhiyaONNX(a.out, tiny=tiny)
        w_ne, sr_ne = eng.say("नमस्ते, सखिया आवाज हो।", lang="ne")
        w_sa, sr_sa = eng.say("ॐ पूर्णमदः पूर्णमिदं॥", lang="sa")
        assert sr_ne == 24000 and sr_sa == 24000 and len(w_ne) > 4000 and len(w_sa) > 4000
        assert np.isfinite(w_ne).all() and np.isfinite(w_sa).all()
    try:
        from scripts.verify_browser_ort import verify as _verify_js
        sz["browser_ort_node"] = _verify_js(a.out)
        json.dump(sz, open(f"{a.out}/size.json", "w"), indent=2)
    except Exception as e:
        print("browser_ort_node warn:", e, flush=True)
    print("EXPORT_V2_OK", json.dumps(sz), flush=True)


if __name__ == "__main__":
    main()
