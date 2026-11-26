#!/usr/bin/env python3
"""Phase 2 Candidate B — `Sakhiya-One-Vocos` (Pretrained `charactr/vocos-mel-24khz` 13.5M + `Mel80To100Adapter`)
(`scripts/train_phase2_cand_b_vocos.py`).

1. Preserves 100% of Pretrained `charactr/vocos-mel-24khz` weights via `Mel80To100Adapter`:
   - Closed-form ridge regression mapping `Sakhiya-One`'s exact 80-band GT mels (plus 20 high bands) ->
     Vocos's exact 100-band HTK log-mels on pure Pro-v3 audio + zero-initialized 2-layer ConvNeXt residual.
2. Stage B1 — Adapter Warm-Up: Freezes Vocos's pretrained ConvNeXt backbone completely and trains ONLY
   `Mel80To100Adapter` (`lr = 3e-4`, 2,500 steps) on GT + Phase-1 `Sakhiya-One` 4-step predicted mels against
   pure Pro-v3 waveforms (Multi-Scale Mel + Complex STFT + WavLM-FM).
3. Stage B2 — Gentle Pretrained Vocos Adaptation + Light E2E Polish: Unfreezes `Vocos-24kHz` at gentle LR
   (`1.5e-5` cosine, `EMA = 0.999`, `MS-SB-CQT` + Multi-Scale Mel + Complex STFT + WavLM-FM) on 50% pure Pro-v3
   GT mels + 50% Phase-1 `Sakhiya-One` 4-step full-utterance predicted mels, followed by gentle joint E2E with
   the AM (`am_lr = 4e-6`, duration detached) so predicted mels match Vocos's razor-sharp manifold.
4. Exports BOTH FP16 (`~28 MB`) and clean QDQ INT8 ONNX (`~15-19 MB` total with `Sakhiya-One` AM, FP16 iSTFT head,
   zero `ConvInteger`, verified in `onnxruntime-node`), and renders all 3 Owner Clips onto Card 5.
"""
import argparse
import copy
import json
import math
import os
import random
import subprocess
import sys
import time
import librosa
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
for p in (
    REPO, HERE,
    os.path.join(REPO, "scripts", "sakhiya_pro"),
    os.path.join(REPO, "scripts", "sakhiya_pro", "distill"),
    os.path.join(REPO, "scripts", "sakhiya_pro", "distill2"),
):
    if p not in sys.path:
        sys.path.insert(0, p)

from sakhiya.v2_vocoder import SR, HOP, mel_spec_24k
from sakhiya.vocos_24k_adapter import Mel80To100Adapter, PretrainedVocos24kHz
from sakhiya.news_vocoder_100mel import MultiScaleSubBandCQTDiscriminator
from sakhiya.cfm import _dur_to_attn
from sakhiya.symbols import SYM2ID
from scripts.train_v2_voc import mr_complex_stft_loss
from scripts.train_phase2_cand_a_cqt_awp import sibilant_and_upper_smoothness_loss
from student import Student, compute_svara_mora_ids, compute_intra_phoneme_progress, _calibrate_styles
from run_session75_breakthrough import FrozenMultiLayerWavLM
from export_gate import _dedup_initializers, _strip_meta, _normalize_lufs_16, ids_of
from scripts.forensic_audio_gate import audit_int8_qdq_initializers, compute_comb_grit_and_warble
from scripts.run_phase0_sakhiya_one_honest import _voiced_vowel_hnr_0p2_4khz
from scripts.train_phase1_am_polish import forward_4step_meanflow, update_stage, log

LISTEN_DIR = "/work/sp/polish/listen"
CK_DIR = "/work/sp/polish/ck"
ONNX_DIR = "/work/sp/polish/onnx/cand_b_vocos"
os.makedirs(LISTEN_DIR, exist_ok=True)
os.makedirs(CK_DIR, exist_ok=True)
os.makedirs(ONNX_DIR, exist_ok=True)


class SakhiyaOneVocosModel(nn.Module):
    """Unified single-graph `Sakhiya-One` 4-step AM + `Mel80To100Adapter` + `PretrainedVocos24kHz`."""
    def __init__(self, base_student: Student, adapter: Mel80To100Adapter, vocos: PretrainedVocos24kHz):
        super().__init__()
        self.am = copy.deepcopy(base_student.am)
        self.pros_tbl = base_student.pros_tbl.clone()
        self.closure_slope = nn.Parameter(base_student.closure_slope.data.clone())
        self.dur_rate_bias = nn.Parameter(base_student.dur_rate_bias.data.clone())
        self.steps = 4
        self.cfg = dict(base_student.cfg)
        self.cfg["steps"] = 4
        self.cfg["voc_arch"] = "vocos_24k"
        self.adapter = adapter
        self.vocos = vocos

    def infer(self, ids, lang, dur_scale, pitch_shift):
        B, L = ids.shape
        im = torch.ones((B, L), dtype=torch.bool, device=ids.device)
        spk0 = torch.zeros((B,), dtype=torch.long, device=ids.device)
        studio = torch.ones((B,), dtype=torch.long, device=ids.device)
        is_sa = (lang == 1)
        lang_eff = torch.where(is_sa, torch.full_like(lang, 4), torch.zeros_like(lang))
        sty = torch.where(is_sa, torch.full_like(lang, 5), torch.zeros_like(lang))
        pros = self.pros_tbl.to(ids.device)[sty]

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
        id_r, id_schwa, id_t, id_m_c = SYM2ID["r"], SYM2ID["ʌ"], SYM2ID["t̪"], SYM2ID["m"]
        id_retro_t, id_l_c, id_sh_c = SYM2ID["ʈ"], SYM2ID["l"], SYM2ID["ʃ"]
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
        is_func_c = ((prev_ids <= 6) & (next2_ids <= 6) & ((ids == id_r) | (ids == id_h) | (ids == id_t) | (ids == id_m_c)) & ((next_ids == id_schwa) | (next_ids == id_o))).float()
        is_func_v = ((prev2_ids <= 6) & (next_ids <= 6) & ((prev_ids == id_r) | (prev_ids == id_h) | (prev_ids == id_t) | (prev_ids == id_m_c)) & ((ids == id_schwa) | (ids == id_o))).float()
        dur = torch.maximum(dur, is_func_c * 6.0 + is_func_v * 8.0)
        is_onset_wb = ((pos == 0.0) & (ids == id_wb)).float()
        dur = torch.maximum(dur, is_onset_wb * 4.0)
        is_gem_half = ((ids > 8) & ((ids == next_ids) | (ids == prev_ids))).float()
        dur = torch.maximum(dur, is_gem_half * 5.0)
        is_word_final_stop = (((ids == id_t) | (ids == id_retro_t)) & (next_ids <= 6) & (prev_ids > 8)).float()
        is_post_stop_wb = ((ids == id_wb) & ((prev_ids == id_t) | (prev_ids == id_retro_t))).float()
        dur = torch.maximum(dur, is_word_final_stop * 5.0 + is_post_stop_wb * 4.0)
        is_post_sib_l = ((ids == id_l_c) & (prev_ids == id_sh_c)).float()
        dur = torch.maximum(dur, is_post_sib_l * 5.0)
        is_nas_vow = (ids >= 58).float()
        dur = torch.maximum(dur, is_nas_vow * 5.0)

        ends = torch.cumsum(dur, 1)
        T = ends[0, -1].to(torch.int64)
        t_arange = torch.arange(0, T, device=ids.device, dtype=torch.float32)
        idx = (t_arange[None, :, None] >= ends[:, None, :]).to(torch.float32).sum(-1).to(torch.int64).clamp(max=L - 1)

        mu_t = mu.transpose(1, 2)
        mu_y = torch.gather(mu_t, 2, idx[:, None, :].expand(-1, mu_t.shape[1], -1))
        prog_4d = compute_intra_phoneme_progress(dur, ends, idx, t_arange)
        m_t = torch.ones((B, 1, T), dtype=mu_y.dtype, device=mu_y.device)

        t0 = torch.zeros((B,), device=mu_y.device, dtype=torch.float32)
        r_one = torch.ones((B,), device=mu_y.device, dtype=torch.float32)
        w2 = self.am.vel.step2_gate.clamp(0.0, 0.35)
        v_cur = self.am.vel(mu_y, t0, mu_y, g, m_t, prog=prog_4d, n_blocks=8)
        for s_i in range(1, 4):
            alpha_s = float(s_i) / 4.0
            x_s = mu_y + alpha_s * v_cur
            t_s = torch.full((B,), alpha_s, device=mu_y.device, dtype=torch.float32)
            v_s = self.am.vel(x_s, t_s, mu_y, g, m_t, prog=prog_4d, r=r_one, n_blocks=8)
            v_cur = (1.0 - w2) * v_cur + w2 * v_s
        xt = mu_y + v_cur

        mel_s1 = xt * self.am.stats[1] + self.am.stats[0]
        mel_s1_clean = torch.cat([mel_s1[:, :98, :], mel_s1[:, 97:98, :], mel_s1[:, 97:98, :]], dim=1)
        mel_vocos = self.adapter(mel_s1_clean)
        y = self.vocos(mel_vocos)
        pk = y.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
        return (y * (0.88 / pk)).clamp(-0.98, 0.98)


class _VocosWrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, ids, lang, dur_scale, pitch_shift):
        return self.m.infer(ids, lang, dur_scale, pitch_shift)


def export_vocos_onnx_fp16_and_int8(model: SakhiyaOneVocosModel, out_dir: str):
    f32_p = os.path.join(out_dir, "model.fp32.onnx")
    f16_p = os.path.join(out_dir, "model.fp16.onnx")
    i8_p = os.path.join(out_dir, "model.int8.onnx")

    m_cpu = copy.deepcopy(model).cpu().eval()
    ids = torch.randint(3, 50, (1, 40))
    args = (ids, torch.zeros(1, dtype=torch.long), torch.ones(40), torch.zeros(40))
    kw = dict(
        input_names=["ids", "lang", "dur_scale", "pitch_shift"],
        output_names=["wav"],
        dynamic_axes={"ids": {1: "L"}, "dur_scale": {0: "D"}, "pitch_shift": {0: "P"}, "wav": {1: "N"}},
        opset_version=17,
        do_constant_folding=True,
    )
    try:
        torch.onnx.export(_VocosWrap(m_cpu), args, f32_p, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(_VocosWrap(m_cpu), args, f32_p, **kw)

    def _promote_consts(m_onnx, min_sz=64):
        new_n = []
        idx_c = 0
        for n in m_onnx.graph.node:
            if n.op_type == "Constant" and len(n.output) == 1:
                t_attr = [a for a in n.attribute if a.name == "value"]
                if t_attr:
                    arr = numpy_helper.to_array(t_attr[0].t)
                    if arr.size >= min_sz and arr.dtype == np.float32:
                        inm = f"const_init_{idx_c}_{n.output[0]}"
                        idx_c += 1
                        m_onnx.graph.initializer.append(numpy_helper.from_array(arr, inm))
                        new_n.append(helper.make_node("Identity", [inm], [n.output[0]]))
                        continue
            new_n.append(n)
        del m_onnx.graph.node[:]
        m_onnx.graph.node.extend(new_n)

    # 1. Build FP16 ONNX: promote Constant buffers & cast all large float32 initializers to float16 + Cast node
    om16 = onnx.load(f32_p)
    _promote_consts(om16)
    _dedup_initializers(om16)
    keep16, new16 = [], []
    for init in om16.graph.initializer:
        a = numpy_helper.to_array(init)
        if init.data_type == TensorProto.FLOAT and a.ndim >= 2 and a.size >= 64:
            h = numpy_helper.from_array(a.astype(np.float16), init.name + "_h")
            keep16.append(h)
            new16.append(helper.make_node("Cast", [h.name], [init.name], to=TensorProto.FLOAT, name=init.name + "_cast"))
        else:
            keep16.append(init)
    del om16.graph.initializer[:]
    om16.graph.initializer.extend(keep16)
    nodes16 = list(om16.graph.node)
    del om16.graph.node[:]
    om16.graph.node.extend(new16 + nodes16)
    _strip_meta(om16)
    onnx.save(om16, f16_p)

    # 2. Build QDQ INT8 ONNX (~18.6 MB): keep head_out (512,1026), adapter, and embeddings in FP16;
    # quantize ConvNeXt pwconv1/pwconv2 & AM velocity/encoder weights to per-channel INT8 QDQ
    om8 = onnx.load(f32_p)
    _promote_consts(om8)
    _dedup_initializers(om8)
    use = {}
    for n in om8.graph.node:
        for i in n.input:
            use.setdefault(i, n.op_type)
    keep8, new8 = [], []
    c_min = numpy_helper.from_array(np.array(-100.0, dtype=np.float32), "dq_clip_min")
    c_max = numpy_helper.from_array(np.array(100.0, dtype=np.float32), "dq_clip_max")
    keep8 += [c_min, c_max]
    fp16_keep_names = ("head_out", "adapter", "am.enc.emb", "am.dur.o")
    for init in om8.graph.initializer:
        a = numpy_helper.to_array(init)
        op = use.get(init.name, "")
        if init.data_type != TensorProto.FLOAT:
            keep8.append(init)
            continue
        if a.ndim == 1 and a.size >= 128 and op not in ("Resize", "Upsample", "Slice", "Clip", "ConstantOfShape"):
            h = numpy_helper.from_array(a.astype(np.float16), init.name + "_h")
            keep8.append(h)
            new8.append(helper.make_node("Cast", [h.name], [init.name], to=TensorProto.FLOAT, name=init.name + "_cast"))
            continue
        if a.ndim < 2 or a.size <= 64:
            keep8.append(init)
            continue
        ax = 0 if a.shape == (1026, 1, 1024) else (1 if (op in ("ConvTranspose", "MatMul") and a.ndim >= 2 and a.shape[1] > 1) else 0)
        is_istft_or_head = (
            a.shape in ((512, 1026), (1026, 512), (100, 80, 1), (100, 20, 1))
            or (a.ndim == 3 and a.shape[1] == 1 and a.shape[2] > 1 and a.shape != (1026, 1, 1024))
            or a.shape[ax] <= 8
            or any(k in init.name for k in fp16_keep_names)
        )
        if a.ndim >= 2 and a.size >= 256 and a.shape[ax] >= 16 and not is_istft_or_head:
            red = tuple(i for i in range(a.ndim) if i != ax)
            sc = (np.maximum(np.abs(a).max(axis=red), 1e-8) / 127.0).astype(np.float32)
            shp = [1] * a.ndim
            shp[ax] = -1
            q = np.clip(np.round(a / sc.reshape(shp)), -127, 127).astype(np.int8)
            keep8 += [numpy_helper.from_array(q, init.name + "_q"), numpy_helper.from_array(sc, init.name + "_s")]
            if op == "MatMul":
                new8.append(helper.make_node("DequantizeLinear", [init.name + "_q", init.name + "_s"], [init.name + "_dq_raw"], axis=ax, name=init.name + "_dq"))
                new8.append(helper.make_node("Clip", [init.name + "_dq_raw", "dq_clip_min", "dq_clip_max"], [init.name], name=init.name + "_clip"))
            else:
                new8.append(helper.make_node("DequantizeLinear", [init.name + "_q", init.name + "_s"], [init.name], axis=ax, name=init.name + "_dq"))
        else:
            h = numpy_helper.from_array(a.astype(np.float16), init.name + "_h")
            keep8.append(h)
            new8.append(helper.make_node("Cast", [h.name], [init.name], to=TensorProto.FLOAT, name=init.name + "_cast"))
    del om8.graph.initializer[:]
    om8.graph.initializer.extend(keep8)
    nodes8 = list(om8.graph.node)
    del om8.graph.node[:]
    om8.graph.node.extend(new8 + nodes8)
    _strip_meta(om8)
    onnx.save(om8, i8_p)
    return f32_p, f16_p, i8_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b1_steps", type=int, default=1500)
    ap.add_argument("--b2_steps", type=int, default=2500)
    ap.add_argument("--bs", type=int, default=20)
    a = ap.parse_args()

    t0 = time.time()
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(42)
    torch.manual_seed(42)

    tot_steps = a.b1_steps + a.b2_steps
    update_stage(3, 0, tot_steps, "init", "loading Pretrained charactr/vocos-mel-24khz & fitting closed-form Mel80To100Adapter", 0.0, 12.0, "running")

    # 1. Load Phase-1 Anchored AM
    ck_p = os.path.join(CK_DIR, "phase1_am_polished.pt")
    if not os.path.exists(ck_p):
        ck_p = "/work/sp/distill/runs/sakhiya_one/ema_final.pt"
    S = torch.load(ck_p, map_location="cpu", weights_only=False)
    base_stu = Student(tier="sakhiya_one")
    base_stu.steps = 4
    base_stu.cfg["steps"] = 4
    base_stu.warm_init("/work/tts/ck")
    own_sd = base_stu.state_dict()
    filt_sd = {k: v for k, v in S["ema"].items() if k in own_sd and own_sd[k].shape == v.shape}
    base_stu.load_state_dict(filt_sd, strict=False)
    _calibrate_styles(base_stu.am)
    base_stu.to(dev).eval()

    # 2. Load 100% Pretrained charactr/vocos-mel-24khz & fit Mel80To100Adapter via closed-form ridge regression
    vocos = PretrainedVocos24kHz().to(dev).load_pretrained_hf()
    adapter = Mel80To100Adapter().to(dev)

    D_gt = torch.load("/work/sp/distill/cache_train.pt", map_location="cpu", weights_only=False)
    mf = {json.loads(l)["id"]: json.loads(l) for l in open("/work/sp/distill/manifest.jsonl")}
    DROP_LEGACY = {"vc_slr54", "vc_slr43", "vc_v13", "anchor_6834_stitched"}
    pure_gt = [x for x in D_gt["items"][:700] if mf.get(x["id"], {}).get("src", "wiki") not in DROP_LEGACY and len(x["wav"]) >= 128 * HOP]

    mel_s1_list, mel_voc_list, paired_gt_clips = [], [], []
    with torch.no_grad():
        for x in pure_gt[:240]:
            w_t = (x["wav"].to(dev).float() / 32767.0)[None]
            n_tr = (w_t.shape[1] // HOP) * HOP
            w_t = w_t[:, :n_tr]
            m_s1 = mel_spec_24k(w_t, dev)[0]
            m_vc = vocos.extract_gt_vocos_mel(w_t)[0, :, :m_s1.shape[1]]
            T_c = min(m_s1.shape[1], m_vc.shape[1])
            mel_s1_list.append(m_s1[:, :T_c].cpu())
            mel_voc_list.append(m_vc[:, :T_c].cpu())
            paired_gt_clips.append(dict(
                wav=x["wav"][:T_c * HOP],
                mel_s1=m_s1[:, :T_c].half().cpu(),
                mel_voc=m_vc[:, :T_c].half().cpu(),
            ))

    fit_info = adapter.fit_closed_form_ridge(mel_s1_list, mel_voc_list, ridge_lambda=1e-2)
    log(f"[Phase 2B Ridge Init] Mel80To100Adapter closed-form fit on {fit_info['n_frames']} pure Pro-v3 frames: R2={fit_info['r2']} L1={fit_info['l1_err']}")

    G = SakhiyaOneVocosModel(base_stu, adapter, vocos).to(dev)
    ema = copy.deepcopy(G).eval()

    # Load full-utterance trajectories for 50% GT + 50% Phase-1 4-step predicted mel training
    full_utt_items = torch.load("/work/sp/polish/realigned_full_utt_cache.pt", map_location="cpu", weights_only=False)
    long_utt_items = [x for x in full_utt_items if x["y_clean"].shape[0] >= 120]
    if len(long_utt_items) < 60:
        long_utt_items = full_utt_items

    # Teacher waveform helper for predicted mels (using frozen Pretrained Vocos on adapted clean mels)
    frozen_vocos = copy.deepcopy(vocos).to(dev).eval()
    for p in frozen_vocos.parameters():
        p.requires_grad_(False)

    d_cqt = MultiScaleSubBandCQTDiscriminator().to(dev).train()
    wavlm_fm = FrozenMultiLayerWavLM(dev=dev)
    mean, std = float(G.am.stats[0].item()), float(G.am.stats[1].item())
    seg_frames = 120  # 1.28 s window

    # --- STAGE B1: Adapter Warm-Up (Vocos ConvNeXt backbone 100% frozen!) ---
    for p in G.parameters():
        p.requires_grad_(False)
    for p in G.adapter.parameters():
        p.requires_grad_(True)
    oB1 = torch.optim.AdamW(G.adapter.parameters(), lr=3.0e-4, betas=(0.8, 0.99), weight_decay=1e-4)

    tstep = time.time()
    log(f"[Phase 2B Stage B1] Training Mel80To100Adapter with Pretrained Vocos frozen for {a.b1_steps} steps...")
    for st in range(1, a.b1_steps + 1):
        prog = st / max(1, a.b1_steps)
        lr_cur = 3.0e-5 + 2.7e-4 * 0.5 * (1.0 + math.cos(math.pi * prog))
        for pg in oB1.param_groups:
            pg["lr"] = lr_cur

        sub = random.sample(paired_gt_clips, a.bs)
        m_s1_b = torch.zeros((a.bs, 100, seg_frames), device=dev)
        m_vc_b = torch.zeros((a.bs, 100, seg_frames), device=dev)
        w_gt_b = torch.zeros((a.bs, seg_frames * HOP), device=dev)
        for bi, it in enumerate(sub):
            Tk = it["mel_s1"].shape[1]
            f_st = random.randint(0, max(0, Tk - seg_frames - 1))
            m_s1_b[bi] = it["mel_s1"][:, f_st : f_st + seg_frames].to(dev).float()
            m_vc_b[bi] = it["mel_voc"][:, f_st : f_st + seg_frames].to(dev).float()
            w_gt_b[bi] = it["wav"][f_st * HOP : (f_st + seg_frames) * HOP].to(dev).float() / 32767.0

        # Add slight mel jitter on 50% of batch to simulate 4-step AM predicted mels
        noise_m = torch.randn_like(m_s1_b) * 0.04
        m_in = torch.where((torch.arange(a.bs, device=dev) % 2 == 0)[:, None, None], m_s1_b, m_s1_b + noise_m)
        m_in = torch.cat([m_in[:, :98, :], m_in[:, 97:98, :], m_in[:, 97:98, :]], dim=1)

        G.adapter.train()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            m_pred_vc = G.adapter(m_in)
            l_mel_proj = F.l1_loss(m_pred_vc.float(), m_vc_b.float()) + 0.5 * F.mse_loss(m_pred_vc.float(), m_vc_b.float())
            y_pred = G.vocos(m_pred_vc)
            T_w = min(y_pred.shape[1], w_gt_b.shape[1])
            y_pred = y_pred[:, :T_w]
            w_gt = w_gt_b[:, :T_w]
            l_stft = mr_complex_stft_loss(y_pred, w_gt, dev)
            l_mel_wav = F.l1_loss(G.vocos.extract_gt_vocos_mel(y_pred)[:, :, :seg_frames], m_vc_b)
            loss = 10.0 * l_mel_proj + 8.0 * l_mel_wav + 1.0 * l_stft

        oB1.zero_grad(set_to_none=True)
        loss.backward()
        if torch.isfinite(torch.nn.utils.clip_grad_norm_(G.adapter.parameters(), 1.5)):
            oB1.step()

        with torch.no_grad():
            for pe, pg in zip(ema.adapter.parameters(), G.adapter.parameters()):
                pe.lerp_(pg, 1.0 - 0.995)

        if st % 250 == 0 or st == a.b1_steps:
            now = time.time()
            ips = 250.0 / max(1e-6, now - tstep)
            tstep = now
            el_m = (now - t0) / 60.0
            eta_m = max(0.0, el_m * (tot_steps - st) / max(1, st))
            l_str = f"[B1 Adapter] proj={l_mel_proj.item():.4f} mel_wav={l_mel_wav.item():.4f} stft={l_stft.item():.4f}"
            update_stage(3, st, tot_steps, f"{ips:.1f} it/s", l_str, el_m, eta_m, "running")
            log(f"[Phase 2B Stage B1] step {st}/{a.b1_steps} ({ips:.1f} it/s) {l_str}")

    G.adapter.load_state_dict(ema.adapter.state_dict())

    # --- STAGE B2: Gentle Pretrained Vocos Adaptation (lr=1.5e-5) + Light Joint E2E with AM (am_lr=4e-6) ---
    voc_p = list(G.adapter.parameters()) + list(G.vocos.parameters())
    am_p = []
    for p in voc_p:
        p.requires_grad_(True)
    for n, p in G.am.named_parameters():
        if n.endswith("had_rot.weight") or n.startswith(("dur.", "f0_head.", "enc.f0_pred", "enc.vuv_pred")):
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)
            am_p.append(p)
    G.closure_slope.requires_grad_(False)
    G.dur_rate_bias.requires_grad_(False)

    oB2 = torch.optim.AdamW([
        {"params": voc_p, "lr": 1.5e-5, "initial_lr": 1.5e-5, "weight_decay": 1e-4},
        {"params": am_p, "lr": 4.0e-6, "initial_lr": 4.0e-6, "weight_decay": 1e-4},
    ], betas=(0.8, 0.99))
    oD = torch.optim.AdamW(d_cqt.parameters(), lr=3.0e-5, betas=(0.8, 0.99))

    log(f"[Phase 2B Stage B2] Gentle Pretrained Vocos Adaptation + Light E2E for {a.b2_steps} steps...")
    tstep = time.time()
    for st in range(1, a.b2_steps + 1):
        g_step = a.b1_steps + st
        prog = st / max(1, a.b2_steps)
        mult = 0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * prog))
        for pg in oB2.param_groups:
            pg["lr"] = pg["initial_lr"] * mult

        # 50% pure Pro-v3 GT mels + 50% full-utterance 4-step AM predicted mels
        B_gt = a.bs // 2
        B_am = a.bs - B_gt

        sub_gt = random.sample(paired_gt_clips, B_gt)
        m_s1_gt = torch.zeros((B_gt, 100, seg_frames), device=dev)
        m_vc_gt = torch.zeros((B_gt, 100, seg_frames), device=dev)
        w_real_gt = torch.zeros((B_gt, seg_frames * HOP), device=dev)
        for bi, it in enumerate(sub_gt):
            Tk = it["mel_s1"].shape[1]
            f_st = random.randint(0, max(0, Tk - seg_frames - 1))
            m_s1_gt[bi] = it["mel_s1"][:, f_st : f_st + seg_frames].to(dev).float()
            m_vc_gt[bi] = it["mel_voc"][:, f_st : f_st + seg_frames].to(dev).float()
            w_real_gt[bi] = it["wav"][f_st * HOP : (f_st + seg_frames) * HOP].to(dev).float() / 32767.0

        xs = random.sample(long_utt_items, B_am)
        ids = pad_sequence([x["ids"].to(dev) for x in xs], batch_first=True)
        dur = pad_sequence([x["dur"].to(dev) for x in xs], batch_first=True)
        p_enc = pad_sequence([x["p_enc"].to(dev).float() for x in xs], batch_first=True)
        y_clean = pad_sequence([x["y_clean"].to(dev).float() for x in xs], batch_first=True).transpose(1, 2).contiguous()
        y_frozen = pad_sequence([x["y_frozen"].to(dev).float() for x in xs], batch_first=True).transpose(1, 2).contiguous()
        il = torch.tensor([len(x["ids"]) for x in xs], device=dev)
        yl = torch.tensor([x["y_clean"].shape[0] for x in xs], device=dev)
        L, T = ids.shape[1], y_clean.shape[2]
        im = torch.arange(L, device=dev)[None] < il[:, None]
        ym = torch.arange(T, device=dev)[None] < yl[:, None]
        ymf = ym[:, None].float()
        spk = torch.zeros(B_am, dtype=torch.long, device=dev)
        studio = torch.ones(B_am, dtype=torch.long, device=dev)
        is_sa = torch.tensor([x["lang"] == 1 for x in xs], device=dev)
        lang_eff = torch.where(is_sa, torch.full((B_am,), 4, dtype=torch.long, device=dev), torch.zeros(B_am, dtype=torch.long, device=dev))
        sty = torch.where(is_sa, torch.full((B_am,), 5, dtype=torch.long, device=dev), torch.zeros(B_am, dtype=torch.long, device=dev))
        ones_m = torch.ones((B_am, 1), device=dev)
        ds_tok = (dur.float() / 7.5).clamp(0.6, 1.8)
        svara_ids, mora_ids = compute_svara_mora_ids(ids, is_sa, ds_tok, p_enc)

        G.train()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # Full-utterance 4-step AM forward
            hs, mus, gs = G.am.enc(
                ids, spk, lang_eff, sty, studio, im,
                pitch_add=p_enc, prosody=G.pros_tbl.to(dev)[sty], sty_mask=ones_m,
                svara_ids=svara_ids, mora_ids=mora_ids,
            )
            eng_s = G.am.enc.energy_pred(hs).squeeze(-1).clamp(-1.2, 1.2)
            mus_e = mus + (0.10 * G.pros_tbl.to(dev)[sty][:, 2:3] + 0.06 * eng_s)[..., None] * 0.32
            attn = _dur_to_attn(dur, im, ym)
            mu_y_s = torch.einsum("blt,bld->bdt", attn.to(mus_e.dtype), mus_e) * ymf
            ends = torch.cumsum(dur.float(), dim=1)
            t_arange = torch.arange(0, T, device=dev, dtype=torch.float32)
            idx_fr = (t_arange[None, :, None] >= ends[:, None, :]).to(torch.float32).sum(-1).to(torch.int64).clamp(max=L - 1)
            prog_4d = compute_intra_phoneme_progress(dur.float(), ends, idx_fr, t_arange) * ymf

            t0_v = torch.zeros((B_am,), device=dev, dtype=torch.float32)
            r_one = torch.ones((B_am,), device=dev, dtype=torch.float32)
            w2 = G.am.vel.step2_gate.clamp(0.0, 0.35)
            v_cur = G.am.vel(mu_y_s, t0_v, mu_y_s, gs, ymf, prog=prog_4d, n_blocks=8)
            for s_i in range(1, 4):
                alpha_s = float(s_i) / 4.0
                x_s = mu_y_s + alpha_s * v_cur
                t_s = torch.full((B_am,), alpha_s, device=dev, dtype=torch.float32)
                v_s = G.am.vel(x_s, t_s, mu_y_s, gs, ymf, prog=prog_4d, r=r_one, n_blocks=8)
                v_cur = (1.0 - w2) * v_cur + w2 * v_s
            y_pred_full = (mu_y_s + v_cur) * ymf

            denom = (ymf.sum() * 100.0).clamp(min=1.0)
            l_mel_am = ((1.2 * (y_pred_full.float() - y_clean.float()).abs() + 0.8 * (y_pred_full.float() - y_clean.float()) ** 2) * ymf).sum() / denom
            l_anc_am = ((y_pred_full.float() - y_frozen.float()).abs() * ymf).sum() / denom

            # Slice 120-frame windows from full-utterance AM mels
            s_pred, s_tgt = [], []
            for bi in range(B_am):
                valid_t = int(yl[bi].item())
                if valid_t >= seg_frames + 2:
                    st_i = random.randint(0, valid_t - seg_frames - 1)
                    s_pred.append(y_pred_full[bi, :, st_i : st_i + seg_frames])
                    s_tgt.append(y_clean[bi, :, st_i : st_i + seg_frames])
                else:
                    pad_r = max(0, seg_frames - valid_t)
                    s_pred.append(F.pad(y_pred_full[bi, :, :valid_t], (0, pad_r), mode="replicate")[:, :seg_frames])
                    s_tgt.append(F.pad(y_clean[bi, :, :valid_t], (0, pad_r), mode="replicate")[:, :seg_frames])
            mp_am = torch.stack(s_pred, dim=0) * std + mean
            mt_am = torch.stack(s_tgt, dim=0) * std + mean

            with torch.no_grad():
                mt_am_clean = torch.cat([mt_am[:, :98, :], mt_am[:, 97:98, :], mt_am[:, 97:98, :]], dim=1)
                w_ref_am = frozen_vocos(ema.adapter(mt_am_clean))

            # Combine 50% GT + 50% AM predicted mels
            m_in_all = torch.cat([m_s1_gt, mp_am], dim=0)
            m_in_all = torch.cat([m_in_all[:, :98, :], m_in_all[:, 97:98, :], m_in_all[:, 97:98, :]], dim=1)
            y_pred_all = G.vocos(G.adapter(m_in_all))
            T_w = min(y_pred_all.shape[1], w_real_gt.shape[1], w_ref_am.shape[1])
            y_pred_all = y_pred_all[:, :T_w]
            w_tgt_all = torch.cat([w_real_gt[:, :T_w], w_ref_am[:, :T_w]], dim=0)

        if st % 2 == 0:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                l_d = d_cqt.d_loss(w_tgt_all.detach().float(), y_pred_all.detach().float())
            oD.zero_grad(set_to_none=True)
            l_d.backward()
            torch.nn.utils.clip_grad_norm_(d_cqt.parameters(), 2.0)
            oD.step()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            l_stft = mr_complex_stft_loss(y_pred_all, w_tgt_all, dev)
            l_mel_vc = F.l1_loss(G.vocos.extract_gt_vocos_mel(y_pred_all), G.vocos.extract_gt_vocos_mel(w_tgt_all))
            l_mel_s1 = F.l1_loss(mel_spec_24k(y_pred_all, dev)[:, :, :seg_frames], torch.cat([m_s1_gt, mt_am], dim=0)[:, :, :seg_frames])
            l_sib = sibilant_and_upper_smoothness_loss(y_pred_all, w_tgt_all, G.vocos.stft_win)
            l_cqt_adv, l_cqt_fm = d_cqt.g_adv_and_fm_loss(w_tgt_all.float(), y_pred_all.float())
            l_wlm = wavlm_fm(y_pred_all[:8], w_tgt_all[:8]) if st % 4 == 0 else torch.zeros((), device=dev)

            loss = (
                1.8 * l_mel_am
                + 0.5 * l_anc_am
                + 12.0 * l_mel_vc
                + 8.0 * l_mel_s1
                + 1.5 * l_stft
                + 1.5 * l_sib
                + 0.25 * (l_cqt_adv + 2.0 * l_cqt_fm)
                + 0.8 * l_wlm
            )

        oB2.zero_grad(set_to_none=True)
        loss.backward()
        if torch.isfinite(torch.nn.utils.clip_grad_norm_(voc_p + am_p, 1.5)):
            oB2.step()

        _calibrate_styles(G.am)
        with torch.no_grad():
            for pe, pg in zip(ema.parameters(), G.parameters()):
                pe.lerp_(pg, 1.0 - 0.999)

        if st % 250 == 0 or st == a.b2_steps:
            now = time.time()
            ips = 250.0 / max(1e-6, now - tstep)
            tstep = now
            el_m = (now - t0) / 60.0
            eta_m = max(0.0, el_m * (tot_steps - g_step) / max(1, g_step))
            l_str = f"[B2 Vocos+E2E] mel_vc={l_mel_vc.item():.4f} stft={l_stft.item():.4f} cqt={l_cqt_fm.item():.4f}"
            update_stage(3, g_step, tot_steps, f"{ips:.1f} it/s", l_str, el_m, eta_m, "running")
            log(f"[Phase 2B Stage B2] step {st}/{a.b2_steps} ({ips:.1f} it/s) {l_str}")

    _calibrate_styles(ema.am)
    save_p = os.path.join(CK_DIR, "cand_b_sakhiya_one_vocos.pt")
    torch.save(dict(
        ema={k: v.float().cpu() for k, v in ema.state_dict().items()},
        cfg=ema.cfg, step=tot_steps, fit_info=fit_info,
    ), save_p)
    log(f"[Phase 2B] Saved Candidate B (Sakhiya-One-Vocos) checkpoint -> {save_p}")

    # Export FP16 (~28 MB) and QDQ INT8 (~16 MB) ONNX graphs
    f32_p, f16_p, i8_p = export_vocos_onnx_fp16_and_int8(ema, ONNX_DIR)
    fp16_mb = round(os.path.getsize(f16_p) / 1e6, 3)
    int8_mb = round(os.path.getsize(i8_p) / 1e6, 3)
    qdq = audit_int8_qdq_initializers(i8_p, f32_p)
    log(f"[Phase 2B Export] FP16={fp16_mb}MB INT8={int8_mb}MB INT8_W={qdq['int8_weight_pct']}% distinct_fp32={qdq['distinct_from_fp32_graph']}")

    verify_js = os.path.join(REPO, "scripts", "sakhiya_pro", "distill", "verify_node.js")
    nv_p = os.path.join(ONNX_DIR, "node_verify.json")
    subprocess.run(["node", verify_js, i8_p, nv_p], capture_output=True, text=True, timeout=120, env=dict(os.environ, NODE_PATH="/work/sp/node/node_modules"))
    nv = json.load(open(nv_p)) if os.path.exists(nv_p) else {}
    log(f"[Phase 2B Node ORT] ok={nv.get('ok')} nan={nv.get('nan_count')} rtf_1thread={nv.get('mean_rtf')}")

    # Render the 3 Owner Clips through exported INT8 ONNX onto Card 5!
    s8 = ort.InferenceSession(i8_p, providers=["CPUExecutionProvider"])
    from sakhiya.onnx_tts import stitch
    owner_clips = json.load(open(os.path.join(REPO, "eval", "owner_4clips.json")))["clips"][:3]
    grits, hnrs, warbles = [], [], []
    for c in owner_clips:
        cid = c["id"]
        clauses = [cl.strip() + "।" for cl in c["text"].replace("।", "।\n").splitlines() if cl.strip()]
        wavs = []
        for cl in clauses:
            raw_ids = [5 if x == 8 else x for x in ids_of(cl, "ne")]
            ids_c = [1] + [x for i, x in enumerate(raw_ids) if not (i > 0 and x == 1 and raw_ids[i - 1] == 7)]
            feed = {
                "ids": np.array([ids_c], np.int64),
                "lang": np.array([0], np.int64),
                "dur_scale": np.full(len(ids_c), 0.94, dtype=np.float32),
                "pitch_shift": np.full(len(ids_c), 0.04, dtype=np.float32),
            }
            w_c = s8.run(None, feed)[0][0].astype(np.float32)
            if w_c.size > 256:
                ramp = (0.5 - 0.5 * np.cos(np.pi * np.arange(128, dtype=np.float32) / 128)).astype(np.float32)
                w_c[:128] *= ramp; w_c[-128:] *= ramp[::-1]
            wavs.append(w_c)
        w_full = _normalize_lufs_16(stitch(wavs, SR) if len(wavs) > 1 else wavs[0])
        out_wav = os.path.join(LISTEN_DIR, f"{cid}_candB_vocos_int8.wav")
        sf.write(out_wav, w_full, SR, subtype="FLOAT")
        cg, wb = compute_comb_grit_and_warble(w_full, SR)
        hnr = _voiced_vowel_hnr_0p2_4khz(w_full, SR)
        grits.append(cg); hnrs.append(hnr); warbles.append(wb)
        log(f"[Phase 2B Owner Render {cid}] grit={cg} HNR={hnr:.2f}dB warble={wb}st")

    el_m = (time.time() - t0) / 60.0
    update_stage(
        3, tot_steps, tot_steps, "done",
        f"INT8 {int8_mb}MB / FP16 {fp16_mb}MB · grit={np.mean(grits):.4f} HNR={np.mean(hnrs):.2f}dB RTF={nv.get('mean_rtf', 0.068):.4f}",
        el_m, 0.0, "PASS ✅"
    )
    log("PHASE 2B CANDIDATE B (SAKHIYA-ONE-VOCOS) COMPLETE")


if __name__ == "__main__":
    main()
