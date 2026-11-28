"""Single-Graph Analytical ONNX Exporter (`FP32`, `FP16`, and Selective Transformer-Only `INT8`) & Parity Verifier for Sakhiya-Voca v2."""

from __future__ import annotations

import math
from pathlib import Path
import time

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ALL_PULSES, NO_PULSES, NepaliPolish, SakhiyaVoca


class ExportableSakhiyaVoca(nn.Module):
    """Single-graph ONNX-exportable wrapper for `SakhiyaVoca` using compact runtime-constructed DFT kernels (<4 KB storage)."""
    def __init__(self, model: SakhiyaVoca, polish: NepaliPolish = NepaliPolish()):
        super().__init__()
        self.model = model
        self.polish = polish
        c = model.config
        v = model.vocoder

        causal = list(range(c.cep_causal))
        anti = [c.synth_fft - 1 - i for i in range(c.cep_anti)]
        self.register_buffer("harm_pos", torch.tensor(causal + anti, dtype=torch.float32), persistent=False)
        self.register_buffer("noise_pos", torch.arange(c.noise_cep, dtype=torch.float32), persistent=False)

        for name, win in (("synth", v.synth_window), ("noise", v.noise_window)):
            fft = len(win)
            bins = fft // 2 + 1
            mult = torch.full((bins, 1, 1), 2.0 / fft, dtype=torch.float32)
            mult[0, 0, 0] = 1.0 / fft
            mult[bins - 1, 0, 0] = 1.0 / fft
            self.register_buffer(f"{name}_win", win.float().contiguous(), persistent=False)
            self.register_buffer(f"{name}_mult", mult, persistent=False)
            self.register_buffer(f"{name}_env", (win.double().square()).float()[None, None].contiguous(), persistent=False)

        self.register_buffer("ramp", torch.arange(c.hop, dtype=torch.float32) / c.hop, persistent=False)
        self.phase_steps = 1 << 24

        chosen = torch.ones(c.cep_causal, dtype=torch.float32)
        chosen[polish.sharpen_from:polish.sharpen_to] = polish.sharpen
        mirror = torch.ones(c.cep_anti, dtype=torch.float32)
        self.register_buffer("sharpen_vec", torch.cat((chosen, mirror)), persistent=False)

        k = polish.pitch_smooth
        tri = torch.cat((torch.arange(1, k + 2), torch.arange(k, 0, -1))).float()
        self.register_buffer("pitch_kernel", (tri / tri.sum())[None, None], persistent=False)
        self.register_buffer("rumble_cut", torch.sigmoid((v.synth_frequency - 90.0) / 12.0)[None, :, None], persistent=False)

    def _cep_basis(self, pos: torch.Tensor, fft: int, zero: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bins = fft // 2 + 1
        k = torch.arange(bins, device=pos.device, dtype=torch.float32)[None, :]
        ang = (-2.0 * math.pi / fft) * ((pos[:, None] + zero) * k)
        return torch.cos(ang), torch.sin(ang)

    def _dft_kernels(self, name: str, fft: int, zero: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bins = fft // 2 + 1
        win = getattr(self, f"{name}_win")[None, :]
        mult = getattr(self, f"{name}_mult")
        n = torch.arange(fft, device=win.device, dtype=torch.float32)[None, :] + zero
        k = torch.arange(bins, device=win.device, dtype=torch.float32)[:, None]
        ang = (-2.0 * math.pi / fft) * (k * n)
        a_re = (torch.cos(ang) * win)[:, None, :]
        a_im = (torch.sin(ang) * win)[:, None, :]
        syn = torch.cat((a_re * mult, a_im * mult), dim=0)
        return a_re, a_im, syn

    def _analysis(self, source: torch.Tensor, ana_re: torch.Tensor, ana_im: torch.Tensor, fft: int, hop: int) -> tuple[torch.Tensor, torch.Tensor]:
        padded = F.pad(source[:, None], (fft // 2, fft))
        return (
            F.conv1d(padded, ana_re, stride=hop),
            F.conv1d(padded, ana_im, stride=hop),
        )

    def _synthesis(
        self,
        s_re: torch.Tensor,
        s_im: torch.Tensor,
        h_re: torch.Tensor,
        h_im: torch.Tensor,
        syn_k: torch.Tensor,
        name: str,
        fft: int,
        hop: int,
        samples: torch.Tensor | int,
    ) -> torch.Tensor:
        t_frames = s_re.shape[-1]
        h_re_t = h_re[:, :t_frames].transpose(1, 2)
        h_im_t = h_im[:, :t_frames].transpose(1, 2)
        prod = torch.cat((s_re * h_re_t - s_im * h_im_t, s_re * h_im_t + s_im * h_re_t), dim=1)
        wave = F.conv_transpose1d(prod, syn_k, stride=hop)
        env = F.conv_transpose1d(torch.ones_like(s_re[:, :1]), getattr(self, f"{name}_env"), stride=hop)
        return (wave / env)[:, 0, fft // 2:fft // 2 + samples]

    def _pulses(self, f0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.model.config
        knots = torch.log(f0.clamp(c.f0_min, c.f0_max))
        knots = torch.cat((knots, knots[:, -1:]), dim=1)
        low, high = knots[:, :-1, None], knots[:, 1:, None]
        freq = torch.exp(low * (1.0 - self.ramp) + high * self.ramp)
        within = torch.cumsum(freq / c.sample_rate, dim=2)
        whole_fr = within[:, :, -1]
        steps = torch.round((whole_fr - torch.floor(whole_fr)) * self.phase_steps).long()
        start = torch.remainder(torch.cumsum(steps, dim=1) - steps, self.phase_steps).to(within.dtype) / self.phase_steps
        cycles = within + start[:, :, None]
        theta = (2.0 * math.pi * (cycles - torch.floor(cycles))).reshape(1, -1)
        freq = freq.reshape(1, -1)
        harmonics = (c.harmonic_limit * c.sample_rate / freq).clamp_min(1.0)
        whole = torch.floor(harmonics)
        denom = 2.0 * torch.sin(theta / 2.0)
        tiny = denom.abs() < 1e-6
        core = torch.where(tiny, whole, torch.sin((whole + 0.5) * theta) / torch.where(tiny, torch.ones_like(denom), denom) - 0.5)
        wave = (core + (harmonics - whole) * torch.cos((whole + 1.0) * theta)) * torch.sqrt(2.0 / harmonics)
        return wave, theta

    def forward(
        self,
        tokens: torch.Tensor,
        stress: torch.Tensor,
        voice: torch.Tensor,
        speed: torch.Tensor,
        pitch_shift: torch.Tensor,
        pitch_range: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.model
        c = m.config
        v = m.vocoder
        p = self.polish

        tok_mask = torch.ones_like(tokens, dtype=torch.bool)
        x = m.tokens(tokens) + m.stress(stress) + m.voices(voice)[:, None]
        if m.styles is not None:
            x = x + m.styles(torch.zeros_like(voice))[:, None]
        hidden = m.encoder(x, tok_mask)

        out_dur = m.duration(hidden, tok_mask)
        frames = torch.round((torch.exp(out_dur[..., 0].clamp(0.0, math.log1p(400.0))) - 1.0) / speed).long().clamp_min(1)
        present = m.phone_tokens[tokens] | (out_dur[..., 1] > 0.0)
        durations = torch.where(present, frames, torch.zeros_like(frames))

        ends = durations.cumsum(1)
        total = ends[0, -1].clamp_min(1)
        frame = torch.arange(total, device=tokens.device)[None, :]
        idx = (ends[:, None, :] <= frame[:, :, None]).sum(-1).clamp(max=tokens.shape[1] - 1)
        owned = durations.gather(1, idx).clamp_min(1)
        offset = frame - (ends.gather(1, idx) - durations.gather(1, idx))
        rel = (offset.to(hidden.dtype) + 0.5) / owned.to(hidden.dtype)
        feat = torch.stack((rel, 1.0 - rel, torch.log1p(owned.to(hidden.dtype)) / 4.0), dim=-1)
        expanded = hidden.gather(1, idx[..., None].expand(-1, -1, hidden.shape[-1]))
        expanded = expanded + m.frame_position(feat) + m.frame_voices(voice)[:, None]
        if m.frame_styles is not None:
            expanded = expanded + m.frame_styles(torch.zeros_like(voice))[:, None]

        frm_mask = torch.ones_like(frame, dtype=torch.bool)
        pros_h = m.prosody_stack(expanded, frm_mask)
        pros = m.prosody_head(pros_h)

        pitch = pros[..., 0]
        if p.pitch_smooth > 0:
            k = p.pitch_smooth
            pitch = F.conv1d(F.pad(pitch[:, None], (k, k), mode="replicate"), self.pitch_kernel)[:, 0]
        frm_tokens = tokens.gather(1, idx)
        vowel = m.vowel_tokens[frm_tokens]
        voicing = torch.where(vowel, pros[..., 1].clamp_min(2.0), pros[..., 1])
        voiced_bool = voicing > 0.0
        voiced = voiced_bool.to(pros.dtype)
        v_mean = (pitch * voiced).sum(1, keepdim=True) / voiced.sum(1, keepdim=True).clamp_min(1.0)
        lim = math.log(p.octave_fix)
        pitch = v_mean + (pitch - v_mean).clamp(-lim, lim)
        pitch = v_mean + (pitch - v_mean) * pitch_range
        cond = torch.stack((pitch.clamp(-2.0, 1.5), voiced, pros[..., 2].clamp(-4.5, 3.0)), dim=-1)
        acoustic = m.acoustic_stack(pros_h + m.prosody_projection(cond), frm_mask)
        f0 = (c.pitch_reference_hz * torch.exp(pitch + pitch_shift)).clamp(c.f0_min, c.f0_max)

        # Harmonic cepstral envelope (2 sub-frames per frame)
        last = total - 1
        h = v.harmonic(acoustic)
        split = c.cep_causal + c.cep_anti
        steps_h = v.harmonic_steps
        sub_h = torch.minimum(torch.arange(steps_h * total + 3, device=acoustic.device), steps_h * total - 1)
        here_h = torch.div(sub_h, steps_h, rounding_mode="floor")
        next_h = torch.minimum(torch.div(sub_h + 1, steps_h, rounding_mode="floor"), last)
        env_h = 0.5 * (h[..., :split].index_select(1, here_h) + h[..., :split].index_select(1, next_h)) * v.harmonic_lifter * self.sharpen_vec

        # Adaptive lifter on causal cepstra
        f0_sub = f0.index_select(1, here_h)
        t0 = (c.sample_rate / f0_sub.clamp_min(c.f0_min))[..., None]
        q_idx = torch.arange(c.cep_causal, device=acoustic.device, dtype=torch.float32)[None, None, :]
        q_phase = ((q_idx - c.adaptive_lifter * t0) / ((1.0 - c.adaptive_lifter) * t0).clamp_min(1.0)).clamp(0.0, 1.0)
        q_taper = 0.5 * (1.0 + torch.cos(math.pi * q_phase))
        q_taper = torch.cat((torch.ones_like(q_taper[..., :1]), q_taper[..., 1:], torch.ones((1, sub_h.shape[0], c.cep_anti), device=acoustic.device)), dim=-1)
        env_h = env_h * q_taper

        zero = (speed * 0.0).reshape(1, 1)
        harm_re_b, harm_im_b = self._cep_basis(self.harm_pos, c.synth_fft, zero)
        noise_re_b, noise_im_b = self._cep_basis(self.noise_pos, c.noise_fft, zero)
        s_ana_re, s_ana_im, s_syn = self._dft_kernels("synth", c.synth_fft, zero)
        n_ana_re, n_ana_im, n_syn = self._dft_kernels("noise", c.noise_fft, zero)

        gain_h = h[..., split:].flatten(1).index_select(1, sub_h)[None]
        gain_h = F.conv1d(F.pad(gain_h, (1, 1), mode="replicate"), v.gain_smooth_h)[:, 0, :, None]
        env_h = torch.cat((env_h[..., :1] + gain_h, env_h[..., 1:]), dim=-1)
        mag_h = torch.exp((env_h @ harm_re_b).clamp(-14.0, 6.0))
        ang_h = env_h @ harm_im_b
        h_re, h_im = mag_h * torch.cos(ang_h), mag_h * torch.sin(ang_h)

        # Noise cepstral envelope (4 sub-frames per frame)
        n = v.noise(acoustic)
        steps_n = v.noise_steps
        sub_n = torch.minimum(torch.arange(steps_n * total + 5, device=acoustic.device), steps_n * total - 1)
        here_n = torch.div(sub_n, steps_n, rounding_mode="floor")
        next_n = torch.minimum(torch.div(sub_n + 1, steps_n, rounding_mode="floor"), last)
        env_n = 0.5 * (n[..., :c.noise_cep].index_select(1, here_n) + n[..., :c.noise_cep].index_select(1, next_n)) * v.noise_lifter
        gain_n = n[..., c.noise_cep:].flatten(1).index_select(1, sub_n)[None]
        gain_n = F.conv1d(F.pad(gain_n, (2, 2), mode="replicate"), v.gain_smooth_n)[:, 0, :, None]
        env_n = torch.cat((env_n[..., :1] + gain_n, env_n[..., 1:]), dim=-1)
        mag_n = torch.exp((env_n @ noise_re_b).clamp(-14.0, 6.0))
        ang_n = env_n @ noise_im_b
        n_re, n_im = mag_n * torch.cos(ang_n), mag_n * torch.sin(ang_n)

        pulses, theta = self._pulses(f0)
        # Deterministic Box-Muller pseudo-Gaussian noise using Floor+Sub (ONNX opset 17 native)
        t_idx = torch.arange(pulses.shape[1], device=pulses.device, dtype=torch.float32)[None, :]
        s1 = torch.sin(t_idx * 12.9898 + 78.233) * 43758.5453
        s2 = torch.sin(t_idx * 93.9898 + 12.345) * 28461.1357
        u1 = (s1 - torch.floor(s1)).clamp(1e-6, 1.0 - 1e-6)
        u2 = s2 - torch.floor(s2)
        noise = torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * math.pi * u2)
        bs1 = torch.sin(t_idx * 41.1234 + 19.876) * 31415.9265
        bs2 = torch.sin(t_idx * 67.5432 + 54.321) * 27182.8182
        b_u1 = (bs1 - torch.floor(bs1)).clamp(1e-6, 1.0 - 1e-6)
        b_u2 = bs2 - torch.floor(bs2)
        puff = (1.0 - 0.30 * torch.cos(theta)) * (1.0 / math.sqrt(1.0 + 0.5 * 0.30 * 0.30))
        breath = torch.sqrt(-2.0 * torch.log(b_u1)) * torch.cos(2.0 * math.pi * b_u2) * puff * math.sqrt(0.5 / c.harmonic_limit)

        # Split-band harmonic + breath excitation
        kind = m.sound_kind[frm_tokens]
        mvf = torch.full_like(pitch, ALL_PULSES)
        mvf = torch.where(kind == 0, torch.full_like(mvf, NO_PULSES), mvf)
        mvf = torch.where(kind == 1, torch.full_like(mvf, p.open_limit), mvf)
        mvf = torch.where((kind == 2) | (kind == 4), torch.where(voiced_bool, torch.full_like(mvf, p.consonant_limit), torch.full_like(mvf, NO_PULSES)), mvf)
        mvf = torch.where(kind == 3, torch.full_like(mvf, NO_PULSES), mvf)

        p_re, p_im = self._analysis(pulses, s_ana_re, s_ana_im, c.synth_fft, c.synth_hop)
        b_re, b_im = self._analysis(breath, s_ana_re, s_ana_im, c.synth_fft, c.synth_hop)
        t_stft = p_re.shape[-1]
        stft_sub = torch.minimum(torch.arange(t_stft, device=acoustic.device), steps_h * total - 1)
        lim = mvf.index_select(1, torch.div(stft_sub, steps_h, rounding_mode="floor"))[:, None, :]
        freqs = v.synth_frequency[None, :, None]
        width = torch.where(lim > 2000.0, torch.full_like(lim, 1600.0), torch.full_like(lim, 400.0))
        keep = 0.5 * (1.0 + torch.cos(math.pi * ((freqs - (lim - 0.5 * width)) / width).clamp(0.0, 1.0)))
        keep = torch.where(lim <= 10.0, torch.zeros_like(keep), keep)
        other = torch.sqrt((1.0 - keep.square()).clamp_min(0.0)) * self.rumble_cut

        harm_wave = self._synthesis(p_re * keep + b_re * other, p_im * keep + b_im * other, h_re, h_im, s_syn, "synth", c.synth_fft, c.synth_hop, pulses.shape[1])
        n_s_re, n_s_im = self._analysis(noise, n_ana_re, n_ana_im, c.noise_fft, c.noise_hop)
        hiss_wave = self._synthesis(n_s_re, n_s_im, n_re, n_im, n_syn, "noise", c.noise_fft, c.noise_hop, pulses.shape[1])

        return harm_wave + hiss_wave, durations


def export_onnx_models(
    model: SakhiyaVoca,
    out_dir: str | Path,
    prefix: str = "model",
) -> dict:
    """Export `FP32`, `FP16`, and Selective Transformer-Only `INT8` ONNX graphs and run parity + 1T CPU RTF verification."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wrapper = ExportableSakhiyaVoca(model.cpu().eval()).eval()

    tokens = torch.tensor([[1, 14, 7, 36, 5, 42, 31, 9, 4, 29, 7, 39, 5, 34, 8, 3, 2]], dtype=torch.long)
    stress = torch.tensor([[0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.long)
    voice = torch.tensor([0], dtype=torch.long)
    speed = torch.tensor([1.0], dtype=torch.float32)
    pitch_shift = torch.tensor([0.0], dtype=torch.float32)
    pitch_range = torch.tensor([1.0], dtype=torch.float32)
    inputs = (tokens, stress, voice, speed, pitch_shift, pitch_range)

    fp32_path = out_dir / f"{prefix}.fp32.onnx"
    fp16_path = out_dir / f"{prefix}.fp16.onnx"
    int8_path = out_dir / f"{prefix}.int8.onnx"

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            inputs,
            str(fp32_path),
            opset_version=17,
            input_names=["tokens", "stress", "voice", "speed", "pitch_shift", "pitch_range"],
            output_names=["audio", "durations"],
            dynamic_axes={
                "tokens": {1: "N"},
                "stress": {1: "N"},
                "audio": {1: "S"},
                "durations": {1: "N"},
            },
            do_constant_folding=True,
            dynamo=False,
        )

    def _compress_float_constants(graph) -> None:
        replaced_nodes = []
        extra_inits = []
        for idx_n, node in enumerate(list(graph.node)):
            if node.op_type == "Constant" and len(node.output) == 1:
                val_attr = next((a for a in node.attribute if a.name == "value"), None)
                if val_attr is not None:
                    arr = numpy_helper.to_array(val_attr.t)
                    if arr.dtype == np.float32 and arr.size >= 256:
                        fp16_name = f"const_{idx_n}_fp16"
                        extra_inits.append(numpy_helper.from_array(arr.astype(np.float16), name=fp16_name))
                        replaced_nodes.append(helper.make_node("Cast", inputs=[fp16_name], outputs=[node.output[0]], to=TensorProto.FLOAT))
                        continue
            replaced_nodes.append(node)
        del graph.node[:]
        graph.node.extend(replaced_nodes)
        graph.initializer.extend(extra_inits)

    # 1. Build FP16 ONNX (weights stored in FP16 with Cast->FP32 for bit-exact oscillator stability)
    base_m = onnx.load(str(fp32_path))
    g16 = base_m.graph
    _compress_float_constants(g16)
    new_inits_16 = []
    cast_nodes_16 = []
    for init in list(g16.initializer):
        arr = numpy_helper.to_array(init)
        if arr.dtype == np.float32 and arr.size >= 256:
            fp16_name = init.name + "_fp16"
            new_inits_16.append(numpy_helper.from_array(arr.astype(np.float16), name=fp16_name))
            cast_nodes_16.append(helper.make_node("Cast", inputs=[fp16_name], outputs=[init.name], to=TensorProto.FLOAT))
        else:
            new_inits_16.append(init)
    del g16.initializer[:]
    g16.initializer.extend(new_inits_16)
    for n in reversed(cast_nodes_16):
        g16.node.insert(0, n)
    onnx.save(base_m, str(fp16_path))

    # 2. Build Selective Acoustic-Transformer Per-Channel INT8 + FP16 ONNX
    # Keep encoder + duration + prosody_stack in FP16 so durations (len_diff_int8 == 0) and F0 phase are bit-identical to FP16
    m8 = onnx.load(str(fp32_path))
    g8 = m8.graph
    _compress_float_constants(g8)
    cand_2d_names = [
        init.name for init in g8.initializer
        if numpy_helper.to_array(init).dtype == np.float32
        and numpy_helper.to_array(init).ndim == 2
        and min(numpy_helper.to_array(init).shape) >= 128
        and not any(k in init.name for k in ("tokens", "harmonic", "noise"))
    ]
    n_acoustic_mats = 4 * model.config.acoustic_layers
    int8_target_names = set(cand_2d_names[-n_acoustic_mats:]) if n_acoustic_mats > 0 else set()

    new_inits_8 = []
    new_nodes_8 = []
    for init in list(g8.initializer):
        arr = numpy_helper.to_array(init)
        if init.name in int8_target_names:
            max_val = np.maximum(np.max(np.abs(arr), axis=0, keepdims=True), 1e-8)
            scale = (max_val / 127.0).astype(np.float16)
            q = np.clip(np.round(arr / scale.astype(np.float32)), -127, 127).astype(np.int8)
            q_name, s_name = init.name + "_int8", init.name + "_scale16"
            qf_name, sf_name = init.name + "_qf32", init.name + "_sf32"
            new_inits_8.append(numpy_helper.from_array(q, name=q_name))
            new_inits_8.append(numpy_helper.from_array(scale, name=s_name))
            new_nodes_8.extend([
                helper.make_node("Cast", inputs=[q_name], outputs=[qf_name], to=TensorProto.FLOAT),
                helper.make_node("Cast", inputs=[s_name], outputs=[sf_name], to=TensorProto.FLOAT),
                helper.make_node("Mul", inputs=[qf_name, sf_name], outputs=[init.name]),
            ])
        elif arr.dtype == np.float32 and arr.size >= 256:
            fp16_name = init.name + "_fp16"
            new_inits_8.append(numpy_helper.from_array(arr.astype(np.float16), name=fp16_name))
            new_nodes_8.append(helper.make_node("Cast", inputs=[fp16_name], outputs=[init.name], to=TensorProto.FLOAT))
        else:
            new_inits_8.append(init)
    del g8.initializer[:]
    g8.initializer.extend(new_inits_8)
    for n in reversed(new_nodes_8):
        g8.node.insert(0, n)
    onnx.save(m8, str(int8_path))

    # 3. Verify Parity & 1-Thread CPU RTF
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess32 = ort.InferenceSession(str(fp32_path), sess_options=so, providers=["CPUExecutionProvider"])
    sess16 = ort.InferenceSession(str(fp16_path), sess_options=so, providers=["CPUExecutionProvider"])
    sess8 = ort.InferenceSession(str(int8_path), sess_options=so, providers=["CPUExecutionProvider"])

    feed = {
        "tokens": tokens.numpy(),
        "stress": stress.numpy(),
        "voice": voice.numpy(),
        "speed": speed.numpy(),
        "pitch_shift": pitch_shift.numpy(),
        "pitch_range": pitch_range.numpy(),
    }
    w32, d32 = sess32.run(None, feed)
    w16, d16 = sess16.run(None, feed)
    w8, d8 = sess8.run(None, feed)

    for _ in range(3):
        sess8.run(None, feed)
    t0 = time.perf_counter()
    iters = 10
    for _ in range(iters):
        sess8.run(None, feed)
    infer_ms = (time.perf_counter() - t0) * 1000.0 / iters
    audio_sec = max(w8.shape[-1] / 24000.0, 0.1)
    rtf_1t = (infer_ms / 1000.0) / audio_sec

    min_16 = min(w32.shape[-1], w16.shape[-1])
    min_8 = min(w16.shape[-1], w8.shape[-1])
    peak_diff_16 = float(np.max(np.abs(w32[..., :min_16] - w16[..., :min_16])) / max(float(np.max(np.abs(w32[..., :min_16]))), 1e-5))
    peak_diff_8 = float(np.max(np.abs(w16[..., :min_8] - w8[..., :min_8])) / max(float(np.max(np.abs(w16[..., :min_8]))), 1e-5))

    return {
        "fp32_mb": round(fp32_path.stat().st_size / (1024 * 1024), 3),
        "fp16_mb": round(fp16_path.stat().st_size / (1024 * 1024), 3),
        "int8_mb": round(int8_path.stat().st_size / (1024 * 1024), 3),
        "len_diff_fp16": int(abs(w32.shape[-1] - w16.shape[-1])),
        "len_diff_int8": int(abs(w16.shape[-1] - w8.shape[-1])),
        "peak_rel_diff_fp16": round(peak_diff_16, 6),
        "peak_rel_diff_int8": round(peak_diff_8, 6),
        "nan_count": int(np.isnan(w8).sum() + np.isnan(w16).sum()),
        "rtf_1t_cpu": round(rtf_1t, 5),
        "realtime_x_1t_cpu": round(1.0 / max(rtf_1t, 1e-6), 1),
    }
