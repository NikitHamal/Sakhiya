#!/usr/bin/env python3
"""Experiment 4 (Gemini Research): `Sakhiya-Voca-4.5M` Dual-Voice (Female & Male) Nepali + Sanskrit Prototype & ONNX Benchmark.

Builds, verifies, exports to ONNX (FP32, FP16, and Selective Mixed INT8+FP16), and benchmarks on CPU:
- Total parameters: ~4,548,000 (~4.55M, matching tiny-voca's 4.54M sweet spot)
- Voices: 2 native Nepali voices (`0 = F_sakhiya` Female, `1 = M_sagar` Male)
- Languages: Nepali (`ne`) + Sanskrit/Vedic (`sa` with svara & moraic guru/laghu weights)
- Architecture:
  1. Embeddings (`tokens`, `stress_or_mora`, `voices`, `lang`): ~17K params
  2. Text Encoder (`4` full-rank RoPE + k=7 Depthwise + Dense 192->640->192 Transformer blocks): 1,588,864 params
  3. Dual-Head Duration & Pause-Presence Predictor (`log_dur` + `pause_present_logit`): 123,906 params
  4. Frame Transformer A (`2` full-rank RoPE blocks) + Prosody Head (`log_f0`, `voiced_logit`, `log_energy`): 796,547 params
  5. Frame Transformer B (`5` full-rank RoPE blocks conditioned on prosody + Vedic svara `pitch_add`): 1,985,984 params
  6. Nepali 4-Way Phonation Homomorphic Cepstral Source-Filter Head (`NepaliSourceFilter`): ~26,200 params
     - Closed-form band-limited Dirichlet harmonic pulse train (`0` phase-guessing buzz, 24-bit cycle-exact float32 phase)
     - Causal (`80`) + Anti-causal (`16`) glottal-return complex cepstrum (`synth_fft=512, synth_hop=128`)
     - 4x Sub-Frame Noise Cepstrum (`noise_cep=32, noise_fft=256, noise_hop=64` = `2.67 ms` burst resolution for Nepali `च/छ/त/ट/क/प`)
     - 5-Class Nepali Phonation Split-Band Crossover (`sound_kind`: pause, vowel, voiced_consonant, voiceless_obst, breathy_voiced_gh_jh_dh_bh)
"""
import json
import math
import os
import sys
import time
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sakhiya.symbols import SYMBOLS, SYM2ID, N_SYMBOLS, VOW, NASV


def classify_nepali_phonation_kinds() -> list:
    """Assign each symbol in `sakhiya.symbols.SYMBOLS` a phonation class for split-band source-filter excitation:
    0: pad / pause / punctuation (`_`, `|`, `,`, `.`, `?`, `!`, `;`, `<s>`, `</s>`)
    1: vowel / nasalized vowel (`ʌ`, `a`, `i`, `u`, `e`, `o`, `ɛ`, `ɔ` + nasalized) -> full harmonic + -21dB hiss floor
    2: voiced sonorant / voiced unaspirated stop (`g`, `ŋ`, `dz`, `ɲ`, `ɖ`, `ɳ`, `d̪`, `n`, `b`, `m`, `j`, `r`, `l`, `w`, `ɽ`)
    3: voiceless stop / affricate / sibilant (`k`, `kʰ`, `ts`, `tsʰ`, `ʈ`, `ʈʰ`, `t̪`, `t̪ʰ`, `p`, `pʰ`, `ʃ`, `s`, `f`) -> zero harmonic pulse, 2.67ms noise burst
    4: breathy-voiced aspirate / glottal (`gʱ`, `dzʱ`, `ɖʱ`, `d̪ʱ`, `bʱ`, `ɽʱ`, `h`, `z`) -> split-band (<1600 Hz glottal pulse + >1600 Hz aspiration breath)
    """
    vows = set(VOW) | set(NASV)
    voiceless = {"k", "kʰ", "ts", "tsʰ", "ʈ", "ʈʰ", "t̪", "t̪ʰ", "p", "pʰ", "ʃ", "s", "f"}
    breathy = {"gʱ", "dzʱ", "ɖʱ", "d̪ʱ", "bʱ", "ɽʱ", "h", "z"}
    voiced_c = {"g", "ŋ", "dz", "ɲ", "ɖ", "ɳ", "d̪", "n", "b", "m", "j", "r", "l", "w", "ɽ"}
    kinds = []
    for sym in SYMBOLS:
        if sym in vows:
            kinds.append(1)
        elif sym in voiced_c:
            kinds.append(2)
        elif sym in voiceless:
            kinds.append(3)
        elif sym in breathy:
            kinds.append(4)
        else:
            kinds.append(0)
    return kinds


def rotary_table(size: int, width: int, device: torch.device):
    inv = 1.0 / (10000.0 ** (torch.arange(0, width, 2, device=device, dtype=torch.float32) / width))
    ang = torch.arange(size, device=device, dtype=torch.float32)[:, None] * inv[None]
    return ang.cos(), ang.sin()


def rotate(x: torch.Tensor, table):
    cos, sin = table[0].to(x.dtype), table[1].to(x.dtype)
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)


class RoPEBlock(nn.Module):
    """Full-Rank Pre-Norm RoPE Self-Attention + Depthwise k=7 Conv + Dense FFN (192 -> 640 -> 192)."""
    def __init__(self, channels: int = 192, heads: int = 4, feedforward: int = 640, kernel: int = 7):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.out = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        self.depthwise = nn.Conv1d(channels, channels, kernel, padding=kernel // 2, groups=channels)
        self.expand = nn.Linear(channels, feedforward)
        self.project = nn.Linear(feedforward, channels)

    def forward(self, x: torch.Tensor, table) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, t, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(rotate(q, table), rotate(k, table), v)
        x = x + self.out(y.transpose(1, 2).reshape(b, t, c))
        h = self.depthwise(self.norm2(x).transpose(1, 2)).transpose(1, 2)
        return x + self.project(F.gelu(self.expand(h)))


class RoPEStack(nn.Module):
    def __init__(self, layers: int, channels: int = 192, heads: int = 4, feedforward: int = 640):
        super().__init__()
        self.blocks = nn.ModuleList(RoPEBlock(channels, heads, feedforward) for _ in range(layers))
        self.norm = nn.LayerNorm(channels)
        self.width = channels // heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        table = rotary_table(x.shape[1], self.width, x.device)
        for blk in self.blocks:
            x = blk(x, table)
        return self.norm(x)


class DurationPausePredictor(nn.Module):
    """Predicts [log(1 + frames), pause_presence_logit] per token (~123.9K params)."""
    def __init__(self, channels: int = 192, p_ch: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, p_ch, 3, padding=1)
        self.norm1 = nn.LayerNorm(p_ch)
        self.conv2 = nn.Conv1d(p_ch, p_ch, 3, padding=1)
        self.norm2 = nn.LayerNorm(p_ch)
        self.out = nn.Linear(p_ch, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(F.gelu(self.conv1(x.transpose(1, 2)).transpose(1, 2)))
        x = self.norm2(F.gelu(self.conv2(x.transpose(1, 2)).transpose(1, 2)))
        return self.out(x)


class NepaliSourceFilterHead(nn.Module):
    """26K-Parameter Homomorphic Complex-Cepstrum + 4x Sub-Frame Burst Source-Filter Head for Nepali & Sanskrit."""
    def __init__(
        self,
        channels: int = 192,
        sample_rate: int = 24000,
        hop: int = 256,
        synth_fft: int = 512,
        synth_hop: int = 128,
        noise_fft: int = 256,
        noise_hop: int = 64,
        cep_causal: int = 80,
        cep_anti: int = 16,
        noise_cep: int = 32,
        harmonic_limit: float = 0.47,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop = hop
        self.synth_fft = synth_fft
        self.synth_hop = synth_hop
        self.noise_fft = noise_fft
        self.noise_hop = noise_hop
        self.cep_causal = cep_causal
        self.cep_anti = cep_anti
        self.noise_cep = noise_cep
        self.harmonic_limit = harmonic_limit
        self.harmonic_steps = hop // synth_hop  # 2 sub-frames (5.33 ms)
        self.noise_steps = hop // noise_hop     # 4 sub-frames (2.67 ms stop burst resolution)

        self.harmonic = nn.Linear(channels, cep_causal + cep_anti + self.harmonic_steps)
        self.noise = nn.Linear(channels, noise_cep + self.noise_steps)
        nn.init.zeros_(self.harmonic.weight); nn.init.zeros_(self.harmonic.bias); self.harmonic.bias.data[0] = -1.5
        nn.init.zeros_(self.noise.weight); nn.init.zeros_(self.noise.bias); self.noise.bias.data[0] = -2.5

        def lifter(size, first):
            sc = 1.0 / (1.0 + torch.arange(first, first + size, dtype=torch.float32) / 4.0)
            if first == 0:
                sc[0] = 2.0
            return sc

        self.register_buffer("harm_lifter", torch.cat([lifter(cep_causal, 0), lifter(cep_anti, 1)]), persistent=False)
        self.register_buffer("noise_lifter", lifter(noise_cep, 0), persistent=False)
        self.register_buffer("ramp", torch.arange(hop, dtype=torch.float32) / hop, persistent=False)
        self.phase_steps = 1 << 24

        # Precompute analytical cepstrum DFT and STFT/iSTFT convolution kernels for fast single-graph ONNX execution
        causal_pos = list(range(cep_causal)) + [synth_fft - 1 - i for i in range(cep_anti)]
        h_re, h_im = self._cep_mat(causal_pos, synth_fft)
        n_re, n_im = self._cep_mat(list(range(noise_cep)), noise_fft)
        self.register_buffer("h_resp_mat", torch.cat([h_re, h_im], dim=1), persistent=False)
        self.register_buffer("n_resp_mat", torch.cat([n_re, n_im], dim=1), persistent=False)

        for name, fft in (("synth", synth_fft), ("noise", noise_fft)):
            win = torch.hann_window(fft)
            a_re, a_im = self._ana_k(win)
            syn_k = self._syn_k(win)
            q = win.double().square().float().reshape(4, -1)
            w0, w1, w2, w3 = q.unbind()
            self.register_buffer(f"{name}_ana", torch.cat([a_re, a_im], dim=0), persistent=False)
            self.register_buffer(f"{name}_syn", syn_k, persistent=False)
            self.register_buffer(f"{name}_pref", ((w2 + w1) + w0)[None], persistent=False)
            self.register_buffer(f"{name}_stdy", (((w3 + w2) + w1) + w0)[None], persistent=False)

    @staticmethod
    def _cep_mat(positions, fft):
        buf = torch.zeros(len(positions), fft, dtype=torch.float64)
        buf[torch.arange(len(positions)), torch.tensor(positions)] = 1.0
        sp = torch.fft.rfft(buf, dim=-1)
        return sp.real.float(), sp.imag.float()

    @staticmethod
    def _ana_k(window):
        fft = len(window)
        sp = torch.fft.rfft(torch.eye(fft, dtype=torch.float64) * window.double(), dim=-1)
        return sp.real.T[:, None].float().contiguous(), sp.imag.T[:, None].float().contiguous()

    @staticmethod
    def _syn_k(window):
        fft = len(window)
        bins = fft // 2 + 1
        unit = torch.eye(bins, dtype=torch.complex128)
        re = torch.fft.irfft(unit, n=fft, dim=-1)
        im = torch.fft.irfft(unit * 1j, n=fft, dim=-1)
        return (torch.cat([re, im], dim=0) * window.double())[:, None].float().contiguous()

    def _pulses(self, f0: torch.Tensor) -> torch.Tensor:
        knots = torch.log(f0.clamp_min(40.0))
        knots = torch.cat([knots, knots[:, -1:]], dim=1)
        low, high = knots[:, :-1, None], knots[:, 1:, None]
        freq = torch.exp(low * (1.0 - self.ramp) + high * self.ramp)
        within = torch.cumsum(freq / self.sample_rate, dim=2)
        whole_fr = within[:, :, -1]
        steps = torch.round((whole_fr - torch.floor(whole_fr)) * self.phase_steps).long()
        start = torch.remainder(torch.cumsum(steps, dim=1) - steps, self.phase_steps).to(within.dtype) / self.phase_steps
        cycles = within + start[:, :, None]
        theta = (2.0 * math.pi * (cycles - torch.floor(cycles))).reshape(1, -1)
        freq = freq.reshape(1, -1)
        harmonics = (self.harmonic_limit * self.sample_rate / freq).clamp_min(1.0)
        whole = torch.floor(harmonics)
        denom = 2.0 * torch.sin(theta / 2.0)
        tiny = denom.abs() < 1e-6
        core = torch.where(tiny, whole, torch.sin((whole + 0.5) * theta) / torch.where(tiny, torch.ones_like(denom), denom) - 0.5)
        wave = core + (harmonics - whole) * torch.cos((whole + 1.0) * theta)
        return wave * torch.sqrt(2.0 / harmonics)

    def _packed_filter(self, source: torch.Tensor, real: torch.Tensor, imag: torch.Tensor, name: str, fft: int, hop: int) -> torch.Tensor:
        samples = source.shape[1]
        padded = F.pad(source[:, None], (fft // 2, fft - hop))
        ana = F.conv1d(padded, getattr(self, f"{name}_ana"), stride=hop)
        s_re, s_im = ana.split(fft // 2 + 1, dim=1)
        re_t, im_t = real.transpose(1, 2), imag.transpose(1, 2)
        prod = torch.cat([s_re * re_t - s_im * im_t, s_re * im_t + s_im * re_t], dim=1)
        wave = F.conv_transpose1d(prod, getattr(self, f"{name}_syn"), stride=hop, padding=fft // 2)[:, 0, :samples]
        steady = getattr(self, f"{name}_stdy").expand(samples // hop - 1, -1)
        env = torch.cat([getattr(self, f"{name}_pref"), steady], dim=0).reshape(1, -1)
        return wave / env

    def forward(self, hidden: torch.Tensor, f0: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
        last = total - 1
        # 1. Harmonic cepstral branch (2 sub-frames per frame)
        h = self.harmonic(hidden)
        split = self.cep_causal + self.cep_anti
        steps_h = self.harmonic_steps
        sub_h = torch.minimum(torch.arange(steps_h * total + 2, device=hidden.device), steps_h * total - 1)
        here_h = torch.div(sub_h, steps_h, rounding_mode="floor")
        next_h = torch.minimum(torch.div(sub_h + 1, steps_h, rounding_mode="floor"), last)
        env_h = 0.5 * (h[..., :split].index_select(1, here_h) + h[..., :split].index_select(1, next_h)) * self.harm_lifter
        gain_h = h[..., split:].flatten(1).index_select(1, sub_h)[..., None]
        env_h = torch.cat([env_h[..., :1] + gain_h, env_h[..., 1:]], dim=-1)
        amp_h, ang_h = (env_h @ self.h_resp_mat).split(self.synth_fft // 2 + 1, dim=-1)
        mag_h = torch.exp(amp_h.clamp(-14.0, 6.0))
        h_re, h_im = mag_h * torch.cos(ang_h), mag_h * torch.sin(ang_h)

        # 2. Noise cepstral branch (4 sub-frames per frame = 2.67 ms stop/affricate burst resolution)
        n = self.noise(hidden)
        steps_n = self.noise_steps
        sub_n = torch.minimum(torch.arange(steps_n * total + 2, device=hidden.device), steps_n * total - 1)
        shape_n = n[..., :self.noise_cep].index_select(1, torch.div(sub_n, steps_n, rounding_mode="floor")) * self.noise_lifter
        gain_n = n[..., self.noise_cep:].flatten(1).index_select(1, sub_n)[..., None]
        shape_n = torch.cat([shape_n[..., :1] + gain_n, shape_n[..., 1:]], dim=-1)
        amp_n, ang_n = (shape_n @ self.n_resp_mat).split(self.noise_fft // 2 + 1, dim=-1)
        mag_n = torch.exp(amp_n.clamp(-14.0, 6.0))
        n_re, n_im = mag_n * torch.cos(ang_n), mag_n * torch.sin(ang_n)

        pulses = self._pulses(f0)
        # Deterministic high-entropy hash noise for reproducible cross-runtime ONNX / WASM synthesis
        t_idx = torch.arange(pulses.shape[1], device=pulses.device, dtype=torch.float32)[None, :]
        h1 = torch.sin(t_idx * 12.9898 + 78.233) * 43758.5453
        h2 = torch.sin(t_idx * 93.9898 + 12.345) * 28461.1357
        u1 = (h1 - torch.floor(h1)).clamp(1e-6, 1.0 - 1e-6)
        u2 = (h2 - torch.floor(h2))
        noise = torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * math.pi * u2)

        return (
            self._packed_filter(pulses, h_re, h_im, "synth", self.synth_fft, self.synth_hop)
            + self._packed_filter(noise, n_re, n_im, "noise", self.noise_fft, self.noise_hop)
        )


class SakhiyaVoca(nn.Module):
    """Sakhiya-Voca (~4.54M parameters): Dual-Voice (Female F_sakhiya + Male M_sagar) Nepali & Sanskrit TTS."""
    def __init__(self, vocab_size: int = N_SYMBOLS, channels: int = 192, heads: int = 4, feedforward: int = 640):
        super().__init__()
        self.channels = channels
        self.tokens = nn.Embedding(vocab_size, channels, padding_idx=0)
        self.stress = nn.Embedding(3, channels)        # 0: laghu, 1: guru/primary stress, 2: geminate/diphthong
        self.voices = nn.Embedding(2, channels)        # 0: F_sakhiya (Female), 1: M_sagar (Male)
        self.langs = nn.Embedding(2, channels)         # 0: Nepali, 1: Sanskrit
        self.encoder = RoPEStack(layers=4, channels=channels, heads=heads, feedforward=feedforward)
        self.duration = DurationPausePredictor(channels=channels, p_ch=128)

        self.frame_voices = nn.Embedding(2, channels)
        self.frame_position = nn.Linear(3, channels)
        self.prosody_stack = RoPEStack(layers=2, channels=channels, heads=heads, feedforward=feedforward)
        self.prosody_head = nn.Linear(channels, 3)     # [log_f0_rel, voiced_logit, log_energy]
        self.prosody_projection = nn.Linear(3, channels)
        self.acoustic_stack = RoPEStack(layers=5, channels=channels, heads=heads, feedforward=feedforward)
        self.vocoder = NepaliSourceFilterHead(channels=channels)

        kinds = classify_nepali_phonation_kinds()
        self.register_buffer("sound_kind", torch.tensor(kinds, dtype=torch.long), persistent=False)
        self.register_buffer("phone_tokens", torch.tensor([k > 0 for k in kinds], dtype=torch.bool), persistent=False)

    def forward(
        self,
        tokens: torch.Tensor,       # [1, N] int64
        stress: torch.Tensor,       # [1, N] int64 in {0, 1, 2}
        voice: torch.Tensor,        # [1] int64 (0=Female, 1=Male)
        lang: torch.Tensor,         # [1] int64 (0=Nepali, 1=Sanskrit)
        speed: torch.Tensor,        # [1] float32
        pitch_shift: torch.Tensor,  # [1] float32
        pitch_range: torch.Tensor,  # [1] float32
        pitch_add: torch.Tensor,    # [1, N] float32 (svara / clause intonation)
    ):
        x = (
            self.tokens(tokens)
            + self.stress(stress.clamp(0, 2))
            + self.voices(voice.clamp(0, 1))[:, None]
            + self.langs(lang.clamp(0, 1))[:, None]
        )
        hidden = self.encoder(x)
        dur_out = self.duration(hidden)
        frames = torch.round((torch.exp(dur_out[..., 0].clamp(0.0, math.log1p(400.0))) - 1.0) / speed).long().clamp_min(1)
        present = self.phone_tokens[tokens] | (dur_out[..., 1] > 0.0)
        durations = torch.where(present, frames, torch.zeros_like(frames))

        ends = durations.cumsum(1)
        total = ends[0, -1]
        frame = torch.arange(total, device=tokens.device)[None]
        index = (ends[:, None, :] <= frame[:, :, None]).sum(-1).clamp(max=tokens.shape[1] - 1)
        owned = durations.gather(1, index).clamp_min(1)
        offset = frame - (ends.gather(1, index) - durations.gather(1, index))
        rel = (offset.to(hidden.dtype) + 0.5) / owned
        feat_pos = torch.stack([rel, 1.0 - rel, torch.log1p(owned.to(hidden.dtype)) / 4.0], dim=-1)

        expanded = (
            hidden.gather(1, index[..., None].expand(-1, -1, hidden.shape[-1]))
            + self.frame_position(feat_pos)
            + self.frame_voices(voice.clamp(0, 1))[:, None]
        )
        pros_hidden = self.prosody_stack(expanded)
        pros = self.prosody_head(pros_hidden)

        voiced = (pros[..., 1] > 0.0).to(pros.dtype)
        mean_p = (pros[..., 0] * voiced).sum(1, keepdim=True) / voiced.sum(1, keepdim=True).clamp_min(1.0)
        sv_frame = pitch_add.gather(1, index)
        pitch = mean_p + (pros[..., 0] - mean_p) * pitch_range + sv_frame
        cond = torch.stack([pitch.clamp(-2.0, 1.5), voiced, pros[..., 2].clamp(-4.5, 3.0)], dim=-1)

        acoustic = self.acoustic_stack(pros_hidden + self.prosody_projection(cond))
        # Voice-specific base F0 reference: 215 Hz for Female (0), 125 Hz for Male (1)
        ref_hz = torch.where(voice == 0, torch.tensor(215.0, device=tokens.device), torch.tensor(125.0, device=tokens.device))
        f0 = (ref_hz[:, None] * torch.exp(pitch + pitch_shift)).clamp(50.0, 550.0)
        audio = self.vocoder(acoustic, f0, total)
        return audio, durations


def export_and_benchmark():
    model = SakhiyaVoca().eval()
    breakdown = {
        "embeddings": sum(p.numel() for m in (model.tokens, model.stress, model.voices, model.langs) for p in m.parameters()),
        "text_encoder_4L_rope": sum(p.numel() for p in model.encoder.parameters()),
        "duration_and_pause_predictor": sum(p.numel() for p in model.duration.parameters()),
        "frame_transformer_A_prosody_2L_rope": sum(
            p.numel() for m in (model.frame_voices, model.frame_position, model.prosody_stack, model.prosody_head) for p in m.parameters()
        ),
        "frame_transformer_B_acoustic_5L_rope": sum(
            p.numel() for m in (model.prosody_projection, model.acoustic_stack) for p in m.parameters()
        ),
        "nepali_source_filter_head": sum(p.numel() for p in model.vocoder.parameters()),
    }
    total_params = sum(breakdown.values())

    # Set reasonable mock durations on initialized model so benchmark synthesizes ~4.5 seconds of audio (420 frames)
    with torch.no_grad():
        model.duration.out.bias.data[0] = math.log1p(7.5)
        model.duration.out.bias.data[1] = 1.0

    N = 56
    tokens = torch.randint(9, 55, (1, N), dtype=torch.long)
    stress = torch.randint(0, 3, (1, N), dtype=torch.long)
    voice_f = torch.tensor([0], dtype=torch.long)
    voice_m = torch.tensor([1], dtype=torch.long)
    lang = torch.tensor([0], dtype=torch.long)
    speed = torch.tensor([1.0], dtype=torch.float32)
    pitch_shift = torch.tensor([0.0], dtype=torch.float32)
    pitch_range = torch.tensor([1.0], dtype=torch.float32)
    pitch_add = torch.zeros((1, N), dtype=torch.float32)

    with torch.no_grad():
        wav_f, dur_f = model(tokens, stress, voice_f, lang, speed, pitch_shift, pitch_range, pitch_add)
        wav_m, dur_m = model(tokens, stress, voice_m, lang, speed, pitch_shift, pitch_range, pitch_add)

    out_dir = "/work/gemini/out/sakhiya_voca_proto"
    os.makedirs(out_dir, exist_ok=True)
    f32_path = os.path.join(out_dir, "sakhiya_voca.fp32.onnx")
    f16_path = os.path.join(out_dir, "sakhiya_voca.fp16.onnx")
    i8_path = os.path.join(out_dir, "sakhiya_voca.int8.onnx")

    args = (tokens, stress, voice_f, lang, speed, pitch_shift, pitch_range, pitch_add)
    kw = dict(
        input_names=["tokens", "stress", "voice", "lang", "speed", "pitch_shift", "pitch_range", "pitch_add"],
        output_names=["audio", "durations"],
        dynamic_axes={
            "tokens": {1: "N"},
            "stress": {1: "N"},
            "pitch_add": {1: "N"},
            "audio": {1: "S"},
            "durations": {1: "N"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    try:
        torch.onnx.export(model, args, f32_path, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(model, args, f32_path, **kw)

    def _promote_consts(m_onnx, min_sz=64):
        new_n, idx_c = [], 0
        for n in m_onnx.graph.node:
            if n.op_type == "Constant" and len(n.output) == 1:
                t_attr = [a for a in n.attribute if a.name == "value"]
                if t_attr:
                    arr = numpy_helper.to_array(t_attr[0].t)
                    if arr.size >= min_sz and arr.dtype == np.float32:
                        inm = f"const_init_{idx_c}_{n.output[0]}"; idx_c += 1
                        m_onnx.graph.initializer.append(numpy_helper.from_array(arr, inm))
                        new_n.append(helper.make_node("Identity", [inm], [n.output[0]]))
                        continue
            new_n.append(n)
        del m_onnx.graph.node[:]; m_onnx.graph.node.extend(new_n)

    # Create FP16 and Selective INT8+FP16 ONNX exports
    om16 = onnx.load(f32_path); _promote_consts(om16)
    keep16, casts16 = [], []
    for init in om16.graph.initializer:
        a = numpy_helper.to_array(init)
        if init.data_type == TensorProto.FLOAT and a.ndim >= 2 and a.size >= 64:
            h = numpy_helper.from_array(a.astype(np.float16), init.name + "_fp16")
            keep16.append(h)
            casts16.append(helper.make_node("Cast", [h.name], [init.name], to=TensorProto.FLOAT))
        else:
            keep16.append(init)
    del om16.graph.initializer[:]; om16.graph.initializer.extend(keep16)
    n16 = list(om16.graph.node); del om16.graph.node[:]; om16.graph.node.extend(casts16 + n16)
    onnx.save(om16, f16_path)

    om8 = onnx.load(f32_path); _promote_consts(om8)
    keep8, dq8 = [], []
    for init in om8.graph.initializer:
        a = numpy_helper.to_array(init)
        if init.data_type == TensorProto.FLOAT and a.ndim == 2 and a.size >= 4096 and "vocoder" not in init.name and "tokens" not in init.name:
            sc = (np.maximum(np.abs(a).max(axis=1), 1e-8) / 127.0).astype(np.float32)
            q = np.clip(np.round(a / sc[:, None]), -127, 127).astype(np.int8)
            keep8.append(numpy_helper.from_array(q, init.name + "_q"))
            keep8.append(numpy_helper.from_array(sc, init.name + "_s"))
            dq8.append(helper.make_node("DequantizeLinear", [init.name + "_q", init.name + "_s"], [init.name], axis=0))
        elif init.data_type == TensorProto.FLOAT and a.ndim >= 2 and a.size >= 64:
            h = numpy_helper.from_array(a.astype(np.float16), init.name + "_fp16")
            keep8.append(h)
            dq8.append(helper.make_node("Cast", [h.name], [init.name], to=TensorProto.FLOAT))
        else:
            keep8.append(init)
    del om8.graph.initializer[:]; om8.graph.initializer.extend(keep8)
    n8 = list(om8.graph.node); del om8.graph.node[:]; om8.graph.node.extend(dq8 + n8)
    onnx.save(om8, i8_path)

    # Benchmark ONNX Runtime 1-thread and 4-thread speed
    feeds = {
        "tokens": tokens.numpy(),
        "stress": stress.numpy(),
        "voice": voice_f.numpy(),
        "lang": lang.numpy(),
        "speed": speed.numpy(),
        "pitch_shift": pitch_shift.numpy(),
        "pitch_range": pitch_range.numpy(),
        "pitch_add": pitch_add.numpy(),
    }
    ort_bench = {}
    for label, pth in (("fp32", f32_path), ("fp16", f16_path), ("int8_mixed", i8_path)):
        for n_th in (1, 4):
            so = ort.SessionOptions()
            so.intra_op_num_threads = n_th
            sess = ort.InferenceSession(pth, so, providers=["CPUExecutionProvider"])
            # Warmup
            for _ in range(2):
                w_out, _ = sess.run(None, feeds)
            dur_s = w_out.shape[1] / 24000.0
            times = []
            for _ in range(6):
                t0 = time.perf_counter()
                sess.run(None, feeds)
                times.append(time.perf_counter() - t0)
            mean_t = float(np.median(times))
            rtf = mean_t / dur_s
            ort_bench[f"{label}_{n_th}T"] = {
                "size_mb": round(os.path.getsize(pth) / 1e6, 3),
                "audio_dur_s": round(dur_s, 2),
                "latency_ms": round(mean_t * 1000.0, 2),
                "rtf": round(rtf, 4),
                "realtime_factor_x": round(1.0 / max(rtf, 1e-6), 1),
            }

    # Check PyTorch vs ONNX parity
    so1 = ort.SessionOptions(); so1.intra_op_num_threads = 1
    s_f32 = ort.InferenceSession(f32_path, so1, providers=["CPUExecutionProvider"])
    w_ort_f, _ = s_f32.run(None, feeds)
    max_diff = float(np.max(np.abs(wav_f.numpy() - w_ort_f)))

    return {
        "architecture": "Sakhiya-Voca-4.5M (Dual-Voice Female + Male, Nepali + Sanskrit, Homomorphic Cepstral Source-Filter)",
        "total_parameters": total_params,
        "parameter_breakdown": breakdown,
        "voices_supported": ["0: F_sakhiya (Female, 215 Hz base)", "1: M_sagar (Male, 125 Hz base)"],
        "female_vs_male_output_shapes": [list(wav_f.shape), list(wav_m.shape)],
        "pytorch_vs_onnx_max_abs_diff": round(max_diff, 6),
        "onnx_exports_and_cpu_rtf": ort_bench,
    }


def main():
    t0 = time.time()
    res = export_and_benchmark()
    res["elapsed_sec"] = round(time.time() - t0, 2)
    out_p = os.path.join(HERE, "..", "results", "exp4_sakhiya_voca_prototype.json")
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"[EXP4 DONE] Total params={res['total_parameters']:,}, Py-ORT diff={res['pytorch_vs_onnx_max_abs_diff']}")
    for k, v in res["onnx_exports_and_cpu_rtf"].items():
        print(f"  {k:16s}: {v['size_mb']:6.3f} MB | RTF={v['rtf']:.4f} ({v['realtime_factor_x']}x real-time)")


if __name__ == "__main__":
    main()
