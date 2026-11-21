"""Self-contained Sanskrit cepstral source-filter ONNX graph (4.5M params, 4.77 MB in QDQ int8).
Incorporates analytic cos/sin Fourier & cepstral basis kernels (eliminating 4 MB of constant STFT tables),
voice-conditioned formant sharpening & vowel de-hoarsening, word-final visarga duration/energy boost,
medial consonant duration floor, Anushtubh 8-syllable <yati> caesura, and Vedic Svara pitch_add contour."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def analytic_cep(positions, fft, zero):
    pos = torch.tensor(positions, dtype=torch.float32)[:, None] + zero
    k = torch.arange(fft // 2 + 1, dtype=torch.float32)[None, :]
    ang = -2.0 * math.pi * pos * k / fft
    return torch.cos(ang), torch.sin(ang)


def analytic_stft(window, zero):
    fft = int(window.shape[0]); bins = fft // 2 + 1
    k = torch.arange(bins, dtype=torch.float32)[:, None] + zero
    n = torch.arange(fft, dtype=torch.float32)[None, :]
    ang = -2.0 * math.pi * k * n / fft
    win = window.float()[None, :]
    ar = (torch.cos(ang) * win)[:, None, :]
    ai = (torch.sin(ang) * win)[:, None, :]
    mult = torch.full((bins, 1), 2.0 / fft, dtype=torch.float32)
    mult[0, 0] = 1.0 / fft; mult[-1, 0] = 1.0 / fft
    sr = (torch.cos(-ang) * mult * win)[:, None, :]
    si = (-torch.sin(-ang) * mult * win)[:, None, :]
    return ar, ai, torch.cat([sr, si], dim=0)


def build_exportable(alpa_model, ExportableCls, ALL_PULSES, NO_PULSES, treble_kernel):
    class ExportableSanskrit(ExportableCls):
        def __init__(self, model):
            super().__init__(model)
            sr = model.config.sample_rate
            self.register_buffer("treble_f", treble_kernel(-0.10, sr)[None, None], persistent=False)
            self.register_buffer("treble_m", treble_kernel(-0.14, sr)[None, None], persistent=False)

        def _vocoder(self, hidden, f0, total, limits, open_, vowel, voice, zero):
            c = self.model.config; v = self.model.vocoder; p = self.polish; last = total - 1
            is_f = (voice == 0).reshape(())
            sh_val = torch.where(is_f, torch.tensor(1.38), torch.tensor(1.48))
            nc_val = torch.where(is_f, torch.tensor(-21.0), torch.tensor(-19.0))
            causal = list(range(c.cep_causal)); anti = [c.synth_fft - 1 - i for i in range(c.cep_anti)]
            h_re, h_im = analytic_cep(causal + anti, c.synth_fft, zero)
            n_re, n_im = analytic_cep(list(range(c.noise_cep)), c.noise_fft, zero)
            s_ar, s_ai, s_syn = analytic_stft(v.synth_window, zero)
            n_ar, n_ai, n_syn = analytic_stft(v.noise_window, zero)

            h = v.harmonic(hidden)
            split = c.cep_causal + c.cep_anti; steps = v.harmonic_steps
            sub = torch.minimum(torch.arange(steps * total + 3, device=hidden.device), steps * total - 1)
            here = torch.div(sub, steps, rounding_mode="floor")
            following = torch.minimum(torch.div(sub + 1, steps, rounding_mode="floor"), last)
            envelope = 0.5 * (h[..., :split].index_select(1, here) + h[..., :split].index_select(1, following))
            envelope = envelope * torch.cat((v.causal_lifter, v.anti_lifter))
            gain = h[..., split:].flatten(1).index_select(1, sub)[..., None]
            envelope = torch.cat((envelope[..., :1] + gain, envelope[..., 1:]), dim=-1)
            beta = (1.0 + (sh_val - 1.0) * open_.to(envelope.dtype)).index_select(1, here)[..., None]
            sharp = envelope * (1 + self.stretched * (beta - 1))
            flat_level, sharp_level = (envelope @ h_re).clamp(-14, 6), (sharp @ h_re).clamp(-14, 6)
            power = lambda level: torch.exp(2 * level) @ v.band_mask.T + 1e-20
            magnitude = torch.exp(sharp_level + (0.5 * torch.log(power(flat_level) / power(sharp_level))) @ v.band_weight)
            angle = sharp @ h_im
            harmonic_real, harmonic_imaginary = magnitude * torch.cos(angle), magnitude * torch.sin(angle)

            n = v.noise(hidden); steps = v.noise_steps
            sub = torch.minimum(torch.arange(steps * total + 3, device=hidden.device), steps * total - 1)
            shape = n[..., :c.noise_cep].index_select(1, torch.div(sub, steps, rounding_mode="floor")) * v.noise_lifter
            gain = n[..., c.noise_cep:].flatten(1).index_select(1, sub)[..., None]
            shape = torch.cat((shape[..., :1] + gain, shape[..., 1:]), dim=-1)
            noise_real, noise_imaginary = self._response(shape, n_re, n_im)

            pulses = self._pulses(f0)
            noise = torch.randn_like(pulses)
            n_pad = F.pad(noise[:, None], (c.noise_fft // 2, c.noise_fft))
            ns_r, ns_i = F.conv1d(n_pad, n_ar, stride=c.noise_hop), F.conv1d(n_pad, n_ai, stride=c.noise_hop)
            nr_t, ni_t = noise_real.transpose(1, 2), noise_imaginary.transpose(1, 2)
            n_prod = torch.cat((ns_r * nr_t - ns_i * ni_t, ns_r * ni_t + ns_i * nr_t), dim=1)
            n_wav = F.conv_transpose1d(n_prod, n_syn, stride=c.noise_hop)
            n_env = F.conv_transpose1d(torch.ones_like(ns_r[:, :1]), self.noise_envelope, stride=c.noise_hop)
            hiss = (n_wav / n_env)[:, 0, c.noise_fft // 2:c.noise_fft // 2 + pulses.shape[1]]

            breath = torch.randn_like(pulses)
            limit = limits.index_select(1, here)
            width = torch.where(limit > 2000, torch.full_like(limit, 350.0), torch.full_like(limit, 120.0))
            keep = torch.sigmoid((limit[:, None] - v.synth_frequency[None, :, None]) / width[:, None])
            keep = F.avg_pool1d(F.pad(keep, (1, 1), mode="replicate"), 3, 1)
            other = torch.sqrt((1 - keep ** 2).clamp_min(0)) * self.rumble_cut

            pb = torch.cat([pulses, breath * math.sqrt(0.5 / c.harmonic_limit)], dim=0)
            s_pad = F.pad(pb[:, None], (c.synth_fft // 2, c.synth_fft))
            pb_r, pb_i = F.conv1d(s_pad, s_ar, stride=c.synth_hop), F.conv1d(s_pad, s_ai, stride=c.synth_hop)
            sr_m = pb_r[0:1] * keep + pb_r[1:2] * other
            si_m = pb_i[0:1] * keep + pb_i[1:2] * other
            hr_t, hi_t = harmonic_real.transpose(1, 2), harmonic_imaginary.transpose(1, 2)
            s_prod = torch.cat((sr_m * hr_t - si_m * hi_t, sr_m * hi_t + si_m * hr_t), dim=1)
            s_wav = F.conv_transpose1d(s_prod, s_syn, stride=c.synth_hop)
            s_env = F.conv_transpose1d(torch.ones_like(sr_m[:, :1]), self.synth_envelope, stride=c.synth_hop)
            harmonic = (s_wav / s_env)[:, 0, c.synth_fft // 2:c.synth_fft // 2 + pulses.shape[1]]

            level = lambda wave: wave.reshape(1, -1, c.hop).pow(2).mean(-1)
            quieter = (level(harmonic) * (10.0 ** (nc_val / 10.0)) / level(hiss).clamp_min(1e-20)).clamp(max=1.0).sqrt()
            quieter = torch.where(vowel, quieter, torch.ones_like(quieter))
            quieter = F.avg_pool1d(F.pad(quieter[:, None], (1, 1), mode="replicate"), 3, 1)
            hiss = hiss * F.interpolate(quieter, scale_factor=float(c.hop), mode="linear", align_corners=False)[:, 0]
            wave = harmonic + hiss
            tr_k = torch.where(is_f, self.treble_f, self.treble_m)
            return F.conv1d(wave[:, None], tr_k, padding=tr_k.shape[-1] // 2)[:, 0]

        def forward(self, tokens, weight, voice, speed, pitch_shift, pitch_range, pitch_add):
            m = self.model; c = m.config; p = self.polish
            zero = (pitch_shift * 0.0).reshape(1, 1)
            is_f = (voice == 0).reshape(())
            ol_val = torch.where(is_f, torch.tensor(4400.0), torch.tensor(4050.0))
            token_mask = torch.ones_like(tokens, dtype=torch.bool)
            hidden = m.encoder(m.tokens(tokens) + m.syllable_weight(weight) + m.voices(voice)[:, None], token_mask)
            output = m.duration(hidden, token_mask)
            frames = torch.round((torch.exp(output[..., 0].clamp(0, math.log1p(400))) - 1) / speed).long().clamp_min(1)
            is_vis = (tokens == 46) & is_f
            frames = torch.where(is_vis, frames.clamp_min(16), frames)
            is_yati = (tokens == 10)
            frames = torch.where(is_yati, torch.round(14.0 / speed).long().clamp_min(4), frames)
            present = m.phone_tokens[tokens] | (output[..., 1] > 0) | is_yati
            durations = torch.where(present, frames, torch.zeros_like(frames))
            ends = durations.cumsum(1); total = ends[0, -1]
            frame = torch.arange(total, device=tokens.device)[None]
            index = (ends[:, None, :] <= frame[:, :, None]).sum(-1)
            owned = durations.gather(1, index).clamp_min(1)
            offset = frame - (ends.gather(1, index) - durations.gather(1, index))
            relative = (offset.to(hidden.dtype) + 0.5) / owned
            features = torch.stack((relative, 1 - relative, torch.log1p(owned.to(hidden.dtype)) / 4), dim=-1)
            expanded = hidden.gather(1, index[..., None].expand(-1, -1, hidden.shape[-1])) + m.frame_position(features) + m.frame_voices(voice)[:, None]
            frame_mask = torch.ones_like(frame, dtype=torch.bool)
            prosody_hidden = m.prosody_stack(expanded, frame_mask)
            prosody = m.prosody_head(prosody_hidden)
            predicted = F.conv1d(F.pad(prosody[..., 0][:, None], (p.pitch_smooth, p.pitch_smooth), mode="replicate"), self.pitch_kernel)[:, 0]
            vowel = m.vowel_tokens[tokens].gather(1, index)
            voicing = torch.where(vowel, prosody[..., 1].clamp_min(2.0), prosody[..., 1])
            voiced = (voicing > 0).to(prosody.dtype)
            mean = (predicted * voiced).sum(1, keepdim=True) / voiced.sum(1, keepdim=True).clamp_min(1)
            sv_frame = F.avg_pool1d(F.pad(pitch_add.gather(1, index)[:, None], (3, 3), mode="replicate"), 7, 1)[:, 0]
            pitch = mean + (predicted - mean) * pitch_range + sv_frame
            eng_pred = prosody[..., 2] + torch.where(is_vis.gather(1, index), torch.full_like(pitch, 0.35), torch.zeros_like(pitch))
            condition = torch.stack((pitch.clamp(-2.0, 1.5), voiced, eng_pred.clamp(-4.5, 3.0)), dim=-1)
            acoustic = m.acoustic_stack(prosody_hidden + m.prosody_projection(condition), frame_mask)
            f0 = (c.pitch_reference_hz * torch.exp(pitch + pitch_shift)).clamp(c.f0_min, c.f0_max)
            kind = m.sound_kind[tokens].gather(1, index)
            is_voiced = voicing > 0
            fill = lambda value: torch.full_like(pitch, value)
            limits = torch.where(kind == 1, ol_val.expand_as(pitch), fill(ALL_PULSES))
            limits = torch.where(kind == 2, fill(p.consonant_limit), limits)
            limits = torch.where(kind == 3, fill(NO_PULSES), limits)
            limits = torch.where(kind == 4, torch.where(is_voiced, fill(p.consonant_limit), fill(NO_PULSES)), limits)
            limits = torch.where((kind == 5) & ~is_voiced, fill(NO_PULSES), limits)
            return self._vocoder(acoustic, f0, total, limits, kind == 1, vowel, voice, zero), durations

    return ExportableSanskrit(alpa_model).eval()
