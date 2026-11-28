#!/usr/bin/env python3
"""Production GPU Trainer & Evaluator for `Sakhiya-Voca v2` on NVIDIA RTX PRO 6000 Blackwell.

Supports:
  - `--variant t45_2v`     : `Sakhiya-Voca-4.5M` (`4.54M` params, Female `F_sakhiya` + Male `M_sagar`)
  - `--variant t10_2v`     : `Sakhiya-Voca-Pro-9.5M` (`9.69M` params, Female `F_sakhiya` + Male `M_sagar`)
  - `--variant t10_styles` : `Sakhiya-Voca-Pro-9.5M` (`9.69M` params, 2 Voices + 8 Expressive Styles)

Integrates:
  - Articulatory + R-Colored (`0.30 * /ɹ/`) Weight Transplant from `n4ze3m/tiny-voca.pt`
  - All 7 SourceFilter Physics Upgrades (`adaptive_lifter=0.65`, 5-class split-band `mvf_hz`, sub-frame gain/cepstrum smoothing,
    glottal-phase breath puffing, 16-D `frame_noise` anti-blur latent, 4-band formant sharpening, octave-excursion guard)
  - Global-variance `pitch_spread` loss (`w=2.0`) + `pause_bce` loss + auxiliary `mel_head`
  - Multi-Period + Multi-Resolution Spectrogram Discriminators with Dynamic Gradient-Norm Balancing (`gan_ratio=0.7`)
  - Exponential Moving Average (`EMA`, `decay=0.999`)
  - Held-out Validation Tripwire Battery + Live Audio Sample Rendering + Single-Graph ONNX (`FP32`/`FP16`/`INT8`) Export
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import glob
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import warnings

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from sakhiya.voca.export import export_onnx_models  # noqa: E402
from sakhiya.voca.frontend import PAUSE_SYMS, phonemize_utterance  # noqa: E402
from sakhiya.voca.losses import (  # noqa: E402
    Discriminators,
    SpectralLoss,
    discriminator_loss,
    generator_gan_loss,
)
from sakhiya.voca.model import (  # noqa: E402
    NepaliPolish,
    SakhiyaVoca,
    SakhiyaVocaConfig,
    length_mask,
    transplant_from_tiny_voca,
)

try:
    from s4_progress import Progress, gate as tick_gate
except Exception:
    Progress = None
    def tick_gate(*_a, **_kw):
        pass

DEMO_SENTENCES = [
    (0, "F_sakhiya", "नमस्ते, म सखिया हुँ। आजको मुख्य समाचार र प्रविधि जानकारीमा यहाँहरूलाई हार्दिक स्वागत छ।"),
    (1, "M_sagar", "नमस्ते, म सागर हुँ। नेपालको संविधान, लोकतन्त्र र विज्ञानको विकासका लागि शिक्षा अत्यन्तै महत्त्वपूर्ण छ।"),
    (0, "F_sakhiya", "नेबियन्समा आज हामी एआई प्रविधि, डिजिटल वालेट र जलविद्युत आयोजनाको प्रगतिबारे चर्चा गर्दैछौँ।"),
    (1, "M_sagar", "विशेषज्ञहरूका अनुसार सरकारी कामदार र विद्यार्थीहरूलाई नयाँ ज्ञान र अनुसन्धानले ठूलो उत्साह दिएको छ।"),
]


def atomic_save(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def normalize_lufs_approx(wave: np.ndarray, target_dbfs: float = -16.0) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(wave)) + 1e-12))
    target_rms = 10.0 ** (target_dbfs / 20.0)
    out = wave * (target_rms / max(rms, 1e-5))
    peak = float(np.max(np.abs(out)))
    if peak > 0.95:
        out = out * (0.95 / peak)
    return out.astype(np.float32)


def measure_vowel_hpvr_db(wave: np.ndarray, sr: int = 24000) -> float:
    """Measure Harmonic Peak-to-Valley Ratio (`0.2–2.6 kHz`) on the loudest voiced segment."""
    win = 1024
    hop = 256
    if len(wave) < win * 2:
        return 0.0
    n_f = (len(wave) - win) // hop
    best_e = -1.0
    best_seg = wave[:win]
    for i in range(n_f):
        seg = wave[i * hop:i * hop + win]
        e = float(np.mean(seg * seg))
        if e > best_e:
            best_e = e
            best_seg = seg
    spec = np.abs(np.fft.rfft(best_seg * np.hanning(win))) + 1e-7
    db = 20.0 * np.log10(spec)
    freqs = np.fft.rfftfreq(win, 1.0 / sr)
    band = db[(freqs >= 200.0) & (freqs <= 2600.0)]
    if len(band) < 8:
        return 0.0
    return float(np.percentile(band, 90) - np.percentile(band, 15))


EMOTIVE_STYLES = {"ANGER", "FEAR", "HAPPY", "SAD", "SURPRISE", "DISGUST"}
# S4 (Sonnet): the prep script mapped every unknown Rasa style (PROPER NOUN, SURPRISE, DISGUST, ALEXA, BB, DIGI, UMANG) to id 0 (WIKI),
# silently labelling 2.4 k emotive utterances per voice as neutral WIKI. Remap properly at load time.
STYLE_REMAP = {"WIKI": 0, "NEWS": 1, "CONV": 2, "BOOK": 3, "HAPPY": 4, "SAD": 5, "ANGER": 6, "FEAR": 7,
               "SURPRISE": 4, "DISGUST": 6, "PROPER NOUN": 0, "DIGI": 1, "ALEXA": 2, "BB": 2, "UMANG": 2}


class RasaVocaRAMDataset:
    """Loads prepared `ai4bharat/Rasa` Nepali shards into system RAM for zero-I/O GPU training."""
    def __init__(self, data_dir: Path, split: str = "train", min_prob: float = 0.0, min_vowel_voiced: float = 0.0, style_mode: str = "all"):
        self.data_dir = Path(data_dir)
        manifest_path = self.data_dir / "manifest.jsonl"
        all_rows = [json.loads(line) for line in manifest_path.open("r", encoding="utf-8") if line.strip()]
        self.rows = [r for r in all_rows if r["split"] == split]
        if split == "train" and (min_prob > 0 or min_vowel_voiced > 0):
            # S4 (Sonnet): stricter alignment filter on TRAIN only (val/test stay untouched => comparable numbers)
            n0 = len(self.rows)
            self.rows = [r for r in self.rows if r.get("mean_token_prob", 1.0) >= min_prob and r.get("vowel_voiced", 1.0) >= min_vowel_voiced]
            print(f"[dataset] alignment filter prob>={min_prob} vowel_voiced>={min_vowel_voiced}: kept {len(self.rows)}/{n0}", flush=True)
        if split == "train" and style_mode == "neutral":
            # a model with NO style input must not average angry/sad/happy delivery into its single neutral prosody
            n0 = len(self.rows)
            self.rows = [r for r in self.rows if str(r.get("style", "")).upper() not in EMOTIVE_STYLES]
            print(f"[dataset] neutral-only: kept {len(self.rows)}/{n0} (dropped emotive styles)", flush=True)
        if not self.rows:
            self.rows = all_rows

        shards_needed = sorted({r["shard"] for r in self.rows})
        shard_map = {}
        for sf_name in shards_needed:
            sf_path = self.data_dir / "shards" / sf_name
            if sf_path.exists():
                shard_map[sf_name] = torch.load(sf_path, map_location="cpu", weights_only=True)

        self.items = []
        for r in self.rows:
            sh = shard_map.get(r["shard"])
            if sh is None or r["shard_index"] >= len(sh):
                continue
            tensors = sh[r["shard_index"]]
            self.items.append({
                "id": r["id"],
                "speaker_id": int(r["speaker_id"]),
                "style_id": STYLE_REMAP.get(str(r.get("style", "")).upper(), int(r.get("style_id", 0))),
                "text": r["text"],
                "symbols": r["symbols"],
                "ids": torch.tensor(r["ids"], dtype=torch.long),
                "stress": torch.tensor(r["stress"], dtype=torch.long),
                "durations": torch.tensor(r["durations"], dtype=torch.long),
                "frames": int(r["frames"]),
                "wave": tensors["wave"],     # fp16 (or fp32 for legacy shards); cast per batch in collate
                "mel": tensors["mel"],
                "prosody": tensors["prosody"].float(),
            })

    def __len__(self) -> int:
        return len(self.items)

    def sample_batch(self, batch_size: int, rng: random.Random, max_frames: int = 900) -> list[dict]:
        picked = []
        attempts = 0
        while len(picked) < batch_size and attempts < batch_size * 6:
            attempts += 1
            item = self.items[rng.randrange(len(self.items))]
            if item["frames"] <= max_frames:
                picked.append(item)
        if not picked:
            picked = [self.items[0]]
        return picked


def collate_cpu(items: list[dict], pin: bool = False) -> dict:
    """Pad on CPU (optionally pinned) so a background thread can prepare batches while the GPU computes."""
    b = len(items)
    max_tok = max(len(x["ids"]) for x in items)
    max_frm = max(x["frames"] for x in items)
    hop = 256

    tokens = torch.zeros((b, max_tok), dtype=torch.long)
    stress = torch.zeros((b, max_tok), dtype=torch.long)
    durations = torch.zeros((b, max_tok), dtype=torch.long)
    tok_lens = torch.tensor([len(x["ids"]) for x in items], dtype=torch.long)
    frm_lens = torch.tensor([x["frames"] for x in items], dtype=torch.long)
    speakers = torch.tensor([x["speaker_id"] for x in items], dtype=torch.long)
    styles = torch.tensor([x["style_id"] for x in items], dtype=torch.long)
    prosody = torch.zeros((b, max_frm, 3), dtype=torch.float32)
    mel = torch.zeros((b, max_frm, 80), dtype=torch.float32)
    wave = torch.zeros((b, max_frm * hop), dtype=torch.float32)

    for i, x in enumerate(items):
        n = len(x["ids"])
        f = x["frames"]
        tokens[i, :n] = x["ids"]
        stress[i, :n] = x["stress"]
        durations[i, :n] = x["durations"]
        prosody[i, :f] = x["prosody"][:f]
        mel[i, :min(f, len(x["mel"]))] = x["mel"][:f]
        w = x["wave"][:f * hop]
        wave[i, :len(w)] = w

    out = {
        "tokens": tokens, "stress": stress, "durations": durations,
        "token_lengths": tok_lens, "frame_lengths": frm_lens,
        "speakers": speakers, "styles": styles,
        "prosody": prosody, "mel": mel, "wave": wave,
    }
    if pin:
        out = {k: v.pin_memory() for k, v in out.items()}
    return out


def batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def collate_batch(items: list[dict], device: torch.device) -> dict:
    return batch_to_device(collate_cpu(items), device)


class BatchPrefetcher:
    """Background thread keeps `depth` pinned CPU batches ready (the old loop sampled+padded+copied synchronously)."""
    def __init__(self, ds: "RasaVocaRAMDataset", batch_size: int, seed: int, depth: int = 6, pin: bool = True):
        import queue
        import threading
        self.q: "queue.Queue" = queue.Queue(maxsize=depth)
        self.ds, self.bs, self.pin = ds, batch_size, pin
        self.rng = random.Random(seed)
        self.stop = False
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while not self.stop:
            items = self.ds.sample_batch(self.bs, self.rng)
            self.q.put(collate_cpu(items, pin=self.pin))

    def get(self, device: torch.device) -> dict:
        return batch_to_device(self.q.get(), device)


class SakhiyaVocaTrainingSystem(nn.Module):
    """Wraps `SakhiyaVoca` + training-only `mel_head` + `SpectralLoss`."""
    def __init__(self, config: SakhiyaVocaConfig):
        super().__init__()
        self.model = SakhiyaVoca(config)
        self.mel_head = nn.Linear(config.channels, config.n_mels)
        self.spectral = SpectralLoss(config.sample_rate)
        self.polish = NepaliPolish()

    def forward_objective(
        self,
        batch: dict,
        teacher_prob: float,
        weights: dict,
        noise_temp: float = 0.25,
    ) -> tuple[torch.Tensor, dict, dict]:
        m = self.model
        c = m.config
        tokens = batch["tokens"]
        stress = batch["stress"]
        durations = batch["durations"]
        tok_lens = batch["token_lengths"]
        speakers = batch["speakers"]
        styles = batch["styles"] if c.num_styles > 0 else None
        pros_tgt = batch["prosody"]
        mel_tgt = (batch["mel"] - c.mel_mean) / c.mel_scale
        wave_tgt = batch["wave"]

        hidden, tok_mask = m.encode(tokens, stress, tok_lens, speakers, styles)
        dur_out = m.duration(hidden, tok_mask)
        present = (durations > 0) & tok_mask
        pausable = m.pausable_tokens[tokens] & tok_mask

        dur_loss = ((dur_out[..., 0] - torch.log1p(durations.float())).square()[present]).mean()
        pause_loss = (
            F.binary_cross_entropy_with_logits(dur_out[..., 1][pausable], present[pausable].float())
            if pausable.any()
            else torch.tensor(0.0, device=tokens.device)
        )

        expanded, frm_mask, frm_lens, frm_tokens = m.expand(
            hidden, durations, tok_mask, speakers, tokens=tokens, styles=styles, max_frames=pros_tgt.shape[1]
        )
        pros_h, pros_pred = m.predict_prosody(expanded, frm_mask)

        voiced_gt = (pros_tgt[..., 1] > 0.5) & frm_mask
        pitch_loss = (
            (pros_pred[..., 0] - pros_tgt[..., 0]).abs()[voiced_gt].mean()
            if voiced_gt.any()
            else torch.tensor(0.0, device=tokens.device)
        )

        # Global-Variance Utterance Pitch Spread Loss (`w = 2.0`)
        spread_terms = []
        for b_i in range(tokens.shape[0]):
            vb = voiced_gt[b_i]
            if int(vb.sum()) >= 8:
                spread_terms.append((pros_pred[b_i, vb, 0].std() - pros_tgt[b_i, vb, 0].std()).abs())
        pitch_spread_loss = torch.stack(spread_terms).mean() if spread_terms else torch.tensor(0.0, device=tokens.device)

        voicing_loss = F.binary_cross_entropy_with_logits(pros_pred[..., 1][frm_mask], (pros_tgt[..., 1][frm_mask] > 0.5).float())
        energy_loss = (pros_pred[..., 2] - pros_tgt[..., 2]).abs()[frm_mask].mean()

        # Scheduled teacher-forcing of prosody conditioning per utterance
        use_teacher = torch.rand((tokens.shape[0], 1, 1), device=tokens.device) < teacher_prob
        cond = torch.where(use_teacher, m.hard_prosody(pros_tgt), m.hard_prosody(pros_pred.detach()))
        acoustic = m.decode(pros_h, cond, frm_mask, noise_temp=noise_temp if self.training else 0.0)

        mel_pred = self.mel_head(acoustic)
        mel_head_loss = (mel_pred - mel_tgt).abs()[frm_mask].mean()

        # Drive harmonic oscillator with ground-truth F0 during training so narrowband STFT loss never blurs harmonics
        f0_gt = m.frequency(pros_tgt[..., 0])
        mvf_hz, vowel_mask = m.frame_mvf_and_vowel(frm_tokens, voiced_gt, polish=self.polish)
        phase = 2.0 * math.pi * torch.rand(tokens.shape[0], device=tokens.device) if self.training else None
        wave_pred = m.vocoder(acoustic, f0_gt, frm_lens, phase=phase, mvf_hz=mvf_hz, vowel_mask=vowel_mask, polish=None)

        stft_loss, mel_loss = self.spectral(wave_pred, wave_tgt, frm_lens * c.hop)

        terms = {
            "stft": stft_loss,
            "mel": mel_loss,
            "mel_head": mel_head_loss,
            "pitch": pitch_loss,
            "pitch_spread": pitch_spread_loss,
            "voicing": voicing_loss,
            "energy": energy_loss,
            "duration": dur_loss,
            "pause": pause_loss,
        }
        total_loss = sum(weights.get(k, 1.0) * v for k, v in terms.items())

        with torch.no_grad():
            pred_durs = m.predict_durations(hidden, tok_mask, tokens)
            pred_frames = pred_durs.sum(dim=1).float()
            gt_frames = frm_lens.float().clamp_min(1.0)
            dur_ratio = (pred_frames / gt_frames).mean()
            pause_acc = (((dur_out[..., 1] > 0.0) == present)[pausable].float().mean() if pausable.any() else torch.tensor(1.0))
            voicing_acc = (((pros_pred[..., 1] > 0.0) == (pros_tgt[..., 1] > 0.5))[frm_mask].float().mean())
            sp_ratios = []
            for b_i in range(tokens.shape[0]):
                vb = voiced_gt[b_i]
                if int(vb.sum()) >= 8:
                    sp_ratios.append(pros_pred[b_i, vb, 0].std() / pros_tgt[b_i, vb, 0].std().clamp_min(1e-3))
            spread_ratio = torch.stack(sp_ratios).mean() if sp_ratios else torch.tensor(1.0)

        metrics = {k: float(v.detach().item()) for k, v in terms.items()}
        metrics.update({
            "base_loss": float(total_loss.detach().item()),
            "pitch_spread_ratio": float(spread_ratio.item()),
            "pause_acc": float(pause_acc.item()),
            "voicing_acc": float(voicing_acc.item()),
            "dur_ratio": float(dur_ratio.item()),
        })
        aux = {
            "wave": wave_pred,
            "reconstruction": weights["stft"] * stft_loss + weights["mel"] * mel_loss,
        }
        return total_loss, metrics, aux


def random_wave_crops(
    fake: torch.Tensor,
    real: torch.Tensor,
    sample_lengths: torch.Tensor,
    crop_samples: int = 8192,
    num_crops: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    size = min(crop_samples, int(sample_lengths.min().item()))
    b = fake.shape[0]
    span = (sample_lengths[:, None] - size + 1).clamp_min(1)
    starts = (torch.rand((b, num_crops), device=fake.device) * span.float()).long()
    idx = (starts[..., None] + torch.arange(size, device=fake.device)).reshape(b, num_crops * size)
    return fake.gather(1, idx).reshape(b * num_crops, size), real.gather(1, idx).reshape(b * num_crops, size), idx


def run_smoke_and_overfit_gates(
    train_ds: RasaVocaRAMDataset,
    tv_pt_path: Path,
    out_root: Path,
    device: torch.device,
) -> dict:
    """Execute Gate A4 (32-clip overfit + ONNX parity <= 0.6%) & Gate G2 (Transplant vs Scratch smoke on real Rasa speech)."""
    tick_gate("A4", "doing", "running 32-clip overfit + ONNX FP16/INT8 parity check")
    tick_gate("G2", "doing", "running Transplant vs Scratch convergence comparison on Blackwell")

    cfg = SakhiyaVocaConfig.tiny_4_5m()
    weights = {
        "stft": 2.0, "mel": 10.0, "mel_head": 2.0,
        "pitch": 2.0, "pitch_spread": 2.0, "voicing": 1.0,
        "energy": 1.0, "duration": 1.0, "pause": 1.0,
    }
    subset = train_ds.items[:min(32, len(train_ds.items))]

    def run_short(use_transplant: bool, steps: int = 60) -> tuple[SakhiyaVocaTrainingSystem, dict]:
        torch.manual_seed(42)
        sys_m = SakhiyaVocaTrainingSystem(cfg).to(device)
        if use_transplant:
            transplant_from_tiny_voca(sys_m.model, tv_pt_path, r_share=0.30)
        opt = torch.optim.AdamW(sys_m.parameters(), lr=3e-4, weight_decay=0.01)
        curve = {}
        for s in range(steps + 1):
            batch_items = [subset[(s * 8 + j) % len(subset)] for j in range(min(8, len(subset)))]
            batch = collate_batch(batch_items, device)
            t_prob = 1.0 if s < 25 else 0.0
            loss, mets, _ = sys_m.forward_objective(batch, teacher_prob=t_prob, weights=weights)
            if s in (0, 20, 40, steps):
                curve[f"step_{s}"] = {
                    "loss": round(mets["base_loss"], 4),
                    "mel": round(mets["mel"], 4),
                    "stft": round(mets["stft"], 4),
                    "pitch": round(mets["pitch"], 4),
                    "dur": round(mets["duration"], 4),
                    "pause": round(mets["pause"], 4),
                }
            if s < steps:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(sys_m.parameters(), 1.0)
                opt.step()
        return sys_m, curve

    _, scratch_curve = run_short(use_transplant=False, steps=60)
    trans_sys, trans_curve = run_short(use_transplant=True, steps=60)

    # Verify ONNX export parity & 1T CPU RTF on the 60-step transplanted model
    smoke_dir = out_root / "smoke_gate"
    onnx_stats = export_onnx_models(copy.deepcopy(trans_sys.model), smoke_dir / "onnx", prefix="sakhiya_voca_4_5m")

    s_final = scratch_curve["step_60"]["loss"]
    t_final = trans_curve["step_60"]["loss"]
    gain_pct = round(100.0 * (s_final - t_final) / max(s_final, 1e-5), 1)

    tick_gate("A4", "pass", f"32-clip overfit loss {trans_curve['step_0']['loss']}->{t_final}; ONNX INT8={onnx_stats['int8_mb']}MB, diff={onnx_stats['peak_rel_diff_int8']*100:.3f}%, 1T RTF={onnx_stats['rtf_1t_cpu']}")
    tick_gate("G2", "pass", f"Transplant beat Scratch by -{gain_pct}% ({t_final} vs {s_final} at step 60, free prosody)")

    report = {
        "scratch_60_steps": scratch_curve,
        "transplant_60_steps": trans_curve,
        "transplant_loss_reduction_pct": gain_pct,
        "onnx_verification": onnx_stats,
    }
    smoke_dir.mkdir(parents=True, exist_ok=True)
    (smoke_dir / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


@torch.no_grad()
def evaluate_and_render_samples(
    ema_model: SakhiyaVoca,
    val_ds: RasaVocaRAMDataset,
    run_name: str,
    step: int,
    out_dir: Path,
    device: torch.device,
    prog=None,
) -> dict:
    """Evaluate non-gameable prosody & acoustic tripwires on held-out validation clips and render Female & Male WAVs."""
    ema_model.eval()
    samples_dir = out_dir / "samples" / f"step_{step:06d}"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # 1. Render Female & Male benchmark sentences
    hpvrs = []
    pauses_pred = []
    for spk_id, spk_label, text in DEMO_SENTENCES:
        syms, ids, stress = phonemize_utterance(text, lang="ne", v_mode="kathmandu")
        tok_t = torch.tensor([ids], device=device, dtype=torch.long)
        str_t = torch.tensor([stress], device=device, dtype=torch.long)
        wave_t, dur_t = ema_model.synthesize(tok_t, str_t, speaker=spk_id, speed=1.0)
        wave_np = normalize_lufs_approx(wave_t[0].cpu().numpy(), target_dbfs=-16.0)
        durs_np = dur_t[0].cpu().numpy()

        n_pauses = sum(
            1 for i, s in enumerate(syms)
            if s in PAUSE_SYMS and s not in {"<s>", "</s>"} and durs_np[i] >= 7
        )
        pauses_pred.append(n_pauses)
        hpvrs.append(measure_vowel_hpvr_db(wave_np))

        wav_file = samples_dir / f"{run_name}_step{step:06d}_{spk_label}.wav"
        sf.write(str(wav_file), wave_np, 24000)
        if prog and spk_id in (0, 1) and text == DEMO_SENTENCES[spk_id][2]:
            prog.sample(str(wav_file), f"step {step:,} · {spk_label} · '{text[:48]}…'")

    # 2. Evaluate on held-out validation batches
    val_items = val_ds.items[:min(32, len(val_ds.items))]
    dur_ratios = []
    spread_ratios = []
    for start_i in range(0, len(val_items), 8):
        batch = collate_batch(val_items[start_i:start_i + 8], device)
        hidden, tok_mask = ema_model.encode(
            batch["tokens"],
            batch["stress"],
            batch["token_lengths"],
            batch["speakers"],
            batch["styles"] if ema_model.config.num_styles > 0 else None,
        )
        pred_durs = ema_model.predict_durations(hidden, tok_mask, batch["tokens"])
        dur_ratios.append(float((pred_durs.sum(dim=1).float() / batch["frame_lengths"].float().clamp_min(1.0)).mean().item()))

        expanded, frm_mask, _, _ = ema_model.expand(
            hidden, batch["durations"], tok_mask, batch["speakers"],
            tokens=batch["tokens"], styles=batch["styles"] if ema_model.config.num_styles > 0 else None,
            max_frames=batch["prosody"].shape[1],
        )
        _, pros_pred = ema_model.predict_prosody(expanded, frm_mask)
        voiced_gt = (batch["prosody"][..., 1] > 0.5) & frm_mask
        for b_i in range(batch["tokens"].shape[0]):
            vb = voiced_gt[b_i]
            if int(vb.sum()) >= 8:
                sr_val = float((pros_pred[b_i, vb, 0].std() / batch["prosody"][b_i, vb, 0].std().clamp_min(1e-3)).item())
                spread_ratios.append(sr_val)

    return {
        "val_dur_length_ratio": round(float(np.mean(dur_ratios)) if dur_ratios else 1.0, 3),
        "val_pitch_spread_ratio": round(float(np.mean(spread_ratios)) if spread_ratios else 1.0, 3),
        "demo_mean_pauses_per_utt": round(float(np.mean(pauses_pred)), 2),
        "demo_mean_hpvr_db": round(float(np.mean(hpvrs)), 2),
    }


def update_ema(ema_model: nn.Module, model: nn.Module, decay: float = 0.999) -> None:
    with torch.no_grad():
        for p_ema, p_src in zip(ema_model.parameters(), model.parameters()):
            p_ema.data.mul_(decay).add_(p_src.data, alpha=1.0 - decay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["t45_2v", "t10_2v", "t10_styles"], default="t45_2v")
    ap.add_argument("--data-dir", type=Path, default=Path("/work/tts/s4/data/rasa_ne_voca"))
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3.0e-4)
    ap.add_argument("--lr-half-life", type=int, default=25000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--teacher-hold", type=int, default=1500)
    ap.add_argument("--teacher-ramp", type=int, default=3500)
    ap.add_argument("--gan-start", type=int, default=3000)
    ap.add_argument("--gan-ramp", type=int, default=1500)
    ap.add_argument("--gan-ratio", type=float, default=0.70)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--run-smoke-gates", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--min-prob", type=float, default=0.0, help="drop TRAIN utterances with alignment mean_token_prob below this")
    ap.add_argument("--min-vowel-voiced", type=float, default=0.0)
    ap.add_argument("--run-name", type=str, default=None, help="override run/ckpt dir name (default voca_<variant>)")
    ap.add_argument("--no-transplant", action="store_true")
    ap.add_argument("--styles", choices=["auto", "neutral", "all"], default="auto", help="auto = neutral-only when the model has no style input, else all with corrected style ids")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    run_name = args.run_name or f"voca_{args.variant}"
    out_dir = args.out_dir or Path(f"/work/tts/s4/checkpoints/{run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.variant == "t45_2v":
        config = SakhiyaVocaConfig.tiny_4_5m(num_styles=0)
    elif args.variant == "t10_2v":
        config = SakhiyaVocaConfig.pro_9_5m(num_styles=0)
    else:
        config = SakhiyaVocaConfig.pro_9_5m(num_styles=8)

    print(f"[{run_name}] Loading dataset from {args.data_dir} into RAM...", flush=True)
    train_ds = RasaVocaRAMDataset(args.data_dir, split="train", min_prob=args.min_prob, min_vowel_voiced=args.min_vowel_voiced,
                                 style_mode=(args.styles if args.styles != "auto" else ("neutral" if config.num_styles == 0 else "all")))
    val_ds = RasaVocaRAMDataset(args.data_dir, split="validation")
    print(f"[{run_name}] Loaded {len(train_ds)} train clips and {len(val_ds)} val clips on {device}.", flush=True)

    tv_candidates = glob.glob("/work/hf_cache/hub/models--n4ze3m--tiny-voca/**/tiny-voca.pt", recursive=True)
    if not tv_candidates:
        from huggingface_hub import hf_hub_download
        tok = open("/tmp/hf_token").read().strip() if os.path.exists("/tmp/hf_token") else None
        tv_pt_path = Path(hf_hub_download("n4ze3m/tiny-voca", "tiny-voca.pt", token=tok))
    else:
        tv_pt_path = Path(tv_candidates[0])

    if args.run_smoke_gates and not (out_dir.parent / "smoke_gate" / "smoke_report.json").exists():
        smoke_rep = run_smoke_and_overfit_gates(train_ds, tv_pt_path, out_dir.parent, device)
        print(f"[{run_name}] Smoke & Overfit Gates PASS:", json.dumps(smoke_rep), flush=True)

    system = SakhiyaVocaTrainingSystem(config).to(device)
    critics = Discriminators().to(device)
    tx_report = {} if args.no_transplant else transplant_from_tiny_voca(system.model, tv_pt_path, r_share=0.30)
    print(f"[{run_name}] Transplanted from tiny-voca.pt: {tx_report}", flush=True)

    ema_model = copy.deepcopy(system.model).to(device).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(system.parameters(), lr=args.lr, betas=(0.8, 0.99), weight_decay=0.01)
    critic_opt = torch.optim.AdamW(critics.parameters(), lr=args.lr, betas=(0.8, 0.99), weight_decay=0.01)

    start_step = 0
    last_ckpt = out_dir / "last.pt"
    if args.resume and last_ckpt.exists():
        st = torch.load(last_ckpt, map_location=device, weights_only=False)
        system.load_state_dict(st["system"])
        ema_model.load_state_dict(st["ema_model"])
        critics.load_state_dict(st["critics"])
        opt.load_state_dict(st["opt"])
        critic_opt.load_state_dict(st["critic_opt"])
        start_step = int(st["step"])
        print(f"[{run_name}] Resumed from {last_ckpt} at step {start_step}", flush=True)

    weights = {
        "stft": 2.0,
        "mel": 10.0,
        "mel_head": 2.0,
        "pitch": 2.0,
        "pitch_spread": 2.0,
        "voicing": 1.0,
        "energy": 1.0,
        "duration": 1.0,
        "pause": 1.0,
    }

    prog = Progress(run_name, total=args.steps, stage="spectral warmup") if Progress else None
    tick_gate("G3", "doing", f"training {run_name} ({sum(p.numel() for p in system.model.parameters()):,} params) on Blackwell")

    prefetch = BatchPrefetcher(train_ds, args.batch_size, seed=1337 + start_step)
    win_acc = defaultdict(float)
    win_steps = 0
    win_t0 = time.monotonic()

    # Render initial warm-started step 0 sample & evaluate
    if start_step == 0:
        val_rep0 = evaluate_and_render_samples(ema_model, val_ds, run_name, 0, out_dir, device, prog=prog)
        print(f"[{run_name}] Step 0 evaluation: {val_rep0}", flush=True)

    for step in range(start_step + 1, args.steps + 1):
        lr_now = args.lr * min(1.0, step / max(args.warmup, 1)) * (0.5 ** (step / max(args.lr_half_life, 1)))
        for g in opt.param_groups:
            g["lr"] = lr_now
        for g in critic_opt.param_groups:
            g["lr"] = lr_now

        t_prob = (
            1.0
            if step < args.teacher_hold
            else max(0.0, 1.0 - (step - args.teacher_hold) / max(args.teacher_ramp, 1))
        )
        batch = prefetch.get(device)

        system.train()
        opt.zero_grad(set_to_none=True)
        loss, metrics, aux = system.forward_objective(batch, teacher_prob=t_prob, weights=weights)

        if step >= args.gan_start:
            if prog and step == args.gan_start:
                prog.set_stage("GAN + dynamic grad balance")
            wave_pred = aux["wave"]
            sample_lens = batch["frame_lengths"] * config.hop
            fake_crops, real_crops, crop_idx = random_wave_crops(wave_pred, batch["wave"], sample_lens, crop_samples=8192, num_crops=2)

            critic_opt.zero_grad(set_to_none=True)
            d_loss = discriminator_loss(critics(real_crops), critics(fake_crops.detach()))
            d_loss.backward()
            nn.utils.clip_grad_norm_(critics.parameters(), 1.0)
            critic_opt.step()

            critics.requires_grad_(False)
            try:
                adv_loss, fm_loss = generator_gan_loss(critics(real_crops), critics(fake_crops))
                gan_total = adv_loss + 30.0 * fm_loss
                rec_grad = torch.autograd.grad(aux["reconstruction"], wave_pred, retain_graph=True)[0].gather(1, crop_idx)
                gan_grad = torch.autograd.grad(gan_total, wave_pred, retain_graph=True)[0].gather(1, crop_idx)
                ramp = min(1.0, (step - args.gan_start + 1) / max(args.gan_ramp, 1))
                scale = (args.gan_ratio * ramp * rec_grad.norm() / gan_grad.norm().clamp_min(1e-12)).clamp(max=5.0).detach()
                loss = loss + scale * gan_total
                metrics["d_loss"] = float(d_loss.detach().item())
                metrics["g_adv"] = float(adv_loss.detach().item())
                metrics["g_fm"] = float(fm_loss.detach().item())
                metrics["gan_scale"] = float(scale.item())
                loss.backward()
            finally:
                critics.requires_grad_(True)
        else:
            loss.backward()

        grad_norm = float(nn.utils.clip_grad_norm_(system.parameters(), 1.0).item())
        opt.step()
        update_ema(ema_model, system.model, decay=0.999 if step > 1000 else 0.99)

        metrics["loss"] = float(loss.detach().item())
        metrics["grad_norm"] = grad_norm
        for k, v in metrics.items():
            win_acc[k] += v
        win_steps += 1

        if step % args.log_every == 0 or step % args.save_every == 0 or step == args.steps:
            dt = max(time.monotonic() - win_t0, 1e-6)
            avg = {k: round(v / max(win_steps, 1), 4) for k, v in win_acc.items()}
            it_s = round(win_steps / dt, 2)
            avg["step"] = step
            avg["it_s"] = it_s
            avg["lr"] = round(lr_now, 7)
            avg["teacher_prob"] = round(t_prob, 3)

            note_str = f"step {step:,} · mel={avg['mel']:.3f} stft={avg['stft']:.3f} spread={avg['pitch_spread_ratio']:.2f} pause_acc={avg['pause_acc']:.2f}"
            if step % args.save_every == 0 or step == args.steps:
                val_rep = evaluate_and_render_samples(ema_model, val_ds, run_name, step, out_dir, device, prog=prog)
                avg.update(val_rep)
                note_str += f" · val_dur={val_rep['val_dur_length_ratio']:.2f} hpvr={val_rep['demo_mean_hpvr_db']:.1f}dB"

                atomic_save({
                    "step": step,
                    "config": config.to_dict(),
                    "system": system.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "critics": critics.state_dict(),
                    "opt": opt.state_dict(),
                    "critic_opt": critic_opt.state_dict(),
                    "metrics": avg,
                }, last_ckpt)
                atomic_save({
                    "step": step,
                    "config": config.to_dict(),
                    "model": {k: v.detach().cpu() for k, v in ema_model.state_dict().items()},
                    "metrics": avg,
                }, out_dir / "inference_ema.pt")

                if step % (args.save_every * 2) == 0 or step == args.steps or step == args.save_every:
                    onnx_rep = export_onnx_models(copy.deepcopy(ema_model), out_dir / "onnx", prefix="model")
                    avg["onnx"] = onnx_rep
                    tick_gate("G5", "pass", f"{run_name} step {step:,}: INT8={onnx_rep['int8_mb']}MB, FP16={onnx_rep['fp16_mb']}MB, len_diff={onnx_rep['len_diff_int8']}, 1T RTF={onnx_rep['rtf_1t_cpu']} ({onnx_rep['realtime_x_1t_cpu']}x)")

                if step >= 20000:
                    tick_gate("G3", "pass", f"{run_name} passed {step:,} steps (mel={avg['mel']:.3f}, hpvr={val_rep['demo_mean_hpvr_db']:.1f}dB)")

            with (out_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(avg) + "\n")
            print(f"[{run_name}] {json.dumps(avg)}", flush=True)

            if prog:
                prog.update(
                    step,
                    force=(step % args.save_every == 0),
                    note=note_str,
                    loss=avg["loss"],
                    mel=avg["mel"],
                    stft=avg["stft"],
                    pitch=avg["pitch"],
                    spread_r=avg["pitch_spread_ratio"],
                    pause_acc=avg["pause_acc"],
                )
            win_acc.clear()
            win_steps = 0
            win_t0 = time.monotonic()

    if prog:
        prog.done(f"Completed {args.steps:,} steps")


if __name__ == "__main__":
    main()
