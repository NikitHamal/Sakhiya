"""High-level synthesis API:  Sakhiya().say("नमस्ते", rate=1.0, style="storytelling")"""
import numpy as np
import torch
from . import g2p, symbols as S
from .acoustic import Acoustic
from .vocoder import Vocoder, load_compat
from .audio import SR

STYLES = {   # commercial controls: rate, pitch_shift (z), pitch_scale, energy_scale, pause_scale
    "neutral": dict(rate=1.0, pitch_shift=0.0, pitch_scale=1.0, energy_scale=1.0, pause_scale=1.0),
    "conversational": dict(rate=1.08, pitch_shift=0.1, pitch_scale=1.15, energy_scale=1.0, pause_scale=0.8),
    "storytelling": dict(rate=0.88, pitch_shift=0.0, pitch_scale=1.35, energy_scale=1.05, pause_scale=1.4),
    "news": dict(rate=1.0, pitch_shift=-0.1, pitch_scale=0.85, energy_scale=1.1, pause_scale=1.0),
    "slow_clear": dict(rate=0.78, pitch_shift=0.0, pitch_scale=1.0, energy_scale=1.0, pause_scale=1.3),
}
PAUSE_IDS = tuple(S.SYM2ID[p] for p in [S.WORD] + S.PUNCT)


class Sakhiya:
    def __init__(self, am_ckpt, voc_ckpt, device="cpu"):
        c = torch.load(am_ckpt, map_location=device, weights_only=False)
        self.am = Acoustic().to(device).eval()
        self.am.load_state_dict(c["M"])
        self.spk2id = c["spk2id"]
        self.voc = Vocoder().to(device).eval()
        sd = torch.load(voc_ckpt, map_location=device)
        load_compat(self.voc, sd["G"] if "G" in sd else sd)
        self.device = device

    def say(self, text, speaker=None, lang="ne", style="neutral", **ctl):
        k = dict(STYLES[style]); k.update(ctl)
        ph = g2p.text_to_phones(text, lang)
        ids = torch.tensor([S.encode(ph)], device=self.device)
        spk = self.spk2id.get(speaker, 0) if speaker else 0
        with torch.no_grad():
            mel, dur = self.am.infer(ids, spk=spk, lang=S.LANG2ID.get(lang, 0), pause_ids=PAUSE_IDS, **k)
            wav = self.voc(mel)[0].clamp(-1, 1).cpu().numpy()
        return wav, SR
