"""Python drop-in wrapper for sakhiya-cli (GGML C++ inference engine).
Provides the exact same `.say(text, speaker='sakhiya', style='news', lang='ne', ...)` API as `SakhiyaONNX`.
"""
import os
import subprocess
import tempfile
from typing import Optional, Tuple
import numpy as np
import soundfile as sf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class SakhiyaGGML:
    """High-speed, bit-exact GGML C++ TTS wrapper around `sakhiya-cli`."""

    def __init__(
        self,
        model_path: str = "/work/tts/models/sakhiya_f32.gguf",
        cli_path: Optional[str] = None,
        threads: int = 8,
    ):
        self.model_path = model_path
        if cli_path is None:
            cand = os.path.join(REPO_ROOT, "build", "sakhiya-cli")
            if not os.path.exists(cand):
                cand = os.path.join(REPO_ROOT, "sakhiya-cli")
            cli_path = cand
        self.cli_path = cli_path
        self.threads = int(threads)
        self.sr = 24000

    def say(
        self,
        text: str,
        speaker: str = "sakhiya",
        style: str = "news",
        lang: str = "ne",
        rate: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        pitch_scale: Optional[float] = None,
        pause_scale: Optional[float] = None,
        energy_scale: Optional[float] = None,
        svara: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        with tempfile.TemporaryDirectory() as tmpd:
            wav_p = os.path.join(tmpd, "out.wav")
            cmd = [
                self.cli_path,
                "-m", self.model_path,
                "-t", text,
                "-l", lang,
                "-s", style,
                "-o", wav_p,
                "-j", str(self.threads),
                "--svara", str(svara),
            ]
            if rate is not None:
                cmd += ["--rate", str(rate)]
            if pitch_shift is not None:
                cmd += ["--pitch-shift", str(pitch_shift)]
            if pitch_scale is not None:
                cmd += ["--pitch-scale", str(pitch_scale)]
            if pause_scale is not None:
                cmd += ["--pause-scale", str(pause_scale)]
            if energy_scale is not None:
                cmd += ["--energy-scale", str(energy_scale)]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            wav, sr = sf.read(wav_p, dtype="float32")
            return wav, int(sr)
