"""Python wrapper for the Sakhiya GGML C++ engine (``cpp/sakhiya.h``) — drop-in sibling of ``SakhiyaONNX``.

    from sakhiya.ggml_tts import SakhiyaGGML
    tts = SakhiyaGGML.from_pretrained("atom-lite")            # downloads + caches sakhiya_atom_lite_q8_0.gguf (~3.0 MB)
    wav, sr = tts.say("नेपालको राजधानी काठमाडौँ हो।", style="news", speed=1.0, pitch=0.0, energy=1.0)
    wavs = tts.say_batch([("नमस्ते", "ne", "warm_soft"), ("ॐ भूर्भुवः स्वः।", "sa", "vedic_chant")])

Backends (auto-selected): in-process ``libsakhiya`` via ctypes (zero subprocess, zero-copy numpy view), else the ``sakhiya-cli``
binary. No PyTorch / ONNX needed. Tiers: ``v1`` | ``nano`` | ``atom`` | ``atom-lite``; quants: ``q8_0`` (default, smallest),
``q8_0_exact`` (waveform-exact vs PyTorch), ``f16``, ``f32``.
"""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HF_REPO = "NikitHamal/sakhiya-gguf"
SAMPLE_RATE = 24000
TIER_ALIASES = {
    "v1": "v1", "v25": "v1", "flagship": "v1",
    "nano": "nano", "voca": "nano",
    "atom": "atom", "sano": "atom",
    "atom-lite": "atom_lite", "atom_lite": "atom_lite", "atomlite": "atom_lite", "lite": "atom_lite",
}
STYLES = ("neutral", "news", "conversational", "storytelling", "warm_soft", "slow_clear", "expressive", "vedic_chant")


class _Params(ctypes.Structure):
    _fields_ = [("lang", ctypes.c_char_p), ("style", ctypes.c_char_p), ("rate", ctypes.c_float),
                ("pitch_shift", ctypes.c_float), ("pitch_scale", ctypes.c_float), ("pause_scale", ctypes.c_float),
                ("energy_scale", ctypes.c_float), ("svara", ctypes.c_float)]


def _find_lib() -> Optional[str]:
    names = ["libsakhiya.so", "libsakhiya.dylib", "sakhiya.dll"]
    cands = [os.environ.get("SAKHIYA_LIB", "")]
    for d in (os.path.dirname(__file__), os.path.join(REPO_ROOT, "build"), os.path.join(REPO_ROOT, "build-lib"), REPO_ROOT):
        cands += [os.path.join(d, n) for n in names]
    return next((c for c in cands if c and os.path.exists(c)), None)


def build_native(out_dir: Optional[str] = None) -> str:
    """Compile libsakhiya (single g++ call, no ggml / no third-party deps) from cpp/ and return its path."""
    out_dir = out_dir or os.path.join(REPO_ROOT, "build")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "libsakhiya.so")
    cmd = ["g++", "-std=c++17", "-O3", "-fPIC", "-shared", "-DSAKHIYA_STANDALONE_GGUF", "-fopenmp", "-funroll-loops",
           "-fno-math-errno", "-fno-trapping-math", "-I", os.path.join(REPO_ROOT, "cpp"),
           os.path.join(REPO_ROOT, "cpp", "sakhiya_capi.cpp"), "-o", out]
    if os.uname().machine in ("x86_64", "AMD64"):
        cmd[5:5] = ["-mavx2", "-mfma"]
    subprocess.run(cmd, check=True)
    return out


def download_gguf(tier: str = "atom-lite", quant: str = "q8_0", cache_dir: Optional[str] = None,
                  repo_id: str = HF_REPO, revision: Optional[str] = None) -> str:
    """Download (and cache under ~/.cache/huggingface or ``cache_dir``) ``sakhiya_<tier>_<quant>.gguf`` from Hugging Face."""
    key = TIER_ALIASES.get(tier.strip().lower())
    if key is None:
        raise ValueError(f"unknown tier {tier!r}; choose one of {sorted(set(TIER_ALIASES.values()))}")
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id, f"sakhiya_{key}_{quant}.gguf", cache_dir=cache_dir, revision=revision,
                           token=os.environ.get("HF_TOKEN") or None)


class SakhiyaGGML:
    """High-speed GGML C++ TTS (bit-exact with the PyTorch reference in the F32 / q8_0_exact profiles)."""

    def __init__(self, model_path: str, cli_path: Optional[str] = None, threads: int = 4, lib_path: Optional[str] = None,
                 backend: str = "auto"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        self.model_path = model_path
        self.threads = int(threads)
        self.sr = SAMPLE_RATE
        self._lib = None
        self._ctx = None
        self.cli_path = cli_path or next((c for c in (os.path.join(REPO_ROOT, "build", "sakhiya-cli"),
                                                       os.path.join(REPO_ROOT, "build", "sakhiya-cli-lite"),
                                                       os.path.join(REPO_ROOT, "sakhiya-cli"), shutil.which("sakhiya-cli") or "")
                                          if c and os.path.exists(c)), None)
        if backend in ("auto", "lib"):
            lp = lib_path or _find_lib()
            if lp is None and backend == "lib":
                lp = build_native()
            if lp:
                self._load_lib(lp)
            elif backend == "lib":
                raise RuntimeError("libsakhiya not found")
        if self._ctx is None and not self.cli_path:
            raise RuntimeError("Neither libsakhiya nor sakhiya-cli found. Run `python -c 'from sakhiya.ggml_tts import build_native; "
                               "build_native()'` or set SAKHIYA_LIB / pass cli_path.")

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_pretrained(cls, tier: str = "atom-lite", quant: str = "q8_0", threads: int = 4, cache_dir: Optional[str] = None,
                        **kw) -> "SakhiyaGGML":
        return cls(download_gguf(tier, quant, cache_dir), threads=threads, **kw)

    def _load_lib(self, path: str) -> None:
        L = ctypes.CDLL(path)
        L.sakhiya_default_params.restype = _Params
        L.sakhiya_init_from_file.restype = ctypes.c_void_p
        L.sakhiya_init_from_file.argtypes = [ctypes.c_char_p, ctypes.c_int]
        L.sakhiya_free.argtypes = [ctypes.c_void_p]
        L.sakhiya_synthesize.restype = ctypes.c_int64
        L.sakhiya_synthesize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, _Params, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
                                         ctypes.POINTER(ctypes.c_int)]
        L.sakhiya_free_pcm.argtypes = [ctypes.c_void_p]
        L.sakhiya_last_error.restype = ctypes.c_char_p
        L.sakhiya_model_tier.restype = ctypes.c_char_p
        L.sakhiya_model_tier.argtypes = [ctypes.c_void_p]
        ctx = L.sakhiya_init_from_file(self.model_path.encode(), self.threads)
        if not ctx:
            raise RuntimeError("sakhiya_init_from_file failed: " + (L.sakhiya_last_error() or b"").decode())
        self._lib, self._ctx = L, ctx

    @property
    def tier(self) -> str:
        return self._lib.sakhiya_model_tier(self._ctx).decode() if self._ctx else os.path.basename(self.model_path)

    @property
    def backend(self) -> str:
        return "libsakhiya" if self._ctx else "sakhiya-cli"

    def close(self) -> None:
        if self._ctx and self._lib:
            self._lib.sakhiya_free(self._ctx)
            self._ctx = None

    def __del__(self):  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self): return self
    def __exit__(self, *a): self.close()

    # ------------------------------------------------------------------ synthesis
    def say(self, text: str, speaker: str = "sakhiya", style: str = "news", lang: str = "ne",
            rate: Optional[float] = None, pitch_shift: Optional[float] = None, pitch_scale: Optional[float] = None,
            pause_scale: Optional[float] = None, energy_scale: Optional[float] = None, svara: float = 1.0,
            *, speed: Optional[float] = None, pitch: Optional[float] = None, energy: Optional[float] = None,
            chandas: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """Synthesize ``text`` -> (float32 mono wav, 24000).

        speed (=rate) 0.5-2.0, pitch (=pitch_shift, octaves) -0.6..0.6, pitch_scale 0.5-1.8 (expressiveness),
        pause_scale 0.3-2.5, energy (=energy_scale) 0.6-1.6, svara/chandas = Vedic Svara depth (lang="sa"). ``lang="auto"``
        routes mixed Devanagari/Roman text. Unset values use the style preset."""
        rate = speed if speed is not None else rate
        pitch_shift = pitch if pitch is not None else pitch_shift
        energy_scale = energy if energy is not None else energy_scale
        svara = chandas if chandas is not None else svara
        if self._ctx:
            return self._say_lib(text, lang, style, rate, pitch_shift, pitch_scale, pause_scale, energy_scale, svara), self.sr
        return self._say_cli(text, lang, style, rate, pitch_shift, pitch_scale, pause_scale, energy_scale, svara), self.sr

    def _say_lib(self, text, lang, style, rate, ps, psc, pause, eng, svara) -> np.ndarray:
        p = self._lib.sakhiya_default_params()
        p.lang, p.style = lang.encode(), style.encode()
        if rate is not None: p.rate = rate
        if ps is not None: p.pitch_shift = ps
        if psc is not None: p.pitch_scale = psc
        if pause is not None: p.pause_scale = pause
        if eng is not None: p.energy_scale = eng
        p.svara = svara
        out = ctypes.POINTER(ctypes.c_float)()
        sr = ctypes.c_int(0)
        n = self._lib.sakhiya_synthesize(self._ctx, text.encode("utf-8"), p, ctypes.byref(out), ctypes.byref(sr))
        if n < 0:
            raise RuntimeError("sakhiya_synthesize failed: " + (self._lib.sakhiya_last_error() or b"").decode())
        try:
            return np.ctypeslib.as_array(out, shape=(int(n),)).copy()
        finally:
            self._lib.sakhiya_free_pcm(ctypes.cast(out, ctypes.c_void_p))

    def _say_cli(self, text, lang, style, rate, ps, psc, pause, eng, svara) -> np.ndarray:
        import soundfile as sf
        with tempfile.TemporaryDirectory() as tmpd:
            wav_p = os.path.join(tmpd, "out.wav")
            cmd = [self.cli_path, "-m", self.model_path, "-t", text, "-l", lang, "-s", style, "-o", wav_p,
                   "-j", str(self.threads), "--svara", str(svara)]
            for flag, v in (("--rate", rate), ("--pitch-shift", ps), ("--pitch-scale", psc), ("--pause-scale", pause),
                            ("--energy-scale", eng)):
                if v is not None:
                    cmd += [flag, str(v)]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            wav, _ = sf.read(wav_p, dtype="float32")
            return wav

    def say_batch(self, items: Iterable[Union[str, Sequence]], lang: str = "ne", style: str = "news", workers: int = 1,
                  **kw) -> List[np.ndarray]:
        """Batch synthesis. Each item is ``text`` or ``(text, lang[, style[, kwargs_dict]])``. The engine context is reentrant for
        reads, so ``workers>1`` synthesizes several utterances concurrently (set ``threads=1`` for best scaling)."""
        jobs = []
        for it in items:
            if isinstance(it, str):
                jobs.append((it, lang, style, kw))
            else:
                t = list(it) + [None] * 4
                jobs.append((t[0], t[1] or lang, t[2] or style, {**kw, **(t[3] or {})}))

        def run(j):
            return self.say(j[0], lang=j[1], style=j[2], **j[3])[0]

        if workers <= 1 or len(jobs) < 2:
            return [run(j) for j in jobs]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(run, jobs))

    def save(self, path: str, text: str, **kw) -> str:
        import soundfile as sf
        wav, sr = self.say(text, **kw)
        sf.write(path, wav, sr)
        return path


if __name__ == "__main__":  # python -m sakhiya.ggml_tts atom-lite "नमस्ते" out.wav
    tier = sys.argv[1] if len(sys.argv) > 1 else "atom-lite"
    txt = sys.argv[2] if len(sys.argv) > 2 else "नेपालको राजधानी काठमाडौँ हो।"
    out = sys.argv[3] if len(sys.argv) > 3 else "sakhiya_ggml.wav"
    SakhiyaGGML.from_pretrained(tier).save(out, txt)
    print("wrote", out)
