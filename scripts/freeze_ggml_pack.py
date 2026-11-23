#!/usr/bin/env python3
"""Freeze the 15-clip master pack rendered by the GGML C++ engine (sakhiya-cli) for a GGUF model, e.g.
   python scripts/freeze_ggml_pack.py --tag v15_ggml_atom_lite --gguf /tmp/gguf/sakhiya_atom_lite_q8_0.gguf --cli build/sakhiya-cli --label "..."
Immutable: refuses to overwrite a directory that already has a .frozen lock."""
import argparse, datetime, json, os, re, subprocess, sys, tempfile
import numpy as np, soundfile as sf
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scripts.freeze_version_pack import MASTER_15_CLIPS

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--label", required=True); ap.add_argument("--gguf", required=True)
ap.add_argument("--cli", default="build/sakhiya-cli"); ap.add_argument("--notes", default=""); ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--extra-metrics", default="{}")
a = ap.parse_args()
out_dir = os.path.join("docs", "samples", "versions", a.tag)
lock = os.path.join(out_dir, ".frozen")
if os.path.exists(lock):
    sys.exit(f"[IMMUTABLE GUARD] {out_dir} already frozen")
os.makedirs(out_dir, exist_ok=True)
try:
    import imageio_ffmpeg; ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    ffmpeg = "ffmpeg"
clips, rtfs = [], []
for cid, lang, style, text in MASTER_15_CLIPS:
    with tempfile.TemporaryDirectory() as t:
        wav = os.path.join(t, "o.wav")
        r = subprocess.run([a.cli, "-m", a.gguf, "-t", text, "-l", lang, "-s", style, "-o", wav, "-j", str(a.threads)], check=True, capture_output=True, text=True)
        m = re.search(r"RTF=([0-9.]+)", r.stderr + r.stdout); rtfs.append(float(m.group(1)) if m else float("nan"))
        w, sr = sf.read(wav, dtype="float32")
        assert np.isfinite(w).all()
        mp3 = os.path.join(out_dir, f"{cid}.mp3")
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", wav, "-codec:a", "libmp3lame", "-qscale:a", "3", mp3], check=True)
    clips.append(dict(id=cid, lang=lang, style=style, text=text, mp3=f"samples/versions/{a.tag}/{cid}.mp3", duration_s=round(len(w) / sr, 2)))
    print("rendered", cid, clips[-1]["duration_s"], "s RTF", rtfs[-1], flush=True)
sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
metrics = dict(ne_pcer=None, ne_wer=None, sa_pcer=None, utmos=None, sv_cos_sim=None, cross_style_sv=None, rigveda_svara_st=None,
               style_f0_ratio=None, glitch_frames=None, word_drops=None, ggml_gguf_bytes=os.path.getsize(a.gguf),
               ggml_mean_rtf=round(float(np.mean(rtfs)), 4), ggml_threads=a.threads, **json.loads(a.extra_metrics))
entry = dict(tag=a.tag, label=a.label, commit_sha=sha, timestamp_utc=now, notes=a.notes, metrics=metrics, clips=clips)
json.dump(entry, open(os.path.join(out_dir, "manifest.json"), "w"), indent=2, ensure_ascii=False)
json.dump(clips, open(os.path.join(out_dir, "clips_meta.json"), "w"), indent=2, ensure_ascii=False)
open(lock, "w").write(f"Frozen at {now} commit {sha}\n")
reg_path = os.path.join("docs", "samples", "versions", "registry.json")
reg = json.load(open(reg_path))
reg["versions"] = [v for v in reg["versions"] if v["tag"] != a.tag] + [entry]
json.dump(reg, open(reg_path, "w"), indent=2, ensure_ascii=False)
print("FROZEN", a.tag, len(clips), "clips")
