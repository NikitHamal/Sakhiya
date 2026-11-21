"""Sakhiya v8 owner cards - core helpers (plain-English, listenable). Reads /work/tts only, never raises.
Audio cards must NOT re-render every few seconds (that would stop playback), so they are driven by a shuffle button."""
import base64, glob, io, json, os, random, re, subprocess, time, html as _h
R = os.environ.get("TTS_ROOT", "/work/tts")
REPO = os.environ.get("REPO_ROOT", "/work/tts/repo")
E = lambda s: _h.escape(str(s))
STYLES = {"storytelling": ("Story narration", "warm novel / storytelling voice"),
          "conversational": ("Everyday talk", "natural speech with ! and ? feeling"),
          "news": ("News reading", "crisp, clear newsreader"),
          "vedic_chant": ("Vedic chanting", "mantras with melody (swar) and rhythm (laya)")}
CSS = """<style>
.v8{font-family:Inter,system-ui,sans-serif;color:#e8ecf4;background:#121826;border:1px solid #26314a;border-radius:16px;padding:18px 20px;margin:6px 0}
.v8 h2{margin:0 0 4px;font-size:21px}.v8 h3{margin:14px 0 6px;font-size:15px;color:#9fb4ff}.v8 p,.v8 small{color:#aab4c8;font-size:14px;line-height:1.5}
.v8 .row{display:flex;flex-wrap:wrap;gap:12px}.v8 .big{background:#1a2338;border-radius:12px;padding:12px 16px;min-width:150px;flex:1}
.v8 .big b{display:block;font-size:26px;color:#fff}.v8 .big span{font-size:13px;color:#9aa6bf}
.v8 .bar{height:12px;background:#222c44;border-radius:7px;overflow:hidden;margin:4px 0 10px}.v8 .bar i{display:block;height:100%;background:linear-gradient(90deg,#5b8cff,#8e6bff)}
.v8 .pill{display:inline-block;padding:3px 11px;border-radius:99px;font-size:12px;font-weight:600;margin-right:6px}
.v8 .ok{background:#12351f;color:#5ee38b}.v8 .wait{background:#3a3010;color:#ffd35c}.v8 .no{background:#3a1a1a;color:#ff8a8a}.v8 .tag{background:#1f2c4d;color:#9fb4ff}
.v8 .clip{background:#1a2338;border-radius:12px;padding:10px 14px;margin:8px 0}.v8 .clip audio{width:100%;height:36px;margin-top:6px}
.v8 .dev{font-size:18px;line-height:1.7;color:#fff;margin:4px 0}.v8 .ex{color:#ff9d5c;font-weight:700}.v8 .qm{color:#5cd0ff;font-weight:700}.v8 .cm{color:#ffd35c;font-weight:700}
.v8 .step{display:flex;align-items:center;gap:10px;padding:7px 0;border-bottom:1px solid #222c44;font-size:15px}.v8 .dot{width:11px;height:11px;border-radius:50%;background:#555}
.v8 .dot.on{background:#43e07f;box-shadow:0 0 8px #43e07f}.v8 .dot.done{background:#5b8cff}</style>"""

def jl(p, cap=None):
    out = []
    try:
        for i, l in enumerate(open(p, errors="ignore")):
            if cap and i >= cap: break
            try: out.append(json.loads(l))
            except Exception: pass
    except Exception: pass
    return out

def count_lines(p):
    try: return sum(1 for _ in open(p, errors="ignore"))
    except Exception: return 0

_cache = {}
def audio(path, t0=None, dur=None, maxs=8.0):
    """Return an <audio> tag with the clip embedded (ogg/vorbis, or the mp3 itself). '' if unreadable."""
    try:
        key = (path, t0, dur)
        if key not in _cache:
            if path.endswith(".mp3"):
                uri = "data:audio/mpeg;base64," + base64.b64encode(open(path, "rb").read()).decode()
            else:
                import soundfile as sf
                info = sf.info(path); sr = info.samplerate
                a = int((t0 or 0) * sr); n = int(min(dur or maxs, maxs) * sr)
                x, _ = sf.read(path, start=a, frames=n, dtype="float32", always_2d=False)
                if x.ndim > 1: x = x.mean(1)
                b = io.BytesIO(); sf.write(b, x, sr, format="OGG", subtype="VORBIS")
                uri = "data:audio/ogg;base64," + base64.b64encode(b.getvalue()).decode()
            if len(_cache) > 80: _cache.clear()
            _cache[key] = uri
        return f'<audio controls preload="none" src="{_cache[key]}"></audio>'
    except Exception:
        return ""

def dev_text(t):
    """Colour the expressive punctuation so a non-coder can SEE the feeling."""
    t = E(t or "")
    t = re.sub(r"(!+)", r'<span class="ex">\1</span>', t); t = re.sub(r"(\?+)", r'<span class="qm">\1</span>', t)
    return re.sub(r"(,)", r'<span class="cm">\1</span>', t)

def pill(t, k="tag"): return f'<span class="pill {k}">{E(t)}</span>'
def big(v, l): return f'<div class="big"><b>{E(v)}</b><span>{E(l)}</span></div>'
def bar(frac): return f'<div class="bar"><i style="width:{max(0, min(1, frac)) * 100:.0f}%"></i></div>'
def wrap(inner): return CSS + f'<div class="v8">{inner}</div>'

def hours(cat):
    try: return sum(os.path.getsize(f) for f in glob.glob(f"{R}/youtube_raw/{cat}/*.wav")) / 48000 / 3600
    except Exception: return 0.0

def running():
    """Plain-English list of what the GPU box is doing right now."""
    names = [("mine_", "Finding & downloading new audio"), ("yt_separate_segment", "Removing music, cutting into sentences"),
             ("yt_gate_speakers", "Checking quality & one-speaker purity"), ("yt_asr_ensemble", "Writing down the words (2 listeners)"),
             ("yt_prosody_pack", "Adding ! ? , feeling marks & packing"), ("v5_vc", "Converting voices into sakhiya"),
             ("export_private", "Backing up privately (HuggingFace / Kaggle)"), ("rsi_v6", "Training Sakhiya v6 models"), ("rsi_youtube_daemon", "Autopilot (keeps everything moving)")]
    try:
        ps = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True, timeout=5).stdout
    except Exception: ps = ""
    return [(lab, any(k in l for l in ps.splitlines())) for k, lab in names]

def gpu():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.split(",")
        return [float(x) for x in o]
    except Exception: return None
