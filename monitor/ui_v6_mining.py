"""v6 mining monitor: 5 live cards (Mining · Separation/VAD · Quality gate · ASR/Punctuation/Pack · VC/Archive) + BAD counter.
Pure functions -> HTML (reuse the flat M3X dark palette from ui26). Reads /work/tts state files only; never raises."""
import glob, json, os, time
try:
    import ui26 as U
except Exception:
    U = None
R = os.environ.get("TTS_ROOT", "/work/tts")
E = lambda s: __import__("html").escape(str(s))
chip = lambda t, k="": f'<span class="chip {k}">{E(t)}</span>'
tile = lambda v, l: f'<div class="tl"><b>{E(v)}</b><small>{E(l)}</small></div>'

def _jl(p):
    try: return [json.loads(l) for l in open(p)]
    except Exception: return []

def _hours(d):
    try: return sum(os.path.getsize(f) for f in glob.glob(d + "/*.wav")) / 48000 / 3600
    except Exception: return 0.0

def _bad():
    n = 0
    for f in glob.glob(f"{R}/logs/*.log"):
        try: n += sum(1 for l in open(f, errors="ignore") if "Traceback" in l or " ERR " in l or "FAIL" in l and "HF snapshot" not in l)
        except Exception: pass
    return n

def _card(title, body, ok=True):
    return f'<div class="cd"><h3>{E(title)}<span class="sp"></span>{chip("BAD 0" if ok else "check logs", "ok" if ok else "bad")}</h3>{body}</div>'

def _state():
    """Collect all v6 pipeline telemetry once per refresh (every reader swallows its own errors)."""
    st = {}
    try: st = json.load(open(f"{R}/mining_state.json"))
    except Exception: pass
    cats = ["storytelling", "vedic_chant", "news", "conversational"]
    S = {"st": st, "cats": cats, "bad": _bad()}
    S["h"] = {c: _hours(f"{R}/youtube_raw/{c}") for c in cats}
    S["seg"] = len(glob.glob(f"{R}/segments/*/*.done"))
    S["nseg"] = 0
    for p in glob.glob(f"{R}/segments/segments.*.jsonl"):
        try: S["nseg"] += sum(1 for _ in open(p))
        except Exception: pass
    S["gate"] = [r for p in glob.glob(f"{R}/gate/gate_*.jsonl") for r in _jl(p)]
    S["passed"] = _jl(f"{R}/gate/passed.jsonl"); S["asr"] = _jl(f"{R}/asr/asr.jsonl"); S["meta"] = _jl(f"{R}/dataset/meta.jsonl")
    S["vc"] = len(glob.glob(f"{R}/dataset/vc/*.parquet")); S["raw"] = len(glob.glob(f"{R}/dataset/raw/*.parquet"))
    return S

def _gpu():
    try:
        import subprocess
        o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.split(",")
        return [float(x) for x in o]
    except Exception: return None

def _wrap(fn):
    def w():
        try:
            S = _state(); body, ok = fn(S)
            return (U.CSS if U else "") + f'<div class="sk" style="display:block">{_card(*body, ok=S["bad"] < 5)}</div>' if ok is None else ""
        except Exception as e:
            return f'<div class="cd">{chip("monitor soft-fail", "wait")} <small>{E(e)}</small></div>'
    return w

@_wrap
def card1(S):
    st = S["st"]
    return ("1 · Discovery & ingestion", '<div class="tls">' + "".join(tile(f"{v:.1f} h", k) for k, v in S["h"].items()) + "</div>" + f'<small>stage <b>{E(st.get("stage"))}</b> · cycle {st.get("cycle")} · sources: Internet Archive + HF Vagdhenu/Rasa (YouTube bot-walled on datacenter IP)</small>'), None

@_wrap
def card2(S):
    return ("2 · Demucs vocals + Silero-VAD", f'<div class="tls">{tile(S["seg"], "files done")}{tile(S["nseg"], "2.5–14 s segments")}</div><small>htdemucs vocal stem · 150 ms tail kept · cuts only at ≥0.9 s gaps</small>'), None

@_wrap
def card3(S):
    g = S["gate"]; um = sum(r.get("utmos", 0) for r in g) / max(1, len(g))
    return ("3 · Quality & speaker gate", f'<div class="tls">{tile(len(g), "scored")}{tile(len(S["passed"]), "passed")}{tile(f"{um:.2f}", "mean UTMOS22")}</div><small>UTMOS · SNR · clipping · clicks · WavLM-SV switch score</small>'), None

@_wrap
def card4(S):
    meta = S["meta"]; sty = {}
    for m in meta: sty[m.get("style", "?")] = sty.get(m.get("style", "?"), 0) + m.get("dur", 0) / 3600
    gold = sum(1 for m in meta if m.get("tier") == "gold")
    return ("4 · ASR · punctuation · alignment", f'<div class="tls">{tile(len(S["asr"]), "ASR+aligned")}{tile(len(meta), "packed")}{tile(gold, "gold tier")}</div><small>' + (" · ".join(f"{k} {v:.1f} h" for k, v in sty.items()) or "no clip packed yet") + "</small>"), None

@_wrap
def card5(S):
    return ("5 · sakhiya VC & archive", f'<div class="tls">{tile(S["vc"], "VC shards")}{tile(S["raw"], "raw shards")}</div><small>HF private NikitHamal/sakhiya-nepali-vedic-private · Kaggle private · GH release v6-ckpt</small>'), None

def header():
    try:
        g = _gpu(); hb = (json.load(open(f"{R}/heartbeat.json")) or {}).get("t")
        age = time.time() - hb if hb else None
        gc = (tile(f"{g[0]:.0f} %", "GPU util") + tile(f"{g[1]/1024:.1f}/{g[2]/1024:.0f} GB", "VRAM") + tile(f"{g[3]:.0f} °C", "temp") + tile(f"{g[4]:.0f} W", "power")) if g else tile("-", "GPU")
        return (U.CSS if U else "") + (f'<div class="sk" style="display:block"><div class="cd w"><h3>Sakhiya v6 · YouTube/Open-Web Nepali + Vedic mining on RTX PRO 6000 Blackwell{chip("keep-alive live" if age is not None and age < 150 else "starting", "ok" if age is not None and age < 150 else "wait")}'
                f'<span class="sp"></span>{chip(time.strftime("%H:%M:%S"))}</h3><div class="tls">{gc}</div></div></div>')
    except Exception as e:
        return f'<div class="cd">{chip("monitor soft-fail", "wait")} <small>{E(e)}</small></div>'

def cards():
    return header() + card1() + card2() + card3() + card4() + card5()
