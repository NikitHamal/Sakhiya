"""Sakhiya v8 owner cards: 5 plain-English cards (progress, hear-the-cleaning, hear-kept-clips, hear-sakhiya, backups)."""
import glob, json, os, random, time
import importlib.util as _iu, sys
_here = os.path.dirname(os.path.abspath(__file__))
_s = _iu.spec_from_file_location("ui8_core", _here + "/ui8_core.py"); C = _iu.module_from_spec(_s); sys.modules["ui8_core"] = C; _s.loader.exec_module(C)
E, R, pill, big, bar, wrap = C.E, C.R, C.pill, C.big, C.bar, C.wrap

def safe(fn):
    def w(*a, **k):
        try: return fn(*a, **k)
        except Exception as e: return wrap(f"<h2>Loading…</h2><p>This card will fill in by itself. ({E(type(e).__name__)})</p>")
    return w

def header():
    g = C.gpu(); r = C.running(); on = sum(1 for _, x in r if x)
    gp = (big(f"{g[0]:.0f}%", "GPU busy") + big(f"{g[1]/1024:.0f} of {g[2]/1024:.0f} GB", "GPU memory in use")) if g else big("-", "GPU")
    return wrap(f"<h2>Sakhiya · Nepali & Vedic voice factory</h2><p>This page shows, in plain words, how our collected audio is turning into the Sakhiya voice. "
                f"Cards 2, 3 and 4 let you <b>listen</b>. Press <b>New clips</b> to hear different examples.</p><div class='row'>{gp}{big(f'{on} jobs', 'running right now')}"
                f"{big(time.strftime('%H:%M'), 'last refreshed (box time)')}</div>")

@safe
def card1():
    seg = C.count_lines(f"{R}/segments/segments.0.jsonl") if False else sum(C.count_lines(p) for p in glob.glob(f"{R}/segments/segments.*.jsonl"))
    scored = sum(C.count_lines(p) for p in glob.glob(f"{R}/gate/gate_*.jsonl"))
    passed = C.count_lines(f"{R}/gate/passed.jsonl"); asr = C.jl(f"{R}/asr/asr.jsonl")
    agree = sum(1 for a in asr if a.get("agree", 0) >= 0.94); packed = C.count_lines(f"{R}/dataset/meta.jsonl")
    tot = max(1, seg)
    funnel = [("Sentences cut from the audio", seg, "after removing background music"), ("Checked for quality", scored, "sound clean, one speaker, no clicks"),
              ("Passed quality check", passed, "good enough to learn from"), ("Words written down", len(asr), "two independent listeners"),
              ("Both listeners agree (94%+)", agree, "so we trust the text"), ("Packed with feeling marks", packed, "ready for training")]
    rows = "".join(f"<div><b>{E(n)}</b> — {v:,} <small>· {E(d)}</small>{bar(v / tot)}</div>" for n, v, d in funnel)
    hrs = "".join(big(f"{C.hours(k):.1f} h", C.STYLES[k][0]) for k in C.STYLES)
    steps = "".join(f"<div class='step'><span class='dot {'on' if o else ''}'></span>{E(l)}{pill('working', 'ok') if o else pill('idle', 'tag')}</div>" for l, o in C.running())
    return wrap(f"<h2>1 · Where we are</h2><h3>Hours of audio collected</h3><div class='row'>{hrs}</div><h3>From raw audio to training-ready clips</h3>{rows}"
                f"<h3>What the machine is doing now</h3>{steps}")

@safe
def card2(shuffle=None):
    segs = [r for p in glob.glob(f"{R}/segments/segments.*.jsonl") for r in C.jl(p, 4000) if 4 <= r.get("dur", 0) <= 9]
    out = []
    for cat, (name, desc) in C.STYLES.items():
        pool = [r for r in segs if r.get("cat") == cat]
        if not pool: out.append(f"<div class='clip'><b>{E(name)}</b> {pill('none yet', 'wait')}<small> — nothing cut for this style yet</small></div>"); continue
        r = random.choice(pool); raw = f"{R}/youtube_raw/{cat}/{r['src']}.wav"
        b = C.audio(raw, r["t0"], r["dur"]); a = C.audio(r["path"])
        out.append(f"<div class='clip'><b>{E(name)}</b> <small>· {E(desc)}</small><div class='row'><div style='flex:1;min-width:240px'><small>BEFORE (as found, may have music)</small>{b}</div>"
                   f"<div style='flex:1;min-width:240px'><small>AFTER (voice only, cleaned)</small>{a}</div></div></div>")
    return wrap("<h2>2 · Listen: music removed</h2><p>The same moment, before and after the machine removes background music and echo.</p>" + "".join(out))

@safe
def card3(shuffle=None):
    meta = C.jl(f"{R}/dataset/meta.jsonl", 3000); asr = C.jl(f"{R}/asr/asr.jsonl")
    pool = meta or sorted(asr, key=lambda a: -a.get("agree", 0))[:150]
    out = []
    for cat, (name, desc) in C.STYLES.items():
        p = [r for r in pool if r.get("cat", r.get("style")) == cat]
        if not p: out.append(f"<div class='clip'><b>{E(name)}</b> {pill('not ready', 'wait')}<small> — no clip has finished all checks yet</small></div>"); continue
        r = random.choice(p); path = r.get("path") or r.get("wav", "")
        txt = r.get("text") or r.get("text_m") or r.get("text_w", "")
        out.append(f"<div class='clip'><b>{E(name)}</b> {pill('quality ' + format(r.get('utmos', 0), '.2f'), 'ok')}{pill('agreement ' + format(r.get('agree', 0) * 100, '.0f') + '%', 'tag')}"
                   f"<div class='dev'>{C.dev_text(txt)}</div>{C.audio(path)}</div>")
    return wrap("<h2>3 · Listen: clips that passed our checks</h2><p>Real human recordings kept for training, with the words the machine heard. "
                "Marks: <span class='ex'>!</span> excitement, <span class='qm'>?</span> question, <span class='cm'>,</span> breath pause. "
                "Quality is a 1–5 score (higher is better).</p>" + "".join(out))

@safe
def card4(shuffle=None):
    V = f"{C.REPO}/docs/samples/versions"; names = {"ne_story": "Nepali storytelling", "ne_expressive": "Nepali with feeling", "ne_chat": "Nepali everyday chat", "ne_kathmandu": "Nepali place names", "sa_gayatri": "Gayatri Mantra"}
    vers = [("v5_final", "Previous (v5)"), ("v6_teacher", "Newest big model (v6)"), ("v6_voca_4m", "Small browser model (~4.5 MB)"), ("v6_sano_1p5m", "Tiny browser model (~1.6 MB)")]
    keys = [k for k in names if any(os.path.exists(f"{V}/{v}/{k}.mp3") for v, _ in vers)] or list(names)
    ks = random.sample(keys, min(2, len(keys))); out = []
    for k in ks:
        cols = "".join(f"<div style='flex:1;min-width:220px'><small>{E(l)}</small>{C.audio(f'{V}/{v}/{k}.mp3')}</div>" for v, l in vers if os.path.exists(f"{V}/{v}/{k}.mp3"))
        out.append(f"<div class='clip'><b>{E(names[k])}</b><div class='row'>{cols}</div></div>")
    vc = len(glob.glob(f"{R}/dataset/vc/*.parquet"))
    return wrap("<h2>4 · Listen: the Sakhiya voice</h2><p>Same sentence, different model sizes. Please judge honestly — the two tiny models are still experimental and are not yet as good as the big one.</p>"
                + "".join(out) + f"<p>Mined voices converted into Sakhiya so far: {pill(str(vc) + ' packs', 'ok' if vc else 'wait')}</p>")

_hf = {"t": 0, "v": None}
def _hf_info():
    if time.time() - _hf["t"] < 300: return _hf["v"]
    _hf["t"] = time.time()
    try:
        from huggingface_hub import HfApi
        i = HfApi(token=open("/tmp/hf_token").read().strip()).dataset_info("NikitHamal/sakhiya-nepali-vedic-private", files_metadata=True)
        _hf["v"] = (i.private, len(i.siblings or []), sum((s.size or 0) for s in (i.siblings or [])))
    except Exception: _hf["v"] = None
    return _hf["v"]

@safe
def card5():
    st = {}
    try: st = json.load(open(f"{R}/v6_state.json"))
    except Exception: pass
    h = _hf_info()
    hf = (pill("saved · private", "ok") + f" {h[1]} files, {h[2]/1e9:.2f} GB") if h and h[1] > 1 else (pill("repository ready, no audio uploaded yet", "wait") if h else pill("not created yet", "no"))
    kg = pill("uploaded", "ok") if st.get("kaggle") else pill("nothing uploaded yet", "wait")
    gh = pill("saved", "ok") if st.get("gh_release") else pill("nothing saved yet", "wait")
    disk = sum(os.path.getsize(f) for f in glob.glob(f"{R}/youtube_raw/*/*.wav")) / 1e9
    return wrap(f"<h2>5 · Is everything safely backed up?</h2><p>Your collected audio and models are copied to private places only you can open.</p>"
                f"<div class='step'>HuggingFace (private) · NikitHamal/sakhiya-nepali-vedic-private {hf}</div><div class='step'>Kaggle (private) {kg}</div>"
                f"<div class='step'>GitHub model archive (v6-ckpt) {gh}</div><div class='step'>On the GPU box right now: {disk:.0f} GB of raw audio {pill('temporary — box can disappear', 'wait')}</div>"
                f"<p>Next up: finish the quality + word checks on more clips, convert them to the Sakhiya voice, then train the new models.</p>")
