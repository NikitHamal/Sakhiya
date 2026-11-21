"""Sakhiya v13 owner cards: adds card 8 'Gemini gold words' (live progress + listen) and makes the audit card (6) start from Gemini's words.
Data: /work/tts/asr/gemini.jsonl written by scripts/gemini_transcribe.py.  Builds on ui12.  Never raises."""
import json, os, random, sys, time
import importlib.util as _iu
_here = os.path.dirname(os.path.abspath(__file__))
_s = _iu.spec_from_file_location("ui12_cards", _here + "/ui12_cards.py"); U = _iu.module_from_spec(_s); sys.modules["ui12_cards"] = U; _s.loader.exec_module(U)
C, E, R, pill, big, wrap, safe = U.C, U.E, U.R, U.pill, U.big, U.wrap, U.safe
GEM = f"{R}/asr/gemini.jsonl"; BADL = {"tamil", "telugu", "malayalam", "kannada", "english", "other", "none", ""}
_c = {"pos": 0, "rows": {}, "n": 0}
def rows():
    """incremental read of gemini.jsonl (cheap even at 50k rows)"""
    try:
        with open(GEM, "rb") as f:
            f.seek(_c["pos"]); data = f.read(); cut = data.rfind(b"\n")
            if cut >= 0:
                _c["pos"] += cut + 1
                for l in data[:cut].splitlines():
                    try: r = json.loads(l); _c["rows"][r["id"]] = r
                    except Exception: pass
    except Exception: pass
    return _c["rows"]
def state():
    try: return json.load(open(f"{R}/asr/gemini_state.json"))
    except Exception: return {}
_u11 = sys.modules.get("ui11_cards"); _old_pool = _u11._pool if _u11 else None
def _pool(kind, style):
    out = _old_pool(kind, style); g = rows()
    for o in out:
        r = g.get(o["id"])
        if r and r.get("text") and r["lang"] not in BADL:
            o["old"] = o["text"]; o["text"] = r["text"]; o["gem"] = True
    return out
_old_html = _u11.clip_html if _u11 else None
def clip_html(o):
    h = _old_html(o)
    if o.get("gem"): h = h.replace("</div></div>", "</div><div><small>" + pill("words below start from Gemini's listening", "ok") + (" · older guess: " + E(o.get("old", "")[:100]) if o.get("old") else "") + "</small></div></div>", 1) if False else h + "<div class='v8'><small>" + pill("words start from Gemini's listening", "ok") + "</small></div>"
    return h
if _u11: _u11._pool, _u11.clip_html = _pool, clip_html
globals().update({k: getattr(U, k) for k in dir(U) if not k.startswith("__") and k not in globals()})
def _row(r, tag, extra=""):
    p = r["path"]
    return (f"<div class='v8'><div class='clip'>{tag}{pill(C.STYLES.get(r['style'], (r['style'],))[0], 'tag')}<small> id {E(r['id'][:44])}</small>{C.audio(p, maxs=14)}"
            f"<div class='dev' style='font-size:16px'>{E(r['text'][:260]) or '<small>(no words heard)</small>'}</div>{extra}</div></div>")
@safe
def card8_stats():
    g = rows(); st = state(); n = len(g)
    if not n: return wrap("<h2>8 · Gemini gold words</h2><p>Gemini is starting to listen to every clip... " + pill("starting", "ok") + "</p>")
    tot = max(st.get("total", n), n); pct = 100 * n / tot; by = {}
    for r in g.values(): by.setdefault(r["style"], [0, 0]); by[r["style"]][0] += 1; by[r["style"]][1] += bool(r["text"])
    v = [r for r in g.values() if r["style"] == "vedic_chant"]; vs = [r for r in v if r["lang"] not in BADL and r["text"]]
    langs = {}
    for r in v:
        if r not in vs: langs[r["lang"] or "no speech"] = langs.get(r["lang"] or "no speech", 0) + 1
    rate = ""
    try:
        ts = sorted(r["t"] for r in g.values())[-200:]; cpm = len(ts) / max(1, ts[-1] - ts[0]) * 60; left = max(0, st.get("total", n) - n); eta = left / max(cpm, .1) / 60; rate = f"· speed {cpm:.0f} clips / minute · about {eta:.0f} hours left for all {left:,} remaining (Vedic and kept clips go first; Google allows only ~20,000 audio-tokens a minute)" if len(ts) > 20 else ""
    except Exception: pass
    park = st.get("parked") or {}
    bar = f"<div style='background:#333;border-radius:8px;height:16px'><div style='background:#22c55e;height:16px;border-radius:8px;width:{pct:.1f}%'></div></div>"
    lang_txt = " · ".join(f"{k}: {c}" for k, c in sorted(langs.items(), key=lambda x: -x[1])[:5]) or "none yet"
    h = (f"<h2>8 · Gemini gold words</h2><p>Google Gemini is listening to <b>every</b> mined clip again and writing the words. For Vedic clips it also says <b>which language it hears</b>, "
         f"so Tamil / Telugu / English talk is thrown out and only real Sanskrit stays.</p>{bar}<p><small>{n:,} of {tot:,} clips done ({pct:.1f}%) {rate} {pill('finished', 'ok') if st.get('done') else pill('working', 'ok')}</small></p>"
         f"<div class='grid'>{big(f'{n:,}', 'clips re-listened')}{big(f'{sum(x[1] for x in by.values()):,}', 'with words written')}{big(f'{len(vs):,}', 'Vedic = real Sanskrit')}{big(f'{len(v) - len(vs):,}', 'Vedic thrown out (wrong language)')}</div>"
         f"<p><small>Wrong-language Vedic clips heard as -> {E(lang_txt)}</small></p>" + (f"<p><small>Models resting (daily limit reached): {E(', '.join(park))}</small></p>" if park else ""))
    return wrap(h)
@safe
def card8_clips(seed=None):
    g = rows(); rnd = random.Random(seed if seed is not None else time.time()); L = list(g.values())
    ne = [r for r in L if r["style"] != "vedic_chant" and r["text"]]; vs = [r for r in L if r["style"] == "vedic_chant" and r["lang"] not in BADL and r["text"]]; vb = [r for r in L if r["style"] == "vedic_chant" and (r["lang"] in BADL or not r["text"])]
    if not L: return wrap("<p>Clips will appear here once Gemini has listened to some.</p>")
    pick = ([(r, pill("Nepali · Gemini words", "ok"), "") for r in rnd.sample(ne, min(3, len(ne)))] + [(r, pill("Real Sanskrit ✓", "ok") + pill("heard: " + (r["lang"] or "?"), "tag"), "") for r in rnd.sample(vs, min(3, len(vs)))]
            + [(r, pill("THROWN OUT", "no") + pill("Gemini heard: " + (r["lang"] or "no speech"), "tag"), "") for r in rnd.sample(vb, min(3, len(vb)))])
    return wrap("<h3>Listen: what Gemini wrote</h3><p><small>3 Nepali, 3 real Sanskrit, 3 thrown-out Vedic clips. Press <b>New clips</b> for others. To fix any words yourself use card 6 - it now starts from Gemini's words.</small></p>" + "".join(_row(*x) for x in pick))
