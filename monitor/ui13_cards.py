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
        if r and r.get("text") and r["lang"] not in BADL and r.get("style") != "vedic_chant":
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
_old = {"t": 0, "d": {}}
def _whisper(cid):
    """older Whisper/MMS guess for the same clip (to show next to Gemini's words); cached for 10 min"""
    if time.time() - _old["t"] > 600:
        d = {}
        try:
            for l in open(f"{R}/asr/asr.jsonl"):
                try: r = json.loads(l); d[os.path.basename(r["path"])[:-4]] = r.get("text_w") or r.get("text") or ""
                except Exception: pass
        except Exception: pass
        _old.update(t=time.time(), d=d)
    return _old["d"].get(cid, "")
def _nep(g): return [r for r in g.values() if r["style"] != "vedic_chant" and "/vedic_chant/" not in r.get("path", "")]
@safe
def card8_stats():
    g = rows(); st = state(); L = _nep(g); n = len(L)
    if not n: return wrap("<h2>8 · Gemini Nepali words</h2><p>Gemini is starting to listen to the Nepali clips... " + pill("starting", "ok") + "</p>")
    done = sum(bool(r["text"]) for r in L); tot = max(st.get("total", n), n); pct = 100 * done / max(tot, 1); by = {}
    for r in L: by.setdefault(r["style"], [0, 0]); by[r["style"]][0] += 1; by[r["style"]][1] += bool(r["text"])
    rate = ""
    try:
        ts = sorted(r["t"] for r in L)[-200:]; cpm = len(ts) / max(1, ts[-1] - ts[0]) * 60; left = max(0, tot - n); eta = left / max(cpm, .1) / 60
        rate = f"Speed {cpm:.0f} clips a minute. Google lets us stream audio only at real-time speed, so the best clips go first (kept, then checked, then the rest); the whole backlog (about {left:,} clips) needs roughly {eta:.0f} hours." if len(ts) > 20 else ""
    except Exception: pass
    conf = sum(1 for r in L if r['text'] and r['lang'] == 'nepali')
    bar = f"<div style='background:#333;border-radius:8px;height:16px'><div style='background:#22c55e;height:16px;border-radius:8px;width:{min(pct, 100):.1f}%'></div></div>"
    sty = " · ".join(f"{C.STYLES.get(k, (k,))[0]}: {v[1]:,}" for k, v in sorted(by.items()))
    h = (f"<h2>8 · Gemini Nepali words</h2><p>Google Gemini listens to <b>Nepali clips only</b> and writes the words with ? ! and , as it hears them. "
         f"<b>Sanskrit / Vedic is NOT sent to Gemini</b> — our own local tools judge that (verse matching + language check).</p>{bar}"
         f"<p><small>{done:,} of {tot:,} Nepali clips have Gemini words ({pct:.1f}%). {rate} {pill('working', 'ok')}</small></p>"
         f"<div class='grid'>{big(f'{done:,}', 'Nepali clips with Gemini words')}{big(f'{tot - n:,}', 'still waiting')}{big(f'{conf:,}', 'confirmed Nepali')}</div><p><small>{E(sty)}</small></p>")
    return wrap(h)
@safe
def card8_clips(seed=None):
    g = rows(); rnd = random.Random(seed if seed is not None else time.time()); ne = [r for r in _nep(g) if r["text"]]
    if not ne: return wrap("<p>Clips will appear here once Gemini has listened to some.</p>")
    out = []
    for r in rnd.sample(ne, min(6, len(ne))):
        w = _whisper(r["id"]); ex = f"<div class='dev'><small>Before (old model): {E(w[:200]) or '—'}</small></div>" if w else ""
        out.append(_row(r, pill("Nepali · Gemini words", "ok"), ex))
    return wrap("<h3>Listen: what Gemini wrote (Nepali)</h3><p><small>Press <b>New clips</b> for others. To fix any word yourself use card 6 — it starts from Gemini's words and saves instantly.</small></p>" + "".join(out))
