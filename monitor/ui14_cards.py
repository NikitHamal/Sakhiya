"""Sakhiya v14 owner cards: card 8 = ALL Nepali listening engines at once (Gemini + AssemblyAI + Groq Whisper v3 / turbo), one engine per clip.
Data: asr/gemini.jsonl, asr/cloud_asr.jsonl, asr/cloud_state.json.  Builds on ui13 (keeps cards 1-7).  Never raises."""
import json, os, random, sys, time
import importlib.util as _iu
_here = os.path.dirname(os.path.abspath(__file__))
_s = _iu.spec_from_file_location("ui13_cards", _here + "/ui13_cards.py"); U13 = _iu.module_from_spec(_s); sys.modules["ui13_cards"] = U13; _s.loader.exec_module(U13)
C, E, R, pill, big, wrap, safe = U13.C, U13.E, U13.R, U13.pill, U13.big, U13.wrap, U13.safe
FILES = {f"{R}/asr/gemini.jsonl": "gemini", f"{R}/asr/cloud_asr.jsonl": None}
NAMES = {"gemini": "Google Gemini", "assemblyai": "AssemblyAI", "groq-whisper-v3": "Groq Whisper large-v3", "groq-whisper-v3-turbo": "Groq Whisper turbo"}
COL = {"gemini": "#3b82f6", "assemblyai": "#a855f7", "groq-whisper-v3": "#f59e0b", "groq-whisper-v3-turbo": "#f97316"}
_c = {"pos": {}, "rows": {}}
def rows():
    """incremental read of every engine's jsonl -> {clip id: row with .provider}; one engine per clip so ids never clash"""
    for f, dflt in FILES.items():
        try:
            with open(f, "rb") as fh:
                fh.seek(_c["pos"].get(f, 0)); data = fh.read(); cut = data.rfind(b"\n")
                if cut < 0: continue
                _c["pos"][f] = _c["pos"].get(f, 0) + cut + 1
                for l in data[:cut].splitlines():
                    try: r = json.loads(l); r["provider"] = r.get("provider") or dflt; _c["rows"][r["id"]] = r
                    except Exception: pass
        except Exception: pass
    return _c["rows"]
U13.rows = rows      # ui13's audit pool / helpers now see AssemblyAI + Groq words too
def cstate():
    try: return json.load(open(f"{R}/asr/cloud_state.json"))
    except Exception: return {}
_u11 = getattr(U13, "_u11", None); _orig_pool = U13._old_pool; _orig_html = U13._old_html
def _pool(kind, style):
    out = _orig_pool(kind, style); g = rows()
    for o in out:
        r = g.get(o["id"])
        if r and r.get("text") and r.get("lang") not in U13.BADL and r.get("style") != "vedic_chant":
            o["old"] = o["text"]; o["text"] = r["text"]; o["gem"] = True; o["eng"] = NAMES.get(r["provider"], r["provider"])
    return out
def clip_html(o):
    h = _orig_html(o)
    if o.get("eng"): h = h.replace("words below start from Gemini's listening", "words below start from " + E(o["eng"]) + "'s listening")
    return h
if _u11: _u11._pool, _u11.clip_html = _pool, clip_html
globals().update({k: getattr(U13, k) for k in dir(U13) if not k.startswith("__") and k not in globals()})
def _nep(g): return [r for r in g.values() if r.get("style") != "vedic_chant" and "/vedic_chant/" not in r.get("path", "") and r.get("text")]
def _status(k, st, gs):
    cool = st.get("cool", {})
    if k == "assemblyai":
        note = "spent about $%.2f of your $100 free credit (stops at $%.0f to be safe)" % (st.get("aai_usd", 0), st.get("aai_budget", 90))
        return note, pill("working", "ok") if time.time() - st.get("ts", 0) < 120 else pill("paused", "bad")
    if k.startswith("groq"):
        m = "whisper-large-v3-turbo" if k.endswith("turbo") else "whisper-large-v3"; cs = cool.get(m, 0)
        note = "free daily quota used up, tries again later" if cs > 1800 else "resting %d min (free per-minute limit), then continues" % (cs // 60 + 1) if cs > 0 else "listening now"
        return note, pill("resting", "bad") if cs > 0 else pill("working", "ok")
    return "streams audio at real-time speed", pill("working", "ok") if time.time() - gs.get("ts", 0) < 300 else pill("paused", "bad")
@safe
def card8_stats():
    g = rows(); L = _nep(g); st = cstate(); gs = {}
    try: gs = json.load(open(f"{R}/asr/gemini_state.json"))
    except Exception: pass
    tot = max(st.get("total", 0), gs.get("total", 0), len(L)); done = len(L); pct = 100 * done / max(tot, 1); cnt = {}
    for r in L: cnt[r["provider"]] = cnt.get(r["provider"], 0) + 1
    bar = f"<div style='background:#333;border-radius:8px;height:16px'><div style='background:#22c55e;height:16px;border-radius:8px;width:{min(pct, 100):.1f}%'></div></div>"
    ts = sorted(r.get("t", 0) for r in L)[-400:]; cpm = len(ts) / max(1, ts[-1] - ts[0]) * 60 if len(ts) > 30 else 0
    eta = f"About <b>{max(0, tot - done) / max(cpm, .1) / 60:.1f} hours</b> to go at today's speed ({cpm:.0f} clips a minute)." if cpm else ""
    rows_h = ""
    for k, nm in NAMES.items():
        c = cnt.get(k, 0); w = 100 * c / max(done, 1); note, stt = _status(k, st, gs)
        rows_h += (f"<div style='margin:6px 0'><b style='color:{COL[k]}'>{E(nm)}</b> {stt} <small>{c:,} clips · {E(note)}</small>"
                   f"<div style='background:#333;border-radius:6px;height:8px'><div style='background:{COL[k]};height:8px;border-radius:6px;width:{w:.1f}%'></div></div></div>")
    spent = "$%.2f" % st.get("aai_usd", 0); left = max(0, tot - done)
    h = ("<h2>8 · Nepali words — all listeners at once</h2><p>Four listeners work <b>at the same time</b> on <b>different clips</b> (never two on the same clip): Gemini, AssemblyAI (language set to auto) and Groq Whisper large-v3 then turbo. "
         "<b>Sanskrit / Vedic is never sent to any of them</b> — only our own local tools judge it.</p>" + bar +
         f"<p><small>{done:,} of {tot:,} Nepali clips have words ({pct:.1f}%). {eta}</small></p>"
         f"<div class='grid'>{big(f'{done:,}', 'clips with words')}{big(f'{left:,}', 'still waiting')}{big(spent, 'AssemblyAI spent')}</div>{rows_h}"
         "<p><small>Qwen 3.8 (27B) on Groq reads text and pictures only, not sound, so it is not used for listening.</small></p>")
    return wrap(h)
@safe
def card8_clips(seed=None):
    g = rows(); rnd = random.Random(seed if seed is not None else time.time()); L = _nep(g); by = {}
    for r in L: by.setdefault(r["provider"], []).append(r)
    if not L: return wrap("<p>Clips will appear here once the listeners have written some words.</p>")
    out = []
    for k in NAMES:
        for r in rnd.sample(by.get(k, []), min(2, len(by.get(k, [])))):
            ex = f"<div class='dev'><small>detected language: {E(str(r.get('lang', '')))}</small></div>"
            out.append(U13._row(r, f"<span style='color:{COL[k]};font-weight:700'>{E(NAMES[k])}</span> ", ex))
    return wrap("<h3>Listen: what each engine wrote (Nepali)</h3><p><small>Press <b>New clips</b> for others. To fix any word yourself use card 6 — it starts from these words and saves instantly.</small></p>" + "".join(out))
