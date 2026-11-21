"""Sakhiya v11 owner cards: adds the AUDIT card ("Check & fix the words").  Builds on ui10 (verified Sanskrit, clip browser, funnel, listen, backups).
The owner listens to a clip, edits the words, picks a verdict; every change is auto-saved to /work/tts/corrections.json (gold transcripts).
Fixes typed on the GitHub-Pages Clip Library can be pasted / uploaded here.  Never raises."""
import glob, json, os, sys, time, html
import importlib.util as _iu
_here = os.path.dirname(os.path.abspath(__file__))
_s = _iu.spec_from_file_location("ui10_cards", _here + "/ui10_cards.py"); U = _iu.module_from_spec(_s); sys.modules["ui10_cards"] = U; _s.loader.exec_module(U)
C, E, R, pill, big, wrap, safe = U.C, U.E, U.R, U.pill, U.big, U.wrap, U.safe
header, card1, card2, card3, card4, card5 = U.header, U.card1, U.card2, U.card3, U.card4, U.card5
LISTS, STYLE_OPTS = U.LISTS, U.STYLE_OPTS
FIXFILE = f"{R}/corrections.json"
AUDIT_LISTS = ["Vedic: verified Sanskrit", "Kept clips (passed every check)", "Words written down (2 listeners)"]
VERD = ["Not checked yet", "✓ Words are right", "✎ I fixed the words", "✗ Garbage / wrong language"]
_V2K = {VERD[1]: "ok", VERD[2]: "fix", VERD[3]: "bad"}; _K2V = {"ok": VERD[1], "fix": VERD[2], "bad": VERD[3]}
PER = 5

def load_fixes():
    try: return json.load(open(FIXFILE))
    except Exception: return {}

def _write(d):
    tmp = FIXFILE + ".tmp"; json.dump(d, open(tmp, "w"), ensure_ascii=False, indent=1); os.replace(tmp, FIXFILE)

def _pool(kind, style):
    out = []
    try:
        if kind.startswith("Vedic"):
            if style not in ("", "vedic_chant"): return []
            for o in U.gen_pool(): out.append({"id": os.path.basename(o["path"])[:-4], "path": o["path"], "text": o["text"], "style": "vedic_chant", "src": o["src"]})
        elif kind.startswith("Kept"):
            for r in C.jl(f"{R}/dataset/meta.jsonl"):
                if style and r.get("style") != style: continue
                p = next(iter(glob.glob(f"{R}/segments/*/{r['id']}.wav")), "")
                if p: out.append({"id": r["id"], "path": p, "text": r.get("text", ""), "style": r.get("style", ""), "src": r["id"].rsplit("_", 1)[0][:40]})
        else:
            for r in C.jl(f"{R}/asr/asr.jsonl"):
                if style and r.get("cat") != style: continue
                p = r.get("path", "")
                if p: out.append({"id": os.path.basename(p)[:-4], "path": p, "text": r.get("text_w", ""), "style": r.get("cat", ""), "src": os.path.basename(p).rsplit("_", 1)[0][:40]})
    except Exception: pass
    return [o for o in out if os.path.exists(o["path"])]

def audit_items(kind=AUDIT_LISTS[0], style_label="All styles", page=1, only_todo=True):
    """-> (items for this page, total, pages, page).  Un-reviewed clips come first so every page is useful work."""
    fx = load_fixes(); pool = _pool(kind, STYLE_OPTS.get(style_label, ""))
    pool.sort(key=lambda o: (o["id"] in fx, o["id"])); n = len(pool); pages = max(1, -(-n // PER)); page = int(min(max(1, page or 1), pages))
    items = pool[(page - 1) * PER: page * PER]
    cand = {r["id"]: r for r in C.jl(f"{R}/asr/candidates.jsonl")} if kind[0] != "V" else {}
    for o in items:
        o["cand"] = cand.get(o["id"]); f = fx.get(o["id"], {}); o["saved_text"] = f.get("text", o["text"]); o["saved_v"] = _K2V.get(f.get("verdict"), VERD[0])
    return items, n, pages, page

def clip_html(o):
    tag = pill({"vag": "Studio chanting", "gss": "Gita verse", "rec": "Gita class"}.get(o.get("src"), (o.get("src") or "")[:30]), "tag")
    cd = o.get("cand"); sug = ""
    if cd and cd.get("cands"):
        names = {"A": "Whisper-large-v3", "B": "MMS Nepali", "C": "IndicConformer"}
        sug = f"<small>Other listeners (agreement {cd.get('consensus', 0) * 100:.0f}%):</small>" + "".join(f"<div class='dev' style='font-size:14px'><small>{names[k]}:</small> {E(v[:140])}</div>" for k, v in cd["cands"].items())
    return f"<div class='v8'><div class='clip'>{tag}{pill(C.STYLES.get(o['style'], (o['style'],))[0], 'tag')}<small> id {E(o['id'][:44])}</small>{C.audio(o['path'], maxs=14)}{sug}</div></div>"

def vtext(items, verd, txt):
    """Auto-save: called with the widgets' values; writes only real changes. Returns the status line (HTML)."""
    fx = load_fixes(); changed = 0
    for o, v, t in zip(items, verd, txt):
        k = _V2K.get(v, ""); t = (t or "").strip(); old = fx.get(o["id"], {})
        if not k and t == o["saved_text"].strip() and not old: continue
        if not k and t != o["text"].strip(): k = "fix"
        if k == "ok" and t != o["text"].strip(): k = "fix"
        if not k: continue
        new = {"text": t, "verdict": k, "orig": o["text"], "style": o["style"], "src": o["src"], "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        if (old.get("text"), old.get("verdict")) != (new["text"], new["verdict"]): fx[o["id"]] = new; changed += 1
    if changed: _write(fx)
    return status(changed)

def status(changed=0):
    fx = load_fixes(); n = len(fx); c = lambda k: sum(1 for v in fx.values() if v.get("verdict") == k)
    msg = f"Saved ✓ ({changed} just now)" if changed else "Everything you change is saved automatically."
    return wrap(f"<p>{pill(msg, 'ok')} You have checked <b>{n}</b> clips: {c('ok')} right · {c('fix')} fixed · {c('bad')} garbage.</p>")

def import_fixes(text):
    """Merge a fixes file typed on the website (JSON text) into corrections.json."""
    try:
        d = json.loads(text); assert isinstance(d, dict)
    except Exception: return wrap("<p>" + pill("That text is not a fixes file", "no") + " Use the <b>Copy</b> or <b>Download</b> button on the website's Clip Library page.</p>")
    fx = load_fixes(); n = 0
    for k, v in d.items():
        if isinstance(v, dict) and v.get("verdict") in ("ok", "fix", "bad") or (isinstance(v, dict) and v.get("text")):
            v = dict(v); v["ts"] = time.strftime("%Y-%m-%d %H:%M:%S"); v["via"] = "website"; v.setdefault("verdict", "fix"); fx[k] = v; n += 1
    _write(fx); return wrap("<p>" + pill(f"Imported {n} fixes", "ok") + f" · total {len(fx)}. They are now your gold transcripts.</p>")

@safe
def card6_head():
    return wrap("<h2>6 · Check &amp; fix the words</h2><p>Listen to a clip, then fix the words under it or mark it right / garbage. <b>Nothing to save by hand</b> - it saves itself. "
                "Un-checked clips come first. Your fixes become the <b>gold transcripts</b>: they replace the machine's guess for training and show on the website. "
                "<small>Garbage clips are removed from training.</small></p>")
