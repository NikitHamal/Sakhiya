"""Sakhiya v10 owner cards. Builds on ui9: the Vedic card now shows ONLY verified Sanskrit (exact reference text, never an ASR guess) and the clip browser
gets a 'Vedic: verified genuine' list.  Sources: studio Vagdhenu chanting, Gita-supersite verses (force-aligned, exact text), and Gita-class clips recovered by
matching their sound to a real verse.  Never raises."""
import glob, json, os, random, sys
import importlib.util as _iu
_here = os.path.dirname(os.path.abspath(__file__))
_s = _iu.spec_from_file_location("ui9_cards", _here + "/ui9_cards.py"); U = _iu.module_from_spec(_s); sys.modules["ui9_cards"] = U; _s.loader.exec_module(U)
C, E, R, pill, big, wrap, safe = U.C, U.E, U.R, U.pill, U.big, U.wrap, U.safe
header, card1, card4, card5, STYLE_OPTS = U.header, U.card1, U.card4, U.card5, U.STYLE_OPTS
LISTS = dict(U.LISTS); LISTS = {"Vedic: verified genuine Sanskrit (exact words)": "v_gen", **{k: v for k, v in LISTS.items() if v != "v_ok"}}
LABEL = {"vag": "Studio chanting (Vagdhenu)", "gss": "Bhagavad Gita verses (exact text)", "rec": "Recovered from Gita class recordings"}
_c = {"t": 0, "v": None}

def gen_pool():
    """All verified genuine Sanskrit clips: [{path,text,src,score}] (cached 60 s)."""
    import time
    if time.time() - _c["t"] < 60 and _c["v"] is not None: return _c["v"]
    try: ref = json.load(open(f"{R}/vedic/ref_text.json"))
    except Exception: ref = {}
    out = []
    for p, t in ref.items():
        b = os.path.basename(p)
        if b.startswith("vag_") or b.startswith("gss_"): out.append({"path": p, "text": t, "src": "vag" if b.startswith("vag_") else "gss"})
    for r in C.jl(f"{R}/vedic/verified.jsonl"):
        if r.get("ok"): out.append({"path": r["path"], "text": r.get("text_ref", ""), "src": "rec", "score": r.get("score")})
    out = [o for o in out if os.path.exists(o["path"])]; _c.update(t=time.time(), v=out); return out

def _clip(o, k="ok"):
    sc = pill(f"sound matches verse {o['score']:.0f}%", "tag") if o.get("score") else ""
    return f"<div class='clip'>{pill(LABEL[o['src']], k)}{sc}<div class='dev'>{E(o['text'][:160])}</div>{C.audio(o['path'], maxs=14)}</div>"

@safe
def card2(shuffle=None):
    pool = gen_pool(); rep = {}
    for n in ("verified_report", "gss_report"):
        try: rep[n] = json.load(open(f"{R}/vedic/{n}.json"))
        except Exception: rep[n] = None
    by = {k: [o for o in pool if o["src"] == k] for k in LABEL}
    stats = "".join(big(f"{len(v):,}", LABEL[k]) for k, v in by.items())
    if not pool: return wrap("<h2>2 · Is the chanting real Sanskrit?</h2><p>The verified Sanskrit library is being built right now. Check again in a few minutes.</p>")
    bad = [r for r in U._vrows() if r["verdict"] == "rejected" and os.path.exists(r["path"])]
    good = "".join(_clip(o) for k in LABEL for o in random.sample(by[k], min(2, len(by[k]))))
    rej = "".join(f"<div class='clip'>{pill('REMOVED', 'no')}<small>{E(U._why(r.get('why', [])))}</small>{C.audio(r['path'])}</div>" for r in random.sample(bad, min(2, len(bad))))
    return wrap("<h2>2 · Is the chanting real Sanskrit?</h2><p>Yes — now every Vedic clip must be <b>proven</b>: its words come from a real Sanskrit text (never a machine guess), and clips from long "
                "class recordings are kept only when their sound matches a real Gita verse. Talks in Tamil / Malayalam / English cannot match, so they stay out.</p>"
                f"<div class='row'>{stats}</div><h3>Listen: verified chanting (words are the real verse)</h3>{good}<h3>Listen: examples that were thrown away</h3>{rej}")

@safe
def card3(kind_label="Kept clips (passed every check)", style_label="All styles", page=1):
    if LISTS.get(kind_label) != "v_gen": return U.card3(kind_label, style_label, page)
    pool = gen_pool(); per = 8; n = len(pool); pages = max(1, -(-n // per)); page = int(min(max(1, page or 1), pages))
    if not n: return wrap("<h2>3 · Clip browser</h2><p>No verified Sanskrit clips yet.</p>")
    return wrap(f"<h2>3 · Clip browser · verified Sanskrit</h2><p>{n:,} clips · page {page} of {pages}. Use the Page box above to hear more.</p>" + "".join(_clip(o) for o in pool[(page - 1) * per: page * per]))
