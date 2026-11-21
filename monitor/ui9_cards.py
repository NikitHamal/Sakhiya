"""Sakhiya v9 owner cards (plain English, you can LISTEN to everything). Builds on ui8_cards; adds
 card2 = "Is the chanting real Sanskrit?" (per-source verdicts + audio of accepted / rejected clips),
 card3 = "Clip browser" (pick a list + style + page, hear any clip with its words).  Never raises."""
import glob, json, os, random, sys
import importlib.util as _iu
_here = os.path.dirname(os.path.abspath(__file__))
_s = _iu.spec_from_file_location("ui8_cards", _here + "/ui8_cards.py"); B = _iu.module_from_spec(_s); sys.modules["ui8_cards"] = B; _s.loader.exec_module(B)
C, E, R, pill, big, wrap, safe = B.C, B.E, B.R, B.pill, B.big, B.wrap, B.safe
header, card1, card4, card5 = B.header, B.card1, B.card4, B.card5
LISTS = {"Kept clips (passed every check)": "kept", "Words written down (2 listeners)": "asr", "Vedic: real Sanskrit (accepted)": "v_ok",
         "Vedic: thrown away (talk / other language)": "v_bad", "All cleaned sentences (before word check)": "gate"}
STYLE_OPTS = {"All styles": "", **{v[0]: k for k, v in C.STYLES.items()}}
WHY = {"Dravidian": "sounds like Tamil / Telugu / Malayalam / Kannada", "English": "English speech", "not Sanskrit": "does not sound like Sanskrit",
       "Devanagari": "no clean Sanskrit words could be written", "voiced": "mostly talking, no sustained singing tone"}

def _why(ws):
    out = [v for w in ws for k, v in WHY.items() if k in w]
    return "; ".join(out) or "failed a check"

def _vrows():
    return C.jl(f"{R}/vedic/verdicts.jsonl")

@safe
def card2(shuffle=None):
    rows = _vrows(); rep = {}
    try: rep = json.load(open(f"{R}/vedic/source_report.json"))
    except Exception: pass
    if not rows:
        return wrap("<h2>2 · Is the chanting real Sanskrit?</h2><p>The purity check has not run yet. It listens to every Vedic clip and keeps only genuine Sanskrit chanting.</p>")
    ok = [r for r in rows if r["verdict"] in ("pure_sanskrit", "trusted_reference")]; bad = [r for r in rows if r["verdict"] == "rejected"]
    srcs = "".join(f"<div class='step'>{pill('KEPT', 'ok') if d['pure_pct'] >= 60 or d.get('trusted') else pill('REMOVED', 'no')}<b>{E(s[:46])}</b><small> · {d['pure_pct']:.0f}% genuine · mostly heard as "
                   f"{E(', '.join(l for l, _ in d['top_langs'][:2]))}</small></div>" for s, d in sorted(rep.items(), key=lambda x: -x[1]["pure_pct"]))
    def ex(pool, tag, kind, k=3):
        o = ""
        for r in random.sample(pool, min(k, len(pool))):
            note = (f"<small>Why removed: {E(_why(r.get('why', [])))}</small>" if kind == "bad" else
                    f"<small>Heard as {E(r['top_lang'])} · {'melodic chant' if r.get('chant_like') else 'recitation'} · {r['dur']:.0f} s</small>")
            o += f"<div class='clip'>{pill(tag, 'ok' if kind == 'ok' else 'no')}<div class='dev'>{E(r.get('text_sa', '')[:90])}</div>{note}{C.audio(r['path'])}</div>"
        return o or "<p><small>none yet</small></p>"
    return wrap(f"<h2>2 · Is the chanting real Sanskrit?</h2><p>Earlier, many “Vedic” clips were really long talks (Tamil, Malayalam, English). Now every clip is checked: "
                f"language it sounds like, written Sanskrit words, and whether it is sustained chanting. <b>{len(ok)}</b> accepted · <b>{len(bad)}</b> thrown away so far.</p>"
                f"<h3>Sources</h3>{srcs}<h3>Listen: accepted as real Sanskrit</h3>{ex(ok, 'REAL SANSKRIT', 'ok')}<h3>Listen: thrown away</h3>{ex(bad, 'REMOVED', 'bad')}")

def _pool(kind, style):
    if kind == "kept": rows = C.jl(f"{R}/dataset/meta.jsonl"); f = lambda r: (r.get("style"), "", r.get("text", ""))
    elif kind == "asr": rows = C.jl(f"{R}/asr/asr.jsonl"); f = lambda r: (r.get("cat"), r.get("path"), r.get("text_w", ""))
    elif kind == "gate": rows = C.jl(f"{R}/gate/passed.jsonl"); f = lambda r: (r.get("cat"), r.get("path"), "")
    else:
        rows = [r for r in _vrows() if (r["verdict"] != "rejected") == (kind == "v_ok")]; f = lambda r: ("vedic_chant", r["path"], r.get("text_sa", ""))
    out = []
    for r in rows:
        st, p, t = f(r)
        if style and st != style: continue
        if kind == "kept": p = next(iter(glob.glob(f"{R}/segments/*/{r['id']}.wav")), "")   # packed rows keep audio in shards; use the cleaned wav
        if p and os.path.exists(p): out.append((st, p, t, r))
    return out

@safe
def card3(kind_label="Kept clips (passed every check)", style_label="All styles", page=1):
    kind = LISTS.get(kind_label, "kept"); style = STYLE_OPTS.get(style_label, ""); pool = _pool(kind, style); n = len(pool); per = 6
    pages = max(1, -(-n // per)); page = int(min(max(1, page or 1), pages)); view = pool[(page - 1) * per: page * per]
    if not view: return wrap(f"<h2>3 · Clip browser</h2><p>No clips in “{E(kind_label)}” for “{E(style_label)}” yet. Pick another list above.</p>")
    cards = ""
    for st, p, t, r in view:
        tags = pill(C.STYLES.get(st, (st, ""))[0], "tag") + (pill(f"quality {r['utmos']:.2f}", "ok") if r.get("utmos") else "") + (pill(f"agreement {r['agree'] * 100:.0f}%", "tag") if r.get("agree") is not None else "")
        cards += f"<div class='clip'>{tags}<div class='dev'>{C.dev_text(t) or '<small>(words not written yet)</small>'}</div>{C.audio(p)}</div>"
    return wrap(f"<h2>3 · Clip browser</h2><p>Choose a list, a style and a page. {n:,} clips in this view · page {page} of {pages}. "
                f"The same clips are on the website: <b>Clip Library</b> page.</p>{cards}")
