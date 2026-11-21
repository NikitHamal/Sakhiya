"""Sakhiya v12 owner cards: adds the VEDIC GOLD card (card 7) and puts the gold/doubtful lists into the audit card (card 6).
Gold = a second, independent speech model heard the clip and its sound matches the written verse (scripts/vedic_gold_audit.py -> /work/tts/vedic/gold.json).
Builds on ui11 (audit card, import) and ui10.  Never raises."""
import json, os, random, sys, time
import importlib.util as _iu
_here = os.path.dirname(os.path.abspath(__file__))
_s = _iu.spec_from_file_location("ui11_cards", _here + "/ui11_cards.py"); U = _iu.module_from_spec(_s); sys.modules["ui11_cards"] = U; _s.loader.exec_module(U)
C, E, R, pill, big, wrap, safe = U.C, U.E, U.R, U.pill, U.big, U.wrap, U.safe
GOLD_LISTS = ["Vedic GOLD: sound matches the verse", "Vedic GOLD: melodic swara only", "Vedic doubtful: please check"]
U.AUDIT_LISTS = GOLD_LISTS + [x for x in U.AUDIT_LISTS if not x.startswith("Vedic")]
_old_pool, _old_html = U._pool, U.clip_html
_gc = {"t": 0, "v": {}}
WHY = {"does not match": "what is chanted does not match the written verse", "letters/s": "speed is not natural chanting", "voiced": "too little voiced sound", "length": "too short or too long"}

def gold():
    if time.time() - _gc["t"] > 30:
        try: _gc.update(t=time.time(), v=json.load(open(f"{R}/vedic/gold.json")))
        except Exception: _gc.update(t=time.time())
    return _gc["v"]

def _gpool(kind):
    out = []
    try: ref = json.load(open(f"{R}/vedic/ref_text.json"))
    except Exception: ref = {}
    for p, g in gold().items():
        if not os.path.exists(p): continue
        if kind.startswith("Vedic doubtful") != (g["tier"] == "doubtful"): continue
        if "melodic" in kind and not g.get("melodic"): continue
        out.append({"id": os.path.basename(p)[:-4], "path": p, "text": ref.get(p, ""), "style": "vedic_chant", "src": {"vag_": "vag", "gss_": "gss"}.get(os.path.basename(p)[:4], "rec"), "g": g})
    out.sort(key=lambda o: (-o["g"]["match"] if not kind.startswith("Vedic doubtful") else o["g"]["match"]))
    return out

def _pool(kind, style):
    if kind.startswith("Vedic"): return _gpool(kind) if style in ("", "vedic_chant") else []
    return _old_pool(kind, style)

def clip_html(o):
    g = o.get("g")
    if not g: return _old_html(o)
    t = pill("GOLD" if g["tier"] == "gold" else "OK" if g["tier"] == "ok" else "Doubtful", "ok" if g["tier"] != "doubtful" else "no") + pill("melodic swara" if g.get("melodic") else "recited", "tag")
    t += pill({"vag": "Studio chanting", "gss": "Gita verse", "rec": "Gita class"}.get(o["src"], o["src"]), "tag") + pill(f"sound matches verse {g['match']:.0f}%", "tag")
    why = "; ".join(v for k, v in WHY.items() if any(k in w for w in g.get("why", [])))
    return (f"<div class='v8'><div class='clip'>{t}{C.audio(o['path'], maxs=18)}" + (f"<div><small>Why doubtful: {E(why)}</small></div>" if why else "")
            + f"<div class='dev' style='font-size:14px'><small>Second listener heard:</small> {E(g.get('hyp', '')[:120])}</div></div></div>")

U._pool, U.clip_html = _pool, clip_html
globals().update({k: getattr(U, k) for k in dir(U) if not k.startswith("__") and k not in globals()})
AUDIT_LISTS = U.AUDIT_LISTS

@safe
def card7(seed=None):
    g = gold(); n = len(g)
    if not n: return wrap("<h2>7 · Vedic GOLD audit</h2><p>The second listener is still working... " + pill("running", "ok") + "</p>")
    cnt = lambda t: sum(1 for v in g.values() if v["tier"] == t); mel = sum(1 for v in g.values() if v["tier"] == "gold" and v.get("melodic"))
    ng, nok, nd = cnt("gold"), cnt("ok"), cnt("doubtful"); rnd = random.Random(seed if seed is not None else time.time())
    gp = [o for o in _gpool(GOLD_LISTS[1])] or _gpool(GOLD_LISTS[0]); dp = _gpool(GOLD_LISTS[2])
    pick = rnd.sample(gp, min(3, len(gp))) + rnd.sample(dp, min(2, len(dp)))
    h = (f"<h2>7 · Vedic GOLD audit</h2><p>A <b>second, independent listener</b> heard every Vedic clip and compared it with the written verse. "
         f"Only clips whose sound really matches the verse are GOLD.</p><div class='grid'>{big(f'{ng:,}', 'GOLD clips (use for training)')}{big(f'{mel:,}', 'of them melodic swara')}"
         f"{big(f'{nok:,}', 'probably right')}{big(f'{nd:,}', 'doubtful (kept out)')}{big(f'{n:,}', 'clips audited')}</div>"
         f"<p><small>Below: 3 GOLD examples and 2 doubtful ones. Press <b>New clips</b> for others. To judge any clip yourself use card 6 -> Check -> Vedic GOLD / doubtful.</small></p>")
    return wrap(h + "".join(clip_html(o) for o in pick))
