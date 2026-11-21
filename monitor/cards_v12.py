"""v12 card: cleaned V1 vocoder, chunk by chunk, each chunk judged by the real audio gate vs shipped V1. Read-only, never raises."""
import base64, glob, html, json, os, re, time
R = "/work/tts"; V = f"{R}/v12"
CSS = """<style>.s12{font-family:system-ui,sans-serif;font-size:14px;color:#1f2937}.s12 h3{margin:.2em 0 .5em;font-size:17px}
.s12 table{border-collapse:collapse;width:100%}.s12 td,.s12 th{border-bottom:1px solid #e5e7eb;padding:5px 8px;text-align:left}
.s12 th{background:#f9fafb}.s12 .ok{background:#dcfce7}.s12 .no{background:#fee2e2}.s12 .mut{color:#6b7280}.s12 .pill{display:inline-block;padding:1px 8px;border-radius:999px;background:#f3f4f6;margin-right:4px}
.s12 audio{height:32px;width:240px}</style>"""


def _lines(p):
    try: return open(p, encoding="utf8", errors="ignore").read().splitlines()
    except Exception: return []


def card_v12():
    L = _lines(f"{V}/train.log"); rows = [json.loads(l) for l in _lines(f"{V}/chunks.jsonl") if l.strip()]
    gate = next((l for l in L if "[GATE]" in l), "speed check pending")
    inl = [l for l in L if "in-loop audio check" in l][-3:]
    last = next((l for l in reversed(L) if "[v12] step=" in l), "")
    done = next((l for l in L if "[v12] DONE" in l), None)
    state = "finished" if done else ("training" if L and time.time() - os.path.getmtime(f"{V}/train.log") < 180 else "waiting")
    tr = ""
    for r in rows:
        b, c = r["base"], r["cand"]
        tr += (f"<tr class='{'ok' if r['ok'] else 'no'}'><td>{r['chunk']}</td><td>{r['step']:,}</td><td>{b['score']}</td><td><b>{c['score']}</b></td>"
               f"<td>{c['drops']} / {c['stutters']} / {c['clicks']}</td><td>{c['sv_cos']}</td><td>{'OK' if r['ok'] else 'not OK'}</td></tr>")
    dec = ""
    try:
        d = json.load(open(f"{R}/repo11/eval/v12_voc_decision.json"))
        ck = " ".join(f'<span class="pill {"ok" if v else "no"}">{k.replace("_", " ")}</span>' for k, v in d["checks"].items())
        stl = "".join(f"<tr><td>{k}</td><td>{d['v1']['styles'].get(k)}</td><td>{d['candidate']['styles'].get(k)}</td></tr>" for k in d["v1"]["styles"])
        dec = f"<h3 style='margin-top:10px'>Decision: {'SHIPPED' if d['ship'] else 'not shipped'} (chunk {d['chunk']})</h3><div>{ck}</div><table><tr><th>Style</th><th>V1</th><th>Clean vocoder</th></tr>{stl}</table>"
    except Exception:
        pass
    return CSS + f"""<div class="s12"><h3>V1 vocoder clean-up <span class="pill">{state}</span></h3>
<div class="mut">Acoustic model untouched. Only the vocoder learns from real voices, in 10-min chunks. After every chunk the model is exported and
scored on the full audio gate against shipped V1 (same clips). It stops by itself when a chunk is worse.</div>
<div style="margin-top:6px" class="mut">{html.escape(gate)}<br>{html.escape(last[-200:])}</div>
<div class="mut">{'<br>'.join(html.escape(l[-230:]) for l in inl)}</div>
<table style="margin-top:8px"><tr><th>Chunk</th><th>Step</th><th>V1 score</th><th>Chunk score</th><th>Drops / stutters / clicks</th><th>Voice match</th><th>Result</th></tr>{tr or '<tr><td colspan=7 class=mut>first chunk result about 12 min after start</td></tr>'}</table>{dec}</div>"""


def card_v12_listen():
    w = sorted(glob.glob(f"{V}/samples/*.wav"))
    if not w: return CSS + '<div class="s12"><h3>Listen: V1 vs clean vocoder</h3><span class="mut">Rendered after the last chunk (same sentences, int8 graphs the site serves).</span></div>'
    by = {}
    for p in w:
        sid, name = os.path.basename(p)[:-4].split("__"); by.setdefault(sid, {})[name] = p
    rows = "".join(f"<tr><td>{html.escape(sid[2:])}</td>" + "".join(f"<td>{n}<br><audio controls preload='none' src='data:audio/wav;base64,{base64.b64encode(open(d[n], 'rb').read()).decode()}'></audio></td>" for n in ("V1", "V1-cleanvoc") if n in d) + "</tr>" for sid, d in sorted(by.items()))
    return CSS + f'<div class="s12"><h3>Listen: V1 vs clean vocoder (same sentence)</h3><table>{rows}</table></div>'


CELLS_V12 = {
"tts_v12_voc": '''
import sys as _sys, importlib.util as _iu, marimo as _mo
_s = _iu.spec_from_file_location("cards_v12", "/work/tts/repo11/monitor/cards_v12.py"); v12ui = _iu.module_from_spec(_s); _sys.modules["cards_v12"] = v12ui; _s.loader.exec_module(v12ui)
v12_refresh = _mo.ui.refresh(default_interval="20s", label="vocoder clean-up live (20 s)")
v12_refresh
''',
"tts_v12_voc_body": '''
import marimo as _mo
v12_refresh.value
_mo.Html(v12ui.card_v12())
''',
"tts_v12_listen": '''
import marimo as _mo
v12_refresh.value
_mo.Html(v12ui.card_v12_listen())
''',
}


async def install_v12():
    import marimo._code_mode as cm
    async with cm.get_context(skip_staleness_check=True) as ctx:
        cells = list(ctx.cells.values()); by = {c.name: c.id for c in cells}
        first = next((c.id for c in cells if not c.name.startswith("tts_v12_")), None)
        ids, prev = {}, None
        for name, src in CELLS_V12.items():
            if name in by: ids[name] = by[name]; ctx.edit_cell(ids[name], src.strip(), hide_code=True)
            elif prev is None and first is not None: ids[name] = ctx.create_cell(src.strip(), name=name, hide_code=True, before=first)
            else: ids[name] = ctx.create_cell(src.strip(), name=name, hide_code=True, after=prev)
            prev = ids[name]
        for name in CELLS_V12: ctx.run_cell(ids[name])
        print("v12 cards:", list(CELLS_V12))
