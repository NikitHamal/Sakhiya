"""HTML renderers for the v2.6 cards. Pure functions -> str, each wrapped so a failure shows a soft card, never a traceback.
Flat M3X dark palette: tonal surfaces, no shadows, no gradients."""
import html, time
import live26 as L

CSS = """<style>
.sk{font-family:Poppins,system-ui,sans-serif;color:#e3e8f7;display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px;margin:6px 0}
.cd{background:#0e1730;border:1.5px solid #2a3552;border-radius:28px;padding:20px 22px;min-width:0}.cd.w{grid-column:1/-1}
.cd h3{margin:0 0 10px;font-size:16px;font-weight:600;display:flex;align-items:center;gap:10px;flex-wrap:wrap}.sp{flex:1}
.cd small{color:#9fabc9;font-size:12px;display:block}.tls{display:grid;grid-template-columns:repeat(auto-fit,minmax(86px,1fr));gap:8px;margin:10px 0}
.tl{background:#16213f;border-radius:18px;padding:9px 13px}.tl b{font-size:18px;font-weight:600;display:block}
.bar{height:9px;border-radius:9px;background:#212c47;overflow:hidden}.bar i{display:block;height:100%;border-radius:9px}
.chip{border-radius:12px;padding:3px 11px;font-size:12px;font-weight:600;background:#2a3858;color:#dfe7fa;display:inline-block;margin:1px}
.chip.run{background:#0b3fa0;color:#dbe4ff}.chip.ok{background:#0a5a3a;color:#c9f5dc}.chip.bad{background:#6b1d1d;color:#ffdad6}.chip.wait{background:#5a4a12;color:#ffe9a8}
.big{font-size:22px;font-weight:600;line-height:1.3}.steps{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}
.st{border-radius:16px;padding:7px 14px;font-size:13px;background:#16213f;color:#9fabc9}.st.on{background:#0b3fa0;color:#fff}.st.done{background:#0a5a3a;color:#c9f5dc}
pre.lg{background:#0a1226;border-radius:16px;padding:10px 12px;font-size:11px;line-height:1.5;color:#b9c4e2;overflow:hidden;white-space:pre-wrap;margin:8px 0 0;max-height:130px}
table.bt{width:100%;border-collapse:collapse;font-size:13px}table.bt th{color:#9fabc9;font-size:11px;text-align:left;padding:6px}table.bt td{padding:8px 6px;border-top:1px solid #2a3552;vertical-align:top}
</style>"""
E = lambda s: html.escape(str(s))
chip = lambda t, k="": f'<span class="chip {k}">{E(t)}</span>'
tile = lambda v, l: f'<div class="tl"><b>{E(v)}</b><small>{E(l)}</small></div>'
bar = lambda p, c="#b2c5ff": f'<div class="bar"><i style="width:{max(0, min(100, p)):.1f}%;background:{c}"></i></div>'
import os


def fmt(v, d=4, na="-"):
    try: return f"{float(v):.{d}f}" if v is not None else na
    except Exception: return na


def dur(s):
    try: return f"{s:.0f} s" if s < 90 else (f"{s/60:.0f} min" if s < 5400 else f"{s//3600:.0f} h {s%3600/60:.0f} min")
    except Exception: return "-"


def spark(ys, w=360, h=64, col="#b2c5ff"):
    try:
        if not ys or len(ys) < 2: return '<small>waiting for loss points...</small>'
        lo, hi = min(ys), max(ys)
        pts = " ".join(f"{i/(len(ys)-1)*w:.1f},{h-(v-lo)/(hi-lo+1e-9)*(h-8)-4:.1f}" for i, v in enumerate(ys))
        return f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" preserveAspectRatio="none"><polyline fill="none" stroke="{col}" stroke-width="2.5" stroke-linejoin="round" points="{pts}"/></svg>'
    except Exception:
        return ""


def safe(fn):
    def w(*a, **k):
        try: return CSS + fn(*a, **k)
        except Exception as e: return CSS + f'<div class="sk"><div class="cd w"><h3>{chip("Waiting for Stage...", "wait")}</h3><small>{E(type(e).__name__)}</small></div></div>'
    return w


def pill():
    hb = (L.rj("/work/tts/heartbeat.json", {}) or {}).get("t")
    age = time.time() - hb if hb else None
    ok = age is not None and age < 150
    return CSS + f'<div class="sk" style="display:block">{chip("keep-alive " + ("live" if ok else "starting"), "ok" if ok else "wait")}{chip("heartbeat " + (dur(age) + " ago" if age is not None else "none"))}</div>'


STAGES = ["P1 v6_chandas (Vedic Laya+Svara, no GPU)", "P2 v6_teacher (single-stage iSTFT, 150m cap)", "P3 v6_nano (<3 MB, 150m cap)", "P4 v6_studio (cartridges + Pages)"]
V6 = "/work/tts/runs/v6_chain.json"
VERS = "/work/tts/repo/docs/samples/versions"


def _chain():
    c = L.rj(V6, {}) or {}
    return c if isinstance(c, dict) else {}


def _stage_state(c, name):
    s = (c.get("stages") or {}).get(name) or {}
    if not s: return "wait", "waiting"
    if s.get("state") == "running": return "on", "training"
    if s.get("state") in ("aborted", "failed"): return "bad", s.get("state")
    if s.get("gate_pass") is False: return "bad", "gate not passed"
    if s.get("shipped"): return "done", "shipped"
    return "on", s.get("state", "?")


def _countdown(c, name):
    s = (c.get("stages") or {}).get(name) or {}
    if s.get("state") != "running" or not s.get("started"): return None
    return max(0.0, s.get("cap_min", 150) - (time.time() - s["started"]) / 60.0)


@safe
def mission():
    S = L.snapshot(); g = S["gpu"]; c = _chain()
    busy = bool(S["procs"]) or g.get("util", 0) >= 5
    st_t, _ = _stage_state(c, "v6_teacher"); st_n, _ = _stage_state(c, "v6_nano")
    studio = os.path.exists("/work/tts/repo/docs/cartridges.json")
    kinds = ["done", st_t, st_n, "done" if studio else "wait"]
    steps = "".join(f'<span class="st {"done" if k == "done" else ("on" if k == "on" else "")}">{E(s)}{" · FAILED GATE" if k == "bad" else ""}</span>' for s, k in zip(STAGES, kinds))
    run = next((n for n in ("v6_teacher", "v6_nano") if _countdown(c, n) is not None), None)
    cd = _countdown(c, run) if run else None
    p = S["am"] if run == "v6_teacher" else S["voc"]
    say = (f"{run} running · step {p.get('step') or 0:,} · {fmt(p.get('sps'), 1)} steps/s · ETA {fmt(p.get('eta_min'), 0)} min · hard cap left {fmt(cd, 0)} min"
           if run else (c.get("v6_chandas") and "v6_chandas shipped · chain idle / waiting for next stage") or "waiting for v6 chain")
    gc = (f'<div class="cd"><h3>RTX PRO 6000 Blackwell (98 GB){chip("busy" if busy else "idle", "run" if busy else "wait")}</h3>{bar(g.get("util", 0))}'
          f'<small style="margin:5px 0 10px">GPU {fmt(g.get("util"), 0)} %</small>{bar(g.get("mem", 0) / (g.get("mem_total") or 98) * 100, "#7fd2f2")}'
          f'<div class="tls">{tile(fmt(g.get("mem"), 1) + " / " + fmt(g.get("mem_total", 98), 0) + " GB", "VRAM")}{tile(fmt(g.get("temp"), 0) + " °C", "temp")}{tile(fmt(g.get("power"), 0) + " W", "power")}</div></div>')
    rows = "".join(f'<tr><td>{p_["pid"]}</td><td>{dur(p_["up"])}</td><td>{E(p_["cmd"])}</td></tr>' for p_ in S["procs"]) or '<tr><td colspan="3"><small>No v6 job running right now</small></td></tr>'
    caps = "".join(chip(f"{n} cap left {fmt(_countdown(c, n), 0)} min", "run") for n in ("v6_teacher", "v6_nano") if _countdown(c, n) is not None) or chip("no stage timer running", "wait")
    return (f'<div class="sk"><div class="cd w"><h3>Sakhiya v6 · Chandas Mantra Engine → single-stage teacher → sub-3 MB Nano → Studio<span class="sp"></span>{chip(time.strftime("%H:%M:%S"))}</h3>'
            f'<div class="big">{E(say)}</div><div class="steps">{steps}</div><div style="margin-top:10px">{caps}{chip("ships only on a real audio gate pass", "ok")}</div></div>{gc}'
            f'<div class="cd"><h3>Active PIDs &amp; v6 chain</h3><table class="bt"><tr><th>pid</th><th>up</th><th>command</th></tr>{rows}</table></div></div>')


def _tcard(title, d, fields):
    if not d.get("ok"): return f'<div class="cd"><h3>{title}{chip("Waiting for Stage...", "wait")}</h3><small>No log yet</small></div>'
    ts = "".join(tile((f"{float(d[k]):.1e}" if d.get(k) is not None and k == "lr" else fmt(d.get(k), 4)) if k not in ("step",) else f'{d.get("step") or 0:,}', n) for k, n in fields)
    live = d["age"] < 180
    return (f'<div class="cd"><h3>{title}{chip("live" if live else "idle " + dur(d["age"]), "run" if live else "")}</h3><div class="tls">{ts}</div>'
            f'{spark(d.get("y"))}<pre class="lg">{E(chr(10).join(d.get("tail") or ["(empty)"]))}</pre></div>')


@safe
def training():
    S = L.snapshot(); a, v = S["am"], S["voc"]; c = _chain()
    for d, n in ((a, "v6_teacher"), (v, "v6_nano")):
        cd = _countdown(c, n)
        if cd is not None: d["rem_cap_min"] = cd
    if "v6_nano" not in (c.get("stages") or {}): v.clear(); v["ok"] = False
    af = [("step", "step"), ("sps", "steps/s"), ("eta_min", "ETA min"), ("rem_cap_min", "cap left (min)"), ("mel_l1", "mel_l1"), ("stft_loss", "rollout mel l1")]
    vf = [("step", "step"), ("sps", "steps/s"), ("eta_min", "ETA min"), ("rem_cap_min", "cap left (min)"), ("mel_l1", "mel_l1"), ("dur_l1", "dur_l1")]
    return f'<div class="sk">{_tcard("v6_teacher (single-stage iSTFT · 100 min cap)", a, af)}{_tcard("v6_nano (sub-3 MB looped student · 100 min cap)", v, vf)}</div>'


def _forensic(name):
    return L.rj(f"/work/tts/repo/eval/forensic_{name}.json", {}) or {}


@safe
def bench200():
    cols = [("v5_c1 (baseline)", "forensic_v5_c1"), ("v6_chandas", "forensic_v6_chandas"), ("v6_teacher", "forensic_v6_istft_teacher"), ("v6_nano", "forensic_v6_nano")]
    F = [_forensic(n.replace("forensic_", "")) for _, n in cols]
    c = _chain(); tg = ((c.get("stages") or {}).get("v6_teacher") or {}).get("gate_pass"); ng = ((c.get("stages") or {}).get("v6_nano") or {}).get("gate_pass")
    def row(label, fn):
        return f"<tr><td><b>{label}</b></td>" + "".join(f"<td>{E(fn(f)) if f else 'not run'}</td>" for f in F) + "</tr>"
    nas = lambda f: fmt(((f.get("nasal_scores") or {}).get("काठमाडौँ") or {}).get("ctc_nasal_prob"), 3)
    head = "".join(f"<th>{E(t)}</th>" for t, _ in cols)
    badge = chip("teacher gate PASS" if tg else ("teacher gate not passed" if tg is False else "teacher pending"), "ok" if tg else ("bad" if tg is False else "wait")) + chip("nano gate PASS" if ng else ("nano gate not passed" if ng is False else "nano pending"), "ok" if ng else ("bad" if ng is False else "wait"))
    return (f'<div class="sk"><div class="cd w"><h3>Real audio forensic gate · single voice sakhiya (F){badge}</h3>'
            f'<table class="bt"><tr><th>measured on synthesized 24 kHz audio</th>{head}</tr>'
            + row("word drops / stutters / F0 shutters", lambda f: f'{f.get("word_drop_count")} / {f.get("stutter_count")} / {f.get("f0_shutters")}')
            + row("clicks (15 master clips)", lambda f: f.get("clicks"))
            + row("काठमाडौँ nasal prob (gate ≥ 0.50)", nas)
            + row("LTAS MCD ne↔sa / SV cosine", lambda f: f'{fmt(f.get("ltas_mcd_ne_sa"), 2)} / {fmt(f.get("sv_cos_ne_sa"), 3)}')
            + row("ne pCER / sa pCER", lambda f: f'{fmt(f.get("ne_pcer"), 4)} / {fmt(f.get("sa_pcer"), 4)}')
            + row("UTMOS22 (info only, English-trained)", lambda f: fmt(f.get("utmos_15"), 2))
            + '</table></div></div>')


@safe
def audio_ab():
    import base64
    vers = [("v5_c1", "v5-C1"), ("v6_chandas", "v6 Chandas"), ("v6_istft_teacher", "v6 Teacher"), ("v6_nano", "v6 Nano")]
    rows = []
    for cid, label in [("ne_kathmandu", "1. काठमाडौँ (News)"), ("ne_story", "2. Storytelling"), ("sa_gayatri", "11. Gayatri Mantra"), ("sa_rv", "9. Rigveda 1.1.1"), ("sa_gita", "12. Gita 1.1"), ("sa_om", "13. ॐ पूर्णमदः")]:
        def tag(ver):
            p = f"{VERS}/{ver}/{cid}.mp3"
            if not os.path.exists(p): return "<small>not shipped</small>"
            b64 = base64.b64encode(open(p, "rb").read()).decode()
            return f'<audio controls preload="none" style="height:30px;max-width:155px" src="data:audio/mp3;base64,{b64}"></audio>'
        rows.append(f"<tr><td><b>{E(label)}</b></td>" + "".join(f"<td>{tag(v)}</td>" for v, _ in vers) + "</tr>")
    return (f'<div class="sk"><div class="cd w"><h3>Immutable version player{chip("only gate-passing versions appear", "ok")}</h3>'
            f'<table class="bt"><tr><th>Master clip</th>{"".join(f"<th>{E(t)}</th>" for _, t in vers)}</tr>{"".join(rows)}</table></div></div>')


@safe
def gate():
    G = L.gate(); sents, g = G["sents"], G["gate"]
    voices = ["sakhiya", "sandesh"]; res = g.get("voices", {}) if isinstance(g.get("voices"), dict) else {}
    def cell(vo, sid):
        r = ((res.get(vo) or {}).get("sentences") or {}).get(sid)
        if not isinstance(r, dict): return chip("waiting", "wait")
        bad = r.get("bad") or []
        return chip("PASS", "ok") if r.get("pass") else chip("FAIL " + ", ".join(map(str, bad)), "bad")
    rows = "".join(f'<tr><td><b>{E(s.get("label", s.get("id")))}</b></td>' + "".join(f"<td>{cell(v, s.get('id'))}</td>" for v in voices) + "</tr>" for s in sents) or '<tr><td colspan="3"><small>Waiting for sentence list</small></td></tr>'
    sv = "".join(chip(f"{v} WavLM-SV {fmt((res.get(v) or {}).get('cos_sim'), 3)}", "ok" if ((res.get(v) or {}).get("cos_sim") or 0) >= 0.88 else "wait") for v in voices)
    st = g.get("style", {}) if isinstance(g.get("style"), dict) else {}
    f0n, f0s = st.get("news_f0_std"), st.get("storytelling_f0_std")
    sep = chip(f"F0-std storytelling {fmt(f0s, 1)} vs news {fmt(f0n, 1)}", "ok" if (f0n and f0s and f0s > f0n * 1.15) else "wait")
    sv_st = g.get("svara_st"); svc = chip(f"Rigveda svara {fmt(sv_st, 2)} st", "ok" if (sv_st or 0) >= 4.0 else "wait")
    upd = g.get("updated"); when = f"updated {dur(time.time() - upd)} ago" if upd else "Waiting for Stage 4 gate run"
    return (f'<div class="sk"><div class="cd w"><h3>v2.6 pronunciation &amp; quality gate{chip("promote" if g.get("all_pass") else "not promoted", "ok" if g.get("all_pass") else "wait")}</h3>'
            f'<small>{when}</small><table class="bt"><tr><th>owner sentence</th><th>sakhiya</th><th>sandesh</th></tr>{rows}</table>'
            f'<div style="margin-top:12px">{sv}{sep}{svc}</div></div></div>')
