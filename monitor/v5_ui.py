"""Sakhiya v5 recovery dashboard helpers (5 cards). Every function is defensive: it never raises, it returns HTML/mo objects.
State lives in /work/tts/v5_recovery/state/*.json (written by the pipeline scripts); audio in /work/tts/v5_recovery/listen/<group>/."""
import datetime, glob, html, json, os, subprocess

V5 = "/work/tts/v5_recovery"
ST = V5 + "/state"
LISTEN = V5 + "/listen"
GROUPS = ["nepali", "episode01", "vedic", "micro", "pico"]
GROUP_LABEL = {"nepali": "Clean Nepali (v5 master)", "episode01": "Episode 01 (v5 broadcast)", "vedic": "Isolated Vedic",
               "micro": "Micro (INT8 · live v4 baseline)", "pico": "Pico (INT8 · live v4 baseline)"}
STYLE = ("<style>.v5{font-family:system-ui,sans-serif;background:#10151c;color:#e8edf3;padding:14px 16px;border-radius:12px;line-height:1.5}"
         ".v5 h3{margin:0 0 8px;font-size:17px}.v5 table{border-collapse:collapse;width:100%;font-size:13px}"
         ".v5 td,.v5 th{border-bottom:1px solid #26303c;padding:4px 8px;text-align:left}.ok{color:#4ade80}.bad{color:#f87171}"
         ".warn{color:#fbbf24}.dim{color:#8b98a8}</style>")


def _sh(c, t=6):
    try:
        return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t).stdout.strip()
    except Exception:
        return ""


def _j(name, default=None):
    try:
        return json.load(open(f"{ST}/{name}"))
    except Exception:
        return default


def _wrap(title, body):
    return f'{STYLE}<div class="v5"><h3>{html.escape(title)}</h3>{body}</div>'


def _deadline():
    try:
        s = open("/work/sp/deadline_utc.txt").read().strip().rstrip("Z")
        return datetime.datetime.fromisoformat(s).replace(tzinfo=datetime.timezone.utc)
    except Exception:
        return datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=9)


def card1():
    try:
        now = datetime.datetime.now(datetime.timezone.utc)
        dl = _deadline(); stop = dl - datetime.timedelta(minutes=30)
        left = dl - now; sleft = stop - now
        f = lambda d: f"{int(d.total_seconds() // 3600)}h {int(d.total_seconds() % 3600 // 60):02d}m" if d.total_seconds() > 0 else "0h 00m"
        npt = _sh("TZ=Asia/Kathmandu date '+%a %d %b · %H:%M:%S NPT'")
        gpu = _sh("nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader")
        g = [x.strip() for x in gpu.split(",")] if gpu else []
        gl = f"{g[0]} · util {g[1]} · {g[2]} / {g[3]} · {g[4]} °C" if len(g) >= 5 else "GPU not responding"
        bk = _sh("pgrep -f 'hf_backup[.]py' | head -1")
        last = _sh(f"tail -n 1 {V5}/../logs/hf_backup_daemon.log | cut -c1-120")
        cls = "ok" if sleft.total_seconds() > 3600 else ("warn" if sleft.total_seconds() > 0 else "bad")
        body = (f"<table><tr><td>Now</td><td>{npt}</td></tr>"
                f"<tr><td>Box expiry</td><td>{dl:%H:%M} UTC · <b>{f(left)}</b> left</td></tr>"
                f"<tr><td>Hard stop (30 min safety)</td><td class='{cls}'><b>{f(sleft)}</b> until all exports/pushes must be done</td></tr>"
                f"<tr><td>GPU</td><td>{html.escape(gl)}</td></tr>"
                f"<tr><td>15-min HF backup daemon</td><td class='{'ok' if bk else 'bad'}'>{'running (pid ' + bk + ')' if bk else 'NOT running'}"
                f" <span class='dim'>· {html.escape(last)}</span></td></tr></table>")
        return _wrap("1 · Box countdown & GPU health", body)
    except Exception as e:
        return _wrap("1 · Box countdown & GPU health", f"<span class='warn'>refreshing… {html.escape(repr(e)[:80])}</span>")


def card2():
    try:
        a = _j("rollback_audit.json")
        if not a:
            return _wrap("2 · Rollback audit", "<span class='dim'>audit not run yet</span>")
        rows = ""
        for k, v in (a.get("nepali_accent") or {}).items():
            vals = " · ".join(f"{kk}={vv}" for kk, vv in v.items() if kk != "src" and isinstance(vv, (int, float)))
            rows += f"<tr><td>{html.escape(k)}</td><td>{html.escape(vals[:200])}</td></tr>"
        v4 = "".join(f"<tr><td>{html.escape(r['pack'].replace('sakhiya_pro_', ''))}/{r['id']}</td>"
                     f"<td class='{'ok' if r['ok'] else 'bad'}'>{'PASS' if r['ok'] else 'REJECT'}</td>"
                     f"<td>{html.escape(', '.join(r['rejects'])[:140])}</td></tr>" for r in a.get("v4_vedic", []))
        nep = ""
        for f in sorted(glob.glob(ST + "/nepali_*.json")):
            try:
                j = json.load(open(f)); g = j["agg"]
                nep += (f"<tr><td>{html.escape(j['tag'])}</td><td>gate {g['gate_pass']}/{g['clips']} · all-thresholds {g['all_pass']}/{g['clips']} · "
                        f"end_slope {g['end_slope']} · rise {g['rise_frac']} · CER {g['cer']} · UTMOS {g['utmos']}</td></tr>")
            except Exception:
                pass
        cause = "".join(f"<li>{html.escape(c)}</li>" for c in a.get("root_cause", []))
        body = (f"<b>Poisoned v4 Vedic:</b> <span class='bad'>{a.get('v4_rejected')}/{len(a.get('v4_vedic', []))} rejected</span> · "
                f"leaked-prompt clips <b>{a.get('v4_leak_clips')}</b> · repeated-loop (stutter) clips <b>{a.get('v4_loop_clips')}</b>"
                f"<table>{v4}</table><br><b>Nepali v5 candidates (v3b-600, gate + thresholds)</b><table>{nep}</table><br><b>Nepali checkpoints (stored evals)</b><table>{rows}</table><br><b>Root cause</b><ul>{cause}</ul>")
        return _wrap("2 · Rollback audit (Pro v3b 400/200/600 vs poisoned v4)", body)
    except Exception as e:
        return _wrap("2 · Rollback audit", f"<span class='warn'>{html.escape(repr(e)[:100])}</span>")


def _kv_card(title, name, empty):
    try:
        s = _j(name)
        if not s:
            return _wrap(title, f"<span class='dim'>{html.escape(empty)}</span>")
        rows = "".join(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(json.dumps(v, ensure_ascii=False)[:260])}</td></tr>" for k, v in s.items())
        return _wrap(title, f"<table>{rows}</table>")
    except Exception as e:
        return _wrap(title, f"<span class='warn'>{html.escape(repr(e)[:100])}</span>")


def card3():
    return _kv_card("3 · Isolated Vedic LoRA + Chandas metrics", "vedic.json", "Vedic adapter work has not started (Nepali is locked first; weights stay frozen).")


def card4():
    return _kv_card("4 · Serial shrink ladder (Micro ≤ 4.5 MB · Pico ≤ 1.2 MB)", "shrink.json", "Shrink ladder has not started.")


def listen_files(group):
    out = []
    for ext in ("wav", "mp3"):
        out += sorted(glob.glob(f"{LISTEN}/{group}/*.{ext}"))
    return [p for p in out if os.path.isfile(p) and os.path.getsize(p) > 1000]


def card5_head():
    n = {g: len(listen_files(g)) for g in GROUPS}
    return _wrap("5 · Listening desk", " · ".join(f"{GROUP_LABEL[g]}: <b>{n[g]}</b>" for g in GROUPS))


def desk(clip_name, mo):
    """Play the same clip name from every group side by side (only files that exist)."""
    cols = []
    for g in GROUPS:
        p = next((x for x in listen_files(g) if os.path.splitext(os.path.basename(x))[0] == clip_name), None)
        if p:
            try:
                cols.append(mo.vstack([mo.md(f"**{GROUP_LABEL[g]}**"), mo.audio(open(p, "rb").read())]))
            except Exception:
                pass
    return mo.hstack(cols, justify="start", wrap=True) if cols else mo.md("_no clip with this name yet_")


def clip_names():
    names = []
    for g in GROUPS:
        for p in listen_files(g):
            n = os.path.splitext(os.path.basename(p))[0]
            if n not in names:
                names.append(n)
    return names or ["(none yet)"]
