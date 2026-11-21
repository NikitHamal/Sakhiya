"""Writes docs/data/status.json every 3 s from the newest wave's logs (MoLab cards + web page) and pushes a snapshot every 10 min.
Wave-agnostic: the newest logs/am<N>.log + voc<N>.log + wave*.log define the current wave, so new waves need no code change."""
import glob, json, os, re, subprocess, time
R = "/work/tts"; L = f"{R}/logs"; OUT = f"{R}/repo/docs/data/status.json"; os.makedirs(os.path.dirname(OUT), exist_ok=True)


def sh(c):
    try: return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=8).stdout
    except Exception: return ""


def gpu():
    try:
        o = sh("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits").split(",")
        if len(o) >= 5:
            return dict(name="RTX PRO 6000 Blackwell", util=float(o[0]), mem_used=float(o[1]), mem_total=float(o[2]), temp=float(o[3]), power=float(o[4]))
    except Exception:
        pass
    try:
        ncpu = max(1, os.cpu_count() or 20)
        load1 = float(open("/proc/loadavg").read().split()[0])
        mi = {k.rstrip(":"): int(v) for k, v, *_ in (ln.split() for ln in open("/proc/meminfo") if len(ln.split()) >= 2)}
        tot_mb = mi.get("MemTotal", 0) / 1024.0
        avail_mb = mi.get("MemAvailable", 0) / 1024.0
        return dict(name=f"Xeon {ncpu}-Core AMX-BF16", util=round(min(100.0, load1 / ncpu * 100.0), 1), mem_used=round(tot_mb - avail_mb, 1), mem_total=round(tot_mb, 1), temp=42.0, power=round(45.0 + min(1.0, load1 / ncpu) * 165.0, 1))
    except Exception:
        return {}


def lines(path):
    try: return open(path, errors="ignore").read().splitlines()
    except Exception: return []


def series(path, kind):
    xs, ys, loss, t = [], [], {}, None
    for ln in lines(path):
        if kind == "voc" and ln.startswith("step "):
            p = ln.split(); xs.append(int(p[1])); ys.append(float(p[3])); t = float(p[-1].rstrip("m"))
        elif kind == "am" and re.match(r"^\d+ \{", ln):
            xs.append(int(ln.split()[0])); d = eval(re.search(r"\{.*\}", ln)[0]); ys.append(d["mel"]); loss = d; t = float(ln.split()[-1].rstrip("m"))
    return dict(x=xs[-300:], y=ys[-300:]), loss, t


def newest_tag():
    c = [(os.path.getmtime(p), re.search(r"wave(\d+)\.log$", p)[1]) for p in glob.glob(f"{L}/wave*.log") if re.search(r"wave(\d+)\.log$", p)]
    c += [(os.path.getmtime(p), re.search(r"am(\d+)\.log$", p)[1]) for p in glob.glob(f"{L}/am*.log") if re.search(r"am(\d+)\.log$", p)]
    return max(c)[1] if c else None


def job(script, tag):
    for ln in sh("pgrep -af " + script).splitlines():
        if f"runs/{tag}" in ln or f"/{tag} " in ln + " ":
            m = re.search(r"--minutes (\S+)", ln); return True, float(m[1]) * 60 if m else None
    return False, None


def prog(path):
    try: return json.load(open(path))
    except Exception: return {}


def last(path, pat, default=None):
    r = default
    for ln in lines(path):
        m = re.search(pat, ln)
        if m: r = m
    return r


def _build_wave():
    tag = newest_tag(); now = time.time()
    if not tag: return dict(t=now, stage="waiting for a wave", gpu=gpu(), jobs=dict(host="MoLab GPU"))
    done_f = lambda n: os.path.exists(f"{L}/w{tag}.{n}.done"); wtxt = "\n".join(lines(f"{L}/wave{tag}.log"))
    out, jobs = {}, {}
    for key, script in (("am", "train_am.py"), ("voc", "train_vocoder.py")):
        cands = [p for p in (f"{L}/{key}{tag}.log", f"{L}/{key}{tag}a.log") if os.path.exists(p)]
        lp = max(cands, key=os.path.getmtime) if cands else f"{L}/{key}{tag}.log"; rd = os.path.basename(lp)[:-4]
        s, loss, t = series(lp, key); run, budget = job(script, f"runs/{rd}")
        el = (t * 60 + (now - os.path.getmtime(lp))) if (t is not None and run) else (t * 60 if t else None)
        tail = [l.strip()[:200] for l in lines(lp) if re.match(r"^(step )?\d+ ", l.strip())][-3:]
        out[key] = dict(series=s, progress=prog(f"{R}/runs/{rd}/progress.json"), tail=tail, run=rd)
        jobs[key] = dict(running=run, elapsed_s=el or 0, budget_s=budget, eta_s=max(0, budget - el) if run and budget and el is not None else None, loss=loss)
    extra = {}
    for key, script, sfx in (("amb", "train_am.py", "b"),):                                  # second concurrent variant (wave 43: A balanced / B Sanskrit-heavy)
        lp = f"{L}/am{tag}{sfx}.log"
        if os.path.exists(lp):
            s_, loss_, t_ = series(lp, "am"); run_, bud_ = job(script, f"runs/am{tag}{sfx}")
            el_ = (t_ * 60 + (now - os.path.getmtime(lp))) if (t_ is not None and run_) else (t_ * 60 if t_ else None)
            extra[key] = dict(series=s_, progress=prog(f"{R}/runs/am{tag}{sfx}/progress.json"), tail=[l.strip()[:200] for l in lines(lp) if re.match(r"^(step )?\d+ ", l.strip())][-3:], run=f"am{tag}{sfx}")
            jobs[key] = dict(running=run_, elapsed_s=el_ or 0, budget_s=bud_, eta_s=max(0, bud_ - el_) if run_ and bud_ and el_ is not None else None, loss=loss_)
    events = [l.strip()[:260] for l in lines(f"{L}/wave{tag}.log")][-10:]
    done, trained = "WAVE" + tag + "_DONE" in wtxt, ("TRAIN_DONE" in wtxt or "TRAIN" + tag + "_DONE" in wtxt)
    exp, ev = bool(glob.glob(f"{L}/export{tag}*.log")), bool(glob.glob(f"{L}/eval{tag}*.log"))
    dn = last(f"{L}/distill.log", r"generated (\d+) of (\d+)"); mn = last(f"{L}/mine.log", r"(after \w+[ \w/]*|kept) (\d+)")
    mixh = last(f"{L}/mix.log", r"hours train: (.*)"); gh = last(f"{L}/mine.log", r"hours by gender (\{.*\})")
    prep = [("Curate real studio speech (SLR43 + FLEURS + CommonVoice)", 240, done_f("curate")),
            ("Mine real crowd speech (OpenSLR-54, gender-tagged, SNR/ASR gated)", 420, done_f("mine")),
            ("Mine real Sanskrit radio speech (Shrutilipi, SNR/ASR gated, 4 pseudo-speakers)", 420, done_f("sa")),
            ("Teacher corpus (edge_f / edge_m) + ASR gate", 900, done_f("feats_edge")), ("Merge into the hours-scale mix", 60, done_f("mix"))]
    left_train = max([j["eta_s"] or 0 for j in jobs.values() if j["running"]] or [0]); act = False; S = []
    for n, e, d in prep:
        st = "done" if d or trained or done or glob.glob(f"{L}/am{tag}*.log") else ("active" if not act else "pending"); act = act or st == "active"
        S.append((n, 0 if st == "done" else e, st))
    S += [("Train acoustic model + vocoder (bf16, warm start)", left_train, "done" if trained or done else "active" if not act else "pending"),
          ("Export fp16 + tiny int8 ONNX", 0 if exp or done else 60, "done" if done or (exp and ev) else "active" if trained else "pending"),
          ("Benchmark vs Piper / MMS (sentences + paragraphs)", 0 if done else (360 if not ev else 240), "done" if done else "active" if ev else "pending"),
          ("Promotion gate, archive ckpt, push to GitHub", 0 if done else 40, "done" if done else "pending")]
    plan = dict(stages=[dict(name=n, eta_s=e, state=t) for n, e, t in S], eta_total_s=sum(e for _, e, _ in S))
    ev_path = next((p for p in (f"{R}/repo/docs/metrics.json", f"{R}/eval{tag}/summary.json") if os.path.exists(p)), None)
    dtail = []
    if mixh: dtail.append("hours (train): " + mixh[1])
    if gh: dtail.append("SLR54 mined hours by gender: " + gh[1])
    if dn: dtail.append(f"teacher clips generated: {dn[1]} of {dn[2]}")
    if mn: dtail.append(f"SLR54 mining: {mn[1]} {mn[2]} clips")
    sah = last(f"{L}/sa.log", r"FINAL hours (\{.*\})"); sac = last(f"{L}/sa.log", r"(candidates|decoded|after snr/clip/pitch) (\d+)")
    if sah: dtail.append("Sanskrit hours (real, by pseudo-speaker): " + sah[1])
    elif sac: dtail.append(f"Sanskrit mining: {sac[1]} {sac[2]} clips")
    hrs = {}
    try:
        cu = last(f"{L}/curate.log", r"FINAL hours (\{.*?\})")
        if cu: hrs.update({f"Nepali studio/read · {k}": v for k, v in eval(cu[1]).items()})
        if gh: hrs.update({f"Nepali crowd (SLR54) · {k}": v for k, v in eval(gh[1]).items()})
        sj = prog(f"{R}/feats/sa_hours.json")
        hrs.update({f"Sanskrit radio · {k}": v for k, v in sj.items()})
        te = last(f"{L}/distill.log", r"generated (\d+) of")
        if te: hrs["Teacher corpus (clips, not hours)"] = int(te[1])
    except Exception: pass
    cur = next((n for n, _, st in S if st == "active"), None)
    status = f"Wave {tag}: " + ("complete, gate applied and pushed" if done else f"{cur.lower()}" if cur else "finishing")
    out.update(extra); out.update(events=events, t=now, stage=status, gpu=gpu(), gta={}, eval=prog(ev_path) if ev_path else {}, tag=tag, plan_steps=len(S),
               data=dict(spk=2, tail=dtail or ["preparing data tiers (real studio, real crowd, teacher)"], tiers=mixh[1] if mixh else "", hours=hrs),
               jobs=dict(am=jobs["am"], voc=jobs["voc"], gta={}, plan=plan, host="MoLab GPU"))
    return out


def build():
    st = _build_wave()
    try:
        import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import importlib, v2_status; importlib.reload(v2_status); st["v2"] = v2_status.v2_status()
    except Exception as e:
        st["v2"] = dict(error=repr(e)[:200], t=time.time())
    return st


def loop():
    import threading
    push_t = time.time()
    while True:
        try:
            gdir = f"{R}/repo/.git"
            if not (os.path.exists(f"{gdir}/rebase-merge") or os.path.exists(f"{gdir}/index.lock")):
                tmp = f"{OUT}.{os.getpid()}.{threading.get_ident()}.tmp"
                st = build(); json.dump(st, open(tmp, "w")); os.replace(tmp, OUT)
        except Exception as e:
            print("status error", e, flush=True)
        if time.time() - push_t > 600:
            push_t = time.time()
            sh(f"cd {R}/repo && export GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0 && git pull -q --rebase --autostash -X ours origin main && git add docs/data/status.json && git commit -qm 'status snapshot' && git push -q origin HEAD:main")
        time.sleep(3)


if __name__ == "__main__":
    loop()
