#!/usr/bin/env python3
"""Every ~20 min: export + REAL int8 gate on the newest checkpoint of every ladder tier, keep the best passing candidate per tier
(never a regression), refresh /work/sp/tiers/leaderboard.json, the listening desk (/work/sp/listen/student_<tier>/),
and trigger a live Marimo card refresh.
usage (detached):  python3 ladder_daemon.py [--every 1200] [--tiers t30,t15,t10,t5,t1] [--once]"""
import os, sys, json, glob, time, shutil, subprocess, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
RUNS, OUT, LISTEN = '/work/sp/distill/runs', '/work/sp/tiers', '/work/sp/listen'

REFS = [
    dict(name='Sakhiya-Pro v1 (ft3_s200 Teacher)', size_mb=2400.0, utmos=3.973, cer=0.064, sa_cer=0.137, drops=0, stutters=0, clicks=35, spk=0.670, rtf='GPU', status='anchor teacher'),
    dict(name='Sakhiya-Pro v2 (s2_300 Teacher)', size_mb=2400.0, utmos=3.796, cer=0.066, sa_cer=0.147, drops=0, stutters=0, clicks=44, spk=0.669, rtf='GPU', status='shipped v2'),
    dict(name='Sakhiya-V1 (Live Site)', size_mb=15.98, utmos=3.957, cer=0.014, sa_cer=0.025, drops=0, stutters=0, clicks=0, spk=0.987, rtf=0.1884, status='shipped'),
    dict(name='Sakhiya-V1-Nano (Live Site)', size_mb=8.93, utmos=3.253, cer=0.027, sa_cer=0.046, drops=0, stutters=0, clicks=0, spk=0.960, rtf=0.0217, status='shipped'),
    dict(name='Sakhiya-V1-Atom (Live Site)', size_mb=6.51, utmos=3.262, cer=0.017, sa_cer=0.068, drops=0, stutters=0, clicks=0, spk=0.955, rtf=0.0189, status='shipped'),
    dict(name='Sakhiya-V1-Atom-Lite (Live Site)', size_mb=2.72, utmos=2.944, cer=0.017, sa_cer=0.040, drops=0, stutters=0, clicks=0, spk=0.940, rtf=0.0155, status='shipped'),
]


def leaderboard():
    rows = list(REFS)
    for tier in ('t30', 't15', 't10', 't5', 't1'):
        bp = f'{OUT}/{tier}/best/gate.json'
        if not os.path.exists(bp):
            cands = sorted(glob.glob(f'{OUT}/{tier}/c_*/gate.json'))
            bp = cands[-1] if cands else bp
        if os.path.exists(bp):
            g = json.load(open(bp)); s = g.get('summary', {})
            drops = s.get('word_drop_count', s.get('word_drop_proxy', 0))
            stut = s.get('stutter_count', 0)
            rows.append(dict(
                name=f'Sakhiya-{tier} (step {g["step"]}{", QAT" if g.get("qat") else ""})',
                size_mb=g['int8_mb'],
                utmos=s.get('utmos'),
                cer=s.get('cer_ne'),
                sa_cer=s.get('cer_sa'),
                drops=drops,
                stutters=stut,
                clicks=s.get('clicks'),
                spk=s.get('spk_sim'),
                rtf=g.get('rtf_1thread'),
                status=('PASS' if g['gate_pass'] else 'FAIL') + f" · corr {g['int8_vs_fp32_mel_corr']} · drops {drops} · stutters {stut} · clicks {s.get('clicks')} · sa-CER {s.get('cer_sa')}"
            ))
        else:
            rows.append(dict(
                name=f'Sakhiya-{tier}',
                size_mb='~' + str({'t30': 29.6, 't15': 14.5, 't10': 9.5, 't5': 4.8, 't1': 1.05}[tier]),
                utmos='-', cer='-', sa_cer='-', drops='-', stutters='-', clicks='-', spk='-', rtf='-',
                status='training / awaiting INT8 gate'
            ))
    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, open(OUT + '/leaderboard.json', 'w'), ensure_ascii=False, indent=1)


def evaluate(tier, ck, tag):
    d = f'{OUT}/{tier}/c_{tag}'; os.makedirs(d, exist_ok=True)
    r = subprocess.run([sys.executable, f'{HERE}/export_gate.py', tier, ck, d, '--mp3'], capture_output=True, text=True, timeout=1500, env=dict(os.environ, PYTHONPATH='/work/sp'))
    open(d + '/export.log', 'w').write(r.stdout[-3000:] + r.stderr[-3000:])
    gp = d + '/gate.json'
    if not os.path.exists(gp):
        print(tier, 'export failed', r.stderr[-300:], flush=True)
        return
    g = json.load(open(gp)); best = f'{OUT}/{tier}/best/gate.json'
    cur = json.load(open(best)) if os.path.exists(best) else None
    if g['gate_pass'] and (cur is None or g.get('score', -9) > cur.get('score', -9)):
        shutil.rmtree(f'{OUT}/{tier}/best', ignore_errors=True)
        shutil.copytree(d, f'{OUT}/{tier}/best', ignore=shutil.ignore_patterns('model.fp32.onnx'))
        ld = f'{LISTEN}/student_{tier}'; shutil.rmtree(ld, ignore_errors=True); os.makedirs(ld, exist_ok=True)
        for f in glob.glob(d + '/wav/*.mp3') + glob.glob(d + '/wav/*.wav'):
            shutil.copy(f, ld)
        print(f'[{time.strftime("%T")}] {tier} PROMOTED step {g["step"]} score {g.get("score")} {g["int8_mb"]} MB', flush=True)
    else:
        print(f'[{time.strftime("%T")}] {tier} step {g["step"]} not promoted (pass={g["gate_pass"]} score={g.get("score")} best={cur and cur.get("score")})', flush=True)
    if not (g['gate_pass'] and cur is None):
        shutil.rmtree(d + '/wav', ignore_errors=True)
    for old in sorted(glob.glob(f'{OUT}/{tier}/c_*'))[:-2]:
        shutil.rmtree(old, ignore_errors=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--every', type=int, default=1200)
    ap.add_argument('--tiers', default='t30,t15,t10,t5,t1')
    ap.add_argument('--once', action='store_true')
    a = ap.parse_args()
    last = {}
    leaderboard()
    while True:
        for t in a.tiers.split(','):
            fin = f'{RUNS}/{t}/ema_final.pt'
            ck = fin if os.path.exists(fin) else f'{RUNS}/{t}/ema_latest.pt'
            if not os.path.exists(ck):
                continue
            key = (ck, os.path.getmtime(ck))
            if last.get(t) == key or (t in last and last[t][0] == fin):
                continue
            if not a.once and time.time() - os.path.getmtime(ck) < 20:
                continue
            last[t] = key
            try:
                evaluate(t, ck, time.strftime('%H%M%S'))
            except Exception as e:
                print(t, 'ERR', str(e)[:200], flush=True)
            leaderboard()
        leaderboard()
        if a.once:
            break
        time.sleep(a.every)
