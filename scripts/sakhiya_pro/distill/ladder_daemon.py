#!/usr/bin/env python3
"""Every ~20 min: export + REAL int8 gate on the newest checkpoint of every ladder tier, keep the best passing candidate per tier
(never a regression), refresh /work/sp/tiers/leaderboard.json and the listening desk (/work/sp/listen/student_<tier>/).
usage (detached):  python3 ladder_daemon.py [--every 1200] [--tiers t30,t15,t10,t5,t1]"""
import os, sys, json, glob, time, shutil, subprocess, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
RUNS, OUT, LISTEN = '/work/sp/distill/runs', '/work/sp/tiers', '/work/sp/listen'
REFS = [dict(name='Sakhiya-V1 (old gate)', size_mb=15.98, utmos=3.957, cer=0.014, spk='-', rtf=0.188, status='shipped'),
        dict(name='Atom-Lite (old gate)', size_mb=2.72, utmos='-', cer='-', spk='-', rtf='-', status='shipped')]


def leaderboard():
    rows = list(REFS)
    for p in ('/work/sp/evals/pron_v2_final/pron_scores.json',):
        pass
    t = '/work/sp/tiers/teacher.json'
    if os.path.exists(t): rows.append(json.load(open(t)))
    for tier in ('t30', 't15', 't10', 't5', 't1'):
        bp = f'{OUT}/{tier}/best/gate.json'
        if os.path.exists(bp):
            g = json.load(open(bp)); s = g.get('summary', {})
            rows.append(dict(name=f'Sakhiya-{tier} (step {g["step"]}{", QAT" if g.get("qat") else ""})', size_mb=g['int8_mb'], utmos=s.get('utmos'), cer=s.get('cer_ne'), spk=s.get('spk_sim'), rtf=g.get('rtf_1thread'),
                             status=('PASS' if g['gate_pass'] else 'FAIL') + f" · corr {g['int8_vs_fp32_mel_corr']} · clicks {s.get('clicks')} · drops {s.get('word_drop_proxy')} · sa-CER {s.get('cer_sa')}"))
        else: rows.append(dict(name=f'Sakhiya-{tier}', size_mb='~' + str({'t30': 30, 't15': 14.5, 't10': 9.5, 't5': 4.8, 't1': 1.2}[tier]), utmos='-', cer='-', spk='-', rtf='-', status='training / no passing export yet'))
    os.makedirs(OUT, exist_ok=True); json.dump(rows, open(OUT + '/leaderboard.json', 'w'), ensure_ascii=False, indent=1)


def evaluate(tier, ck, tag):
    d = f'{OUT}/{tier}/c_{tag}'; os.makedirs(d, exist_ok=True)
    r = subprocess.run([sys.executable, f'{HERE}/export_gate.py', tier, ck, d, '--mp3'], capture_output=True, text=True, timeout=1500, env=dict(os.environ, PYTHONPATH='/work/sp'))
    open(d + '/export.log', 'w').write(r.stdout[-3000:] + r.stderr[-3000:])
    gp = d + '/gate.json'
    if not os.path.exists(gp): print(tier, 'export failed', r.stderr[-300:], flush=True); return
    g = json.load(open(gp)); best = f'{OUT}/{tier}/best/gate.json'
    cur = json.load(open(best)) if os.path.exists(best) else None
    if g['gate_pass'] and (cur is None or g.get('score', -9) > cur.get('score', -9)):
        shutil.rmtree(f'{OUT}/{tier}/best', ignore_errors=True); shutil.copytree(d, f'{OUT}/{tier}/best', ignore=shutil.ignore_patterns('model.fp32.onnx'))
        ld = f'{LISTEN}/student_{tier}'; shutil.rmtree(ld, ignore_errors=True); os.makedirs(ld, exist_ok=True)
        for f in glob.glob(d + '/wav/*.mp3'): shutil.copy(f, ld)
        print(f'[{time.strftime("%T")}] {tier} PROMOTED step {g["step"]} score {g.get("score")} {g["int8_mb"]} MB', flush=True)
    else: print(f'[{time.strftime("%T")}] {tier} step {g["step"]} not promoted (pass={g["gate_pass"]} score={g.get("score")} best={cur and cur.get("score")})', flush=True)
    shutil.rmtree(d + '/wav', ignore_errors=True) if not (g['gate_pass'] and cur is None) else None
    for old in sorted(glob.glob(f'{OUT}/{tier}/c_*'))[:-2]: shutil.rmtree(old, ignore_errors=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--every', type=int, default=1200); ap.add_argument('--tiers', default='t30,t15,t10,t5,t1'); a = ap.parse_args()
    last = {}
    while True:
        for t in a.tiers.split(','):
            fin = f'{RUNS}/{t}/ema_final.pt'; ck = fin if os.path.exists(fin) else f'{RUNS}/{t}/ema_latest.pt'
            if not os.path.exists(ck): continue
            key = (ck, os.path.getmtime(ck))
            if last.get(t) == key or (t in last and last[t][0] == fin): continue
            if time.time() - os.path.getmtime(ck) < 20: continue
            last[t] = key
            try: evaluate(t, ck, time.strftime('%H%M'))
            except Exception as e: print(t, 'ERR', str(e)[:200], flush=True)
            leaderboard()
        leaderboard(); time.sleep(a.every)
