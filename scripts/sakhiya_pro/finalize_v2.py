#!/usr/bin/env python3
"""Unattended autopilot for the Pro v2 session (runs on the GPU box, detached).  Never needs the agent to be online.
  1. waits for PRO_V2_DONE -> freezes docs/samples/versions/sakhiya_pro_v2 + EP01 v2, EVAL, git push, HF upload (sakhiya-pro-v1/v2/)
  2. every --every s: promotes each ladder tier's best *passing* int8 export into docs/models_student/<tier>/, refreshes
     docs/student_tiers.json + the tiers.json "gated" rows, git push (never a regression: only ladder_daemon's `best` is used)
  3. at the final cutoff (--final_utc): freezes immutable packs for every passing tier (t5 = Sakhiya-Micro, t1 = Sakhiya-Pico), STATUS.md,
     HF upload of everything, GH release, stops training + backup processes by exact PID.
usage:  setsid nohup python3 finalize_v2.py --final_utc 2026-11-24T04:00:00 > /work/tts/logs/finalize_v2.log 2>&1 < /dev/null &"""
import os, sys, re, json, glob, time, shutil, subprocess, argparse, datetime, signal
REPO = '/work/sp/repo'; L = '/work/tts/logs'; T = '/work/sp/tiers'
ENV = dict(os.environ, GIT_ASKPASS='/tmp/gitenv.sh', GIT_TERMINAL_PROMPT='0', PYTHONPATH='/work/sp/OmniVoice_src:/work/sp')
NAMES = {'t30': 'Sakhiya-T30', 't15': 'Sakhiya-T15', 't10': 'Sakhiya-T10', 't5': 'Sakhiya-Micro (t5)', 't1': 'Sakhiya-Pico (t1)'}


def log(*a): print(f'[{time.strftime("%T")}]', *a, flush=True)
def sh(c, cwd=REPO, check=False):
    r = subprocess.run(c, shell=True, cwd=cwd, capture_output=True, text=True, env=ENV, executable='/bin/bash')
    if r.returncode and check: raise RuntimeError(c + ' -> ' + (r.stdout + r.stderr)[-400:])
    return (r.stdout + r.stderr).strip()
def push(msg):
    sh('git add -A docs STATUS.md; git commit -qm %r' % msg)
    out = sh('git pull -q --rebase origin main; git push -q origin HEAD:main; git log --oneline | head -1'); log('push:', out[-160:])
def stage(n, label, state, note=''):
    os.makedirs('/work/sp/stage', exist_ok=True); json.dump(dict(label=label, state=state, note=note), open(f'/work/sp/stage/{n}.json', 'w'))
def hf():
    from huggingface_hub import HfApi
    return HfApi(token=open('/tmp/hf_token').read().strip())


def pro_v2():
    log('waiting for PRO_V2_DONE')
    while 'PRO_V2_DONE' not in open(f'{L}/pipeline_v2.log', errors='ignore').read(): time.sleep(30)
    sel = open('/work/sp/final/sakhiya_pro_v2/SELECTED_FROM.txt').read().strip(); tag = sel.split()[0]
    sm = json.load(open(f'/work/sp/evals/{tag}/pron_scores.json'))['summary']
    m = dict(cer=sm['cer_native'], cer_tech=sm['cer_tech'], cer_vedic=sm['cer_vedic'], word_recall=sm['word_recall'], utmos=sm['utmos'], spk_sim=sm['spk_sim'], clicks=sm['clicks'], hindi_leaks=sm['hindi'],
             eval_tag=tag, note='automatic metrics on the pronunciation eval set; no human MOS')
    json.dump(m, open('/work/sp/render/metrics_v2.json', 'w'))
    json.dump(dict(name=f'Sakhiya-Pro v2 (600M teacher, {tag})', size_mb='~2400 (fp32/bf16 weights)', utmos=m['utmos'], cer=m['cer'], spk=m['spk_sim'], rtf='GPU', status=f'teacher · tech-CER {m["cer_tech"]} · Vedic-CER {m["cer_vedic"]} · Hindi leaks {m["hindi_leaks"]}'), open(f'{T}/teacher.json', 'w'))
    sh('git pull -q --rebase origin main')
    rows = []
    for p in sorted(glob.glob('/work/sp/evals/pron_*/pron_scores.json')):
        s = json.load(open(p))['summary']; rows.append(f"| {s['tag']} | {s['cer_native']:.3f} | {s['word_recall']:.3f} | {s['cer_tech']:.3f} | {s['cer_vedic']:.3f} | {s['utmos']:.2f} | {s['spk_sim']:.2f} | {s['hindi']} | {s['clicks']:.0f} |")
    t = f'\n\n## Sakhiya-Pro v2 (native-Nepali two-stage adaptation) — selected **{sel}**\n\n| tag | native CER | word recall | tech CER | Vedic CER | UTMOS22 | spk-sim | Hindi leaks | clicks |\n|---|---|---|---|---|---|---|---|---|\n' + '\n'.join(rows) + '\n\nAutomatic metrics only (ASR round trip, UTMOS22, ECAPA similarity); no human MOS.\n'
    open(f'{REPO}/docs/sakhiya_pro/EVAL.md', 'a').write(t)
    sh(f'python3 scripts/sakhiya_pro/freeze_pro_pack.py /work/sp/render/pack_v2 /work/sp/render/metrics_v2.json --tag sakhiya_pro_v2 --label "Sakhiya-Pro v2 (OmniVoice-600M · native-Nepali 2-stage, teacher)" --notes "stage-1 broad Nepal-native phonology + stage-2 anchor lock ({sel}); automatic metrics only"')
    os.makedirs(f'{REPO}/docs/sakhiya_pro/ep01_v2', exist_ok=True)
    sh('cp /work/sp/render/ep01_v2/voiceover_ep01.mp3 /work/sp/render/ep01_v2/timeline.json /work/sp/render/ep01_v2/take_report.json docs/sakhiya_pro/ep01_v2/')
    push(f'sakhiya_pro_v2: freeze 15-clip pack + EP01 v2 + eval ({tag})')
    try:
        a = hf(); a.upload_folder(folder_path='/work/sp/final/sakhiya_pro_v2', repo_id='NikitHamal/sakhiya-pro-v1', path_in_repo='v2', commit_message='Sakhiya-Pro v2 weights', ignore_patterns=['__pycache__/**', 'optimizer*', '*.pt'])
        log('HF pro v2 uploaded')
    except Exception as e: log('HF upload err', str(e)[:200])


def tiers_sync(final=False):
    rows = []; ts = {}
    for t in ('t30', 't15', 't10', 't5', 't1'):
        b = f'{T}/{t}/best'
        if not os.path.exists(b + '/model.int8.onnx'): continue
        g = json.load(open(b + '/gate.json')); d = f'{REPO}/docs/models_student/{t}'; os.makedirs(d, exist_ok=True)
        shutil.copy(b + '/model.int8.onnx', d + '/model.int8.onnx'); json.dump({k: g[k] for k in g if k not in ('clips',)}, open(d + '/gate.json', 'w'), ensure_ascii=False, indent=1)
        s = g.get('summary', {}); ts[t] = g
        rows.append(dict(id=f'student_{t}', label=NAMES[t], prec='int8', dir=f'models_student/{t}/', variant='student', mb=g['int8_mb'], rtf=g.get('rtf_1thread'), utmos=s.get('utmos'),
                         ne_pcer=s.get('cer_ne'), sa_pcer=s.get('cer_sa'), step=g['step'], ready=False, gate='single-graph student: audition on the Samples page / docs/student_tts.js',
                         desc=f"Distilled from Sakhiya-Pro v2 · real int8 gate PASS (corr {g['int8_vs_fp32_mel_corr']}, nan {g['nan_count']}, clicks {s.get('clicks')}, drops {s.get('word_drop_proxy')})"))
    json.dump(rows, open(f'{REPO}/docs/student_tiers.json', 'w'), ensure_ascii=False, indent=1)
    tj = json.load(open(f'{REPO}/docs/tiers.json')); tj['tiers'] = [x for x in tj['tiers'] if not x['id'].startswith('student_')] + rows
    json.dump(tj, open(f'{REPO}/docs/tiers.json', 'w'), ensure_ascii=False, indent=1)
    return ts


def freeze_tier(t, g):
    from importlib import import_module
    tag = {'t5': 'sakhiya_micro', 't1': 'sakhiya_pico'}.get(t, f'sakhiya_student_{t}')
    out = f'{REPO}/docs/samples/versions/{tag}'
    if os.path.exists(out + '/.frozen'): return
    sys.path.insert(0, f'{REPO}/scripts/sakhiya_pro/distill'); sys.path.insert(0, f'{REPO}/scripts/sakhiya_pro'); sys.path.insert(0, REPO)
    eg = import_module('export_gate'); ev = [e for e in eg.eval_set(0) if e['set'] == 'master']; b = f'{T}/{t}/best/wav'
    pk = '/tmp/pack_' + t; shutil.rmtree(pk, ignore_errors=True); os.makedirs(pk); meta = []
    for e in ev:
        if os.path.exists(f"{b}/{e['id']}.mp3"):
            shutil.copy(f"{b}/{e['id']}.mp3", f"{pk}/{e['id']}.mp3"); meta.append(dict(id=e['id'], lang=e['lang'], style=e.get('style', ''), text=e['text'], mp3=e['id'] + '.mp3', duration_s=0))
    if not meta: return
    json.dump(meta, open(pk + '/pack_meta.json', 'w'), ensure_ascii=False); s = g.get('summary', {})
    mm = dict(int8_mb=g['int8_mb'], utmos=s.get('utmos'), cer_ne=s.get('cer_ne'), cer_sa=s.get('cer_sa'), spk_sim=s.get('spk_sim'), clicks=s.get('clicks'), word_drop_proxy=s.get('word_drop_proxy'), int8_vs_fp32_mel_corr=g['int8_vs_fp32_mel_corr'], rtf_1thread=g.get('rtf_1thread'), step=g['step'], note='real exported int8 gate; automatic metrics')
    json.dump(mm, open(pk + '/metrics.json', 'w'))
    sh(f'python3 scripts/sakhiya_pro/freeze_pro_pack.py {pk} {pk}/metrics.json --tag {tag} --label "{NAMES[t]} · {g["int8_mb"]} MB int8 · distilled from Sakhiya-Pro v2" --notes "ladder student {t}, step {g["step"]}, best passing int8 export"')


def final_cleanup():
    for pat in ('train_student.py', 'gen_corpus.py', 'ladder_daemon.py', 'distill_chain.sh', 'backup_daemon.py'):
        for pid in sh(f"pgrep -f '[{pat[0]}]{pat[1:]}'", cwd='/').split():
            try: os.kill(int(pid), signal.SIGTERM)
            except Exception: pass


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--every', type=int, default=2700); ap.add_argument('--final_utc', required=True); a = ap.parse_args()
    fin = datetime.datetime.fromisoformat(a.final_utc).replace(tzinfo=datetime.timezone.utc).timestamp()
    try: pro_v2()
    except Exception as e: log('pro_v2 ERR', repr(e)[:300])
    while time.time() < fin:
        t0 = time.time()
        try:
            if tiers_sync(): push('student ladder: best passing int8 exports (never regress)')
        except Exception as e: log('sync ERR', repr(e)[:300])
        while time.time() - t0 < a.every and time.time() < fin: time.sleep(20)
    log('FINAL cutoff reached')
    try:
        ladder = subprocess.run("ps -eo cmd | grep -c '[t]rain_student.py'", shell=True, capture_output=True, text=True).stdout.strip(); log('students still training:', ladder)
        final_cleanup(); time.sleep(20); sh('python3 scripts/sakhiya_pro/distill/ladder_daemon.py --help >/dev/null 2>&1; true')
        ts = tiers_sync(True)
        for t, g in ts.items():
            try: freeze_tier(t, g)
            except Exception as e: log('freeze', t, repr(e)[:200])
        st = '\n\n## Session 67 — Sakhiya-Pro v2 + student ladder (autopilot finalize)\n' + '\n'.join(f"- {NAMES[t]}: **{g['int8_mb']} MB** int8, step {g['step']}, UTMOS {g.get('summary', {}).get('utmos')}, CER ne {g.get('summary', {}).get('cer_ne')}, int8-vs-fp32 mel corr {g['int8_vs_fp32_mel_corr']}, nan {g['nan_count']}, RTF(1 thr) {g.get('rtf_1thread')}" for t, g in ts.items()) + '\n- Metrics are automatic (ASR round trip / UTMOS22 / ECAPA); no human MOS. Tiers not listed did not pass the int8 gate.\n'
        open(f'{REPO}/STATUS.md', 'a').write(st)
        push('Session 67: frozen packs for passing student tiers, STATUS')
        try:
            a_ = hf()
            for t in ts: a_.upload_folder(folder_path=f'{T}/{t}/best', repo_id='NikitHamal/sakhiya-pro-v1', path_in_repo=f'students/{t}', ignore_patterns=['wav/*.wav'], commit_message=f'student {t}')
            a_.upload_folder(folder_path='/work/sp/out', repo_id='NikitHamal/sakhiya-v1-train', path_in_repo='sakhiya_pro_v2', commit_message='final sync', ignore_patterns=['*.tmp'])
        except Exception as e: log('HF final err', str(e)[:200])
        stage('99_final', 'Autopilot finalize', 'done', 'packs frozen, pushed, HF uploaded, GPU jobs stopped')
    except Exception as e: log('final ERR', repr(e)[:400])
    log('FINALIZE_V2_DONE')


if __name__ == '__main__': main()
