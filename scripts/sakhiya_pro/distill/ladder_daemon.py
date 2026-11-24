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
    dict(name='Sakhiya-Pro v3 (v3b_s600 Native-Nepal Teacher)', size_mb=2400.0, utmos=3.918, cer=0.032, sa_cer=0.068, drops=0, stutters=0, clicks=0, spk=0.852, rtf='GPU', status='shipped Pro v3'),
    dict(name='Sakhiya-V1 (Live Site)', size_mb=15.98, utmos=3.957, cer=0.014, sa_cer=0.025, drops=0, stutters=0, clicks=0, spk=0.987, rtf=0.1884, status='shipped'),
    dict(name='Sakhiya-V1-Nano (Live Site)', size_mb=8.93, utmos=3.253, cer=0.027, sa_cer=0.046, drops=0, stutters=0, clicks=0, spk=0.960, rtf=0.0217, status='shipped'),
    dict(name='Sakhiya-V1-Atom (Live Site)', size_mb=6.51, utmos=3.262, cer=0.017, sa_cer=0.068, drops=0, stutters=0, clicks=0, spk=0.955, rtf=0.0189, status='shipped'),
    dict(name='Sakhiya-V1-Atom-Lite (Live Site)', size_mb=2.72, utmos=2.944, cer=0.017, sa_cer=0.040, drops=0, stutters=0, clicks=0, spk=0.940, rtf=0.0155, status='shipped'),
]


# publish (git/HF) only when the INT8 student's UTMOS reaches the shipped baseline of its size class
BASELINE_UTMOS = dict(t30=3.957, t15=3.957, t10=3.253, t5=3.262, t1=2.944)

TIER_LABELS = {
    't30': 'Sakhiya-T30 (29.6 MB INT8 Distilled Student)',
    't15': 'Sakhiya-T15 (14.5 MB INT8 Flagship Compact Student)',
    't10': 'Sakhiya-T10 (9.5 MB INT8 Compact Student)',
    't5':  'Sakhiya-Micro t5 (4.8 MB INT8 Hero Sub-5 MB Student)',
    't1':  'Sakhiya-Pico t1 (1.05 MB INT8 Breakthrough Looped Student)',
}


def refresh_marimo_cards():
    try:
        if not os.path.exists('/tmp/molab_url'):
            open('/tmp/molab_url', 'w').write('http://127.0.0.1:8080')
        if not os.path.exists('/tmp/molab_token'):
            import re as _re
            cmd1 = subprocess.check_output(['ps', '-p', '1', '-o', 'args='], text=True)
            m_tok = _re.search(r'--token-password\s+(\S+)', cmd1)
            if m_tok:
                open('/tmp/molab_token', 'w').write(m_tok.group(1))
        sys.path.insert(0, os.path.join(REPO, 'scripts'))
        import molab_exec as M
        # pro5 cards self-refresh from the files this daemon writes; reinstalling would reset the listening-desk selection
        M.run_py("pass", 30)
    except Exception as e:
        print('marimo refresh skip:', str(e)[:120], flush=True)


def promote_repo_and_hf(tier, g, best_dir):
    import datetime
    s = g.get('summary', {})
    # 1. Update docs/student_tiers.json
    st_path = os.path.join(REPO, 'docs', 'student_tiers.json')
    try:
        st_list = json.load(open(st_path)) if os.path.exists(st_path) else []
        if not isinstance(st_list, list): st_list = []
    except Exception:
        st_list = []
    entry_t = dict(
        tier=tier, label=TIER_LABELS.get(tier, f'Sakhiya-{tier}'),
        step=g.get('step', 0), qat=bool(g.get('qat')),
        int8_mb=g.get('int8_mb'), fp32_mb=g.get('fp32_mb'),
        int8_vs_fp32_mel_corr=g.get('int8_vs_fp32_mel_corr'),
        rtf_1thread=g.get('rtf_1thread'),
        utmos=s.get('utmos'), cer_ne=s.get('cer_ne'), cer_sa=s.get('cer_sa'),
        spk_sim=s.get('spk_sim'), clicks=s.get('clicks', 0),
        drops=s.get('word_drop_count', s.get('word_drop_proxy', 0)),
        stutters=s.get('stutter_count', 0),
        by_style=s.get('by_style', {}),
        node_verify=g.get('node_verify', {}),
        gate_pass=bool(g.get('gate_pass')),
        score=g.get('score')
    )
    alias_map = {'t5': ('sakhiya_v4_micro', 'micro', 'models_micro'), 't1': ('sakhiya_v4_pico', 'pico', 'models_pico')}
    st_list = [x for x in st_list if x.get('tier') != tier] + [entry_t]
    if tier in alias_map:
        v4_tag, _, _ = alias_map[tier]
        entry_v4 = dict(entry_t, tier=v4_tag, label=TIER_LABELS.get(tier, v4_tag))
        st_list = [x for x in st_list if x.get('tier') != v4_tag] + [entry_v4]
    json.dump(st_list, open(st_path, 'w'), indent=2, ensure_ascii=False)

    # 2. Freeze 15-clip master pack into docs/samples/versions/sakhiya_v3_<tier>/ (and sakhiya_v4_micro/pico + live models_micro/pico)
    tag = f'sakhiya_v3_{tier}'
    tags_to_write = [tag]
    extra_git_paths = []
    if tier in alias_map:
        v4_tag, tier_id, mdir_name = alias_map[tier]
        tags_to_write.append(v4_tag)
        src_i8 = os.path.join(best_dir, 'model.int8.onnx')
        dst_mdir = os.path.join(REPO, 'docs', mdir_name)
        if os.path.exists(src_i8) and os.path.isdir(dst_mdir):
            shutil.copy(src_i8, os.path.join(dst_mdir, 'model.int8.onnx'))
            extra_git_paths.append(f'docs/{mdir_name}/model.int8.onnx')
        tiers_p = os.path.join(REPO, 'docs', 'tiers.json')
        if os.path.exists(tiers_p):
            try:
                td = json.load(open(tiers_p))
                for tr_row in td.get('tiers', []):
                    if tr_row.get('id') == tier_id:
                        tr_row['mb'] = g.get('int8_mb')
                        tr_row['onnx_mb'] = g.get('int8_mb')
                        tr_row['rtf'] = g.get('rtf_1thread')
                        tr_row['rtf_onnx'] = g.get('rtf_1thread')
                        tr_row['utmos'] = s.get('utmos')
                        tr_row['ne_pcer'] = s.get('cer_ne')
                        tr_row['sa_pcer'] = s.get('cer_sa')
                json.dump(td, open(tiers_p, 'w'), indent=1, ensure_ascii=False)
                extra_git_paths.append('docs/tiers.json')
            except Exception:
                pass

    reg_p = os.path.join(REPO, 'docs', 'samples', 'versions', 'registry.json')
    reg = json.load(open(reg_p))
    base_clips = reg['versions'][0]['clips']
    sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO, text=True).strip()
    now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    for cur_tag in tags_to_write:
        vdir = os.path.join(REPO, 'docs', 'samples', 'versions', cur_tag)
        os.makedirs(vdir, exist_ok=True)
        clips_out = []
        for c in base_clips:
            cid = c['id']
            src_mp3 = os.path.join(best_dir, 'wav', f'{cid}.mp3')
            src_wav = os.path.join(best_dir, 'wav', f'{cid}.wav')
            dst_mp3 = os.path.join(vdir, f'{cid}.mp3')
            if os.path.exists(src_mp3):
                shutil.copy(src_mp3, dst_mp3)
            elif os.path.exists(src_wav):
                subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', src_wav, '-b:a', '96k', dst_mp3])
            dur_s = c.get('duration_s', 4.0)
            for gc in g.get('clips', []):
                if gc.get('id') == cid:
                    dur_s = gc.get('dur', dur_s)
                    break
            clips_out.append(dict(id=cid, lang=c['lang'], style=c['style'], text=c['text'], mp3=f'samples/versions/{cur_tag}/{cid}.mp3', duration_s=dur_s))
        v_entry = dict(
            tag=cur_tag,
            label=f"{TIER_LABELS.get(tier, cur_tag)} (step {g.get('step',0)}, {g.get('int8_mb')} MB INT8 ONNX)",
            commit_sha=sha, timestamp_utc=now,
            notes=f"Distilled from Sakhiya-Pro v3 + Phase-Locked Harmonic-iSTFT; int8_mb={g.get('int8_mb')}, corr={g.get('int8_vs_fp32_mel_corr')}, rtf_1thread={g.get('rtf_1thread')}",
            metrics=entry_t, clips=clips_out
        )
        json.dump(v_entry, open(os.path.join(vdir, 'manifest.json'), 'w'), indent=2, ensure_ascii=False)
        open(os.path.join(vdir, '.frozen'), 'w').write(f'Frozen at {now} commit {sha}\n')
        reg['versions'] = [v for v in reg['versions'] if v['tag'] != cur_tag] + [v_entry]
        extra_git_paths.append(f'docs/samples/versions/{cur_tag}')
    json.dump(reg, open(reg_p, 'w'), indent=2, ensure_ascii=False)

    # 3. Git pull --rebase & push
    env = dict(os.environ, GIT_ASKPASS='/tmp/gitenv.sh', GIT_TERMINAL_PROMPT='0')
    subprocess.run(['git', 'add', 'docs/student_tiers.json', 'docs/samples/versions/registry.json'] + extra_git_paths, cwd=REPO, env=env)
    subprocess.run(['git', 'commit', '-m', f"student({tier}): promote step {g.get('step')} ({g.get('int8_mb')} MB INT8, UTMOS {s.get('utmos')}, CER {s.get('cer_ne')})"], cwd=REPO, env=env)
    subprocess.run(['git', 'pull', '--rebase', 'origin', 'main'], cwd=REPO, env=env)
    subprocess.run(['git', 'push', 'origin', 'HEAD:main'], cwd=REPO, env=env)

    # 4. Upload to Hugging Face
    try:
        from huggingface_hub import HfApi
        tok = open('/tmp/hf_token').read().strip()
        api = HfApi(token=tok)
        api.upload_folder(folder_path=best_dir, repo_id='NikitHamal/sakhiya-pro-v1', path_in_repo=f'students/{tier}', commit_message=f'Promote {tier} step {g.get("step")}')
        api.upload_folder(folder_path=best_dir, repo_id='NikitHamal/sakhiya-v1-train', path_in_repo=f'sakhiya_pro_v3/students/{tier}', commit_message=f'Promote {tier} step {g.get("step")}')
    except Exception as he:
        print('HF student upload warn:', str(he)[:160], flush=True)


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
        _u = (g.get('summary') or {}).get('utmos'); _b = BASELINE_UTMOS.get(tier, 0.0)
        if _u is None or float(_u) < _b:
            print(f'[{time.strftime("%T")}] {tier} gate PASS but UTMOS {_u} < baseline {_b}: kept for the listening desk only, NOT published', flush=True)
        else:
            try:
                promote_repo_and_hf(tier, g, f'{OUT}/{tier}/best')
            except Exception as ex:
                print(f'[{time.strftime("%T")}] {tier} promote hook err: {ex}', flush=True)
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
            if not a.once:
                refresh_marimo_cards()
        leaderboard()
        if a.once:
            break
        time.sleep(a.every)
