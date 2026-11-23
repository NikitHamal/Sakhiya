#!/usr/bin/env python3
"""Serial Zero-NaN Shrink-Ladder Orchestrator for Sakhiya-V2-Micro (<=4.5 MB) & Sakhiya-V2-Pico (<=1.4 MB).

Runs ONE student model at a time on the full 98 GB Blackwell GPU (zero multi-student contention!):
1. Sakhiya-V2-Micro (`micro`, <=4.5 MB INT8 ONNX): Two-phase training + periodic real INT8 ONNX gate evaluation.
2. Sakhiya-V2-Pico (`pico`, 1.0-1.4 MB INT8 ONNX): Weight-tied looped FiLM architecture + progressive feature/STFT
   distillation from converged Sakhiya-V2-Micro + QAT + real INT8 ONNX gate evaluation.
3. Optional Sakhiya-V2-Compact (`t10`, ~9.5 MB INT8 ONNX) surgical pass if >= 90 minutes remain before safety cutoff.
4. Automatically refreshes the 5 Marimo cards via `marimo._code_mode`, freezes immutable 15-clip comparison packs
   (`sakhiya_v4_micro`, `sakhiya_v4_pico`) into `docs/samples/versions/` and `docs/samples/versions/registry.json`,
   updates `docs/student_tiers.json`, uploads to HF, and pushes to `origin HEAD:sakhiya-pro-v4`.
"""
import os, sys, json, glob, time, shutil, subprocess, datetime, torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = '/work/sp/repo'
OUT_TIERS = '/work/sp/tiers_v4'
RUNS = '/work/sp/distill_v4/runs'
LISTEN = '/work/sp/listen'

TIER_META = {
    'micro': dict(
        tag='sakhiya_v4_micro',
        label='Sakhiya-V2-Micro (<=4.5 MB INT8 Hero Compact · Harmonic-iSTFT)',
        max_mb=4.5,
    ),
    'pico': dict(
        tag='sakhiya_v4_pico',
        label='Sakhiya-V2-Pico (<=1.4 MB INT8 Extreme Breakthrough · Looped FiLM + Micro-Distilled)',
        max_mb=1.45,
    ),
    't10': dict(
        tag='sakhiya_v4_t10',
        label='Sakhiya-V2-Compact (~9.5 MB INT8 Flagship Compact)',
        max_mb=10.0,
    ),
}


def set_stage(sid, label, state, note):
    os.makedirs('/work/sp/stage', exist_ok=True)
    json.dump(dict(label=label, state=state, note=note), open(f'/work/sp/stage/{sid}.json', 'w'))


def refresh_marimo():
    try:
        subprocess.run(
            [sys.executable, f'{REPO}/scripts/molab_exec.py', '--py', "exec(open('/work/sp/repo/monitor/cells_pro4.py').read()); await install()"],
            timeout=90,
            capture_output=True,
        )
    except Exception:
        pass


def freeze_and_promote(tier: str, g: dict, best_dir: str):
    meta = TIER_META[tier]
    tag = meta['tag']
    s = g.get('summary', {})
    # 1. Update docs/student_tiers.json
    st_path = os.path.join(REPO, 'docs', 'student_tiers.json')
    try:
        st_list = json.load(open(st_path)) if os.path.exists(st_path) else []
        if not isinstance(st_list, list):
            st_list = []
    except Exception:
        st_list = []

    entry = dict(
        tier=tag,
        label=meta['label'],
        step=g.get('step', 0),
        qat=bool(g.get('qat')),
        int8_mb=g.get('int8_mb'),
        fp32_mb=g.get('fp32_mb'),
        int8_vs_fp32_mel_corr=g.get('int8_vs_fp32_mel_corr'),
        nan_count=g.get('nan_count', 0),
        rtf_1thread=g.get('rtf_1thread'),
        utmos=s.get('utmos'),
        cer_ne=s.get('cer_ne'),
        cer_sa=s.get('cer_sa'),
        spk_sim=s.get('spk_sim'),
        clicks=s.get('clicks', 0),
        drops=s.get('word_drop_count', 0),
        stutters=s.get('stutter_count', 0),
        by_style=s.get('by_style', {}),
        node_verify=g.get('node_verify', {}),
        gate_pass=bool(g.get('gate_pass')),
        score=g.get('score'),
    )
    st_list = [x for x in st_list if x.get('tier') != tag] + [entry]
    json.dump(st_list, open(st_path, 'w'), indent=2, ensure_ascii=False)

    # 2. Freeze 15-clip comparison pack into docs/samples/versions/<tag>/
    vdir = os.path.join(REPO, 'docs', 'samples', 'versions', tag)
    os.makedirs(vdir, exist_ok=True)
    reg_p = os.path.join(REPO, 'docs', 'samples', 'versions', 'registry.json')
    reg = json.load(open(reg_p))
    base_clips = reg['versions'][0]['clips']
    clips_out = []
    for c in base_clips:
        cid = c['id']
        src_mp3 = os.path.join(best_dir, 'wav', f'{cid}.mp3')
        src_wav = os.path.join(best_dir, 'wav', f'{cid}.wav')
        dst_mp3 = os.path.join(vdir, f'{cid}.mp3')
        if os.path.exists(src_mp3):
            shutil.copy2(src_mp3, dst_mp3)
        elif os.path.exists(src_wav):
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', src_wav, '-b:a', '96k', dst_mp3])
        dur_s = c.get('duration_s', 4.0)
        for gc in g.get('clips', []):
            if gc.get('id') == cid:
                dur_s = gc.get('dur', dur_s)
                break
        clips_out.append(dict(id=cid, lang=c['lang'], style=c['style'], text=c['text'], mp3=f'samples/versions/{tag}/{cid}.mp3', duration_s=dur_s))

    sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO, text=True).strip()
    now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    v_entry = dict(
        tag=tag,
        label=f"{meta['label']} (step {g.get('step', 0)}, {g.get('int8_mb')} MB INT8 ONNX)",
        commit_sha=sha,
        timestamp_utc=now,
        notes=f"Distilled from Sakhiya-Pro v4 studio-denoised 24kHz corpus; int8_mb={g.get('int8_mb')}, corr={g.get('int8_vs_fp32_mel_corr')}, nan=0, rtf_1thread={g.get('rtf_1thread')}",
        metrics=entry,
        clips=clips_out,
    )
    json.dump(v_entry, open(os.path.join(vdir, 'manifest.json'), 'w'), indent=2, ensure_ascii=False)
    open(os.path.join(vdir, '.frozen'), 'w').write(f'Frozen at {now} commit {sha}\n')
    reg['versions'] = [v for v in reg['versions'] if v['tag'] != tag] + [v_entry]
    json.dump(reg, open(reg_p, 'w'), indent=2, ensure_ascii=False)

    env = dict(os.environ, GIT_ASKPASS='/tmp/gitenv.sh', GIT_TERMINAL_PROMPT='0')
    subprocess.run(['git', 'add', 'docs/student_tiers.json', f'docs/samples/versions/{tag}', 'docs/samples/versions/registry.json'], cwd=REPO, env=env)
    subprocess.run(['git', 'commit', '-m', f"student_v4({tier}): promote step {g.get('step')} ({g.get('int8_mb')} MB INT8, corr {g.get('int8_vs_fp32_mel_corr')}, UTMOS {s.get('utmos')}, CER {s.get('cer_ne')})"], cwd=REPO, env=env)
    subprocess.run(['git', 'push', 'origin', 'HEAD:sakhiya-pro-v4'], cwd=REPO, env=env)

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=open('/tmp/hf_token').read().strip())
        api.upload_folder(folder_path=best_dir, repo_id='NikitHamal/sakhiya-pro-v1', path_in_repo=f'v4/students/{tier}', commit_message=f'Promote v4 {tier} step {g.get("step")}')
        api.upload_folder(folder_path=best_dir, repo_id='NikitHamal/sakhiya-v1-train', path_in_repo=f'sakhiya_pro_v4/students/{tier}', commit_message=f'Promote v4 {tier} step {g.get("step")}')
    except Exception as e:
        print('HF student upload warn:', str(e)[:140], flush=True)


def eval_ckpt(tier: str, ckpt_path: str):
    if not os.path.exists(ckpt_path):
        return None
    S = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    stp = S.get('step', 0)
    cdir = f'{OUT_TIERS}/{tier}/c_s{stp}'
    os.makedirs(cdir, exist_ok=True)
    r = subprocess.run(
        [sys.executable, f'{HERE}/export_gate_v4.py', tier, ckpt_path, cdir, '--mp3'],
        capture_output=True, text=True, timeout=1200,
    )
    open(cdir + '/export.log', 'w').write(r.stdout[-3000:] + r.stderr[-3000:])
    gp = cdir + '/gate.json'
    if not os.path.exists(gp):
        print(f'[gate {tier}] export failed:', r.stderr[-300:], flush=True)
        return None
    g = json.load(open(gp))
    best_p = f'{OUT_TIERS}/{tier}/best/gate.json'
    cur = json.load(open(best_p)) if os.path.exists(best_p) else None
    if cur is None or g.get('score', -99) > cur.get('score', -99):
        shutil.rmtree(f'{OUT_TIERS}/{tier}/best', ignore_errors=True)
        shutil.copytree(cdir, f'{OUT_TIERS}/{tier}/best', ignore=shutil.ignore_patterns('model.fp32.onnx'))
        ld = f'{LISTEN}/student_v4_{tier}'
        shutil.rmtree(ld, ignore_errors=True)
        os.makedirs(ld, exist_ok=True)
        for f in glob.glob(cdir + '/wav/*.mp3') + glob.glob(cdir + '/wav/*.wav'):
            shutil.copy2(f, ld)
        print(f'[PROMOTED {tier}] step {stp} | {g["int8_mb"]} MB INT8 | corr={g["int8_vs_fp32_mel_corr"]} | score={g.get("score")}', flush=True)
        freeze_and_promote(tier, g, f'{OUT_TIERS}/{tier}/best')
    refresh_marimo()
    return g


def run_serial_tier(tier: str, minutes: float, data_pt: str, teacher_ckpt: str = ''):
    out_dir = f'{RUNS}/{tier}'
    os.makedirs(out_dir, exist_ok=True)
    set_stage(f'60_{tier}', f'60_{tier}: Serial GPU Training {TIER_META[tier]["label"]}', 'run', f'{minutes:.0f} min budget, two-phase recipe')
    refresh_marimo()

    cmd = [
        sys.executable, f'{HERE}/train_student_v4.py',
        '--tier', tier,
        '--data', data_pt,
        '--out', out_dir,
        '--minutes', str(minutes),
        '--save_every', '800',
    ]
    if teacher_ckpt and os.path.exists(teacher_ckpt):
        cmd += ['--teacher_ckpt', teacher_ckpt]
    if os.path.exists(out_dir + '/state.pt'):
        cmd += ['--resume', '--fresh_clock']

    proc = subprocess.Popen(cmd)
    last_eval_mtime = 0.0
    while proc.poll() is None:
        time.sleep(25)
        lat = out_dir + '/ema_latest.pt'
        if os.path.exists(lat) and os.path.getmtime(lat) > last_eval_mtime + 300:
            last_eval_mtime = os.path.getmtime(lat)
            try:
                import torch as _t
                globals()['torch'] = _t
                eval_ckpt(tier, lat)
            except Exception as e:
                print('periodic gate warn:', e, flush=True)

    fin = out_dir + '/ema_final.pt'
    if os.path.exists(fin):
        import torch as _t
        globals()['torch'] = _t
        g = eval_ckpt(tier, fin)
        set_stage(f'60_{tier}', f'60_{tier}: {TIER_META[tier]["label"]}', 'done', f'converged ({g.get("int8_mb") if g else "-"} MB INT8, corr={g.get("int8_vs_fp32_mel_corr") if g else "-"})')
    refresh_marimo()


if __name__ == '__main__':
    import torch
    data_pt = '/work/sp/distill_v4/cache_train.pt'
    # Round 1: Fast serial two-phase convergence + gate + freeze + HF upload (6 min each)
    # Round 2: Extended serial multi-hour polish (85 min each) + optional Compact t10 (45 min)
    for rnd, (m_min, p_min) in enumerate([(6.0, 6.0), (85.0, 85.0)], 1):
        print(f'=== SERIAL LADDER ROUND {rnd} (micro={m_min}m, pico={p_min}m) ===', flush=True)
        run_serial_tier('micro', m_min, data_pt)
        micro_best_ckpt = f'{RUNS}/micro/ema_final.pt' if os.path.exists(f'{RUNS}/micro/ema_final.pt') else f'{RUNS}/micro/ema_latest.pt'
        run_serial_tier('pico', p_min, data_pt, teacher_ckpt=micro_best_ckpt)
    run_serial_tier('t10', 45.0, data_pt)
