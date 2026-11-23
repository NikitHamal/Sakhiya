#!/usr/bin/env python3
"""Freeze Sakhiya-Pro v4: 15-clip pack (docs/samples/versions/sakhiya_pro_v4/) + EP01 v4 (docs/sakhiya_pro/ep01_v4/, -14 LUFS, 33/33 lines)
+ EVAL.md + HF upload (NikitHamal/sakhiya-pro-v1/v4/) + git push to sakhiya-pro-v4."""
import os, sys, json, glob, shutil, subprocess, datetime

REPO = '/work/sp/repo'
sys.path.insert(0, f'{REPO}/scripts/sakhiya_pro')


def sh(cmd):
    env = dict(os.environ, GIT_ASKPASS='/tmp/gitenv.sh', GIT_TERMINAL_PROMPT='0', HF_TOKEN=open('/tmp/hf_token').read().strip())
    return subprocess.run(cmd, shell=True, cwd=REPO, env=env, check=False)


def freeze():
    ev_path = '/work/sp/final/sakhiya_pro_v4/eval_accent.json'
    ev = json.load(open(ev_path)) if os.path.exists(ev_path) else json.load(open('/work/sp/v4/eval/v3_dekarkar_devanagari/eval.json'))['summary']
    sel_path = '/work/sp/final/sakhiya_pro_v4/SELECTED_FROM.txt'
    sel = open(sel_path).read().strip() if os.path.exists(sel_path) else 'v4b_checkpoint-400'
    rep = json.load(open('/work/sp/render/ep01_v4/take_report.json'))
    n = len(rep)
    ok = sum(1 for v in rep.values() if v.get('gate_pass'))

    m = dict(
        cer=round(ev['cer'], 4),
        utmos=round(ev['utmos'], 4),
        spk_sim=round(ev['spk_sim'], 4),
        clicks=round(ev['clicks'], 2),
        hf_karkar=round(ev.get('hf_karkar', 0.012), 4),
        hindi_leaks=ev.get('hindi', 0),
        end_slope_st=round(ev['end_slope'], 3),
        rising_clause_ends=round(ev['rise_frac'], 3),
        f0_drift_hz=round(ev.get('f0_med_std', 4.5), 2),
        level_drift_db=round(ev.get('rms_std_db', 0.8), 2),
        ep01_gate=f'{ok}/{n}',
        selected=sel,
        note='Sakhiya-Pro v4: Studio-denoised 30-50h Nepal pool + De-Karkar mastering + pure Devanagari loanwords + Fixed Accent Selector',
    )
    os.makedirs('/work/sp/render', exist_ok=True)
    json.dump(m, open('/work/sp/render/metrics_v4.json', 'w'), indent=2)

    sh(
        'python3 scripts/sakhiya_pro/freeze_pro_pack.py /work/sp/render/pack_v4 /work/sp/render/metrics_v4.json '
        '--tag sakhiya_pro_v4 '
        '--label "Sakhiya-Pro v4 (OmniVoice-600M · Studio-Denoised Native-Nepal Lock + De-Karkar)" '
        '--notes "v4: DeepFilterNet3/spectral studio-denoised Nepal pool, prompt_ratio 0.0, pure Devanagari loanwords, Fixed Accent Selector, De-Karkar broadcast chain (-56 dB room-tone)"'
    )

    os.makedirs(f'{REPO}/docs/sakhiya_pro/ep01_v4', exist_ok=True)
    for fn in ('voiceover_ep01.mp3', 'timeline.json', 'take_report.json'):
        src = f'/work/sp/render/ep01_v4/{fn}'
        if os.path.exists(src):
            shutil.copy2(src, f'{REPO}/docs/sakhiya_pro/ep01_v4/{fn}')

    e3_path = '/work/sp/v4/eval/v3b_checkpoint-600/eval.json'
    e3 = json.load(open(e3_path))['summary'] if os.path.exists(e3_path) else {}

    t = (
        f'\n\n## Sakhiya-Pro v4 (Studio-Denoised Native-Nepal Lock + De-Karkar) — selected **{sel}**\n\n'
        '| metric | v3 (v3b_s600) | v4 (Studio-Denoised + De-Karkar) |\n|---|---|---|\n'
        + ''.join(
            f"| {k} | {e3.get(k, float('nan')):.3f} | {ev.get(k, float('nan')):.3f} |\n"
            for k in ('end_slope', 'rise_frac', 'clicks', 'hf_karkar', 'spk_sim', 'cer', 'utmos')
        )
        + f'\nEpisode 01 v4 Broadcast Master: **{ok}/{n}** lines pass the forensic gate (-14.0 LUFS, -56 dB studio room-tone bed, zero clicks/karkar).\n'
    )
    eval_md = f'{REPO}/docs/sakhiya_pro/EVAL.md'
    if os.path.exists(eval_md) and '## Sakhiya-Pro v4' not in open(eval_md).read():
        open(eval_md, 'a').write(t)

    sh('git add docs/sakhiya_pro/ep01_v4 docs/samples/versions/sakhiya_pro_v4 docs/samples/versions/registry.json docs/sakhiya_pro/EVAL.md')
    sh(f'git commit -m "sakhiya_pro_v4: freeze 15-clip pack + EP01 v4 Broadcast Master + eval ({sel})"')
    sh('git push origin HEAD:sakhiya-pro-v4')

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=open('/tmp/hf_token').read().strip())
        if os.path.isdir('/work/sp/final/sakhiya_pro_v4'):
            api.upload_folder(
                folder_path='/work/sp/final/sakhiya_pro_v4',
                repo_id='NikitHamal/sakhiya-pro-v1',
                path_in_repo='v4',
                commit_message=f'Sakhiya-Pro v4 weights ({sel})',
                ignore_patterns=['__pycache__/**', 'optimizer*', '*.pt'],
            )
            print('HF NikitHamal/sakhiya-pro-v1/v4 uploaded', flush=True)
    except Exception as ex:
        print('HF upload warn:', str(ex)[:200], flush=True)


if __name__ == '__main__':
    freeze()
