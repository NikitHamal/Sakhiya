#!/usr/bin/env python3
"""Freeze Sakhiya-Pro v3: 15-clip pack + EP01 v3 (-14 LUFS, 33 lines) + EVAL.md + HF upload + git push.  usage: finalize_v3.py freeze"""
import os, sys, json, glob, shutil
sys.path.insert(0, '/work/sp/repo/scripts/sakhiya_pro')
from finalize_v2 import sh, push, log, hf, REPO

def freeze():
    sh('git pull -q --rebase origin main')
    ev = json.load(open('/work/sp/final/sakhiya_pro_v3/eval_accent.json')); sel = open('/work/sp/final/sakhiya_pro_v3/SELECTED_FROM.txt').read().strip()
    e2 = json.load(open('/work/sp/v3/eval/v2/eval.json'))['summary'] if os.path.exists('/work/sp/v3/eval/v2/eval.json') else {}
    rep = json.load(open('/work/sp/render/ep01_v3/take_report.json')); n = len(rep); ok = sum(1 for v in rep.values() if v['gate_pass'])
    six = {k: rep[k] for k in ('c3', 'n1a', 'n2a', 'n4a', 'f1', 'o1') if k in rep}
    m = dict(cer=ev['cer'], utmos=ev['utmos'], spk_sim=ev['spk_sim'], clicks=ev['clicks'], hindi_leaks=ev['hindi'], end_slope_st=ev['end_slope'], rising_clause_ends=ev['rise_frac'],
             f0_drift_hz=ev['f0_med_std'], level_drift_db=ev['rms_std_db'], ep01_gate=f'{ok}/{n}', selected=sel, note='automatic metrics; no human MOS; v2 baseline under the same config: ' + json.dumps({k: round(v, 3) for k, v in e2.items() if k in ('end_slope', 'rise_frac', 'spk_sim', 'cer', 'utmos')}))
    json.dump(m, open('/work/sp/render/metrics_v3.json', 'w'))
    sh('python3 scripts/sakhiya_pro/freeze_pro_pack.py /work/sp/render/pack_v3 /work/sp/render/metrics_v3.json --tag sakhiya_pro_v3 --label "Sakhiya-Pro v3 (OmniVoice-600M · native-Nepal weight lock, teacher)" '
       '--notes "v3: language tag npi, instruct None, stitched 6-14 s native utterances, prompt-ratio 0.10, cosine 1.4e-5 -> 2e-6, single-anchor lock; nativized loanwords; accent-aware best-of-N"')
    os.makedirs(f'{REPO}/docs/sakhiya_pro/ep01_v3', exist_ok=True)
    sh('cp /work/sp/render/ep01_v3/voiceover_ep01.mp3 /work/sp/render/ep01_v3/timeline.json /work/sp/render/ep01_v3/take_report.json docs/sakhiya_pro/ep01_v3/')
    rows = ''.join(f"| {k} | {v['gate_pass']} | {max(v['cands'], key=lambda c: c['sel'])['cer']:.3f} | {max(v['cands'], key=lambda c: c['sel'])['clicks']} |\n" for k, v in six.items())
    t = ('\n\n## Sakhiya-Pro v3 (native-Nepal accent lock) — selected **%s**\n\n| metric | v2 | v3 |\n|---|---|---|\n' % sel
         + ''.join(f"| {k} | {e2.get(k, float('nan')):.3f} | {ev[k]:.3f} |\n" for k in ('end_slope', 'rise_frac', 'spk_sim', 'cer', 'utmos', 'clicks', 'f0_med_std', 'rms_std_db'))
         + f'\nEpisode 01: **{ok}/{n}** lines pass the forensic gate. Previously failing lines:\n\n| line | gate | CER | clicks |\n|---|---|---|---|\n' + rows
         + '\nend_slope < 0 = clause endings close downward (Nepal-native); > 0 = floating English-style rise. All numbers automatic (F0 contour, Kriti ASR, UTMOS22, ECAPA); no human MOS.\n')
    open(f'{REPO}/docs/sakhiya_pro/EVAL.md', 'a').write(t)
    push(f'sakhiya_pro_v3: freeze 15-clip pack + EP01 v3 + accent eval ({sel})')
    try:
        hf().upload_folder(folder_path='/work/sp/final/sakhiya_pro_v3', repo_id='NikitHamal/sakhiya-pro-v1', path_in_repo='v3', commit_message='Sakhiya-Pro v3 weights', ignore_patterns=['__pycache__/**', 'optimizer*', '*.pt']); log('HF pro v3 uploaded')
    except Exception as ex: log('HF upload err', str(ex)[:200])

if __name__ == '__main__':
    freeze()
