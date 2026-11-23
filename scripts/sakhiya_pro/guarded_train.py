#!/usr/bin/env python3
"""Guarded OmniVoice full-backbone fine-tune for Sakhiya-Pro.

Guardrails (per project rules):
  * hard wall-clock cap (default 150 min) -> kills the whole process group
  * mandatory step-50 throughput check: projected total time must fit the budget, else abort (exit 3)
  * checkpoints every save_steps (evaluate those on synthesized audio with eval_ckpt.py, not loss alone)

usage: guarded_train.py --name ft1 --steps 1600 [--lr 5e-6] [--batch_tokens 6000] [--init k2-fsa/OmniVoice]
"""
import argparse, json, os, re, signal, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument('--name', default='ft1')
ap.add_argument('--steps', type=int, default=1600)
ap.add_argument('--lr', type=float, default=5e-6)
ap.add_argument('--batch_tokens', type=int, default=6000)
ap.add_argument('--save_steps', type=int, default=200)
ap.add_argument('--budget_min', type=float, default=150)
ap.add_argument('--init', default='/work/sp/models/OmniVoice')
ap.add_argument('--tokens', default='/work/sp/ft/tokens')
ap.add_argument('--lora', action='store_true')
ap.add_argument('--prompt_ratio', type=float, default=None, help='fixed prompt ratio (v3: 0.10); default = OmniVoice (0.0,0.3)')
ap.add_argument('--min_lr_ratio', type=float, default=0.0, help='cosine decays to lr*ratio (v3: 2e-6/1.4e-5)')
ap.add_argument('--warmup_ratio', type=float, default=0.05)
a = ap.parse_args()

SRC = '/work/sp/OmniVoice_src'
out = f'/work/sp/exp/{a.name}'; os.makedirs(out, exist_ok=True)
cfg = json.load(open(f'{SRC}/examples/config/train_config_finetune_sdpa.json'))
cfg.update(init_from_checkpoint=a.init, learning_rate=a.lr, steps=a.steps, batch_tokens=a.batch_tokens,
           logging_steps=10, eval_steps=a.save_steps, save_steps=a.save_steps, keep_last_n_checkpoints=-1,
           warmup_ratio=a.warmup_ratio, num_workers=4)
if a.prompt_ratio is not None:
    cfg.update(prompt_ratio_range=[a.prompt_ratio, a.prompt_ratio], instruct_ratio=0.0, only_instruct_ratio=0.0)
if a.lora:
    cfg.update(use_lora=True, learning_rate=1e-4)
json.dump(cfg, open(f'{out}/train_cfg.json', 'w'), indent=1)
dcfg = dict(train=[dict(manifest_path=[f'{a.tokens}/train/data.lst'])], dev=[dict(manifest_path=[f'{a.tokens}/dev/data.lst'])])
json.dump(dcfg, open(f'{out}/data_cfg.json', 'w'), indent=1)

env = dict(os.environ, PYTHONPATH=SRC + ':' + os.environ.get('PYTHONPATH', ''), HF_TOKEN=open('/tmp/hf_token').read().strip(), MIN_LR_RATIO=str(a.min_lr_ratio))
cmd = ['accelerate', 'launch', '--num_processes', '1', '--mixed_precision', 'bf16', '-m', 'omnivoice.cli.train',
       '--train_config', f'{out}/train_cfg.json', '--data_config', f'{out}/data_cfg.json', '--output_dir', out]
logf = open(f'/work/tts/logs/train_{a.name}.log', 'w')
t0 = time.time(); budget = a.budget_min * 60
p = subprocess.Popen(cmd, cwd=SRC, env=env, stdout=logf, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
print('launched pid', p.pid, flush=True)
def kill(why, code):
    print('ABORT:', why, flush=True)
    try: os.killpg(p.pid, signal.SIGTERM)
    except Exception: pass
    time.sleep(5)
    try: os.killpg(p.pid, signal.SIGKILL)
    except Exception: pass
    sys.exit(code)
t_first = None; checked = False
while p.poll() is None:
    time.sleep(5)
    if time.time() - t0 > budget: kill(f'hard cap {a.budget_min} min', 4)
    lp = f'/work/tts/logs/train_{a.name}.log'  # 'Step N |' lines go to stdout log (not out/train.log)
    txt = open(lp, errors='ignore').read() if os.path.exists(lp) else ''
    steps = [(int(m.group(1))) for m in re.finditer(r'Step (\d+) \|', txt)]
    if steps and t_first is None:
        t_first = (time.time(), steps[0])
    if steps and not checked and max(steps) >= 50:
        checked = True
        el = time.time() - t_first[0]; ds = max(steps) - t_first[1]
        sps = ds / max(el, 1e-6); eta = (a.steps - max(steps)) / max(sps, 1e-6) + (time.time() - t0)
        print(f'STEP50 CHECK: {sps:.2f} steps/s, projected total {eta/60:.1f} min (budget {a.budget_min})', flush=True)
        if eta > budget: kill('projected ETA exceeds budget at step 50', 3)
print('train exited', p.returncode, 'after', round((time.time() - t0) / 60, 1), 'min', flush=True)
sys.exit(p.returncode)
