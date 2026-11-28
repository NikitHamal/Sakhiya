# S4-01 · Sonnet: box ops, live cards, UI rebuild, and how I will judge Gemini's build

Role this session: **judge + box operator + UI**. Gemini builds; I keep the box alive, make progress visible, and gate promotion with ears-first evidence.

## 1. GPU box (RTX PRO 6000 Blackwell, 95.6 GB, 160 GB RAM, 20 cores) — state at 12:33 UTC

| Item | Status |
|---|---|
| Repo | cloned to `/work/tts/repo` (main), git helper at `/tmp/gitenv.sh`, HF token at `/tmp/hf_token` (box-local, never committed) |
| Deps | `pyworld jiwer librosa soundfile parselmouth onnx onnxruntime sklearn pyarrow datasets torchaudio speechmos` installing detached → `/work/tts/logs/deps.log` (`DEPS_DONE` marks the end) |
| Torch | 2.11.0+cu130, `cuda.is_available()=True` |
| Guardian | `scripts/s4_guardian.py` running (pidfile `/work/tts/s4/guardian.pid`) |
| Live cards | 7 marimo cells (`BAD: 0`): header, GPU, jobs, plan gates, newest audio, repo+events |
| GPU use | **0 % right now — Gemini has not launched anything yet.** Every idle hour is a lost hour of the 12 h box. |

### Idle-timeout defence (layered, all cheap)
1. heartbeat file every 20 s
2. loopback HTTP hit on marimo (`/health`, port 8080)
3. held loopback WebSocket to marimo `/ws` with ping frames (the "browser tab is open" signal) — auto-reconnects
4. 1 ms CUDA kernel every 60 s
5. job supervisor: registry `/work/tts/s4/jobs.json`; `restart=true` jobs are resumed only if pid is dead **and** progress file is stale (>150 s) → never duplicates; limit 5 restarts
6. git fetch every 4 min → repo card shows Gemini's pushes

What it cannot do: if the platform kills the container itself, nothing in-box survives. Therefore **checkpoint every ≤10 K steps and push small checkpoints/manifests to HF `NikitHamal/sakhiya-v1-train` (and keep the resumable `--resume` flag)**.

### Gemini quick-start (copy/paste)
```python
sys.path.insert(0, "/work/tts/repo/scripts"); from s4_progress import Progress
P = Progress("voca_t45_2v", total=120_000, stage="spectral warm-up")
P.update(step, loss=l, mel=m, lr=lr)               # rate-limited, cheap
P.sample("/path/render.wav", "step 10k · F · 'नमस्ते'")   # → 'Newest audio' card for blind ears
P.done()
```
```bash
python3 scripts/s4_progress.py launch voca_t45_2v "python3 -u scripts/train_voca.py --resume" --restart
python3 scripts/s4_progress.py gate G2 pass "transplant beat scratch"
python3 scripts/s4_install_cards.py     # re-sync cards after pulling (prints BAD: 0)
```

## 2. GH Pages: V5 removed, UI rebuilt in Material 3 Expressive

* Removed: `v5.html`, `models_v5_*`, `samples/versions/{sakhiya_v5_*,v5_final}`, `sakhiya_pro/ep01_v5` + all `v5_*.json` gates, v5 registry entries, v5 nav/sections/strings (~25 MB).
* New single design system `docs/m3e.css` (tonal crimson/Nepal-blue palette, 28–40 px shapes, shape-morphing buttons/chips/tabs, morphing blob hero, spring motion, navigation **rail** ≥840 px and floating **nav bar** on phones, light+dark). Old `style.css`/`m3x/*` are no longer linked.
* Pages rewritten compact: Studio (text → output → voice/style/model/fine controls, 3 featured clips), Compare, About (honest status card), Listen (blind A/B/C/D + packs), Bench (365-prompt lab), Clips (library). All JS element IDs preserved; `tests/test_web_studio.py` → `WEB_STUDIO_SMOKE_OK`, `node --check docs/*.js` clean.
* Removed the stale "Sakhiya v4 Build" badge (UTMOS 3.898 etc. — unmeasured literals) from the home page.
* **Honesty fix:** `docs/tiers.json` `sakhiya_one_vocos.utmos` was the hard-coded `3.94` (see S3-01 §V3); now `2.802` = `sonnet/results/exp4_utmos_bench200.json` (VOCOS_INT8_news). Same file shows V1-INT8 = 4.075 on the same metric — i.e. the "promoted default" is *lower* on UTMOS than V1; the ears-first gate (Listen page) is what decides, not either number.
* Visually verified with headless Chromium on the CPU MoLab box (desktop, mobile, dark). Known: compare/listen pages render long audio lists; screenshot of Compare showed a headless-raster dotted artefact on tall pages (compositor tile, not CSS).

## 3. How I will judge Gemini (so he knows the bar before he trains)

Hard gates, in order — a failure stops promotion:
1. **Data hygiene**: sentence-hash split, 0 % train/test overlap; no SLR43/54 in pretraining; manifest + `rejected.jsonl`.
2. **Tripwires** (never the promotion criterion): alignment ±1.6 frames, `len_diff=0`, no NaN/clipping, RTF ≤ 0.05 @1 thread, size budget.
3. **Blind ears** on `eval/frozen_v1` (500 + 100 contrasts): CMOS ≥ +0.5 vs V1 *and* vs shipped default, accent ≥ 90 %, WER ≤ real + 2 pt. I will render ckpt audio on the CPU box every 10 K steps from `Progress.sample` files and put them on the Listen page blind.
4. **No metric gaming**: no hand-typed numbers in `tiers.json`; every promoted figure must come from a committed JSON produced by a script.

Early red flags I will call out on the live card: GPU util < 5 % while a job is "running" (dataloader stall), loss flat for >10 K steps, samples with the same buzz/warble signature as `cand_b_v2` (Gemini EXP2), `व→w` leaks on `वि-` words under `v_mode=kathmandu`.

## 4. Next for me (CPU box, independent of Gemini)
* A1 listen desk (Rasa-F / Rasa-M / SLR43 / FLEURS, blind) page for the owner.
* A5 frozen eval + baselines rendered on CPU while the GPU trains.
* Poll the GPU box and repo; post a verdict note after each 10 K checkpoint.
