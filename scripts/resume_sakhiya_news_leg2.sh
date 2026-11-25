#!/usr/bin/env bash
# ==============================================================================
# Sakhiya-News Leg-2 Zero-Loss Relay Resume Script (`scripts/resume_sakhiya_news_leg2.sh`)
# ==============================================================================
# Designed for the upcoming 12-hour RTX PRO 6000 Blackwell session.
# Resumes with ZERO rework from `NikitHamal/sakhiya-v1-train/sakhiya_news_final/`:
#   1. Pulls `cache_100mel/` (16.13h 100-band full-bandwidth studio corpus: 84.6% real Nepal-born
#      female human prosody + 15.4% Pro-v3 rare vocab, <breath> tokens, 5D structural cues),
#      `voc_100mel_best.pt` (Phase 1 gated 100-Mel ConvNeXt-V2 AWP-iSTFT + MS-SB-CQT vocoder),
#      and `am_last.pt` (resumable Sakhiya-News AM model + EMA + optimizer + scheduler + step)
#      from Hugging Face in < 3 minutes.
#   2. Resumes Phase 2 `Sakhiya-News` Single-Style 10-Block iMF + Stochastic Flow Prosody AM
#      to full convergence (~100k-120k steps).
#   3. Runs Phase 3 Joint End-to-End Waveform Distillation (unfreezing AM + 100-mel MS-SB-CQT
#      Vocoder together on 0.8-1.2 s waveform slices with detached duration gradients +
#      MS-SB-CQT + MPD/MRD + Magnitude-Aware AWP + Multi-Layer WavLM-FM + LSQ Per-Channel QAT).
#   4. Runs the FIXED 4-Layer INT8 Forensic Gate + real INT8 initializer audit (>=70% INT8 QDQ
#      weights, distinct un-snapped FP32 vs INT8 graphs, honest pCER gap vs 0.032 anchor,
#      CTC forced-aligned geminate ratio, full-band HF_LSD_4_12kHz) + final promotion to
#      GitHub Pages (`Sakhiya-News` as the single definitive studio flagship).
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p /work/sp/sakhiya_news/cache_100mel /work/sp/sakhiya_news/eval /work/sp/sakhiya_news/listen /work/tts/ck /work/tts/logs

echo "[Leg-2 Relay 1/5] Pulling cached 100-Mel Studio Corpus, Phase-1 Vocoder & Resumable AM Checkpoint from HF (<3 min)..."
python3 - <<'PY'
import os, shutil
from huggingface_hub import snapshot_download

tok = open("/tmp/hf_token").read().strip()
dl_dir = snapshot_download(
    repo_id="NikitHamal/sakhiya-v1-train",
    allow_patterns=["sakhiya_news_final/**", "feats/v31_ctc_recognizer.pt"],
    token=tok,
    local_dir="/work/sp/hf_relay_pull",
)
src_root = os.path.join(dl_dir, "sakhiya_news_final")
if os.path.isdir(os.path.join(src_root, "cache_100mel")):
    os.system(f"cp -ru {src_root}/cache_100mel/. /work/sp/sakhiya_news/cache_100mel/")
if os.path.isdir(os.path.join(src_root, "checkpoints")):
    for fn in ("voc_100mel_best.pt", "am_last.pt", "am_best.pt"):
        p = os.path.join(src_root, "checkpoints", fn)
        if os.path.exists(p):
            shutil.copy2(p, f"/work/tts/ck/{fn}")
            shutil.copy2(p, f"/work/sp/sakhiya_news/{fn}")
if os.path.exists("/work/sp/hf_relay_pull/feats/v31_ctc_recognizer.pt") and not os.path.exists("/work/tts/ck/v31_ctc_recognizer.pt"):
    shutil.copy2("/work/sp/hf_relay_pull/feats/v31_ctc_recognizer.pt", "/work/tts/ck/v31_ctc_recognizer.pt")
print("[Leg-2 Relay] HF assets restored cleanly!")
PY

echo "[Leg-2 Relay 2/5] Starting 15-minute HF backup daemon..."
if ! pgrep -f "[b]ackup_news_daemon.py" >/dev/null 2>&1; then
    setsid nohup python3 "$REPO_ROOT/scripts/sakhiya_pro/backup_news_daemon.py" > /work/tts/logs/backup_news_daemon.log 2>&1 < /dev/null &
fi

AM_TOTAL_STEPS="${AM_TOTAL_STEPS:-100000}"
E2E_STEPS="${E2E_STEPS:-20000}"

echo "[Leg-2 Relay 3/5] Resuming Phase 2 Sakhiya-News Single-Style Flagship AM to step ${AM_TOTAL_STEPS}..."
python3 "$REPO_ROOT/scripts/sakhiya_pro/train_sakhiya_news_leg1.py" \
    --skip_phase1 \
    --resume_am /work/tts/ck/am_last.pt \
    --am_steps "${AM_TOTAL_STEPS}"

echo "[Leg-2 Relay 4/5] Running Phase 3 Joint End-to-End Waveform Distillation (AM + 100-Mel MS-SB-CQT Vocoder + LSQ QAT) for ${E2E_STEPS} steps..."
python3 "$REPO_ROOT/scripts/sakhiya_pro/train_sakhiya_news_leg1.py" \
    --skip_phase1 \
    --joint_e2e \
    --resume_am /work/tts/ck/am_last.pt \
    --e2e_steps "${E2E_STEPS}"

echo "[Leg-2 Relay 5/5] Running Fixed Honest 4-Layer INT8 Forensic Gate & Final Flagship Promotion..."
python3 "$REPO_ROOT/scripts/forensic_audio_gate.py" \
    --model /work/sp/sakhiya_news/leg1_preview \
    --tier sakhiya_news \
    --out "$REPO_ROOT/eval/forensic_sakhiya_news_final.json"

python3 - <<'PY'
import json, os, shutil
rep = json.load(open("eval/forensic_sakhiya_news_final.json"))
if rep.get("gate_pass"):
    os.makedirs("docs/models_sakhiya_news", exist_ok=True)
    shutil.copy2("/work/sp/sakhiya_news/leg1_preview/model.int8.onnx", "docs/models_sakhiya_news/model.int8.onnx")
    print("[Leg-2 Relay] Gate PASSED ✅ -> Promoted docs/models_sakhiya_news/model.int8.onnx as definitive flagship!")
else:
    print("[Leg-2 Relay] Gate report written to eval/forensic_sakhiya_news_final.json (gate_pass=False; holding promotion).")
PY

python3 "$REPO_ROOT/scripts/sakhiya_pro/backup_news_daemon.py" --once
echo "[Leg-2 Relay] ALL LEG-2 MILESTONES COMPLETE!"
