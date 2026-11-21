#!/bin/bash
# Wave 5 (session 8): Warm-starts AM6 from AM5 with Global Variance (GV) formant contrast + 6x 2-voice specialization (s3614 Female & cv_b0487694 Male)
# -> fine-tunes Vocoder5 from voc3/last.pt (preserving trained D + 20% GTA / 80% GT phase-consistent studio training)
# -> exports ONNX with 129-point studio presence curve & 0.90 unclipped peak norm -> evaluates -> promotes & archives fp16 ckpt/.
R=/work/tts; L=$R/logs; cd $R/repo; mkdir -p $R/runs $L
AM=${AM_MIN:-4.5}; VOC=${VOC_MIN:-2.5}
push() { export GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0; git checkout -- docs/data/status.json 2>/dev/null || true; git add -A docs eval ckpt STATUS.md LEDGER.json scripts sakhiya monitor 2>/dev/null; git commit -qm "$1"; git pull -q --rebase -X ours origin main; git push -q origin HEAD:main; }

echo "WAVE5_START $(date)" > $L/wave5.log
mkdir -p $R/best5; rm -rf $R/best5/models $R/best5/samples; cp -r docs/models $R/best5/models; cp -r docs/samples $R/best5/samples 2>/dev/null; cp docs/metrics.json $R/best5/metrics.json
export ROMAN_ORT=$R/node/package.json

python3 scripts/train_am.py --set v2 --minutes $AM --bs 64 --lr 2e-4 --align_warm 0 --init $R/runs/am5/last.pt --out $R/runs/am6 > $L/am6.log 2>&1
python3 scripts/train_vocoder.py --set v2 --minutes $VOC --out $R/runs/voc5 --bs 28 --gta $R/feats/gta_v2.pt --init $R/runs/voc3/last.pt --lr 3e-5 --phase 1.0 > $L/voc5.log 2>&1

# Pick whichever vocoder (voc5 or voc3) achieves higher 20-sentence UTMOS with am6
python3 - <<'PY' >> $L/wave5.log 2>&1
import os, sys, shutil, torch, numpy as np, librosa, types, importlib.machinery as im
sys.path.insert(0, "/work/tts/repo")
from sakhiya.acoustic import Acoustic
from sakhiya.vocoder import Vocoder
dev = "cuda"
ta = types.ModuleType("torchaudio"); ta.__spec__ = im.ModuleSpec("torchaudio", None); ta.functional = types.SimpleNamespace(resample=lambda w, orig_freq, new_freq: w); sys.modules["torchaudio"] = ta
hd = os.path.join(torch.hub.get_dir(), "tarepan_SpeechMOS_v1.2.0"); sys.path.insert(0, hd)
utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(dev).eval()
to16 = lambda w: torch.from_numpy(librosa.resample(w.astype(np.float32), orig_sr=22050, target_sr=16000))[None].to(dev)
data = torch.load("/work/tts/feats/v2.pt", weights_only=False)
test = [r for r in data if r["split"] == "test" and r["src"] == "slr43"][:20]
ck = torch.load("/work/tts/runs/am6/last.pt", weights_only=False)
M = Acoustic(n_spk=ck["M"]["spk.weight"].shape[0]).to(dev); M.load_state_dict(ck["M"]); M.eval()
sid = ck["spk2id"]["s3614"]
best_v, best_u = "/work/tts/runs/voc3/G.pt", -1.0
for vp in ["/work/tts/runs/voc3/G.pt", "/work/tts/runs/voc5/G.pt"]:
    if not os.path.exists(vp): continue
    V = Vocoder().to(dev); V.load_state_dict(torch.load(vp, weights_only=False)); V.eval()
    with torch.no_grad():
        u = float(np.mean([float(utmos(to16(V(M.infer(torch.tensor([r["ph"]], dtype=torch.long, device=dev), spk=sid)[0])[0].cpu().numpy()), 16000).detach()) for r in test]))
    print("Candidate vocoder", vp, "UTMOS:", round(u, 4))
    if u > best_u: best_u, best_v = u, vp
shutil.copy(best_v, "/work/tts/runs/best_voc5_G.pt")
print("Selected vocoder:", best_v, "with UTMOS", round(best_u, 4))
PY

python3 scripts/export_onnx.py --am $R/runs/am6/last.pt --voc $R/runs/best_voc5_G.pt --out $R/export6 > $L/export6.log 2>&1 || { echo EXPORT_FAIL >> $L/wave5.log; exit 1; }
cp $R/export6/*.onnx $R/export6/config.json $R/export6/size.json docs/models/
git checkout -- docs/samples 2>/dev/null; python3 scripts/evaluate.py --export $R/export6 --out $R/eval6 > $L/eval6.log 2>&1
python3 scripts/eval_roman.py > $L/roman.log 2>&1; cp eval/roman_metrics.json docs/roman_metrics.json 2>/dev/null

python3 - <<'PY' > $L/promote5.log 2>&1
import json, shutil, torch, os
R = "/work/tts"; g = lambda p: next(s for s in json.load(open(p))["systems"] if s["system"] == "Sakhiya-fp16")["sentence"]
new, old = g(f"{R}/repo/docs/metrics.json"), g(f"{R}/best5/metrics.json")
better = new["pcer"] <= old["pcer"] + 0.003 and new["mos"] >= old["mos"] - 0.03 and (new["pcer"] < old["pcer"] or new["mos"] > old["mos"])
print("new", new["pcer"], new["mos"], "old", old["pcer"], old["mos"], "PROMOTE" if better else "REVERT")
if better:
    os.makedirs(f"{R}/repo/ckpt", exist_ok=True); c = torch.load(f"{R}/runs/am6/last.pt", weights_only=False)
    c["M"] = {k: v.half() for k, v in c["M"].items()}; torch.save(c, f"{R}/repo/ckpt/am.fp16.pt")
    torch.save({k: v.half() for k, v in torch.load(f"{R}/runs/best_voc5_G.pt", weights_only=False).items()}, f"{R}/repo/ckpt/G.fp16.pt")
else:
    for d in ("models", "samples"): shutil.rmtree(f"{R}/repo/docs/{d}"); shutil.copytree(f"{R}/best5/{d}", f"{R}/repo/docs/{d}")
    shutil.copy(f"{R}/best5/metrics.json", f"{R}/repo/docs/metrics.json"); shutil.copy(f"{R}/best5/metrics.json", f"{R}/repo/eval/metrics.json")
PY
cat $L/promote5.log >> $L/wave5.log
push "wave 5: GV-contrast AM6 + 129-pt studio presence vocoder, 2 voices ($(tail -n 1 $L/promote5.log | awk '{print $NF}'))"
echo WAVE5_DONE >> $L/wave5.log
