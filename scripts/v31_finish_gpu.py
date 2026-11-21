#!/usr/bin/env python3
"""Sakhiya v3.1 Automated GPU Finisher (Phases 3, 4, 5, 6) + Clean Stop After Promotion."""
import copy, json, math, os, random, re, shutil, signal, subprocess, sys, time, urllib.request
import numpy as np, torch, torch.nn.functional as F

R = "/work/tts"
REPO = "/work/tts/repo" if os.path.exists("/work/tts/repo") else os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from sakhiya.cfm import SakhiyaAM
from sakhiya.ctc_aligner import MelCTCAligner
from sakhiya.symbols import SYM2ID, LANG2ID
from scripts.train_v3_am import _batch_prosody, dump_gta
from scripts.run_v31_pipeline import run_phase3_realign


def run_phase4_am(data_p, dev="cuda"):
    out_dir = f"{R}/runs/v31_am"; os.makedirs(out_dir, exist_ok=True)
    D = torch.load(data_p, weights_only=False)
    mean, std = float(D["mean"]), float(D["std"])
    items = D["items"]
    for x in items:
        x["is_fem"] = 1 if (x["spk"] in (0, 2) or 4 <= x["spk"] <= 13) else 0
        x["mel"] = x["mel"].to(dev)
    tr = [x for x in items if x.get("split") == "train"]
    va = [x for x in items if x.get("split") == "val"][:64]
    POOL = [[i for i, x in enumerate(tr) if x["is_fem"] == k] for k in (1, 0)]
    net = SakhiyaAM().to(dev); net.stats.copy_(torch.tensor([mean, std])); ema = copy.deepcopy(net).eval()
    for init_p in (f"{R}/ckpt_v3_snap/v3_am_last.pt", f"{R}/runs/v3_am/last.pt", f"{REPO}/ckpt/am.fp16.pt"):
        if os.path.exists(init_p):
            ck = torch.load(init_p, map_location=dev, weights_only=False)
            sd = {k: v.float() if torch.is_tensor(v) else v for k, v in ck.get("ema", ck.get("net", ck)).items()}
            net.load_state_dict(sd, strict=False); ema = copy.deepcopy(net).eval()
            break
    ctc_p = f"{R}/feats/v31_ctc_recognizer.pt" if os.path.exists(f"{R}/feats/v31_ctc_recognizer.pt") else f"{R}/feats/v3_ctc_recognizer.pt"
    if os.path.exists(ctc_p):
        ctc = MelCTCAligner().to(dev).train()
        ctc.load_state_dict(torch.load(ctc_p, map_location=dev, weights_only=False)["model"])
        for p in ctc.parameters(): p.requires_grad_(False)
        ctc.gru.dropout = 0.0; ctc.train(); net.ctc_rec = ctc

    opt = torch.optim.AdamW(net.parameters(), lr=8e-5, betas=(0.9, 0.99), weight_decay=1e-2)
    best_score, flat_cnt = 999.0, 0
    # Phase 4A (12,000 steps, lambda_ctc=0.22) + Phase 4B (5,000 steps, lambda_ctc=0.0 coarticulation polish) with strict early stopping
    for step in range(1, 17001):
        in_4b = (step > 12000)
        cur_lr = (2e-5 + 6e-5 * 0.5 * (1 + math.cos(math.pi * step / 12000))) if not in_4b else (5e-6 + 1.5e-5 * 0.5 * (1 + math.cos(math.pi * (step - 12000) / 5000)))
        lam_ctc = 0.0 if in_4b else 0.22
        for g_ in opt.param_groups: g_["lr"] = cur_lr
        b_idx = random.sample(POOL[0], 12) + random.sample(POOL[1], 12)
        xs = [tr[i] for i in b_idx]; L = max(len(x["ids"]) for x in xs); T = max(x["mel"].shape[1] for x in xs)
        ids = torch.zeros(len(xs), L, dtype=torch.long, device=dev); dur_gt = torch.zeros(len(xs), L, dtype=torch.long, device=dev)
        y = torch.zeros(len(xs), 100, T, device=dev)
        for k, x in enumerate(xs):
            Lk, Tm = len(x["ids"]), x["mel"].shape[1]
            ids[k, :Lk] = x["ids"].to(dev); dur_gt[k, :Lk] = x["dur_ctc"].to(dev).long()
            y[k, :, :Tm] = (x["mel"].float() - mean) / std
        il = torch.tensor([len(x["ids"]) for x in xs], device=dev); yl = torch.tensor([x["mel"].shape[1] for x in xs], device=dev)
        spks = torch.tensor([0 if x["is_fem"] == 1 else 1 for x in xs], device=dev)
        langs = torch.tensor([0 if (x["lang"] == "sa" and random.random() < 0.78) else LANG2ID[x["lang"]] for x in xs], device=dev)
        stys = torch.tensor([x.get("v3_style_id", 0) for x in xs], device=dev)
        pros = _batch_prosody(y, yl, il)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, parts = net.loss(ids, il, y, yl, spks, langs, stys, torch.ones_like(spks), None, None, pros, dur_gt, lambda_ctc=lam_ctc)
        opt.zero_grad(set_to_none=True); loss.backward()
        if torch.isfinite(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)): opt.step()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), net.parameters()): pe.lerp_(p, 0.002)
        if step % 2500 == 0:
            score = parts["dur_l1"] + (parts["ctc_pron"] if not in_4b else parts["cfm"])
            if score < best_score - 0.0015:
                best_score = score; flat_cnt = 0
            else:
                flat_cnt += 1
            sd_ema = {k: v for k, v in ema.state_dict().items() if not k.startswith("ctc_rec.")}
            torch.save(dict(ema=sd_ema, step=step, mean=mean, std=std, pre_baked_v25=True, version="3.1"), f"{out_dir}/last.pt")
            json.dump(dict(step=step, phase="4B" if in_4b else "4A", **parts), open(f"{out_dir}/progress.json", "w"), indent=2)
            if flat_cnt >= 2 and step >= 7500 and not in_4b:
                step = 12000; flat_cnt = 0   # Jump immediately to Phase 4B coarticulation polish on 4A plateau!
            elif flat_cnt >= 2 and in_4b:
                break
    dump_gta(ema, items, mean, std, dst=f"{R}/feats/v31_gta_24k.pt", dev=dev)
    return f"{out_dir}/last.pt"


def archive_v31_release(am_p, voc_p, ctc_p):
    tok = re.search(r"echo (\S+)", open("/tmp/gitenv.sh").read())[1]
    h = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json", "User-Agent": "curl/8.5.0"}
    try:
        rel = json.load(urllib.request.urlopen(urllib.request.Request("https://api.github.com/repos/NikitHamal/Sakhiya/releases/tags/v31-ckpt", headers=h), timeout=30))
    except Exception:
        b = json.dumps(dict(tag_name="v31-ckpt", name="Sakhiya v3.1 Checkpoint Release", body="Archived fp16 checkpoints (v31_am, v31_voc, v31_ctc).", prerelease=False)).encode()
        rel = json.load(urllib.request.urlopen(urllib.request.Request("https://api.github.com/repos/NikitHamal/Sakhiya/releases", data=b, method="POST", headers=h), timeout=30))
    for name, src in (("v31_am.fp16.pt", am_p), ("v31_voc.fp16.pt", voc_p), ("v31_ctc.fp16.pt", ctc_p)):
        if not src or not os.path.exists(src): continue
        for a in rel.get("assets", []):
            if a["name"] == name:
                subprocess.run(["curl", "-s", "-X", "DELETE", "-H", f"Authorization: Bearer {tok}", a["url"]])
        u = f"https://uploads.github.com/repos/NikitHamal/Sakhiya/releases/{rel['id']}/assets?name={name}"
        subprocess.run(["curl", "-s", "-X", "POST", "-H", f"Authorization: Bearer {tok}", "-H", "Content-Type: application/octet-stream", "--data-binary", f"@{src}", u])


if __name__ == "__main__":
    if os.path.exists(f"{R}/feats/v3_24k/v3_studio_aligned_24k.pt"):
        dp = run_phase3_realign()
        if dp:
            am_p = run_phase4_am(dp)
            subprocess.run(["python3", f"{REPO}/scripts/train_v3_voc.py", "--init", f"{R}/runs/v3_voc/G_best.pt", "--gta", f"{R}/feats/v31_gta_24k.pt", "--gta_p", "0.75", "--lr", "3.0e-5", "--minutes", "22", "--out", f"{R}/runs/v31_voc_3b"])
            subprocess.run(["python3", f"{REPO}/scripts/export_v2_onnx.py", "--am", am_p, "--voc", f"{R}/runs/v31_voc_3b/G_best.pt", "--out", f"{R}/export_v31"])
            subprocess.run(["python3", f"{REPO}/scripts/eval_v3.py", "--model", f"{R}/export_v31", "--limit", "100", "--promote"], cwd=REPO)
            archive_v31_release(am_p, f"{R}/runs/v31_voc_3b/G_best.pt", f"{R}/feats/v31_ctc_recognizer.pt")
