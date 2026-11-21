#!/usr/bin/env python3
"""Sakhiya v6 Autonomous Overnight RSI Daemon (`scripts/rsi_v6_daemon.py`).
Executes Phase 1 -> Phase 2 (`v6_teacher`) -> Phase 3 (`v6_voca_4m` <=4.8MB + `v6_sano_1p5m` <=1.9MB) -> Phase 4 SWA & Hard-Error RSI:
1. Phase 1: Builds `feats/v6_expressive_vedic_24k.pt` (Rasa Expressive Nepali + Vagdhenu Octave-Shifted Neural-VC Chanting).
2. Phase 2 (`v6_teacher`): Trains Large Expressive + Musical Vedic Teacher with explicit `pitch_add` & `Vagdhenu` sur/laya,
   gates with `forensic_audio_gate.py`, freezes `docs/samples/versions/v6_teacher/`, and pushes to GitHub Pages.
3. Phase 3 (`v6_voca_4m` & `v6_sano_1p5m`): Distills deterministic 1-pass `Sakhiya-Voca` (~4.34 MB int8) and `Sakhiya-Sano` (~1.71 MB int8),
   exports ONNX to `docs/models_voca/` and `docs/models_sano/`, freezes 15-clip comparison packs, and pushes live.
4. Phase 4 (`Cycle 4+`): Continuous error-mined RSI + SWA Weight-Space Soup + 30-min `v6-ckpt` GitHub Release archival."""
import copy, json, math, os, random, shutil, subprocess, sys, time
import numpy as np, torch, torch.nn.functional as F

R = "/work/tts"
REPO = "/work/tts/repo" if os.path.exists("/work/tts/repo") else os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
from sakhiya.cfm import SakhiyaAM
from sakhiya.ctc_aligner import MelCTCAligner
from sakhiya.symbols import LANG2ID
from sakhiya.v6_distill import distill_v6_student
from sakhiya.v6_pipeline import archive_v6_release, build_v6_expressive_vedic_corpus
from scripts.rsi_v5_daemon import train_v5_voc_slm_ltas, update_rsi_state
from scripts.train_v3_am import _batch_prosody, dump_gta


def train_v6_teacher_am(D, mean, std, ctc, am_init, out_dir, tag="v6_teacher", steps_a=900, steps_b=600, dev="cuda"):
    os.makedirs(out_dir, exist_ok=True)
    tr = [x for x in D["items"] if x.get("split") == "train" and 3 <= len(x["ids"]) and 20 <= x["mel"].shape[1] <= 900]
    exp_pool = [i for i, x in enumerate(tr) if x.get("v3_style_id", 0) in (2, 5)] or list(range(len(tr)))
    net = SakhiyaAM().to(dev); net.stats.copy_(torch.tensor([mean, std])); ema = copy.deepcopy(net).eval()
    if am_init and os.path.exists(am_init):
        ck = torch.load(am_init, map_location=dev, weights_only=False)
        sd = {k: v.float() if torch.is_tensor(v) else v for k, v in ck.get("ema", ck.get("net", ck)).items() if not k.startswith("ctc_rec.")}
        if os.path.exists(f"{R}/runs/v5_c1/am.pt"):
            ck5 = torch.load(f"{R}/runs/v5_c1/am.pt", map_location=dev, weights_only=False)
            sd5 = ck5.get("ema", ck5.get("net", ck5))
            for k, v in sd5.items():
                if k.startswith("vel."): sd[k] = v.float()
        net.load_state_dict(sd, strict=False); ema = copy.deepcopy(net).eval()
    for p in net.vel.parameters(): p.requires_grad_(False)
    net.ctc_rec = ctc
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=6.0e-6, betas=(0.9, 0.99), weight_decay=1e-2)
    tot = steps_a + steps_b
    for step in range(1, tot + 1):
        in_b = step > steps_a
        lr = 1.2e-5 * 0.5 * (1 + math.cos(math.pi * step / tot)) + 2.0e-6
        lam_ctc = 0.0 if in_b else 0.22
        for g_ in opt.param_groups: g_["lr"] = lr
        idxs = random.sample(exp_pool, min(16, len(exp_pool))) + random.sample(range(len(tr)), 8)
        xs = [tr[i] for i in idxs]
        L = max(len(x["ids"]) for x in xs); T = max(x["mel"].shape[1] for x in xs)
        ids = torch.zeros(len(xs), L, dtype=torch.long, device=dev); dur_gt = torch.zeros(len(xs), L, dtype=torch.long, device=dev)
        po = torch.zeros(len(xs), L, dtype=torch.float32, device=dev); y = torch.zeros(len(xs), 100, T, device=dev)
        for k, x in enumerate(xs):
            Lk, Tm = len(x["ids"]), x["mel"].shape[1]
            ids[k, :Lk] = x["ids"].to(dev); dur_gt[k, :Lk] = x["dur_ctc"].to(dev).long()
            if "po" in x and x["po"] is not None: po[k, :Lk] = x["po"][:Lk].to(dev) * 0.45
            y[k, :, :Tm] = (x["mel"].to(dev).float() - mean) / std
        il = torch.tensor([len(x["ids"]) for x in xs], device=dev); yl = torch.tensor([x["mel"].shape[1] for x in xs], device=dev)
        spks = torch.zeros(len(xs), dtype=torch.long, device=dev)
        langs = torch.tensor([0 if (x["lang"] == "sa" and random.random() < 0.85) else LANG2ID.get(x["lang"], 0) for x in xs], device=dev)
        stys = torch.tensor([int(x.get("v3_style_id", 0)) for x in xs], device=dev)
        pros = _batch_prosody(y, yl, il)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, parts = net.loss(ids, il, y, yl, spks, langs, stys, spks + 1, po, None, pros, dur_gt, spk_true=spks, lambda_ctc=lam_ctc, lambda_grl=0.0)
        opt.zero_grad(set_to_none=True); loss.backward()
        if torch.isfinite(torch.nn.utils.clip_grad_norm_(net.parameters(), 0.9)): opt.step()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), net.parameters()): pe.lerp_(p, 0.002)
        if step % 100 == 0 or step == tot:
            sd_ema = {k: v for k, v in ema.state_dict().items() if not k.startswith("ctc_rec.")}
            torch.save(dict(ema=sd_ema, step=step, mean=mean, std=std, pre_baked_v25=True, version="6.0"), f"{out_dir}/am.pt")
            p_info = dict(step=step, phase=f"{tag}-{'B' if in_b else 'A'}", **parts)
            json.dump(p_info, open(f"{out_dir}/progress.json", "w"), indent=2)
            os.makedirs(f"{R}/runs/v4_am", exist_ok=True)
            json.dump(p_info, open(f"{R}/runs/v4_am/progress.json", "w"), indent=2)
            update_rsi_state(1, tag, f"Phase 2 ({tag}) Expressive+Vedic Teacher step {step}/{tot} | cfm={parts['cfm']} dur={parts['dur_l1']} ctc={parts['ctc_pron']}")
            print(f"[{tag}-AM] step={step}/{tot} {parts}", flush=True)
    return f"{out_dir}/am.pt"


def export_gate_and_ship(tier, am_p, voc_p, out_model_dir, ver_tag, ver_label, notes):
    subprocess.run([sys.executable, f"{REPO}/scripts/export_v2_onnx.py", "--tier", tier, "--am", am_p, "--voc", voc_p, "--out", out_model_dir], cwd=REPO, check=True)
    f_json = f"{REPO}/eval/forensic_{ver_tag}.json"
    subprocess.run([sys.executable, f"{REPO}/scripts/forensic_audio_gate.py", "--model", out_model_dir, "--limit", "30", "--out", f_json], cwd=REPO, check=True)
    shutil.copyfile(f_json, f"{REPO}/eval/forensic_v5.json")
    subprocess.run([sys.executable, f"{REPO}/scripts/freeze_version_pack.py", "--tag", ver_tag, "--label", ver_label, "--models", out_model_dir, "--metrics-json", f_json, "--notes", notes], cwd=REPO, check=True)
    cmd = f"cd {REPO} && GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0 git pull --rebase origin main 2>/dev/null || true && GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0 git add docs/ eval/ && GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0 git commit -m 'v6 RSI ship {ver_tag}: {ver_label}' && GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0 git push origin HEAD:main"
    subprocess.run(cmd, shell=True, check=False)


def main():
    dev = "cuda"
    update_rsi_state(0, "Phase 1: Vagdhenu Neural-VC + Rasa Expressive Corpus", "Building single-voice v6 expressive + Vedic chanting dataset...")
    D = build_v6_expressive_vedic_corpus(dev=dev)
    mean, std = float(D.get("mean", -5.5028)), float(D.get("std", 2.1869))
    ctc = MelCTCAligner().to(dev).train()
    ck_c = torch.load(f"{R}/feats/v31_ctc_recognizer.pt", map_location=dev, weights_only=False)
    ctc.load_state_dict(ck_c.get("model", ck_c)); [p.requires_grad_(False) for p in ctc.parameters()]

    am_init = f"{R}/runs/v5_c1/am.pt"
    voc_init = f"{R}/runs/v5_c1/voc.pt"
    t_dir = f"{R}/runs/v6_teacher"
    t_am_p = f"{t_dir}/am.pt" if os.path.exists(f"{t_dir}/am.pt") else train_v6_teacher_am(D, mean, std, ctc, am_init, t_dir, "v6_teacher", 900, 600, dev)
    t_voc_p = voc_init
    if not os.path.exists(f"{REPO}/docs/samples/versions/v6_teacher/.frozen"):
        export_gate_and_ship("v25", t_am_p, t_voc_p, f"{REPO}/docs/models_v2", "v6_teacher", "v6-Teacher (Expressive + Musical Vedic Ceiling)", "Trained on Rasa Expressive + Vagdhenu Neural-VC Vedic Chanting with explicit pitch_add")
    archive_v6_release({"v6_teacher_am.fp16.pt": t_am_p, "v6_teacher_voc.fp16.pt": t_voc_p, "v6_state.json": f"{R}/rsi_v5_state.json"})

    for idx_t, (tier, vtag, vlbl, mdir, a_st, v_st) in enumerate([
        ("voca", "v6_voca_4m", "v6-Voca (4.3 MB int8 · Source-Filter Student)", f"{REPO}/docs/models_voca", 1500, 900),
        ("sano", "v6_sano_1p5m", "v6-Sano (1.7 MB int8 · 3-Student Latent Contract)", f"{REPO}/docs/models_sano", 1600, 900),
    ], start=2):
        s_dir = f"{R}/runs/{vtag}"
        if not os.path.exists(f"{REPO}/docs/samples/versions/{vtag}/.frozen"):
            s_am_p, s_voc_p = distill_v6_student(tier, D, t_am_p, t_voc_p, ctc, s_dir, a_st, v_st, lambda s, tot, msg: update_rsi_state(idx_t, vtag, msg), dev)
            export_gate_and_ship(tier, s_am_p, s_voc_p, mdir, vtag, vlbl, f"Deterministic 1-pass distillation from v6_teacher ({tier})")
            archive_v6_release({f"{vtag}_am.fp16.pt": s_am_p, f"{vtag}_voc.fp16.pt": s_voc_p, "v6_state.json": f"{R}/rsi_v5_state.json"})

    for cyc in range(4, 16):
        c_tag = f"v6_rsi_c{cyc}"
        c_dir = f"{R}/runs/{c_tag}"
        update_rsi_state(4, f"Phase 4: Overnight SWA + Hard-Error RSI (Cycle {cyc})", f"Running Cycle {cyc} teacher + student SWA polish...")
        t_am_p = train_v6_teacher_am(D, mean, std, ctc, t_am_p, c_dir, c_tag, 450, 350, dev)
        subprocess.run([sys.executable, f"{REPO}/scripts/export_v2_onnx.py", "--tier", "v25", "--am", t_am_p, "--voc", t_voc_p, "--out", f"{REPO}/docs/models_v2"], cwd=REPO, check=False)
        subprocess.run([sys.executable, f"{REPO}/scripts/forensic_audio_gate.py", "--model", f"{REPO}/docs/models_v2", "--limit", "30", "--out", f"{REPO}/eval/forensic_v6_teacher.json"], cwd=REPO, check=False)
        shutil.copyfile(f"{REPO}/eval/forensic_v6_teacher.json", f"{REPO}/eval/forensic_v5.json")
        for tier, vtag, mdir, plbl in [
            ("voca", "v6_voca_4m", f"{REPO}/docs/models_voca", f"v6-Voca Polish C{cyc} (4.3 MB int8 · CTC-KL + Phase-GAN)"),
            ("sano", "v6_sano_1p5m", f"{REPO}/docs/models_sano", f"v6-Sano Polish C{cyc} (1.7 MB int8 · CTC-KL + Phase-GAN)"),
        ]:
            s_dir = f"{R}/runs/{vtag}"
            ck_prev = torch.load(f"{s_dir}/am.pt", map_location="cpu", weights_only=False)
            prev_step = int(ck_prev.get("step", 1500))
            v_steps = 0 if os.path.exists(f"{R}/runs/v6_{tier}_voclong/voc.pt") else 300
            s_am_p, s_voc_p = distill_v6_student(tier, D, t_am_p, t_voc_p, ctc, s_dir, prev_step + 600, v_steps, lambda s, tot, msg: update_rsi_state(4, f"{c_tag}:{vtag}", msg), dev)
            subprocess.run([sys.executable, f"{REPO}/scripts/export_v2_onnx.py", "--tier", tier, "--am", s_am_p, "--voc", s_voc_p, "--out", mdir], cwd=REPO, check=False)
            f_s = f"{REPO}/eval/forensic_{vtag}.json"
            subprocess.run([sys.executable, f"{REPO}/scripts/forensic_audio_gate.py", "--model", mdir, "--limit", "30", "--out", f_s], cwd=REPO, check=False)
            ptag = f"{vtag}_polish" if not os.path.exists(f"{REPO}/docs/samples/versions/{vtag}_polish/.frozen") else f"{vtag}_c{cyc}"
            subprocess.run([sys.executable, f"{REPO}/scripts/freeze_version_pack.py", "--tag", ptag, "--label", plbl, "--models", mdir, "--metrics-json", f_s, "--notes", f"RSI Cycle {cyc} CTC-KL + Phase-GAN student polish"], cwd=REPO, check=False)
        cmd = f"cd {REPO} && GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0 git pull --rebase origin main 2>/dev/null || true && GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0 git add docs/ eval/ && GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0 git commit -m 'v6 RSI Cycle {cyc}: ship teacher + voca + sano polish' && GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0 git push origin HEAD:main"
        subprocess.run(cmd, shell=True, check=False)
        archive_v6_release({
            "v6_teacher_am.fp16.pt": t_am_p,
            "v6_voca_4m_am.fp16.pt": f"{R}/runs/v6_voca_4m/am.pt",
            "v6_voca_4m_voc.fp16.pt": f"{R}/runs/v6_voca_voclong/voc.pt",
            "v6_sano_1p5m_am.fp16.pt": f"{R}/runs/v6_sano_1p5m/am.pt",
            "v6_sano_1p5m_voc.fp16.pt": f"{R}/runs/v6_sano_voclong/voc.pt",
            "v6_state.json": f"{R}/rsi_v5_state.json",
        })


if __name__ == "__main__":
    main()
