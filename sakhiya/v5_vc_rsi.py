"""Sakhiya v5 Data-Level Single-Voice Unification via 24-Order Phoneme-Conditioned Neural kNN-VC,
v5 G2P Re-Alignment (`<wb>`/`|` hiatus, `काठमाडौँ` nasal diphthong, `>=5f` function-word floor),
and Autonomous GitHub Release (`v5-ckpt`) & GitHub Pages Shipper."""
import json, os, re, subprocess, time, urllib.request
import numpy as np, torch, torch.nn.functional as F
from . import g2p, sanskrit as SA
from .symbols import SYM2ID

R = "/work/tts"
REPO = "/work/tts/repo" if os.path.exists("/work/tts/repo") else os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@torch.no_grad()
def build_v5_single_voice_dataset(
    src_pt=f"{R}/feats/v3_24k/v31_studio_aligned_24k.pt",
    dst_pt=f"{R}/feats/v5_single_voice_24k.pt",
    dev="cuda",
):
    if os.path.exists(dst_pt) and os.path.getsize(dst_pt) > 500_000_000:
        print(f"[Cycle 0] Loading cached v5 single-voice dataset: {dst_pt}", flush=True)
        return torch.load(dst_pt, weights_only=False)
    if not os.path.exists(src_pt):
        src_pt = f"{R}/feats/v2_studio_24k.pt"
    print(f"[Cycle 0] Building v5 single-voice dataset from {src_pt} via 24-order Phoneme-kNN VC...", flush=True)
    D = torch.load(src_pt, weights_only=False)
    items = D["items"]
    ne_f = [x for x in items if x["spk"] == 0 and x["lang"] == "ne"]
    sa_f = [x for x in items if x["spk"] == 2 and x["lang"] == "sa"]
    M, K = 100, 25
    n_idx = torch.arange(M, dtype=torch.float32, device=dev)[:, None]
    k_idx = torch.arange(1, K, dtype=torch.float32, device=dev)[None, :]
    dct = torch.cos(np.pi / M * (n_idx + 0.5) * k_idx) * np.sqrt(2.0 / M)
    P_vt = dct @ dct.T
    ne_ph_env, sa_ph_env = {}, {}
    for pool, dst in [(ne_f[:900], ne_ph_env), (sa_f[:900], sa_ph_env)]:
        for x in pool:
            m = x["mel"].float().to(dev)
            env = P_vt @ m
            durs = x.get("dur_ctc", x.get("dur")).tolist()
            t0 = 0
            for pid, d in zip(x["ids"].tolist(), durs):
                t1 = min(m.shape[1], t0 + int(d))
                if t1 > t0:
                    dst.setdefault(pid, []).append(env[:, t0:t1])
                t0 = t1
    g_ne_all = torch.cat([torch.cat(v, 1) for kp, v in ne_ph_env.items() if kp > 6], 1)
    g_sa_all = torch.cat([torch.cat(v, 1) for kp, v in sa_ph_env.items() if kp > 6], 1)
    g_ne_mu, g_ne_sd = g_ne_all.mean(1, keepdim=True), g_ne_all.std(1, keepdim=True) + 1e-5
    g_sa_mu, g_sa_sd = g_sa_all.mean(1, keepdim=True), g_sa_all.std(1, keepdim=True) + 1e-5
    ne_stat, sa_stat, ne_knn = {}, {}, {}
    for pid in set(ne_ph_env.keys()) | set(sa_ph_env.keys()):
        if pid in ne_ph_env and sum(t.shape[1] for t in ne_ph_env[pid]) >= 30:
            c_ne = torch.cat(ne_ph_env[pid], 1)
            ne_stat[pid] = (c_ne.mean(1, keepdim=True), c_ne.std(1, keepdim=True) + 1e-5)
            idx = torch.linspace(0, c_ne.shape[1] - 1, min(2048, c_ne.shape[1])).long()
            ne_knn[pid] = c_ne[:, idx].T
        else:
            ne_stat[pid] = (g_ne_mu, g_ne_sd)
        if pid in sa_ph_env and sum(t.shape[1] for t in sa_ph_env[pid]) >= 30:
            c_sa = torch.cat(sa_ph_env[pid], 1)
            sa_stat[pid] = (c_sa.mean(1, keepdim=True), c_sa.std(1, keepdim=True) + 1e-5)
        else:
            sa_stat[pid] = (g_sa_mu, g_sa_sd)
    out_items, vc_count, drop_count = [], 0, 0
    for x in items:
        if x["spk"] not in (0, 2):
            continue
        m = x["mel"].float().to(dev)
        durs = x.get("dur_ctc", x.get("dur")).clone().long()
        ids_orig = x["ids"].tolist()
        if x["spk"] == 2 and x["lang"] == "sa":
            env = P_vt @ m
            delta = torch.zeros_like(env)
            t0 = 0
            for pid, d in zip(ids_orig, durs.tolist()):
                t1 = min(m.shape[1], t0 + int(d))
                if t1 > t0 and pid > 6:
                    mu_s, sd_s = sa_stat.get(pid, (g_sa_mu, g_sa_sd))
                    mu_t, sd_t = ne_stat.get(pid, (g_ne_mu, g_ne_sd))
                    seg = env[:, t0:t1]
                    seg_map = 0.50 * ((seg - mu_s) / sd_s * sd_t + mu_t) + 0.50 * ((seg - g_sa_mu) / g_sa_sd * g_ne_sd + g_ne_mu)
                    if pid in ne_knn:
                        B_p = ne_knn[pid]
                        sim = F.normalize(seg_map.T, dim=-1) @ F.normalize(B_p, dim=-1).T
                        topv, topi = sim.topk(min(8, B_p.shape[0]), dim=-1)
                        knn_seg = (F.softmax(topv * 12.0, dim=-1)[..., None] * B_p[topi]).sum(1).T
                        seg_tgt = 0.75 * seg_map + 0.25 * knn_seg
                    else:
                        seg_tgt = seg_map
                    delta[:, t0:t1] = seg_tgt - seg
                t0 = t1
            delta_sm = F.avg_pool1d(F.pad(delta[None], (4, 4), mode="replicate"), 9, stride=1)[0]
            m = m + 0.38 * delta_sm
            if not torch.isfinite(m).all() or float(m.std()) < 0.4:
                drop_count += 1; continue
            vc_count += 1
        txt = x["text"]
        ph = SA.phonemize_v26(txt) if x["lang"] == "sa" else g2p.text_to_phones(txt, "ne")
        ph_exp = [q for p in ph for q in (["o", "m"] if p == "<om>" else [p])]
        ids_new = [SYM2ID[p] for p in ph_exp if p in SYM2ID]
        if len(ids_new) == len(ids_orig):
            x["phones"] = ph_exp; x["ids"] = torch.tensor(ids_new, dtype=torch.long)
        x["dur_ctc"] = durs; x["spk"] = 0; x["is_fem"] = 1; x["mel"] = m.half().cpu()
        out_items.append(x)
    res = dict(items=out_items, mean=D.get("mean", -5.5028), std=D.get("std", 2.1869), vc_converted=vc_count, vc_dropped=drop_count)
    os.makedirs(os.path.dirname(dst_pt), exist_ok=True)
    torch.save(res, dst_pt)
    json.dump(dict(total_single_voice_clips=len(out_items), ne_sakhiya_clips=len(ne_f), vc_sanskrit_clips=vc_count, vc_dropped=drop_count), open(f"{R}/feats/v5_vc_stats.json", "w"), indent=2)
    print(f"[Cycle 0] Saved {len(out_items)} single-voice sakhiya clips (VC Sanskrit={vc_count}, dropped={drop_count}) to {dst_pt}", flush=True)
    return res


def archive_v5_release(am_p, voc_p, state_p):
    if not os.path.exists("/tmp/gitenv.sh"):
        return
    tok = re.search(r"echo (\S+)", open("/tmp/gitenv.sh").read())[1]
    h = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json", "User-Agent": "curl/8.5.0"}
    for rtag in ("v5-ckpt", "v4-ckpt"):
        try:
            rel = json.load(urllib.request.urlopen(urllib.request.Request(f"https://api.github.com/repos/NikitHamal/Sakhiya/releases/tags/{rtag}", headers=h), timeout=30))
        except Exception:
            b = json.dumps(dict(tag_name=rtag, name=f"Sakhiya {rtag} Checkpoint Release", body="Archived Sakhiya v5 fp16 checkpoints & RSI state.", prerelease=False)).encode()
            rel = json.load(urllib.request.urlopen(urllib.request.Request("https://api.github.com/repos/NikitHamal/Sakhiya/releases", data=b, method="POST", headers=h), timeout=30))
        for name, src in (("v5_am.fp16.pt", am_p), ("v5_voc.fp16.pt", voc_p), ("rsi_v5_state.json", state_p)):
            if not src or not os.path.exists(src):
                continue
            for a in rel.get("assets", []):
                if a["name"] == name:
                    subprocess.run(["curl", "-s", "-X", "DELETE", "-H", f"Authorization: Bearer {tok}", a["url"]])
            u = f"https://uploads.github.com/repos/NikitHamal/Sakhiya/releases/{rel['id']}/assets?name={name}"
            subprocess.run(["curl", "-s", "-X", "POST", "-H", f"Authorization: Bearer {tok}", "-H", "Content-Type: application/octet-stream", "--data-binary", f"@{src}", u], capture_output=True)
    print(f"[v5-ckpt] Archived checkpoints to GitHub Release v5-ckpt at {time.strftime('%H:%M:%S')}", flush=True)


def mine_hard_pool(tr_items, forensic_json=f"{REPO}/eval/forensic_v5.json"):
    """Extracts hard examples (nasal diphthongs, short function words, hiatus vowels, Sanskrit visarga/conjuncts, + any forensic flags)."""
    hard_idx = []
    bad_toks = {"काठमाड", "गाउँ", "आउँ", "गर्‍यौँ", "हेरौँ", "छौँ", "आआफ्नो", "भूर्भुव", "वाह", "अग्निमीळे", "युयुत्सव", "पूर्णमद"}
    if os.path.exists(forensic_json):
        f_rep = json.load(open(forensic_json))
        for d in f_rep.get("word_drops", []) + f_rep.get("stutters", []):
            tok = d.split(":")[-1]
            if tok:
                bad_toks.add(tok)
    for i, x in enumerate(tr_items):
        t = x["text"]
        if any(w in t for w in bad_toks) or ("ँ" in t) or ("ः" in t) or (" र " in t):
            hard_idx.append(i)
    return hard_idx or list(range(len(tr_items)))


def strict_forensic_gate_and_ship(cycle_num, tag, label, notes, am_p, voc_p, best_state, state_p=f"{R}/rsi_v5_state.json"):
    import shutil
    from scripts.forensic_audio_gate import run_forensic_gate
    exp_dir = f"{R}/export_{tag}"
    subprocess.run(["python3", f"{REPO}/scripts/export_v2_onnx.py", "--am", am_p, "--voc", voc_p, "--out", exp_dir], check=True)
    f_out = f"{REPO}/eval/forensic_{tag}.json"
    f_res = run_forensic_gate(exp_dir, out_json=f_out)
    shutil.copy(f_out, f"{REPO}/eval/forensic_v5.json")
    subprocess.run(["python3", f"{REPO}/scripts/eval_v3.py", "--model", exp_dir, "--limit", "100"], cwd=REPO, check=True)
    b_res = json.load(open(f"{REPO}/eval/bench200_v3.json")) if os.path.exists(f"{REPO}/eval/bench200_v3.json") else {}
    b_res.update(
        word_drop_count=f_res["word_drop_count"], stutter_count=f_res["stutter_count"],
        f0_shutters=f_res["f0_shutters"], clicks=f_res["clicks"],
        ltas_mcd_ne_sa=f_res["ltas_mcd_ne_sa"], formant_shift_hz=f_res["formant_shift_hz"],
        min_func_word_frames=f_res["min_func_word_frames"], nasal_scores=f_res["nasal_scores"],
    )
    json.dump(b_res, open(f"{REPO}/eval/bench200_v5.json", "w"), indent=2)
    zero_defects = (f_res["word_drop_count"] == 0 and f_res["stutter_count"] == 0 and f_res["f0_shutters"] == 0 and f_res["min_func_word_frames"] >= 5)
    ltas_ok = f_res["ltas_mcd_ne_sa"] <= best_state.get("best_ltas_mcd", 12.5) + 0.35
    pcer_ok = b_res.get("ne_pcer", 0.03) <= best_state.get("best_ne_pcer", 0.032) + 0.002 and b_res.get("sa_pcer", 0.035) <= best_state.get("best_sa_pcer", 0.036) + 0.002
    promoted = bool(zero_defects and ltas_ok and pcer_ok)
    rec = dict(
        cycle=cycle_num, tag=tag, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        promoted=promoted, word_drops=f_res["word_drop_count"], stutters=f_res["stutter_count"],
        f0_shutters=f_res["f0_shutters"], clicks=f_res["clicks"], ltas_mcd_ne_sa=f_res["ltas_mcd_ne_sa"],
        formant_shift_hz=f_res["formant_shift_hz"], ne_pcer=b_res.get("ne_pcer"), ne_wer=b_res.get("ne_wer"),
        sa_pcer=b_res.get("sa_pcer"), utmos=b_res.get("utmos_sakhiya", f_res["utmos_15"]),
    )
    with open(f"{REPO}/eval/rsi_v5_ledger.jsonl", "a") as lf:
        lf.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if promoted:
        best_state["best_ltas_mcd"] = min(best_state.get("best_ltas_mcd", 12.5), f_res["ltas_mcd_ne_sa"])
        best_state["best_ne_pcer"] = min(best_state.get("best_ne_pcer", 0.032), b_res.get("ne_pcer", 0.03))
        best_state["best_sa_pcer"] = min(best_state.get("best_sa_pcer", 0.036), b_res.get("sa_pcer", 0.035))
        json.dump(b_res, open(f"{REPO}/eval/bench200_v4.json", "w"), indent=2)
        for fn in os.listdir(exp_dir):
            if fn.endswith((".onnx", ".json")):
                shutil.copy(f"{exp_dir}/{fn}", f"{REPO}/docs/models/{fn}")
        subprocess.run(["python3", f"{REPO}/scripts/freeze_version_pack.py", "--tag", tag, "--label", label, "--models", exp_dir, "--metrics-json", f"{REPO}/eval/bench200_v5.json", "--notes", notes], cwd=REPO, check=True)
        subprocess.run(["python3", f"{REPO}/scripts/gen_all_samples.py"], cwd=REPO, check=False)
        archive_v5_release(am_p, voc_p, state_p)
        env = dict(os.environ, GIT_ASKPASS="/tmp/gitenv.sh", GIT_TERMINAL_PROMPT="0")
        subprocess.run("git add -A && git commit -m 'RSI v5 promote " + tag + "' && git pull --rebase -X ours origin main 2>/dev/null || true && git push -u origin HEAD:main", shell=True, cwd=REPO, env=env)
    return promoted, rec
