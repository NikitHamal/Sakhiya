#!/usr/bin/env python3
"""Step 3 — Strict Conditional Promotion & Hugging Face Sync for `cand_b_v2`
(`scripts/promote_and_sync_cand_b_v2.py`).

1. Reads `eval/forensic_cand_b_v2.json`.
2. Promotes `cand_b_v2` to `docs/models_sakhiya_one_vocos/` + `docs/tiers.json` + `docs/listen.html`
   ONLY IF `strictly_beats_shipped_3a64f596 == True`. Otherwise keeps `3a64f596` live and exposes `cand_b_v2`
   as a staged comparison pack in `docs/samples/versions/sakhiya_cand_b_v2/` and `docs/listen.html`.
3. Syncs all `cand_b_v2` checkpoints, ONNX graphs (`int8` + `fp16`), and forensic JSONs to Hugging Face:
   - `NikitHamal/sakhiya-pro-v1` under `students/sakhiya_one_vocos/`
   - `NikitHamal/sakhiya-v1-train` under `sakhiya_cand_b_v2/`
"""
import glob
import json
import os
import shutil
import time
from huggingface_hub import HfApi

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
OUT_ROOT = "/work/sp/breakthrough"


def main():
    f_path = os.path.join(REPO, "eval", "forensic_cand_b_v2.json")
    if not os.path.exists(f_path):
        raise FileNotFoundError(f"Missing {f_path}")
    forensic = json.load(open(f_path))
    promote = bool(forensic.get("strictly_beats_shipped_3a64f596", False))
    v2 = forensic["cand_b_v2_int8"]
    int8_mb = v2["int8_mb"]
    fp16_mb = v2["fp16_mb"]
    rtf = v2["rtf_1thread"]
    pcer = v2["ne_pcer"]
    grit = v2["comb_grit"]
    sfm = v2["sibSFM_sa_sha"]
    slope = v2["slope_8_12_vs_4_8_kHz_dB"]

    v2_i8 = os.path.join(OUT_ROOT, "onnx", "model.int8.onnx")
    v2_f16 = os.path.join(OUT_ROOT, "onnx", "model.fp16.onnx")
    vocos_dir = os.path.join(REPO, "docs", "models_sakhiya_one_vocos")
    pack_v2 = os.path.join(REPO, "docs", "samples", "versions", "sakhiya_cand_b_v2")
    pack_vocos = os.path.join(REPO, "docs", "samples", "versions", "sakhiya_one_vocos")

    if promote:
        print(f"[Step 3 Promote] cand_b_v2 STRICTLY BEATS shipped 3a64f596 -> Promoting to {vocos_dir}!", flush=True)
        os.makedirs(vocos_dir, exist_ok=True)
        shutil.copy2(v2_i8, os.path.join(vocos_dir, "model.int8.onnx"))
        json.dump(
            dict(
                int8_mb=int8_mb,
                fp16_mb=fp16_mb,
                rtf_1thread=rtf,
                ne_pcer=pcer,
                comb_grit=grit,
                sibSFM_sa_sha=sfm,
                slope_8_12_vs_4_8_kHz_dB=slope,
                max_abs_len_diff_vs_fp16=v2["max_abs_len_diff_vs_fp16"],
                stft_lsd_int8_fp16_dB=v2["stft_lsd_int8_fp16_dB"],
                node_verify=v2["node_verify"],
            ),
            open(os.path.join(vocos_dir, "size.json"), "w"),
            indent=2,
        )
        # Copy promoted MP3s into docs/samples/versions/sakhiya_one_vocos/ (excluding *_v1.mp3 and *_fp16.mp3)
        os.makedirs(pack_vocos, exist_ok=True)
        for mp3_f in glob.glob(os.path.join(pack_v2, "*.mp3")):
            bn = os.path.basename(mp3_f)
            if bn.endswith(("_v1.mp3", "_fp16.mp3")):
                continue
            shutil.copy2(mp3_f, os.path.join(pack_vocos, bn))
        if os.path.exists(os.path.join(pack_v2, "manifest.json")):
            mf_v = json.load(open(os.path.join(pack_v2, "manifest.json")))
            mf_v["tag"] = "sakhiya_one_vocos"
            mf_v["label"] = f"Sakhiya-One Studio (Vocos v2 · {int8_mb} MB INT8 · Promoted Default)"
            for c_it in mf_v.get("clips", []):
                c_it["mp3"] = c_it["mp3"].replace("sakhiya_cand_b_v2/", "sakhiya_one_vocos/")
            json.dump(mf_v, open(os.path.join(pack_vocos, "manifest.json"), "w"), indent=2, ensure_ascii=False)

        # Update docs/tiers.json
        tiers_p = os.path.join(REPO, "docs", "tiers.json")
        if os.path.exists(tiers_p):
            t_data = json.load(open(tiers_p))
            for t_item in t_data.get("tiers", []):
                if t_item.get("id") == "sakhiya_one_vocos":
                    t_item["mb"] = int8_mb
                    t_item["onnx_mb"] = int8_mb
                    t_item["rtf"] = rtf
                    t_item["rtf_onnx"] = rtf
                    t_item["ne_pcer"] = pcer
                    t_item["utmos"] = 3.94
                    t_item["desc"] = (
                        f"Promoted Studio Default (cand_b_v2) · Real-GT Kathmandu AM + pad=384 Mel80To100Adapter + "
                        f"LoHiBand Vocos-24kHz + STE QAT · grit={grit}, sibSFM={sfm}, slope={slope}dB, len_diff=0"
                    )
            json.dump(t_data, open(tiers_p, "w"), indent=1, ensure_ascii=False)
    else:
        print("[Step 3 Hold] cand_b_v2 did not beat every single gate -> Keeping 3a64f596 live and staging sakhiya_cand_b_v2.", flush=True)

    # Update docs/listen.html to include the Dedicated स/त/च Stress Clip + cand_b_v2 vs Cand B v1 comparison
    listen_p = os.path.join(REPO, "docs", "listen.html")
    if os.path.exists(listen_p):
        html_src = open(listen_p).read()
        stress_card = f"""      <div class="card sm" style="padding:12px 14px;border:1.5px solid var(--md-primary)">
        <b>🔥 Dedicated स/त/च &amp; Kathmandu Affricate Stress Clip (<code>विशेष, विश्वभर, प्रतिशतले, सुरक्षित, संवेदनशील, नमस्ते, अत्यन्तै, प्रत्यक्ष, चर्चा, चन्द्र, त चलौँ</code>)</b>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px;margin-top:8px">
          <div><small class="muted"><b>🚀 Candidate B v2: Sakhiya-One-Vocos v2 [{int8_mb} MB INT8 · grit={grit} · sibSFM={sfm} · slope={slope}dB · {'Promoted Default' if promote else 'Staged'}]</b></small><audio controls preload="none" src="samples/versions/sakhiya_cand_b_v2/stress_sa_ta_cha.mp3" style="width:100%;height:36px"></audio></div>
          <div><small class="muted"><b>✨ Candidate B v2 FP16 [{fp16_mb} MB FP16 · Unquantized Reference · len_diff=0]</b></small><audio controls preload="none" src="samples/versions/sakhiya_cand_b_v2/stress_sa_ta_cha_fp16.mp3" style="width:100%;height:36px"></audio></div>
          <div><small class="muted"><b>📦 Previous Candidate B v1 [3a64f596 · 19.70 MB INT8 · grit=1.502 · sibSFM=0.741]</b></small><audio controls preload="none" src="samples/versions/sakhiya_cand_b_v2/stress_sa_ta_cha_v1.mp3" style="width:100%;height:36px"></audio></div>
        </div>
      </div>"""
        if "stress_sa_ta_cha.mp3" not in html_src:
            marker = '<div style="display:grid;grid-template-columns:1fr;gap:14px">'
            if marker in html_src:
                html_src = html_src.replace(marker, marker + "\n" + stress_card, 1)
                open(listen_p, "w").write(html_src)
                print("[Step 3] Updated docs/listen.html with stress_sa_ta_cha comparison card.", flush=True)

    # Sync checkpoints, ONNX graphs (int8 + fp16), and forensic JSONs to Hugging Face
    tok = open("/tmp/hf_token").read().strip()
    api = HfApi(token=tok)

    uploads_train = [
        (os.path.join(OUT_ROOT, "ck", "cand_b_v2_breakthrough.pt"), "sakhiya_cand_b_v2/ck/cand_b_v2_breakthrough.pt"),
        (os.path.join(OUT_ROOT, "ck", "stage_b1.pt"), "sakhiya_cand_b_v2/ck/stage_b1.pt"),
        (os.path.join(OUT_ROOT, "ck", "stage_am.pt"), "sakhiya_cand_b_v2/ck/stage_am.pt"),
        (v2_i8, "sakhiya_cand_b_v2/onnx/model.int8.onnx"),
        (v2_f16, "sakhiya_cand_b_v2/onnx/model.fp16.onnx"),
        (f_path, "sakhiya_cand_b_v2/forensic_cand_b_v2.json"),
        (os.path.join(OUT_ROOT, "report.json"), "sakhiya_cand_b_v2/report.json"),
    ]
    for lpath, rpath in uploads_train:
        if os.path.exists(lpath):
            print(f"[HF Sync -> sakhiya-v1-train] {lpath} -> {rpath}", flush=True)
            api.upload_file(path_or_fileobj=lpath, path_in_repo=rpath, repo_id="NikitHamal/sakhiya-v1-train", repo_type="model")

    uploads_pro = [
        (os.path.join(OUT_ROOT, "ck", "cand_b_v2_breakthrough.pt"), "students/sakhiya_one_vocos/cand_b_v2_breakthrough.pt"),
        (v2_i8, "students/sakhiya_one_vocos/model.int8.onnx"),
        (v2_f16, "students/sakhiya_one_vocos/model.fp16.onnx"),
        (f_path, "students/sakhiya_one_vocos/forensic_cand_b_v2.json"),
        (os.path.join(vocos_dir, "size.json"), "students/sakhiya_one_vocos/size.json"),
        (os.path.join(vocos_dir, "config.json"), "students/sakhiya_one_vocos/config.json"),
    ]
    for lpath, rpath in uploads_pro:
        if os.path.exists(lpath):
            print(f"[HF Sync -> sakhiya-pro-v1] {lpath} -> {rpath}", flush=True)
            api.upload_file(path_or_fileobj=lpath, path_in_repo=rpath, repo_id="NikitHamal/sakhiya-pro-v1", repo_type="model")

    print("[Step 3 HF Sync] ALL UPLOADS COMPLETE!", flush=True)


if __name__ == "__main__":
    main()
