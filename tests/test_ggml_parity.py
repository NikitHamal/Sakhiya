#!/usr/bin/env python3
"""Hard Stage-by-Stage Diff Gate for sakhiya-cli (ggml C++ CLI) vs PyTorch (v5_c1/am.pt + v5_c1/voc.pt) & sakhiya/onnx_tts.py.

Verifies at every boundary:
1. Phoneme IDs: 100% exact match with sakhiya/g2p.py + sakhiya/sanskrit.py + sakhiya/chandas.py
2. Encoder output x_enc ([L, 294]), logdur ([L]), and integer durations dur ([L]): max_abs_diff < 1e-3 (dur 100% exact)
3. Mel spectrogram output ([100, T]): max_abs_diff < 1e-3
4. Final 24 kHz PCM waveform: max_abs_diff < 1e-3 and cosine_sim >= 0.998 + real measured RTF on CPU
5. Outputs side-by-side pytorch.wav vs sakhiya_cpp.wav audio proof.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import numpy as np
import soundfile as sf
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from sakhiya import g2p, sanskrit as SA, chandas as CH, symbols as S
from sakhiya.onnx_tts import STYLES, PAUSE, _declick_np
from scripts.export_v2_onnx import EncExport, DecExport, VocSpecExport
from convert_sakhiya_gguf import load_v5_c1_models


TEST_SENTENCES = [
    ("ne_kathmandu", "ne", "news", "नेपालको राजधानी काठमाडौँ हिमालको काखमा रहेको एउटा ऐतिहासिक सहर हो।"),
    ("ne_expressive", "ne", "expressive", "वाह! के अद्भुत दृश्य हो यो! माछापुच्छ्रे हिमाल बिहानको घाममा सुनजस्तै चम्किरहेको छ!"),
    ("ne_loan_num", "ne", "news", "सन् २०२६ मा नेपालको प्रविधि क्षेत्रमा १५ प्रतिशत वृद्धि भएको छ।"),
    ("ne_hiatus", "ne", "storytelling", "गाउँलेहरू आआफ्नो खेतबारीमा काम गर्न निस्किए।"),
    ("sa_om", "sa", "vedic_chant", "ॐ पूर्णमदः पूर्णमिदं पूर्णात्पूर्णमुदच्यते। पूर्णस्य पूर्णमादाय पूर्णमेवावशिष्यते॥"),
    ("sa_rv", "sa", "vedic_chant", "अ॒ग्निमी॑ळे पु॒रोहि॑तं य॒ज्ञस्य॑ दे॒वमृ॒त्विज॑म् । होता॑रं रत्न॒धात॑मम् ॥"),
    ("sa_gita", "sa", "storytelling", "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः। मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय॥"),
]

SANSKRIT_SLP1_SENTENCES = [
    "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥",
    "अ॒ग्निमी॑ळे पु॒रोहि॑तं य॒ज्ञस्य॑ दे॒वमृ॒त्विज॑म् । होता॑रं रत्न॒धात॑मम् ॥",
    "agnim īḷe purohitam tatra'sti",
    "ॐ भूर्भुवः स्वः। तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि। धियो यो नः प्रचोदयात्॥",
]


def pytorch_reference_stages(am, voc, text: str, lang: str, style: str, steps: int = 8, temp: float = 0.667):
    """Run exact PyTorch reference (EncExport -> duration rules -> DecExport -> VocSpecExport -> _ola64 -> _declick_np)."""
    enc_w = EncExport(am).eval()
    dec_w = DecExport(am, steps=steps, temp=temp).eval()
    voc_w = VocSpecExport(voc).eval()

    k = dict(STYLES[style])
    if lang == "sa" and style not in ("chant", "vedic_chant", "storytelling"):
        k["rate"] = k["rate"] * 0.96
        k["pause_scale"] *= 1.05
        k["pitch_scale"] = 1.00

    if lang == "sa":
        ch_info = CH.compile_chandas(text, bpm=75.0, tonic_hz=215.0)
        ph, off, dm = ch_info["ph"], ch_info["off"], ch_info["dm"]
        has_vm = bool(ch_info.get("has_vedic_marks"))
        sv_sc = 0.52 if has_vm else 0.48
    else:
        ph, off = g2p.text_to_phones_acc(text, lang)
        dm = g2p.phone_dm(ph)
        sv_sc = 1.15

    sid = [S.SYM2ID.get(p, -1) for p in ph]
    dm_arr = np.array([d for d, i in zip(dm, sid) if i >= 0], dtype=np.float32)
    ids_np = np.array([[i for i in sid if i >= 0]], dtype=np.int64)
    p_add_np = np.array([[o * sv_sc for o, i in zip(off, sid) if i >= 0]], dtype=np.float32)
    ph_valid = [p for p, i in zip(ph, sid) if i >= 0]

    with torch.no_grad():
        ids_t = torch.from_numpy(ids_np)
        spk_t = torch.tensor([0], dtype=torch.long)
        lang_t = torch.tensor([0], dtype=torch.long)
        sty_t = torch.tensor([int(k["style_id"])], dtype=torch.long)
        ps_t = torch.tensor(float(k["pitch_shift"]), dtype=torch.float32)
        psc_t = torch.tensor(float(k["pitch_scale"]), dtype=torch.float32)
        esc_t = torch.tensor(float(k["energy_scale"]), dtype=torch.float32)
        pa_t = torch.from_numpy(p_add_np)

        x_out_t, ld_t = enc_w(ids_t, spk_t, lang_t, ps_t, psc_t, esc_t, pa_t, sty_t)
        x = x_out_t.numpy().copy()
        ld = ld_t.numpy().copy()

        dur = np.clip(np.exp(ld[0]) - 1, 1, None) * (dm_arr if dm_arr.size == ld[0].size else 1.0)
        dur = np.where(np.isin(ids_np[0], PAUSE), dur * k["pause_scale"], dur)
        dur = np.maximum(np.round(dur / k["rate"]), 1).astype(np.int32)
        for idx_p, p in enumerate(ph_valid):
            if p not in ("|", ",", ".", "?", "!", ";"):
                dur[idx_p] = max(int(dur[idx_p]), 3)
            if p == "h" and (idx_p + 1 == len(ph_valid) or ph_valid[idx_p + 1] in ("|", ",", ".", "?", "!", ";")):
                dur[idx_p] = min(int(dur[idx_p]), 6)
            if idx_p >= 1 and idx_p + 1 < len(ph_valid) and p == "|" and ph_valid[idx_p - 1] == ph_valid[idx_p + 1] and ph_valid[idx_p - 1] in ("a", "i", "u", "e", "o", "ʌ"):
                if idx_p >= 2 and ph_valid[idx_p - 2] == "|":
                    dur[idx_p - 2] = 12
                    x[0, idx_p - 2, :100] = x[0, -1, :100]
                dur[idx_p - 1] = max(int(dur[idx_p - 1]), 19)
                dur[idx_p] = 7
                dur[idx_p + 1] = max(int(dur[idx_p + 1]), 18)
                x[0, idx_p - 1, :100] = x[0, idx_p + 1, :100]
                x[0, idx_p, :100] = 0.45 * x[0, idx_p + 1, :100] + 0.55 * x[0, -1, :100]
            if lang == "sa" and p == "o" and idx_p + 1 < len(ph_valid) and ph_valid[idx_p + 1] == "m" and (idx_p == 0 or ph_valid[idx_p - 1] in ("|", ",", ";", ".")):
                dur[idx_p] = int(np.clip(dur[idx_p], 34, 44))
                dur[idx_p + 1] = int(np.clip(dur[idx_p + 1], 16, 22))
                if idx_p + 2 < len(ph_valid) and ph_valid[idx_p + 2] == ",":
                    dur[idx_p + 2] = int(np.clip(dur[idx_p + 2], 14, 22))
            if idx_p >= 1 and idx_p + 2 < len(ph_valid) and ph_valid[idx_p - 1] in ("|", ",") and ph_valid[idx_p + 2] in ("|", ",") and p in ("r", "t̪", "n", "m", "h", "tsʰ") and ph_valid[idx_p + 1] in ("ʌ", "o"):
                dur[idx_p] = int(np.clip(dur[idx_p], 5, 9))
                dur[idx_p + 1] = int(np.clip(dur[idx_p + 1], 6, 11))

        xe = np.repeat(x, dur, axis=1)
        mel_t = dec_w(torch.from_numpy(xe))
        re_t, im_t = voc_w(mel_t)

        mel_np = mel_t[0].numpy()
        re_np = re_t[0].numpy()
        im_np = im_t[0].numpy()

        # _ola64
        n_fft, hop = 64, 16
        win = np.hanning(n_fft + 1)[:-1].astype(np.float32)
        spec = re_np + 1j * im_np
        T16 = spec.shape[1]
        fr = np.fft.irfft(spec, n_fft, axis=0).T * win
        L_ola = (T16 - 1) * hop + n_fft
        y = np.zeros(L_ola, np.float32)
        env = np.zeros(L_ola, np.float32)
        w2 = win ** 2
        for t in range(T16):
            y[t * hop:t * hop + n_fft] += fr[t]
            env[t * hop:t * hop + n_fft] += w2
        p = (n_fft - hop) // 2
        w = (y[p:-p] / np.maximum(env[p:-p], 1e-8)).astype(np.float32)
        pk = float(np.max(np.abs(w))) if w.size else 0.0
        if pk > 1e-4:
            w = np.clip(w * (0.90 / pk), -1.0, 1.0)
        if w.size > 256:
            ramp = (0.5 - 0.5 * np.cos(np.pi * np.arange(128, dtype=np.float32) / 128)).astype(np.float32)
            w[:128] *= ramp
            w[-128:] *= ramp[::-1]
            pad_n = int(0.055 * 24000)
            w = np.concatenate([np.zeros(pad_n, dtype=np.float32), w, np.zeros(pad_n // 2, dtype=np.float32)])
        w = _declick_np(w)

    return dict(
        ph=ph,
        ids=ids_np[0],
        pitch_add=p_add_np[0],
        dm=dm_arr,
        x_enc=x[0],
        logdur=ld[0],
        dur=dur,
        mel=mel_np,
        spec_re=re_np,
        spec_im=im_np,
        wav=w,
    )


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a64 = a.astype(np.float64).ravel()
    b64 = b.astype(np.float64).ravel()
    n = min(a64.size, b64.size)
    a64, b64 = a64[:n], b64[:n]
    denom = np.linalg.norm(a64) * np.linalg.norm(b64)
    return float(np.dot(a64, b64) / max(denom, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", default=os.path.join(REPO_ROOT, "build", "sakhiya-cli"))
    ap.add_argument("--am", default="/work/tts/runs/v5_c1/am.pt")
    ap.add_argument("--voc", default="/work/tts/runs/v5_c1/voc.pt")
    ap.add_argument("--gguf", default="/work/tts/models/sakhiya_f32.gguf")
    ap.add_argument("--gguf-f16", default="/work/tts/models/sakhiya_f16.gguf")
    ap.add_argument("--gguf-q8", default="/work/tts/models/sakhiya_q8_0.gguf")
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "eval", "ggml_proof"))
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    assert os.path.exists(args.cli), f"sakhiya-cli binary not found at {args.cli}"

    print("0. Verifying all MASTER_15_CLIPS & frontend G2P / Sanskrit phoneme IDs 100% exact parity...")
    from scripts.freeze_version_pack import MASTER_15_CLIPS
    for cid, lg, st, txt in MASTER_15_CLIPS:
        if lg == "auto":
            continue
        cpp_g2p = json.loads(subprocess.check_output([args.cli, "--g2p-only", "-t", txt, "-l", lg], text=True))
        if lg == "sa":
            ch_info = CH.compile_chandas(txt, bpm=75.0, tonic_hz=215.0)
            ph, off, dm = ch_info["ph"], ch_info["off"], ch_info["dm"]
            sv_sc = 0.52 if ch_info.get("has_vedic_marks") else 0.48
        else:
            ph, off = g2p.text_to_phones_acc(txt, lg)
            dm = g2p.phone_dm(ph)
            sv_sc = 1.15
        sid = [S.SYM2ID.get(p, -1) for p in ph]
        py_ids = [i for i in sid if i >= 0]
        py_pa = np.array([o * sv_sc for o, i in zip(off, sid) if i >= 0], dtype=np.float32)
        py_dm = np.array([d for d, i in zip(dm, sid) if i >= 0], dtype=np.float32)
        assert cpp_g2p["ph"] == ph, f"MASTER_15 ph mismatch on {cid}: {cpp_g2p['ph']} vs {ph}"
        assert cpp_g2p["ids"] == py_ids, f"MASTER_15 ids mismatch on {cid}: {cpp_g2p['ids']} vs {py_ids}"
        assert float(np.max(np.abs(np.array(cpp_g2p["pitch_add"], dtype=np.float32) - py_pa))) < 2e-4
        assert float(np.max(np.abs(np.array(cpp_g2p["dm"], dtype=np.float32) - py_dm))) < 1e-6
    print("   [PASS] All MASTER_15_CLIPS phoneme IDs, pitch_add, and dm 100% exact match.")

    print("1. Verifying Sanskrit SLP1 & Laya Scales 100% parity...")
    for s_txt in SANSKRIT_SLP1_SENTENCES:
        py_ids, py_wt, py_off, py_syms = SA.phonemize_slp1(s_txt)
        cpp_out = json.loads(subprocess.check_output([args.cli, "--slp1-only", "-t", s_txt], text=True))
        assert cpp_out["ids"] == py_ids, f"SLP1 ids mismatch on {s_txt}: {cpp_out['ids']} vs {py_ids}"
        assert cpp_out["weight"] == py_wt, f"SLP1 weight mismatch on {s_txt}"
        assert cpp_out["syms"] == py_syms, f"SLP1 syms mismatch on {s_txt}"
        off_diff = float(np.max(np.abs(np.array(cpp_out["off"]) - np.array(py_off))))
        assert off_diff < 1e-6, f"SLP1 off diff {off_diff} on {s_txt}"

        l_ph, l_off, l_dm = SA.laya_scales(s_txt, True, True)
        cpp_laya = json.loads(subprocess.check_output([args.cli, "--laya-only", "-t", s_txt], text=True))
        assert cpp_laya["ph"] == l_ph, f"laya_scales ph mismatch on {s_txt}: {cpp_laya['ph']} vs {l_ph}"
        assert float(np.max(np.abs(np.array(cpp_laya["off"]) - np.array(l_off)))) < 1e-6
        assert float(np.max(np.abs(np.array(cpp_laya["dm"]) - np.array(l_dm)))) < 1e-6
    print("   [PASS] Sanskrit SLP1 + laya_scales 100% exact match.")

    print("2. Loading PyTorch v5_c1 reference models...")
    am, voc = load_v5_c1_models(args.am, args.voc)

    report = {"sentences": [], "all_passed": True}
    for cid, lang, style, text in TEST_SENTENCES:
        t_py0 = time.perf_counter()
        ref = pytorch_reference_stages(am, voc, text, lang, style)
        py_ms = (time.perf_counter() - t_py0) * 1000.0

        with tempfile.TemporaryDirectory() as tmpd:
            cpp_wav_path = os.path.join(tmpd, "cpp.wav")
            subprocess.run(
                [args.cli, "-m", args.gguf, "-t", text, "-l", lang, "-s", style,
                 "-o", cpp_wav_path, "-j", str(args.threads), "--dump-stages", tmpd],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            meta = json.load(open(os.path.join(tmpd, "meta.json")))
            cpp_ids = np.fromfile(os.path.join(tmpd, "ids.i64.bin"), dtype=np.int64)
            cpp_pa = np.fromfile(os.path.join(tmpd, "pitch_add.f32.bin"), dtype=np.float32)
            cpp_dm = np.fromfile(os.path.join(tmpd, "dm.f32.bin"), dtype=np.float32)
            cpp_x = np.fromfile(os.path.join(tmpd, "x_enc.f32.bin"), dtype=np.float32).reshape(meta["L"], 294)
            cpp_ld = np.fromfile(os.path.join(tmpd, "logdur.f32.bin"), dtype=np.float32)
            cpp_dur = np.fromfile(os.path.join(tmpd, "dur.i32.bin"), dtype=np.int32)
            cpp_mel = np.fromfile(os.path.join(tmpd, "mel.f32.bin"), dtype=np.float32).reshape(100, meta["T_mel"])
            cpp_re = np.fromfile(os.path.join(tmpd, "spec_re.f32.bin"), dtype=np.float32).reshape(33, meta["T_spec"])
            cpp_im = np.fromfile(os.path.join(tmpd, "spec_im.f32.bin"), dtype=np.float32).reshape(33, meta["T_spec"])
            cpp_wav = np.fromfile(os.path.join(tmpd, "wav.f32.bin"), dtype=np.float32)

        ids_exact = bool(np.array_equal(cpp_ids, ref["ids"]))
        dur_exact = bool(np.array_equal(cpp_dur, ref["dur"]))
        pa_diff = float(np.max(np.abs(cpp_pa - ref["pitch_add"])))
        dm_diff = float(np.max(np.abs(cpp_dm - ref["dm"])))
        x_diff = float(np.max(np.abs(cpp_x - ref["x_enc"])))
        ld_diff = float(np.max(np.abs(cpp_ld - ref["logdur"])))
        mel_diff = float(np.max(np.abs(cpp_mel - ref["mel"])))
        re_diff = float(np.max(np.abs(cpp_re - ref["spec_re"])))
        im_diff = float(np.max(np.abs(cpp_im - ref["spec_im"])))
        wav_diff = float(np.max(np.abs(cpp_wav - ref["wav"])))
        wav_cos = cosine_sim(cpp_wav, ref["wav"])

        item_res = dict(
            id=cid, lang=lang, style=style,
            L=int(meta["L"]), T_mel=int(meta["T_mel"]), wav_samples=int(meta["wav_samples"]),
            ids_exact=ids_exact, dur_exact=dur_exact,
            pitch_add_max_diff=pa_diff, dm_max_diff=dm_diff,
            x_enc_max_diff=x_diff, logdur_max_diff=ld_diff,
            mel_max_abs_diff=mel_diff,
            spec_re_max_diff=re_diff, spec_im_max_diff=im_diff,
            wav_max_abs_diff=wav_diff, wav_cosine_sim=wav_cos,
            cpp_synth_ms=round(meta["synth_ms"], 2),
            py_synth_ms=round(py_ms, 2),
            cpp_rtf=round(meta["rtf"], 4),
            audio_sec=round(meta["audio_sec"], 3),
        )
        report["sentences"].append(item_res)
        print(f"   [{cid}] ids_exact={ids_exact} dur_exact={dur_exact} "
              f"mel_diff={mel_diff:.2e} wav_diff={wav_diff:.2e} cos={wav_cos:.6f} "
              f"RTF={meta['rtf']:.4f} ({meta['synth_ms']:.1f}ms / {meta['audio_sec']:.2f}s)")

        if cid == "ne_kathmandu":
            sf.write(os.path.join(args.out_dir, "pytorch.wav"), ref["wav"], 24000, subtype="PCM_16")
            sf.write(os.path.join(args.out_dir, "sakhiya_cpp.wav"), cpp_wav, 24000, subtype="PCM_16")
            sf.write(os.path.join(REPO_ROOT, "pytorch.wav"), ref["wav"], 24000, subtype="PCM_16")
            sf.write(os.path.join(REPO_ROOT, "sakhiya_cpp.wav"), cpp_wav, 24000, subtype="PCM_16")
        if cid == "sa_om":
            sf.write(os.path.join(args.out_dir, "pytorch_sa.wav"), ref["wav"], 24000, subtype="PCM_16")
            sf.write(os.path.join(args.out_dir, "sakhiya_cpp_sa.wav"), cpp_wav, 24000, subtype="PCM_16")

        wav_mae = float(np.mean(np.abs(cpp_wav - ref["wav"])))
        item_res["wav_mean_abs_diff"] = wav_mae
        assert ids_exact, f"Phoneme IDs mismatch on {cid}: {cpp_ids} vs {ref['ids']}"
        assert dur_exact, f"Duration mismatch on {cid}: {cpp_dur} vs {ref['dur']}"
        assert pa_diff < 1e-3, f"pitch_add max_abs_diff {pa_diff} >= 1e-3 on {cid}"
        assert x_diff < 1e-3, f"x_enc max_abs_diff {x_diff} >= 1e-3 on {cid}"
        assert ld_diff < 1e-3, f"logdur max_abs_diff {ld_diff} >= 1e-3 on {cid}"
        assert mel_diff < 1e-3, f"Mel max_abs_diff {mel_diff} >= 1e-3 on {cid}"
        assert wav_diff < 1e-3, f"Wav max_abs_diff {wav_diff} >= 1e-3 on {cid}"
        assert wav_mae < 1e-3, f"Wav mean_abs_diff {wav_mae} >= 1e-3 on {cid}"
        assert wav_cos >= 0.998, f"Wav cosine_sim {wav_cos} < 0.998 on {cid}"

    # Also benchmark FP16 and Q8_0 GGUF models on ne_kathmandu and sa_om
    report["quantized_tiers"] = {}
    for q_tag, q_path in [("f32", args.gguf), ("f16", args.gguf_f16), ("q8_0", args.gguf_q8)]:
        if not os.path.exists(q_path):
            continue
        sz_mb = round(os.path.getsize(q_path) / (1024 * 1024), 2)
        q_items = []
        for cid, lang, style, text in [TEST_SENTENCES[0], TEST_SENTENCES[4]]:
            ref = pytorch_reference_stages(am, voc, text, lang, style)
            with tempfile.TemporaryDirectory() as tmpd:
                subprocess.run(
                    [args.cli, "-m", q_path, "-t", text, "-l", lang, "-s", style,
                     "-o", os.path.join(tmpd, "q.wav"), "-j", str(args.threads), "--dump-stages", tmpd],
                    check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                meta = json.load(open(os.path.join(tmpd, "meta.json")))
                cpp_mel = np.fromfile(os.path.join(tmpd, "mel.f32.bin"), dtype=np.float32).reshape(100, meta["T_mel"])
                cpp_wav = np.fromfile(os.path.join(tmpd, "wav.f32.bin"), dtype=np.float32)
            mel_d = float(np.max(np.abs(cpp_mel - ref["mel"])))
            wav_cos = cosine_sim(cpp_wav, ref["wav"])
            q_items.append(dict(
                id=cid, mel_max_abs_diff=mel_d, wav_cosine_sim=wav_cos,
                cpp_rtf=round(meta["rtf"], 4), cpp_synth_ms=round(meta["synth_ms"], 2)
            ))
            assert wav_cos >= 0.998, f"{q_tag} cosine_sim {wav_cos} < 0.998 on {cid}"
        report["quantized_tiers"][q_tag] = dict(size_mb=sz_mb, results=q_items)
        print(f"   [GGUF {q_tag:4s} | {sz_mb:5.2f} MB] ne_cos={q_items[0]['wav_cosine_sim']:.6f} sa_cos={q_items[1]['wav_cosine_sim']:.6f} RTF={q_items[0]['cpp_rtf']:.4f}")

    json.dump(report, open(os.path.join(args.out_dir, "diff_gate_report.json"), "w"), indent=2, ensure_ascii=False)
    print("\nALL STAGE-BY-STAGE DIFF GATES PASSED!")


if __name__ == "__main__":
    main()
