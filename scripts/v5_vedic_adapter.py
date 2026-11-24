#!/usr/bin/env python3
"""Vedic-only LoRA adapter: load / unload + gated benchmark render.

Isolation contract: Nepali rendering uses `load_base()` (no adapter code touched). The adapter is injected into a *separate*
model instance only when style == 'vedic_chant' (`load_vedic(base, adapter)`), and the base checkpoint on disk is never modified.

Pure-prompt Vedic inference (no leak): ref_audio = a full, UNSLICED verified clip (6-8 s, ASR match >= 90) and ref_text = that
clip's verbatim Devanagari; target text uses natural ॐ and pada breaks (। / ॥) -- no 'ऽऽ' stretch hack, no post-hoc Om splice.

usage: v5_vedic_adapter.py --adapter <dir|none> --tag vl1_200 [--nref 3] [--seeds 3]
"""
import argparse, json, os, sys, glob
import numpy as np, soundfile as sf, torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "sakhiya_pro"))
V = "/work/tts/v5_vedic"; BASE = "/work/tts/v5_recovery/hf/sakhiya_pro_v3/exp/v3b/latest"

BENCH = [
    ("sa_rv", "अग्निमीळे पुरोहितं यज्ञस्य देवमृत्विजम्। होतारं रत्नधातमम्॥"),
    ("sa_om", "ॐ पूर्णमदः पूर्णमिदं पूर्णात्पूर्णमुदच्यते। पूर्णस्य पूर्णमादाय पूर्णमेवावशिष्यते॥"),
    ("sa_gayatri", "ॐ भूर्भुवः स्वः। तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि। धियो यो नः प्रचोदयात्॥"),
    ("sa_gita", "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः। मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय॥"),
]


def load_base(dtype=torch.bfloat16):
    from omnivoice.models.omnivoice import OmniVoice
    return OmniVoice.from_pretrained(BASE, device_map="cuda", dtype=dtype)


def load_vedic(model, adapter_dir):
    """Inject LoRA into model.llm (in place) and load weights. Call ONLY for style == 'vedic_chant' on a dedicated instance."""
    from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    c = json.load(open(f"{adapter_dir}/adapter_config.json"))
    inject_adapter_in_model(LoraConfig(r=c["r"], lora_alpha=c["alpha"], target_modules=c["targets"], lora_dropout=0.0), model.llm)
    sd = load_file(f"{adapter_dir}/vedic_lora.safetensors")
    r = set_peft_model_state_dict(model.llm, sd)
    bad = [k for k in getattr(r, "unexpected_keys", [])]
    assert not bad, f"unexpected adapter keys {bad[:3]}"
    model.llm.to(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)
    return model


def pick_refs(n=3, lo=6.0, hi=8.0, min_match=float(os.environ.get("REF_MATCH", 90)), min_voiced=float(os.environ.get("REF_VOICED", 0.85))):
    """Razor-clean 6-8 s prompts: full clip, verbatim text, high ASR match, no Gita/Nepali, distinct texts."""
    gold = json.load(open(f"{V}/meta/vedic/gold.json")); rows = [json.loads(l) for l in open(f"{V}/ft/dev.jsonl")]
    rows += [json.loads(l) for l in open(f"{V}/ft/train.jsonl")]
    c = []
    for r in rows:
        if r["src"] != "vag" or not (lo <= r["dur"] <= hi): continue
        g = gold.get(f"/work/tts/segments/vedic_chant/{r['id']}.wav", {})
        if float(g.get("match", 0)) < min_match or float(g.get("voiced", 0)) < min_voiced: continue
        c.append((float(g["match"]) + 20 * float(g.get("steady", 0)), r))
    c.sort(key=lambda x: -x[0]); return [r for _, r in c[:n]]


def render(model, out, nref=3, seeds=3, num_step=32, gs=2.0):
    from omnivoice.models.omnivoice import OmniVoice  # noqa
    os.makedirs(out, exist_ok=True); refs = pick_refs(nref); man = []
    json.dump(refs, open(f"{out}/refs.json", "w"), ensure_ascii=False, indent=1)
    for ri, ref in enumerate(refs):
        for cid, txt in BENCH:
            for sd in range(seeds):
                torch.manual_seed(1000 * ri + sd)
                a = model.generate(text=txt, language="sa", ref_audio=ref["audio_path"], ref_text=ref["text"], num_step=num_step,
                                   guidance_scale=gs, instruct=None)
                w = np.asarray(a[0], dtype=np.float32)
                if os.environ.get("V5_RAW") != "1":   # same zero-click de-karkar chain as the Nepali master (removes RVQ crackle/hiss)
                    from v4_pro_render import declick_and_deess
                    w = declick_and_deess(w, model.sampling_rate)
                p = f"{out}/{cid}_r{ri}_s{sd}.wav"; sf.write(p, w, model.sampling_rate)
                man.append(dict(wav=p, target_text=txt, ref_text=ref["text"], mode="vedic", id=cid, ref=ri, seed=sd))
    json.dump(man, open(f"{out}/manifest.json", "w"), ensure_ascii=False, indent=1); return man


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--adapter", default="none"); ap.add_argument("--tag", required=True)
    ap.add_argument("--nref", type=int, default=3); ap.add_argument("--seeds", type=int, default=3); a = ap.parse_args()
    m = load_base()
    if a.adapter != "none": load_vedic(m, a.adapter)
    out = f"{V}/eval/{a.tag}"; man = render(m, out, a.nref, a.seeds); del m; torch.cuda.empty_cache()
    import subprocess
    subprocess.call([sys.executable, f"{HERE}/leak_loop_gate.py", "--manifest", f"{out}/manifest.json", "--out", f"{out}/gate.json"])
    g = json.load(open(f"{out}/gate.json")); best = {}
    for c in g["clips"]:
        cid = os.path.basename(c["wav"]).split("_r")[0]
        if c["ok"]: best.setdefault(cid, c["wav"])
    summ = dict(tag=a.tag, adapter=a.adapter, passed=g["passed"], n=g["n"], per_chant_pass={k: sum(1 for c in g["clips"] if os.path.basename(c["wav"]).startswith(k + "_") and c["ok"]) for k, _ in BENCH},
                chants_with_a_clean_take=sorted(best), best=best)
    json.dump(summ, open(f"{out}/summary.json", "w"), ensure_ascii=False, indent=1); print("SUMMARY", json.dumps(summ, ensure_ascii=False))


if __name__ == "__main__":
    main()
