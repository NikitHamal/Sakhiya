"""Export Sakhiya to ONNX (3 graphs) + fp16-weight / int8 compaction with 2 flagship voices (Female & Male).
python scripts/export_onnx.py --am runs/am/last.pt --voc runs/voc/G.pt --out /work/tts/export"""
import os, sys, argparse, json
import numpy as np, onnx, torch
from onnx import numpy_helper, helper, TensorProto
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sakhiya.acoustic import Acoustic
from sakhiya.vocoder import Vocoder, load_compat

ap = argparse.ArgumentParser()
ap.add_argument("--am", default="/work/tts/runs/v2_am_uni/last.pt"); ap.add_argument("--voc", default="/work/tts/runs/v2_voc_3b1/G_best.pt"); ap.add_argument("--out", default="/work/tts/export")
a, extra = ap.parse_known_args(); os.makedirs(a.out, exist_ok=True)

c = torch.load(a.am, map_location="cpu", weights_only=False)
if isinstance(c, dict) and ("ema" in c or "net" in c) and "M" not in c:
    import subprocess
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "export_v2_onnx.py"), "--am", a.am, "--voc", a.voc, "--out", a.out] + extra, check=True)
    sys.exit(0)
am = Acoustic(n_spk=c["M"]["spk.weight"].shape[0]); am.load_state_dict(c["M"]); am.eval()
with torch.no_grad():   # zero-shot Tharu tracks inherit the trained Nepali language embedding
    for i in (1, 2, 3) + (() if c.get("has_sa") else (4,)): am.lang.weight[i] = am.lang.weight[0]
sd = torch.load(a.voc, map_location="cpu", weights_only=False)
voc = Vocoder(); load_compat(voc, sd["G"] if isinstance(sd, dict) and "G" in sd else sd); voc.eval()


def pick_two_voices(spk2id, pitch_stats):
    """Select the single best Native Nepali Female voice (sakhiya / s3614) and single best Native Nepali Male voice (cv_b0487694)."""
    fem_candidates = ["sakhiya", "s3614", "edge_f", "s7957", "s6329", "s2139", "s0546", "s0258"]
    male_candidates = ["edge_m", "cv_b0487694", "cv_13b830fe", "cv_626e8569", "cv_0142a5d1", "fl_m1", "fl_m2"]
    f_id = next((spk2id[k] for k in fem_candidates if k in spk2id), next((v for k, v in spk2id.items() if k.startswith("s")), 0))
    m_id = next((spk2id[k] for k in male_candidates if k in spk2id), next((v for k, v in spk2id.items() if not k.startswith("s")), min(1, len(spk2id) - 1)))
    f_key = next((k for k, v in spk2id.items() if v == f_id), "female")
    m_key = next((k for k, v in spk2id.items() if v == m_id), "male")
    return f_id, m_id, f_key, m_key


class Enc(torch.nn.Module):
    def __init__(s): super().__init__(); s.m = am
    def forward(s, ids, spk, lang, pitch_shift, pitch_scale, energy_scale, pitch_add):
        h, ld, p, e = s.m.encode(ids, spk, lang)
        return s.m.condition(h, p * pitch_scale + pitch_shift + pitch_add, e * energy_scale), ld


class Dec(torch.nn.Module):
    def __init__(s): super().__init__(); s.m = am
    def forward(s, x): return s.m.decode(x)


class Voc(torch.nn.Module):
    def __init__(s): super().__init__(); s.m = voc
    def forward(s, mel): return s.m.spectrum(mel)


def fp16_weights(path):
    """Store big float32 initializers as float16 + Cast->float32 (halves size, exact fp32 compute)."""
    m = onnx.load(path); new_nodes = []; keep = []
    for init in m.graph.initializer:
        if init.data_type == TensorProto.FLOAT and np.prod(init.dims) > 512:
            arr = numpy_helper.to_array(init).astype(np.float16)
            h = numpy_helper.from_array(arr, init.name + "_h"); keep.append(h)
            new_nodes.append(helper.make_node("Cast", [h.name], [init.name], to=TensorProto.FLOAT, name=init.name + "_cast"))
        else:
            keep.append(init)
    del m.graph.initializer[:]; m.graph.initializer.extend(keep)
    nodes = list(m.graph.node); del m.graph.node[:]; m.graph.node.extend(new_nodes + nodes)
    onnx.save(m, path)


def slim_onnx(path):
    """Convert remaining float32 Constant/1D tensors (>128 floats) to fp16+Cast and compact internal symbol names."""
    m = onnx.load(path); new_inits, kept_nodes = [], []
    for idx, n in enumerate(m.graph.node):
        n.doc_string = ""
        if n.op_type == "Constant" and len(n.output) == 1:
            t = next((at.t for at in n.attribute if at.name == "value"), None)
            if t is not None and t.data_type == TensorProto.FLOAT and np.prod(t.dims) > 128:
                hn = f"c{idx}"; new_inits.append(numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float16), hn))
                kept_nodes.append(helper.make_node("Cast", [hn], [n.output[0]], to=TensorProto.FLOAT)); continue
        kept_nodes.append(n)
    m.graph.initializer.extend(new_inits); del m.graph.node[:]; m.graph.node.extend(kept_nodes)
    cast_nodes, final_inits = [], []
    for idx, init in enumerate(m.graph.initializer):
        if init.data_type == TensorProto.FLOAT and np.prod(init.dims) > 128:
            hn = f"h{idx}"; final_inits.append(numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), hn))
            cast_nodes.append(helper.make_node("Cast", [hn], [init.name], to=TensorProto.FLOAT))
        else:
            final_inits.append(init)
    del m.graph.initializer[:]; m.graph.initializer.extend(final_inits)
    nodes = list(m.graph.node); del m.graph.node[:]; m.graph.node.extend(cast_nodes + nodes)
    io_names = {x.name for x in list(m.graph.input) + list(m.graph.output)}; remap = {}
    def rn(s):
        if not s or s in io_names: return s
        if s not in remap: remap[s] = f"t{len(remap)}"
        return remap[s]
    for init in m.graph.initializer: init.name = rn(init.name)
    for n in m.graph.node:
        n.name = ""
        for i in range(len(n.input)): n.input[i] = rn(n.input[i])
        for i in range(len(n.output)): n.output[i] = rn(n.output[i])
    del m.graph.value_info[:]
    onnx.save(m, path)


def int8_weights(src, dst):
    """Weight-only per-channel int8 (QDQ): int8 initializer -> DequantizeLinear -> float32 weight."""
    m = onnx.load(src); use = {}
    for n in m.graph.node:
        for i in n.input: use.setdefault(i, n.op_type)
    new_nodes, keep = [], []
    for init in m.graph.initializer:
        if init.data_type == TensorProto.FLOAT and np.prod(init.dims) > 512 and len(init.dims) >= 2:
            w = numpy_helper.to_array(init); ax = 1 if use.get(init.name) == "MatMul" and w.ndim == 2 else 0
            red = tuple(i for i in range(w.ndim) if i != ax); sc = np.maximum(np.abs(w).max(axis=red), 1e-8) / 127.0
            shp = [1] * w.ndim; shp[ax] = -1
            q = np.clip(np.round(w / sc.reshape(shp)), -127, 127).astype(np.int8)
            keep += [numpy_helper.from_array(q, init.name + "_q"), numpy_helper.from_array(sc.astype(np.float32), init.name + "_s")]
            new_nodes.append(helper.make_node("DequantizeLinear", [init.name + "_q", init.name + "_s"], [init.name], axis=ax, name=init.name + "_dq"))
        else:
            keep.append(init)
    del m.graph.initializer[:]; m.graph.initializer.extend(keep)
    nodes = list(m.graph.node); del m.graph.node[:]; m.graph.node.extend(new_nodes + nodes)
    onnx.save(m, dst)
    slim_onnx(dst)


if __name__ == "__main__":
    T = 40
    spec = {
        "encoder": (Enc(), (torch.randint(4, 60, (1, T)), torch.tensor([0]), torch.tensor([0]), torch.tensor(0.0), torch.tensor(1.0), torch.tensor(1.0), torch.zeros(1, T)),
                    ["ids", "spk", "lang", "pitch_shift", "pitch_scale", "energy_scale", "pitch_add"], ["x", "logdur"],
                    {"ids": {1: "T"}, "pitch_add": {1: "T"}, "x": {1: "T"}, "logdur": {1: "T"}}),
        "decoder": (Dec(), (torch.randn(1, 200, 256),), ["x"], ["mel"], {"x": {1: "F"}, "mel": {2: "F"}}),
        "vocoder": (Voc(), (torch.randn(1, 80, 100),), ["mel"], ["real", "imag"], {"mel": {2: "F"}, "real": {2: "F"}, "imag": {2: "F"}}),
    }
    sizes = {}
    for name, (m, args, inn, outn, dyn) in spec.items():
        p32 = f"{a.out}/{name}.fp32.onnx"
        torch.onnx.export(m, args, p32, input_names=inn, output_names=outn, dynamic_axes=dyn, opset_version=17, dynamo=False)
        p16 = f"{a.out}/{name}.onnx"; onnx.save(onnx.load(p32), p16); fp16_weights(p16)
        sizes[name] = dict(fp32_mb=os.path.getsize(p32) / 1e6, fp16_mb=os.path.getsize(p16) / 1e6)
        pi8 = f"{a.out}/{name}.int8.onnx"; int8_weights(p32, pi8)
        sizes[name]["int8_mb"] = os.path.getsize(pi8) / 1e6
        os.remove(p32)

    sa_ckpt = next((p for p in ("/work/tts/runs/sa_cepstral.pt", "/work/tts/alpa/alpa-v0.2.pt") if os.path.exists(p)), None)
    if sa_ckpt and os.path.exists("/work/tts/alpa/alpa"):
        sys.path.insert(0, "/work/tts/alpa")
        from alpa.infer import load_model as _load_sa
        from alpa.export import Exportable as _ExpCls
        from alpa.model import ALL_PULSES, NO_PULSES, treble_kernel
        from sakhiya.sanskrit_cepstral import build_exportable
        sa_m = build_exportable(_load_sa(sa_ckpt, device="cpu"), _ExpCls, ALL_PULSES, NO_PULSES, treble_kernel)
        sa_p32 = f"{a.out}/sanskrit.fp32.onnx"; sa_out = f"{a.out}/sanskrit.onnx"
        sa_args = (torch.randint(5, 47, (1, 24)), torch.zeros((1, 24), dtype=torch.long), torch.tensor([0]),
                   torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([1.0]), torch.zeros((1, 24), dtype=torch.float32))
        torch.onnx.export(sa_m, sa_args, sa_p32,
                          input_names=["tokens", "weight", "voice", "speed", "pitch_shift", "pitch_range", "pitch_add"],
                          output_names=["audio", "durations"],
                          dynamic_axes={"tokens": {1: "T"}, "weight": {1: "T"}, "pitch_add": {1: "T"}, "audio": {1: "S"}, "durations": {1: "T"}},
                          opset_version=17, dynamo=False)
        int8_weights(sa_p32, sa_out)
        sz_sa = os.path.getsize(sa_out) / 1e6
        sizes["sanskrit"] = dict(fp32_mb=os.path.getsize(sa_p32) / 1e6, fp16_mb=sz_sa, int8_mb=sz_sa)
        os.remove(sa_p32)

f_id, m_id, f_key, m_key = pick_two_voices(c["spk2id"], c["pitch_stats"])
speakers = [
    {"name": "female", "label": "सखिया · Female (Native Nepali)", "id": int(f_id), "src_spk": f_key},
    {"name": "male", "label": "सन्देश · Male (Native Nepali)", "id": int(m_id), "src_spk": m_key},
]
try:   # real-human voices (best SLR43/SLR54 female + male), written by build_mix.py
    rv = json.load(open("/work/tts/feats/real_voices.json"))["best"]
    for g, nm, lab in (("f", "human_female", "Human-data Female (real recordings)"), ("m", "human_male", "Human-data Male (real recordings)")):
        if rv.get(g) in c["spk2id"]: speakers.append({"name": nm, "label": lab, "id": int(c["spk2id"][rv[g]]), "src_spk": rv[g]})
except Exception as e: print("no real voices", e)
for g, nm, lab in (("f", "sanskrit_female", "Sanskrit Female (real radio speech)"), ("m", "sanskrit_male", "Sanskrit Male (real radio speech)")):
    k = next((s_ for s_ in sorted(c["spk2id"]) if s_.startswith(f"sa_{g}")), None)
    if k and c.get("has_sa"): speakers.append({"name": nm, "label": lab, "id": int(c["spk2id"][k]), "src_spk": k, "lang": "sa"})
ONE_VOICE = len(c["spk2id"]) == 1 or bool(os.environ.get("SAKHIYA_ONE_VOICE"))   # session 30: one female voice for ne / thr / sa, no male
if ONE_VOICE:
    speakers = [{"name": "female", "label": "सखिया · Sakhiya (one voice: Nepali · Tharu · Sanskrit)", "id": int(f_id), "src_spk": f_key}]
spk2id_out = dict(c["spk2id"])
for sp_ in speakers: spk2id_out[sp_["name"]] = sp_["id"]
spk2id_out["female"] = int(f_id)
if not ONE_VOICE: spk2id_out["male"] = int(m_id)
cfg = dict(pitch_stats=c["pitch_stats"], spk2id=spk2id_out, speakers=speakers, default_speaker="female", lang2id={**__import__("sakhiya.symbols", fromlist=["x"]).LANG2ID, **({} if c.get("has_sa") else {"sa": 0})}, sample_rate=22050, hop=256, n_fft=1024, step=c["step"])
json.dump(cfg, open(f"{a.out}/config.json", "w"), ensure_ascii=False)
tot16 = sum(v["fp16_mb"] for v in sizes.values()); toti8 = sum(v.get("int8_mb", v["fp16_mb"]) for v in sizes.values())
sizes["total_fp16_mb"], sizes["total_tiny_mb"] = tot16, toti8
json.dump(sizes, open(f"{a.out}/size.json", "w"), indent=1); print(json.dumps(sizes, indent=1))
