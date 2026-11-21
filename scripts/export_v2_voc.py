"""Export Sakhiya-v2-Voc (24 kHz 16x iSTFTNet) to ONNX: fp32 -> fp16-weight (Cast) + int8 weight-only QDQ (DequantizeLinear, no ConvInteger).
Verifies against torch with onnxruntime.  usage: python3 scripts/export_v2_voc.py --g runs/v2_voc/G_best.pt --out /work/tts/export_v2"""
import os, sys, argparse, json
import numpy as np, onnx, torch
from onnx import numpy_helper, helper, TensorProto
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sakhiya.v2_vocoder import VocoderV2

ap = argparse.ArgumentParser()
ap.add_argument("--g", default="/work/tts/runs/v2_voc/G_best.pt"); ap.add_argument("--out", default="/work/tts/export_v2")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)


def fp16_weights(path):
    m = onnx.load(path); new_nodes, keep = [], []
    for init in m.graph.initializer:
        if init.data_type == TensorProto.FLOAT and np.prod(init.dims) > 256:
            h = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name + "_h"); keep.append(h)
            new_nodes.append(helper.make_node("Cast", [h.name], [init.name], to=TensorProto.FLOAT, name=init.name + "_cast"))
        else:
            keep.append(init)
    del m.graph.initializer[:]; m.graph.initializer.extend(keep)
    nodes = list(m.graph.node); del m.graph.node[:]; m.graph.node.extend(new_nodes + nodes); onnx.save(m, path)


def int8_weights(src, dst):
    m = onnx.load(src); use = {}
    for n in m.graph.node:
        for i in n.input: use.setdefault(i, n.op_type)
    new_nodes, keep = [], []
    for init in m.graph.initializer:
        if init.data_type == TensorProto.FLOAT and np.prod(init.dims) > 256 and len(init.dims) == 3 and use.get(init.name) in ("Conv", "ConvTranspose"):
            w = numpy_helper.to_array(init); ax = 1 if use[init.name] == "ConvTranspose" else 0
            red = tuple(i for i in range(w.ndim) if i != ax); sc = np.maximum(np.abs(w).max(axis=red), 1e-8) / 127.0
            shp = [1] * w.ndim; shp[ax] = -1
            q = np.clip(np.round(w / sc.reshape(shp)), -127, 127).astype(np.int8)
            keep += [numpy_helper.from_array(q, init.name + "_q"), numpy_helper.from_array(sc.astype(np.float32), init.name + "_s")]
            new_nodes.append(helper.make_node("DequantizeLinear", [init.name + "_q", init.name + "_s"], [init.name], axis=ax, name=init.name + "_dq"))
        elif init.data_type == TensorProto.FLOAT and np.prod(init.dims) > 256:
            h = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name + "_h"); keep.append(h)
            new_nodes.append(helper.make_node("Cast", [h.name], [init.name], to=TensorProto.FLOAT, name=init.name + "_cast"))
        else:
            keep.append(init)
    del m.graph.initializer[:]; m.graph.initializer.extend(keep)
    nodes = list(m.graph.node); del m.graph.node[:]; m.graph.node.extend(new_nodes + nodes); onnx.save(m, dst)


class Wrap(torch.nn.Module):
    def __init__(s, g):
        super().__init__(); s.g = g

    def forward(s, mel):
        return s.g.forward_onnx(mel)


if __name__ == "__main__":
    sd = torch.load(a.g, map_location="cpu", weights_only=False)
    G = VocoderV2(); G.load_state_dict(sd.get("G_ema", sd.get("G", sd)) if isinstance(sd, dict) and ("G_ema" in sd or "G" in sd) else sd)
    G.eval(); mel = torch.randn(1, 100, 64) * 1.5 - 5.0
    with torch.no_grad(): ref = G(mel).numpy(); ref2 = G.forward_onnx(mel).numpy()
    n = min(ref.shape[1], ref2.shape[1]); print("torch ola64 vs forward_onnx max abs diff:", float(np.abs(ref[:, :n] - ref2[:, :n]).max()), flush=True)
    G.fuse_weight_norm(); p32 = f"{a.out}/vocoder_v2_fp32.onnx"
    torch.onnx.export(Wrap(G), (mel,), p32, input_names=["mel"], output_names=["wav"], dynamic_axes={"mel": {2: "T"}, "wav": {1: "S"}}, opset_version=17, dynamo=False)
    p16, p8 = f"{a.out}/vocoder_v2_fp16.onnx", f"{a.out}/vocoder_v2_int8.onnx"
    m = onnx.load(p32); onnx.save(m, p16); fp16_weights(p16); int8_weights(p32, p8)
    import onnxruntime as ort
    res = dict(params=sum(p.numel() for p in G.parameters()), frame_rate_upsample=16)
    for name, p in (("fp32", p32), ("fp16", p16), ("int8", p8)):
        s = ort.InferenceSession(p, providers=["CPUExecutionProvider"]); y = s.run(None, {"mel": mel.numpy()})[0]
        k = min(y.shape[1], ref.shape[1])
        res[name] = dict(mb=round(os.path.getsize(p) / 1e6, 3), max_abs_diff=float(np.abs(y[:, :k] - ref[:, :k]).max()), ops=sorted({n.op_type for n in onnx.load(p).graph.node}))
        print(name, res[name]["mb"], "MB  diff", res[name]["max_abs_diff"], flush=True)
    assert "ConvInteger" not in res["int8"]["ops"]
    json.dump(res, open(f"{a.out}/export_report.json", "w"), indent=1)
