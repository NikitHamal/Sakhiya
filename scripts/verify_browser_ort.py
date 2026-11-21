"""Verify exported Sakhiya ONNX models (fp16 + tiny-int8) end-to-end in Node `onnxruntime-node`
using the EXACT browser `docs/tts.js` (`SakhiyaWeb`) + `docs/g2p.js` + `docs/sanskrit.js` + `docs/mixed.js` code path.
Checks 24 kHz output, zero NaNs, zero silence/clipping, and logs single-thread CPU RTF.
usage: python3 scripts/verify_browser_ort.py --export /work/tts/export_v2_uni [--out /tmp/browser_ort.json]"""
import argparse, json, os, subprocess, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OWNER_MIXED = ("हेलो साथीहरू। म सखिया हुँ, निकित हमालज्यूले बनाउनुभएको सानो टेक्स्ट टु स्पीच मोडेल। म संस्कृत पनि स्पष्ट बोल्न सक्छु। "
               "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥ आज मौसम निकै रमाइलो र शान्त छ।")


def verify(export_dir, node_modules=None):
    if node_modules is None:
        node_modules = "/work/tts/node/node_modules" if os.path.exists("/work/tts") else "/tmp/ort_node/node_modules"
    export_dir = os.path.abspath(export_dir)
    if not os.path.exists(os.path.join(node_modules, "onnxruntime-node")):
        base_nm = os.path.dirname(node_modules)
        os.makedirs(base_nm, exist_ok=True)
        subprocess.run(["npm", "install", "--prefix", base_nm, "onnxruntime-node", "--no-save", "--silent"], check=True)
    node_code = r"""
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const ort = require("onnxruntime-node");

const exportDir = process.env.EXPORT_DIR;
const rootDir = process.env.ROOT_DIR;
const ownerText = process.env.OWNER_TEXT;

const ortShim = {
  env: { wasm: { wasmPaths: "", numThreads: 1 } },
  Tensor: ort.Tensor,
  InferenceSession: {
    create: async (uri) => {
      const p = uri.startsWith("models/") ? path.join(exportDir, uri.slice(7)) : path.join(exportDir, path.basename(uri));
      return await ort.InferenceSession.create(p, { intraOpNumThreads: 1, interOpNumThreads: 1 });
    }
  }
};

const sandbox = {
  window: null, global: null, ort: ortShim,
  navigator: { hardwareConcurrency: 1 },
  performance: { now: () => Number(process.hrtime.bigint()) / 1e6 },
  Float32Array, Int32Array, Uint32Array, Uint8ClampedArray, BigInt64Array, BigInt, ArrayBuffer, DataView, Math, Date, Promise, Set, Object, Number, String, RegExp, Array, JSON, console,
  Blob: class Blob { constructor(a, b) { this.a = a; this.b = b; } },
  fetch: async (url) => {
    const clean = url.split("?")[0];
    if (clean.startsWith("models/")) {
      const p = path.join(exportDir, clean.slice(7));
      return { json: async () => JSON.parse(fs.readFileSync(p, "utf8")) };
    }
    const p = path.join(rootDir, "docs", clean);
    return { json: async () => JSON.parse(fs.readFileSync(p, "utf8")) };
  }
};
sandbox.window = sandbox;
sandbox.global = sandbox;
vm.createContext(sandbox);

for (const f of ["g2p.js", "sanskrit.js", "chandas.js", "mixed.js", "roman.js", "tts.js"]) {
  if (fs.existsSync(path.join(rootDir, "docs", f))) {
    vm.runInContext(fs.readFileSync(path.join(rootDir, "docs", f), "utf8"), sandbox, { filename: f });
  }
}

(async () => {
  await sandbox.SK.load();
  const web = new sandbox.SakhiyaWeb();
  const out = {};
  for (const variant of ["fp16", "tiny"]) {
    await web.init(variant, "models/");
    const r = await web.say(ownerText, { lang: "auto", spk: 0, style: "news" });
    const rMale = await web.say("अ॒ग्निमी॑ळे पु॒रोहि॑तं य॒ज्ञस्य॑ दे॒वमृ॒त्विज॑म् । होता॑रं रत्न॒धात॑मम् ॥", { lang: "sa", spk: 1, style: "vedic_chant", svara: 1.0 });
    let maxAbs = 0, sumSq = 0, nanCount = 0;
    for (let i = 0; i < rMale.wav.length; i++) if (!Number.isFinite(rMale.wav[i])) nanCount++;
    for (let i = 0; i < r.wav.length; i++) {
      const v = r.wav[i];
      if (!Number.isFinite(v)) nanCount++;
      const a = Math.abs(v);
      if (a > maxAbs) maxAbs = a;
      sumSq += v * v;
    }
    const rms = Math.sqrt(sumSq / Math.max(1, r.wav.length));
    if (r.sr !== 24000) throw new Error(`Expected 24000 Hz, got ${r.sr}`);
    if (nanCount > 0) throw new Error(`NaN/Inf samples: ${nanCount}`);
    if (maxAbs < 0.05 || maxAbs > 1.001 || rms < 0.005) throw new Error(`Silence or clipping: maxAbs=${maxAbs} rms=${rms}`);
    out[variant] = {
      sr: r.sr,
      samples: r.wav.length,
      dur_s: +(r.wav.length / r.sr).toFixed(3),
      rtf_1thread: +r.rtf.toFixed(4),
      max_abs: +maxAbs.toFixed(4),
      rms: +rms.toFixed(4),
      nan_count: nanCount,
      segments: r.segments ? r.segments.map(s => s.lang) : []
    };
  }
  console.log(JSON.stringify(out));
})().catch(e => { console.error(e); process.exit(1); });
"""
    env = dict(os.environ, NODE_PATH=node_modules, EXPORT_DIR=export_dir, ROOT_DIR=ROOT, OWNER_TEXT=OWNER_MIXED)
    raw = subprocess.check_output(["node", "-e", node_code], cwd=ROOT, env=env, text=True).strip().splitlines()[-1]
    return json.loads(raw)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = verify(a.export)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)
    print("BROWSER_ORT_OK", json.dumps(res), flush=True)
