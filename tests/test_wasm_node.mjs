// End-to-end Node test: docs/wasm/sakhiya.wasm (shipped browser binary) vs native sakhiya-cli.
// usage: node tests/test_wasm_node.mjs [--gguf-dir /tmp/gguf] [--cli build/sakhiya-cli] [--tiers atom_lite,atom] [--out eval/ggml_proof/wasm_report.json]
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const GG = arg('--gguf-dir', '/tmp/gguf'), CLI = arg('--cli', path.join(ROOT, 'build', 'sakhiya-cli'));
const TIERS = arg('--tiers', 'atom_lite,atom').split(',');
const OUT = arg('--out', path.join(ROOT, 'eval', 'ggml_proof', 'wasm_report.json'));
const SENT = [
  ['ne_kathmandu', 'ne', 'news', 'नेपालको राजधानी काठमाडौँ हिमालको काखमा रहेको एउटा ऐतिहासिक सहर हो।'],
  ['sa_om', 'sa', 'vedic_chant', 'ॐ पूर्णमदः पूर्णमिदं पूर्णात्पूर्णमुदच्यते। पूर्णस्य पूर्णमादाय पूर्णमेवावशिष्यते॥'],
];

const { createSakhiyaEngine } = await import(pathToFileURL(path.join(ROOT, 'docs/wasm/sakhiya_engine.js')).href);
// The shipped glue is web,worker only (no emsdk needed here): run it unmodified in Node by presenting a Worker-like global and
// a file:// capable fetch (the glue resolves sakhiya.wasm next to itself). build/wasm_node/ (scripts/build_wasm.sh) is used if present.
const nodeGlue = path.join(ROOT, 'build/wasm_node/sakhiya_wasm.js');
let glue = pathToFileURL(path.join(ROOT, 'docs/wasm/sakhiya_wasm.js')).href;
if (fs.existsSync(nodeGlue) && !process.env.SAKHIYA_SHIM) glue = pathToFileURL(nodeGlue).href;
else {
  globalThis.WorkerGlobalScope = function WorkerGlobalScope() {};
  globalThis.self = globalThis;
  const realFetch = globalThis.fetch;
  globalThis.fetch = async (u, o) => {
    const url = String(u);
    if (url.startsWith('file:')) { const b = fs.readFileSync(new URL(url)); return new Response(b, { status: 200, headers: { 'content-type': 'application/wasm' } }); }
    return realFetch(u, o);
  };
}
const factory = (await import(glue)).default;
const wasmBinary = fs.readFileSync(path.join(ROOT, 'docs/wasm/sakhiya.wasm'));

function readWav(p) {
  const b = fs.readFileSync(p);
  let o = 12, data = null, ch = 1, bits = 16, sr = 24000;
  while (o + 8 <= b.length) {
    const id = b.toString('ascii', o, o + 4), sz = b.readUInt32LE(o + 4);
    if (id === 'fmt ') { ch = b.readUInt16LE(o + 10); sr = b.readUInt32LE(o + 12); bits = b.readUInt16LE(o + 22); }
    if (id === 'data') { data = b.subarray(o + 8, o + 8 + sz); break; }
    o += 8 + sz + (sz & 1);
  }
  const n = data.length / 2, f = new Float32Array(n);
  for (let i = 0; i < n; i++) f[i] = data.readInt16LE(i * 2) / 32768;
  return { f, sr };
}
const cosine = (a, b) => { const n = Math.min(a.length, b.length); let d = 0, x = 0, y = 0; for (let i = 0; i < n; i++) { d += a[i] * b[i]; x += a[i] * a[i]; y += b[i] * b[i]; } return d / Math.max(Math.sqrt(x * y), 1e-12); };

const report = { wasm_bytes: wasmBinary.length, results: [], passed: true };
for (const tier of TIERS) {
  const gguf = path.join(GG, `sakhiya_${tier}_q8_0.gguf`);
  const t0 = performance.now();
  const eng = await createSakhiyaEngine({ factory, wasmBinary, skipSimdCheck: false });
  const info = eng.loadModel(fs.readFileSync(gguf));
  const load_ms = performance.now() - t0;
  for (const [id, lang, style, text] of SENT) {
    const s0 = performance.now();
    const { pcm, sampleRate } = eng.synthesize(text, { lang, style });
    const synth_ms = performance.now() - s0;
    const wav = `/tmp/_wasm_ref_${tier}_${id}.wav`;
    execFileSync(CLI, ['-m', gguf, '-t', text, '-l', lang, '-s', style, '-o', wav, '-j', '1'], { stdio: 'pipe' });
    const ref = readWav(wav);
    // native CLI writes 16-bit PCM; quantize the WASM float PCM the same way before comparing
    const q = Float32Array.from(pcm, (v) => Math.round(Math.max(-1, Math.min(1, v)) * 32767) / 32768);
    const nan_count = pcm.reduce((c, v) => c + (Number.isFinite(v) ? 0 : 1), 0);
    const cos = cosine(q, ref.f);
    const dur = pcm.length / sampleRate;
    const r = { tier, id, tier_meta: info, samples: pcm.length, native_samples: ref.f.length, sample_rate: sampleRate, nan_count, cosine_sim: +cos.toFixed(6),
                load_ms: +load_ms.toFixed(0), synth_ms: +synth_ms.toFixed(0), wasm_rtf: +(synth_ms / 1000 / dur).toFixed(4) };
    r.ok = nan_count === 0 && sampleRate === 24000 && pcm.length === ref.f.length && cos >= 0.999;
    report.passed &&= r.ok;
    report.results.push(r);
    console.log(JSON.stringify(r));
  }
  // synthesize_ids path (what docs/g2p.js feeds): ids from the native g2p must give the same audio as synthesize_text
  const g = JSON.parse(execFileSync(CLI, ['--g2p-only', '-t', SENT[0][3], '-l', 'ne'], { encoding: 'utf8' }));
  const a = eng.synthesizeIds(g.ids, { lang: 'ne', style: 'news', pitchAdd: Float32Array.from(g.pitch_add), dm: Float32Array.from(g.dm) });
  const b = eng.synthesize(SENT[0][3], { lang: 'ne', style: 'news' });
  const cids = cosine(a.pcm, b.pcm);
  console.log(JSON.stringify({ tier, synthesize_ids_vs_text_cosine: +cids.toFixed(6), samples: a.pcm.length, text_samples: b.pcm.length }));
  report.results.push({ tier, id: 'synthesize_ids', cosine_sim: +cids.toFixed(6), ok: cids >= 0.999 });
  report.passed &&= cids >= 0.999;
  eng.free();
}
fs.mkdirSync(path.dirname(OUT), { recursive: true });
fs.writeFileSync(OUT, JSON.stringify(report, null, 1));
console.log(report.passed ? 'WASM_NODE_PASSED' : 'WASM_NODE_FAILED');
process.exit(report.passed ? 0 : 1);
