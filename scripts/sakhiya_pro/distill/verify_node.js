// Verify a student ONNX in Node `onnxruntime-node` through the EXACT browser code path (docs/g2p.js + sanskrit.js + student_tts.js).
// usage: NODE_PATH=/work/sp/node/node_modules node verify_node.js MODEL.onnx [out.json]   (1 intra-op thread => honest CPU RTF)
const fs = require("fs"), path = require("path"), vm = require("vm"), ort = require("onnxruntime-node");
const ROOT = path.resolve(__dirname, "..", "..", "..");
const sandbox = { window: null, ort, navigator: { hardwareConcurrency: 1 }, performance, Float32Array, BigInt64Array, BigInt, ArrayBuffer, DataView, Math, Date, Promise, Set, Object, Number, String, RegExp, Array, JSON, console,
  fetch: async (u) => ({ json: async () => JSON.parse(fs.readFileSync(path.join(ROOT, "docs", u.split("?")[0]), "utf8")) }) };
sandbox.window = sandbox; vm.createContext(sandbox);
ort.env = { wasm: {} };
for (const f of ["g2p.js", "sanskrit.js", "student_tts.js"]) vm.runInContext(fs.readFileSync(path.join(ROOT, "docs", f), "utf8"), sandbox, { filename: f });
const TEXTS = [
  ["ne", "नमस्ते, म सखिया हुँ। आजको मौसम निकै राम्रो छ।"],
  ["ne", "काठमाडौँ उपत्यकामा आज बिहानै हल्का वर्षा भयो र गाउँगाउँमा किसानहरू खुसी देखिए।"],
  ["ne", "प्रधानमन्त्रीले आज संसद्‌मा बजेट प्रस्तुत गर्दै कुल १२.५ अर्ब रुपैयाँ विनियोजन गरेको जानकारी दिए।"],
  ["sa", "ॐ भूर्भुवः स्वः । तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि । धियो यो नः प्रचोदयात् ॥"],
  ["sa", "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः । मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय ॥"]];
(async () => {
  const s = new sandbox.SakhiyaStudent();
  s.sess = await ort.InferenceSession.create(process.argv[2], { intraOpNumThreads: 1, interOpNumThreads: 1 });
  await sandbox.SK.load(); const res = []; let nan = 0;
  for (const [lang, t] of TEXTS) {
    const r = await s.say(t, { lang }); let mx = 0, ss = 0;
    for (const v of r.wav) { if (!Number.isFinite(v)) nan++; mx = Math.max(mx, Math.abs(v)); ss += v * v; }
    res.push({ lang, dur_s: +(r.wav.length / r.sr).toFixed(2), rtf_1thread: +r.rtf.toFixed(4), max_abs: +mx.toFixed(3), rms: +Math.sqrt(ss / Math.max(1, r.wav.length)).toFixed(4) });
    if (process.argv[3]) fs.writeFileSync(process.argv[3].replace(/\.json$/, `_${res.length}.f32`), Buffer.from(r.wav.buffer, r.wav.byteOffset, r.wav.byteLength));
  }
  const out = { model: process.argv[2], nan_count: nan, mean_rtf: +(res.reduce((a, b) => a + b.rtf_1thread, 0) / res.length).toFixed(4), clips: res, ok: nan === 0 && res.every(c => c.max_abs > 0.05 && c.max_abs < 1.5 && c.rms > 0.005) };
  console.log(JSON.stringify(out)); if (process.argv[3]) fs.writeFileSync(process.argv[3], JSON.stringify(out));
  process.exit(out.ok ? 0 : 2);
})().catch(e => { console.error(e); process.exit(1); });
