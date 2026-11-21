// Node bridge to the browser front-end: reads JSON [{text,lang}] on stdin, prints JSON [normalised Devanagari].
// Runs the *same* docs/g2p.js + docs/roman.js + docs/translit/ the web studio uses.   node scripts/roman_node.mjs < in.json
import fs from "node:fs"; import vm from "node:vm"; import path from "node:path"; import { createRequire } from "node:module"; import { fileURLToPath } from "node:url";
const D = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "docs");
const require = createRequire(process.env.ROMAN_ORT || import.meta.url); const ort = require("onnxruntime-node");
const rd = (n) => (n.endsWith(".onnx") ? new Uint8Array(fs.readFileSync(path.join(D, "translit", n))) : fs.readFileSync(path.join(D, "translit", n), "utf8"));
const ctx = { console, fetch: async (u) => ({ ok: true, json: async () => JSON.parse(fs.readFileSync(path.join(D, u.split("?")[0]), "utf8")) }), TextDecoder, URLSearchParams, Promise, ort,
  LOADER: async (n) => rd(n), importer: (u) => import(u) };
ctx.window = ctx; vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(D, "g2p.js"), "utf8"), ctx, { filename: "g2p.js" });
vm.runInContext(fs.readFileSync(path.join(D, "roman.js"), "utf8").replace('import("./translit/tlm.js")', `importer("file://${D}/translit/tlm.js")`)
  .replace('base: "translit"', "loader: LOADER").replace('["wasm"]', '["cpu"]'), ctx);
await ctx.SK.load();
const inp = JSON.parse(fs.readFileSync(0, "utf8")), out = [];
for (const r of inp) out.push(r.fast ? ctx.SK.normalize(r.text, r.lang) : await ctx.SK.normalizeAsync(r.text, r.lang));
process.stdout.write(JSON.stringify(out));
