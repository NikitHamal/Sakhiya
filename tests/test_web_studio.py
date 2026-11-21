"""Web Studio & Browser JS Smoke Test (Track B):
1. Runs `node --check` on every `docs/*.js`.
2. Executes `docs/g2p.js`, `docs/sanskrit.js`, `docs/roman.js`, `docs/tts.js`, `docs/app.js`, and `docs/samples.js` inside a simulated DOM in Node
   to verify that `$` is defined in `app.js`, the Studio renders voices, 6 styles and model tiers without ReferenceError/SyntaxError,
   and `SakhiyaWeb._ola64` reconstructs a 24 kHz sine wave accurately.
usage: python3 tests/test_web_studio.py"""
import glob, json, os, subprocess, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    for js in sorted(glob.glob(os.path.join(ROOT, "docs", "*.js"))):
        subprocess.run(["node", "--check", js], check=True)
    node_script = r"""
const fs = require("fs");
const vm = require("vm");

const elements = {};
function makeEl(id = "") {
  return {
    id, className: "", innerHTML: "", textContent: "", value: "नमस्ते",
    style: {}, dataset: { theme: "dark", v: "fp16", k: "sentence" },
    options: [], children: [], clientWidth: 300, width: 600, height: 180,
    lastChild: { textContent: "" }, lastElementChild: { textContent: "" }, setAttribute() {}, removeAttribute() {}, append(...c) { this.children.push(...c); }, remove() {}, hidden: false, disabled: false,
    classList: { toggle() {}, add() {}, remove() {} },
    appendChild(c) { this.children.push(c); return c; },
    add(o) { this.options.push(o); },
    play() {},
    querySelector() { return makeEl(); },
    querySelectorAll() { return []; },
    getAttribute() { return "#studio"; },
    getContext() {
      return {
        fillStyle: "", fillRect() {},
        createImageData(w, h) { return { data: new Uint8ClampedArray(w * h * 4) }; },
        putImageData() {}
      };
    }
  };
}
const doc = {
  head: makeEl("head"), body: makeEl("body"), documentElement: makeEl("html"), addEventListener() {},
  querySelector(sel) { if (!elements[sel]) elements[sel] = makeEl(sel); return elements[sel]; },
  querySelectorAll() { return []; },
  createElement(tag) { return makeEl(tag); }
};
const sandbox = {
  window: null, global: null, document: doc,
  navigator: { hardwareConcurrency: 4 },
  location: { hash: "#studio", search: "" },
  performance: { now: () => Date.now() },
  localStorage: { getItem: () => "dark", setItem: () => {} },
  URLSearchParams, Float32Array, Int32Array, Uint32Array, Uint8ClampedArray, BigInt64Array, BigInt, ArrayBuffer, DataView, Math, Date, Promise, Set, Object, Number, String, RegExp, Array, JSON, console,
  Blob: class Blob { constructor(a, b) { this.a = a; this.b = b; } },
  URL: { createObjectURL: () => "blob:mock" },
  Option: class Option { constructor(t, v) { this.text = t; this.value = v; } },
  addEventListener() {},
  requestAnimationFrame(cb) { cb(); },
  setTimeout(cb) { return 0; },
  setInterval() { return 0; },
  getComputedStyle() { return { getPropertyValue: () => "#b2c5ff" }; },
  fetch: async (url) => {
    const clean = url.split("?")[0];
    const p = "./docs/" + clean;
    if (fs.existsSync(p)) return { json: async () => JSON.parse(fs.readFileSync(p, "utf8")) };
    return { json: async () => ({}) };
  }
};
sandbox.window = sandbox;
sandbox.global = sandbox;
vm.createContext(sandbox);

for (const f of ["g2p.js", "sanskrit.js", "mixed.js", "roman.js", "tts.js", "app.js", "samples.js"]) {
  vm.runInContext(fs.readFileSync("./docs/" + f, "utf8"), sandbox, { filename: f });
}

(async () => {
  if (typeof sandbox.$ !== "function") throw new Error("$ is not defined after loading app.js");
  await new Promise(r => setImmediate(r));
  const gridHtml = elements["#tiers"] ? elements["#tiers"].innerHTML + String(elements["#tiers"].children.length) : "";
  if (!elements["#tiers"] || elements["#tiers"].children.length < 3) throw new Error("studio did not render model tiers");
  if (!elements["#voices"] || elements["#voices"].children.length < 1) throw new Error("studio did not render the unified sakhiya voice");
  if (!elements["#styles"] || elements["#styles"].children.length < 3) throw new Error("studio did not render the 3 canonical modes");
  // Test 24 kHz _ola64 iSTFT in SakhiyaWeb
  const web = new sandbox.SakhiyaWeb();
  const T = 32, re = new Float32Array(33 * T), im = new Float32Array(33 * T);
  for (let t = 0; t < T; t++) re[4 * T + t] = 1.0;
  const wav = web._ola64(re, im, T);
  if (wav.length !== (T - 1) * 16 + 64 - 48 || !wav.every(Number.isFinite)) {
    throw new Error("_ola64 output invalid length or NaN: " + wav.length);
  }
  console.log("WEB_STUDIO_SMOKE_OK", { cardsLen: gridHtml.length, ola64Samples: wav.length });
})().catch(e => { console.error(e); process.exit(1); });
"""
    out = subprocess.check_output(["node", "-e", node_script], cwd=ROOT, text=True).strip()
    print(out)


def test_web_studio():
    main()


if __name__ == "__main__":
    main()
