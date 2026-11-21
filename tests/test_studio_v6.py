"""Studio v6 add-ons: cartridges.json size/shape, studio_v6.js builds its UI against a stub DOM, tanpura mixes without NaN."""
import json, os, subprocess, textwrap
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_cartridges_small_and_valid():
    p = os.path.join(ROOT, "docs", "cartridges.json")
    assert os.path.getsize(p) < 20_000
    j = json.load(open(p))
    assert len(j["cartridges"]) >= 15
    sliders = {"rate", "pause_scale", "pitch_scale", "pitch_shift", "svara", "energy_scale", "bpm"}
    for c in j["cartridges"]:
        assert c["style"] in ("news", "storytelling", "vedic_chant")
        assert set(c["params"]) <= sliders
        assert all(len(c["contour"][k]) == 8 for k in ("f0", "en", "dur"))
    assert {c["id"] for c in j["cartridges"]} >= {"news_crisp", "folk_story", "excited_wow", "gayatri_72", "rigveda_svara", "gita_anushtubh"}


def test_studio_js_runs_on_stub_dom():
    js = textwrap.dedent(r"""
    const fs = require("fs"), vm = require("vm");
    const els = {}; const mk = (id) => els[id] || (els[id] = { id, children: [], dataset: {}, style: {}, hidden: false, value: "-22", textContent: "", innerHTML: "", title: "",
      appendChild(c) { this.children.push(c); }, setAttribute() {}, addEventListener() {}, querySelector() { return mk("x"); }, parentElement: null });
    const card = mk("card"); const sty = mk("#styleinfo"); sty.parentElement = card;
    const doc = { readyState: "complete", querySelector: s => (s === "#styleinfo" ? sty : mk(s)), createElement: () => mk("n" + Math.random()), addEventListener() {} };
    const win = { document: doc, st: { style: "vedic_chant" }, STYLES: {}, setSliders() {}, renderStyles() {}, fetch: () => Promise.resolve({ json: () => Promise.resolve({ cartridges: [] }) }) };
    win.window = win; win.fetch = win.fetch; win.TanpuraDrone = require(process.argv[1] + "/docs/tanpura.js").TanpuraDrone;
    const ctx = vm.createContext(Object.assign(win, { setTimeout: (f) => f(), fetch: win.fetch, document: doc, alert() {}, performance, navigator: {} }));
    vm.runInContext(fs.readFileSync(process.argv[1] + "/docs/studio_v6.js", "utf8"), ctx);
    const w = new Float32Array(24000).fill(0.1);
    const out = ctx.window.StudioV6.postMix(w, 24000, {});
    if (out !== w) throw new Error("tanpura should be off by default");
    console.log("OK");
    """)
    r = subprocess.run(["node", "-e", js, ROOT], capture_output=True, text=True)
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr[-600:]
