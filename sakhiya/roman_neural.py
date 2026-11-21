"""Python access to the context-aware Roman->Devanagari front-end (same code the browser runs, via scripts/roman_node.mjs)."""
import json, os, subprocess
ROOT = os.path.join(os.path.dirname(__file__), "..")
def convert_many(items, fast=False, node_path=None):
    """items: [(text, lang)] -> [devanagari]. fast=True = old lexicon/rule converter (baseline). Raises on failure."""
    env = dict(os.environ); 
    if node_path: env["ROMAN_ORT"] = node_path
    p = subprocess.run(["node", os.path.join(ROOT, "scripts", "roman_node.mjs")], input=json.dumps([{"text": t, "lang": l, "fast": fast} for t, l in items]),
                       capture_output=True, text=True, env=env, timeout=1800)
    if p.returncode: raise RuntimeError(p.stderr[-400:])
    return json.loads(p.stdout)
