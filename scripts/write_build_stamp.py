#!/usr/bin/env python3
"""Write docs/build.json (commit SHA + generated UTC) so the site's transparency badge is never stale.
Call after every promoted checkpoint / sample run (scripts/gen_all_samples.py, promote.py) and before the push."""
import json, os, subprocess, time
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
try:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
except Exception:
    sha = ""
out = {"commit": sha, "generated_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "generated_epoch": int(time.time())}
json.dump(out, open(os.path.join(ROOT, "docs/build.json"), "w"), indent=1)
print(out)
