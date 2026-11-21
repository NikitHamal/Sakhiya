"""Waits for a wave to finish (logs/wave<TAG>.log contains WAVE<TAG>_DONE), then records its real gate result + metrics in STATUS.md and LEDGER.json and pushes.
usage: python3 scripts/finalize_wave.py 12 [session_id]   (run on the GPU box from the repo dir; nothing is written before the gate has run)"""
import json, os, re, subprocess, sys, time
tag = sys.argv[1]; sid = sys.argv[2] if len(sys.argv) > 2 else "12"
R = os.environ.get("SAKHIYA_ROOT", "/work/tts"); L = f"{R}/logs"; repo = f"{R}/repo"; log = f"{L}/wave{tag}.log"
while not (os.path.exists(log) and f"WAVE{tag}_DONE" in open(log).read()): time.sleep(15)
gate = open(f"{L}/promote{tag}.log").read().strip().splitlines()[-1]
M = json.load(open(f"{repo}/eval/metrics.json")); S = {s["system"]: s for s in M["systems"]}
f, t, h = S["Sakhiya-fp16"], S["Sakhiya-tiny-int8"], S["Human recording (ceiling)"]
piper = next((s for s in M["systems"] if s["system"].lower().startswith("piper")), None)
fmt = lambda v, n=3: "n/a" if v is None else f"{v:.{n}f}"
row = lambda s, k: f"pCER {fmt(s[k]['pcer'])}, CER {fmt(s[k]['cer'])}, WER {fmt(s[k]['wer'])}, UTMOS22 {fmt(s[k]['mos'], 2)}, MCD {fmt(s[k]['mcd'], 2)}, F0 corr {fmt(s[k]['f0corr'], 2)}"
am = json.load(open(f"{R}/runs/am{tag}/progress.json")) if os.path.exists(f"{R}/runs/am{tag}/progress.json") else {}
sec = (f"\n## Session {sid} — wave {tag} (gate: `{gate}`)\n"
       f"- Warm-start continuation wave (time-boxed bf16 AM + GT-mel vocoder concurrently, then GTA vocoder; exact minutes in `scripts/wave{tag}.sh`) on the ~37 h two-voice teacher corpus (`edge_f` 18.7 h, `edge_m` 18.4 h after the ASR gate). AM step {am.get('step', 'n/a')}.\n"
       f"- `Sakhiya-fp16` ({f['size_mb']} MB) sentences: {row(f, 'sentence')}, DNSMOS {fmt(f.get('dnsmos'), 2)}, RTF {fmt(f['sentence']['rtf'], 4)}; paragraphs: {row(f, 'paragraph')}.\n"
       f"- `Sakhiya-tiny-int8` ({t['size_mb']} MB) sentences: {row(t, 'sentence')}; paragraphs: {row(t, 'paragraph')}.\n"
       f"- References: human recordings UTMOS22 {fmt(h['sentence']['mos'], 2)}, pCER {fmt(h['sentence']['pcer'])}" + (f"; {piper['system']} UTMOS22 {fmt(piper['sentence']['mos'], 2)}, pCER {fmt(piper['sentence']['pcer'])}." if piper else ".") + "\n"
       "- Gate semantics: PROMOTE = models/samples/ckpt/metrics replaced; REVERT = previous shipped models restored (numbers above are then the *candidate's*, kept only for honesty in `/work/tts/eval" + tag + "`).\n")
txt = open(f"{repo}/STATUS.md").read(); i = txt.index("\n## ") if "\n## " in txt else len(txt)
open(f"{repo}/STATUS.md", "w").write(txt[:i] + "\n" + sec + txt[i:])
led = json.load(open(f"{repo}/LEDGER.json")); led["metrics"][f"wave{tag}"] = dict(gate=gate, fp16_pcer=f["sentence"]["pcer"], fp16_utmos=f["sentence"]["mos"], fp16_wer=f["sentence"]["wer"], fp16_f0corr=f["sentence"]["f0corr"],
    tiny_pcer=t["sentence"]["pcer"], tiny_utmos=t["sentence"]["mos"], size_mb_fp16=f["size_mb"], size_mb_tiny=t["size_mb"])
json.dump(led, open(f"{repo}/LEDGER.json", "w"), indent=1, ensure_ascii=False)
subprocess.run(f"cd {repo} && export GIT_ASKPASS=/tmp/gitenv.sh GIT_TERMINAL_PROMPT=0 && git checkout -- docs/data/status.json; git add STATUS.md LEDGER.json && git commit -qm 'session {sid}: wave {tag} results recorded ({gate.split()[-1]})' && git pull -q --rebase --autostash -X ours origin main; git push -q origin HEAD:main", shell=True)
print("FINALIZED", tag, gate, flush=True)
