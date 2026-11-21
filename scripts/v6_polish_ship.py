#!/usr/bin/env python3
"""Fast-track shipper for the v6 RSI polish cycles (runs alongside scripts/rsi_v6_daemon.py, never edits it).

At --deadline (epoch seconds) it waits for the daemon to cross a cycle boundary (state `right_now` changes, meaning the
previous cycle's checkpoints were archived and no torch.save is in flight), stops the daemon by exact PID, then exports
Voca + Sano from the newest student checkpoints, gates them, freezes immutable `*_polish` packs and pushes to GitHub Pages.
"""
import argparse, json, os, signal, subprocess, sys, time

R = "/work/tts"
REPO = f"{R}/repo"
sys.path.insert(0, REPO)
from scripts.rsi_v6_daemon import export_gate_and_ship  # noqa: E402


def state():
    try:
        return json.load(open(f"{R}/rsi_v5_state.json")).get("right_now", "")
    except Exception:
        return ""


def daemon_pids():
    out = subprocess.run(["pgrep", "-f", "scripts/rsi_v6_daemon.py"], capture_output=True, text=True).stdout.split()
    return [int(p) for p in out if int(p) != os.getpid()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline", type=float, required=True)
    a = ap.parse_args()
    while time.time() < a.deadline and daemon_pids():
        time.sleep(5)
    pids = daemon_pids()
    if pids:  # stop only at a cycle boundary so no checkpoint is half-written
        s0 = state()
        t0 = time.time()
        while state() == s0 and time.time() - t0 < 420 and daemon_pids():
            time.sleep(1)
        time.sleep(2)
        for p in daemon_pids():
            os.kill(p, signal.SIGTERM)
        time.sleep(3)
    print("daemon stopped; shipping polish packs", flush=True)
    rr = f"{R}/runs"
    sv, ss = f"{rr}/v6_voca_4m", f"{rr}/v6_sano_1p5m"
    t_am = f"{rr}/v6_teacher/am.pt"
    for tier, vtag, d, mdir, lbl in [
        ("voca", "v6_voca_4m_polish", sv, f"{REPO}/docs/models_voca", "v6-Voca polish (4.3 MB int8 · RSI cycles)"),
        ("sano", "v6_sano_1p5m_polish", ss, f"{REPO}/docs/models_sano", "v6-Sano polish (1.7 MB int8 · RSI cycles)"),
    ]:
        try:
            vl = f"{rr}/v6_{'voca' if tier == 'voca' else 'sano'}_voclong/voc.pt"
            voc_p = vl if os.path.exists(vl) else f"{d}/voc.pt"  # prefer the long wall-clock-trained vocoder
            print(f"{tier}: voc={voc_p}", flush=True)
            export_gate_and_ship(tier, f"{d}/am.pt", voc_p, mdir, vtag, lbl, f"Overnight RSI polish of {tier} student; teacher={os.path.basename(os.path.dirname(t_am))}")
        except Exception as e:  # keep going so the other tier still ships
            print(f"ship {vtag} failed: {e}", flush=True)
    print("polish ship done", flush=True)


if __name__ == "__main__":
    main()
