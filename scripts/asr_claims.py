#!/usr/bin/env python3
"""Shared claim ledger so ONE clip is never transcribed by two cloud engines (owner rule: Gemini and AssemblyAI never on the same clip;
Groq clips are separate too).  SQLite (WAL) works across processes: claim() is an atomic INSERT OR IGNORE.
seed(): every clip Gemini already finished is claimed by 'gemini' so nobody repeats it."""
import json, os, sqlite3, time
ROOT = os.environ.get("TTS_ROOT", "/work/tts"); DB = f"{ROOT}/asr/claims.db"

def con():
    c = sqlite3.connect(DB, timeout=30, isolation_level=None); c.execute("PRAGMA journal_mode=WAL")
    c.execute("CREATE TABLE IF NOT EXISTS claims(id TEXT PRIMARY KEY, provider TEXT, state TEXT, ts REAL)"); return c

def claim(cid, provider):
    """True if this provider now owns the clip"""
    c = con(); cur = c.execute("INSERT OR IGNORE INTO claims VALUES(?,?,?,?)", (cid, provider, "claimed", time.time())); c.close()
    return cur.rowcount == 1

def retarget(cid, old, new):
    c = con(); cur = c.execute("UPDATE claims SET provider=?, ts=? WHERE id=? AND provider=?", (new, time.time(), cid, old)); c.close(); return cur.rowcount == 1

def finish(cid, provider):
    c = con(); c.execute("UPDATE claims SET state='done', ts=? WHERE id=? AND provider=?", (time.time(), cid, provider)); c.close()

def release(cid, provider):
    c = con(); c.execute("DELETE FROM claims WHERE id=? AND provider=? AND state!='done'", (cid, provider)); c.close()

def reclaim_stale(max_age=900):
    """claims left 'claimed' by a crashed worker go back to the pool"""
    c = con(); cur = c.execute("DELETE FROM claims WHERE state='claimed' AND ts<?", (time.time() - max_age,)); c.close(); return cur.rowcount

def seed(files=(("asr/gemini.jsonl", "gemini"),)):
    c = con(); n = 0
    for f, prov in files:
        try:
            for l in open(f"{ROOT}/{f}"):
                try: r = json.loads(l)
                except Exception: continue
                if r.get("text"): n += c.execute("INSERT OR IGNORE INTO claims VALUES(?,?,?,?)", (r["id"], r.get("provider", prov), "done", time.time())).rowcount
        except FileNotFoundError: pass
    c.close(); return n

def claimed_ids():
    c = con(); r = {x[0] for x in c.execute("SELECT id FROM claims")}; c.close(); return r

def stats():
    c = con(); rows = c.execute("SELECT provider,state,COUNT(*) FROM claims GROUP BY 1,2").fetchall(); c.close(); return rows

if __name__ == "__main__":
    print("seeded", seed()); print(stats())
