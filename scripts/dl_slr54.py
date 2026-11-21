"""Download + extract the 16 OpenSLR-54 shards (CC-BY-SA 4.0) in parallel from the ELDA mirror -> /work/tts/data/slr54 (resumable)."""
import os, zipfile, urllib.request, concurrent.futures as cf
D = "/work/tts/data/slr54"; os.makedirs(D, exist_ok=True); U = "https://openslr.elda.org/resources/54/"
def shard(c):
    mk = f"{D}/.done_{c}"
    if os.path.exists(mk): return c
    z = f"{D}/asr_nepali_{c}.zip"
    for k in range(4):
        try: urllib.request.urlretrieve(U + f"asr_nepali_{c}.zip", z); zipfile.ZipFile(z).extractall(D); os.remove(z); open(mk, "w").close(); print("shard", c, flush=True); return c
        except Exception as e: print("retry", c, k, e, flush=True)
if not os.path.exists(f"{D}/utt_spk_text.tsv"): urllib.request.urlretrieve(U + "utt_spk_text.tsv", f"{D}/utt_spk_text.tsv")
with cf.ThreadPoolExecutor(8) as ex: list(ex.map(shard, "0123456789abcdef"))
print("SLR54DL_DONE", flush=True)
