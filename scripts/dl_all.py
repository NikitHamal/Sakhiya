"""Download every open corpus used for training (run on the GPU box). SLR43 CC-BY-SA, FLEURS ne_np CC-BY-4.0, CommonVoice ne-NP CC0."""
import os, tarfile, zipfile, subprocess, threading, urllib.request
from huggingface_hub import hf_hub_download as hd, list_repo_files
D = "/work/tts/data"; os.makedirs(D, exist_ok=True)


def slr43():
    z = f"{D}/ne_np_female.zip"
    if not os.path.exists(f"{D}/slr43/ne_np_female/line_index.tsv"):
        for host in ("openslr.elda.org", "www.openslr.org"):   # www.openslr.org throttles to ~30 KB/s from the box; the ELDA mirror is ~100x faster
            try: urllib.request.urlretrieve(f"https://{host}/resources/43/ne_np_female.zip", z); break
            except Exception as e: print("mirror failed", host, e, flush=True)
        os.makedirs(f"{D}/slr43", exist_ok=True); zipfile.ZipFile(z).extractall(f"{D}/slr43"); os.remove(z)
    print("SLR43 ok", flush=True)


def fleurs():
    if all(os.path.isdir(f"{D}/fleurs/{s}") for s in ("train", "dev", "test")): print("FLEURS cached", flush=True); return
    for s in ("train", "dev", "test"):
        for f in (f"data/ne_np/{s}.tsv", f"data/ne_np/audio/{s}.tar.gz"):
            p = hd("google/fleurs", f, repo_type="dataset", local_dir=f"{D}/fleurs")
            if p.endswith("tar.gz") and not os.path.isdir(f"{D}/fleurs/{s}"):
                tarfile.open(p).extractall(f"{D}/fleurs/{s}", filter="data"); os.remove(p)
    print("FLEURS ok", flush=True)


def cv():
    if os.path.isdir(f"{D}/cv") and any(f.endswith(".tsv") for _, _, fs in os.walk(f"{D}/cv") for f in fs): print("CV cached", flush=True); return
    for f in list_repo_files("fsicoli/common_voice_22_0", repo_type="dataset"):
        if "ne-NP" in f and "/invalidated" not in f and (f.endswith(".tar") or f.endswith(".tsv")):
            p = hd("fsicoli/common_voice_22_0", f, repo_type="dataset", local_dir=f"{D}/cv")
            if p.endswith(".tar"): tarfile.open(p).extractall(os.path.dirname(p), filter="data"); os.remove(p)
    print("CV ok", flush=True)


ts = [threading.Thread(target=f) for f in (slr43, fleurs, cv)]
[t.start() for t in ts]; [t.join() for t in ts]; print("DLDONE", flush=True)
