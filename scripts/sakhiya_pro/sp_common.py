"""Shared metrics for Sakhiya-Pro: Kriti Nepali CTC ASR, UTMOS22, ECAPA similarity, click/pause forensics."""
import os, re, sys, json, numpy as np, torch, librosa, soundfile as sf
SW = '/work/sp/models/swar'
_cache = {}
def kriti():
    if 'k' not in _cache:
        sys.path.insert(0, SW + '/app/pipeline')
        from kriti_onnx import KritiONNX
        _cache['k'] = KritiONNX(SW + '/kriti_onnx')
    return _cache['k']
def utmos():
    if 'u' not in _cache:
        if not os.path.isdir('/work/sp/SpeechMOS'):
            import subprocess
            subprocess.run(['git', 'clone', '-q', 'https://github.com/tarepan/SpeechMOS.git', '/work/sp/SpeechMOS'], check=False)
        _cache['u'] = torch.hub.load('/work/sp/SpeechMOS', 'utmos22_strong', source='local', trust_repo=True).cuda().eval()
    return _cache['u']
def utmos_score(w, sr):
    w16 = librosa.resample(w.astype(np.float32), orig_sr=sr, target_sr=16000)
    with torch.no_grad():
        return float(utmos()(torch.from_numpy(w16)[None].cuda(), 16000).item())
def spk():
    if 's' not in _cache:
        import torchaudio
        if not hasattr(torchaudio, 'list_audio_backends'):
            torchaudio.list_audio_backends = lambda: ['soundfile']
        from speechbrain.inference.speaker import EncoderClassifier
        _cache['s'] = EncoderClassifier.from_hparams('speechbrain/spkrec-ecapa-voxceleb', savedir='/work/sp/ecapa', run_opts={'device': 'cuda'})
    return _cache['s']
def spk_emb(w, sr):
    w16 = librosa.resample(w.astype(np.float32), orig_sr=sr, target_sr=16000)
    with torch.no_grad():
        e = spk().encode_batch(torch.from_numpy(w16)[None].cuda())[0, 0]
    return torch.nn.functional.normalize(e, dim=0).cpu().numpy()
_P = re.compile(r'[^\u0900-\u097F\u0020a-zA-Z0-9]')
def norm_txt(t):
    t = re.sub(r'[।॥,?!.\-–—:;"\'()]', ' ', t)
    t = t.replace('\u200d', '').replace('\u200c', '')
    return re.sub(r'\s+', ' ', t).strip()
def cer(ref, hyp):
    import jiwer
    r, h = norm_txt(ref).replace(' ', ''), norm_txt(hyp).replace(' ', '')
    return float(jiwer.cer(r, h)) if r else 1.0
def wer(ref, hyp):
    import jiwer
    return float(jiwer.wer(norm_txt(ref), norm_txt(hyp) or ' '))
HINDI = ['है', 'हैं', 'में', 'मैं', 'नहीं', 'था', 'थी', 'और', 'का', 'की', 'हूँ', 'कर']
def hindi_leak(txt):
    ws = norm_txt(txt).split()
    return sum(w in ('है', 'हैं', 'में', 'मैं', 'नहीं', 'थी', 'और', 'हूँ', 'किया', 'करके') for w in ws)
def forensic(w, sr):
    w = w.astype(np.float32)
    pk = float(np.abs(w).max())
    d = np.abs(np.diff(w)); thr = max(0.25, 12 * np.median(d) + 1e-9)
    clicks = int((d > max(0.35 * max(pk, 1e-3), 0.12)).sum())
    iv = librosa.effects.split(w, top_db=38, frame_length=1024, hop_length=256)
    gaps = [(iv[i + 1][0] - iv[i][1]) / sr for i in range(len(iv) - 1)]
    breaths = [g for g in gaps if 0.15 <= g <= 0.9]
    long_gaps = [g for g in gaps if g > 1.0]
    return dict(dur=len(w) / sr, peak=pk, clicks=clicks, pauses=len(breaths), mean_pause=float(np.mean(breaths)) if breaths else 0.0, long_gaps=len(long_gaps),
                lead_sil=float(iv[0][0] / sr) if len(iv) else 0.0)
def score_take(path, ref_text, ref_emb=None, asr=True):
    w, sr = sf.read(path, dtype='float32')
    if w.ndim > 1: w = w.mean(1)
    f = forensic(w, sr)
    f['utmos'] = utmos_score(w, sr)
    if asr:
        hyp, conf = kriti().read_conf([path])[0]
        f.update(hyp=hyp, asr_conf=conf, cer=cer(ref_text, hyp), wer=wer(ref_text, hyp), hindi=hindi_leak(hyp))
    if ref_emb is not None:
        f['spk_sim'] = float(np.dot(spk_emb(w, sr), ref_emb))
    nsyl = max(1, len(re.findall(r'[क-ह]', ref_text)))
    f['cps'] = nsyl / max(0.1, f['dur'])
    return f
