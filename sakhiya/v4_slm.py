"""Sakhiya-v4 Perceptual SLM Discriminator (`D_SLM`), Single-Voice Timbre Lock (`WavLM-SV`),
4-Term Composite RL/DPO Reward Scorer, and `v4-ckpt` GitHub Release Archiver (0 extra bytes in exported ONNX)."""
import json, os, re, subprocess, time, urllib.request
import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class SLMDiscriminator(nn.Module):
    """StyleTTS 2 / 2026 Paradigm: Conv1D Perceptual Discriminator operating on multi-layer WavLM hidden states."""
    def __init__(s, in_ch=768 * 4, h=256):
        super().__init__()
        s.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(in_ch, h, 1, 1, 0)),
            weight_norm(nn.Conv1d(h, h, 5, 2, 2)),
            weight_norm(nn.Conv1d(h, h, 5, 2, 2)),
            weight_norm(nn.Conv1d(h, h, 3, 1, 1)),
        ])
        s.post = weight_norm(nn.Conv1d(h, 1, 3, 1, 1))

    def forward(s, ssl_feats):
        fmap = []
        x = ssl_feats
        for c in s.convs:
            x = F.leaky_relu(c(x), 0.2)
            fmap.append(x)
        out = s.post(x)
        fmap.append(out)
        return out, fmap


class WavLMSLMHelper(nn.Module):
    """Wraps frozen `microsoft/wavlm-base-plus-sv` to provide:
    1. Multi-layer perceptual features (layers 3, 6, 9, 12) for `SLMDiscriminator`.
    2. Differentiable Speaker Verification embedding (`sv_emb`) for Single-Voice Timbre Lock Loss."""
    def __init__(s, dev="cuda"):
        super().__init__()
        from transformers import WavLMForXVector
        s.wavlm = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(dev).eval()
        for p in s.wavlm.parameters():
            p.requires_grad_(False)

    def extract(s, wav_16k):
        # Normalize 16 kHz waveform per utterance
        mu = wav_16k.mean(dim=-1, keepdim=True)
        sd = wav_16k.std(dim=-1, keepdim=True).clamp(min=1e-5)
        x = (wav_16k - mu) / sd
        out = s.wavlm(x, output_hidden_states=True)
        hs = out.hidden_states
        # Stack layers 3, 6, 9, 12 -> [B, 4*768, T_ssl]
        sel = [hs[i] for i in (3, 6, 9, 12) if i < len(hs)]
        feats = torch.cat(sel, dim=-1).transpose(1, 2)
        emb = F.normalize(out.embeddings, dim=-1)
        return feats, emb


@torch.no_grad()
def score_4term_reward(wav_24k, ids, il, ctc_rec, ctc_mean, ctc_std, slm_helper, anchor_emb, utmos_m, target_f0_std=None, dev="cuda"):
    """Computes the 4-Term Composite Reward [B] on 24 kHz waveforms:
    R_pron (MelCTCAligner) + R_spk (WavLM-SV cosine sim to sakhiya) + R_smooth (zero F0 spike penalty) + R_mos (UTMOS22)."""
    from .v2_vocoder import mel_spec_24k
    B = wav_24k.shape[0]
    mel = (mel_spec_24k(wav_24k, dev) - ctc_mean) / ctc_std
    yl = torch.full((B,), mel.shape[2], dtype=torch.long, device=dev)
    # 1. R_pron per item
    log_p = ctc_rec(mel)
    r_pron = torch.zeros(B, device=dev)
    for b in range(B):
        l_b = F.ctc_loss(log_p[:, b:b + 1], ids[b:b + 1, :il[b]], yl[b:b + 1], il[b:b + 1], blank=0, reduction="mean", zero_infinity=True)
        r_pron[b] = -l_b.clamp(0.0, 3.0)
    # 2. R_spk & 4. R_mos on 16 kHz resampled audio
    wav_16k = F.interpolate(wav_24k[:, None], size=(wav_24k.shape[1] * 2) // 3, mode="linear", align_corners=False)[:, 0]
    _, emb = slm_helper.extract(wav_16k)
    r_spk = F.cosine_similarity(emb, anchor_emb.expand(B, -1), dim=-1)
    r_mos = utmos_m(wav_16k, 16000).view(B)
    # 3. R_smooth: penalize frame-to-frame low-band spectral centroid / F0 jumps
    bins = torch.arange(20, device=dev, dtype=torch.float32)[None, :, None]
    w_f0 = torch.softmax(mel[:, 2:22, :].float() * 2.0, dim=1)
    c_f0 = (w_f0 * bins).sum(1)
    jump = (c_f0[:, 1:] - c_f0[:, :-1]).abs().amax(dim=-1)
    r_smooth = -0.35 * F.relu(jump - 1.8)
    if target_f0_std is not None:
        f0_sd = c_f0.std(dim=-1)
        r_smooth = r_smooth - 0.20 * (f0_sd - target_f0_std).abs()
    tot = 1.25 * r_pron + 2.50 * r_spk + 1.00 * r_smooth + 0.80 * r_mos
    return tot, dict(r_pron=r_pron.mean().item(), r_spk=r_spk.mean().item(), r_smooth=r_smooth.mean().item(), r_mos=r_mos.mean().item())


def archive_v4_release(am_p, voc_p, state_p):
    """Idempotently archives fp16 checkpoints + state JSON to GitHub Release `v4-ckpt` for session survival."""
    if not os.path.exists("/tmp/gitenv.sh"):
        return
    tok = re.search(r"echo (\S+)", open("/tmp/gitenv.sh").read())[1]
    h = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json", "User-Agent": "curl/8.5.0"}
    try:
        rel = json.load(urllib.request.urlopen(urllib.request.Request("https://api.github.com/repos/NikitHamal/Sakhiya/releases/tags/v4-ckpt", headers=h), timeout=30))
    except Exception:
        b = json.dumps(dict(tag_name="v4-ckpt", name="Sakhiya v4 Checkpoint Release", body="Archived Sakhiya v4 fp16 checkpoints & resume state.", prerelease=False)).encode()
        rel = json.load(urllib.request.urlopen(urllib.request.Request("https://api.github.com/repos/NikitHamal/Sakhiya/releases", data=b, method="POST", headers=h), timeout=30))
    for name, src in (("v4_am.fp16.pt", am_p), ("v4_voc.fp16.pt", voc_p), ("v4_state.json", state_p)):
        if not src or not os.path.exists(src):
            continue
        for a in rel.get("assets", []):
            if a["name"] == name:
                subprocess.run(["curl", "-s", "-X", "DELETE", "-H", f"Authorization: Bearer {tok}", a["url"]])
        u = f"https://uploads.github.com/repos/NikitHamal/Sakhiya/releases/{rel['id']}/assets?name={name}"
        subprocess.run(["curl", "-s", "-X", "POST", "-H", f"Authorization: Bearer {tok}", "-H", "Content-Type: application/octet-stream", "--data-binary", f"@{src}", u], capture_output=True)
    print(f"[v4-ckpt] Archived checkpoints to GitHub Release v4-ckpt at {time.strftime('%H:%M:%S')}", flush=True)
