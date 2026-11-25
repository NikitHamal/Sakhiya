import torch
from sakhiya import g2p
from sakhiya.symbols import SYM2ID
from sakhiya.news_vocoder_100mel import (
    StudioVocoder100Mel,
    coherence_masked_awp_losses,
)
from sakhiya.news_am_100mel import (
    SakhiyaNewsFlagship,
    compute_graph_structural_features,
    compute_syllables_since_last_breath,
    BREATH_ID,
)


def test_leg2_dual_source_ola_noise_and_creak_vocoder():
    voc = StudioVocoder100Mel(n_mels=100, ch=320).eval()
    mel = torch.randn(1, 100, 16)
    f0 = torch.full((1, 1, 16), 195.0)
    vuv = torch.ones((1, 1, 16))
    shr = torch.full((1, 1, 16), 0.35)
    with torch.no_grad():
        y1, re1, im1, lmag1, cp1, sp1 = voc.forward_wav(mel, f0, vuv, shr_creak=shr)
        y2, _, _, _, _, _ = voc.forward_wav(mel, f0, vuv, shr_creak=shr)
    assert y1.shape == (1, 16 * 256)
    # Deterministic OLA unit-phasor noise in eval mode guarantees bit-exact reproducibility
    assert torch.allclose(y1, y2, atol=1e-6)


def test_coherence_masked_awp_zero_on_unvoiced():
    cos_p = torch.randn(2, 33, 32, requires_grad=True)
    sin_p = torch.randn(2, 33, 32, requires_grad=True)
    re_gt = torch.randn(2, 33, 32)
    im_gt = torch.randn(2, 33, 32)
    vuv_zero = torch.zeros(2, 1, 32)
    loss_uv = coherence_masked_awp_losses(cos_p, sin_p, re_gt, im_gt, vuv_16=vuv_zero)
    assert float(loss_uv.item()) == 0.0


def test_coupled_prosody_flow_and_cfg0_flagship():
    m = SakhiyaNewsFlagship(steps=2, cfg_scale=0.0).eval()
    m.enc.spk_film.weight.data.fill_(0.05)
    m.enc.studio_24k_film.weight.data.fill_(0.03)
    m.enc.fold_studio_anchor_film()
    assert float(m.enc.spk_film.weight.abs().max().item()) == 0.0
    assert float(m.enc.studio_24k_film.weight.abs().max().item()) == 0.0

    ids = torch.tensor([[1, 15, 45, 8, 22, 46, 5]], dtype=torch.long)
    syl = compute_syllables_since_last_breath(ids)
    assert syl.shape == (1, 1, 7)
    with torch.no_grad():
        wav = m.infer(ids)
    assert wav.dim() == 2 and wav.shape[0] == 1 and wav.shape[1] > 256


def test_nepali_loanword_geminate_sibilant_g2p():
    for w in ("कति", "कत्ति", "पकै", "पक्कै", "चिटिक्क", "ठ्ठयाक्कै", "सफा", "चिसो", "सास", "शान्ति", "छाता", "डिजिटल", "सफ्टवेयर", "कम्प्युटर"):
        ph = g2p.word_to_phones(w, "ne")
        assert len(ph) >= 3 and all(p in SYM2ID for p in ph)
    assert BREATH_ID == SYM2ID[";"]
