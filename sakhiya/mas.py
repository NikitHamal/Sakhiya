"""Monotonic Alignment Search (Glow-TTS style, numba) on [B,L,T] log-likelihoods."""
import numpy as np, torch
from numba import njit, prange


@njit(parallel=True, cache=True)
def _mas(v, path, ls, ts):
    for b in prange(v.shape[0]):
        L, T = ls[b], ts[b]; neg = -1e9; val = v[b]
        for j in range(T):
            for i in range(max(0, L + j - T), min(L, j + 1)):
                a = val[i, j - 1] if (j > 0 and i < j) else neg
                c = val[i - 1, j - 1] if (i > 0 and j > 0) else (0.0 if (i == 0 and j == 0) else neg)
                val[i, j] += max(a, c)
        i = L - 1
        for j in range(T - 1, -1, -1):
            path[b, i, j] = 1
            if i > 0 and (i == j or val[i, j - 1] < val[i - 1, j - 1]): i -= 1


def maximum_path(logp, mask):
    """logp [B,L,T] (float), mask [B,L,T] bool -> hard 0/1 alignment on the same device."""
    dev = logp.device; lp = logp.detach().float().cpu().numpy().copy()
    ls = mask[:, :, 0].sum(1).cpu().numpy().astype(np.int64); ts = mask[:, 0, :].sum(1).cpu().numpy().astype(np.int64)
    path = np.zeros(lp.shape, dtype=np.float32); _mas(lp, path, ls, ts)
    return torch.from_numpy(path).to(dev)
