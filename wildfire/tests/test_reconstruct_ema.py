import numpy as np
import torch

from wildfire.paper.reconstruct import fit


def _data():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 255, (24, 12, 4, 13, 13), dtype=np.uint8)
    y = rng.integers(0, 2, 24)
    idx = {"train": np.arange(16), "val": np.arange(16, 24)}
    return X, y, idx, np.full(4, 127.0, np.float32), np.full(4, 50.0, np.float32)   # wie cache_norm


def test_fit_with_ema_runs_and_returns_averaged_weights():
    X, y, idx, mean, std = _data()
    kw = dict(chs=[0, 1, 2, 3], t0=2, t1=10, mean=mean, std=std, device=torch.device("cpu"), seed=0,
              epochs=3, patience=10, bs=8, lr=1e-2)
    net0, h0 = fit(X, y, idx, **kw)
    net1, h1 = fit(X, y, idx, ema_decay=0.9, **kw)
    assert len(h0) == len(h1) == 3
    w0, w1 = net0.head.weight.detach(), net1.head.weight.detach()
    assert torch.isfinite(w1).all() and not torch.allclose(w0, w1)   # gleicher Seed, andere (gemittelte) Gewichte
