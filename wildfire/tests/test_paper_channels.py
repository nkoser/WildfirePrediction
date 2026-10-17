import numpy as np
import torch

from wildfire.paper.channels import CHANNELS, has_fire, keep_mask, masked_std, summary_columns, tag
from wildfire.paper.compare import summaries
from wildfire.paper.prepare import FIRE_CH
from wildfire.paper.stdctm import batch


def test_channel_sets():
    assert len(CHANNELS["paper55"]) == 55 and len(CHANNELS["no_fire"]) == 54
    assert FIRE_CH not in CHANNELS["no_modis"] and not any(38 <= c <= 53 for c in CHANNELS["no_modis"])
    assert has_fire("paper55") and not has_fire("no_fire") and not has_fire("no_modis")
    assert tag("paper55") == "" and tag("no_fire") == "_no_fire"


def test_summary_columns_select_the_same_channels_in_every_block():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 255, size=(4, 30, 55, 13, 13), dtype=np.uint8)
    full = summaries(X, np.arange(4), 0, 30)
    chs = CHANNELS["no_modis"]
    cols = summary_columns("no_modis")
    assert cols.size == 5 * len(chs)
    # Block k, Kanal c steht in Spalte k * 55 + c
    s = X[:, :, :, 6, 6].astype(np.float32)
    ref = np.concatenate([s[:, -1][:, chs], s[:, -7:].mean(1)[:, chs], s.mean(1)[:, chs],
                          s.min(1)[:, chs], s.max(1)[:, chs]], 1)
    np.testing.assert_allclose(full[:, cols], ref, rtol=1e-6)


def test_loader_gives_the_same_batches_from_numpy_and_from_a_tensor():
    from wildfire.paper.reconstruct import loader
    rng = np.random.default_rng(2)
    X = rng.integers(0, 255, size=(20, 61, 55, 13, 13), dtype=np.uint8)
    mean, std = rng.uniform(50, 150, 55).astype(np.float32), rng.uniform(5, 30, 55).astype(np.float32)
    idx, chs, dev = np.arange(3, 17), CHANNELS["no_modis"], torch.device("cpu")
    a = list(loader(X, idx, chs, 29, 59, mean, std, 4, dev, np.random.default_rng(0)))
    b = list(loader(torch.from_numpy(X), idx, chs, 29, 59, mean, std, 4, dev, np.random.default_rng(0)))
    assert len(a) == len(b) == 4
    for (sa, xa), (sb, xb) in zip(a, b):
        np.testing.assert_array_equal(sa, sb)
        assert xa.shape == (len(sa), len(chs), 30, 13, 13)
        torch.testing.assert_close(xa, xb)


def test_masked_std_zeroes_dropped_channels_after_normalisation():
    rng = np.random.default_rng(1)
    X = rng.integers(0, 255, size=(2, 61, 55, 13, 13), dtype=np.uint8)
    mean, std = np.full(55, 100, np.float32), np.full(55, 20, np.float32)
    xb = batch(X, np.array([0, 1]), np.array([1, 30]), 61, mean, masked_std(std, "no_fire"), torch.device("cpu"))
    assert torch.all(xb[:, :, FIRE_CH] == 0)
    kept = torch.from_numpy(keep_mask("no_fire"))
    assert torch.all(torch.isfinite(xb)) and xb[:, :, kept].abs().sum() > 0
