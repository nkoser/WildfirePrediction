"""Die Batch-Funktionen liefern aus dem numpy-Array und aus dem Tensor (GPU-Cache) dasselbe.

Ein CPU-Tensor nimmt hier den Platz der GPU-Kopie ein; der Code-Pfad ist derselbe.
"""

import numpy as np
import torch

from wildfire.paper import compare, pixel_stdctm, stdctm
from wildfire.paper.reconstruct import gather_windows

DEV = torch.device("cpu")


def _data(seed=0, n=12):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 255, size=(n, 61, 55, 13, 13), dtype=np.uint8)
    mean, std = rng.uniform(50, 150, 55).astype(np.float32), rng.uniform(5, 30, 55).astype(np.float32)
    return X, torch.from_numpy(X), mean, std


def test_gather_windows_numpy_and_tensor():
    X, Xt, _, _ = _data()
    rows, starts = np.array([3, 0, 7, 3]), np.array([0, 31, 12, 29])
    a = gather_windows(X, rows, starts, 30)
    b = gather_windows(Xt, rows, starts, 30)
    assert a.shape == (4, 30, 55, 13, 13)
    np.testing.assert_array_equal(a, b.numpy())
    np.testing.assert_array_equal(a[2], X[7, 12:42])


def test_compare_pair_batch_numpy_and_tensor():
    X, Xt, mean, std = _data(1)
    rows, ks = np.array([1, 5, 5, 9]), np.array([0, 3, 1, 2])
    a = compare.pair_batch(X, rows, ks, 61, mean, std, DEV)
    b = compare.pair_batch(Xt, rows, ks, 61, mean, std, DEV)
    torch.testing.assert_close(a, b)
    t0, t1 = compare.window(61, compare.DTS[3])
    ref = (torch.from_numpy(X[5, t0:t1]).float().permute(1, 0, 2, 3) - torch.from_numpy(mean)[:, None, None, None]) \
        / torch.from_numpy(std)[:, None, None, None]
    torch.testing.assert_close(a[1], ref)


def test_stdctm_batch_numpy_and_tensor():
    X, Xt, mean, std = _data(2)
    rows, dts = np.array([0, 4, 11]), np.array([1, 30, 7])
    torch.testing.assert_close(stdctm.batch(X, rows, dts, 61, mean, std, DEV),
                               stdctm.batch(Xt, rows, dts, 61, mean, std, DEV))


def test_pixel_stdctm_windows_numpy_and_tensor():
    X, Xt, _, _ = _data(3)
    rows, e = np.array([2, 6]), np.array([29, 59])
    a = pixel_stdctm.windows(X, rows, e)
    np.testing.assert_array_equal(a, pixel_stdctm.windows(Xt, rows, e).numpy())
    np.testing.assert_array_equal(a[1], X[6, 30:60])
