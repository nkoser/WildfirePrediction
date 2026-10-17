import numpy as np
import torch

from wildfire.paper.compare import CENTRE, summaries
from wildfire.paper.pixel import UNet3DPixel, pixel_fire_fraction, pixel_summaries


def test_pixel_summaries_match_centre_summaries():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 255, size=(3, 30, 55, 13, 13), dtype=np.uint8)
    S = pixel_summaries(X)
    assert S.shape == (3 * 169, 275)
    ref = summaries(X, np.arange(3), 0, 30)
    centre_rows = np.arange(3) * 169 + CENTRE * 13 + CENTRE
    np.testing.assert_allclose(S[centre_rows], ref, rtol=1e-6)
    # Zeile (s=1, r=2, c=5) gehoert zu genau diesem Pixel
    np.testing.assert_allclose(S[169 + 2 * 13 + 5, :55], X[1, -1, :, 2, 5].astype(np.float32))


def test_pixel_fire_fraction_per_pixel():
    X = np.zeros((2, 30, 55, 13, 13), np.uint8)
    X[1, :15, 0, 4, 7] = 1
    f = pixel_fire_fraction(X).reshape(2, 13, 13)
    assert f[1, 4, 7] == 0.5 and f.sum() == 0.5


def test_unet3d_pixel_outputs_a_map_with_prior_start():
    net = UNet3DPixel(55, prior=0.02).eval()
    out = net(torch.zeros(2, 55, 30, 13, 13))
    assert out.shape == (2, 13, 13)
    np.testing.assert_allclose(torch.sigmoid(out).detach().numpy(), 0.02, rtol=1e-4)
