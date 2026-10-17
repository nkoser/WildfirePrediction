import numpy as np
import torch

from wildfire.paper.pixel_stdctm import PixelSTDCTM, pixel_nll, pixel_targets


def test_pixel_targets_first_fire_and_censoring():
    fire = np.zeros((2, 61, 13, 13), bool)
    fire[0, 40, 6, 6] = True          # vor dem Stichtag 45: zaehlt nicht
    fire[0, 50, 6, 6] = True          # erstes Feuer nach 45 -> t = 5
    fire[0, 55, 6, 6] = True
    fire[1, 60, 0, 12] = True         # am Zieltag -> t = 60 - 29 = 31
    ev, t = pixel_targets(fire, np.array([45, 29]))
    assert ev[0, 6, 6] and t[0, 6, 6] == 5
    assert ev[1, 0, 12] and t[1, 0, 12] == 31
    assert not ev[0, 0, 0] and t[0, 0, 0] == 15      # zensiert am Zieltag: 60 - 45
    assert ev.sum() == 2


def test_pixel_model_shape_and_start_value():
    net = PixelSTDCTM(tau_max=31.0, init_cif=0.05).eval()
    raw = net(torch.zeros(2, 30, 55, 13, 13))
    assert raw.shape == (2, 13, 13, net.dctm.n_out)
    F31 = -torch.expm1(net.dctm.log_survival(raw.reshape(-1, raw.shape[-1]), torch.tensor([31.0])))
    np.testing.assert_allclose(F31.detach().numpy(), 0.05, rtol=1e-3)


def test_pixel_nll_matches_head_likelihood_when_censored_at_horizon():
    net = PixelSTDCTM(tau_max=31.0, init_cif=0.05)
    raw = torch.randn(4, 1, 1, net.dctm.n_out)
    ev = torch.tensor([True, False, True, False]).view(4, 1, 1)
    t = torch.tensor([3, 31, 31, 31]).view(4, 1, 1)
    ref = -net.dctm.log_likelihood(raw.view(4, -1), ev.view(-1), t.view(-1), 31.0).mean()
    np.testing.assert_allclose(pixel_nll(net.dctm, raw, ev, t).item(), ref.item(), rtol=1e-6)
