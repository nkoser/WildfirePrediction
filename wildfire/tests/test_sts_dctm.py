import numpy as np
import torch

from wildfire.paper.sts_dctm import STSDCTM, doy_of, patch_events


def test_forward_shapes_and_monotone_risk():
    torch.manual_seed(0)
    net = STSDCTM(d=32, heads=4, depth=2).eval()
    x = torch.randn(2, 30, 55, 13, 13)
    raw_px, raw_pa = net(x, torch.tensor([180.0, 240.0]))
    assert raw_px.shape == (2, 13, 13, net.px_head.n_out) and raw_pa.shape == (2, net.pa_head.n_out)
    tau = torch.arange(0, 101, dtype=torch.float64)
    F_px = -torch.expm1(net.px_head.log_survival(raw_px.reshape(-1, raw_px.shape[-1]), tau))
    F_pa = -torch.expm1(net.pa_head.log_survival(raw_pa, tau))
    assert torch.all(F_px[:, 0].abs() < 1e-9) and torch.all(F_pa[:, 0].abs() < 1e-9)       # F(0) = 0
    assert torch.all(F_px.diff(dim=1) >= -1e-12) and torch.all(F_pa.diff(dim=1) >= -1e-12)  # monoton


def test_initial_risk_matches_init_cif():
    net = STSDCTM(d=32, heads=4, depth=1, init_cif_px=0.02, init_cif_patch=0.3).eval()
    raw_px, raw_pa = net(torch.zeros(1, 30, 55, 13, 13), torch.tensor([200.0]))
    F_pa = -torch.expm1(net.pa_head.log_survival(raw_pa, torch.tensor([100.0], dtype=torch.float64)))
    assert abs(float(F_pa) - 0.3) < 1e-6                                                    # Kopf startet bei 0 Ausgabe


def test_patch_events_take_the_first_pixel_event():
    ev = np.zeros((2, 13, 13), bool)
    t = np.full((2, 13, 13), 9, np.int64)                 # Zensur nach 9 Tagen
    ev[0, 1, 1], t[0, 1, 1] = True, 4
    ev[0, 5, 5], t[0, 5, 5] = True, 2
    pe, pt = patch_events(ev, t)
    assert pe.tolist() == [True, False] and pt.tolist() == [2, 9]


def test_doy():
    assert doy_of(np.array([20240101, 20241231])).tolist() == [1.0, 366.0]
