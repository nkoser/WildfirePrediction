import math

import numpy as np
import torch

from wildfire.models.surv_heads import BinaryHorizonHead, head_loss, interval_nll
from wildfire.paper.contagion import ContagionSTS
from wildfire.paper.rate_labels import daily_fire
from wildfire.paper.sts_dctm import STSDCTM, load_sts
from wildfire.paper.surv_prepare import DAY_BITS

TAU = torch.arange(0, 101, dtype=torch.float64)


def test_bce_head_labels_mask_and_raw_outputs():
    head = BinaryHorizonHead(100, 0.05, taus=(31, 2, 8))
    assert head.taus == (2, 8, 31)
    # Label je Horizont: Ereignis bis tau; unbekannt, wenn zensiert vor tau
    raw = torch.zeros(3, 3, requires_grad=True)
    ev, t = torch.tensor([True, False, False]), torch.tensor([5, 100, 4])
    loss = head.loss(raw, ev, t)
    b = head.bias
    y = torch.tensor([[0, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=torch.float64)
    known = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 0, 0]], dtype=torch.float64)
    ref = torch.nn.functional.binary_cross_entropy_with_logits(b.expand(3, 3), y, reduction="none")
    assert torch.isclose(loss.double(), (ref * known).sum() / known.sum())
    # rohe Werte an den Horizonten, wenn nicht monotonisiert
    r = torch.tensor([[2.0, -1.0, 0.5]])
    head.monotone = False
    F_ = -torch.expm1(head.log_survival(r, torch.tensor([2.0, 8.0, 31.0])))
    assert torch.allclose(F_, torch.sigmoid(r.double() + b))
    assert head.violations(r).item()


def test_bce_head_monotone_curve_is_a_valid_survival_function():
    head = BinaryHorizonHead(100, 0.05)
    torch.manual_seed(0)
    raw = torch.randn(4, head.n_out) * 3
    ls = head.log_survival(raw, TAU)
    assert torch.all(ls[:, 0].abs() < 1e-12) and torch.all(ls[:, 1:] < ls[:, :-1])
    nll = interval_nll(head, raw, torch.tensor([1, 1, 0, 1]).bool(), torch.tensor([3, 50, 100, 99]))
    assert torch.isfinite(nll)
    # an monotonen Stellen gleich der rohen Ausgabe
    ok = ~head.violations(raw)
    F_ = -torch.expm1(head.log_survival(raw, torch.tensor([float(t) for t in head.taus])))
    assert torch.allclose(F_[ok], head.cif_raw(raw)[ok], atol=1e-6)


def test_rate_head_starts_constant_and_trains():
    torch.manual_seed(0)
    net = STSDCTM(d=32, heads=4, depth=1, head="lh", rate=True, init_rate=(0.03, 0.004))
    x, doy = torch.randn(2, 30, 55, 13, 13), torch.tensor([150.0, 200.0])
    raw_px, raw_pa, lg = net(x, doy, rate=True)
    assert lg.shape == (2, 2, 100)
    p = torch.sigmoid(lg)
    assert torch.allclose(p[:, 0], torch.full_like(p[:, 0], 0.03), atol=1e-5)
    assert torch.allclose(p[:, 1], torch.full_like(p[:, 1], 0.004), atol=1e-6)
    assert len(net(x, doy)) == 2                               # ohne rate wie bisher
    y = torch.zeros(2, 2, 100); y[0, 0, 30] = 1
    loss = torch.nn.functional.binary_cross_entropy_with_logits(lg, y) + head_loss(net.pa_head, raw_pa, torch.tensor([True, False]), torch.tensor([31, 100]))
    loss.backward()
    assert net.rate_head.pa.weight.grad.abs().sum() > 0 and net.rate_head.px.weight.grad.abs().sum() > 0


def test_history_length_only_uses_the_last_days():
    torch.manual_seed(0)
    net = STSDCTM(d=32, heads=4, depth=1, head="lh", hist=7).eval()
    torch.nn.init.normal_(net.pa_lin.weight)                    # Kopf startet bei 0, sonst sieht man nichts
    x, doy = torch.randn(1, 30, 55, 13, 13), torch.tensor([150.0])
    x2 = x.clone(); x2[:, :23] = torch.randn(1, 23, 55, 13, 13)
    x3 = x.clone(); x3[:, -1] += 1.0
    with torch.no_grad():
        a, b, c = net(x, doy)[1], net(x2, doy)[1], net(x3, doy)[1]
    assert torch.allclose(a, b) and not torch.allclose(a, c)


def test_contagion_with_rate_and_short_history():
    torch.manual_seed(0)
    net = ContagionSTS(d=32, heads=4, depth=1, kernel=1, obs_days=30, burn="curve", hist=7, rate=True).eval()
    x = torch.randn(2, 30, 55, 13, 13)
    x[:, :, 0] = 0.0
    x[0, 5, 0, 6, 6] = 1.0                                     # Feuer vor 25 Tagen: ausserhalb der 7 Tage
    with torch.no_grad():
        ls_px, ls_pa, lg = net(x, torch.tensor([150.0, 150.0]), rate=True)
        x[0, 5, 0, 6, 6] = 0.0
        ls_px2, ls_pa2 = net(x, torch.tensor([150.0, 150.0]))
    assert lg.shape == (2, 2, 100) and torch.allclose(ls_pa, ls_pa2) and torch.allclose(ls_px, ls_px2)


def test_checkpoint_roundtrip_keeps_rate_head_history_and_taus(tmp_path):
    net = STSDCTM(d=32, heads=4, depth=1, head="bce", hist=14, rate=True, bce_taus=(2, 31))
    args = {"d": 32, "heads": 4, "depth": 1, "head": "bce", "hist_days": 14, "rate_weight": 1.0, "bce_taus": [2, 31]}
    torch.save({"state": net.state_dict(), "args": args}, tmp_path / "c.pt")
    net2 = load_sts(tmp_path / "c.pt")
    assert net2.hist == 14 and net2.rate_head is not None and net2.pa_head.taus == (2, 31)


def test_daily_fire_marks_every_burning_day_and_matches_first_event():
    H, W = 40, 40
    fires = [(20, 20, 105), (20, 20, 106), (20, 24, 103), (13, 20, 150), (20, 20, 220), (20, 20, 100)]
    keys = np.sort(np.array([(r * W + c) * DAY_BITS + d for r, c, d in fires], np.int64))
    patch, centre = daily_fire(keys, np.array([100]), np.array([20]), np.array([20]), H, W)
    assert np.flatnonzero(patch[0]).tolist() == [2, 4, 5]     # Tage 103, 105, 106; 150 liegt ausserhalb, 220 nach 100 Tagen
    assert np.flatnonzero(centre[0]).tolist() == [4, 5]
    assert not patch[0, -1] and math.isclose(patch.sum(), 3)
