"""Prueft Verlustfunktion und Netz gegen bekannte Werte.

Der Verlust muss dieselbe Groesse sein wie die NLL der Auswertung -
sonst optimiert das Training etwas anderes, als am Ende berichtet wird.
Genau das wird hier numerisch abgeglichen.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wildfire.eval.metrics import negative_log_likelihood
from wildfire.models.losses import cif_from_logits, discrete_survival_nll
from wildfire.models.nets import SpatialHazardUNet

H = 30


def logit(p):
    return float(np.log(p / (1 - p)))


def test_loss_matches_evaluation_nll():
    """Training und Auswertung muessen dieselbe Zahl liefern."""
    rng = np.random.default_rng(0)
    n = 500
    h = rng.random((n, H)) * 0.2 + 1e-4
    ev = rng.random(n) < 0.3
    tev = np.where(ev, rng.integers(1, H + 1, n), H)
    w = rng.random(n) * 10 + 1

    ref = negative_log_likelihood(h, ev, tev, w)
    got = discrete_survival_nll(
        torch.log(torch.tensor(h) / (1 - torch.tensor(h))),
        torch.tensor(ev), torch.tensor(tev), torch.tensor(w)).item()
    assert got == pytest.approx(ref, rel=1e-6), (got, ref)


def test_loss_censored_case_is_analytic():
    p, n = 0.05, 8
    lo = torch.full((n, H), logit(p))
    got = discrete_survival_nll(lo, torch.zeros(n), torch.full((n,), H)).item()
    assert got == pytest.approx(-H * np.log(1 - p), rel=1e-6)


def test_loss_event_case_is_analytic():
    p, k = 0.05, 4
    lo = torch.full((1, H), logit(p))
    got = discrete_survival_nll(lo, torch.ones(1), torch.tensor([k])).item()
    assert got == pytest.approx(-(np.log(p) + (k - 1) * np.log(1 - p)), rel=1e-6)


def test_loss_is_stable_for_extreme_logits():
    """Bei Praevalenz 1e-3 werden Logits sehr negativ - das muss halten."""
    lo = torch.full((4, H), -30.0, requires_grad=True)
    loss = discrete_survival_nll(lo, torch.ones(4), torch.full((4,), 3))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(lo.grad).all()


def test_loss_gradient_points_the_right_way():
    """Richtung der Gradienten bei einem Ereignis an Tag 5.

    Drei Bereiche mit unterschiedlichem Verhalten:
      * Tag 5 selbst: der Hazard soll STEIGEN, Gradient negativ.
      * Tage 1-4: das Pixel hat ueberlebt, die Hazards sollen SINKEN.
      * Tage 6-30: kommen in der Likelihood GAR NICHT vor - das Pixel ist
        ab Tag 5 nicht mehr unter Risiko. Der Gradient ist exakt null.
        Das ist kein Defekt, sondern die Definition diskreter
        Survival-Likelihood.
    """
    lo = torch.full((1, H), -7.0, requires_grad=True)
    discrete_survival_nll(lo, torch.ones(1), torch.tensor([5])).backward()
    g = lo.grad[0]
    assert g[4] < 0, "Hazard am Ereignistag muesste steigen"
    assert torch.all(g[:4] > 0), "Hazards vor dem Ereignis muessten sinken"
    assert torch.all(g[5:] == 0), "Tage nach dem Ereignis duerfen nicht beitragen"


def test_censored_case_constrains_all_days():
    """Gegenprobe: bei Zensierung tragen ALLE Tage bei."""
    lo = torch.full((1, H), -7.0, requires_grad=True)
    discrete_survival_nll(lo, torch.zeros(1), torch.tensor([H])).backward()
    assert torch.all(lo.grad[0] > 0), "alle Hazards muessten sinken wollen"


def test_cif_is_monotone():
    rng = torch.Generator().manual_seed(0)
    lo = torch.randn(20, H, generator=rng) * 2 - 5
    F_ = cif_from_logits(lo)
    assert torch.all(torch.diff(F_, dim=1) >= -1e-7)
    assert torch.all((F_ >= 0) & (F_ <= 1))


def test_unet_shapes_survive_odd_sizes():
    net = SpatialHazardUNet(in_ch=17, horizon=H, width=8)
    for P in (17, 33, 65):
        out = net(torch.zeros(2, 17, P, P))
        assert out.shape == (2, H, P, P), (P, out.shape)


def test_unet_centre_matches_dense_output():
    net = SpatialHazardUNet(in_ch=5, horizon=H, width=8)
    x = torch.randn(3, 5, 33, 33)
    dense, centre = net(x), net.centre_logits(x)
    assert torch.allclose(dense[:, :, 16, 16], centre)


def test_unet_starts_near_the_prevalence():
    """Der Startbias muss auf die Groessenordnung seltener Ereignisse zeigen."""
    net = SpatialHazardUNet(in_ch=5, horizon=H, width=8, init_bias=-7.0)
    h = torch.sigmoid(net(torch.randn(2, 5, 33, 33)))
    assert 1e-4 < h.mean().item() < 5e-3, h.mean().item()


def test_unet_can_overfit_a_single_example():
    """Kleinster Sanity Check aus Phase 4.2: ein Beispiel auswendig lernen."""
    torch.manual_seed(0)
    net = SpatialHazardUNet(in_ch=4, horizon=H, width=8)
    x = torch.randn(1, 4, 33, 33)
    ev, tev = torch.ones(1), torch.tensor([6])
    opt = torch.optim.Adam(net.parameters(), lr=0.05)
    first = None
    for _ in range(120):
        opt.zero_grad()
        loss = discrete_survival_nll(net.centre_logits(x), ev, tev)
        loss.backward(); opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < first * 0.2, (first, loss.item())
