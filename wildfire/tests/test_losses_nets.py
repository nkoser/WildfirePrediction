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
from wildfire.models.losses import (INIT_CIF_H, cif_from_logits,
                                    discrete_survival_nll, init_hazard_logit)
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
    """Der Startbias muss F(H) auf die Praevalenz setzen, nicht den Tages-Hazard.

    Frueher pruefte dieser Test 1e-4 < h < 5e-3 je TAG - das liess init_bias=-7
    durch, also F(100) = 8,7 % bei einer Praevalenz von 0,13 %.
    """
    net = SpatialHazardUNet(in_ch=5, horizon=H, width=8)
    logits = net(torch.randn(2, 5, 33, 33))
    F_H = cif_from_logits(logits.permute(0, 2, 3, 1).reshape(-1, H))[:, -1]
    assert torch.allclose(F_H, torch.full_like(F_H, INIT_CIF_H), rtol=1e-3), F_H.mean()


def test_init_hazard_logit_matches_prevalence():
    b = init_hazard_logit(1.3e-3, 100)
    assert abs(b - (-11.25)) < 0.01, b
    F_H = 1 - (1 - torch.sigmoid(torch.tensor(b, dtype=torch.float64))) ** 100
    assert abs(F_H.item() - 1.3e-3) < 1e-9


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


def test_chunked_shuffle_sampler_is_a_permutation():
    """Blockweises Mischen darf kein Beispiel verlieren oder doppeln."""
    from wildfire.models.torch_data import ChunkedShuffleSampler
    order = np.arange(1000)
    s = ChunkedShuffleSampler(order, chunk=137, seed=3)
    out = list(s)
    assert len(out) == 1000 == len(s)
    assert sorted(out) == list(range(1000))


def test_chunked_shuffle_differs_per_epoch_but_stays_local():
    from wildfire.models.torch_data import ChunkedShuffleSampler
    order = np.arange(4000)
    s = ChunkedShuffleSampler(order, chunk=500, seed=1)
    e0 = list(s)
    s.set_epoch(1)
    e1 = list(s)
    assert e0 != e1                                    # andere Reihenfolge
    assert sorted(e0) == sorted(e1)                    # gleiche Menge
    # Lokalitaet: aufeinanderfolgende Indizes liegen im Mittel naeher beieinander
    # als bei globalem Mischen, wo der Erwartungswert n/3 betraegt.
    step = np.abs(np.diff(np.array(e0)))
    assert step.mean() < len(order) / 3


def test_chunked_shuffle_handles_ragged_last_block():
    from wildfire.models.torch_data import ChunkedShuffleSampler
    s = ChunkedShuffleSampler(np.arange(10), chunk=4, seed=0)
    assert sorted(s) == list(range(10))


def test_weights_shift_hazard_scale_not_ranking():
    """Der Fehler, der den ersten B2-Lauf unbrauchbar machte.

    train.py rief den Verlust ohne das Ziehungsgewicht auf, mlp_survival.py mit.
    B2 lernte damit die Praevalenz der angereicherten STICHPROBE (9,15 %) statt
    die der Grundgesamtheit (1,27 Promille). Der Effekt ist ein gleichmaessiger
    Versatz im Logit: Die Rangordnung bleibt erhalten - die AUC sah deshalb
    normal aus - aber die Kalibrierung ist zerstoert, val NLL 0,0697 statt 0,0125.

    Entscheidend fuer den Test ist ein GETEILTER Parametersatz. Bekaeme jedes
    Beispiel eigene Logits, koennte es seinen Ausgang exakt treffen und die
    Gewichte blieben wirkungslos - so wie in der ersten Fassung dieses Tests.
    """
    torch.manual_seed(0)
    N, H = 4096, 20
    event = (torch.rand(N) < 0.3).float()
    t_event = torch.randint(1, H + 1, (N,))
    # Positive vertreten nur sich selbst, Negative je 50 nicht gezogene
    weight = torch.where(event > 0, torch.ones(N), torch.full((N,), 50.0))

    def fit(w):
        shared = torch.zeros(1, H, requires_grad=True)      # EIN Hazardverlauf
        opt = torch.optim.Adam([shared], lr=0.1)
        for _ in range(400):
            opt.zero_grad()
            discrete_survival_nll(shared.expand(N, H), event, t_event, w).backward()
            opt.step()
        return float(torch.sigmoid(shared.detach()).mean())

    h_weighted, h_plain = fit(weight), fit(None)
    # Ungewichtet landet der Hazard deutlich zu hoch - genau der B2-Fehler.
    assert h_plain > 3 * h_weighted, (h_plain, h_weighted)

    # Und die effektive Praevalenz erklaert den Faktor: ungewichtet 0,3,
    # gewichtet rund 0,3/(0,3 + 50*0,7).
    p_plain = float(event.mean())
    p_weighted = float((weight * event).sum() / weight.sum())
    assert p_plain > 20 * p_weighted
