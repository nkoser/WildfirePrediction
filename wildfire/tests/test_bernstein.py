"""Prueft den stetigen Survival-Kopf.

Die entscheidende Eigenschaft ist Monotonie: F(tau) darf mit tau niemals
fallen, fuer JEDE Eingabe - auch fuer solche, die im Training nie
vorkamen. Bei der Bernstein-Parametrisierung folgt das aus der
Konstruktion, nicht aus dem Training. Genau das wird hier mit
Zufallsgewichten in extremen Bereichen geprueft.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wildfire.models.bernstein import (BernsteinSurvivalHead, bernstein_basis,
                                       bernstein_nll)

M, TAU_MAX = 12, 365.0


def head():
    return BernsteinSurvivalHead(degree=M, tau_max=TAU_MAX)


def test_basis_sums_to_one():
    """Bernstein-Basis ist eine Zerlegung der Eins."""
    u = torch.linspace(0, 1, 50)
    assert torch.allclose(bernstein_basis(u, M).sum(-1), torch.ones(50), atol=1e-5)


def test_basis_is_non_negative():
    assert torch.all(bernstein_basis(torch.linspace(0, 1, 200), M) >= 0)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_cif_is_monotone_for_arbitrary_inputs(seed):
    """Monotonie gilt per Konstruktion, nicht nur nach Training."""
    torch.manual_seed(seed)
    h = head()
    raw = torch.randn(64, M + 1) * 5.0          # bewusst extrem
    tau = torch.linspace(0.0, TAU_MAX, 400)
    F = h.cif(raw, tau)
    assert torch.all(torch.diff(F, dim=1) >= -1e-6)
    assert torch.all((F >= 0) & (F <= 1))


def test_cif_is_monotone_even_for_huge_weights():
    h = head()
    raw = torch.full((4, M + 1), 50.0)
    F = h.cif(raw, torch.linspace(0, TAU_MAX, 100))
    assert torch.all(torch.diff(F, dim=1) >= -1e-6)


def test_evaluable_at_arbitrary_non_integer_times():
    """Der eigentliche Zweck: beliebige Zeitpunkte, nicht nur Gitterpunkte."""
    h = head()
    raw = torch.randn(3, M + 1)
    for t in (0.5, 7.25, 37.5, 99.9, 364.999):
        F = h.cif(raw, torch.tensor([t]))
        assert F.shape == (3, 1) and torch.all(torch.isfinite(F))


def test_interpolates_between_neighbouring_days():
    h = head()
    raw = torch.randn(8, M + 1)
    F = h.cif(raw, torch.tensor([10.0, 10.5, 11.0]))
    assert torch.all(F[:, 0] <= F[:, 1] + 1e-6)
    assert torch.all(F[:, 1] <= F[:, 2] + 1e-6)


def test_init_starts_at_low_risk():
    """Ohne Training soll das Risiko klein sein, nicht bei 0,5."""
    h = head()
    F = h.cif(torch.zeros(4, M + 1), torch.tensor([30.0]))
    assert torch.all(F < 0.05), F


def test_nll_is_finite_and_differentiable():
    h = head()
    raw = torch.randn(32, M + 1, requires_grad=True)
    ev = torch.randint(0, 2, (32,)).float()
    tev = torch.randint(1, 100, (32,))
    loss = bernstein_nll(h, raw, ev, tev, horizon=100.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(raw.grad).all()


def test_nll_rewards_mass_at_the_event_time():
    """Ein Modell, das die Ereigniszeit trifft, muss besser sein."""
    torch.manual_seed(0)
    h = head()
    raw = torch.zeros(1, M + 1, requires_grad=True)
    ev, tev = torch.ones(1), torch.tensor([20])
    opt = torch.optim.Adam([raw], lr=0.2)
    first = bernstein_nll(h, raw, ev, tev, 100.0).item()
    for _ in range(300):
        opt.zero_grad()
        loss = bernstein_nll(h, raw, ev, tev, 100.0)
        loss.backward(); opt.step()
    assert loss.item() < first - 1.0, (first, loss.item())
    # Nach dem Anpassen muss zwischen Tag 19 und 20 spuerbar Masse liegen
    F = h.cif(raw.detach(), torch.tensor([19.0, 20.0]))
    assert (F[0, 1] - F[0, 0]).item() > 0.05


def test_censored_case_pushes_risk_down():
    h = head()
    raw = torch.zeros(1, M + 1, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=0.2)
    before = h.cif(raw.detach(), torch.tensor([100.0])).item()
    for _ in range(200):
        opt.zero_grad()
        bernstein_nll(h, raw, torch.zeros(1), torch.tensor([100]), 100.0).backward()
        opt.step()
    after = h.cif(raw.detach(), torch.tensor([100.0])).item()
    assert after < before


def test_query_beyond_tau_max_is_clamped_not_extrapolated():
    h = head()
    raw = torch.randn(2, M + 1)
    a = h.cif(raw, torch.tensor([TAU_MAX]))
    b = h.cif(raw, torch.tensor([TAU_MAX * 3]))
    assert torch.allclose(a, b), "ausserhalb der Basis darf nicht extrapoliert werden"


def test_cif_is_exactly_zero_at_tau_zero():
    """S(0) = 1 per Konstruktion - frueher lag F(0) bei sigmoid(theta_0)."""
    raw = torch.randn(16, M + 1) * 3.0
    F0 = head().cif(raw, torch.tensor([0.0]))
    assert torch.all(F0 == 0.0), F0


def test_init_cif_matches_prevalence():
    """Bei Null-Ausgabe soll F(tau_max) die Praevalenz treffen, nicht 79 %."""
    from wildfire.models.losses import INIT_CIF_H
    h = BernsteinSurvivalHead(degree=M, tau_max=100.0)
    F = h.cif(torch.zeros(2, M + 1, dtype=torch.float64), torch.tensor([100.0]))
    assert torch.allclose(F, torch.full_like(F, INIT_CIF_H), rtol=1e-9), F


def test_nll_equals_discrete_nll_of_its_hazards():
    """Trainingsverlust und Auswertungs-NLL muessen dieselbe Zahl sein.

    Die Auswertung rechnet ueber hazards() und die diskrete NLL; das Training
    ueber die intervallzensierte Likelihood. Mit F(0) != 0 wichen beide um
    log S(0) je Beispiel voneinander ab.
    """
    from wildfire.eval.metrics import negative_log_likelihood
    torch.manual_seed(1)
    Hh = 100
    h = BernsteinSurvivalHead(degree=M, tau_max=float(Hh))
    raw = torch.randn(200, M + 1, dtype=torch.float64)
    ev = (torch.rand(200) < 0.4).float()
    tev = torch.where(ev.bool(), torch.randint(1, Hh + 1, (200,)), torch.full((200,), Hh))
    w = torch.rand(200, dtype=torch.float64) * 50 + 1
    got = bernstein_nll(h, raw, ev, tev, float(Hh), weight=w).item()
    haz = h.hazards(raw, Hh).numpy()
    ref = negative_log_likelihood(haz, ev.numpy(), tev.numpy(), w.numpy())
    assert got == pytest.approx(ref, rel=1e-8), (got, ref)


def test_low_risk_event_still_has_a_gradient():
    """Tages-Hazard um 1e-9: frueher klemmte F(k)-F(k-1) auf 1e-7, Gradient 0."""
    h = BernsteinSurvivalHead(degree=M, tau_max=100.0, init_offset=-25.0,
                              init_cif=1e-7)
    raw = torch.zeros(1, M + 1, requires_grad=True)
    loss = bernstein_nll(h, raw, torch.ones(1), torch.tensor([50]), 100.0)
    loss.backward()
    assert torch.isfinite(loss) and loss.item() > 15.0, loss.item()
    assert raw.grad.abs().sum().item() > 0.1, raw.grad
