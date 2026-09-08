"""Prueft die Umrechnung von Horizont-Ankern in eine Survival-Kurve.

B1 trainiert je Horizont einen EIGENEN Klassifikator. Deren Ausgaben sind
nicht automatisch miteinander vertraeglich: F(1) kann groesser herauskommen
als F(7), obwohl das kumulative Risiko nur wachsen darf. Aus vier
Ankerpunkten muss trotzdem eine gueltige Kurve werden - monoton, im
Intervall [0,1], und an den Ankern mit den vorhergesagten Werten
uebereinstimmend.
"""

from __future__ import annotations

import numpy as np
import pytest

from wildfire.eval.metrics import cif_from_hazard
from wildfire.models.tabular import hazards_from_anchors, prior_shift

# Eigene Anker statt des Modul-Globals: ``HORIZONS`` wird zur Laufzeit auf
# den Horizont des Landmark-Manifests beschnitten. Ein Test, der daran
# haengt, bricht bei jeder Aenderung des Horizonts - und prueft dann nicht
# mehr die Umrechnung, sondern die Konfiguration.
HORIZONS = (1, 7, 14, 30)
H = 30


def test_anchors_are_reproduced():
    """An den Ankern muss die abgeleitete Kurve die Vorhersage treffen."""
    F = np.array([[0.01, 0.05, 0.09, 0.20],
                  [0.001, 0.002, 0.004, 0.010]])
    h = hazards_from_anchors(F, HORIZONS, H)
    got = cif_from_hazard(h)[:, [t - 1 for t in HORIZONS]]
    assert np.allclose(got, F, atol=1e-9), got - F


def test_result_is_monotone():
    rng = np.random.default_rng(0)
    F = np.sort(rng.random((200, len(HORIZONS))) * 0.3, axis=1)
    F_out = cif_from_hazard(hazards_from_anchors(F, HORIZONS, H))
    assert np.all(np.diff(F_out, axis=1) >= -1e-12)


def test_non_monotone_input_is_repaired():
    """F(1) > F(7) darf nicht zu negativen Hazards fuehren."""
    F = np.array([[0.20, 0.05, 0.09, 0.02]])
    h = hazards_from_anchors(F, HORIZONS, H)
    assert np.all(h >= 0) and np.all(h <= 1)
    out = cif_from_hazard(h)
    assert np.all(np.diff(out, axis=1) >= -1e-12)
    # das laufende Maximum bleibt erhalten
    assert out[0, 0] == pytest.approx(0.20, abs=1e-9)
    assert out[0, HORIZONS[-1] - 1] == pytest.approx(0.20, abs=1e-6)


def test_hazard_is_constant_between_anchors():
    """Sparsamste Annahme: zwischen zwei Ankern aendert sich nichts."""
    F = np.array([[0.01, 0.05, 0.09, 0.20]])
    h = hazards_from_anchors(F, HORIZONS, H)[0]
    for a, b in zip((0,) + HORIZONS[:-1], HORIZONS):
        seg = h[a:b]
        assert np.allclose(seg, seg[0]), f"Segment {a}:{b} nicht konstant"


def test_zero_risk_gives_zero_hazard():
    F = np.full((3, len(HORIZONS)), 1e-7)
    h = hazards_from_anchors(F, HORIZONS, H)
    assert np.all(h < 1e-5)


def test_certain_event_saturates():
    F = np.array([[0.999999, 0.999999, 0.999999, 0.999999]])
    out = cif_from_hazard(hazards_from_anchors(F, HORIZONS, H))
    assert out[0, -1] > 0.999


def test_shape_and_range():
    rng = np.random.default_rng(1)
    F = np.sort(rng.random((50, len(HORIZONS))), axis=1)
    h = hazards_from_anchors(F, HORIZONS, H)
    assert h.shape == (50, H)
    assert np.all((h > 0) & (h < 1))


# ------------------------------------------------------------ Prior-Korrektur

def test_prior_shift_is_identity_when_priors_match():
    p = np.array([0.01, 0.3, 0.9])
    assert np.allclose(prior_shift(p, 0.2, 0.2), p, atol=1e-9)


def test_prior_shift_lowers_probabilities_for_rarer_truth():
    """Training angereichert, Wahrheit selten -> Vorhersagen muessen sinken."""
    p = np.array([0.1, 0.5, 0.9])
    out = prior_shift(p, pi_train=0.09, pi_true=0.0005)
    assert np.all(out < p)
    assert np.all(out > 0)


def test_prior_shift_preserves_ranking():
    """Die Korrektur ist monoton - sie darf die Rangfolge nicht aendern."""
    p = np.array([0.02, 0.11, 0.35, 0.77])
    out = prior_shift(p, 0.09, 0.0005)
    assert np.all(np.diff(out) > 0)


def test_prior_shift_moves_mean_towards_true_prevalence():
    rng = np.random.default_rng(3)
    p = rng.random(100000) * 0.2
    out = prior_shift(p, pi_train=0.09, pi_true=0.0014)
    assert out.mean() < p.mean() / 10
