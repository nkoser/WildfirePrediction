"""Prueft die Survival-Metriken gegen analytisch bekannte Faelle.

Metriken sind der gefaehrlichste Code im Projekt: Ein Fehler darin macht
sich nicht durch einen Absturz bemerkbar, sondern durch Zahlen, die
plausibel aussehen und falsch sind. Deshalb hier durchweg Faelle, deren
Ergebnis sich von Hand ausrechnen laesst.
"""

from __future__ import annotations

import numpy as np
import pytest

from wildfire.eval import metrics as M


def const_hazard(n, H, h):
    return np.full((n, H), float(h))


def test_survival_is_monotone_decreasing():
    rng = np.random.default_rng(0)
    h = rng.random((50, 30)) * 0.3
    S = M.survival_from_hazard(h)
    assert np.all(np.diff(S, axis=1) <= 1e-12)
    assert np.all((S >= 0) & (S <= 1))


def test_cif_is_monotone_increasing():
    rng = np.random.default_rng(1)
    F = M.cif_from_hazard(rng.random((50, 30)) * 0.3)
    assert np.all(np.diff(F, axis=1) >= -1e-12)


def test_survival_matches_geometric_formula():
    """Bei konstantem Hazard ist S(k) = (1-h)^k."""
    h = 0.1
    S = M.survival_from_hazard(const_hazard(1, 10, h))[0]
    assert np.allclose(S, [(1 - h) ** k for k in range(1, 11)])


def test_nll_of_censored_case_is_analytic():
    """Alle zensiert bei H, konstanter Hazard -> NLL = -H*log(1-h)."""
    h, H, n = 0.05, 20, 7
    nll = M.negative_log_likelihood(const_hazard(n, H, h),
                                    np.zeros(n), np.full(n, H))
    assert nll == pytest.approx(-H * np.log(1 - h), rel=1e-9)


def test_nll_of_event_at_k_is_analytic():
    """Ereignis bei k -> -[log h + (k-1) log(1-h)]."""
    h, H, k = 0.05, 20, 4
    nll = M.negative_log_likelihood(const_hazard(1, H, h), np.ones(1), np.array([k]))
    assert nll == pytest.approx(-(np.log(h) + (k - 1) * np.log(1 - h)), rel=1e-9)


def test_nll_event_at_first_step():
    """Randfall k=1: keine vorangehenden Ueberlebensterme."""
    h = 0.2
    nll = M.negative_log_likelihood(const_hazard(1, 10, h), np.ones(1), np.array([1]))
    assert nll == pytest.approx(-np.log(h), rel=1e-9)


def test_nll_is_minimised_by_the_true_hazard():
    """Die NLL muss beim datengenerierenden Hazard ihr Minimum haben."""
    rng = np.random.default_rng(3)
    true_h, H, n = 0.03, 30, 40000
    T = rng.geometric(true_h, size=n)
    event = T <= H
    t_ev = np.where(event, T, H)
    losses = {c: M.negative_log_likelihood(const_hazard(n, H, c), event, t_ev)
              for c in (0.01, 0.02, 0.03, 0.05, 0.10)}
    assert min(losses, key=losses.get) == 0.03, losses


def test_brier_of_perfect_prediction_is_zero():
    n, H = 100, 30
    h = np.zeros((n, H)); h[:, 9] = 1.0            # Ereignis sicher an Tag 10
    b = M.brier_at(h, np.ones(n), np.full(n, 10), tau=10)
    assert b == pytest.approx(0.0, abs=1e-6)


def test_brier_of_constant_predictor_equals_variance():
    """Vorhersage p fuer alle -> Brier = p^2(1-r) + (1-p)^2 r bei Rate r."""
    n, H, p, r = 10000, 30, 0.3, 0.2
    y = np.zeros(n, bool); y[: int(r * n)] = True
    h = np.zeros((n, H)); h[:, 0] = p              # F(tau)=p fuer alle tau
    t_ev = np.where(y, 1, H)
    b = M.brier_at(h, y, t_ev, tau=5)
    assert b == pytest.approx(p ** 2 * (1 - r) + (1 - p) ** 2 * r, rel=1e-6)


def test_weighting_changes_prevalence_dependent_metrics():
    """Gewichte muessen wirken - sonst waere die Schichtung wirkungslos."""
    n, H = 2000, 30
    ev = np.zeros(n, bool); ev[:200] = True
    t_ev = np.where(ev, 5, H)
    h = const_hazard(n, H, 0.02)
    w = np.where(ev, 1.0, 500.0)
    assert M.brier_at(h, ev, t_ev, 10) != pytest.approx(
        M.brier_at(h, ev, t_ev, 10, w), rel=1e-3)


def test_auc_of_random_scores_is_half():
    rng = np.random.default_rng(5)
    n, H = 20000, 30
    ev = rng.random(n) < 0.2
    t_ev = np.where(ev, rng.integers(1, H + 1, n), H)
    h = np.tile(rng.random((n, 1)) * 0.05, (1, H))     # Score ohne Bezug zum Label
    auc, _ = M.auc_and_pr_at(h, ev, t_ev, H)
    assert abs(auc - 0.5) < 0.02, auc


def test_auc_of_oracle_is_one():
    n, H = 1000, 30
    ev = np.zeros(n, bool); ev[:100] = True
    t_ev = np.where(ev, 3, H)
    h = np.where(ev[:, None], 0.9, 0.001) * np.ones((n, H))
    auc, ap = M.auc_and_pr_at(h, ev, t_ev, 5)
    assert auc > 0.999 and ap > 0.999


def test_calibration_is_perfect_for_a_calibrated_predictor():
    rng = np.random.default_rng(7)
    n, H = 60000, 30
    p = rng.random(n) * 0.4
    y = rng.random(n) < p
    h = np.zeros((n, H)); h[:, 0] = p
    t_ev = np.where(y, 1, H)
    assert M.calibration(h, y, t_ev, tau=5)["ece"] < 0.01


def test_calibration_detects_a_biased_predictor():
    rng = np.random.default_rng(8)
    n, H = 60000, 30
    p = rng.random(n) * 0.4
    y = rng.random(n) < p
    h = np.zeros((n, H)); h[:, 0] = np.clip(p * 2, 0, 0.99)   # doppelt so hoch
    t_ev = np.where(y, 1, H)
    assert M.calibration(h, y, t_ev, tau=5)["ece"] > 0.1


def test_recall_at_far_is_between_zero_and_one():
    rng = np.random.default_rng(9)
    n, H = 20000, 30
    ev = rng.random(n) < 0.05
    t_ev = np.where(ev, rng.integers(1, H + 1, n), H)
    h = np.tile((0.001 + 0.05 * ev)[:, None], (1, H))
    rec, thr = M.recall_at_false_alarm_rate(h, ev, t_ev, H, far=0.01)
    assert 0.0 <= rec <= 1.0 and np.isfinite(thr)


def test_lead_time_is_zero_when_alarm_comes_at_the_event_day():
    n, H = 10, 30
    h = np.zeros((n, H)); h[:, 9] = 1.0        # F ueberschreitet erst an Tag 10
    lt = M.lead_time(h, np.ones(n, bool), np.full(n, 10), threshold=0.5)
    assert lt["mean"] == pytest.approx(0.0)
    assert lt["detected_share"] == pytest.approx(1.0)


def test_lead_time_grows_with_earlier_alarm():
    n, H = 10, 30
    h = np.zeros((n, H)); h[:, 2] = 1.0        # Alarm ab Tag 3
    lt = M.lead_time(h, np.ones(n, bool), np.full(n, 10), threshold=0.5)
    assert lt["mean"] == pytest.approx(7.0)


def test_evaluate_returns_all_planned_metrics():
    rng = np.random.default_rng(11)
    n, H = 5000, 30
    ev = rng.random(n) < 0.1
    t_ev = np.where(ev, rng.integers(1, H + 1, n), H)
    r = M.evaluate(const_hazard(n, H, 0.01), ev, t_ev)
    assert {"nll", "integrated_brier", "horizons", "lead_time"} <= set(r)
    for t in ("1", "7", "14", "30"):
        assert {"brier", "roc_auc", "pr_auc", "ece"} <= set(r["horizons"][t])


# --------------------------------------------------------------- Bindungen
# Regressionstests: ein konstanter Praediktor lieferte AUC 1,0, weil die
# Landmark-Tabellen alle Positiven zuerst fuehren und argsort stabil ist.

def test_constant_score_gives_auc_one_half():
    n, H = 1000, 30
    ev = np.zeros(n, bool); ev[:100] = True      # Positive ZUERST, wie in echten Dateien
    t_ev = np.where(ev, 5, H)
    h = M.__dict__["np"].full((n, H), 0.01)
    auc, ap = M.auc_and_pr_at(h, ev, t_ev, 10)
    assert auc == pytest.approx(0.5, abs=1e-9), f"AUC {auc} statt 0.5"
    assert ap == pytest.approx(0.1, abs=1e-6), f"AP {ap} statt Praevalenz 0.1"


def test_constant_score_auc_is_order_invariant():
    n, H = 600, 30
    for pos_first in (True, False):
        ev = np.zeros(n, bool)
        ev[: 60] = True
        if not pos_first:
            ev = ev[::-1].copy()
        t_ev = np.where(ev, 5, H)
        auc, _ = M.auc_and_pr_at(np.full((n, H), 0.02), ev, t_ev, 10)
        assert auc == pytest.approx(0.5, abs=1e-9)


def test_partial_ties_are_averaged():
    """Zwei Score-Stufen, innerhalb jeder Stufe gemischt."""
    y = np.array([1, 0, 1, 0, 1, 0, 0, 0], float)
    score = np.array([.9, .9, .9, .9, .1, .1, .1, .1])
    w = np.ones(8)
    auc, _, _ = M._roc_pr(y, score, w)
    # obere Stufe: 2 von 3 Positiven, 2 von 5 Negativen
    assert 0.5 < auc < 0.75, auc


def test_calibration_of_constant_predictor_is_defined():
    n, H = 1000, 30
    ev = np.zeros(n, bool); ev[:50] = True
    t_ev = np.where(ev, 3, H)
    cal = M.calibration(np.full((n, H), 0.0), ev, t_ev, tau=10)
    assert np.isfinite(cal["ece"]), "ECE darf bei konstantem Praediktor nicht nan sein"
    assert cal["ece"] == pytest.approx(0.05, abs=1e-6)


def test_concordance_constant_predictor_is_half():
    """Gleiche Regressionsfalle wie bei der AUC: Bei identischen Scores darf
    die Eingabereihenfolge das Ergebnis nicht bestimmen."""
    from wildfire.eval.metrics import concordance_index
    rng = np.random.default_rng(0)
    n = 400
    event = np.zeros(n, int); event[:120] = 1          # Positive zuerst, wie in den Dateien
    t_event = rng.integers(1, 30, n)
    c, pairs = concordance_index(np.full(n, 0.5), event, t_event)
    assert pairs > 0
    assert abs(c - 0.5) < 1e-12


def test_concordance_perfect_and_inverted():
    from wildfire.eval.metrics import concordance_index
    # frueheres Ereignis = hoeheres Risiko  -> C = 1
    t_event = np.array([1, 2, 3, 4, 5])
    event = np.ones(5, int)
    risk = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert abs(concordance_index(risk, event, t_event)[0] - 1.0) < 1e-12
    assert abs(concordance_index(-risk, event, t_event)[0] - 0.0) < 1e-12


def test_concordance_censored_are_comparable_only_as_later():
    """Ein Zensierter bei t=100 ist mit jedem frueheren Ereignis vergleichbar,
    aber nie selbst der 'frueher gezuendete' Partner."""
    from wildfire.eval.metrics import concordance_index
    t_event = np.array([10, 100])
    event = np.array([1, 0])
    assert abs(concordance_index(np.array([1.0, 0.0]), event, t_event)[0] - 1.0) < 1e-12
    assert abs(concordance_index(np.array([0.0, 1.0]), event, t_event)[0] - 0.0) < 1e-12
    # zwei Zensierte allein ergeben kein vergleichbares Paar
    c, pairs = concordance_index(np.array([1.0, 0.0]), np.array([0, 0]), t_event)
    assert pairs == 0 and np.isnan(c)
