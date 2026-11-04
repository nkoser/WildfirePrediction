import datetime as dt

import numpy as np

from wildfire.eval.metrics import d_calibration
from wildfire.paper.calib_eval import (conformal_threshold, coverage, groups, hazard_from_F, lower_bound, pit_fit,
                                       pit_apply, pit_map)
from wildfire.paper.surv_eval import DAY0, H


def _sample(n, factor=1.0, seed=0):
    """Wahre Tagesrate lam, tagesgenaue Zeit, Zensur bei H; Modell-F mit lam * factor."""
    rng = np.random.default_rng(seed)
    lam = np.exp(rng.uniform(np.log(0.002), np.log(0.06), n))
    T = np.ceil(-np.log1p(-rng.uniform(size=n)) / lam).astype(int)
    e = T <= H
    t = np.minimum(T, H)
    days = np.arange(1, H + 1)
    F = 1 - np.exp(-np.outer(lam * factor, days))
    return F, e, t


def test_pit_map_is_identity_for_a_calibrated_model_and_fixes_a_biased_one():
    F, e, t = _sample(20000)
    u, G = pit_map(F, e, t)
    assert np.all(np.diff(G) >= 0) and G[0] == 0 and G[-1] == 1
    assert np.abs(G - u).max() < 0.03                                   # schon kalibriert: G ~ Identitaet
    Fb, eb, tb = _sample(20000, factor=2.0, seed=1)                     # Modell ueberschaetzt
    Ft, et, tt = _sample(20000, factor=2.0, seed=2)
    before = d_calibration(Ft, et, tt)["d_cal_max_dev"]
    u, G = pit_fit(Fb, eb, tb)
    after = d_calibration(pit_apply(Ft, u, G), et, tt)["d_cal_max_dev"]
    assert before > 0.05 and after < 0.02


def test_hazard_from_F_round_trip():
    F, _, _ = _sample(50)
    h = hazard_from_F(F)
    S = np.cumprod(1 - h, axis=1)
    assert np.allclose(1 - S, F, atol=1e-9)


def test_lower_bound_semantics():
    F = np.array([[0.1, 0.3, 0.6], [0.0, 0.0, 0.05]])
    import wildfire.paper.calib_eval as ce
    old = ce.H
    try:
        ce.H = 3
        assert lower_bound(F, 0.3).tolist() == [2, 4]                   # zweite Zeile: nie erreicht -> H + 1
        assert lower_bound(F, -np.inf).tolist() == [1, 1]
        assert lower_bound(F, np.array([0.6, 0.05])).tolist() == [3, 3]
    finally:
        ce.H = old


def test_conformal_lower_bound_covers_at_least_one_minus_alpha():
    Fv, ev, tv = _sample(20000, factor=0.5, seed=3)                     # Modell unterschaetzt: naiv zu optimistisch
    Ft, et, tt = _sample(20000, factor=0.5, seed=4)
    for alpha in (0.05, 0.1, 0.2):
        s_hat = conformal_threshold(Fv, ev, tv, alpha)
        cov = coverage(lower_bound(Ft, s_hat), et, tt).mean()
        naive = coverage(lower_bound(Ft, alpha), et, tt).mean()
        assert cov >= 1 - alpha - 0.01                                  # Garantie (bis auf Zufall)
        assert cov <= 1 - alpha + 0.03                                  # und nicht unnoetig vorsichtig
        assert naive < 1 - alpha - 0.01                                 # ohne Konformalisierung verfehlt


def test_groups_cover_every_landmark_once_per_family():
    rng = np.random.default_rng(0)
    n = 1000
    years = rng.choice([2023, 2024, 2025], n)
    L_abs = np.array([(dt.date(int(y), 7, 1) - DAY0).days for y in years])
    meta = {"L_abs": L_abs, "doy": rng.integers(91, 274, n).astype(float), "col": rng.integers(0, 4000, n),
            "fire_win_patch": rng.random(n) < 0.1}
    g = groups(meta, "patch")
    for prefix in ("jahr_", "monat_", "band_"):
        assert np.all(sum(m.astype(int) for k, m in g.items() if k.startswith(prefix)) == 1)
    assert np.array_equal(g["jahr_2024"], years == 2024)
    assert np.all(g["feuer_im_fenster"] ^ g["kein_feuer_im_fenster"])


def test_ks_fit_flattens_the_pit_distribution():
    from wildfire.paper.calib_eval import ks_fit, ks_map
    Fb, eb, tb = _sample(20000, factor=2.0, seed=5)
    Ft, et, tt = _sample(20000, factor=2.0, seed=6)
    a, b = ks_fit(Fb, eb, tb)
    assert np.all(np.diff(ks_map(np.linspace(0.01, 0.99, 50), a, b)) > 0)
    before = d_calibration(Ft, et, tt)["d_cal_max_dev"]
    after = d_calibration(ks_map(Ft, a, b), et, tt)["d_cal_max_dev"]
    assert after < 0.5 * before
