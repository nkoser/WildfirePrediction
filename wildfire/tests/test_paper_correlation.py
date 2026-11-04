import numpy as np
from scipy.stats import spearmanr

from wildfire.paper.correlation import (DETERMINISTIC, LAG_DRIVERS, LAG_TARGETS, STATIC, auc_by_month,
                                        clusters, day_months, eta, lagged, partial_and_vif, r2_from,
                                        ranks, spearman)
from wildfire.paper.prepare import CHANNEL_NAMES, LANDCOVER_CH


def test_spearman_matches_scipy_with_ties():
    rng = np.random.default_rng(0)
    Z = rng.integers(0, 6, size=(500, 4)).astype(np.float32)   # viele Bindungen wie bei uint8
    Z[:, 1] += Z[:, 0]
    ref = spearmanr(Z).statistic
    assert np.allclose(spearman(Z), ref, atol=1e-10)


def test_constant_column_gives_nan_not_crash():
    Z = np.c_[np.arange(10.0), np.ones(10)]
    R = spearman(Z)
    assert R[0, 0] == 1.0 and np.isnan(R[0, 1]) and np.isnan(R[1, 1])


def test_lagged_finds_the_shift():
    rng = np.random.default_rng(1)
    n, L, k0 = 200, 61, 3
    a = rng.normal(size=(n, L + k0))
    Z = np.stack([a[:, k0:], a[:, :-k0]], -1)       # Kanal 1 folgt Kanal 0 um k0 Tage
    Zr = ranks(Z.reshape(-1, 2)).reshape(n, L, 2)
    lag = lagged(Zr, 6)
    assert int(np.argmax(lag[:, 0, 1])) == k0 and lag[k0, 0, 1] > 0.99
    assert abs(lag[0, 0, 1]) < 0.05
    assert np.allclose(lag[0], lag[0].T)


def test_partial_vif_and_r2_on_known_structure():
    rng = np.random.default_rng(2)
    x1, x2 = rng.normal(size=(2, 20000))
    y = x1 + x2 + 0.1 * rng.normal(size=20000)
    R = np.corrcoef(np.c_[x1, x2, y].T)
    pc, vif = partial_and_vif(R)
    assert abs(pc[0, 1]) > 0.9                      # gegeben y haengen x1 und x2 stark zusammen
    assert vif[2] > 50 and vif[0] > 25
    assert r2_from(R, 2, [0, 1]) > 0.99 and r2_from(R, 2, [0]) < 0.6


def test_clusters_cut_at_rho():
    R = np.array([[1, .95, .1], [.95, 1, .05], [.1, .05, 1]])
    groups, order, _ = clusters(R, ["a", "b", "c"], rho=0.7)
    assert groups == [["a", "b"], ["c"]] and sorted(order) == [0, 1, 2]


def test_day_months_cross_year():
    m = day_months(np.array([20190102]), 5)          # 29.12.2018 .. 02.01.2019
    assert m.tolist() == [[12, 12, 12, 1, 1]]


def test_eta_and_auc_by_month():
    cls = np.repeat([0, 1], 100)
    assert eta(cls.astype(float), cls) > 0.99 and np.isnan(eta(np.ones(200), cls))
    # Merkmal trennt nur ueber den Monat: Januar ohne Feuer und niedrig, Juli halb Feuer und
    # hoch, innerhalb des Juli Zufall -> global hohe AUC, je Monat ~0,5
    rng = np.random.default_rng(3)
    month = np.repeat([1, 7, 7], [600, 200, 200])
    y = np.repeat([0, 0, 1], [600, 200, 200])
    x = np.where(month == 7, 1.0, 0.0) + 0.01 * rng.normal(size=1000)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, x) > 0.7
    assert abs(auc_by_month(x, y, month) - 0.5) < 0.1


def test_channel_lists_are_valid():
    for tgt, src in DETERMINISTIC.items():
        assert tgt in CHANNEL_NAMES and all(s in CHANNEL_NAMES for s in src)
    dyn = {CHANNEL_NAMES[c] for c in range(len(CHANNEL_NAMES)) if c not in STATIC}
    assert set(LAG_DRIVERS) <= dyn and set(LAG_TARGETS) <= dyn
    assert LANDCOVER_CH in STATIC and CHANNEL_NAMES[STATIC[0]] == "OSM_road"
    assert CHANNEL_NAMES[37] == "TWI" and 37 in STATIC and 38 not in STATIC
