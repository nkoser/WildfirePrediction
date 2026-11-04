import math

import numpy as np
import pytest
import torch

from wildfire.eval.metrics import antolini_c, cif_from_hazard, d_calibration
from wildfire.models.surv_heads import HEADS, CoxHead, DeepHitHead, LogisticHazardHead, head_loss, interval_nll, make_head
from wildfire.paper.linear_tm import NAMES, LinearTM, features, select
from wildfire.paper.sts_dctm import STSDCTM

TAU = torch.arange(0, 101, dtype=torch.float64)


@pytest.mark.parametrize("kind", HEADS)
def test_heads_start_at_one_are_monotone_and_hit_init_cif(kind):
    head = make_head(kind, 100.0, 12, 0.05)
    raw = torch.zeros(3, head.n_out)
    ls = head.log_survival(raw, TAU)
    assert torch.allclose(ls[:, 0], torch.zeros(3, dtype=torch.float64))
    assert (ls[:, 1:] <= ls[:, :-1] + 1e-12).all()
    assert abs(-math.expm1(float(ls[0, -1])) - 0.05) < 1e-6
    torch.manual_seed(0)
    raw = torch.randn(5, head.n_out, requires_grad=True)
    loss = head_loss(head, raw, torch.tensor([1, 0, 1, 0, 1]).bool(), torch.tensor([3, 100, 1, 40, 99]))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(raw.grad).all()


def test_logistic_hazard_nll_is_the_discrete_likelihood():
    head = LogisticHazardHead(10, 0.1)
    raw = torch.randn(2, 10, dtype=torch.float64)
    h = torch.sigmoid(raw + head.bias)
    ll_ev = torch.log1p(-h[0, :3]).sum() + torch.log(h[0, 3])      # Ereignis an Tag 4
    ll_ce = torch.log1p(-h[1, :10]).sum()                          # zensiert bei 10
    nll = interval_nll(head, raw, torch.tensor([True, False]), torch.tensor([4, 10]))
    assert torch.allclose(nll, -(ll_ev + ll_ce) / 2)


def test_deephit_survival_is_the_tail_of_the_softmax():
    head = DeepHitHead(10, 0.1)
    raw = torch.randn(4, 11, dtype=torch.float64)
    p = torch.softmax(raw + head.bias, -1)
    S = head.log_survival(raw, torch.arange(11.0)).exp()
    for k in range(11):
        assert torch.allclose(S[:, k], p[:, k:].sum(-1))


def test_cox_partial_likelihood_and_breslow():
    head = CoxHead(10, 0.1)
    eta = torch.tensor([0.5, -0.2, 0.1, 0.0], dtype=torch.float64)
    ev, t = torch.tensor([True, True, False, True]), torch.tensor([2, 5, 5, 7])
    # Risikomengen: t=2 alle; t=5 {1,2,3}; t=7 {3}
    e = eta.exp()
    ll = (eta[0] - e.sum().log()) + (eta[1] - e[1:].sum().log()) + (eta[3] - e[3].log())
    assert torch.allclose(head.loss(eta[:, None], ev, t), -ll / 3)
    head.fit_baseline(torch.zeros(4), ev, t)                       # eta = 0: Nelson-Aalen
    na = np.cumsum([0, 0, 1 / 4, 0, 0, 1 / 3, 0, 1 / 1, 0, 0, 0])
    assert np.allclose(head.base.numpy(), na)


def test_cloglog_shift_is_proportional_hazards():
    head = make_head("dctm_cloglog", 100.0, 12, 0.05)
    raw = torch.randn(1, head.n_out, dtype=torch.float64)
    shifted = raw.clone(); shifted[:, 0] += 0.7
    lam, lam2 = -head.log_survival(raw, TAU[1:]), -head.log_survival(shifted, TAU[1:])
    assert torch.allclose(lam2 / lam, torch.full_like(lam, math.exp(0.7)))


@pytest.mark.parametrize("kind", HEADS)
def test_sts_with_every_head(kind):
    net = STSDCTM(d=32, heads=4, depth=1, head=kind).eval()
    raw_px, raw_pa = net(torch.randn(2, 30, 55, 13, 13), torch.tensor([150.0, 200.0]))
    assert raw_px.shape == (2, 13, 13, net.px_head.n_out) and raw_pa.shape == (2, net.pa_head.n_out)
    hz = net.pa_head.hazards(raw_pa, 100)
    assert hz.shape == (2, 100) and (hz >= 0).all() and (hz <= 1).all()


def test_antolini_matches_brute_force():
    rng = np.random.default_rng(0)
    n, H = 60, 12
    F = cif_from_hazard(rng.uniform(0, 0.2, (n, H)))
    ev = rng.random(n) < 0.6
    T = np.where(ev, rng.integers(1, H + 1, n), H)
    conc = tot = 0.0
    for i in np.flatnonzero(ev):
        for j in range(n):
            if j != i and (T[j] > T[i] or (T[j] == T[i] and not ev[j])):
                tot += 1
                a, b = F[i, T[i] - 1], F[j, T[i] - 1]
                conc += 1.0 if a > b else 0.5 if a == b else 0.0
    assert abs(antolini_c(F, ev, T) - conc / tot) < 1e-12


def test_d_calibration_of_the_true_model_is_flat():
    rng = np.random.default_rng(1)
    n, H = 40000, 30
    h = np.full((n, H), 0.03)
    F = cif_from_hazard(h)
    T = rng.geometric(0.03, n)
    ev = T <= H
    T = np.minimum(T, H)
    d = d_calibration(F, ev, T)
    assert abs(sum(d["d_cal_bins"]) - 1) < 1e-9 and d["d_cal_max_dev"] < 0.01
    bad = d_calibration(cif_from_hazard(h * 3), ev, T)
    assert bad["d_cal_max_dev"] > 3 * d["d_cal_max_dev"]


def test_linear_tm_features_and_shift():
    X = np.random.default_rng(0).integers(0, 3, size=(5, 30, 55, 13, 13), dtype=np.uint8)
    Z = features(X, np.array([300, 100, 200]), rows=np.array([4, 0, 2]))
    assert Z.shape == (3, len(NAMES))
    assert np.isclose(Z[0, 0], X[4, -7:, 1, 6, 6].mean())
    assert list(select(np.array([0.1, 2.0, 1.0]), np.array([[0, 0.9, 0.1], [0.9, 0, 0.2], [0.1, 0.2, 0]]), 0.7)) == [1, 2]
    m = LinearTM(len(NAMES), "cloglog")
    x = torch.from_numpy((Z - Z.mean(0)) / (Z.std(0) + 1)).double()
    assert m.raw(x).shape == (3, 13)


def test_clayton_aggregation_stays_within_the_frechet_bounds():
    from wildfire.paper.patch_consistency import clayton_log_s
    torch.manual_seed(0)
    ls = -torch.rand(4, 169, 5, dtype=torch.float64) * 0.05             # S_i in [0.95, 1]
    lo = torch.log((1 + torch.expm1(ls).sum(1)).clamp_min(1e-300))       # log(1 - sum F_i)
    hi = ls.min(1).values
    for th in (0.0, 0.1, 1.0, 10.0, float("inf")):
        a = clayton_log_s(ls, th)
        assert (a <= hi + 1e-12).all() and (a >= lo - 1e-12).all()
    assert torch.allclose(clayton_log_s(ls, 1e-6), ls.sum(1), atol=1e-6)
    assert torch.allclose(clayton_log_s(ls, 1e4), hi, atol=1e-3)
