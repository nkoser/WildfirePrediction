import numpy as np
import torch

from wildfire.paper.sts_dctm import EMA, STSDCTM, doy_of, lr_factor, patch_events, spatial_holdout


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


def test_backbone_variants_run_and_differ_only_in_the_trunk():
    import pytest
    from wildfire.paper.sts_dctm import Block
    x = torch.randn(2, 30, 55, 13, 13)
    doy = torch.tensor([120.0, 200.0])
    for attn, stem in (("joint", "2d"), ("local", "2d"), ("factorized", "3d")):
        net = STSDCTM(d=32, heads=4, depth=1, head="lh", attn=attn, stem=stem).eval()
        raw_px, raw_pa = net(x, doy)
        assert raw_px.shape == (2, 13, 13, 100) and torch.isfinite(raw_px).all()
    # lokale Attention: Randpixel haben weniger gueltige Nachbarn, Mitte alle 9
    kv, pad = Block._neighbour_keys(torch.randn(1, 169, 30, 8), 169)
    assert kv.shape == (169, 270, 8)
    assert int((~pad[0]).sum()) == 4 * 30 and int((~pad[6 * 13 + 6]).sum()) == 9 * 30
    with pytest.raises(ValueError):
        Block(8, 2, 0.0, "bogus")


def test_ema_tracks_weights_and_keeps_integer_buffers():
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2))
    ema = EMA(net, decay=0.9)
    w0 = net[0].weight.detach().clone()
    with torch.no_grad():
        net[0].weight.add_(1.0)                           # Gewichte springen um +1
    net[1].num_batches_tracked.fill_(7)
    ema.update(net)                                       # n = 1: d = min(0.9, 2 / 11)
    d = 2 / 11
    sd = ema.state(net.state_dict())
    assert torch.allclose(sd["0.weight"], w0 + (1 - d))
    assert int(sd["1.num_batches_tracked"]) == 7 and sd["1.num_batches_tracked"].dtype == torch.long
    for _ in range(200):                                  # konstante Gewichte: EMA laeuft gegen sie
        ema.update(net)
    assert torch.allclose(ema.state(net.state_dict())["0.weight"], net[0].weight, atol=1e-6)


def test_spatial_holdout_keeps_a_buffer_around_held_blocks():
    rng = np.random.default_rng(0)
    row, col = rng.integers(0, 2000, 50000), rng.integers(0, 2000, 50000)
    hold, buf = spatial_holdout(row, col, share=0.2, block=64, buffer=13)
    assert not np.any(hold & buf)
    assert 0.1 < hold.mean() < 0.3
    tr = ~hold & ~buf
    # kein Trainings-Stichtag liegt naeher als 13 Pixel (Chebyshev) an einem herausgehaltenen
    d = np.maximum(np.abs(row[tr][:, None] - row[hold][None, :400]), np.abs(col[tr][:, None] - col[hold][None, :400]))
    assert d.min() > 13
    # gleicher Seed -> gleiche Bloecke
    hold2, _ = spatial_holdout(row, col, share=0.2, block=64, buffer=13)
    assert np.array_equal(hold, hold2)


def test_interval_nll_matches_bernstein_log_likelihood_when_censored_at_horizon():
    # Survival-NLL fuer Auswahl und Diagnose laeuft fuer alle Koepfe ueber interval_nll; beim
    # Bernstein-Kopf muss das die bisherige log_likelihood treffen (Zensur bei 100 Tagen)
    from wildfire.models.surv_heads import interval_nll, make_head
    torch.manual_seed(0)
    head = make_head("dctm", 100.0, 12, 0.05)
    raw = torch.randn(64, head.n_out, dtype=torch.float64) * 0.3
    ev = torch.rand(64) < 0.3
    t = torch.where(ev, torch.randint(1, 101, (64,)), torch.full((64,), 100))
    ll_old = head.log_likelihood(raw, ev, t, 100.0).sum()
    assert torch.allclose(-interval_nll(head, raw, ev, t) * 64, ll_old, rtol=1e-6)
    lh = make_head("lh", 100.0, 12, 0.05)                                  # neue Koepfe: kein log_likelihood noetig
    assert torch.isfinite(interval_nll(lh, torch.randn(64, lh.n_out), ev, t))


def test_lr_factor_warmup_then_constant_or_cosine():
    assert lr_factor(0, 10, 100) == 0.1 and lr_factor(9, 10, 100) == 1.0
    assert lr_factor(50, 10, 100, "const") == 1.0
    f = [lr_factor(s, 10, 100, "cosine", 0.1) for s in range(10, 101)]
    assert abs(f[0] - 1.0) < 1e-12 and abs(f[-1] - 0.1) < 1e-12
    assert all(a >= b for a, b in zip(f, f[1:]))                        # faellt monoton
    assert abs(lr_factor(55, 10, 100, "cosine") - 0.5) < 1e-12          # Mitte: halbe Lernrate
