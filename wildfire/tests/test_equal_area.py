import numpy as np

from wildfire.paper.equal_area import (P, EqualAreaGrid, cell_index, ea_daily_fire, ea_fire_between, ea_pixel_events,
                                       read_wide, resample, window_ea)
from wildfire.paper.pixel_landmarks import pixel_events
from wildfire.paper.rate_labels import daily_fire

RES = 0.04491576420597608
DAY_BITS = 1 << 15


def grid_at(lat_centre: float, H: int = 60, W: int = 200) -> EqualAreaGrid:
    return EqualAreaGrid(H, W, lat_centre + RES * H / 2, RES)


def unit_grid(H: int = 40, W: int = 80) -> EqualAreaGrid:
    """Gitter mit s = 1: flaechentreu = Originalpixel."""
    g = grid_at(50.0, H, W)
    g.scale = np.ones(H)
    g.wh_row = np.full(H, 7)
    return g


def test_cells_are_square_and_scale_follows_latitude():
    for lat, s_exp in ((49.0, 1.52), (60.0, 1.99), (70.0, 2.91)):
        g = grid_at(lat)
        R = g.H // 2
        assert abs(g.scale[R] - s_exp) < 0.02
        assert abs(g.cell_km[R] - 4.99) < 0.03                       # Nord-Sued-Kante = Ost-West-Kante
        ew_px_km = g.cell_km[R] / g.scale[R]
        assert abs(ew_px_km * g.scale[R] - g.cell_km[R]) < 1e-9


def test_weights_sum_to_one_and_assignment_partitions_columns():
    for lat in (43.0, 55.0, 70.0, 80.0):
        g = grid_at(lat)
        Wt, A, wh = g.tables(g.H // 2)
        assert np.allclose(Wt.sum(-1), 1.0, atol=1e-5)
        for i in range(P):
            counts = np.bincount(A[i][A[i] >= 0], minlength=P)
            assert counts.min() >= 1                                  # jede Zelle hat mindestens ein Pixel
            # Pixel einer Zelle liegen zusammenhaengend, Zellen von West nach Ost
            ks = A[i][A[i] >= 0]
            assert np.all(np.diff(ks) >= 0)
        s = g.scale[g.H // 2]
        assert abs(Wt[0].sum() - P) < 1e-4 and abs((Wt[0] > 0).sum(-1).mean() - (s + 1)) < 1.5


def test_centre_cell_is_centred_on_the_landmark_pixel():
    g = grid_at(60.0)
    Wt, A, wh = g.tables(g.H // 2)
    j = np.arange(-wh, wh + 1)
    w = Wt[6, 6]
    centre = float((w * (j + 0.5)).sum())                                # Schwerpunkt in Spaltenkoordinaten relativ zu c
    assert abs(centre - 0.5) < 1e-5
    assert A[6][j == 0][0] == 6


def test_unit_scale_reproduces_the_original_window():
    g = unit_grid()
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(4, g.H, g.W, 3), dtype=np.uint8)
    r, c = 20, 40
    Wt, A, wh = g.tables(r)
    wide = read_wide(arr, r, c, wh, g.H, g.W, lead=(slice(0, 4),))
    out = resample(wide, Wt, A, binary=(0,), categorical=(1,))
    ref = arr[:, r - 6:r + 7, c - 6:c + 7]
    assert np.array_equal(out, ref)


def test_constant_field_stays_constant_and_binary_is_any():
    g = grid_at(65.0)
    r = g.H // 2
    Wt, A, wh = g.tables(r)
    wide = np.full((P, 2 * wh + 1, 3), 77, np.uint8)
    wide[..., 0] = 0
    wide[3, wh + 1, 0] = 1                                                # ein brennendes Pixel neben der Mitte
    wide[..., 2] = 5
    wide[:, :wh, 2] = 9                                                   # westliche Haelfte Klasse 9
    out = resample(wide, Wt, A, binary=(0,), categorical=(2,))
    assert np.all(out[..., 1] == 77)
    assert out[..., 0].sum() == 1 and out[3, A[3, wh + 1], 0] == 1
    assert np.all(out[:, :6, 2] == 9) and np.all(out[:, 7:, 2] == 5)


def test_pixel_events_match_the_original_with_unit_scale():
    g = unit_grid()
    rng = np.random.default_rng(1)
    n_fire = 400
    pix = rng.integers(0, g.H * g.W, n_fire)
    day = rng.integers(0, 300, n_fire)
    keys = np.unique(pix.astype(np.int64) * DAY_BITS + day)
    rr, cc = rng.integers(0, g.H, 50), rng.integers(0, g.W, 50)
    Ls = rng.integers(0, 200, 50)
    e0, t0 = pixel_events(keys, Ls, rr, cc, g.H, g.W)
    e1, t1 = ea_pixel_events(keys, Ls, rr, cc, g)
    assert np.array_equal(e0, e1) and np.array_equal(t0, t1)
    p0, c0 = daily_fire(keys, Ls, rr, cc, g.H, g.W)
    p1, c1 = ea_daily_fire(keys, Ls, rr, cc, g)
    assert np.array_equal(p0, p1) and np.array_equal(c0, c1)


def test_cell_event_is_the_first_fire_of_its_pixels():
    g = grid_at(70.0)
    r, c = g.H // 2, 100
    Wt, A, wh = g.tables(r)
    cols = np.where(A[6] == 8)[0] - wh + c                                # Gitterspalten der Zelle (6, 8)
    assert cols.size >= 2
    keys = np.sort(np.array([(r * g.W + cols[0]) * DAY_BITS + 30, (r * g.W + cols[-1]) * DAY_BITS + 12,
                             (r * g.W + c) * DAY_BITS + 50], np.int64))
    ev, t = ea_pixel_events(keys, np.array([10]), np.array([r]), np.array([c]), g)
    assert ev[0, 6, 8] and t[0, 6, 8] == 2
    assert ev[0, 6, 6] and t[0, 6, 6] == 40
    assert ev[0].sum() == 2
    hit = ea_fire_between(keys, np.array([12]), np.array([12]), np.array([r]), np.array([c]), g)
    assert hit[0, 6, 8] and hit[0].sum() == 1
    flat, cell, ok = cell_index(g, np.array([r]), np.array([c]))
    assert ok.sum() == np.sum(A >= 0)


class FakeCube:
    def __init__(self, H, W, T=40):
        rng = np.random.default_rng(2)
        self.H, self.W = H, W
        self.dyn = rng.integers(0, 255, size=(T, H, W, 42), dtype=np.uint8)
        self.dyn[..., 0] = rng.random((T, H, W)) < 0.05
        self.stat = rng.integers(0, 17, size=(H, W, 13), dtype=np.uint8)
        self._pos = {t: ("2001", t) for t in range(T)}

    def _arrays(self, year):
        return self.dyn, self.stat


def test_window_ea_equals_original_window_with_unit_scale():
    from wildfire.paper.surv_prepare import window
    g = unit_grid()
    cube = FakeCube(g.H, g.W)
    for r, c in ((20, 40), (2, 3), (38, 79)):                             # Mitte und Raender
        assert np.array_equal(window_ea(cube, g, 35, r, c, 30), window(cube, 35, r, c))


def test_window_ea_aggregates_wider_columns_at_high_latitude():
    g = grid_at(70.0)
    cube = FakeCube(g.H, g.W)
    r, c = g.H // 2, 100
    x = window_ea(cube, g, 35, r, c, 30)
    assert x.shape == (30, 55, P, P) and x.dtype == np.uint8
    # Feuerkanal: Zelle brennt genau dann, wenn eines ihrer Pixel brennt
    Wt, A, wh = g.tables(r)
    raw = cube.dyn[35, r - 6:r + 7, c - wh:c + wh + 1, 0]
    any_k = np.array([[raw[i][A[i] == k].max() for k in range(P)] for i in range(P)])
    assert np.array_equal(x[-1, 0], any_k)
    # stetiger Kanal (Temperatur, Wuerfel-dyn 1): gewichtetes Mittel
    t2m = cube.dyn[35, r - 6:r + 7, c - wh:c + wh + 1, 1].astype(float)
    ref = np.rint(np.einsum("ikj,ij->ik", Wt, t2m))
    assert np.abs(x[-1, 1].astype(int) - ref).max() <= 1
