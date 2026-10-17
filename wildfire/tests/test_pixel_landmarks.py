import numpy as np

from wildfire.paper.pixel_landmarks import H_FOLLOW, pixel_events
from wildfire.paper.surv_prepare import DAY_BITS


def test_pixel_events_first_fire_per_pixel_and_censoring():
    H, W = 50, 60
    fires = [(20, 30, 105), (20, 30, 140), (21, 30, 150), (20, 31, 300), (26, 36, 101)]
    keys = np.sort(np.array([(r * W + c) * DAY_BITS + d for r, c, d in fires], np.int64))
    ev, t = pixel_events(keys, np.array([100]), np.array([20]), np.array([30]), H, W)
    assert ev.shape == (1, 13, 13)
    assert ev[0, 6, 6] and t[0, 6, 6] == 5              # erstes Feuer nach L, nicht das zweite
    assert ev[0, 7, 6] and t[0, 7, 6] == 50
    assert not ev[0, 6, 7] and t[0, 6, 7] == H_FOLLOW    # nach 200 Tagen: zensiert
    assert ev[0, 12, 12] and t[0, 12, 12] == 1          # Ecke des Ausschnitts
    assert ev.sum() == 3


def test_pixel_events_outside_grid_is_censored():
    keys = np.array([5 * DAY_BITS + 3], np.int64)
    ev, t = pixel_events(keys, np.array([1]), np.array([0]), np.array([0]), 10, 10)
    assert not ev[0, :6].any() and not ev[0, :, :6].any()
    assert (t[0, :6] == H_FOLLOW).all()
