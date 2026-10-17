import numpy as np

from wildfire.paper import compare, targets
from wildfire.paper.pixel_stdctm import pixel_targets


def test_window_moves_one_day_later_for_next_day_target():
    L = 61                                         # Cache-Index 60 = D
    for dt in (1, 7, 15, 30):
        t0, t1 = compare.window(L, dt)             # Ziel D: Eingabe D-dt-30 .. D-dt-1
        assert (t0, t1 - 1) == (60 - dt - 30, 60 - dt - 1)
        n0, n1 = compare.window(L + targets.shift("next_patch"), dt)   # Ziel D+1: D-dt-29 .. D-dt
        assert (n0, n1 - 1) == (t0 + 1, t1) and n0 >= 0 and n1 <= L


def test_target_files(tmp_path):
    np.save(tmp_path / "y.npy", np.array([1, 0, 1], np.int8))
    np.save(tmp_path / "y_next_patch.npy", np.array([0, 0, 1], np.int8))
    fn = np.zeros((3, 13, 13), np.uint8)
    fn[2, 6, 6] = 1
    np.save(tmp_path / "fire_next.npy", fn)
    X = np.zeros((3, 61, 55, 13, 13), np.uint8)
    X[0, -1, 0, 0, 0] = 1
    assert targets.labels(tmp_path, "cache").tolist() == [1, 0, 1]
    assert targets.labels(tmp_path, "next_patch").tolist() == [0, 0, 1]
    assert targets.centre_labels(tmp_path, X, "next_patch").tolist() == [0, 0, 1]
    assert targets.fire_map(tmp_path, X, "cache")[0, 0, 0] and targets.tag("cache") == ""


def test_pixel_events_reach_the_appended_next_day():
    fire = np.zeros((1, 62, 13, 13), bool)        # 61 Cache-Tage + D+1
    fire[0, 61, 4, 4] = True
    e = np.array([59])                             # letzter Eingabetag D-1 (dt = 1)
    ev, t = pixel_targets(fire, e)
    assert ev[0, 4, 4] and t[0, 4, 4] == 2         # erstes Feuer nach 2 Tagen = am Zieltag D+1
    assert not ev[0, 0, 0] and t[0, 0, 0] == 2     # sonst zensiert am Zieltag
