import numpy as np

from wildfire.paper.prepare import FIRE_CH
from wildfire.paper.surv_train import fire_in_window


def test_fire_in_window_patch_vs_centre():
    X = np.zeros((3, 30, 55, 13, 13), np.uint8)
    X[1, 10, FIRE_CH, 2, 3] = 1          # nur ein Nachbar brennt
    X[2, 29, FIRE_CH, 6, 6] = 1          # die Mittelzelle brennt am letzten Tag
    X[0, 5, FIRE_CH + 1, 6, 6] = 200     # anderer Kanal zaehlt nicht
    assert fire_in_window(X, "patch").tolist() == [False, True, True]
    assert fire_in_window(X, "centre").tolist() == [False, False, True]
