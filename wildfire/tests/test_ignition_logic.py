"""Prueft die Zuendungsdefinition an von Hand gerechneten Mustern.

Die Regel lautet: ein Feuer zaehlt als Zuendung, wenn im Umkreis R und in
den C Tagen davor kein Feuer war. Radius R trennt Entstehung von
Ausbreitung - genau daran haengt, ob unser Ziel "Zuendung" oder
"Feueraufkommen" heisst. Die Muster hier sind klein genug, um die
erwartete Antwort ohne Code zu bestimmen.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

COOLDOWN = 7


def ignitions(frames: list[np.ndarray], radius: int, cooldown: int = COOLDOWN):
    """Dieselbe Logik wie wildfire.data.events, isoliert und ohne IO."""
    struct = np.ones((2 * radius + 1, 2 * radius + 1), bool) if radius else None
    ring: list[np.ndarray] = []
    out = []
    for fire in frames:
        recent = (np.logical_or.reduce(ring) if ring
                  else np.zeros_like(fire, bool))
        blocked = recent if radius == 0 else ndimage.binary_dilation(recent, struct)
        out.append(fire & ~blocked)
        ring.append(fire)
        if len(ring) > cooldown:
            ring.pop(0)
    return out


def empty(n=5):
    return np.zeros((n, n), bool)


def test_isolated_fire_is_an_ignition():
    f = empty(); f[2, 2] = True
    res = ignitions([empty(), f], radius=1)
    assert res[1][2, 2]


def test_same_pixel_next_day_is_continuation():
    f = empty(); f[2, 2] = True
    res = ignitions([f, f], radius=0)
    assert res[0][2, 2] and not res[1][2, 2]


def test_neighbour_next_day_is_spread_not_ignition():
    """Der Kern der Radiusregel: das Feuer wandert ins Nachbarpixel."""
    a = empty(); a[2, 2] = True
    b = empty(); b[2, 3] = True
    r0 = ignitions([a, b], radius=0)
    r1 = ignitions([a, b], radius=1)
    assert r0[1][2, 3], "bei R=0 zaehlt Ausbreitung faelschlich als Zuendung"
    assert not r1[1][2, 3], "bei R=1 muss Ausbreitung unterdrueckt werden"


def test_distant_fire_stays_an_ignition():
    """Zwei Pixel Abstand ist mehr als R=1 - das ist ein neuer Brand."""
    a = empty(7); a[3, 1] = True
    b = empty(7); b[3, 4] = True
    res = ignitions([a, b], radius=1)
    assert res[1][3, 4]


def test_reignition_after_cooldown():
    f = empty(); f[2, 2] = True
    frames = [f] + [empty()] * COOLDOWN + [f]
    res = ignitions(frames, radius=0)
    assert res[0][2, 2]
    assert res[-1][2, 2], "nach vollem Cooldown muss erneut gezaehlt werden"


def test_no_reignition_one_day_short_of_cooldown():
    f = empty(); f[2, 2] = True
    frames = [f] + [empty()] * (COOLDOWN - 1) + [f]
    res = ignitions(frames, radius=0)
    assert not res[-1][2, 2], "Cooldown wurde zu frueh aufgehoben"


def test_radius_is_monotone():
    """Groesserer Radius kann nie MEHR Zuendungen liefern."""
    rng = np.random.default_rng(0)
    frames = [rng.random((12, 12)) < 0.05 for _ in range(20)]
    counts = [sum(int(x.sum()) for x in ignitions(frames, r)) for r in (0, 1, 2)]
    assert counts[0] >= counts[1] >= counts[2], counts


def test_first_frame_always_counts():
    """Ohne Vorgeschichte ist jedes Feuer eine Zuendung."""
    f = empty(); f[1, 1] = True; f[3, 3] = True
    res = ignitions([f], radius=2)
    assert int(res[0].sum()) == 2
