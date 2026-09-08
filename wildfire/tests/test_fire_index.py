"""Prueft die CSR-Abfragen des Feuerindex gegen eine naive Referenz.

Der Index kodiert (Pixel, Tag) als ``pixel * DAY_BITS + tag`` in EIN
sortiertes int64-Array, damit Abfragen vektorisiert per ``searchsorted``
laufen. Diese Kodierung ist der Kern der Labelableitung - ein Fehler darin
verschiebt still jedes Ereignis. Deshalb wird sie hier gegen eine
offensichtlich korrekte, langsame Implementierung geprueft.
"""

from __future__ import annotations

import numpy as np
import pytest

from wildfire.data.landmarks import DAY_BITS, FireIndex


class FakeIndex(FireIndex):
    """FireIndex ohne Datei, direkt aus (Pixel, Tag)-Paaren gebaut."""

    def __init__(self, W, fire_pairs, ign_pairs, events=None):
        self.H, self.W = 100, W
        self.dates = [f"d{i}" for i in range(DAY_BITS // 100)]
        self.fire_keys = np.sort(np.array(
            [p * DAY_BITS + t for p, t in fire_pairs], np.int64))
        order = sorted(range(len(ign_pairs)),
                       key=lambda i: (ign_pairs[i][0], ign_pairs[i][1]))
        self.ign_keys = np.array(
            [ign_pairs[i][0] * DAY_BITS + ign_pairs[i][1] for i in order], np.int64)
        self.ign_events = np.array(
            [(events or list(range(len(ign_pairs))))[i] for i in order], np.int32)
        # Seit der Umstellung auf gemeldete Braende traegt der Index auch
        # Ursache und Flaeche. Der Doppelgaenger muss dieselbe Schnittstelle
        # anbieten, sonst prueft er ein anderes Objekt als das echte.
        self.ign_cause = np.zeros(self.ign_keys.size, np.int8)
        self.ign_size = np.full(self.ign_keys.size, np.nan, np.float32)
        self.label_source = "test"


def brute_fire_between(pairs, p, lo, hi):
    return any(pp == p and lo <= tt <= hi for pp, tt in pairs)


def brute_next_ign(pairs, p, t):
    cand = sorted(tt for pp, tt in pairs if pp == p and tt > t)
    return cand[0] if cand else None


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_queries_match_bruteforce(seed):
    rng = np.random.default_rng(seed)
    W = 50
    pixels = rng.integers(0, 500, size=40).tolist()
    fire = sorted({(int(p), int(t)) for p in pixels
                   for t in rng.integers(0, 200, size=int(rng.integers(1, 6)))})
    ign = sorted({(p, t) for p, t in fire if rng.random() < 0.4})
    idx = FakeIndex(W, fire, ign)

    qp = np.array([int(p) for p in rng.integers(0, 500, size=200)], np.int64)
    qt = np.array([int(t) for t in rng.integers(0, 200, size=200)], np.int64)

    got = idx.has_fire_between(qp, qt - 6, qt)
    exp = np.array([brute_fire_between(fire, int(p), int(t) - 6, int(t))
                    for p, t in zip(qp, qt)])
    assert np.array_equal(got, exp)

    day, _, _, found = idx.next_ignition(qp, qt)
    for i, (p, t) in enumerate(zip(qp.tolist(), qt.tolist())):
        e = brute_next_ign(ign, p, t)
        if e is None:
            assert not found[i], f"Treffer erfunden fuer Pixel {p}, t={t}"
        else:
            assert found[i] and day[i] == e, f"Pixel {p}, t={t}: {day[i]} statt {e}"


def test_empty_pixel_never_matches():
    idx = FakeIndex(50, [(10, 5)], [(10, 5)])
    q = np.array([999], np.int64)
    assert not idx.has_fire_between(q, np.array([0]), np.array([100]))[0]
    _, _, _, found = idx.next_ignition(q, np.array([0]))
    assert not found[0]


def test_next_ignition_is_strictly_after_t():
    """Eine Zuendung AM Landmarktag darf nicht als zukuenftig zaehlen."""
    idx = FakeIndex(50, [(7, 20)], [(7, 20)])
    q = np.array([7], np.int64)
    _, _, _, found = idx.next_ignition(q, np.array([20], np.int64))
    assert not found[0], "Zuendung bei t selbst wurde als zukuenftig gewertet"
    day, _, _, found = idx.next_ignition(q, np.array([19], np.int64))
    assert found[0] and day[0] == 20


def test_key_encoding_has_no_collisions():
    """Tag muss unterhalb DAY_BITS bleiben, sonst laufen Pixel ineinander."""
    assert DAY_BITS > 9497, "DAY_BITS zu klein fuer 26 Jahre Tagesraster"
    a = 5 * DAY_BITS + (DAY_BITS - 1)
    b = 6 * DAY_BITS + 0
    assert a < b
