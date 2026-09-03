"""Prueft Stichprobengewichte und Gitterflaechen.

Zwei Rechnungen, bei denen ein Fehler nicht auffaellt, weil das Ergebnis
plausibel aussieht:

  * Die Landmark-Tabelle ist GESCHICHTET gezogen - alle Positiven, aber nur
    ein Bruchteil der Negativen. Ohne korrektes Gewicht ist jede Praevalenz,
    jeder Brier-Score und jede Kalibrierungskurve um Groessenordnungen
    daneben.
  * Das Gitter ist in Grad regelmaessig, nicht in Metern. Ohne
    Flaechengewicht kehrt sich die Feuerdichte zwischen Nord und Sued um.
"""

from __future__ import annotations

import numpy as np
import pytest

from wildfire.data.geometry import row_areas


def weighted_mean(values, weights):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    return float((values * weights).sum() / weights.sum())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_stratified_weights_recover_population_prevalence(seed):
    """Alle Positiven + Stichprobe der Negativen, mit Gewicht -> echte Rate."""
    rng = np.random.default_rng(seed)
    n_pop = 2_000_000
    true_prev = 4e-4
    pop = rng.random(n_pop) < true_prev
    pos_idx = np.flatnonzero(pop)
    neg_idx = np.flatnonzero(~pop)

    n_neg_sampled = 20_000
    take_neg = rng.choice(neg_idx, n_neg_sampled, replace=False)
    w_neg = neg_idx.size / n_neg_sampled

    y = np.concatenate([np.ones(pos_idx.size), np.zeros(n_neg_sampled)])
    w = np.concatenate([np.ones(pos_idx.size), np.full(n_neg_sampled, w_neg)])

    est = weighted_mean(y, w)
    exact = pop.mean()
    assert abs(est - exact) / exact < 0.02, f"{est:.3e} statt {exact:.3e}"
    # ohne Gewichte waere es voellig daneben
    assert y.mean() > 20 * exact


def test_unweighted_mean_is_biased():
    """Gegenprobe: ohne Gewicht ist die Datei-Praevalenz stark ueberhoeht."""
    y = np.concatenate([np.ones(100), np.zeros(1000)])
    w = np.concatenate([np.ones(100), np.full(1000, 500.0)])
    assert y.mean() > 100 * weighted_mean(y, w)


def test_weights_are_invariant_to_negative_sample_size():
    """Doppelt so viele Negative, halbes Gewicht - gleiche Schaetzung."""
    for n in (1_000, 4_000, 16_000):
        y = np.concatenate([np.ones(50), np.zeros(n)])
        w = np.concatenate([np.ones(50), np.full(n, 100_000 / n)])
        assert abs(weighted_mean(y, w) - 50 / (50 + 100_000)) < 1e-9


# ------------------------------------------------------------------ Geometrie

RES = 0.04491576420597608
LAT_TOP = 83.34569206060921


def test_cell_area_shrinks_towards_the_pole():
    a = row_areas(928, LAT_TOP, RES)
    assert np.all(np.diff(a) > 0), "Flaeche muss nach Sueden zunehmen"


def test_cell_area_magnitudes_full_grid():
    """Ueber das GANZE Gitter, inklusive der Zeilen im Nordpolarmeer."""
    a = row_areas(928, LAT_TOP, RES)
    assert 2.8e6 < a.min() < 3.1e6, f"kleinste Zelle {a.min()/1e6:.2f} km2 (Zeile 0, 83.3degN)"
    assert 18.0e6 < a.max() < 19.0e6, f"groesste Zelle {a.max()/1e6:.2f} km2 (Zeile 927, 41.7degN)"
    assert a.max() / a.min() > 6


def test_area_matches_independent_cos_approximation():
    """Gegen eine unabhaengige Naeherung statt gegen eine gemerkte Zahl.

    Die Zellflaeche ist naeherungsweise
        (Kantenlaenge N-S) x (Kantenlaenge O-W bei cos(Breite)).
    Die exakte Ellipsoidrechnung darf davon nur um Bruchteile eines
    Prozents abweichen. Das prueft die Formel, nicht ein Ergebnis, das ich
    mir vorher aufgeschrieben habe - und es haengt an keiner Maske.
    """
    a = row_areas(928, LAT_TOP, RES)
    for lat, tol in ((45.0, 0.004), (60.0, 0.004), (75.0, 0.006)):
        i = int((LAT_TOP - lat) / RES)
        centre = LAT_TOP - RES * (i + 0.5)
        ns = RES * (111132.92 - 559.82 * np.cos(2 * np.radians(centre)))
        ew = RES * 111412.84 * np.cos(np.radians(centre))
        approx = ns * ew
        rel = abs(a[i] - approx) / approx
        assert rel < tol, f"{lat}degN: exakt {a[i]/1e6:.3f} vs Naeherung {approx/1e6:.3f} km2 ({100*rel:.2f} %)"


def test_area_at_equator_matches_nominal_5km():
    """Am Aequator muss die Zelle rund 5 x 5 km sein - dort gilt das Etikett."""
    a = row_areas(3, 1.5 * RES, RES)
    assert abs(a[1] / 1e6 - 25.0) < 0.5, f"{a[1]/1e6:.2f} km2 statt ~25"


def test_total_area_of_a_full_band_matches_sphere():
    """Ein ganzer Breitenring muss die analytische Kugelzonenflaeche ergeben."""
    n_cols = round(360 / RES)
    a = row_areas(2, 0.5 * RES, RES)          # Zeile um den Aequator
    band = a[0] * n_cols
    R = 6371008.8
    exact = 2 * np.pi * R ** 2 * abs(np.sin(np.radians(0.5 * RES))
                                     - np.sin(np.radians(-0.5 * RES)))
    assert abs(band - exact) / exact < 0.01, f"{band/1e12:.3f} vs {exact/1e12:.3f} Mio km2"
