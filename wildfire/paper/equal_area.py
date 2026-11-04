"""Flaechentreue Ausschnitte: 13 x 13 quadratische Zellen gleicher Flaeche statt 13 x 13 Gitterpixel.

Das Gitter ist EPSG:4326 mit 0.0449 Grad (data/geometry.py). Nord-Sued ist ein Pixel ueberall
rund 5 km lang, Ost-West aber nur 5 km * cos(Breite) breit: 3.3 km bei 49 Grad N, 1.7 km bei
70 Grad N. Ein 13 x 13-Ausschnitt deckt deshalb im Sueden 65 x 43 km ab, im Norden 65 x 22 km,
und das Ziel "Feuer im Ausschnitt" haengt von der Breite ab.

Hier wird jeder Ausschnitt lokal neu gerastert, ohne globale Umprojektion:

  Zeilen   wie im Gitter (eine Gitterzeile = eine Zelle, Nord-Sued-Laenge ns(phi))
  Spalten  je Gitterzeile R eine eigene Breite: Zelle k (k = -6 .. 6) ueberdeckt die
           Gitterspalten [c + 0.5 + (k - 0.5) s_R, c + 0.5 + (k + 0.5) s_R) mit
           s_R = ns(phi_R) / ew(phi_R) >= 1 Gitterpixeln, also genau so breit wie hoch.
           Die Mittelzelle ist um das Pixel c zentriert.

Alle Zellen sind damit quadratisch mit rund 5 x 5 km (Flaechenunterschied ueber Kanada < 1 %),
der Ausschnitt deckt ueberall 65 x 65 km ab.

Aggregation der Gitterpixel einer Zelle:
  stetige Kanaele   flaechengewichtetes Mittel (Anteil jedes Pixels an der Zelle), gerundet auf uint8
  Feuer (Kanal 0)   irgendein Pixel der Zelle brennt (Zuordnung ueber die Pixelmitte)
  Landbedeckung     Klasse mit der groessten Flaeche in der Zelle
  Ereignisse        erster Feuertag ueber die Pixel der Zelle (Zuordnung ueber die Pixelmitte)

Die Zuordnung ueber die Pixelmitte zerlegt die Pixel ohne Ueberlappung: jedes Pixel gehoert zu
genau einer Zelle, weil s_R >= 1 bekommt jede Zelle mindestens eines. Ausserhalb des Gitters
ist alles 0, wie bei den bisherigen Fenstern.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HALF = 6
P = 2 * HALF + 1
CUBE_FIRE = 0           # Wuerfel: dynamischer Kanal 0 = MCD14A1 (Cache-Kanal 0)
CUBE_LANDCOVER = 12     # Wuerfel: statischer Kanal 12 = Landbedeckung IGBP (Cache-Kanal 54)


def ns_m_per_deg(lat: np.ndarray) -> np.ndarray:
    """Meridianbogen je Grad (m), WGS84-Reihe wie in geometry.py."""
    p = np.radians(lat)
    return 111132.92 - 559.82 * np.cos(2 * p) + 1.175 * np.cos(4 * p)


def ew_m_per_deg(lat: np.ndarray) -> np.ndarray:
    """Breitenkreisbogen je Grad (m)."""
    p = np.radians(lat)
    return 111412.84 * np.cos(p) - 93.5 * np.cos(3 * p) + 0.118 * np.cos(5 * p)


class EqualAreaGrid:
    """Spaltengewichte je Gitterzeile; ein Ausschnitt um (r, c) nutzt die Zeilen r - 6 .. r + 6.

    weights(R)  (13, 2 * wh + 1) float32: Anteil von Gitterspalte c + j an Zelle k, Zeilen summieren auf 1
    assign(R)   (2 * wh + 1,) int: Zelle (0..12) der Gitterspalte c + j nach ihrer Mitte, sonst -1
    mit j = -wh .. wh, wh = halbe Fensterbreite dieser Zeile.
    """

    def __init__(self, H: int, W: int, lat_top: float, res: float):
        self.H, self.W, self.res = H, W, res
        lat = lat_top - res * (np.arange(H) + 0.5)
        self.lat = lat
        self.scale = ns_m_per_deg(lat) / ew_m_per_deg(lat)                 # Gitterpixel je Zelle, Ost-West
        self.cell_km = ns_m_per_deg(lat) * res / 1000                       # Kantenlaenge der Zelle
        self.wh_row = np.ceil((HALF + 0.5) * self.scale - 0.5).astype(int) + 1
        self._w: dict[int, np.ndarray] = {}
        self._a: dict[int, np.ndarray] = {}

    @classmethod
    def from_experiment(cls, exp: Path) -> "EqualAreaGrid":
        man = json.loads((exp / "data_manifests" / "d0_masks.json").read_text())
        tr = man["grid"]["transform"]
        return cls(man["grid"]["height"], man["grid"]["width"], tr[5], abs(tr[0]))

    def half_width(self, r: int) -> int:
        """Halbe Fensterbreite in Gitterspalten, die alle 13 Zeilen des Ausschnitts um r abdeckt."""
        lo, hi = max(0, r - HALF), min(self.H, r + HALF + 1)
        return int(self.wh_row[lo:hi].max()) if hi > lo else 1

    def _row(self, R: int, wh: int):
        key = (R, wh)
        if key not in self._w:
            s = float(self.scale[R])
            j = np.arange(-wh, wh + 1)
            left = 0.5 + (np.arange(P) - HALF - 0.5)[:, None] * s           # Zellgrenzen relativ zu Spalte c
            right = left + s
            ov = np.clip(np.minimum(right, j[None] + 1.0) - np.maximum(left, j[None].astype(float)), 0, None)
            self._w[key] = (ov / s).astype(np.float32)
            mid = j + 0.5
            k = np.floor((mid - 0.5) / s + HALF + 0.5).astype(int)
            self._a[key] = np.where((k >= 0) & (k < P), k, -1)
        return self._w[key], self._a[key]

    def tables(self, r: int, wh: int | None = None):
        """Gewichte (13, 13, 2wh+1) und Zuordnung (13, 2wh+1) fuer die 13 Zeilen um r; Zeilen ausserhalb: 0 / -1."""
        wh = self.half_width(r) if wh is None else wh
        Wt = np.zeros((P, P, 2 * wh + 1), np.float32)
        A = np.full((P, 2 * wh + 1), -1, int)
        for i in range(P):
            R = r - HALF + i
            if 0 <= R < self.H:
                Wt[i], A[i] = self._row(R, wh)
        return Wt, A, wh


def read_wide(arr, r: int, c: int, wh: int, H: int, W: int, lead: tuple = ()) -> np.ndarray:
    """Gitterausschnitt Zeilen r-6..r+6, Spalten c-wh..c+wh aus arr[..., H, W, C], ausserhalb 0.

    lead: Indizes vor den Gitterachsen (z. B. ein Tagesslice)."""
    r0, r1 = max(0, r - HALF), min(H, r + HALF + 1)
    c0, c1 = max(0, c - wh), min(W, c + wh + 1)
    src = np.asarray(arr[lead + (slice(r0, r1), slice(c0, c1))])
    out = np.zeros(src.shape[:len(src.shape) - 3] + (P, 2 * wh + 1, src.shape[-1]), src.dtype)
    if r1 > r0 and c1 > c0:
        out[..., r0 - (r - HALF):r1 - (r - HALF), c0 - (c - wh):c1 - (c - wh), :] = src
    return out


def resample(wide: np.ndarray, Wt: np.ndarray, A: np.ndarray, binary=(), categorical=()) -> np.ndarray:
    """(..., 13, 2wh+1, C) uint8 -> (..., 13, 13, C) uint8.

    stetig: flaechengewichtetes Mittel; binary: Maximum ueber die Pixel der Zelle (Pixelmitte);
    categorical: Klasse mit der groessten Flaeche."""
    lead = wide.shape[:-3]
    C = wide.shape[-1]
    x = wide.astype(np.float32)
    out = np.empty(lead + (P, P, C), np.float32)
    for i in range(P):                                                 # je Zeile (13, w) @ (..., w, C)
        out[..., i, :, :] = np.matmul(Wt[i], x[..., i, :, :])
    res = np.rint(np.clip(out, 0, 255)).astype(np.uint8)
    for ch in binary:
        v = wide[..., ch]                                              # (..., 13, 2wh+1)
        m = np.zeros(lead + (P, P), np.uint8)
        for k in range(P):
            sel = A == k                                               # (13, 2wh+1)
            m[..., k] = np.where(sel, v, 0).max(-1)
        res[..., ch] = m
    for ch in categorical:
        v = wide[..., ch]
        classes = np.unique(v)
        best = np.full(lead + (P, P), -1.0, np.float32)
        lab = np.zeros(lead + (P, P), np.uint8)
        for cl in classes:
            area = np.einsum("ikj,...ij->...ik", Wt, (v == cl).astype(np.float32))
            better = area > best
            lab = np.where(better, cl, lab)
            best = np.where(better, area, best)
        res[..., ch] = lab
    assert res.shape[-1] == C
    return res


def window_ea(cube, grid: EqualAreaGrid, L: int, r: int, c: int, T: int) -> np.ndarray:
    """Wie surv_prepare.window, aber flaechentreu: (T, 55, 13, 13) uint8 bis einschliesslich L."""
    from wildfire.paper.surv_prepare import CACHE_FROM_CUBE
    year, li = cube._pos[L]
    dyn, stat = cube._arrays(year)
    Wt, A, wh = grid.tables(r)
    d = read_wide(dyn, r, c, wh, cube.H, cube.W, lead=(slice(li - T + 1, li + 1),))   # (T, 13, w, 42)
    s = read_wide(stat, r, c, wh, cube.H, cube.W)                                       # (13, w, 13)
    d = resample(d, Wt, A, binary=(CUBE_FIRE,))                                         # (T, 13, 13, 42)
    s = resample(s, Wt, A, categorical=(CUBE_LANDCOVER,))                               # (13, 13, 13)
    out = np.empty((T, 55, P, P), np.uint8)
    for ch, (src, i) in enumerate(CACHE_FROM_CUBE):
        out[:, ch] = d[..., i] if src == "dyn" else s[None, ..., i]
    return out


def cell_index(grid: EqualAreaGrid, rr: np.ndarray, cc: np.ndarray):
    """Gitterpixel je Ausschnitt und ihre Zelle: (flat (n, m) int64, cell (n, m) int, ok (n, m) bool).

    m = 13 * (2 wh_max + 1); cell = i * 13 + k, -1 fuer Pixel ausserhalb aller Zellen oder des Gitters."""
    whs = np.array([grid.half_width(int(r)) for r in rr])
    wmax = int(whs.max()) if whs.size else 1
    n, m = rr.size, P * (2 * wmax + 1)
    flat = np.zeros((n, m), np.int64)
    cell = np.full((n, m), -1, np.int64)
    j = np.arange(-wmax, wmax + 1)
    for q in range(n):
        r, c = int(rr[q]), int(cc[q])
        _, A, _ = grid.tables(r, wmax)
        R = (r - HALF + np.arange(P))[:, None]
        C = c + j[None]
        ok = (R >= 0) & (R < grid.H) & (C >= 0) & (C < grid.W) & (A >= 0)
        flat[q] = np.where(ok, R * grid.W + C, 0).ravel()
        cell[q] = np.where(ok, np.arange(P)[:, None] * P + A, -1).ravel()
    return flat, cell, cell >= 0


def ea_pixel_events(keys: np.ndarray, Ls: np.ndarray, rr: np.ndarray, cc: np.ndarray, grid: EqualAreaGrid,
                    horizon: int = 100, day_bits: int = 1 << 15, chunk: int = 5000):
    """Wie pixel_landmarks.pixel_events je Zelle: erster Feuertag > L ueber die Pixel der Zelle.

    -> (event (n, 13, 13) bool, t (n, 13, 13) int16), zensiert bei horizon."""
    n = Ls.size
    event = np.zeros((n, P, P), bool)
    t = np.full((n, P, P), horizon, np.int16)
    for a in range(0, n, chunk):
        sl = slice(a, a + chunk)
        flat, cell, ok = cell_index(grid, rr[sl], cc[sl])
        pos = np.searchsorted(keys, flat * day_bits + (Ls[sl, None] + 1))
        nxt = np.where(pos < keys.size, keys[np.minimum(pos, keys.size - 1)], -1)
        same = ok & (nxt // day_bits == flat)
        tte = np.where(same, nxt % day_bits - Ls[sl, None], 10 ** 6)
        best = np.full((flat.shape[0], P * P), 10 ** 6, np.int64)
        qi = np.repeat(np.arange(flat.shape[0]), flat.shape[1]).reshape(flat.shape)
        np.minimum.at(best, (qi[ok], cell[ok]), tte[ok])
        best = best.reshape(-1, P, P)
        event[sl] = best <= horizon
        t[sl] = np.minimum(best, horizon).astype(np.int16)
    return event, t


def ea_fire_between(keys: np.ndarray, lo: np.ndarray, hi: np.ndarray, rr: np.ndarray, cc: np.ndarray,
                    grid: EqualAreaGrid, day_bits: int = 1 << 15, chunk: int = 5000) -> np.ndarray:
    """(n, 13, 13) bool: Feuer in der Zelle an irgendeinem Tag lo .. hi (einschliesslich)."""
    n = rr.size
    out = np.zeros((n, P, P), bool)
    for a in range(0, n, chunk):
        sl = slice(a, a + chunk)
        flat, cell, ok = cell_index(grid, rr[sl], cc[sl])
        pos = np.searchsorted(keys, flat * day_bits + lo[sl, None])
        nxt = np.where(pos < keys.size, keys[np.minimum(pos, keys.size - 1)], -1)
        hit = ok & (nxt // day_bits == flat) & (nxt % day_bits <= hi[sl, None])
        m = np.zeros((flat.shape[0], P * P), bool)
        qi = np.repeat(np.arange(flat.shape[0]), flat.shape[1]).reshape(flat.shape)
        m[qi[hit], cell[hit]] = True
        out[sl] = m.reshape(-1, P, P)
    return out


def ea_daily_fire(keys: np.ndarray, Ls: np.ndarray, rr: np.ndarray, cc: np.ndarray, grid: EqualAreaGrid,
                  horizon: int = 100, day_bits: int = 1 << 15, chunk: int = 5000):
    """Wie rate_labels.daily_fire je Zelle: (patch (n, horizon), centre (n, horizon)) bool."""
    n = Ls.size
    patch = np.zeros((n, horizon), bool)
    centre = np.zeros((n, horizon), bool)
    mid = (P * P) // 2
    for a in range(0, n, chunk):
        sl = slice(a, a + chunk)
        flat, cell, ok = cell_index(grid, rr[sl], cc[sl])
        L = Ls[sl, None].astype(np.int64)
        lo = np.searchsorted(keys, flat * day_bits + L + 1)
        hi = np.searchsorted(keys, flat * day_bits + L + horizon, side="right")
        cnt = np.where(ok, hi - lo, 0).ravel()
        if not cnt.sum():
            continue
        px = np.repeat(np.arange(cnt.size), cnt)
        start = np.repeat(np.cumsum(cnt) - cnt, cnt)
        pos = np.repeat(lo.ravel(), cnt) + np.arange(cnt.sum()) - start
        i = px // flat.shape[1]
        tau = keys[pos] % day_bits - L.ravel()[i]
        patch[a + i, tau - 1] = True
        c = cell.ravel()[px] == mid
        centre[a + i[c], tau[c] - 1] = True
    return patch, centre


def summary(grid: EqualAreaGrid) -> dict:
    """Kennzahlen fuer Manifeste: Zellgroesse und Ausschnittsbreite je Breite."""
    out = {}
    for la in (45, 49, 55, 60, 65, 70):
        R = int(np.argmin(np.abs(grid.lat - la)))
        out[str(la)] = {"cell_km": round(float(grid.cell_km[R]), 3),
                        "grid_px_per_cell_ew": round(float(grid.scale[R]), 3),
                        "pixel_ew_km": round(float(grid.cell_km[R] / grid.scale[R]), 3),
                        "half_width_px": int(grid.wh_row[R])}
    return out
