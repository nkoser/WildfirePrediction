"""Ziel der Paper-Auswertung: Cache-Label (Feuer an D) oder Feuer am Folgetag (D+1).

  cache       /labels des Caches: Feuer im 13x13-Ausschnitt an D (letzter Cache-Tag)
  next_patch  Feuer im Ausschnitt an D+1 (next_labels.py). So passt das Paper:
              trainiert auf diesem Ziel trifft das 3D-U-Net die berichteten
              PR-AUC-Werte (p2_reconstruct_*_next_*).

Fenster wie im Paper: Zieltag t, Eingabe [t - dt - 30, t - dt - 1]. Mit t = D+1
liegt alles einen Tag spaeter als mit t = D. Die Skripte rechnen dafuer mit einer
effektiven Laenge L_eff = L + shift (L = 61 Cache-Tage): der Zieltag hat den
Index L_eff - 1, das Fenster endet (exklusiv) bei L_eff - dt - 1. Fuer dt >= 1
liegt es damit immer im Cache.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wildfire.paper.prepare import FIRE_CH

TARGETS = ("cache", "next_patch")
CENTRE = 6


def shift(target: str) -> int:
    return 0 if target == "cache" else 1


def tag(target: str) -> str:
    """Suffix fuer Manifeste und Dateien; das Cache-Label behaelt die alten Namen."""
    return "" if target == "cache" else "_next"


def labels(d: Path, target: str) -> np.ndarray:
    """(N,) Label je Stichprobe."""
    return np.load(d / ("y.npy" if target == "cache" else "y_next_patch.npy")).astype(int)


def fire_map(d: Path, X, target: str) -> np.ndarray:
    """(N, 13, 13) bool: Feuermaske am Zieltag je Pixel."""
    if target == "cache":
        return np.asarray(X[:, -1, FIRE_CH]) > 0
    return np.load(d / "fire_next.npy").astype(bool)


def centre_labels(d: Path, X, target: str) -> np.ndarray:
    """(N,) Feuer in der Mittelzelle am Zieltag."""
    return fire_map(d, X, target)[:, CENTRE, CENTRE].astype(int)
