"""Zusaetzliche Trainingsstichproben (paper_extra.py) an die Daten eines Caches anhaengen.

  none    nur die 4 347 Train-Stichproben des Papers
  random  + Positive + zufaellige Negative aus 2000-2019
  hnm     + Positive + Negative nach FWI-HNM (unser Score)
  both    + Positive + beide Negativ-Arten (Verhaeltnis dann 1 : 4)

Die angehaengten Zeilen zaehlen zum Split "train"; Validierung und Test bleiben
die Stichproben des Papers und behalten ihre Positionen.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wildfire.paper import targets

MODES = ("none", "random", "hnm", "both")


def tag(mode: str) -> str:
    return "" if mode == "none" else f"_x{mode}"


def load(exp: Path, target: str, mode: str, name: str = "paper_extra") -> dict:
    """X (n, 61, 55, 13, 13) uint8, y (n,), fire_map (n, 13, 13) am Zieltag, fuer die gewaehlte Art."""
    d = exp / "derived_data" / name
    meta = np.load(d / "meta.npz")
    kind = meta["kind"]
    sel = kind == 0
    if mode in ("random", "both"):
        sel |= kind == 1
    if mode in ("hnm", "both"):
        sel |= kind == 2
    rows = np.where(sel)[0]
    X = np.load(d / "X.npy", mmap_mode="r")[rows]
    y = targets.labels(d, target)[rows]
    fmap = (np.asarray(X[:, -1, 0]) > 0) if target == "cache" else np.load(d / "fire_next.npy")[rows].astype(bool)
    print(f"[extra] {mode}: {rows.size:,} Stichproben angehaengt, davon positiv ({target}) {int(y.sum()):,}",
          flush=True)
    return {"X": X, "y": y, "fire_map": fmap, "n": int(rows.size),
            "date": meta["date"][rows] if "date" in meta.files else np.zeros(rows.size, np.int64)}


def append(X: np.ndarray, y: np.ndarray, split: np.ndarray, ex: dict):
    """Cache-Daten + Zusatz -> (X, y, split); Zusatz kommt ans Ende, Split "train"."""
    return (np.concatenate([X, ex["X"]]), np.concatenate([y, ex["y"]]),
            np.concatenate([split, np.full(ex["n"], "train", dtype=split.dtype)]))
