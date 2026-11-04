"""Wetter-Oracle, Daten: das beobachtete Wetter der Tage L + 1 .. L + 30 je Wuerfel-Stichtag.

Obere Schranke fuer lange Vorlaufzeiten: wie viel besser wird dt = 30, wenn das Modell das
kuenftige Wetter kennt (perfekte Vorhersage)? Nur die Reanalyse-Kanaele (ERA5 und daraus das
FWI-System und VPD, Cache-Kanaele 1..25), kein Feuer, keine Satellitenkanaele, die das Feuer
selbst sehen. Gleiche Kodierung wie train_X (uint8, Cache-Reihenfolge, Rand genullt).

  train_Xf.npy  (n, 30, 25, 13, 13) uint8   Tag L + j an Stelle j - 1
  train_kf.npy  (n,) int8                    Zahl der vorhandenen Folgetage (Saisonende: < 30)

Selbsttest: fuer Stichproben mit L + 30 in der Saison ist train_Xf gleich dem Wuerfelfenster bis
L + 30, und das letzte Fenster vor L ist gleich train_X (gleicher Ausschnitt, gleicher Tagesindex).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.data.dataset import CubeReader
from wildfire.paper.surv_prepare import HALF, T_WIN, window

WEATHER_CH = list(range(1, 26))
F_DAYS = 30
P = 2 * HALF + 1


def available_future(cube: CubeReader, L: int, F: int = F_DAYS) -> int:
    """Zahl der Folgetage L + 1 .. L + k im selben Wuerfeljahr, lueckenlos."""
    year, li = cube._pos[L]
    k = 0
    while k < F and cube._pos.get(L + k + 1) == (year, li + k + 1):
        k += 1
    return k


def future_window(cube: CubeReader, L: int, r: int, c: int, F: int = F_DAYS) -> tuple[np.ndarray, int]:
    """(F, 25, 13, 13) uint8 der Tage L + 1 .. L + k, Rest 0; dazu k."""
    out = np.zeros((F, len(WEATHER_CH), P, P), np.uint8)
    k = available_future(cube, L, F)
    if k:
        out[:k] = window(cube, L + k, r, c, T=k)[:, WEATHER_CH]
    return out, k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cube-dir", default="paper_pixel")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    cube = CubeReader(exp)
    cdir = exp / "derived_data" / args.cube_dir
    m = np.load(cdir / "train_meta.npz")
    Ls, rows, cols = m["L"], m["row"], m["col"]
    n = Ls.size
    Xf = np.lib.format.open_memmap(cdir / "train_Xf.npy", mode="w+", dtype=np.uint8, shape=(n, F_DAYS, len(WEATHER_CH), P, P))
    kf = np.zeros(n, np.int8)
    for q, j in enumerate(np.argsort(Ls, kind="mergesort")):
        Xf[j], kf[j] = future_window(cube, int(Ls[j]), int(rows[j]), int(cols[j]))
        if q % 20000 == 0:
            print(f"    {q}/{n}", flush=True)
    Xf.flush()

    # Selbsttest
    X = np.load(cdir / "train_X.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    full = np.flatnonzero(kf == F_DAYS)
    chk = rng.choice(full, min(200, full.size), replace=False)
    ok_f = all(np.array_equal(Xf[j], window(cube, int(Ls[j]) + F_DAYS, int(rows[j]), int(cols[j]))[:, WEATHER_CH]) for j in chk)
    ok_x = all(np.array_equal(X[j], window(cube, int(Ls[j]), int(rows[j]), int(cols[j]))) for j in chk[:50])
    print(f"[future] Selbsttest Folgefenster {ok_f}, Bezug train_X {ok_x}", flush=True)
    if not (ok_f and ok_x):
        raise SystemExit("ABBRUCH: Folgefenster passen nicht zum Wuerfel")
    np.save(cdir / "train_kf.npy", kf)
    summary = {"n": int(n), "share_full_30_days": float((kf == F_DAYS).mean()),
               "k_quantiles": {q: float(np.percentile(kf, q)) for q in (1, 5, 25, 50)},
               "weather_channels": WEATHER_CH}
    print(f"[future] {summary}", flush=True)
    write_manifest(cfg, "p26_future_weather", args.config, {
        "files": ["train_Xf.npy (n, 30, 25, 13, 13) uint8", "train_kf.npy (n,) int8"],
        "definition": "beobachtetes Wetter (Cache-Kanaele 1..25: ERA5, FWI-System, VPD) der Tage L+1..L+k, k <= 30 "
                      "lueckenlos im selben Wuerfeljahr (Saison 1.4.-30.9.), Rest 0",
        "summary": summary,
    })


if __name__ == "__main__":
    main()
