"""Merkmalsextraktion fuer die tabellarischen Baselines (B1).

B1 ist bewusst NICHT raeumlich: Es sieht nur das Landmarkpixel selbst,
keine Umgebung. Damit isoliert der spaetere Vergleich mit B2/B3 genau den
Beitrag des raeumlichen Kontexts - eine der Ablationen aus Phase 9.

Merkmale je Landmark:
  * Wert jedes dynamischen Kanals am Tag t
  * Mittelwert ueber die letzten 7 und 30 Tage
  * Minimum und Maximum ueber 30 Tage
  * Steigung ueber die letzten 7 Tage (einfache Differenz)
  * alle statischen Kanaele
  * Geometrie und Zeit: Breite, Zellflaeche, Tag im Jahr als sin/cos

Die gesperrten Kanaele (Feuermaske, 4-um-Baender) gehen nur mit ihrer
Historie bis ``t - C`` ein - dieselbe Regel wie im Dataset, hier auf die
Aggregate angewandt.

**Zugriffsmuster.** Das kanalletzte Layout macht den Kanalvektor eines
Pixels an einem Tag zu 42 zusammenhaengenden Byte. Die Landmarks werden
nach (Jahr, Tag) sortiert gelesen, damit die Zugriffe moeglichst
fortlaufend sind statt ueber 340 GB zu springen.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

from wildfire.audit.common import assert_readonly, human_bytes, load_config, write_manifest
from wildfire.data.dataset import LOCKED_CHANNELS, CubeReader

LOOKBACK = 30
COOLDOWN = 7
SHORT = 7


def feature_names(n_dyn: int, n_stat: int) -> list[str]:
    return ([f"{c}_last" for c in range(n_dyn)] + [f"{c}_mean7" for c in range(n_dyn)]
            + [f"{c}_mean30" for c in range(n_dyn)] + [f"{c}_min30" for c in range(n_dyn)]
            + [f"{c}_max30" for c in range(n_dyn)] + [f"stat{c}" for c in range(n_stat)]
            + ["lat", "cell_km2", "region", "doy_sin", "doy_cos", "year"])


def row_features(cube: CubeReader, tt: int, r: int, c: int, lat, area, region,
                 locked: np.ndarray) -> np.ndarray:
    """Merkmalsvektor (vor dem Spaltenfilter) fuer einen Pixeltag (t, r, c).

    Auch von wildfire.paper.ours_on_paper benutzt - dieselbe Rechnung wie im
    Training, nicht eine Nachbildung.
    """
    n_dyn = cube.n_dyn
    # Das Historienfenster liegt immer im selben Jahr: die Saison
    # beginnt am 1. Mai, der Wuerfel am 1. April, 30 Tage zurueck
    # bleiben also innerhalb des Jahres. Damit genuegt EIN
    # zusammenhaengender Schnitt statt 30 Einzelzugriffe.
    year, li = cube._pos[tt]
    year0, li0 = cube._pos[tt - LOOKBACK + 1]
    dyn, _ = cube._arrays(year)
    if year0 == year and li0 == li - LOOKBACK + 1:
        series = dyn[li0:li + 1, r, c, :].astype(np.float32)
    else:                                     # Jahresgrenze: Rueckfall
        series = np.empty((LOOKBACK, n_dyn), np.float32)
        for k in range(LOOKBACK):
            yy, ll = cube._pos[tt - (LOOKBACK - 1 - k)]
            series[k] = cube._arrays(yy)[0][ll, r, c, :]
    # Kanalsperre auf die Historie anwenden
    series[LOOKBACK - COOLDOWN:, locked] = np.nan

    with np.errstate(invalid="ignore"):
        last = series[-1]
        mean_s = np.nanmean(series[-SHORT:], axis=0)
        mean_l = np.nanmean(series, axis=0)
        mn = np.nanmin(series, axis=0)
        mx = np.nanmax(series, axis=0)
    stat = cube._arrays(year)[1][r, c, :].astype(np.float32)
    d0 = dt.date.fromisoformat(cube.dates[tt])
    doy = d0.timetuple().tm_yday
    extra = np.array([
        lat[r, c], area[r, c] / 1e6, region[r, c],
        np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
        d0.year,
    ], np.float32)
    return np.concatenate([last, mean_s, mean_l, mn, mx, stat, extra])


def build(cfg, split: str, part: str, n_max: int, seed: int):
    exp = Path(cfg["experiment_root"])
    z = np.load(exp / "derived_data" / "landmarks" / f"{split}_{part}.npz")
    cube = CubeReader(exp)
    avail = np.zeros(len(cube.dates), bool)
    avail[np.fromiter(cube.available_days(), np.int64)] = True

    t = z["t"].astype(np.int64)
    ok = np.ones(t.size, bool)
    for k in range(LOOKBACK):
        d = t - k
        ok &= (d >= 0) & avail[np.clip(d, 0, len(cube.dates) - 1)]
    idx = np.nonzero(ok)[0]
    rng = np.random.default_rng(seed)
    if idx.size > n_max:
        idx = np.sort(rng.choice(idx, n_max, replace=False))
    # nach Zeit sortieren: fortlaufende statt springende Zugriffe
    idx = idx[np.argsort(t[idx], kind="mergesort")]

    row, col = z["row"].astype(np.int64), z["col"].astype(np.int64)
    lat = np.load(exp / "derived_data" / "masks" / "latitude.npy")
    area = np.load(exp / "derived_data" / "masks" / "cell_area_m2.npy")
    region = np.load(exp / "derived_data" / "masks" / "region.npy")

    n_dyn, n_stat = cube.n_dyn, cube.n_stat
    n_feat = 5 * n_dyn + n_stat + 6
    X = np.zeros((idx.size, n_feat), np.float32)
    locked = np.array(sorted(LOCKED_CHANNELS))

    for i, j in enumerate(idx.tolist()):
        X[i] = row_features(cube, int(t[j]), int(row[j]), int(col[j]), lat, area, region, locked)
        if i % 50000 == 0:
            print(f"    {part}: {i}/{idx.size}", flush=True)

    names = feature_names(n_dyn, n_stat)
    # Spalten, die IMMER undefiniert sind, entfernen statt imputieren.
    # Fuer die gesperrten Kanaele gibt es am Tag t und im 7-Tage-Mittel
    # per Konstruktion keinen Wert - das ist keine fehlende Messung,
    # sondern eine bewusst entfernte Information. Eine Imputation wuerde
    # so tun, als liesse sie sich schaetzen.
    all_nan = np.all(np.isnan(X), axis=0)
    dropped = [names[i] for i in np.nonzero(all_nan)[0]]
    X = X[:, ~all_nan]
    names = [n for n, bad in zip(names, all_nan) if not bad]
    # slope wird verworfen: bei gesperrten Kanaelen entstehen NaN-Ketten,
    # und der Zusatznutzen gegenueber mean7/mean30 ist gering.
    return {
        "dropped": dropped,
        "X": X, "names": names, "index": idx,
        "event": z["event"][idx], "t_event": z["t_event"][idx],
        "weight": z["weight"][idx], "event_id": z["event_id"][idx],
        "year": z["year"][idx], "t": t[idx], "row": row[idx], "col": col[idx],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="temporal")
    ap.add_argument("--n-train", type=int, default=400_000)
    ap.add_argument("--n-eval", type=int, default=300_000)
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)

    exp = Path(cfg["experiment_root"])
    out_dir = exp / "derived_data" / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for part in ("train", "val", "test"):
        n = args.n_train if part == "train" else args.n_eval
        print(f"[features] {part}: ziehe bis zu {n} Landmarks")
        d = build(cfg, args.split, part, n, int(cfg["seed"]))
        f = out_dir / f"{args.split}_{part}.npz"
        np.savez_compressed(
            f, **{k: v for k, v in d.items() if k not in ("names", "dropped")},
            names=np.array(d["names"]), dropped=np.array(d["dropped"]))
        nan_share = float(np.isnan(d["X"]).mean())
        if part == "train":
            print(f"[features] entfernte Spalten (immer undefiniert): {d['dropped']}")
        summary[part] = {
            "n": int(d["X"].shape[0]), "n_features": int(d["X"].shape[1]),
            "n_positive": int(d["event"].sum()),
            "nan_share": nan_share,
            "dropped_columns": d["dropped"],
            "file": str(f), "bytes": int(f.stat().st_size),
        }
        print(f"[features] {part}: {d['X'].shape} positiv={int(d['event'].sum())} "
              f"NaN-Anteil={nan_share:.4f}  {human_bytes(f.stat().st_size)}")

    write_manifest(cfg, "f1_features", args.config,
                   {"split": args.split, "lookback": LOOKBACK, "cooldown": COOLDOWN,
                    "short_window": SHORT, "locked_channels": sorted(LOCKED_CHANNELS),
                    "note": "nur Zentrumspixel - B1 ist bewusst nicht raeumlich",
                    "parts": summary})


if __name__ == "__main__":
    main()
