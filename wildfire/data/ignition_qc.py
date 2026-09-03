"""Qualitaetspruefung der abgeleiteten Zuendungen.

Die Monatsverteilung der Zuendungen zeigt einen zweiten Gipfel im Oktober
und November. Kanadas Waldbrandsaison hat ihren Gipfel aber im Juli und
faellt danach steil ab. Der Verdacht: landwirtschaftliches Abbrennen von
Ernterueckstaenden auf den Praerien, das MODIS korrekt detektiert, das
aber kein Waldbrand ist.

Diese Pruefung kreuzt die Zuendungen mit der Landbedeckung und der
geographischen Lage, um den Verdacht zu bestaetigen oder zu verwerfen -
und um zu beziffern, was ein Ausschluss von Ackerland kosten wuerde.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.audit.channel_dict import IGBP_CLASSES

# IGBP-Klassen, die menschgemachtes Abbrennen statt Waldbrand nahelegen.
AGRICULTURAL_CLASSES = (12, 14)   # Ackerland, Ackerland-Naturvegetation-Mosaik
FOREST_CLASSES = (1, 2, 3, 4, 5)
SHRUB_GRASS_CLASSES = (6, 7, 8, 9, 10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ignitions", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)

    exp = Path(cfg["experiment_root"])
    ign_path = Path(args.ignitions) if args.ignitions else (
        exp / "derived_data" / "events" / "ignitions_c7_r1.csv"
    )
    lc = np.load(exp / "derived_data" / "masks" / "landcover.npy") if (
        exp / "derived_data" / "masks" / "landcover.npy"
    ).exists() else None
    if lc is None:
        # Landbedeckung aus dem Referenzjahr nachladen
        from wildfire.data.masks import read_raster

        lc_arr, _ = read_raster(Path(cfg["dataset_root"]), "MCD12Q1", 2015, "01_01")
        lc = lc_arr[0]
        np.save(exp / "derived_data" / "masks" / "landcover.npy", lc)

    rows = []
    with ign_path.open() as fh:
        for rec in csv.DictReader(fh):
            rows.append(rec)
    print(f"[qc] {len(rows)} Zuendungen aus {ign_path.name}")

    by_month_class: dict[str, Counter] = defaultdict(Counter)
    by_class = Counter()
    by_month = Counter()
    lat_of_class: dict[int, list[float]] = defaultdict(list)
    agri_months = Counter()
    forest_months = Counter()

    for rec in rows:
        r, c = int(rec["row"]), int(rec["col"])
        k = int(lc[r, c])
        month = rec["date"][5:7]
        by_month_class[month][k] += 1
        by_class[k] += 1
        by_month[month] += 1
        lat_of_class[k].append(float(rec["lat"]))
        if k in AGRICULTURAL_CLASSES:
            agri_months[month] += 1
        elif k in FOREST_CLASSES:
            forest_months[month] += 1

    total = len(rows)
    n_agri = sum(by_class[k] for k in AGRICULTURAL_CLASSES)
    n_forest = sum(by_class[k] for k in FOREST_CLASSES)
    n_shrub = sum(by_class[k] for k in SHRUB_GRASS_CLASSES)

    print("\n[qc] Zuendungen je Landbedeckungsklasse:")
    for k, n in by_class.most_common():
        print(f"    {k:2d} {IGBP_CLASSES.get(k, '?'):32s} {n:7d} ({100*n/total:5.1f} %)  "
              f"Median-Breite {np.median(lat_of_class[k]):5.2f}")

    print(f"\n[qc] Wald {n_forest} ({100*n_forest/total:.1f} %), "
          f"Busch/Gras {n_shrub} ({100*n_shrub/total:.1f} %), "
          f"Acker {n_agri} ({100*n_agri/total:.1f} %)")

    print("\n[qc] Monat: gesamt | davon Acker | davon Wald | Ackeranteil")
    month_tab = {}
    for mth in sorted(by_month):
        tot_m = by_month[mth]
        a, f = agri_months[mth], forest_months[mth]
        month_tab[mth] = {"total": tot_m, "agricultural": a, "forest": f,
                          "agri_share": a / max(1, tot_m)}
        print(f"    {mth}  {tot_m:6d} | {a:6d} | {f:6d} | {100*a/max(1,tot_m):5.1f} %")

    # Was wuerde ein Ackerausschluss kosten - je Monat und insgesamt?
    autumn = ("10", "11")
    autumn_total = sum(by_month[m] for m in autumn)
    autumn_agri = sum(agri_months[m] for m in autumn)
    summer = ("06", "07", "08")
    summer_total = sum(by_month[m] for m in summer)
    summer_agri = sum(agri_months[m] for m in summer)

    print(f"\n[qc] Oktober+November: {autumn_total} Zuendungen, davon {autumn_agri} auf Acker "
          f"({100*autumn_agri/max(1,autumn_total):.1f} %)")
    print(f"[qc] Juni-August:      {summer_total} Zuendungen, davon {summer_agri} auf Acker "
          f"({100*summer_agri/max(1,summer_total):.1f} %)")
    print(f"[qc] Ackerausschluss wuerde {n_agri} von {total} Zuendungen entfernen "
          f"({100*n_agri/total:.1f} %)")

    write_manifest(
        cfg,
        "d1b_ignition_qc",
        args.config,
        {
            "ignitions_file": str(ign_path),
            "n_ignitions": total,
            "by_landcover_class": {str(k): {"n": n, "name": IGBP_CLASSES.get(k, "?"),
                                            "median_lat": float(np.median(lat_of_class[k]))}
                                   for k, n in by_class.most_common()},
            "grouped": {"forest": n_forest, "shrub_grass": n_shrub, "agricultural": n_agri},
            "by_month": month_tab,
            "autumn_oct_nov": {"total": autumn_total, "agricultural": autumn_agri,
                               "share": autumn_agri / max(1, autumn_total)},
            "summer_jun_aug": {"total": summer_total, "agricultural": summer_agri,
                               "share": summer_agri / max(1, summer_total)},
            "agricultural_classes": list(AGRICULTURAL_CLASSES),
            "cost_of_excluding_agriculture": {
                "n_removed": n_agri, "share_removed": n_agri / total
            },
        },
    )


if __name__ == "__main__":
    main()
