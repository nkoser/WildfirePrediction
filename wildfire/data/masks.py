"""Erzeugt die Gueltigkeitsmasken auf dem MPCFire-Gitter.

Ein Pixel-Tag darf nur dann in Training oder Evaluation eingehen, wenn das
Pixel ueberhaupt ein sinnvoller Kandidat ist. Vier Masken werden gebildet
und zu einer Gesamtmaske verknuepft:

  1. ``land``      - IGBP-Landbedeckung ausser Wasser (17) und Fuellwert (0).
  2. ``canada``    - Staatsgebiet Kanadas, aus Natural Earth rasterisiert.
                     Das gelieferte Gitter ist nur eine Bounding Box und
                     enthaelt rund 30 % Feuer-Pixeltage suedlich von 49degN,
                     ueberwiegend in den USA.
  3. ``no_hotspot``- ohne die dauerhaften Waermequellen (Industrie, urbane
                     Waermeinseln), die als Waldbrand fehldetektiert werden.
  4. ``burnable``  - ohne Landbedeckungsklassen, die nicht brennen koennen
                     (Schnee/Eis, vegetationsarm, bebaut).
  5. ``no_cropland``- ohne Ackerland. Dort dominiert Stoppelbrennen: im
                     April 62 % und im Mai 49 % aller Zuendungen, im
                     Hochsommer nur 4,5 %.

Alle Masken werden als .npy und zusaetzlich als GeoTIFF abgelegt, damit sie
in einem GIS sichtbar geprueft werden koennen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.io import MemoryFile
from rasterio.transform import Affine

from wildfire.audit.common import assert_readonly, load_config, write_manifest

NON_LAND_CLASSES = (0, 17)          # unklassifiziert, Wasser
NON_BURNABLE_CLASSES = (15, 16, 13)  # Schnee/Eis, vegetationsarm, bebaut
# Ackerland wird ausgeschlossen, weil dort Stoppelbrennen stattfindet:
# im April sind 62 % und im Mai 49 % aller Zuendungen auf Ackerland,
# im Hochsommer nur 4,5 %. Das ist geplantes menschgemachtes Feuer und
# kein Waldbrandrisiko (siehe d1b_ignition_qc).
AGRICULTURAL_CLASSES = (12, 14)     # Ackerland, Ackerland-Naturvegetation-Mosaik
HOTSPOT_MIN_FIRE_DAYS = 30
COUNTRY = "Canada"


def read_raster(dataset_root: Path, modality: str, year: int, mmdd: str):
    tar = dataset_root / "Entire_Canada_Maps" / modality / f"{modality}__{year}.tar"
    target = f"{year}_{mmdd}.tif"
    with tarfile.open(tar, "r:") as tf:
        for m in tf:
            if m.isfile() and Path(m.name).name == target:
                blob = tf.extractfile(m).read()
                break
        else:
            raise SystemExit(f"nicht gefunden: {modality}/{target}")
    with MemoryFile(blob) as mem, mem.open() as ds:
        return ds.read(), ds.profile.copy()


def write_geotiff(path: Path, arr: np.ndarray, profile: dict) -> None:
    p = dict(profile)
    p.update(count=1, dtype="uint8", compress="lzw", nodata=None)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype(np.uint8), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--reference-year", type=int, default=2015)
    ap.add_argument(
        "--countries-geojson",
        default="/data/Wildfire/artifacts/ne_50m_admin_0_countries.geojson",
    )
    ap.add_argument("--scratch", default="/data/Wildfire/derived_data/audit_scratch")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)

    root = Path(cfg["dataset_root"])
    out_dir = Path(cfg["experiment_root"]) / "derived_data" / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- Gitter
    lc_arr, profile = read_raster(root, "MCD12Q1", args.reference_year, "01_01")
    lc = lc_arr[0]
    H, W = lc.shape
    transform: Affine = profile["transform"]
    print(f"[masks] Gitter {H}x{W}, CRS {profile['crs']}, Transform {tuple(transform)[:6]}")

    # ---------------------------------------------------------------- Land
    land = ~np.isin(lc, NON_LAND_CLASSES)

    # ---------------------------------------------------------------- Kanada
    gj = json.loads(Path(args.countries_geojson).read_text())
    feats = [f for f in gj["features"] if f["properties"].get("ADMIN") == COUNTRY]
    if not feats:
        raise SystemExit(f"{COUNTRY} nicht in {args.countries_geojson}")
    canada = rasterize(
        [(f["geometry"], 1) for f in feats],
        out_shape=(H, W),
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    ).astype(bool)
    geojson_hash = hashlib.sha256(Path(args.countries_geojson).read_bytes()).hexdigest()[:16]

    # ---------------------------------------------------------------- Hotspots
    burn_count = np.load(Path(args.scratch) / "burn_count.npy")
    hotspot = burn_count >= HOTSPOT_MIN_FIRE_DAYS
    no_hotspot = ~hotspot

    # ---------------------------------------------------------------- brennbar
    burnable = ~np.isin(lc, NON_LAND_CLASSES + NON_BURNABLE_CLASSES)
    no_cropland = ~np.isin(lc, AGRICULTURAL_CLASSES)

    valid = land & canada & no_hotspot & burnable & no_cropland

    masks = {
        "land": land,
        "canada": canada,
        "no_hotspot": no_hotspot,
        "burnable": burnable,
        "no_cropland": no_cropland,
        "valid": valid,
    }
    np.save(out_dir / "landcover.npy", lc)
    for name, m in masks.items():
        np.save(out_dir / f"{name}.npy", m)
        write_geotiff(out_dir / f"{name}.tif", m, profile)

    # ---------------------------------------------------------------- Bilanz
    total_fire = int(burn_count.sum())
    stats = {}
    for name, m in masks.items():
        stats[name] = {
            "n_pixels": int(m.sum()),
            "fraction_of_grid": float(m.mean()),
            "fire_pixel_days_inside": int(burn_count[m].sum()),
            "fire_share_retained": float(burn_count[m].sum() / max(1, total_fire)),
        }
        print(f"[masks] {name:11s} {m.sum():8d} Pixel ({100*m.mean():5.1f} % des Gitters), "
              f"{int(burn_count[m].sum()):7d} Feuer-Pixeltage "
              f"({100*burn_count[m].sum()/max(1,total_fire):5.1f} % erhalten)")

    # Was kostet jede einzelne Einschraenkung, ausgehend von "land"?
    step = land.copy()
    cascade = [("land", step.copy())]
    for name in ("canada", "burnable", "no_cropland", "no_hotspot"):
        step = step & masks[name]
        cascade.append((f"+{name}", step.copy()))
    print("\n[masks] Kaskade:")
    cascade_stats = {}
    for label, m in cascade:
        cascade_stats[label] = {
            "n_pixels": int(m.sum()),
            "fire_pixel_days": int(burn_count[m].sum()),
            "fire_share": float(burn_count[m].sum() / max(1, total_fire)),
        }
        print(f"    {label:14s} {int(m.sum()):8d} Pixel  {int(burn_count[m].sum()):7d} Feuer-Pixeltage "
              f"({100*burn_count[m].sum()/max(1,total_fire):5.1f} %)")

    lc_in_valid = {int(k): int(v) for k, v in zip(*np.unique(lc[valid], return_counts=True))}

    write_manifest(
        cfg,
        "d0_masks",
        args.config,
        {
            "grid": {"height": H, "width": W, "crs": str(profile["crs"]),
                     "transform": list(tuple(transform)[:6])},
            "reference_year": args.reference_year,
            "countries_geojson": args.countries_geojson,
            "countries_geojson_sha256_16": geojson_hash,
            "country": COUNTRY,
            "non_land_classes": list(NON_LAND_CLASSES),
            "non_burnable_classes": list(NON_BURNABLE_CLASSES),
            "agricultural_classes_excluded": list(AGRICULTURAL_CLASSES),
            "hotspot_min_fire_days": HOTSPOT_MIN_FIRE_DAYS,
            "n_hotspot_pixels": int(hotspot.sum()),
            "masks": stats,
            "cascade": cascade_stats,
            "landcover_histogram_in_valid": lc_in_valid,
            "output_dir": str(out_dir),
            "files": sorted(p.name for p in out_dir.iterdir()),
        },
    )


if __name__ == "__main__":
    main()
