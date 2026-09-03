"""Zellflaeche und Breitengeometrie des Gitters.

Das Gitter ist in EPSG:4326 definiert und damit gleichmaessig in GRAD,
nicht in Metern. Die Nord-Sued-Kante bleibt bei rund 5 km, die
Ost-West-Kante schrumpft mit dem Kosinus der Breite. Zwischen dem
Suedrand und dem Nordrand Kanadas unterscheiden sich die Zellflaechen um
knapp den Faktor 6.

Zwei Konsequenzen, die diese Datei bedient:

1. **Metriken.** Jede ueber Pixel gemittelte Groesse gewichtet den Sueden
   staerker, weil dessen Pixel groesser sind. Die Feuerdichte kehrt ihre
   Reihenfolge um, sobald man pro Flaeche statt pro Pixel rechnet. Fuer
   populationsbezogene Aussagen wird deshalb ``area_weight`` gebraucht.

2. **Faltung.** Ein fester Kern deckt im Norden eine andere Flaeche ab als
   im Sueden. Damit ein Netz das kompensieren kann, werden Breite und
   Zellflaeche als statische Kanaele bereitgestellt.

Warum nicht umprojizieren? Eine flaechentreue Projektion (etwa Canada
Albers) haette wirklich gleich grosse Zellen. Sie erfordert aber ein
Resampling bereits aggregierter 5-km-Daten - das verschiebt Information,
ohne welche hinzuzufuegen, und ist bei kategorialen Kanaelen wie der
Landbedeckung heikel. Gewichte und Geometriekanaele sind exakt und
kostenlos; sie werden hier vorgezogen und die Entscheidung dokumentiert.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from wildfire.audit.common import assert_readonly, load_config, write_manifest

WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563


def authalic_q(phi: float, e: float, e2: float) -> float:
    s = math.sin(phi)
    return (1 - e2) * (s / (1 - e2 * s * s)
                       - (1 / (2 * e)) * math.log((1 - e * s) / (1 + e * s)))


def row_areas(height: int, lat_top: float, res: float) -> np.ndarray:
    """Exakte Zellflaeche je Rasterzeile auf dem WGS84-Ellipsoid (m^2)."""
    e2 = WGS84_F * (2 - WGS84_F)
    e = math.sqrt(e2)
    out = np.empty(height, np.float64)
    for i in range(height):
        p1 = math.radians(lat_top - res * i)
        p2 = math.radians(lat_top - res * (i + 1))
        out[i] = (WGS84_A ** 2 * math.radians(res) / 2.0) * abs(
            authalic_q(p1, e, e2) - authalic_q(p2, e, e2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)

    exp = Path(cfg["experiment_root"])
    mask_dir = exp / "derived_data" / "masks"
    man = json.loads((exp / "data_manifests" / "d0_masks.json").read_text())
    H, W = man["grid"]["height"], man["grid"]["width"]
    tr = man["grid"]["transform"]
    res, lat_top = abs(tr[0]), tr[5]

    areas = row_areas(H, lat_top, res)
    area_map = np.repeat(areas[:, None], W, axis=1).astype(np.float32)
    lat = (lat_top - res * (np.arange(H) + 0.5)).astype(np.float32)
    lat_map = np.repeat(lat[:, None], W, axis=1)
    coslat_map = np.cos(np.radians(lat_map)).astype(np.float32)

    valid = np.load(mask_dir / "valid.npy")
    canada = np.load(mask_dir / "canada.npy")

    # Gewicht normiert auf Mittelwert 1 ueber die gueltige Maske, damit es
    # die Groessenordnung von Losses und Metriken nicht verschiebt.
    mean_area = float(area_map[valid].mean())
    area_weight = (area_map / mean_area).astype(np.float32)

    np.save(mask_dir / "cell_area_m2.npy", area_map)
    np.save(mask_dir / "area_weight.npy", area_weight)
    np.save(mask_dir / "latitude.npy", lat_map)
    np.save(mask_dir / "cos_latitude.npy", coslat_map)

    rows_v = np.nonzero(valid.any(1))[0]
    av = areas[rows_v]
    burn = np.load(exp / "derived_data" / "audit_scratch" / "burn_count.npy")

    def density(mask):
        n = int(mask.sum())
        fires = int(burn[mask].sum())
        area_km2 = float(area_map[mask].sum()) / 1e6
        return {"n_pixels": n, "fire_pixel_days": fires, "area_km2": area_km2,
                "per_pixel": fires / max(1, n),
                "per_1000km2": fires / max(1e-9, area_km2) * 1000}

    south = valid & (lat_map < 55)
    north = valid & (lat_map >= 55)

    stats = {
        "grid": {"height": H, "width": W, "resolution_deg": res, "lat_top": lat_top},
        "cell_area_m2": {
            "max": float(av.max()), "min": float(av.min()),
            "ratio_max_min": float(av.max() / av.min()),
            "max_at_lat": float(lat_top - res * rows_v[int(np.argmax(av))]),
            "min_at_lat": float(lat_top - res * rows_v[int(np.argmin(av))]),
            "mean_over_valid": mean_area,
        },
        "total_area_km2": {
            "canada_mask": float(area_map[canada].sum()) / 1e6,
            "valid_mask": float(area_map[valid].sum()) / 1e6,
            "canada_official_km2": 9_984_670,
        },
        "density_south_lt55N": density(south),
        "density_north_ge55N": density(north),
        "patch_extent_km_33px": {
            str(la): {
                "north_south": 33 * res * (111132.92 - 559.82 * math.cos(2 * math.radians(la))) / 1000,
                "east_west": 33 * res * 111412.84 * math.cos(math.radians(la)) / 1000,
            } for la in (49, 55, 60, 70, 78)
        },
        "outputs": [str(mask_dir / f) for f in
                    ("cell_area_m2.npy", "area_weight.npy", "latitude.npy", "cos_latitude.npy")],
        "decision": (
            "Keine Umprojektion. Flaechengewichte und Geometriekanaele statt "
            "Resampling bereits aggregierter 5-km-Daten."
        ),
        "usage": {
            "area_weight": "multiplikativ zu den Ziehungsgewichten, wenn eine Aussage "
                           "pro Flaeche statt pro Pixel gemeint ist",
            "latitude/cos_latitude": "als statische Eingabekanaele, damit das Netz die "
                                     "geometrische Verzerrung kompensieren kann",
        },
    }

    err = abs(stats["total_area_km2"]["canada_mask"] - 9_984_670 / 1e6 * 1e6) 
    print(f"[geometry] Zellflaeche {av.min()/1e6:.2f} .. {av.max()/1e6:.2f} km2 "
          f"(Faktor {av.max()/av.min():.1f})")
    print(f"[geometry] Kanada-Maske {stats['total_area_km2']['canada_mask']/1e6:.3f} Mio. km2 "
          f"gegen amtlich 9.985 Mio. km2 "
          f"({100*abs(stats['total_area_km2']['canada_mask']-9_984_670)/9_984_670:.2f} % Abweichung)")
    print(f"[geometry] Feuerdichte Sueden  {stats['density_south_lt55N']['per_pixel']:.4f}/Pixel  "
          f"{stats['density_south_lt55N']['per_1000km2']:.2f}/1000 km2")
    print(f"[geometry] Feuerdichte Norden  {stats['density_north_ge55N']['per_pixel']:.4f}/Pixel  "
          f"{stats['density_north_ge55N']['per_1000km2']:.2f}/1000 km2")
    print("[geometry] -> die Reihenfolge kippt: pro Pixel fuehrt der Sueden, "
          "pro Flaeche der Norden")

    write_manifest(cfg, "d8_geometry", args.config, stats)


if __name__ == "__main__":
    main()
