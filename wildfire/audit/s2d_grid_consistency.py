"""Stufe 2d: Gitterkonsistenz ueber ALLE Archive.

Das README nennt 1 km Aufloesung, die Dateien tragen 0.04491576420597608
Grad - exakt 5000 m geteilt durch 111319.4908 m/Grad, den Erdumfang durch
360 bei Radius 6378137 m. Das ist der Massstab, mit dem Google Earth Engine
nach EPSG:4326 exportiert.

Stufe 2 hatte das an einer Stichprobe von zwoelf Dateien je Modalitaet
gezeigt. Weil es der Dokumentation widerspricht, wird es hier ueber ALLE
338 Jahresarchive geprueft: je Archiv erstes, mittleres und letztes
Mitglied. Damit ist jede Kombination aus Modalitaet und Jahr abgedeckt -
dort waere ein abweichender Exportlauf am ehesten zu erwarten.

Verglichen werden CRS, Transform, Rastergroesse, dtype und Bandzahl.
"""

from __future__ import annotations

import argparse
import math
import tarfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from rasterio.io import MemoryFile

from wildfire.audit.common import assert_readonly, load_config, write_manifest

EARTH_R = 6378137.0
M_PER_DEG = 2 * math.pi * EARTH_R / 360.0


def probe(path_str: str) -> dict:
    path = Path(path_str)
    out = {"tar": path.name, "modality": path.parent.name, "grids": [], "error": None,
           "n_members": 0, "checked": []}
    try:
        with tarfile.open(path, "r:") as tf:
            members = [m for m in tf if m.isfile()
                       and m.name.lower().endswith((".tif", ".tiff"))]
            members.sort(key=lambda m: m.name)
            out["n_members"] = len(members)
            if not members:
                return out
            pick = {0, len(members) // 2, len(members) - 1}
            for i in sorted(pick):
                m = members[i]
                blob = tf.extractfile(m).read()
                with MemoryFile(blob) as mem, mem.open() as ds:
                    out["grids"].append({
                        "crs": str(ds.crs), "width": ds.width, "height": ds.height,
                        "transform": [round(v, 15) for v in list(ds.transform)[:6]],
                        "dtype": str(ds.dtypes[0]), "count": ds.count,
                        "nodata": ds.nodatavals[0],
                    })
                    out["checked"].append(Path(m.name).name)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def key_of(g: dict) -> str:
    return f"{g['crs']}|{g['width']}x{g['height']}|{g['transform']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)

    maps = Path(cfg["dataset_root"]) / "Entire_Canada_Maps"
    tars = [str(p) for m in cfg["modalities"] for p in sorted((maps / m).glob("*.tar"))]
    print(f"[s2d] {len(tars)} Archive, je 3 Mitglieder -> {3*len(tars)} Dateien")

    grid_counts: Counter = Counter()
    per_modality: dict[str, Counter] = defaultdict(Counter)
    dtype_counts: dict[str, Counter] = defaultdict(Counter)
    band_counts: dict[str, Counter] = defaultdict(Counter)
    nodata_counts: Counter = Counter()
    errors, n_files = [], 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(probe, tars, chunksize=1):
            if res["error"]:
                errors.append({"tar": res["tar"], "error": res["error"]})
                continue
            for g in res["grids"]:
                k = key_of(g)
                grid_counts[k] += 1
                per_modality[res["modality"]][k] += 1
                dtype_counts[res["modality"]][g["dtype"]] += 1
                band_counts[res["modality"]][g["count"]] += 1
                nodata_counts[str(g["nodata"])] += 1
                n_files += 1
            print(f"    {res['modality']:10s} {res['tar']:22s} "
                  f"{res['n_members']:3d} Mitglieder, {len(res['grids'])} geprueft", flush=True)

    print(f"\n[s2d] {n_files} Dateien geprueft, {len(grid_counts)} verschiedene Gitter")
    for k, n in grid_counts.most_common():
        print(f"    {n:6d}x  {k}")

    single = len(grid_counts) == 1
    the_grid = grid_counts.most_common(1)[0][0] if grid_counts else None

    # Aufloesung nachrechnen
    res_deg = None
    if grid_counts:
        import json as _json
        t = _json.loads(the_grid.split("|")[2].replace("'", '"'))
        res_deg = abs(t[0])
    scale_m = res_deg * M_PER_DEG if res_deg else None

    print(f"\n[s2d] Aufloesung {res_deg} Grad")
    print(f"[s2d] x {M_PER_DEG:.4f} m/Grad = {scale_m:.4f} m Exportmassstab")
    print(f"[s2d] 1 km waeren {1000/M_PER_DEG:.17f} Grad")
    print(f"[s2d] alle Modalitaeten auf einem Gitter: {single}")
    print(f"[s2d] dtypes je Modalitaet: "
          f"{ {m: dict(c) for m, c in dtype_counts.items()} }")
    print(f"[s2d] NoData-Werte: {dict(nodata_counts)}")
    if errors:
        print(f"[s2d] FEHLER in {len(errors)} Archiven: {errors[:3]}")

    write_manifest(
        cfg, "s2d_grid_consistency", args.config,
        {
            "n_archives": len(tars),
            "n_files_checked": n_files,
            "members_per_archive": 3,
            "n_distinct_grids": len(grid_counts),
            "single_grid": single,
            "grid": the_grid,
            "grid_counts": dict(grid_counts),
            "per_modality_grids": {m: dict(c) for m, c in per_modality.items()},
            "dtype_per_modality": {m: dict(c) for m, c in dtype_counts.items()},
            "band_count_per_modality": {m: {str(k): v for k, v in c.items()}
                                        for m, c in band_counts.items()},
            "nodata_values": dict(nodata_counts),
            "resolution_deg": res_deg,
            "meters_per_degree": M_PER_DEG,
            "export_scale_m": scale_m,
            "one_km_would_be_deg": 1000 / M_PER_DEG,
            "readme_claim": "1 km daily resolution",
            "conclusion": (
                f"Exportmassstab {scale_m:.0f} m, nicht 1000 m. Die README-Angabe "
                "bezieht sich auf die Quellprodukte (MODIS 1 km), nicht auf das "
                "gelieferte Raster."
            ),
            "errors": errors,
        },
    )


if __name__ == "__main__":
    main()
