"""Stufe 2: Geo-Header und Wertestatistik einer Raster-Stichprobe.

Die GeoTIFFs werden aus den tar-Archiven in den Arbeitsspeicher gelesen
(``rasterio.MemoryFile``) - die Rohdaten werden dabei nur gelesen und es
entsteht keine Kopie im Rohdatenbaum. Optional wird ein Scratch-Pfad
ausserhalb der Rohdaten fuer sehr grosse Raster genutzt.

Erhoben werden: CRS, Affine-Transform, Rastergroesse, Pixelgroesse in
Grad und in Metern, Bandanzahl und -beschreibung, dtype, NoData,
Kompression, Skalierung (scale/offset), TIFF-Tags sowie pro Band
min/max/mean/std, NoData-Anteil und - bei kategorialen Kanaelen - die
Werteverteilung.
"""

from __future__ import annotations

import argparse
import json
import math
import tarfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.io import MemoryFile

from wildfire.audit.common import assert_readonly, load_config, write_manifest

CATEGORICAL_MODALITIES = {"MCD12Q1", "MCD14A1"}


def pick_members(tar_path: Path, k: int) -> list[str]:
    """Gleichmaessig ueber das Archiv verteilte Mitglieder (deterministisch)."""
    with tarfile.open(tar_path, "r:") as tf:
        names = [m.name for m in tf if m.isfile() and m.name.lower().endswith((".tif", ".tiff"))]
    if not names:
        return []
    names.sort()
    if len(names) <= k:
        return names
    idx = [round(i * (len(names) - 1) / (k - 1)) for i in range(k)] if k > 1 else [0]
    return [names[i] for i in sorted(set(idx))]


def read_member(tar_path: Path, member: str) -> bytes:
    with tarfile.open(tar_path, "r:") as tf:
        fh = tf.extractfile(member)
        if fh is None:
            raise OSError(f"kein regulaeres Mitglied: {member}")
        return fh.read()


def meters_per_pixel(transform, crs, height: int, width: int) -> dict:
    """Pixelgroesse in Metern - bei geographischem CRS am Kartenzentrum."""
    px_x, px_y = abs(transform.a), abs(transform.e)
    if crs is not None and crs.is_geographic:
        lat_center = transform.f + transform.e * (height / 2.0)
        m_per_deg_lat = 111_132.92 - 559.82 * math.cos(2 * math.radians(lat_center))
        m_per_deg_lon = 111_412.84 * math.cos(math.radians(lat_center))
        return {
            "pixel_deg_x": px_x,
            "pixel_deg_y": px_y,
            "lat_center": lat_center,
            "pixel_m_x_at_center": px_x * m_per_deg_lon,
            "pixel_m_y_at_center": px_y * m_per_deg_lat,
            "pixel_m_x_at_lat49": px_x * 111_412.84 * math.cos(math.radians(49.0)),
            "pixel_m_x_at_lat70": px_x * 111_412.84 * math.cos(math.radians(70.0)),
        }
    return {"pixel_m_x": px_x, "pixel_m_y": px_y}


def inspect_raster(blob: bytes, modality: str, with_stats: bool) -> dict:
    with MemoryFile(blob) as mem, mem.open() as ds:
        info: dict = {
            "driver": ds.driver,
            "width": ds.width,
            "height": ds.height,
            "count": ds.count,
            "dtypes": list(ds.dtypes),
            "nodata": [None if v is None else float(v) for v in ds.nodatavals],
            "crs": str(ds.crs) if ds.crs else None,
            "crs_wkt_head": (ds.crs.to_wkt()[:220] if ds.crs else None),
            "is_geographic": bool(ds.crs.is_geographic) if ds.crs else None,
            "transform": list(ds.transform)[:6],
            "bounds": list(ds.bounds),
            "res": list(ds.res),
            "band_descriptions": list(ds.descriptions),
            "band_units": list(ds.units),
            "scales": list(ds.scales),
            "offsets": list(ds.offsets),
            "compression": str(ds.compression) if ds.compression else None,
            "interleaving": str(ds.interleaving) if ds.interleaving else None,
            "block_shapes": [list(b) for b in ds.block_shapes],
            "tags": {k: v for k, v in list(ds.tags().items())[:40]},
            "band_tags": {str(b): ds.tags(b) for b in ds.indexes if ds.tags(b)},
            "overviews_band1": ds.overviews(1),
            "geometry": meters_per_pixel(ds.transform, ds.crs, ds.height, ds.width),
        }
        if not with_stats:
            return info

        bands = []
        for b in ds.indexes:
            arr = ds.read(b)
            nod = ds.nodatavals[b - 1]
            mask = np.ones(arr.shape, dtype=bool)
            if nod is not None and not (isinstance(nod, float) and math.isnan(nod)):
                mask &= arr != nod
            if np.issubdtype(arr.dtype, np.floating):
                mask &= np.isfinite(arr)
            valid = arr[mask]
            entry = {
                "band": b,
                "description": ds.descriptions[b - 1],
                "dtype": str(arr.dtype),
                "nodata": None if nod is None else float(nod),
                "n_total": int(arr.size),
                "n_valid": int(valid.size),
                "nodata_fraction": float(1.0 - valid.size / arr.size),
                "nan_fraction": (
                    float(np.isnan(arr).mean()) if np.issubdtype(arr.dtype, np.floating) else 0.0
                ),
            }
            if valid.size:
                entry.update(
                    min=float(valid.min()),
                    max=float(valid.max()),
                    mean=float(valid.mean()),
                    std=float(valid.std()),
                    p01=float(np.percentile(valid, 1)),
                    p50=float(np.percentile(valid, 50)),
                    p99=float(np.percentile(valid, 99)),
                    n_zero=int((valid == 0).sum()),
                )
            if modality in CATEGORICAL_MODALITIES:
                vals, counts = np.unique(arr, return_counts=True)
                order = np.argsort(-counts)[:30]
                entry["value_histogram"] = {
                    str(vals[i]): int(counts[i]) for i in order
                }
            bands.append(entry)
        info["bands"] = bands
        return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--scratch", default="/data/Wildfire/derived_data/audit_scratch")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    Path(args.scratch).mkdir(parents=True, exist_ok=True)

    maps_root = Path(cfg["dataset_root"]) / "Entire_Canada_Maps"
    samp = cfg["sampling"]
    years = [str(y) for y in samp["header_years"]]
    k = int(samp["header_files_per_year"])
    stats_budget = int(samp["stats_rasters_per_modality"])

    out: dict[str, dict] = {}
    for mod in cfg["modalities"]:
        print(f"[s2] === {mod} ===", flush=True)
        entry: dict = {"modality": mod, "samples": [], "errors": []}
        n_stats = 0
        for year in years:
            tar = maps_root / mod / f"{mod}__{year}.tar"
            if not tar.exists():
                entry["errors"].append(f"fehlt: {tar.name}")
                continue
            for member in pick_members(tar, k):
                with_stats = n_stats < stats_budget
                try:
                    blob = read_member(tar, member)
                    info = inspect_raster(blob, mod, with_stats)
                except Exception as exc:
                    entry["errors"].append(f"{tar.name}:{member} -> {type(exc).__name__}: {exc}")
                    continue
                info.update(tar=tar.name, member=member, year=year, blob_bytes=len(blob))
                entry["samples"].append(info)
                n_stats += int(with_stats)
                g = info["geometry"]
                print(
                    f"    {member:30s} {info['width']}x{info['height']} "
                    f"b={info['count']} {info['dtypes'][0]} crs={info['crs']} "
                    f"res={info['res'][0]:.6g} nodata={info['nodata'][0]} "
                    f"~{g.get('pixel_m_x_at_center', g.get('pixel_m_x', float('nan'))):.0f}m",
                    flush=True,
                )
        # Konsistenzpruefung des Rasters innerhalb der Modalitaet
        grids = {
            (s["width"], s["height"], s["crs"], tuple(round(v, 10) for v in s["transform"]))
            for s in entry["samples"]
        }
        entry["n_distinct_grids"] = len(grids)
        entry["grid_consistent"] = len(grids) <= 1
        entry["band_count_values"] = sorted({s["count"] for s in entry["samples"]})
        out[mod] = entry

    # Gitter-Vergleich ueber alle Modalitaeten
    global_grids: dict[str, list[str]] = {}
    for mod, entry in out.items():
        for s in entry["samples"]:
            key = json.dumps(
                {
                    "w": s["width"],
                    "h": s["height"],
                    "crs": s["crs"],
                    "t": [round(v, 10) for v in s["transform"]],
                },
                sort_keys=True,
            )
            global_grids.setdefault(key, [])
            if mod not in global_grids[key]:
                global_grids[key].append(mod)

    print(f"[s2] verschiedene Gitter ueber alle Modalitaeten: {len(global_grids)}")
    for key, mods in global_grids.items():
        print(f"    {key[:150]} <- {mods}")

    write_manifest(
        cfg,
        "s2_rasters",
        args.config,
        {
            "sampled_years": years,
            "files_per_year": k,
            "modalities": out,
            "distinct_grids": [{"grid": json.loads(k_), "modalities": v} for k_, v in global_grids.items()],
            "all_modalities_share_one_grid": len(global_grids) == 1,
        },
    )


if __name__ == "__main__":
    main()
