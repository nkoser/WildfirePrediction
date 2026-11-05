"""The 13 static channels for a region other than Canada, and its masks.

MPCFire built its static channels from sources and per-file normalisations that are not all reproducible
(OpenStreetMap history extracts, a MERIT stack, WorldPop rasters of unknown product). Here each channel gets a raw
stand-in from Earth Engine, computed the same way for Canada and for the new region, and the stand-in is turned into
codes with the mapping that takes Canada's stand-in to MPCFire's released Canadian codes (matching quantiles over
Canadian land). The model thus sees values on the scale it learned, with the same meaning:

  OSMs      distances to built-up land (GHSL; stands in for roads, power lines and buildings) and to surface water
            (MODIS land cover class water), in pixels, log1p
  Worldpop  WorldPop 2020 population per pixel, log1p
  DEMs      MERIT elevation, slope, aspect (sin, cos), hillshade, TPI (11 x 11) and a slope-only wetness index
  MCD12Q1   MODIS land cover (IGBP) of 2024, the most common class, copied as MPCFire does

    LIVE_REGION=europe python -m demo.live.statics --canada-stat <stat_2025.npy> --out <experiment root>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import ee_io
from .grid import REGIONS, Grid

DEG = 1.0


def _raw(region: str) -> dict[str, np.ndarray]:
    """Stand-ins on a region's release grid (computed in a subprocess, since the grid is fixed per process)."""
    cache = Path(os.environ.get("LIVE_CACHE", "live-cache")) / f"static_raw_{region}.npz"
    if cache.exists():
        return dict(np.load(cache))
    env = dict(os.environ, LIVE_REGION=region)
    subprocess.run([sys.executable, "-c", f"from demo.live.statics import _compute; _compute({str(cache)!r})"], env=env, check=True)
    return dict(np.load(cache))


def _compute(out: str) -> None:
    import ee
    from scipy import ndimage
    from .grid import RELEASE
    ee_io.init()
    g: Grid = RELEASE
    proj = dict(crs="EPSG:4326", crsTransform=[g.res, 0, g.x0, 0, -g.res, g.y0])

    def mean(img, band, max_pixels=4096):
        from .grid import R1
        one_km = img.select([band]).rename("v").toFloat().reduceResolution(ee.Reducer.mean(), maxPixels=max_pixels) \
            .reproject(crs="EPSG:4326", crsTransform=[R1, 0, 0, 0, -R1, 0])          # native -> 1 km -> release grid
        im = one_km.reduceResolution(ee.Reducer.mean(), maxPixels=64).reproject(**proj)
        return ee_io.pixels(im.unmask(0), g, ["v"], 4, max_cells=64 * 64)[0]

    dem = mean(ee.Image("MERIT/DEM/v1_0_3"), "dem")
    built = mean(ee.ImageCollection("JRC/GHSL/P2023A/GHS_BUILT_S").filter(ee.Filter.eq("system:index", "2020")).first(), "built_surface")
    pop = mean(ee.ImageCollection("WorldPop/GP/100m/pop").filterDate("2020-01-01", "2021-01-01").mosaic()
               .setDefaultProjection(crs="EPSG:4326", scale=100), "population")
    lc = ee.ImageCollection("MODIS/061/MCD12Q1").filterDate("2024-01-01", "2025-01-01").first().select("LC_Type1")
    lc = lc.reduceResolution(ee.Reducer.mode(), maxPixels=1024).reproject(crs="EPSG:4326", crsTransform=[g.res / 5, 0, 0, 0, -g.res / 5, 0]) \
        .reduceResolution(ee.Reducer.mode(), maxPixels=64).reproject(**proj)
    landcover = ee_io.pixels(lc.unmask(0).toFloat().rename("v"), g, ["v"], 4, max_cells=256 * 256)[0]
    countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    region = {"canada": countries.filter(ee.Filter.eq("country_na", "Canada")),
              "europe": countries.filter(ee.Filter.eq("wld_rgn", "Europe"))}[os.environ.get("LIVE_REGION", "canada")]
    inside = ee_io.pixels(ee.Image(0).paint(region, 1).toFloat().rename("v").reproject(**proj), g, ["v"], 4)[0] > 0

    lat = g.y0 - (np.arange(g.height) + 0.5) * g.res
    dy = 111320.0 * g.res
    dx = (111320.0 * g.res * np.cos(np.radians(lat)))[:, None]
    gy, gx = np.gradient(dem, dy, axis=0), np.gradient(dem, axis=1) / dx
    slope = np.hypot(gx, gy)
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(315), np.radians(35)
    hill = np.clip(np.sin(alt) * np.cos(np.arctan(slope)) + np.cos(alt) * np.sin(np.arctan(slope)) * np.cos(az - aspect), 0, 1)
    tpi = dem - ndimage.uniform_filter(dem, 11)
    twi = -np.log(slope + 1e-4)
    dist = lambda m: np.log1p(ndimage.distance_transform_edt(~m))
    frac = built / 1e4                              # built-up share of the pixel (GHSL: m2 per 100 m cell)
    np.savez(out, osm_roads=dist(frac > 0.005), osm_power=dist(frac > 0.02), osm_buildings=dist(frac > 0.01),
             osm_water=dist(landcover == 17),
             pop=np.log1p(pop), dem=dem, slope=slope, aspect_sin=np.sin(aspect), aspect_cos=np.cos(aspect),
             hillshade=hill, tpi=tpi, twi=twi, landcover=landcover, land=(landcover > 0) & (landcover != 17), inside=inside)


ORDER = ["osm_roads", "osm_power", "osm_buildings", "osm_water", "pop", "dem", "slope", "aspect_sin", "aspect_cos",
         "hillshade", "tpi", "twi"]


def quantile_map(src_canada: np.ndarray, codes_canada: np.ndarray, src_new: np.ndarray) -> np.ndarray:
    """Codes for `src_new` from the monotone relation that matches Canada's stand-in quantiles to its codes."""
    q = np.linspace(0, 1, 1001)
    xs = np.quantile(src_canada, q)
    ys = np.quantile(codes_canada.astype(np.float64), q)
    xs, idx = np.unique(xs, return_index=True)
    return np.clip(np.rint(np.interp(src_new, xs, ys[idx])), 0, 254).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--canada-stat", type=Path, required=True, help="MPCFire's static codes of Canada (stat_<year>.npy)")
    ap.add_argument("--canada-masks", type=Path, required=True, help="Canada's land.npy and canada.npy")
    ap.add_argument("--out", type=Path, required=True, help="experiment root of the new region")
    args = ap.parse_args()
    region = os.environ.get("LIVE_REGION", "canada")
    can, new = _raw("canada"), _raw(region)
    stat = np.load(args.canada_stat)
    m = np.load(args.canada_masks / "land.npy") & np.load(args.canada_masks / "canada.npy")
    out = np.zeros(new["dem"].shape + (13,), np.uint8)
    for i, name in enumerate(ORDER):
        out[..., i] = quantile_map(can[name][m], stat[..., i][m], new[name])
    out[..., 12] = new["landcover"].astype(np.uint8)
    (args.out / "derived_data" / "masks").mkdir(parents=True, exist_ok=True)
    (args.out / "derived_data" / "cube").mkdir(parents=True, exist_ok=True)
    (args.out / "data_manifests").mkdir(parents=True, exist_ok=True)
    np.save(args.out / "derived_data" / "cube" / "stat_static.npy", out)
    np.save(args.out / "derived_data" / "masks" / "land.npy", new["land"])
    np.save(args.out / "derived_data" / "masks" / "canada.npy", new["inside"])      # risk_map.py's name for the area
    g = REGIONS[region]
    (args.out / "data_manifests" / "d0_masks.json").write_text(json.dumps({"grid": {"height": g.height, "width": g.width}}))
    print(f"[statics] {region}: {int((new['land'] & new['inside']).sum()):,} forecast pixels; static codes in {args.out}")


if __name__ == "__main__":
    main()
