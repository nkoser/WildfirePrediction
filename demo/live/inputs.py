"""Rebuild MPCFire's daily input rasters for any date from Earth Engine, as the model was trained on them.

Each builder returns one modality of one day as MPCFire released it: uint8 codes, (bands, 928, 1968) on the release
grid. The recipe follows the FireMPC code (github.com/SynUW/FireMPC) as it was run, quirks included, and the
choices the code leaves open were measured against MPCFire's released 2025 rasters (see demo/live/README.md):

  fire       MOD14A1 (Terra only) FireMask == 9 on Earth Engine's 1 km grid, maximum over a 5 x 5 block
  ERA5       ERA5-Land hourly data aggregated to days over the 00-23 UTC steps on the 0.1-degree grid,
             masked pixels as 0, bilinear to the release grid
  VPD        Tetens from daily mean T2m and dewpoint on the 0.1-degree grid, invalid values interpolated
  FWI        FireMPC's CFFDRS recursion (its equations, not Van Wagner's where they differ) on the 0.1-degree grid

Earth Engine downloads are cached as .npy under LIVE_CACHE, so the local steps can be re-run without
downloading again.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import numpy as np

from . import ee_io
from .encode import RANGES, encode, to_uint8
from .grid import ERA5, ONE_KM, R1, RELEASE, TILED, Grid

CACHE = Path(os.environ.get("LIVE_CACHE", "live-cache"))
if os.environ.get("LIVE_REGION", "canada") != "canada":
    CACHE = CACHE / os.environ["LIVE_REGION"]

# MPCFire downloaded Canada as 2 x 2 tiles and cut each from its own corner, so a 1 km-based channel's release
# pixel (I, J) holds the 1 km block that starts at (5I + dy, 5J + dx), with (dy, dx) set by the tile. Measured on
# every 2025 fire-season day of released fire pixels (demo/live/README.md, "Alignment"): rows from 465 on follow
# the south tiles, columns from 984 on the east tiles; column 983 holds what either tile saw (the union).
COL_SEAM = 984 if TILED else RELEASE.width          # first release column of the east tiles
ROW_SEAM = 465 if TILED else RELEASE.height         # first release row of the south tiles
SEAM_COL_UNION = 983 if TILED else None             # the seam column: maximum of the west and east tiles' blocks
DX_EAST, DY_SOUTH = -1, -2

ERA5_BANDS = ["temperature_2m", "u_component_of_wind_10m", "v_component_of_wind_10m", "snow_cover",
              "total_precipitation_sum", "surface_latent_heat_flux_sum", "dewpoint_temperature_2m",
              "surface_pressure", "volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2",
              "volumetric_soil_water_layer_3", "volumetric_soil_water_layer_4", "temperature_2m_max",
              "skin_temperature_max", "potential_evaporation_sum", "total_evaporation_sum",
              "skin_reservoir_content", "surface_net_solar_radiation_sum"]


def next_day(day: str) -> str:
    return (dt.date.fromisoformat(day) + dt.timedelta(days=1)).isoformat()


def cached(kind: str, day: str, make) -> np.ndarray:
    path = CACHE / kind / f"{day}.npy"
    if path.exists():
        return np.load(path)
    ee_io.init()                                 # every builder downloads from Earth Engine
    arr = make()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    tmp.replace(path)
    return arr


def tile_offsets() -> tuple[np.ndarray, np.ndarray]:
    """Per release row and column, the offset of its 1 km block (dy, dx)."""
    dy = np.where(np.arange(RELEASE.height) >= ROW_SEAM, DY_SOUTH, 0)
    dx = np.where(np.arange(RELEASE.width) >= COL_SEAM, DX_EAST, 0)
    return dy, dx


def has_image(collection: str, day: str) -> bool:
    import ee
    ee_io.init()
    return ee_io.get_info(ee.ImageCollection(collection).filterDate(day, next_day(day)).size()) > 0


# ---- fire: MOD14A1 FireMask == 9, Terra only (MPCFire's combine step paired Terra with itself) -------------------

MARGIN = 5


def fire_1km(day: str) -> np.ndarray:
    """1 km FireMask == 9 of the day around the release grid (MARGIN extra pixels on every side), uint8."""
    def make():
        import ee
        win = ONE_KM.window(-MARGIN, -MARGIN, ONE_KM.height + 2 * MARGIN, ONE_KM.width + 2 * MARGIN)
        img = (ee.ImageCollection("MODIS/061/MOD14A1").filterDate(day, next_day(day)).select("FireMask").mosaic()
               .eq(9).unmask(0).toUint8().reproject(crs="EPSG:4326", crsTransform=[R1, 0, 0, 0, -R1, 0]).rename("f"))
        return ee_io.pixels(img, win, ["f"], 1)[0]
    return cached("MOD14A1_fire9_1km", day, make)


def blocks_max(f1: np.ndarray) -> np.ndarray:
    """Maximum over each release pixel's 5 x 5 block of 1 km pixels, at the tile offsets."""
    dy, dx = tile_offsets()
    rows = MARGIN + 5 * np.arange(RELEASE.height)[:, None] + dy[:, None] + np.arange(5)[None, :]
    cols = MARGIN + 5 * np.arange(RELEASE.width)[:, None] + dx[:, None] + np.arange(5)[None, :]
    a = f1[rows].max(axis=1)                     # (H, W1k + margins)
    out = a[:, cols].max(axis=2)                 # (H, W)
    if SEAM_COL_UNION is not None:               # the seam column: west block and east block together
        j = SEAM_COL_UNION
        east = MARGIN + 5 * j + DX_EAST + np.arange(5)
        out[:, j] = np.maximum(out[:, j], a[:, east].max(axis=1))
    return out


def fire(day: str) -> np.ndarray:
    return blocks_max(fire_1km(day))[None].astype(np.uint8)


def unobserved(day: str) -> np.ndarray:
    """(H, W) bool: the day's MOD14A1 (Terra) could not have seen a fire in the release pixel: at least half of its
    1 km pixels are cloud or not processed (FireMask 0, 1, 2, 4 or 6) and none is a fire (9). For the map's
    "cloud-covered, fire unknown" class; not a model input."""
    def make():
        import ee
        col = ee.ImageCollection("MODIS/061/MOD14A1").filterDate(day, next_day(day)).select("FireMask")
        fm = ee.Image(ee.Algorithms.If(col.size().gt(0), col.mosaic(), ee.Image.constant(0).rename("FireMask")))
        fm = fm.unmask(0)
        blind = fm.lte(2).Or(fm.eq(4)).Or(fm.eq(6)).toFloat()
        fire = fm.eq(9).toFloat()
        one_km = dict(crs="EPSG:4326", crsTransform=[R1, 0, 0, 0, -R1, 0])
        to_release = dict(crs="EPSG:4326", crsTransform=[RELEASE.res, 0, RELEASE.x0, 0, -RELEASE.res, RELEASE.y0])
        img = ee.Image.cat(blind.reproject(**one_km).reduceResolution(ee.Reducer.mean(), maxPixels=64).reproject(**to_release),
                           fire.reproject(**one_km).reduceResolution(ee.Reducer.max(), maxPixels=64).reproject(**to_release)
                           ).rename(["blind", "fire"])
        a = ee_io.pixels(img, RELEASE, ["blind", "fire"], 4, max_cells=512 * 512)
        return ((a[0] >= 0.5) & (a[1] == 0)).astype(np.uint8)
    return cached("MOD14A1_unobserved", day, make) > 0


FIRMS_7D = "https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_Canada_7d.csv"


def fire_nrt(day: str, csv_path: Path) -> np.ndarray:
    """(1, H, W) uint8 fire channel of `day` from FIRMS near-real-time MODIS detections, for days Earth Engine's
    MOD14A1 does not have yet: Terra detections with confidence >= 80 (MOD14 fire class 9, as in MPCFire's label),
    each placed in the 1 km pixel of its centre and pooled into release pixels like MOD14A1. Swath-based, so it
    approximates the gridded product; used only for fires observed after a forecast date."""
    import csv
    f1 = np.zeros((ONE_KM.height + 2 * MARGIN, ONE_KM.width + 2 * MARGIN), np.uint8)
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["acq_date"] != day or row["satellite"] not in ("T", "Terra") or int(row["confidence"]) < 80:
                continue
            r = int(np.floor((ONE_KM.y0 - float(row["latitude"])) / R1)) + MARGIN
            c = int(np.floor((float(row["longitude"]) - ONE_KM.x0) / R1)) + MARGIN
            if 0 <= r < f1.shape[0] and 0 <= c < f1.shape[1]:
                f1[r, c] = 1
    return blocks_max(f1)[None].astype(np.uint8)


# ---- ERA5-Land -------------------------------------------------------------------------------------------------

SUM_BANDS = {"total_precipitation_sum": "total_precipitation", "surface_latent_heat_flux_sum": "surface_latent_heat_flux",
             "potential_evaporation_sum": "potential_evaporation", "total_evaporation_sum": "total_evaporation",
             "surface_net_solar_radiation_sum": "surface_net_solar_radiation"}


MAX_BANDS = {"temperature_2m_max": "temperature_2m", "skin_temperature_max": "skin_temperature"}


WEATHER_END = os.environ.get("LIVE_WEATHER_END")   # after this day, weather repeats its last day (persistence)


def era5_raw(day: str) -> np.ndarray:
    if WEATHER_END and day > WEATHER_END:
        return _era5_raw(WEATHER_END)
    return _era5_raw(day)


def _era5_raw(day: str) -> np.ndarray:
    """(18, ...) ERA5-Land daily values on the 0.1-degree grid as MPCFire downloaded them, float64, masked pixels
    (ocean) as 0, all from Earth Engine's ERA5_LAND/HOURLY.

    Means and maxima are over the day's 24 hourly steps 00 to 23 UTC, as DAILY_AGGR computes them. The five daily
    sums are those 24 steps too (the 00 UTC step holds the previous day's last hour): the definition of DAILY_AGGR
    when MPCFire downloaded it, while today's DAILY_AGGR ends at 00 UTC of the next day. From the hourly
    accumulations A(t): sum = A(day 00:00) - A(previous day 23:00) + A(day 23:00). Every band reproduces MPCFire's
    released codes exactly (2025-03-01, 2025-06-20, 2025-07-14 and 2025-07-15, every land pixel). The hourly data
    also arrives about four days before DAILY_AGGR.
    """
    def make():
        import ee
        proj = dict(crs="EPSG:4326", crsTransform=[ERA5.res, 0, 0, 0, -ERA5.res, 0])
        hourly = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").filterDate(day, next_day(day))
        means = [b for b in ERA5_BANDS if b not in SUM_BANDS and b not in MAX_BANDS]
        out = dict(zip(means, ee_io.pixels(hourly.select(means).mean().unmask(0).toDouble().reproject(**proj),
                                           ERA5, means, 8)))
        mx = hourly.select(list(MAX_BANDS.values())).max().rename(list(MAX_BANDS)).unmask(0).toDouble().reproject(**proj)
        out.update(zip(MAX_BANDS, ee_io.pixels(mx, ERA5, list(MAX_BANDS), 8)))
        prev = (dt.date.fromisoformat(day) - dt.timedelta(days=1)).isoformat()
        acc = {}
        for key, ts in (("a00", f"{day}T00:00:00"), ("a23p", f"{prev}T23:00:00"), ("a23", f"{day}T23:00:00")):
            t0 = ee.Date(ts)
            img = (ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").filterDate(t0, t0.advance(1, "hour"))
                   .select(list(SUM_BANDS.values())).mosaic().unmask(0).toDouble().reproject(**proj))
            acc[key] = ee_io.pixels(img, ERA5, list(SUM_BANDS.values()), 8)
        sums = acc["a00"] - acc["a23p"] + acc["a23"]
        out.update(zip(SUM_BANDS, sums))
        return np.stack([out[b] for b in ERA5_BANDS])
    return cached("ERA5_LAND_mpcfire_hourly", day, make)


def era5_available(day: str) -> bool:
    """True when Earth Engine has the day's 24 hourly ERA5-Land steps (and the previous day's last one)."""
    import ee
    ee_io.init()
    prev = (dt.date.fromisoformat(day) - dt.timedelta(days=1)).isoformat()
    return ee_io.get_info(ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").filterDate(f"{prev}T23:00:00", next_day(day)).size()) == 25


def to_release(arr: np.ndarray, src: Grid, nodata: float = -32768.0) -> np.ndarray:
    """Bilinear onto the release grid as pre_processing_fast.py warps (srcNodata = dstNodata = INIT_DEST = nodata)."""
    import rasterio.warp
    from rasterio.crs import CRS
    arr = np.asarray(arr)
    out = np.full((arr.shape[0], RELEASE.height, RELEASE.width), nodata, np.float64)
    for b in range(arr.shape[0]):
        rasterio.warp.reproject(arr[b].astype(np.float64), out[b], src_transform=src.transform(),
                                src_crs=CRS.from_epsg(4326), dst_transform=RELEASE.transform(),
                                dst_crs=CRS.from_epsg(4326), resampling=rasterio.warp.Resampling.bilinear,
                                src_nodata=nodata, dst_nodata=nodata, init_dest_nodata=True)
    return out


def era5(day: str) -> np.ndarray:
    return encode("ERA5", to_release(era5_raw(day), ERA5))


# ---- VPD (climate_drought_calculation.py with its default spatial interpolation) ------------------------------------

def _interpolate_invalid(data: np.ndarray, lo: float, hi: float, max_distance: float = 3) -> np.ndarray:
    """filter_and_interpolate: values outside [lo, hi] or NaN, within max_distance pixels of a valid one, get the
    linear griddata interpolation of the valid pixels; farther ones keep their value."""
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree
    valid = ~np.isnan(data) & (data >= lo) & (data <= hi)
    result = data.copy()
    if valid.sum() < 4 or valid.all():
        return result
    rows, cols = np.where(valid)
    irows, icols = np.where(~valid)
    points = np.column_stack([rows, cols])
    xi = np.column_stack([irows, icols])
    close = cKDTree(points).query(xi, k=1, workers=-1)[0] <= max_distance
    if close.any():
        vals = griddata(points, data[rows, cols], xi[close], method="linear", fill_value=np.nan)
        ok = np.isfinite(vals)
        result[irows[close][ok], icols[close][ok]] = vals[ok]
    return result


def _es(t_c):
    return 0.611 * np.exp(17.27 * t_c / (t_c + 237.3))


def vpd_grid(day: str) -> np.ndarray:
    raw = era5_raw(day)
    with np.errstate(all="ignore"):
        t = _interpolate_invalid(raw[0] - 273.15, -100, 70)
        td = _interpolate_invalid(raw[6] - 273.15, -100, 60)
        v = np.maximum(_es(t) - _es(td), 0)
        return _interpolate_invalid(v, 0, 10)


def vpd(day: str) -> np.ndarray:
    return encode("VPD", to_release(vpd_grid(day)[None].astype(np.float32), ERA5, nodata=np.nan))


# ---- FWI: FireMPC fwi/fwi_module.py and fwi/fwi_export_daily_components.py, as written ---------------------------

FFMC_INIT, DMC_INIT, DC_INIT = 85.0, 6.0, 15.0


def _ffmc_next(ffmc_prev, temp, rh, wind, rain):
    ffmc_prev = ffmc_prev.astype(np.float32)
    mo = 147.2 * (101.0 - ffmc_prev) / (59.5 + ffmc_prev)
    rf = np.where(rain > 0.5, mo + 42.5 * rain * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rain)), mo)
    rf = np.clip(rf, 0.0, 250.0)
    ed = (0.942 * rh ** 0.679 + 11.0 * np.exp((rh - 100.0) / 10.0) + 0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh)))
    ew = (0.618 * rh ** 0.753 + 10.0 * np.exp((rh - 100.0) / 10.0) + 0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh)))
    ko = 0.424 * (1.0 - (rh / 100.0) ** 1.7) + 0.0694 * np.sqrt(wind) * (1.0 - (rh / 100.0) ** 8)
    kd = ko * 0.581 * np.exp(0.0365 * temp)
    kl = 0.424 * (1.0 - ((100.0 - rh) / 100.0) ** 1.7) + 0.0694 * np.sqrt(wind) * (1.0 - ((100.0 - rh) / 100.0) ** 8)
    kw = kl * 0.581 * np.exp(0.0365 * temp)
    m = np.where(rf > ed, ed + (rf - ed) * np.exp(-kd), np.where(rf < ew, ew - (ew - rf) * np.exp(-kw), rf))
    return np.clip(59.5 * (250.0 - m) / (147.2 + m), 0.0, 101.0).astype(np.float32)


def _dmc_next(dmc_prev, temp, rh, rain, month):
    dmc_prev = np.maximum(dmc_prev.astype(np.float32), 0.0)
    le = {4: 0.9, 5: 12.5, 6: 13.9, 7: 13.9, 8: 12.4, 9: 10.9, 10: 9.4}.get(month, 9.0)
    re = np.where(rain > 1.5, 0.92 * rain - 1.27, 0.0)
    mo = 20.0 + np.exp(5.6348 - dmc_prev / 43.43)
    b = np.where(dmc_prev <= 33, 100.0 / (0.5 + 0.3 * dmc_prev),
                 np.where(dmc_prev <= 65, 14.0 - 1.3 * np.log(dmc_prev), 6.2 * np.log(dmc_prev) - 17.2))
    mr = mo + 1000 * re / (48.77 + b * re)
    pr = 244.72 - 43.43 * np.log(mr - 20.0)
    pr = np.where(rain > 1.5, np.maximum(pr, 0.0), dmc_prev)
    k = np.where(temp > -1.1, 1.894 * (temp + 1.1) * (100.0 - rh) * le * 1e-6, 0.0)
    return np.maximum(pr + 100.0 * k, 0.0).astype(np.float32)


def _dc_next(dc_prev, temp, rain, month):
    dc_prev = np.maximum(dc_prev.astype(np.float32), 0.0)
    lf = {4: -1.6, 5: 1.2, 6: 4.0, 7: 4.0, 8: 3.0, 9: 0.4, 10: -2.0}.get(month, -1.6)
    rd = np.where(rain > 2.8, 0.83 * rain - 1.27, 0.0)
    qo = 800.0 * np.exp(-dc_prev / 400.0)
    qr = qo + 3.937 * rd
    dr = np.maximum(np.where(rain > 2.8, 400.0 * np.log(800.0 / qr), dc_prev), 0.0)
    v = np.maximum(np.where(temp > -2.8, 0.36 * (temp + 2.8) + lf, 0.0), 0.0)
    return np.maximum(dr + 0.5 * v, 0.0).astype(np.float32)


def _isi(ffmc, wind):
    fm = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    return (0.208 * np.exp(0.05039 * wind) * 91.9 * np.exp(-0.1386 * fm) * (1.0 + fm ** 5.31 / 4.93e7)).astype(np.float32)


def _bui(dmc, dc):
    den = dmc + 0.4 * dc
    bui = np.where(den < 1e-6, 0.0, np.where(dmc <= 0.4 * dc, 0.8 * dmc * dc / den,
                                              dmc - (1.0 - 0.8 * dc / den) * (0.92 + (0.0114 * dmc) ** 1.7)))
    return np.maximum(bui, 0.0).astype(np.float32)


def _fwi(isi, bui):
    bb = 0.1 * isi * (0.626 * bui ** 0.809 + 2.0)
    return np.maximum(np.where(bb > 1.0, np.exp(2.72 * (0.434 * np.log(np.maximum(bb, 1e-6))) ** 0.647), bb), 0.0).astype(np.float32)


def fwi_step(state: tuple[np.ndarray, np.ndarray, np.ndarray], day: str):
    """One day of the recursion on the 0.1-degree grid -> (new state, (6, ...) FFMC DMC DC ISI BUI FWI)."""
    raw = era5_raw(day)
    t_max, td, u10, v10, precip = (raw[ERA5_BANDS.index(b)].astype(np.float32) for b in
                                   ("temperature_2m_max", "dewpoint_temperature_2m", "u_component_of_wind_10m",
                                    "v_component_of_wind_10m", "total_precipitation_sum"))
    outside = (t_max == 0.0) | (td == 0.0)
    t_max, td, u10, v10, precip = (np.where(outside, np.nan, a).astype(np.float32) for a in (t_max, td, u10, v10, precip))
    precip = np.maximum(precip, 0.0)
    t_c, td_c = t_max - 273.15, td - 273.15
    with np.errstate(all="ignore"):
        rh = 100.0 * (np.exp(17.625 * td_c / (243.04 + td_c)) / np.exp(17.625 * t_c / (243.04 + t_c)))
        rh = np.clip(rh, 0.0, 100.0).astype(np.float32)
        wind = np.sqrt(u10 ** 2 + v10 ** 2) * 3.6
        rain = precip * 1000.0
        month = dt.date.fromisoformat(day).month
        ffmc = _ffmc_next(state[0], t_c, rh, wind, rain)
        dmc = _dmc_next(state[1], t_c, rh, rain, month)
        dc = _dc_next(state[2], t_c, rain, month)
        isi = _isi(ffmc, wind)
        bui = _bui(dmc, dc)
        fwi = _fwi(isi, bui)
    return (ffmc, dmc, dc), np.stack([ffmc, dmc, dc, isi, bui, fwi])


def fwi_encode(components: np.ndarray) -> np.ndarray:
    return encode("FWI", to_release(components, ERA5))


def fwi_state_from_release(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A starting state on the 0.1-degree grid from released FWI codes (6, H, W): FFMC, DMC and DC read at each
    0.1-degree pixel's centre, decoded to the middle of their code's interval; 255 (no data or >= vmax) gives vmax
    over land and NaN elsewhere."""
    yc = ERA5.y0 - (np.arange(ERA5.height) + 0.5) * ERA5.res
    xc = ERA5.x0 + (np.arange(ERA5.width) + 0.5) * ERA5.res
    r = np.clip(((RELEASE.y0 - yc) / RELEASE.res).astype(int), 0, RELEASE.height - 1)
    c = np.clip(((xc - RELEASE.x0) / RELEASE.res).astype(int), 0, RELEASE.width - 1)
    inside = (((RELEASE.y0 - yc) / RELEASE.res >= 0) & ((RELEASE.y0 - yc) / RELEASE.res < RELEASE.height))[:, None] & \
             (((xc - RELEASE.x0) / RELEASE.res >= 0) & ((xc - RELEASE.x0) / RELEASE.res < RELEASE.width))[None, :]
    state = []
    for b in range(3):
        lo, hi = RANGES["FWI"][b]
        code = codes[b][r[:, None], c[None, :]].astype(np.float32)
        val = lo + (code + 0.5) / 255.0 * (hi - lo)
        val = np.where(code == 255, np.float32(hi), val)
        state.append(np.where(inside, val, np.nan).astype(np.float32))
    return tuple(state)
