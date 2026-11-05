"""MPCFire's MODIS surface channels rebuilt from daily Earth Engine downloads.

Per product and day, Earth Engine applies MPCFire's QA mask and Terra/Aqua choice and returns one 1 km image
(0 = no valid value). Locally, each 1 km pixel keeps its most recent valid value (MPCFire's temporal fill, which has
no age limit; here it starts at a spin-up day), and each release pixel is the mean of the valid values in a
10 x 10 window of 1 km pixels when at least half are valid (MPCFire's downsampling), else no data.

MPCFire averaged per download tile, so each quarter of the map has its own window position: release pixel (I, J)
holds 1 km rows 5I + fy - 5 .. 5I + fy + 4 and columns 5J + fx - 5 .. 5J + fx + 4, with (fy, fx) measured per
quarter on released land temperatures (demo/live/README.md, "Alignment"). The QA and choice rules were calibrated
against MPCFire's released rasters on 121 days of 2025 (README, "Recipe and fidelity per channel").
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import ee_io
from .encode import encode
from .grid import ONE_KM, R1, RELEASE
from .inputs import CACHE, cached, next_day

ROW_SEAM, COL_SEAM = 465, 984
FOOT = {"north": 0, "south": 2, "west": 0, "east": -1}
MARGIN = 8                       # 1 km pixels around the release grid: windows reach 5 back and 6 forward
ONE_KM_TRANSFORM = [R1, 0, 0, 0, -R1, 0]
DOWNLOAD = ONE_KM.window(-MARGIN, -MARGIN, ONE_KM.height + 2 * MARGIN, ONE_KM.width + 2 * MARGIN)


def _bits(img, start: int, n: int):
    return img.rightShift(start).bitwiseAnd((1 << n) - 1)


def _scene(collection: str, day: str, bands: list[str]):
    """The day's mosaic of a MODIS collection, or a fully masked image with the same bands when it has none."""
    import ee
    col = ee.ImageCollection(collection).filterDate(day, next_day(day)).select(bands)
    empty = ee.Image.constant([0] * len(bands)).rename(bands).toInt32().updateMask(0)
    return ee.Image(ee.Algorithms.If(col.size().gt(0), col.mosaic().toInt32(), empty))


# ---- MCD11A1 ---------------------------------------------------------------------------------------------------

LST = ["LST_Day_1km", "LST_Night_1km", "Emis_31", "Emis_32"]


def _mcd11a1_parts(collection: str, day: str):
    img = _scene(collection, day, LST + ["QC_Day", "QC_Night"])
    qd, qn = img.select("QC_Day").unmask(255), img.select("QC_Night").unmask(255)
    md, dqd, eed, led = _bits(qd, 0, 2), _bits(qd, 2, 2), _bits(qd, 4, 2), _bits(qd, 6, 2)
    mn, dqn = _bits(qn, 0, 2), _bits(qn, 2, 2)
    bad_d = md.gte(2).Or(dqd.gte(2))
    bad_dn = bad_d.Or(mn.gte(2).Or(dqn.gte(2)))
    v = img.select(LST).unmask(0)
    # calibrated: the day QC key chooses the sensor for all four bands; LST_Day is masked by the day QC,
    # LST_Night only by Earth Engine's own mask, the emissivities by the day or night QC
    ok = [v.select(0).gt(0).And(bad_d.Not()), v.select(1).gt(0), v.select(2).gt(0).And(bad_dn.Not()),
          v.select(3).gt(0).And(bad_dn.Not())]
    key = md.multiply(64).add(led.multiply(16)).add(dqd.multiply(4)).add(eed)
    return v, ok, key


def mcd11a1_day(day: str):
    import ee
    vm, om, km = _mcd11a1_parts("MODIS/061/MOD11A1", day)
    vy, oy, ky = _mcd11a1_parts("MODIS/061/MYD11A1", day)
    out = []
    for b in range(4):
        pick_y = om[b].Not().And(oy[b]).Or(om[b].And(oy[b]).And(ky.lt(km)))
        val = vm.select(b).where(pick_y, vy.select(b))
        ok = om[b].where(pick_y, oy[b])
        out.append(val.multiply(ok))
    return ee.Image.cat(out).rename(LST).toUint16()


# ---- MCD09GA (surface reflectance b01, b02, b03, b07) -----------------------------------------------------------

REFL = ["sur_refl_b01", "sur_refl_b02", "sur_refl_b03", "sur_refl_b07"]


def _mcd09ga_parts(collection: str, day: str):
    img = _scene(collection, day, REFL + ["QC_500m", "state_1km"])
    q = img.select("QC_500m").unmask(0).bitwiseAnd(0xFFFF)     # MPCFire kept the low 16 bits
    st = img.select("state_1km").unmask(0)
    dq = _bits(q, 2, 4).max(_bits(q, 6, 4)).max(_bits(q, 10, 4)).max(_bits(q, 14, 2))
    key = (_bits(q, 0, 2).multiply(16).add(dq).multiply(4).add(_bits(st, 0, 2)).multiply(4).add(_bits(st, 6, 2))
           .multiply(4).add(_bits(st, 8, 2)).multiply(2).add(_bits(st, 2, 1)).multiply(2).add(_bits(st, 10, 1)))
    ok = st.gt(0).And(_bits(st, 0, 2).neq(1))                  # cloudy pixels dropped
    v = img.select(REFL).unmask(0)
    v = v.where(v.lt(0), 65000)        # MPCFire wrapped negative reflectance into uint16: encodes as 255
    return v, ok, key


def mcd09ga_day(day: str):
    vm, om, km = _mcd09ga_parts("MODIS/061/MOD09GA", day)
    vy, oy, ky = _mcd09ga_parts("MODIS/061/MYD09GA", day)
    pick_y = om.Not().And(oy).Or(om.And(oy).And(ky.lt(km)))
    ok = om.where(pick_y, oy)
    return vm.where(pick_y, vy).add(1).multiply(ok).rename(REFL).toUint16()   # stored +1, so 0 means no value


# ---- products ----------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Product:
    name: str                    # MPCFire modality name (encode ranges, cube slice)
    bands: list[str]
    day_image: object = field(repr=False)   # day -> ee.Image of uint16, 0 = no valid value
    bytes_per_value: int = 2
    offset: int = 0                          # stored value = value + offset (reflectance can be 0)


PRODUCTS = {"MCD11A1": Product("MCD11A1", LST, mcd11a1_day),
            "MCD09GA": Product("MCD09GA", REFL, mcd09ga_day, offset=1)}


def daily_1km(p: Product, day: str) -> np.ndarray:
    """(bands, H1 + 2 MARGIN, W1 + 2 MARGIN) uint16 of the day, around the release grid."""
    def make():
        img = p.day_image(day).reproject(crs="EPSG:4326", crsTransform=ONE_KM_TRANSFORM)
        return ee_io.pixels(img, DOWNLOAD, p.bands, p.bytes_per_value).astype(np.uint16)
    return cached(f"{p.name}_1km", day, make)


def window_means(state: np.ndarray) -> np.ndarray:
    """(bands, H, W) float32: per release pixel the mean of the valid (finite) 1 km values in its 10 x 10 window at
    its quarter's offsets when at least 50 are valid, else -32768."""
    H, W = RELEASE.height, RELEASE.width
    fy = np.where(np.arange(H) >= ROW_SEAM, FOOT["south"], FOOT["north"])
    fx = np.where(np.arange(W) >= COL_SEAM, FOOT["east"], FOOT["west"])
    r = MARGIN + 5 * np.arange(H) + fy - 5
    c = MARGIN + 5 * np.arange(W) + fx - 5
    out = np.empty((state.shape[0], H, W), np.float32)
    for b in range(state.shape[0]):
        valid = np.isfinite(state[b])
        x = np.where(valid, state[b], 0.0)
        cs = np.zeros((x.shape[0] + 1, x.shape[1] + 1))
        np.cumsum(np.cumsum(x, 0, dtype=np.float64), 1, out=cs[1:, 1:])
        cn = np.zeros((x.shape[0] + 1, x.shape[1] + 1), np.int64)
        np.cumsum(np.cumsum(valid, 0, dtype=np.int64), 1, out=cn[1:, 1:])
        R, C = r[:, None], c[None, :]
        s = cs[R + 10, C + 10] - cs[R, C + 10] - cs[R + 10, C] + cs[R, C]
        k = cn[R + 10, C + 10] - cn[R, C + 10] - cn[R + 10, C] + cn[R, C]
        with np.errstate(all="ignore"):
            out[b] = np.where(k >= 50, s / np.maximum(k, 1), -32768.0)
    return out


def series(name: str, days: list[str], spinup: int = 60) -> dict[str, np.ndarray]:
    """{day: (bands, H, W) uint8} for `days` (consecutive), the temporal fill started `spinup` days before the first."""
    p = PRODUCTS[name]
    want = set(days)
    out = {}
    todo = [d for d in days if not (CACHE / f"{p.name}_codes" / f"{d}.npy").exists()]
    if not todo:
        return {d: np.load(CACHE / f"{p.name}_codes" / f"{d}.npy") for d in days}
    d0 = dt.date.fromisoformat(min(days)) - dt.timedelta(days=spinup)
    state = None
    d = d0
    while d.isoformat() <= max(days):
        day = d.isoformat()
        x = daily_1km(p, day)
        if state is None:
            state = np.full(x.shape, np.nan, np.float32)
        ok = x > 0
        state[ok] = x[ok].astype(np.float32) - p.offset
        if day in want:
            out[day] = cached(f"{p.name}_codes", day, lambda: encode(p.name, window_means(state)))
            print(f"[modis] {p.name} {day}", flush=True)
        d += dt.timedelta(days=1)
    return out


# ---- MCD09CMG (brightness temperature, 5600 m, no window mean) ----------------------------------------------------

BT = [f"Coarse_Resolution_Brightness_Temperature_Band_{b}" for b in (20, 21, 31, 32)]


def _cmg_grid():
    from .grid import M_PER_DEG, anchored
    return anchored(5600 / M_PER_DEG, RELEASE)


def mcd09cmg_raw(day: str) -> np.ndarray:
    """(4, ...) K on the 5600 m grid: Terra where valid, else Aqua; 0 = no value (MPCFire masked hardly any)."""
    def make():
        import ee
        g = _cmg_grid()
        def one(c):
            return _scene(f"MODIS/061/{c}09CMG", day, BT).toFloat().unmask(0)
        m, y = one("MOD"), one("MYD")
        img = m.where(m.lte(0), y).reproject(crs="EPSG:4326", crsTransform=[g.res, 0, 0, 0, -g.res, 0])
        return ee_io.pixels(img, g, BT, 4)
    return cached("MCD09CMG_raw", day, make)


def mcd09cmg_series(days: list[str], spinup: int = 30) -> dict[str, np.ndarray]:
    from .inputs import to_release
    g = _cmg_grid()
    d = dt.date.fromisoformat(min(days)) - dt.timedelta(days=spinup)
    state, out = None, {}
    while d.isoformat() <= max(days):
        x = mcd09cmg_raw(d.isoformat())
        state = np.where(x > 0, x, -32765.0) if state is None else np.where(x > 0, x, state)
        if d.isoformat() in days:
            out[d.isoformat()] = encode("MCD09CMG", to_release(state, g, nodata=-32765.0))
        d += dt.timedelta(days=1)
    return out


# ---- MCD15A3H (LAI, FPAR; 4-day composites at 500 m) ----------------------------------------------------------------

def mcd15a3h_raw(start: str) -> np.ndarray:
    """(2, H, W) mean raw LAI and FPAR of a composite over each release pixel's 500 m pixels that pass MPCFire's QA
    (SCF_QC <= 3), -32768 where none."""
    def make():
        import ee
        img = ee.ImageCollection("MODIS/061/MCD15A3H").filterDate(start, next_day(start)).first()
        ok = _bits(img.select("FparLai_QC"), 5, 3).lte(3)
        v = img.select(["Lai", "Fpar"]).unmask(0).updateMask(ok).toFloat()
        mean = v.reduceResolution(ee.Reducer.mean(), maxPixels=256).reproject(
            crs="EPSG:4326", crsTransform=[RELEASE.res, 0, RELEASE.x0, 0, -RELEASE.res, RELEASE.y0])
        return ee_io.pixels(mean.unmask(-32768), RELEASE, ["Lai", "Fpar"], 4, max_cells=MAX_CELLS)
    return cached("MCD15A3H_5km", start, make)


MAX_CELLS = 512 * 512


def mcd15a3h_series(days: list[str], lookback: int = 40) -> dict[str, np.ndarray]:
    """Each day holds the newest composite that starts on or before it, earlier composites filling its gaps."""
    import ee
    ee_io.init()
    d0 = (dt.date.fromisoformat(min(days)) - dt.timedelta(days=lookback)).isoformat()
    starts = ee.ImageCollection("MODIS/061/MCD15A3H").filterDate(d0, next_day(max(days))) \
        .aggregate_array("system:time_start").getInfo()
    starts = sorted(dt.datetime.utcfromtimestamp(t / 1000).date().isoformat() for t in starts)
    state, out, k = None, {}, 0
    for day in sorted(days):
        while k < len(starts) and starts[k] <= day:
            x = mcd15a3h_raw(starts[k])
            state = x.copy() if state is None else np.where(x != -32768, x, state)
            k += 1
        out[day] = encode("MCD15A3H", state)
    return out


# ---- NDVI / EVI from the MCD09GA codes (MPCFire's route) --------------------------------------------------------------

def ndvi_evi(codes: np.ndarray) -> np.ndarray:
    from .encode import to_uint8, RANGES
    rf, nf, bf = (codes[i].astype(np.float64) / 255.0 for i in (0, 1, 2))
    with np.errstate(all="ignore"):
        ndvi = np.where(nf + rf == 0, np.nan, np.clip((nf - rf) / (nf + rf), -1, 1))
        den = nf + 6.0 * rf - 7.5 * bf + 1.0
        evi = np.where(den == 0, np.nan, np.clip(2.5 * (nf - rf) / den, -1, 1))
    return np.stack([to_uint8(ndvi.astype(np.float32), *RANGES["NDVI_EVI"][0]),
                     to_uint8(evi.astype(np.float32), *RANGES["NDVI_EVI"][1])])
