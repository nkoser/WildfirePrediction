"""MPCFire's MODIS surface channels rebuilt on Earth Engine, server side: quality masking, the Terra/Aqua choice,
the temporal fill (each pixel keeps its most recent valid value), and the 10 x 10 one-kilometre window mean that
became each 5 km pixel. Only the 5 km result is downloaded.

MPCFire averaged its MODIS channels per download tile, so each quarter of the map has its own window position:
release pixel (I, J) holds the mean over 1 km rows 5I + fy - 5 .. 5I + fy + 4 and columns 5J + fx - 5 ..
5J + fx + 4, with (fy, fx) per quarter measured on released land temperatures (demo/live/README.md, "Alignment").
The temporal fill in MPCFire has no age limit; here it looks back `fill_days`.
"""

from __future__ import annotations

import datetime as dt

import numpy as np

from . import ee_io
from .encode import encode
from .grid import R1, RELEASE, Grid
from .inputs import cached, next_day

ROW_SEAM, COL_SEAM = 465, 984
FOOT = {"north": 0, "south": 2, "west": 0, "east": -1}     # footprint offsets of the 10 x 10 windows
ONE_KM_TRANSFORM = [R1, 0, 0, 0, -R1, 0]
NODATA = -32768.0
MAX_CELLS = 256 * 256    # output pixels per request: the window means stay within Earth Engine's memory


def _bits(img, start: int, n: int):
    return img.rightShift(start).bitwiseAnd((1 << n) - 1)


def _day(collection: str, day: str):
    import ee
    return ee.ImageCollection(collection).filterDate(day, next_day(day)).mosaic()


def _days_back(day: str, n: int) -> list[str]:
    d = dt.date.fromisoformat(day)
    return [(d - dt.timedelta(days=k)).isoformat() for k in range(n - 1, -1, -1)]


def _window_mean(img, bands: list[str]):
    """10 x 10 mean of the valid 1 km pixels around each pixel (window -5 .. +4), masked where fewer than 50."""
    import ee
    kernel = ee.Kernel.fixed(10, 10, [[1] * 10] * 10, 5, 5)
    one_km = img.reproject(crs="EPSG:4326", crsTransform=ONE_KM_TRANSFORM)
    mean = one_km.reduceNeighborhood(reducer=ee.Reducer.mean(), kernel=kernel, skipMasked=False)
    count = one_km.mask().reduceNeighborhood(reducer=ee.Reducer.sum(), kernel=kernel, skipMasked=False)
    out = mean.rename(bands).updateMask(count.gte(50).rename(bands))
    return out.reproject(crs="EPSG:4326", crsTransform=ONE_KM_TRANSFORM).unmask(NODATA).toFloat()


def _sample_quarters(img, bands: list[str]) -> np.ndarray:
    """(bands, H, W): each quarter sampled at its windows' 1 km focus pixels."""
    out = np.empty((len(bands), RELEASE.height, RELEASE.width), np.float32)
    for rows, fy in ((slice(0, ROW_SEAM), FOOT["north"]), (slice(ROW_SEAM, RELEASE.height), FOOT["south"])):
        for cols, fx in ((slice(0, COL_SEAM), FOOT["west"]), (slice(COL_SEAM, RELEASE.width), FOOT["east"])):
            g = Grid(x0=RELEASE.x0 + (5 * cols.start + fx - 2) * R1, y0=RELEASE.y0 - (5 * rows.start + fy - 2) * R1,
                     res=5 * R1, width=cols.stop - cols.start, height=rows.stop - rows.start)
            out[:, rows, cols] = ee_io.pixels(img, g, bands, 4, max_cells=MAX_CELLS)
    return out


# ---- MCD11A1: land surface temperature (day, night) and emissivity (bands 31, 32) ----------------------------------

LST_BANDS = ["LST_Day_1km", "LST_Night_1km", "Emis_31", "Emis_32"]


def _mcd11a1_one(collection: str, day: str):
    """One sensor's day: the four bands masked where MPCFire's QA mask drops them, and the selection scores."""
    import ee
    img = _day(collection, day)
    qd, qn = img.select("QC_Day").unmask(255), img.select("QC_Night").unmask(255)   # a missing scene scores worst
    md, dqd, eed, led = _bits(qd, 0, 2), _bits(qd, 2, 2), _bits(qd, 4, 2), _bits(qd, 6, 2)
    mn, dqn, een, len_ = _bits(qn, 0, 2), _bits(qn, 2, 2), _bits(qn, 4, 2), _bits(qn, 6, 2)
    bad_d = md.gte(2).Or(dqd.gte(2))
    bad_n = mn.gte(2).Or(dqn.gte(2))
    vals = ee.Image.cat(img.select("LST_Day_1km").updateMask(bad_d.Not()),
                        img.select("LST_Night_1km").updateMask(bad_n.Not()),
                        img.select("Emis_31").updateMask(bad_d.Or(bad_n).Not()),
                        img.select("Emis_32").updateMask(bad_d.Or(bad_n).Not())).toFloat()
    score_d = md.multiply(64).add(led.multiply(16)).add(dqd.multiply(4)).add(eed)
    score_n = mn.multiply(64).add(len_.multiply(16)).add(dqn.multiply(4)).add(een)
    score_e = md.multiply(16).add(eed.multiply(4)).add(dqd).min(mn.multiply(16).add(een.multiply(4)).add(dqn))
    return vals, ee.Image.cat(score_d, score_n, score_e, score_e)


def mcd11a1_combined(day: str):
    """Terra and Aqua of one day, per pixel and band the one with the better QA score (ties: Terra)."""
    import ee
    v_mod, s_mod = _mcd11a1_one("MODIS/061/MOD11A1", day)
    v_myd, s_myd = _mcd11a1_one("MODIS/061/MYD11A1", day)
    pick_myd = s_myd.lt(s_mod)
    return ee.Image(v_mod.where(pick_myd, v_myd)).rename(LST_BANDS).set("system:time_start", ee.Date(day).millis())


def mcd11a1(day: str, fill_days: int = 30) -> np.ndarray:
    """(4, H, W) uint8: the released MCD11A1 channel of `day`."""
    def make():
        import ee
        filled = ee.ImageCollection([mcd11a1_combined(d) for d in _days_back(day, fill_days)]).mosaic()
        return _sample_quarters(_window_mean(filled, LST_BANDS), LST_BANDS)
    arr = cached(f"MCD11A1_fill{fill_days}", day, make)
    return encode("MCD11A1", np.where(arr == NODATA, -32768, arr))
