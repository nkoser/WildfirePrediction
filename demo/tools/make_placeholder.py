"""Placeholder model outputs in the schema of wildfire/paper/risk_map.py, for developing the page.

The risk values are NOT model output. They are a smooth background hazard plus a hazard that decays with
distance to the fires seen in the 30-day input window and with time after the forecast date. The observed
fires (fire_window, fire_today, first_fire) are the real MODIS MCD14A1 masks of MPCFire. The mask is
Canada's land from Natural Earth (1:50m) minus its lakes, a stand-in for land.npy & canada.npy.

Every file carries placeholder=True; export_web.py turns that into a banner on the page.

    python -I demo/tools/make_placeholder.py --mcd14a1 <dir with MCD14A1__<year>.tar> \
        --naturalearth <dir with ne_50m_*.geojson> --out <dir>
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_grid  # noqa: E402

DATES = ["2023-06-01", "2023-08-10", "2024-07-15", "2025-05-28"]   # the dates of the draft's risk-map figure
TAUS = [1, 3, 7, 14, 30]                                          # risk_map.py default --taus
T_WIN = 30


class FireMasks:
    """Daily MODIS fire masks from the yearly MPCFire tar archives (members YYYY_MM_DD.tif)."""

    def __init__(self, root: Path, grid: dict):
        self.root, self.grid, self.tars = root, grid, {}

    def day(self, d: dt.date) -> np.ndarray:
        tar = self.tars.get(d.year)
        if tar is None:
            tar = self.tars[d.year] = tarfile.open(self.root / f"MCD14A1__{d.year}.tar")
        im = Image.open(io.BytesIO(tar.extractfile(f"{d:%Y_%m_%d}.tif").read()))
        scale, tie = im.tag_v2[33550], im.tag_v2[33922]
        g = self.grid
        if not (np.isclose(tie[3], g["lon0"]) and np.isclose(tie[4], g["lat0"]) and np.isclose(scale[0], g["res"])):
            raise SystemExit(f"{d}: GeoTIFF georeference {tie[3:5]}, {scale[0]} does not match grid.json")
        a = np.asarray(im)
        if a.shape != (g["height"], g["width"]):
            raise SystemExit(f"{d}: shape {a.shape} does not match grid.json")
        return a > 0


def rings_of(geom: dict) -> list[np.ndarray]:
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    return [np.asarray(r, np.float64) for p in polys for r in p]


def rasterize(rings: list[np.ndarray], grid: dict) -> np.ndarray:
    """Even-odd scanline fill: a pixel is inside if its centre is inside. Holes and separate parts work
    because all rings are filled together and the parts do not overlap."""
    H, W, lon0, lat0, res = grid["height"], grid["width"], grid["lon0"], grid["lat0"], grid["res"]
    x1, y1, x2, y2 = [], [], [], []
    for r in rings:
        c, rr = (r[:, 0] - lon0) / res, (lat0 - r[:, 1]) / res
        x1.append(c[:-1]); x2.append(c[1:]); y1.append(rr[:-1]); y2.append(rr[1:])
    x1, y1, x2, y2 = (np.concatenate(v) for v in (x1, y1, x2, y2))
    keep = y1 != y2
    x1, y1, x2, y2 = x1[keep], y1[keep], x2[keep], y2[keep]
    ylo, yhi = np.minimum(y1, y2), np.maximum(y1, y2)
    out = np.zeros((H, W), bool)
    for row in range(H):
        y = row + 0.5
        sel = (ylo <= y) & (yhi > y)
        if not sel.any():
            continue
        xs = np.sort(x1[sel] + (y - y1[sel]) * (x2[sel] - x1[sel]) / (y2[sel] - y1[sel]))
        a = np.ceil(xs[0::2] - 0.5).astype(int)
        b = np.floor(xs[1::2] - 0.5).astype(int)
        for s, e in zip(np.clip(a, 0, W), np.clip(b, -1, W - 1)):
            if e >= s:
                out[row, s:e + 1] = True
    return out


def canada_mask(ne: Path, grid: dict) -> np.ndarray:
    countries = json.loads((ne / "ne_50m_admin_0_countries.geojson").read_text())
    geom = next(f["geometry"] for f in countries["features"] if f["properties"].get("ADM0_A3") == "CAN")
    land = rasterize(rings_of(geom), grid)
    lakes = json.loads((ne / "ne_50m_lakes.geojson").read_text())
    lon1 = grid["lon0"] + grid["width"] * grid["res"]
    lat1 = grid["lat0"] - grid["height"] * grid["res"]
    rings = []
    for f in lakes["features"]:
        rs = rings_of(f["geometry"])
        pts = np.concatenate(rs)
        if pts[:, 0].max() >= grid["lon0"] and pts[:, 0].min() <= lon1 and pts[:, 1].max() >= lat1 and pts[:, 1].min() <= grid["lat0"]:
            rings += rs
    water = rasterize(rings, grid) if rings else np.zeros_like(land)
    return land & ~water


def blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur with a unit-sum kernel, via FFT on a zero-padded array."""
    pad = int(4 * sigma) + 2
    P = np.pad(a.astype(np.float64), pad)
    fy = np.fft.fftfreq(P.shape[0])[:, None]
    fx = np.fft.rfftfreq(P.shape[1])[None, :]
    G = np.exp(-2.0 * (np.pi * sigma) ** 2 * (fy ** 2 + fx ** 2))
    out = np.fft.irfft2(np.fft.rfft2(P) * G, s=P.shape)
    return out[pad:-pad, pad:-pad]


def placeholder_F(mask, fire_window, fire_today, date: dt.date, grid: dict, taus, rng) -> dict[str, np.ndarray]:
    H = grid["height"]
    lat = grid["lat0"] - grid["res"] * (np.arange(H) + 0.5)
    lat_factor = (np.exp(-((lat - 56.0) / 7.0) ** 2) + 0.15)[:, None]
    noise = blur(rng.standard_normal(mask.shape), 25.0)
    noise = np.exp(0.6 * noise / noise.std())
    doy = date.timetuple().tm_yday
    season = np.exp(-((doy - 200) / 60.0) ** 2) + 0.2
    lam_bg = 4e-5 * lat_factor * noise * season                       # per day
    sigma = 2.0
    near = blur(fire_window, sigma) * 2 * np.pi * sigma ** 2          # sum of exp(-d^2 / 2 sigma^2) over fires
    lam0 = 0.025 * np.minimum(near, 10.0) + 0.05 * fire_today          # per day at the forecast date
    T = 12.0                                                           # days until the near-fire hazard fades
    out = {}
    for tau in taus:
        cum = lam_bg * tau + lam0 * T * (1 - np.exp(-tau / T))
        F = 1.0 - np.exp(-cum)
        out[f"F_{tau}"] = np.where(mask, F, np.nan).astype(np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mcd14a1", type=Path, required=True)
    ap.add_argument("--naturalearth", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dates", nargs="+", default=DATES)
    ap.add_argument("--taus", default=",".join(map(str, TAUS)))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    grid = load_grid()
    taus = [int(t) for t in args.taus.split(",")]
    fires = FireMasks(args.mcd14a1, grid)
    mask = canada_mask(args.naturalearth, grid)
    print(f"[placeholder] mask: {int(mask.sum()):,} of {mask.size:,} pixels", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for ds in args.dates:
        L = dt.date.fromisoformat(ds)
        win = [fires.day(L - dt.timedelta(days=k)) for k in range(T_WIN - 1, -1, -1)]
        fire_window = np.any(win, axis=0)
        fire_today = win[-1]
        first = np.zeros(mask.shape, np.uint8)
        for d in range(1, max(taus) + 1):
            f = fires.day(L + dt.timedelta(days=d))
            first[(first == 0) & f] = d
        res = placeholder_F(mask, fire_window, fire_today, L, grid, taus, rng)
        out = args.out / f"risk_placeholder_{ds}.npz"
        np.savez_compressed(out, **res, mask=mask, fire_window=fire_window, fire_today=fire_today,
                            first_fire=first, taus=np.array(taus), date=ds, doy=L.timetuple().tm_yday,
                            checkpoints=np.array(["placeholder"]), placeholder=True)
        outside = int((fire_window & ~mask).sum())
        print(f"[placeholder] {ds}: {out.name}; fires in window {int(fire_window.sum()):,} px "
              f"({outside:,} outside the mask), first fire within {max(taus)} d {int((first > 0).sum()):,} px; "
              f"F_7 median {np.nanmedian(res['F_7']):.2e}, p99.9 F_{max(taus)} "
              f"{np.nanpercentile(res[f'F_{max(taus)}'], 99.9):.3f}", flush=True)


if __name__ == "__main__":
    main()
