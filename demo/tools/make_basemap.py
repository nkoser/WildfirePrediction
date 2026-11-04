"""Compact basemap for the page from Natural Earth 1:50m vectors (public domain).

Land and country borders cover the world; lakes and province/state lines cover the data grid plus a
margin. Lines are simplified (Douglas-Peucker), rounded to 0.01 degree and delta-encoded as integer lists:
[lon0*100, lat0*100, dlon1, dlat1, ...]. app.js decodes them.

    python -I demo/tools/make_basemap.py --naturalearth <dir with ne_50m_*.geojson> --out demo/web/basemap.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_grid  # noqa: E402

SCALE = 100


def douglas_peucker(p: np.ndarray, tol: float) -> np.ndarray:
    n = len(p)
    if n < 3:
        return p
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        s, e = stack.pop()
        if e - s < 2:
            continue
        seg = p[e] - p[s]
        rel = p[s + 1:e] - p[s]
        length = float(np.hypot(seg[0], seg[1]))
        d = np.hypot(rel[:, 0], rel[:, 1]) if length == 0 else np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / length
        i = int(np.argmax(d))
        if d[i] > tol:
            m = s + 1 + i
            keep[m] = True
            stack += [(s, m), (m, e)]
    return p[keep]


def encode(p: np.ndarray, tol: float, closed: bool) -> list[int] | None:
    q = np.rint(douglas_peucker(np.asarray(p, np.float64), tol) * SCALE).astype(np.int64)
    q = q[np.r_[True, np.any(np.diff(q, axis=0) != 0, axis=1)]]
    if len(q) < (4 if closed else 2):
        return None
    d = np.vstack([q[:1], np.diff(q, axis=0)])
    return d.ravel().tolist()


def polygons(geom: dict) -> list:
    return [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]


def lines(geom: dict) -> list:
    return [geom["coordinates"]] if geom["type"] == "LineString" else geom["coordinates"]


def intersects(coords: np.ndarray, box) -> bool:
    return bool(coords[:, 0].max() >= box[0] and coords[:, 0].min() <= box[2]
                and coords[:, 1].max() >= box[1] and coords[:, 1].min() <= box[3])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--naturalearth", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tol", type=float, default=0.02, help="simplification tolerance in degrees")
    ap.add_argument("--margin", type=float, default=15.0, help="degrees around the data grid for lakes and provinces")
    args = ap.parse_args()
    g = load_grid()
    box = (g["lon0"] - args.margin, g["lat0"] - g["height"] * g["res"] - args.margin,
           g["lon0"] + g["width"] * g["res"] + args.margin, g["lat0"] + args.margin)
    read = lambda name: json.loads((args.naturalearth / f"ne_50m_{name}.geojson").read_text())["features"]  # noqa: E731

    out = {"scale": SCALE, "land": [], "lakes": [], "borders": [], "provinces": []}
    for key, src, regional in (("land", "land", False), ("lakes", "lakes", True)):
        for f in read(src):
            for poly in polygons(f["geometry"]):
                if regional and not intersects(np.asarray(poly[0]), box):
                    continue
                rings = [encode(np.asarray(r), args.tol, True) for r in poly]
                if rings[0] is not None:
                    out[key].append([r for r in rings if r is not None])
    for key, src, regional in (("borders", "admin_0_boundary_lines_land", False),
                               ("provinces", "admin_1_states_provinces_lines", True)):
        for f in read(src):
            for line in lines(f["geometry"]):
                c = np.asarray(line)
                if regional and not intersects(c, box):
                    continue
                e = encode(c, args.tol, False)
                if e is not None:
                    out[key].append(e)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, separators=(",", ":"))
    args.out.write_text(text)
    counts = {k: len(v) for k, v in out.items() if isinstance(v, list)}
    print(f"[basemap] {args.out} {len(text) / 1024:.0f} KB; features {counts}", flush=True)


if __name__ == "__main__":
    main()
