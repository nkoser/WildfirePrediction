"""Grids of the live input pipeline, all in EPSG:4326.

MPCFire's maps (the model's inputs) are on a grid of 1968 x 928 pixels of 0.04491576 degrees, cut from Earth
Engine's 1 km grid: Earth Engine turns a scale in metres into degrees at 111319.49 m per degree and anchors the
pixel grid at 0 degrees, and the release grid's corner sits on that 1 km grid at (-15697, 9278) pixels with
exactly 5 x 5 one-kilometre pixels per release pixel. ERA5-Land was downloaded at 11132 m, a 0.1000005-degree
grid anchored at 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

M_PER_DEG = 111319.49079327357   # Earth Engine's metres per degree for scales in EPSG:4326
R1 = 1000 / M_PER_DEG            # the 1 km grid: 0.008983152841195215 degrees


@dataclass(frozen=True)
class Grid:
    """A north-up pixel grid: outer top-left corner (x0, y0), square pixels of `res` degrees."""
    x0: float
    y0: float
    res: float
    width: int
    height: int

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0 - self.height * self.res, self.x0 + self.width * self.res, self.y0)

    def transform(self):
        from affine import Affine
        return Affine(self.res, 0.0, self.x0, 0.0, -self.res, self.y0)

    def ee_grid(self) -> dict:
        """The grid as Earth Engine's computePixels takes it."""
        return {"dimensions": {"width": self.width, "height": self.height},
                "affineTransform": {"scaleX": self.res, "shearX": 0, "translateX": self.x0,
                                    "shearY": 0, "scaleY": -self.res, "translateY": self.y0},
                "crsCode": "EPSG:4326"}

    def rows(self, r0: int, r1: int) -> "Grid":
        """Rows r0 .. r1 - 1 as a grid of their own."""
        return Grid(self.x0, self.y0 - r0 * self.res, self.res, self.width, r1 - r0)

    def window(self, r0: int, c0: int, h: int, w: int) -> "Grid":
        return Grid(self.x0 + c0 * self.res, self.y0 - r0 * self.res, self.res, w, h)


import os

# LIVE_REGION picks the grid. "canada" is MPCFire's release grid; "europe" is the same pixel size on the same 1 km
# grid over Europe (11 W to 35 E, 34 to 71.5 N), without MPCFire's download-tile offsets.
REGION = os.environ.get("LIVE_REGION", "canada")
REGIONS = {"canada": Grid(x0=-15697 * R1, y0=9278 * R1, res=5 * R1, width=1968, height=928),
           "europe": Grid(x0=-1225 * R1, y0=7960 * R1, res=5 * R1, width=1025, height=835)}
RELEASE = REGIONS[REGION]
TILED = REGION == "canada"       # MPCFire's 2 x 2 download tiles shifted Canada's 1 km blocks per quarter
ONE_KM = Grid(x0=RELEASE.x0, y0=RELEASE.y0, res=R1, width=RELEASE.width * 5, height=RELEASE.height * 5)


def anchored(res: float, around: Grid, margin: int = 2) -> Grid:
    """The grid of `res` degrees anchored at 0 that covers `around` with `margin` extra pixels on every side."""
    xmin, ymin, xmax, ymax = around.bounds
    c0, c1 = math.floor(xmin / res) - margin, math.ceil(xmax / res) + margin
    r0, r1 = math.floor(ymin / res) - margin, math.ceil(ymax / res) + margin
    return Grid(x0=c0 * res, y0=r1 * res, res=res, width=c1 - c0, height=r1 - r0)


ERA5 = anchored(11132 / M_PER_DEG, RELEASE)   # ERA5-Land as MPCFire downloaded it, around the release grid
