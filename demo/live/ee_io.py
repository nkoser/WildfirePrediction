"""Earth Engine access for the live input pipeline: sign-in, and pixel downloads on a fixed grid.

Credentials come from the Earth Engine client's own store (`earthengine authenticate`); the Cloud project from
EE_PROJECT. Requests carry the workload tag `wildfire-demo-live`, so the project's EECU page reports their cost.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .grid import Grid

WORKLOAD_TAG = "wildfire-demo-live"
MAX_REQUEST_BYTES = 30e6          # computePixels answers up to 48 MB; stay well below
_ready = False


def init() -> None:
    global _ready
    if _ready:
        return
    import ee
    project = os.environ.get("EE_PROJECT")
    if not project:
        raise SystemExit("EE_PROJECT is not set (the Cloud project registered for Earth Engine)")
    ee.Initialize(project=project)
    ee.data.setWorkloadTag(WORKLOAD_TAG)
    _ready = True


def _request(image, grid: Grid, bands: list[str], tries: int = 8) -> np.ndarray:
    import ee
    for attempt in range(tries):
        try:
            raw = ee.data.computePixels({"expression": image, "fileFormat": "NUMPY_NDARRAY",
                                         "grid": grid.ee_grid(), "bandIds": bands})
            return np.stack([raw[b] for b in bands])
        except ee.EEException as e:
            msg = str(e)
            retry = any(s in msg for s in ("Too many", "429", "503", "500", "timed out", "deadline", "Internal error"))
            if not retry or attempt == tries - 1:
                raise
            time.sleep(min(120, 5 * 2 ** attempt))
    raise RuntimeError("unreachable")


def pixels(image, grid: Grid, bands: list[str], bytes_per_value: int = 4, workers: int = 6,
           max_cells: int | None = None) -> np.ndarray:
    """(bands, height, width) array of `image` sampled on `grid`, in blocks that each fit one request.

    Earth Engine samples at pixel centres in the image's own resampling (nearest unless the image says
    otherwise), exactly as an export on the same grid would. `max_cells` caps the output pixels per request for
    images that are costly to compute (Earth Engine's per-request memory), tiling in both directions.
    """
    init()
    # Earth Engine counts a request's size with about one byte more per value than the value itself (a uint8
    # band of 30 M pixels counts as 60 MB against its 48 MB limit), so blocks are sized with that margin.
    cells = int(MAX_REQUEST_BYTES // (len(bands) * (bytes_per_value + 1) * 1.2))
    if max_cells:
        cells = min(cells, max_cells)
    bw = min(grid.width, max(1, cells))
    bh = min(grid.height, max(1, cells // bw))
    blocks = [(r0, min(grid.height, r0 + bh), c0, min(grid.width, c0 + bw))
              for r0 in range(0, grid.height, bh) for c0 in range(0, grid.width, bw)]

    def fetch(b):
        r0, r1, c0, c1 = b
        return b, _request(image, grid.window(r0, c0, r1 - r0, c1 - c0), bands)

    with ThreadPoolExecutor(max_workers=min(workers, len(blocks))) as ex:
        results = list(ex.map(fetch, blocks))
    out = np.empty((len(bands), grid.height, grid.width), dtype=results[0][1].dtype)
    for (r0, r1, c0, c1), part in results:
        out[:, r0:r1, c0:c1] = part
    return out
