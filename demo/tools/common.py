"""Shared helpers for the demo tools: grid, quantization of risk values, image encoding.

Risk values F(tau) are stored for the browser as uint8 on a log10 scale between QUANT["vmin"] and
QUANT["vmax"]; 0 means "no data" (outside the model's mask). With 255 levels over five decades one step is
0.02 decades, a relative error below 2.3 % after rounding, far below what a color scale can show.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

DEMO = Path(__file__).resolve().parents[1]

QUANT = {"kind": "log10", "vmin": 1e-5, "vmax": 1.0, "levels": 255, "nodata": 0}


def load_grid(path: Path | None = None) -> dict:
    return json.loads(Path(path or DEMO / "grid.json").read_text())


def quantize_log(F: np.ndarray, q: dict = QUANT) -> np.ndarray:
    """float array (NaN = no data) -> uint8, 1..levels on a log10 scale, 0 = no data."""
    lo, hi = np.log10(q["vmin"]), np.log10(q["vmax"])
    out = np.zeros(F.shape, np.uint8)
    ok = np.isfinite(F)
    t = (np.log10(np.clip(F[ok], q["vmin"], q["vmax"])) - lo) / (hi - lo)
    out[ok] = (1 + np.rint(t * (q["levels"] - 1))).astype(np.uint8)
    return out


def dequantize_log(v: np.ndarray, q: dict = QUANT) -> np.ndarray:
    lo, hi = np.log10(q["vmin"]), np.log10(q["vmax"])
    v = np.asarray(v, np.float64)
    return np.where(v > 0, 10 ** (lo + (v - 1) / (q["levels"] - 1) * (hi - lo)), np.nan)


def encode_png(a: np.ndarray) -> bytes:
    """Lossless 8-bit grayscale PNG; no gamma or color-profile chunks, so a browser canvas returns the
    stored values unchanged."""
    if a.dtype != np.uint8 or a.ndim != 2:
        raise ValueError(f"expected a 2-D uint8 array, got {a.dtype} {a.shape}")
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")
