"""MPCFire's float-to-uint8 encoding, copied from FireMPC `newset_processing/scale_int_tif.py`
(`_transform_block_to_uint8` and the per-modality band configs in its argument defaults).

code = uint8(clip((x - vmin) / (vmax - vmin) * 255, 0, 255)) in float32, truncating; x in {-32768, 32768, 255, NaN}
gives 255 (for MCD15A3H, -32768 and 32768 give 0). Values at or above vmax also give 255, the same code as no
data, and the configs' scale and offset are not applied: the model was trained on exactly these quirks, so they
are kept.
"""

from __future__ import annotations

import numpy as np


def _parse(config: str) -> list[tuple[float, float]]:
    out = []
    for item in config.split(","):
        band, _scale, _offset, vmin, vmax = item.strip().split(":")
        assert int(band) == len(out)
        out.append((float(vmin), float(vmax)))
    return out


# band:scale:offset:min:max, verbatim from scale_int_tif.py's argument defaults
RANGES = {
    "MCD09GA": _parse("0:0.0001:0:-3.86:8338.89, 1:0.0001:0:4.53:8620.19, 2:0.0001:0:15.45:9426.45, "
                      "3:0.0001:0:-17.73:1937.25"),
    "VPD": _parse("0:1:0:0:1.63"),
    "MCD11A1": _parse("0:0.02:0:7500:20000, 1:0.02:0:7500:20000, 2:0.002:0.49:1:255, 3:0.002:0.49:1:255"),
    "MCD15A3H": _parse("0:0.1:0:0:46.78, 1:0.01:0:0:87.14"),
    "ERA5": _parse(
        "0:1:0:0.005193258648563279:297.7777831997186, 1:1:0:-8.035555467392681:9.75282316575327, "
        "2:1:0:-8.884559637656066:7.783394794594883, 3:1:0:0.00013918371392302747:99.94999949848244, "
        "4:1:0:1.9188684693810958e-07:0.03603348170585861, 5:1:0:-14189138.410743382:2327057.3615739984, "
        "6:1:0:0.005005088835593607:291.85923360787075, 7:1:0:1.7814644058506481:102594.00284017395, "
        "8:1:0:1.2825456326856735e-05:0.7495786155920634, 9:1:0:1.2824859849846285e-05:0.7429968109043169, "
        "10:1:0:1.2793656195784068e-05:0.7264044759278405, 11:1:0:1.2811615921890935e-05:0.7151072856451576, "
        "12:1:0:0.005317793384206413:304.2753351358254, 13:1:0:0.005381247822265205:305.6516323146297, "
        "14:1:0:-0.04698579626080153:0.00023615964263532034, 15:1:0:-0.005673370511294594:0.000825804130791964, "
        "16:1:0:4.4439247570826976e-09:0.0005425088177093472, 17:1:0:413.45007501667135:28043026.67241847"),
    "MCD09CMG": _parse("0:1:0:219:334.4, 1:1:0:193:400, 2:1:0:204:356, 3:1:0:202:386.3"),
    "NDVI_EVI": _parse("0:1:0:-0.5:1, 1:1:0:-1:1"),
    "FWI": _parse("0:1:0:0:101, 1:1:0:0:119, 2:1:0:0:694, 3:1:0:0:11, 4:1:0:0:149, 5:1:0:0:33"),
}


def to_uint8(block: np.ndarray, vmin: float, vmax: float, mcd15a3h: bool = False) -> np.ndarray:
    """One band, as `_transform_block_to_uint8` encodes it."""
    arr = np.asarray(block)
    nan_mask = np.isnan(arr) if np.issubdtype(arr.dtype, np.floating) else np.zeros(arr.shape, bool)
    mask_255, mask_neg, mask_pos = arr == 255, arr == -32768, arr == 32768
    out = np.full(arr.shape, 255, np.uint8)
    if mcd15a3h:
        fixed_zero = mask_neg | mask_pos
        valid = ~(fixed_zero | mask_255 | nan_mask)
        out[fixed_zero] = 0
    else:
        valid = ~(mask_neg | mask_pos | mask_255 | nan_mask)
    if not valid.any():
        return out
    rng = float(vmax) - float(vmin)
    vals = arr.astype(np.float32, copy=False)[valid]
    with np.errstate(all="ignore"):
        scaled = (vals - np.float32(vmin)) / np.float32(rng) * np.float32(255.0)
    if not np.all(np.isfinite(scaled)):
        raise ValueError("encoding produced NaN or Inf for valid pixels")
    np.clip(scaled, 0.0, 255.0, out=scaled)
    out[valid] = scaled.astype(np.uint8)
    return out


def encode(modality: str, bands: np.ndarray) -> np.ndarray:
    """(bands, H, W) float values of one modality -> (bands, H, W) uint8 codes."""
    ranges = RANGES[modality]
    assert bands.shape[0] == len(ranges), (modality, bands.shape)
    return np.stack([to_uint8(b, lo, hi, mcd15a3h=modality == "MCD15A3H") for b, (lo, hi) in zip(bands, ranges)])
