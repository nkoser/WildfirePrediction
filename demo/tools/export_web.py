"""Turn risk-map npz files into the page's data manifest (JSON with every layer embedded as a data URI).

Input: files written by wildfire/paper/risk_map.py (or make_placeholder.py), keys
    F_<tau>      float32 (H, W), pixel probability of a first fire within tau days, NaN outside the mask
    fire_window  bool (H, W), MODIS fire in the 30 input days L-29 .. L
    first_fire   int (H, W), first MODIS fire day after L (1 .. max tau), 0 = none
    taus, date   the lead times and the forecast date L
ign_<tau>, fire_today and the rest are ignored. A file with placeholder=True marks the whole manifest as
placeholder.

Output (schema "wildfire-demo/1"), the contract the page reads and the future API serves:
    grid         CRS, size, outer top-left corner and pixel size of the raster
    quant        how risk layers are stored: uint8, log10 between vmin and vmax, 0 = no data
    display      color-scale range (vmin 1e-4 as in the paper's figure; vmax = 99.9th percentile of F at
                 the largest tau, maximum over forecasts, the figure's rule applied to all dates at once)
    models       id, label, detail
    forecasts    per model and date: taus, risk[tau] (PNG), observed (PNG; bit 7 = fire in the input
                 window, bits 0-6 = first fire day after the date, 0 = none within observed_days)

    python -I demo/tools/export_web.py <npz files> --model-id contagion-sts --model-label Contagion-STS \
        --model-detail "ensemble of 3 seeds, raw probabilities" --out <dir>/data.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import QUANT, data_uri, encode_png, load_grid, quantize_log  # noqa: E402

NOTES = [
    "Forecast dates fall in the held-out test years 2023 to 2025. The model was trained on 2000 to 2019.",
    "Probabilities are the model's raw outputs, without calibration.",
    "Fires are MODIS active-fire detections (MCD14A1) on the same grid of about 5 km.",
]


def nice_up(x: float) -> float:
    """Round up to two significant digits."""
    e = math.floor(math.log10(x)) - 1
    return math.ceil(x / 10 ** e) * 10 ** e


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-id", default="contagion-sts")
    ap.add_argument("--model-label", default="Contagion-STS")
    ap.add_argument("--model-detail", default="ensemble of 3 seeds, raw probabilities")
    ap.add_argument("--note", action="append", help="footnote shown on the page (repeatable; replaces the defaults)")
    ap.add_argument("--vmin", type=float, default=1e-4, help="lower end of the color scale")
    args = ap.parse_args()
    g = load_grid()
    shape = (g["height"], g["width"])

    forecasts, placeholder, tops = [], False, []
    for path in sorted(args.npz):
        z = np.load(path, allow_pickle=False)
        taus = [int(t) for t in z["taus"]]
        date = str(z["date"])
        placeholder |= bool(z["placeholder"]) if "placeholder" in z.files else False
        risk = {}
        for t in taus:
            F = z[f"F_{t}"]
            if F.shape != shape:
                raise SystemExit(f"{path.name}: F_{t} has shape {F.shape}, grid.json says {shape}")
            risk[str(t)] = data_uri(encode_png(quantize_log(F)))
        tops.append(float(np.nanpercentile(z[f"F_{max(taus)}"], 99.9)))
        first = z["first_fire"].astype(np.int64)
        if first.max() > 127:
            raise SystemExit(f"{path.name}: first_fire reaches {first.max()}, the encoding holds 0..127")
        observed = ((z["fire_window"].astype(np.uint8) << 7) | first.astype(np.uint8)).astype(np.uint8)
        forecasts.append({"model": args.model_id, "date": date, "taus": taus, "risk": risk,
                          "observed": data_uri(encode_png(observed)), "observed_days": max(taus)})
        print(f"[export] {path.name}: {date}, taus {taus}", flush=True)

    model = ({"id": "placeholder", "label": "Placeholder values",
              "detail": "stand-in risk from past fires, not model output"} if placeholder else
             {"id": args.model_id, "label": args.model_label, "detail": args.model_detail})
    for f in forecasts:
        f["model"] = model["id"]
    manifest = {
        "schema": "wildfire-demo/1",
        "placeholder": placeholder,
        "grid": {k: g[k] for k in ("crs", "width", "height", "lon0", "lat0", "res")},
        "quant": QUANT,
        "display": {"scale": "log10", "vmin": args.vmin, "vmax": nice_up(max(tops))},
        "input_days": 30,
        "models": [model],
        "forecasts": forecasts,
        "notes": args.note or NOTES,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, separators=(",", ":"))
    args.out.write_text(text)
    print(f"[export] {args.out}: {len(forecasts)} forecasts, {len(text) / 2**20:.1f} MB, "
          f"color scale {args.vmin:g} .. {manifest['display']['vmax']:g}", flush=True)


if __name__ == "__main__":
    main()
