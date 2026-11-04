"""Turn risk-map npz files into the page's data directory: one manifest plus one PNG per layer.

Input: files written by wildfire/paper/risk_map.py (or make_placeholder.py), keys
    F_<tau>      float32 (H, W), pixel probability of a first fire within tau days, NaN outside the mask
    fire_window  bool (H, W), MODIS fire in the 30 input days L-29 .. L
    first_fire   int (H, W), first MODIS fire day after L (1 .. max tau), 0 = none
    taus, date   the lead times and the forecast date L
ign_<tau>, fire_today and the rest are ignored. A file with placeholder=True marks the whole manifest as
placeholder.

Output (schema "wildfire-demo/2"), the contract the page reads and a hosted model would serve:
    <out>/manifest.json
    <out>/<model>/<date>/risk-<tau>.png   F(tau), 8-bit grayscale, quantized as `quant` says
    <out>/<model>/<date>/observed.png     bit 7 = fire in the input window, bits 0-6 = first fire day after
                                          the date (0 = none within observed_days)
Paths in the manifest are relative to the manifest's own URL, so the directory can be served from anywhere.
Manifest fields:
    grid         CRS, size, outer top-left corner and pixel size of the raster
    quant        how risk layers are stored: uint8, log10 between vmin and vmax, 0 = no data
    display      color-scale range (vmin 1e-4 as in the paper's figure; vmax = 99.9th percentile of F at
                 the largest tau, maximum over forecasts, the figure's rule applied to all dates at once)
    models       id, label, detail
    forecasts    per model and date: taus, risk[tau] (path), observed (path), observed_days
    default      date and lead time the page opens on
    notes        footnotes shown on the page

    python -I demo/tools/export_web.py <npz files> --model-id contagion-sts --model-label Contagion-STS \
        --model-detail "ensemble of 3 seeds, raw probabilities" --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import QUANT, encode_png, load_grid, quantize_log  # noqa: E402

SCHEMA = "wildfire-demo/2"
NOTES = [
    "Forecast dates fall in the held-out test years 2023 to 2025. The model was trained on 2000 to 2019.",
    "Probabilities are the model's raw outputs, without calibration.",
    "Fires are MODIS active-fire detections (MCD14A1) on the same grid of about 5 km.",
]


def nice_up(x: float) -> float:
    """Round up to two significant digits."""
    e = math.floor(math.log10(x)) - 1
    return math.ceil(x / 10 ** e) * 10 ** e


def prepare_out(out: Path, overwrite: bool) -> None:
    if out.exists() and any(out.iterdir()):
        if not (overwrite and (out / "manifest.json").is_file()):
            raise SystemExit(f"{out} is not empty; pass --overwrite to replace an earlier export "
                             "(only a directory holding a manifest.json is removed)")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True, help="data directory to write (manifest.json and layers)")
    ap.add_argument("--overwrite", action="store_true", help="replace an earlier export in --out")
    ap.add_argument("--model-id", default="contagion-sts")
    ap.add_argument("--model-label", default="Contagion-STS")
    ap.add_argument("--model-detail", default="ensemble of 3 seeds, raw probabilities")
    ap.add_argument("--note", action="append", help="footnote shown on the page (repeatable; replaces the defaults)")
    ap.add_argument("--vmin", type=float, default=1e-4, help="lower end of the color scale")
    ap.add_argument("--taus", help="comma-separated subset of lead times to export (default: all in the files)")
    ap.add_argument("--dates", nargs="+", help="subset of forecast dates to export (default: all files)")
    ap.add_argument("--default-date", default="2023-06-01", help="date the page opens on (the draft figure's)")
    ap.add_argument("--default-tau", type=int, default=7, help="lead time the page opens on (the draft figure's)")
    args = ap.parse_args()
    keep_taus = [int(t) for t in args.taus.split(",")] if args.taus else None
    g = load_grid()
    shape = (g["height"], g["width"])
    prepare_out(args.out, args.overwrite)

    def is_placeholder(p: Path) -> bool:
        z = np.load(p, allow_pickle=False)
        return "placeholder" in z.files and bool(z["placeholder"])

    placeholder = any(is_placeholder(p) for p in args.npz)   # the model id names the layer folders
    model = ({"id": "placeholder", "label": "Placeholder values",
              "detail": "stand-in risk from past fires, not model output"} if placeholder else
             {"id": args.model_id, "label": args.model_label, "detail": args.model_detail})

    forecasts, tops, n_files, n_bytes = [], [], 0, 0

    def write(rel: str, png: bytes) -> str:
        nonlocal n_files, n_bytes
        path = args.out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png)
        n_files += 1
        n_bytes += len(png)
        return rel

    for path in sorted(args.npz):
        z = np.load(path, allow_pickle=False)
        taus = [int(t) for t in z["taus"]]
        date = str(z["date"])
        if args.dates and date not in args.dates:
            continue
        if keep_taus:
            missing = sorted(set(keep_taus) - set(taus))
            if missing:
                raise SystemExit(f"{path.name}: lead times {missing} are not in the file ({taus})")
            taus = keep_taus
        folder = f"{model['id']}/{date}"
        risk = {}
        for t in taus:
            F = z[f"F_{t}"]
            if F.shape != shape:
                raise SystemExit(f"{path.name}: F_{t} has shape {F.shape}, grid.json says {shape}")
            risk[str(t)] = write(f"{folder}/risk-{t}.png", encode_png(quantize_log(F)))
        tops.append(float(np.nanpercentile(z[f"F_{max(taus)}"], 99.9)))
        first = z["first_fire"].astype(np.int64)
        first[first > max(taus)] = 0                      # observed window = the exported lead times
        if first.max() > 127:
            raise SystemExit(f"{path.name}: first_fire reaches {first.max()}, the encoding holds 0..127")
        observed = ((z["fire_window"].astype(np.uint8) << 7) | first.astype(np.uint8)).astype(np.uint8)
        forecasts.append({"model": model["id"], "date": date, "taus": taus, "risk": risk,
                          "observed": write(f"{folder}/observed.png", encode_png(observed)),
                          "observed_days": max(taus)})
        print(f"[export] {path.name}: {date}, {len(taus)} lead times ({taus[0]} .. {taus[-1]})", flush=True)

    if not forecasts:
        raise SystemExit("no forecast matched --dates")
    manifest = {
        "schema": SCHEMA,
        "placeholder": placeholder,
        "grid": {k: g[k] for k in ("crs", "width", "height", "lon0", "lat0", "res")},
        "quant": QUANT,
        "display": {"scale": "log10", "vmin": args.vmin, "vmax": nice_up(max(tops))},
        "input_days": 30,
        "default": {"date": args.default_date, "tau": args.default_tau},
        "models": [model],
        "forecasts": forecasts,
        "notes": args.note or NOTES,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[export] {args.out}: {len(forecasts)} forecasts, {n_files} layers, {n_bytes / 2**20:.1f} MB, "
          f"color scale {args.vmin:g} .. {manifest['display']['vmax']:g}", flush=True)


if __name__ == "__main__":
    main()
