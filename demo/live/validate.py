"""Compare rebuilt inputs with MPCFire's released rasters, code by code, over Canadian land.

    python -m demo.live.validate --raw <MPCFire root> --masks <experiment root>/derived_data/masks \
        --days 2025-07-13 2025-07-15 --mods fire ERA5 VPD [--fwi-from 2025-06-01]

Per band: the share of land pixels with the released code exactly, within one code, and the mean absolute code
difference. --fwi-from starts the FWI recursion from the released codes of that day and compares every --day
after it, so the drift of the recursion shows.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import tarfile
from pathlib import Path

import numpy as np

from . import inputs

BUILDERS = {"fire": ("MCD14A1", inputs.fire), "ERA5": ("ERA5", inputs.era5), "VPD": ("VPD", inputs.vpd)}


class Released:
    """Released MPCFire rasters of one year, read from its per-modality tar once and kept in memory per day."""

    def __init__(self, raw: Path):
        self.raw = Path(raw)
        self._mem: dict[tuple[str, str], np.ndarray] = {}

    def load(self, modality: str, days: set[str]) -> None:
        from rasterio.io import MemoryFile
        need = {d for d in days if (modality, d) not in self._mem}
        for year in sorted({d[:4] for d in need}):
            tar = self.raw / "Entire_Canada_Maps" / modality / f"{modality}__{year}.tar"
            with tarfile.open(tar, "r:") as tf:
                for m in tf:
                    h = re.search(r"(\d{4})_(\d{2})_(\d{2})\.tif", m.name)
                    key = f"{h[1]}-{h[2]}-{h[3]}" if h else None
                    if key in need:
                        with MemoryFile(tf.extractfile(m).read()) as mem, mem.open() as ds:
                            self._mem[(modality, key)] = ds.read()

    def get(self, modality: str, day: str) -> np.ndarray:
        self.load(modality, {day})
        return self._mem[(modality, day)]


def compare(rebuilt: np.ndarray, released: np.ndarray, mask: np.ndarray) -> list[tuple[float, float, float]]:
    out = []
    for a, b in zip(rebuilt, released):
        d = np.abs(a[mask].astype(np.int16) - b[mask].astype(np.int16))
        out.append((float((d == 0).mean()), float((d <= 1).mean()), float(d.mean())))
    return out


def report(name: str, day: str, stats) -> None:
    exact = " ".join(f"{s[0]:.3f}" for s in stats)
    within = " ".join(f"{s[1]:.3f}" for s in stats)
    mae = " ".join(f"{s[2]:.2f}" for s in stats)
    print(f"{name:8s} {day}  exact [{exact}]\n{'':19s} within 1 [{within}]\n{'':19s} mean |diff| [{mae}]", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True, help="MPCFire root (holds Entire_Canada_Maps/)")
    ap.add_argument("--masks", type=Path, required=True, help="folder with land.npy and canada.npy")
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--mods", nargs="*", default=["fire", "ERA5", "VPD"])
    ap.add_argument("--fwi-from", help="start the FWI recursion from the released codes of this day")
    args = ap.parse_args()
    mask = np.load(args.masks / "land.npy") & np.load(args.masks / "canada.npy")
    rel = Released(args.raw)
    for mod in args.mods:
        name, build = BUILDERS[mod]
        rel.load(name, set(args.days))
        for day in args.days:
            got, want = build(day), rel.get(name, day)
            report(mod, day, compare(got, want, mask))
            if mod == "fire":                        # fire pixels are too rare for the shares above
                g, w = got[0] == 1, want[0] == 1
                print(f"{'':19s} fire px released {int(w.sum())}, rebuilt {int(g.sum())}, both {int((g & w).sum())}", flush=True)
    if args.fwi_from:
        targets = sorted(args.days)
        rel.load("FWI", set(targets) | {args.fwi_from})
        state = inputs.fwi_state_from_release(rel.get("FWI", args.fwi_from))
        day = args.fwi_from
        while day < targets[-1]:
            day = inputs.next_day(day)
            state, comp = inputs.fwi_step(state, day)
            if day in targets:
                report("FWI", day, compare(inputs.fwi_encode(comp), rel.get("FWI", day), mask))


if __name__ == "__main__":
    main()
