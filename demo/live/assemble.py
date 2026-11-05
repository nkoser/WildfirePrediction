"""Write daily inputs as an experiment root that Niklas's CubeReader and wildfire/paper/risk_map.py read unchanged:
data_manifests/d4_cube.json, derived_data/cube/dyn_<year>.mmap and stat_<year>.npy, derived_data/events/
fire_index.npz (its `dates`), and the main root's masks.

Each modality of each day comes from a rebuilt builder (demo/live/inputs.py) or, for the modalities left out
with --released, from an existing cube of MPCFire's released files; mixing the two shows what each rebuilt
modality changes in the model's output.

    python -m demo.live.assemble --root <new root> --main <main experiment root> --start 2025-06-16 \
        --end 2025-08-14 --rebuild fire ERA5 VPD FWI --released-from <root with a released cube> \
        [--fwi-from 2025-06-01] [--static-from <stat_YYYY.npy>]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path

import numpy as np

from . import inputs

# the cube's dynamic channels, in its order (wildfire/data/cube.py channel_layout over channel_dict.CHANNELS)
DYNAMIC = [("MCD14A1", 1), ("ERA5", 18), ("FWI", 6), ("VPD", 1), ("NDVI_EVI", 2), ("MCD15A3H", 2),
           ("MCD09GA", 4), ("MCD11A1", 4), ("MCD09CMG", 4)]
N_DYN, N_STAT = 42, 13
BUILDERS = {"fire": "MCD14A1", "ERA5": "ERA5", "VPD": "VPD", "FWI": "FWI"}


def slices() -> dict[str, slice]:
    out, i = {}, 0
    for mod, n in DYNAMIC:
        out[mod] = slice(i, i + n)
        i += n
    assert i == N_DYN
    return out


def days_between(a: str, b: str) -> list[str]:
    d0, d1 = dt.date.fromisoformat(a), dt.date.fromisoformat(b)
    return [(d0 + dt.timedelta(i)).isoformat() for i in range((d1 - d0).days + 1)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--main", type=Path, required=True, help="main experiment root (masks, mask manifest)")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--rebuild", nargs="*", default=list(BUILDERS), choices=list(BUILDERS))
    ap.add_argument("--released-from", type=Path, nargs="*", default=[],
                    help="experiment roots whose cubes give the modalities that are not rebuilt")
    ap.add_argument("--fwi-from", help="day whose released FWI codes start the FWI recursion (default: --start)")
    ap.add_argument("--static-from", type=Path, help="stat_<year>.npy to copy (default: the newest in --released-from)")
    args = ap.parse_args()

    days = days_between(args.start, args.end)
    sl = slices()
    rebuilt = {BUILDERS[m] for m in args.rebuild}
    readers = []
    if set(sl) - rebuilt:
        from wildfire.data.dataset import CubeReader
        readers = [CubeReader(r) for r in args.released_from]

    def released(day: str, mod: str) -> np.ndarray:
        for rd in readers:
            idx = {d: i for i, d in enumerate(rd.dates)}.get(day)
            if idx is not None and idx in rd._pos:
                year, li = rd._pos[idx]
                return np.moveaxis(np.asarray(rd._arrays(year)[0][li, :, :, sl[mod]]), -1, 0)
        raise SystemExit(f"{mod} of {day} is neither rebuilt nor in a released cube")

    root = args.root
    (root / "data_manifests").mkdir(parents=True, exist_ok=True)
    (root / "derived_data" / "events").mkdir(parents=True, exist_ok=True)
    (root / "derived_data" / "cube").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.main / "data_manifests" / "d0_masks.json", root / "data_manifests" / "d0_masks.json")
    if not (root / "derived_data" / "masks").exists():
        (root / "derived_data" / "masks").symlink_to(args.main / "derived_data" / "masks")
    np.savez(root / "derived_data" / "events" / "fire_index.npz", dates=np.array(days))

    H, W = inputs.RELEASE.height, inputs.RELEASE.width
    per_year = {}
    fwi_state = None
    if "FWI" in rebuilt:
        start = args.fwi_from or args.start
        if not readers:
            from wildfire.data.dataset import CubeReader
            readers = [CubeReader(r) for r in args.released_from]
        fwi_state = inputs.fwi_state_from_release(released(start, "FWI"))
        fwi_day = start
    for year in sorted({d[:4] for d in days}):
        ydays = [d for d in days if d[:4] == year]
        path = root / "derived_data" / "cube" / f"dyn_{year}.mmap"
        dyn = np.memmap(path, dtype=np.uint8, mode="w+", shape=(len(ydays), H, W, N_DYN))
        for t, day in enumerate(ydays):
            for mod, s in sl.items():
                if mod == "FWI" and fwi_state is not None:
                    comp = None
                    while fwi_day < day:                       # the recursion runs every day up to this one
                        fwi_day = inputs.next_day(fwi_day)
                        fwi_state, comp = inputs.fwi_step(fwi_state, fwi_day)
                    arr = inputs.fwi_encode(comp) if comp is not None else released(day, "FWI")
                elif mod in rebuilt:
                    arr = {"MCD14A1": inputs.fire, "ERA5": inputs.era5, "VPD": inputs.vpd}[mod](day)
                else:
                    arr = released(day, mod)
                dyn[t, :, :, s] = np.moveaxis(arr, 0, -1)
            print(f"[assemble] {day}: " + ", ".join(m + ("*" if m in rebuilt else "") for m in sl), flush=True)
        dyn.flush()
        del dyn
        stat_src = args.static_from or sorted(p for r in args.released_from for p in (r / "derived_data" / "cube").glob("stat_*.npy"))[-1]
        stat_dst = root / "derived_data" / "cube" / f"stat_{year}.npy"
        shutil.copyfile(stat_src, stat_dst)
        per_year[year] = {"days": ydays, "T": len(ydays), "dyn_file": str(path), "stat_file": str(stat_dst),
                          "dyn_shape": [len(ydays), H, W, N_DYN], "dyn_bytes": len(ydays) * H * W * N_DYN}
    manifest = {"layout": "(T, H, W, C) uint8, kanalletzt", "grid": {"height": H, "width": W},
                "n_dynamic_channels": N_DYN, "n_static_channels": N_STAT, "per_year": per_year,
                "rebuilt": sorted(rebuilt), "released_from": [str(r) for r in args.released_from],
                "note": "demo/live/assemble.py; modalities in `rebuilt` come from Earth Engine through demo/live/inputs.py"}
    (root / "data_manifests" / "d4_cube.json").write_text(json.dumps(manifest, indent=1))
    print(f"[assemble] {root}: {len(days)} days {days[0]} .. {days[-1]}, rebuilt {sorted(rebuilt)}", flush=True)


if __name__ == "__main__":
    main()
