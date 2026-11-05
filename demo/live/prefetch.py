"""Download (into LIVE_CACHE) what the builders need for a range of days, several days at a time.

    python -m demo.live.prefetch --what era5 fire MCD11A1 --start 2026-07-27 --end 2026-10-30 [--parallel 3]
"""

from __future__ import annotations

import argparse
import datetime as dt
import time
from concurrent.futures import ThreadPoolExecutor

from . import inputs, modis


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--what", nargs="+", required=True, help="era5, fire, or a MODIS product name")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--parallel", type=int, default=3, help="days downloaded at the same time")
    args = ap.parse_args()
    d0, d1 = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    days = [(d0 + dt.timedelta(i)).isoformat() for i in range((d1 - d0).days + 1)]
    for what in args.what:
        fn = {"era5": inputs.era5_raw, "fire": inputs.fire_1km}.get(what) or \
             (lambda day, p=modis.PRODUCTS[what]: modis.daily_1km(p, day))
        t0 = time.time()

        def one(day):
            fn(day)
            return day

        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            for i, day in enumerate(ex.map(one, days), 1):
                print(f"[prefetch] {what} {day} ({i}/{len(days)}, {time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
