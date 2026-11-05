"""Add to risk-map files the days each pixel could not be observed after the forecast date (cloud or no data in
Terra's MOD14A1), for the page's "cloud-covered, fire unknown" class.

    python -m demo.live.add_unobserved <npz files>

Adds `unobserved_bits` (uint32, bit k-1 set when day k after the date was not observed) and `unobserved_days` (how
many days after the date have MOD14A1; later days come from FIRMS, which does not report clouds).
"""
import datetime as dt
import sys

import numpy as np

from . import inputs


def main() -> None:
    for f in sys.argv[1:]:
        z = dict(np.load(f, allow_pickle=False))
        date = str(z["date"])
        n = int(z.get("observed_days", max(int(t) for t in z["taus"])))
        bits = np.zeros(z["first_fire"].shape, np.uint32)
        known = 0
        for k in range(1, min(n, 30) + 1):
            day = (dt.date.fromisoformat(date) + dt.timedelta(days=k)).isoformat()
            if not inputs.has_image("MODIS/061/MOD14A1", day):
                break
            bits |= (inputs.unobserved(day).astype(np.uint32) << np.uint32(k - 1))
            known = k
        z["unobserved_bits"], z["unobserved_days"] = bits, np.int64(known)
        np.savez_compressed(f, **z)
        print(f"[unobserved] {f}: {known} days with cloud information", flush=True)


if __name__ == "__main__":
    main()
