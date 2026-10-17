"""Paper-Protokoll: Labels fuer den Tag nach dem Sample-Datum (D+1) je Stichprobe.

Das HDF5-Attribut forecast = 1 und der Abgleich mit den berichteten Zahlen
(p12_label_variants) sprechen dafuer, dass das Paper gegen Feuer an D+1 bewertet.
Hier entstehen dafuer je Cache, fuer alle Splits:

  y_next_patch.npy   Feuer irgendwo im 13x13-Ausschnitt an D+1 (MCD14A1-Rohraster)
  y_next_centre.npy  Feuer in der Mittelzelle an D+1
  fire_next.npy      (N, 13, 13) Feuermaske des Ausschnitts an D+1 (fuer die Pixel-Modelle)

neben X.npy unter derived_data/paper/<cache>_L61/.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.paper.label_variants import HALF, patch, raw_fire


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--caches", nargs="+", default=["cache_A", "cache_G", "cache_H", "cache_Y"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp, raw = Path(cfg["experiment_root"]), Path(cfg["dataset_root"]) / "Entire_Canada_Maps"

    metas = {c: np.load(exp / "derived_data" / "paper" / f"{c}_L61" / "meta.npy") for c in args.caches}
    to_date = lambda m: dt.date(int(m // 10000), int(m // 100 % 100), int(m % 100))  # noqa: E731
    need = {to_date(m) + dt.timedelta(days=1) for meta in metas.values() for m in meta[:, 0]}
    fire = raw_fire(raw, need)
    missing = sorted(d for d in need if d not in fire)
    if missing:
        raise SystemExit(f"ABBRUCH: {len(missing)} Tage fehlen in den Rohrastern, z. B. {missing[:3]}")

    info = {}
    for c, meta in metas.items():
        d = exp / "derived_data" / "paper" / f"{c}_L61"
        yp, yc = np.zeros(len(meta), np.int8), np.zeros(len(meta), np.int8)
        fmap = np.zeros((len(meta), 2 * HALF + 1, 2 * HALF + 1), np.uint8)   # Ausschnitt an D+1, Rand = 0
        for i, (m, r, col) in enumerate(meta):
            f = fire[to_date(m) + dt.timedelta(days=1)]
            yp[i], yc[i] = patch(f, r, col).any(), f[r, col]
            H, W = f.shape
            r0, c0 = r - HALF, col - HALF
            rr0, cc0, rr1, cc1 = max(0, r0), max(0, c0), min(H, r0 + 2 * HALF + 1), min(W, c0 + 2 * HALF + 1)
            fmap[i, rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0] = f[rr0:rr1, cc0:cc1]
        assert (fmap.reshape(len(meta), -1).max(1) == yp).all() and (fmap[:, HALF, HALF] == yc).all()
        np.save(d / "y_next_patch.npy", yp)
        np.save(d / "y_next_centre.npy", yc)
        np.save(d / "fire_next.npy", fmap)
        y = np.load(d / "y.npy").astype(int)
        split = np.load(d / "split.npy")
        info[c] = {s: {"n": int((split == s).sum()),
                       "base_rate_next_patch": float(yp[split == s].mean()),
                       "base_rate_next_centre": float(yc[split == s].mean()),
                       "agreement_with_cache_label": float((yp[split == s] == y[split == s]).mean()),
                       "pos_still_burning_next_day": float(yp[(split == s) & (y == 1)].mean()),
                       "neg_burning_next_day": float(yp[(split == s) & (y == 0)].mean())}
                   for s in ("train", "val", "test")}
        print(f"[next] {c}: " + "  ".join(
            f"{s} Basisrate {v['base_rate_next_patch']:.3f} (Pos. weiter Feuer {v['pos_still_burning_next_day']:.2f}, "
            f"Neg. mit Feuer {v['neg_burning_next_day']:.3f})" for s, v in info[c].items()), flush=True)

    write_manifest(cfg, "p13_next_labels", args.config, {
        "definition": {"y_next_patch": "Feuer im 13x13-Ausschnitt an D+1 (MCD14A1)",
                       "y_next_centre": "Feuer in der Mittelzelle an D+1"},
        "half": HALF, "caches": info,
    })


if __name__ == "__main__":
    main()
