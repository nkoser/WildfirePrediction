"""Paper-Protokoll, Schritt 4b: FWI-HNM-Gewichte direkt aus den Rohrastern, ganzes Jahr.

hnm.py kalibriert auf unserem Wuerfel (1.4.-30.9.). Hier kommen FWI und die
MODIS-Feuermaske direkt aus Entire_Canada_Maps/{FWI,MCD14A1}/*.tar, fuer alle
Tage der Trainingsjahre, damit Oktober/November und das Fruehjahr mitzaehlen.

Zusaetzlich: wie oft jede Komponente am uint8-Rand liegt (0 bzw. 255), getrennt
nach Feuer- und Nicht-Feuer-Pixeltagen. Die Raster tragen keine Skalierung; ein
Wert 255 ist abgeschnitten und senkt d_i der schwerschwaenzigen Komponenten.

Lesarten (Zeitraum x Feuer-Definition), je mit d_i auf c_i (wie im Paper) und
auf x~ = min(c / P99, 1):
  Zeitraum  ganzes Jahr | 1.5.-31.10. | 1.4.-30.9. (wie unser Wuerfel) | Tag 121-305 (wie die Caches)
  Feuer     am Pixel | irgendwo im 13x13-Ausschnitt
"""

from __future__ import annotations

import argparse
import datetime as dt
import tarfile
import time
from pathlib import Path

import numpy as np
from rasterio.io import MemoryFile
from scipy.ndimage import maximum_filter

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.paper.hnm import NAMES, calibrate

PAPER = {"FFMC": 0.152, "DMC": 0.184}
PERIODS = {
    "all_year": (1, 366),
    "may_oct": (121, 304),                       # 1.5. - 31.10. (ohne Schaltjahr-Feinheit)
    "apr_sep": (91, 273),                        # 1.4. - 30.9., Saison unseres Wuerfels
    "cache_doy": (121, 305),                     # Spanne der Kalendertage in den Caches
}


def members(tar: Path) -> dict[str, tarfile.TarInfo]:
    with tarfile.open(tar, "r:") as tf:
        return {Path(m.name).name: m for m in tf if m.isfile()}


def read(tf: tarfile.TarFile, m: tarfile.TarInfo) -> np.ndarray:
    with MemoryFile(tf.extractfile(m).read()) as mem, mem.open() as src:
        return src.read()                                     # (bands, H, W)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--years", nargs=2, type=int, default=[2000, 2019])
    ap.add_argument("--day-step", type=int, default=2)
    ap.add_argument("--pixel-frac", type=float, default=0.1)
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp, root = Path(cfg["experiment_root"]), Path(cfg["dataset_root"]) / "Entire_Canada_Maps"
    rng = np.random.default_rng(int(cfg["seed"]))

    mdir = exp / "derived_data" / "masks"
    pool = np.load(mdir / "land.npy") & np.load(mdir / "canada.npy")
    pr, pc = np.nonzero(pool)
    xs, fpx, fpa, doys = [], [], [], []
    t0 = time.time()
    for year in range(args.years[0], args.years[1] + 1):
        ft, rt = root / "FWI" / f"FWI__{year}.tar", root / "MCD14A1" / f"MCD14A1__{year}.tar"
        mf, mr = members(ft), members(rt)
        days = sorted(set(mf) & set(mr))[::args.day_step]
        with tarfile.open(ft, "r:") as tff, tarfile.open(rt, "r:") as tfr:
            for name in days:
                fwi = read(tff, mf[name])                     # (6, H, W) uint8
                fire = read(tfr, mr[name])[0] > 0             # (H, W)
                if fwi.shape[0] != 6 or fwi.shape[1:] != pool.shape or fire.shape != pool.shape:
                    raise SystemExit(f"ABBRUCH: unerwartete Form {fwi.shape} / {fire.shape} in {name}")
                m = rng.random(pr.size) < args.pixel_frac
                r, c = pr[m], pc[m]
                xs.append(fwi[:, r, c].T)
                fpx.append(fire[r, c])
                fpa.append(maximum_filter(fire, size=13)[r, c])
                d = dt.date(*map(int, name[:10].split("_")))
                doys.append(np.full(r.size, d.timetuple().tm_yday, np.int16))
        print(f"[hnmraw] {year}: {len(days)} Tage ({time.time() - t0:.0f} s)", flush=True)

    X = np.concatenate(xs)
    fire_px, fire_patch, doy = np.concatenate(fpx), np.concatenate(fpa), np.concatenate(doys)
    print(f"[hnmraw] Pool: {X.shape[0]:,} Pixeltage, Feuer am Pixel {fire_px.sum():,}, "
          f"im Ausschnitt {fire_patch.sum():,}", flush=True)

    # ---- Rand der uint8-Skala, je Komponente und Feuer ja/nein (ganzes Jahr, Feuer am Pixel)
    edges = {}
    for i, n in enumerate(NAMES):
        edges[n] = {lab: {"share_255": float((X[sel, i] == 255).mean()), "share_0": float((X[sel, i] == 0).mean()),
                          "p99": float(np.percentile(X[sel, i], 99))}
                    for lab, sel in (("fire", fire_px), ("no_fire", ~fire_px))}
    print("\n[hnmraw] Anteil am Rand der Skala (Feuer am Pixel, ganzes Jahr)")
    print("    Komponente   255 Feuer   255 kein   0 Feuer   0 kein")
    for n, e in edges.items():
        print(f"    {n:5s}        {100 * e['fire']['share_255']:6.2f} %  {100 * e['no_fire']['share_255']:6.2f} %"
              f"  {100 * e['fire']['share_0']:6.2f} %  {100 * e['no_fire']['share_0']:6.2f} %")

    # ---- Gewichte je Lesart
    weights = {}
    for pname, (lo, hi) in PERIODS.items():
        in_p = (doy >= lo) & (doy <= hi)
        for fname, fv in (("pixel", fire_px), ("patch", fire_patch)):
            for clip, sfx in ((False, ""), (True, "/clipped")):
                p99, d, w = calibrate(X[in_p].astype(np.float64), fv[in_p], clip=clip)
                weights[f"{pname}/{fname}{sfx}"] = {"w": w.tolist(), "d": d.tolist(), "p99": p99.tolist(),
                                                    "n": int(in_p.sum()), "n_fire": int(fv[in_p].sum())}
    print(f"\n[hnmraw] Gewichte (Paper: FFMC {PAPER['FFMC']}, DMC {PAPER['DMC']})")
    print("    " + " " * 28 + "  ".join(f"{n:>5s}" for n in NAMES) + "   d_DMC/d_FFMC")
    for k, v in weights.items():
        ratio = v["d"][1] / v["d"][0] if v["d"][0] > 0 else float("nan")
        print(f"    {k:28s}" + "  ".join(f"{x:5.3f}" for x in v["w"]) + f"   {ratio:5.2f}")
    print(f"    {'Paper':28s}{PAPER['FFMC']:5.3f}  {PAPER['DMC']:5.3f}" + " " * 29
          + f"   {PAPER['DMC'] / PAPER['FFMC']:5.2f}")

    write_manifest(cfg, "p3b_hnm_raw", args.config, {
        "source": "Entire_Canada_Maps/FWI und MCD14A1 (Rohraster, uint8, Gitter 0,0449 Grad)",
        "years": args.years, "day_step": args.day_step, "pixel_frac": args.pixel_frac,
        "pool": "Land & Kanada", "n": int(X.shape[0]), "n_fire_pixel": int(fire_px.sum()),
        "periods_doy": PERIODS, "uint8_edges": edges, "weights": weights, "paper": PAPER,
        "definition": "d_i = max(0, (mu_fire - mu_nofire) / sd_nofire); w_i = d_i / sum d",
    })


if __name__ == "__main__":
    main()
