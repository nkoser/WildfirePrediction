"""Kaplan-Meier-Stichtage (km_meta.npz) fuer die Survival-Stichtage eines Kandidaten-Splits.

surv_eval / horizon_landmarks / state_eval schaetzen die KM-Basis auf derived_data/<surv-dir>/km_meta.npz. Fuer
surv_eval_test stammt sie aus den Trainingsjahren 2000-2019. Fuer surv_eval_ys / surv_eval_bs fehlt die Datei, und
surv_eval.build wuerde sie ohne Split-Maske aus 2000-2019 ziehen (Testjahre bzw. Testbloecke inklusive). Hier
entsteht sie wie in split_landmarks.build_surv aus dem train-Teil des Splits (Allowed), aber ohne X (KM braucht nur
Ereignis und Zeit), gleicher Zufallsstrom wie surv_eval.build fuer km (seed_off 1), n wie dort (200.000).

Ausgabe NEU: derived_data/<out-surv>/km_meta.npz (bricht ab, wenn sie existiert); Manifest p50_split_km_<name><tag>.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.data.dataset import CubeReader
from wildfire.paper.pixel_landmarks import pixel_events
from wildfire.paper.split_audit import load_json
from wildfire.paper.split_landmarks import Allowed
from wildfire.paper.surv_eval import abs_day, fire_in_window, raw_fire_keys, sample
from wildfire.paper.surv_prepare import HALF


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", choices=["year_strat", "block_strat"], required=True)
    ap.add_argument("--split-tag", default="")
    ap.add_argument("--unit", type=int, default=64)
    ap.add_argument("--gap", type=int, default=2 * HALF)
    ap.add_argument("--n-km", type=int, default=200_000)
    ap.add_argument("--out-surv", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    out = exp / "derived_data" / args.out_surv
    if not (out / "eval_meta.npz").exists():
        raise SystemExit(f"ABBRUCH: {out}/eval_meta.npz fehlt - erst split_landmarks")
    if (out / "km_meta.npz").exists():
        raise SystemExit(f"ABBRUCH: {out}/km_meta.npz existiert schon")

    base = load_json(Path(cfg["dataset_root"]) / "cache_G" / "samples_variant_G.json")
    split = np.load(exp / "derived_data" / f"splits_v2{args.split_tag}" / args.name / "split.npy")
    cube = CubeReader(exp)
    year_of = np.array([int(d[:4]) for d in cube.dates])
    ok = Allowed(args.name, split, base, year_of, cube.H, cube.W, args.unit, args.gap)
    years = ({y for y, s in ok.part_of_year.items() if s == "train"} if args.name == "year_strat"
             else set(range(2000, 2026)))
    keys, last_raw = raw_fire_keys(Path(cfg["dataset_root"]))
    land = np.load(exp / "derived_data" / "masks" / "land.npy") & np.load(exp / "derived_data" / "masks" / "canada.npy")
    rng = np.random.default_rng(int(cfg["seed"]) + 31 + 1000 * 1)
    n = args.n_km
    rr, cc, Lc = [], [], []
    while len(rr) < n:
        r, c, L = sample(cube, land, rng, years, n, last_raw)
        keep = ok("train", L, r, c)
        rr += r[keep].tolist(); cc += c[keep].tolist(); Lc += L[keep].tolist()
        print(f"[skm] km: {len(rr):,} / {n:,} Stichtage", flush=True)
    rr, cc, Lc = (np.array(a[:n], np.int64) for a in (rr, cc, Lc))
    La = np.array([abs_day(dt.date.fromisoformat(cube.dates[int(x)])) for x in Lc])
    ev, t = pixel_events(keys, La, rr, cc, cube.H, cube.W)
    pe, pt = ev.reshape(n, -1).any(1), t.reshape(n, -1).min(1)
    fw_patch, fw_centre = fire_in_window(keys, La, rr, cc, cube.H, cube.W)
    doy = np.array([dt.date.fromisoformat(cube.dates[int(x)]).timetuple().tm_yday for x in Lc], np.float32)
    np.savez(out / "km_meta.npz", row=rr, col=cc, L_cube=Lc, L_abs=La, doy=doy,
             ev_centre=ev[:, HALF, HALF], t_centre=t[:, HALF, HALF], ev_patch=pe, t_patch=pt,
             fire_win_patch=fw_patch, fire_win_centre=fw_centre)
    yrs = np.bincount(year_of[Lc] - 2000, minlength=26)
    info = {"n": int(n), "event_patch": float(pe.mean()), "event_centre": float(ev[:, HALF, HALF].mean()),
            "years": {int(2000 + i): int(v) for i, v in enumerate(yrs) if v}}
    print(f"[skm] {info}", flush=True)
    write_manifest(cfg, f"p50_split_km_{args.name}{args.tag}", args.config,
                   {"args": vars(args), "info": info, "output": str(out / "km_meta.npz")})


if __name__ == "__main__":
    main()
