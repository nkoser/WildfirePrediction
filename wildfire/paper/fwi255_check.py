"""Was bedeutet der Wert 255 in den FWI-Kanaelen des Wuerfels: Kappung extremer Werte oder NoData?

Die Rohraster sind uint8 ohne scale/offset (audit/channel_dict.py). In extreme_eval liegen FWI und ISI
in jedem Jahr bei rund 4 % der Zelltage auf 255, unabhaengig von der Saisonschwere. Je Jahr werden
alle Saisontage (jeder --step. Tag) auf dem ganzen Land-Raster gelesen:

  zellen   Anteil der Landzellen, die nie / manchmal / (fast) immer 255 zeigen
  begleit  Mittel von FFMC, DMC, DC, t2m_max, snow_cover, precip an Zelltagen mit 255 gegen < 255
           (Kappung: heiss, trocken, hoher FFMC; NoData: unauffaellig oder Schnee/Wasser)
  monat    Anteil 255 je Monat
  nachbar  Anteil 255 bei FWI-Werten knapp darunter (240-254): Kappung haeuft sich am Rand
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.data.dataset import CubeReader

CH = {"FFMC": 19, "DMC": 20, "DC": 21, "ISI": 22, "BUI": 23, "FWI": 24, "t2m_max": 13, "snow_cover": 4, "precip": 5}
TEST = ("FWI", "ISI", "DC", "BUI")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--years", nargs="+", default=["2010", "2016", "2021", "2023"])
    ap.add_argument("--step", type=int, default=5)
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    cube = CubeReader(exp)
    land = np.load(exp / "derived_data" / "masks" / "land.npy") & np.load(exp / "derived_data" / "masks" / "canada.npy")
    lr, lc = np.nonzero(land)
    names, chs = list(CH), list(CH.values())
    res = {}
    for y in args.years:
        dyn, _ = cube._arrays(y)
        days = cube.per_year[y]["days"]
        days = days if isinstance(days, list) else ast.literal_eval(days)
        idx = list(range(0, dyn.shape[0], args.step))
        cnt255 = {c: np.zeros(lr.size, np.int32) for c in TEST}
        s_hit = {c: np.zeros(len(chs)) for c in TEST}
        s_no = {c: np.zeros(len(chs)) for c in TEST}
        n_hit = {c: 0 for c in TEST}
        n_no = {c: 0 for c in TEST}
        month = {c: {} for c in TEST}
        near = {c: [0, 0] for c in TEST}                         # Zelltage 240-254, Zelltage 255
        for d in idx:
            v = np.asarray(dyn[d][lr, lc][:, chs]).astype(np.float64)
            mo = days[d][5:7]
            for c in TEST:
                j = names.index(c)
                hit = v[:, j] == 255
                cnt255[c] += hit
                s_hit[c] += v[hit].sum(0); n_hit[c] += int(hit.sum())
                s_no[c] += v[~hit].sum(0); n_no[c] += int((~hit).sum())
                a = month[c].setdefault(mo, [0, 0]); a[0] += int(hit.sum()); a[1] += hit.size
                near[c][0] += int(((v[:, j] >= 240) & (v[:, j] < 255)).sum()); near[c][1] += int(hit.sum())
        out = {}
        for c in TEST:
            fr = cnt255[c] / len(idx)
            out[c] = {
                "share_cell_days_255": n_hit[c] / max(n_hit[c] + n_no[c], 1),
                "cells_never": float((fr == 0).mean()), "cells_sometimes": float(((fr > 0) & (fr < 0.9)).mean()),
                "cells_ge_90pct": float((fr >= 0.9).mean()), "cells_always": float((fr == 1).mean()),
                "mean_if_255": {n: s_hit[c][k] / max(n_hit[c], 1) for k, n in enumerate(names)},
                "mean_if_not_255": {n: s_no[c][k] / max(n_no[c], 1) for k, n in enumerate(names)},
                "share_255_by_month": {m: a[0] / max(a[1], 1) for m, a in sorted(month[c].items())},
                "cell_days_240_254": near[c][0], "cell_days_255": near[c][1],
            }
            o = out[c]
            print(f"[255] {y} {c}: Zelltage 255 {o['share_cell_days_255']:.4f}; Zellen nie {o['cells_never']:.3f}, "
                  f"manchmal {o['cells_sometimes']:.3f}, >=90 % {o['cells_ge_90pct']:.3f}, immer {o['cells_always']:.3f}; "
                  f"240-254: {o['cell_days_240_254']:,} gegen 255: {o['cell_days_255']:,}", flush=True)
            print("      Mittel bei 255 / sonst: " + ", ".join(
                f"{n} {o['mean_if_255'][n]:.0f}/{o['mean_if_not_255'][n]:.0f}" for n in names), flush=True)
            print("      je Monat: " + ", ".join(f"{m} {s:.3f}" for m, s in o["share_255_by_month"].items()), flush=True)
        res[y] = out
    write_manifest(cfg, "p31_fwi255_check", args.config, {"years": args.years, "step": args.step, "results": res})


if __name__ == "__main__":
    main()
