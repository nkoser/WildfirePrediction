"""Wie weit springt Feuer pro Tag? Diagnose fuer den Contagion-Kern (Radius 5) direkt aus den MODIS-Feuerpixeln.

Je Feuerpixel-Tag (MCD14A1 == 1, unmaskiert, wie Cache-Kanal 0, ganzes Gitter) mit Tag t:
  neu        das Pixel hat an den Tagen t-1 .. t-Q (--quiet) nicht gebrannt
  Abstand    zum naechsten Feuerpixel der Tage t-1 .. t-G (--gap, Gitterpixel, euklidisch), nur Tage, die
             kalendarisch direkt vor t liegen; ueber --max-dist gilt die Zuendung als unabhaengig
Ausgaben: Verteilung der Abstaende (0 = Pixel brannte schon: Persistenz, nicht neu), Anteil der neuen
Zuendungen innerhalb Radius 1, 2, 3, 5, 7, 10, 15, 20, 30, je Jahrgruppe Training (2000-2019) und Rest;
zusaetzlich fuer grosse Brandtage (>= --big Feuerpixel im Umkreis von 30 Pixeln am Vortag).
Nur Lesen; Ausgabe NEU: metrics/spread_distance.json, Manifest p61_spread_distance.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.data.dataset import CubeReader
from wildfire.paper.surv_prepare import DAY_BITS, fire_keys

RADII = (1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--quiet", type=int, default=3, help="neu = an den Q Vortagen kein Feuer an diesem Pixel")
    ap.add_argument("--gap", type=int, default=2, help="Quellen: Feuer der G Vortage")
    ap.add_argument("--max-dist", type=float, default=30.0)
    ap.add_argument("--big", type=int, default=50)
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    cube = CubeReader(exp)
    W = cube.W
    dates = [dt.date.fromisoformat(s) for s in cube.dates]
    keys = fire_keys(Path(cfg["dataset_root"]), {d: i for i, d in enumerate(cube.dates)}, W)
    day = keys % DAY_BITS
    pix = keys // DAY_BITS
    order = np.argsort(day, kind="stable")
    day, pix = day[order], pix[order]
    bounds = np.searchsorted(day, np.arange(len(dates) + 1))
    per_day = [pix[bounds[t]:bounds[t + 1]] for t in range(len(dates))]
    print(f"[sd] {keys.size:,} Feuerpixel-Tage, {len(dates)} Wuerfeltage", flush=True)

    def prev_days(t, n):
        """Indizes der Wuerfeltage, die kalendarisch 1..n Tage vor t liegen."""
        return [t - k for k in range(1, n + 1) if t - k >= 0 and (dates[t] - dates[t - k]).days == k]

    groups = {"train_2000_2019": [], "other_years": []}
    big = []
    n_persist = {"train_2000_2019": 0, "other_years": 0}
    n_all = {"train_2000_2019": 0, "other_years": 0}
    for t in range(len(dates)):
        cur = per_day[t]
        if cur.size == 0:
            continue
        g = "train_2000_2019" if dates[t].year <= 2019 else "other_years"
        pq = prev_days(t, args.quiet)
        pg = prev_days(t, args.gap)
        if len(pq) < args.quiet or len(pg) < args.gap:
            continue                                       # Saisonanfang/Luecke: Vorgeschichte unbekannt
        recent = np.unique(np.concatenate([per_day[s] for s in pq]))
        new = cur[~np.isin(cur, recent)]
        n_all[g] += cur.size
        n_persist[g] += cur.size - new.size
        if new.size == 0:
            continue
        src = np.unique(np.concatenate([per_day[s] for s in pg]))
        if src.size == 0:
            groups[g].append(np.full(new.size, np.inf))
            continue
        tree = cKDTree(np.stack([src // W, src % W], 1))
        q = np.stack([new // W, new % W], 1)
        d, _ = tree.query(q, distance_upper_bound=args.max_dist)
        groups[g].append(d)
        cnt = np.array([len(v) for v in tree.query_ball_point(q, 30.0)])   # Feuerdichte der Vortage im Umkreis
        big.append(d[cnt >= args.big])
        if t % 500 == 0:
            print(f"    Tag {t}/{len(dates)} {dates[t]}", flush=True)

    def summ(arr, n_new_total=None):
        d = np.concatenate(arr) if arr else np.zeros(0)
        n = d.size
        out = {"n_new": int(n), "independent_beyond_max": float(np.isinf(d).mean()) if n else None}
        for r in RADII:
            out[f"within_{r}"] = float((d <= r).mean()) if n else None
        fin = d[np.isfinite(d)]
        out["conditional_on_source_within_max"] = {f"within_{r}": float((fin <= r).mean()) for r in RADII} if fin.size else None
        out["quantiles_finite"] = {q: float(np.quantile(fin, q)) for q in (0.5, 0.75, 0.9, 0.95, 0.99)} if fin.size else None
        return out

    res = {g: summ(v) for g, v in groups.items()}
    for g in res:
        res[g]["persist_share_of_fire_pixel_days"] = n_persist[g] / max(n_all[g], 1)
    res["big_fire_days"] = summ(big)
    for g, v in res.items():
        print(f"[sd] {g}: {json.dumps(v)}", flush=True)
    out = exp / "metrics" / "spread_distance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f"ABBRUCH: {out} existiert schon")
    out.write_text(json.dumps({"args": vars(args), "results": res}, indent=1))
    write_manifest(cfg, "p61_spread_distance", args.config, {"out": str(out), "results": res, "args": vars(args)})


if __name__ == "__main__":
    main()
