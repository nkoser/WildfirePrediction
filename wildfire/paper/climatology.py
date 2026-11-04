"""Klimatologie je Paper-Stichprobe: wie oft brannte dieser Ausschnitt um dieses Kalenderdatum?

Bei langen Vorlaufzeiten (dt = 30) sagt das aktuelle Wetter wenig ueber den Zieltag. Was
bleibt, ist der Ort und die Jahreszeit. Hier wird das als Merkmal je Stichprobe des
Paper-Caches berechnet, nur aus den Jahren 2001-2019 (vor Val und Test, ohne das eigene
Jahr der Stichprobe), aus derselben Quelle wie alle Ziele (MCD14A1 == 1, unmaskiert):

  day      Anteil der Tage mit Feuer im 13 x 13-Ausschnitt in [Z - 7, Z + 7] (Z = Zieltag D + 1,
           auf das Jahr y verschoben), gemittelt ueber die Jahre: Schaetzer fuer P(Feuer an Z)
  season   Anteil der Jahre mit Feuer im Ausschnitt irgendwann in der Saison (1. April bis 30. Sept.)
  n_years  Zahl der Jahre im Mittel (18 oder 19)

Ausgabe derived_data/paper/<cache>_L61/clim.npz und Manifest p24_climatology_<cache>.
Selbsttest: fuer einen Teil der Stichproben wird `day` einzeln ohne Vektorisierung nachgerechnet.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.data.dataset import CubeReader
from wildfire.paper.rate_labels import daily_fire
from wildfire.paper.surv_prepare import DAY_BITS, HALF, fire_keys

YEARS = range(2001, 2020)
WIN = 7


def to_date(m: int) -> dt.date:
    return dt.date(int(m // 10000), int(m // 100 % 100), int(m % 100))


def season_bounds(y: int) -> tuple[dt.date, dt.date]:
    return dt.date(y, 4, 1), dt.date(y, 9, 30)


def clim_features(keys, t_of, meta, H, W, years=YEARS, win=WIN):
    """(day, season, n_years) je Stichprobe; Zieltag = Datum aus meta + 1 Tag."""
    n = len(meta)
    Z = [to_date(m) + dt.timedelta(days=1) for m in meta[:, 0]]
    rr, cc = meta[:, 1].astype(np.int64), meta[:, 2].astype(np.int64)
    day_sum, season_sum, ny = np.zeros(n), np.zeros(n), np.zeros(n)
    for y in years:
        use = np.array([z.year != y and not (z.month == 2 and z.day == 29) for z in Z])
        a, b = season_bounds(y)
        # Fenster um Z im Jahr y, auf die Saison beschnitten
        lo = np.array([max(dt.date(y, z.month, z.day) - dt.timedelta(days=win), a) if u else a for z, u in zip(Z, use)])
        hi = np.array([min(dt.date(y, z.month, z.day) + dt.timedelta(days=win), b) if u else a for z, u in zip(Z, use)])
        nd = np.array([(h - l).days + 1 for l, h in zip(lo, hi)])
        use &= nd > 0
        L = np.array([t_of[l.isoformat()] - 1 for l in lo])
        p, _ = daily_fire(keys, L, rr, cc, H, W, horizon=2 * win + 1)
        p &= np.arange(2 * win + 1)[None] < nd[:, None]
        day_sum += np.where(use, p.sum(1) / np.maximum(nd, 1), 0.0)
        Ls = np.full(n, t_of[a.isoformat()] - 1)
        ps, _ = daily_fire(keys, Ls, rr, cc, H, W, horizon=(b - a).days + 1)
        season_sum += np.where(use, ps.any(1), 0.0)
        ny += use
    return day_sum / np.maximum(ny, 1), season_sum / np.maximum(ny, 1), ny


def day_reference(keys, t_of, m, r, c, W, years=YEARS, win=WIN) -> float:
    """Unvektorisiert: Anteil der Feuertage im Fenster je Jahr, gemittelt."""
    z = to_date(m) + dt.timedelta(days=1)
    vals = []
    for y in years:
        if y == z.year or (z.month == 2 and z.day == 29):
            continue
        a, b = season_bounds(y)
        lo, hi = max(dt.date(y, z.month, z.day) - dt.timedelta(days=win), a), min(dt.date(y, z.month, z.day) + dt.timedelta(days=win), b)
        if hi < lo:
            continue
        days = [(lo + dt.timedelta(days=k)).isoformat() for k in range((hi - lo).days + 1)]
        hit = 0
        for d in days:
            t = t_of[d]
            for dr in range(-HALF, HALF + 1):
                for dc in range(-HALF, HALF + 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr and 0 <= cc < W:
                        k = (rr * W + cc) * DAY_BITS + t
                        i = np.searchsorted(keys, k)
                        if i < keys.size and keys[i] == k:
                            hit += 1
                            break
                else:
                    continue
                break
        vals.append(hit / len(days))
    return float(np.mean(vals)) if vals else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--n-check", type=int, default=200)
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    cube = CubeReader(exp)
    t_of = {d: i for i, d in enumerate(cube.dates)}
    keys = fire_keys(Path(cfg["dataset_root"]), t_of, cube.W)
    d = exp / "derived_data" / "paper" / f"{args.cache}_L61"
    meta, split = np.load(d / "meta.npy"), np.load(d / "split.npy")
    yrs = meta[:, 0] // 10000
    print("[clim] Jahre je Split: " + "  ".join(f"{s} {yrs[split == s].min()}-{yrs[split == s].max()}" for s in ("train", "val", "test")), flush=True)

    day, season, ny = clim_features(keys, t_of, meta, cube.H, cube.W)
    rng = np.random.default_rng(0)
    chk = rng.choice(len(meta), min(args.n_check, len(meta)), replace=False)
    ref = np.array([day_reference(keys, t_of, *meta[i], cube.W) for i in chk])
    err = float(np.abs(ref - day[chk]).max())
    print(f"[clim] Selbsttest day gegen unvektorisiert ({chk.size} Stichproben): max. Abweichung {err:.2e}", flush=True)
    if err > 1e-9:
        raise SystemExit("ABBRUCH: Klimatologie stimmt nicht mit der Einzelrechnung ueberein")
    np.savez_compressed(d / "clim.npz", day=day, season=season, n_years=ny)

    y = np.load(d / "y_next_patch.npy").astype(int)
    summary = {s: {"n": int((split == s).sum()), "mean_day": float(day[split == s].mean()),
                   "mean_season": float(season[split == s].mean()),
                   "pr_auc_day_vs_next_patch": float(average_precision_score(y[split == s], day[split == s])),
                   "pr_auc_season_vs_next_patch": float(average_precision_score(y[split == s], season[split == s]))}
               for s in ("train", "val", "test")}
    print(f"[clim] {summary}", flush=True)
    write_manifest(cfg, f"p24_climatology_{args.cache}", args.config, {
        "years": [min(YEARS), max(YEARS)], "window_days": WIN, "own_year_excluded": True,
        "features": {"day": "Anteil Feuertage im Ausschnitt in [Z - 7, Z + 7] im Jahr y, Mittel ueber y (Z = D + 1)",
                     "season": "Anteil der Jahre mit Feuer im Ausschnitt in der Saison 1.4.-30.9."},
        "self_test_max_abs_err": err, "summary": summary,
    })


if __name__ == "__main__":
    main()
