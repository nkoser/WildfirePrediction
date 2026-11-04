"""Wie oft brennt ein Pixel nach einer Pause wieder? Grundlage fuer die Frage nach einem Multi-State-Modell
(unverbrannt -> brennend -> erloschen -> wieder brennend).

Liest nur den CSR-Index aller MODIS-Feuertage je Pixel (derived_data/events/fire_index.npz, fire_index.py).
Fuer jedes Paar aufeinanderfolgender Feuertage eines Pixels ist die Luecke g = Tag_{n+1} - Tag_n:
  g = 1          dasselbe Feuer, durchgehend detektiert
  g = 2..3       meist Detektionsluecke (Wolken, Rauch, Ueberflug)
  g = 4..7       unklar: Luecke oder Wiederaufflammen
  g = 8..30      Wiederaufflammen / neues Feuer im selben Pixel, gleiche Saison
  g = 31..183    spaeter in der Saison
  g > 183        Folgejahr(e) (Wiederverbrennung ueber Jahre)
Ausgabe: Verteilung der Luecken (alle Jahre und nur 2023-2025), Anteil der Feuerpixel einer Saison mit
mindestens einer Rueckkehr nach >= 8 / >= 15 / >= 31 Tagen, und Anteil der "ersten Feuer nach einem Stichtag",
die eine Rueckkehr sind (Pixel brannte in den 100 Tagen davor, aber nicht in den 7 Tagen davor).
  -> metrics/paper_multistate/reburn_stats.json (bricht ab, wenn vorhanden)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wildfire.audit.common import load_config

EDGES = (1, 2, 4, 8, 15, 31, 184, 10 ** 9)
LABELS = ("1", "2-3", "4-7", "8-14", "15-30", "31-183", ">183")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    exp = Path(cfg["experiment_root"])
    out = exp / "metrics" / "paper_multistate" / "reburn_stats.json"
    if out.exists():
        raise SystemExit(f"ABBRUCH: {out} existiert")
    z = np.load(exp / "derived_data" / "events" / "fire_index.npz")
    off, days = z["fire_offsets"], z["fire_days"].astype(np.int64)
    dates = np.array([str(d) for d in z["dates"]])
    years = np.array([int(d[:4]) for d in dates])
    n_px = len(off) - 1
    print(f"[reburn] {n_px:,} Pixel mit Feuer, {days.size:,} Feuer-Pixeltage, {dates[0]} .. {dates[-1]}", flush=True)

    pix = np.repeat(np.arange(n_px), np.diff(off))
    same = pix[1:] == pix[:-1]
    gap = (days[1:] - days[:-1])[same]
    gyear = years[days[1:][same]]
    res = {"n_fire_pixels": int(n_px), "n_fire_pixel_days": int(days.size)}
    for name, sel in (("all_years", np.ones_like(gap, bool)), ("test_2023_2025", gyear >= 2023)):
        h = np.histogram(gap[sel], bins=EDGES)[0]
        res[f"gap_hist_{name}"] = dict(zip(LABELS, map(int, h)))
        res[f"gap_share_{name}"] = dict(zip(LABELS, (h / max(h.sum(), 1)).round(4).tolist()))
        print(f"[reburn] Luecken {name}: " + "  ".join(f"{l}: {v:,} ({s:.1%})" for l, v, s in
                                                     zip(LABELS, h, h / max(h.sum(), 1))), flush=True)

    # je Pixel und Saison: Rueckkehr nach >= g Tagen innerhalb derselben Saison
    py = pix * 10000 + years[days]                                        # Pixel x Jahr
    u, start = np.unique(py, return_index=True)
    stop = np.append(start[1:], py.size)
    ret = {g: 0 for g in (8, 15, 31)}
    ret_test = {g: 0 for g in (8, 15, 31)}
    n_season, n_season_test = 0, 0
    for a, b, key in zip(start, stop, u):
        d = days[a:b]
        test = key % 10000 >= 2023
        n_season += 1
        n_season_test += test
        if b - a < 2:
            continue
        mg = np.diff(d).max()
        for g in ret:
            if mg >= g:
                ret[g] += 1
                ret_test[g] += test
    res["pixel_seasons"] = {"n": n_season, "n_test": int(n_season_test)}
    res["share_pixel_seasons_with_return"] = {f">={g}d": round(ret[g] / n_season, 4) for g in ret}
    res["share_pixel_seasons_with_return_test"] = {f">={g}d": round(ret_test[g] / max(n_season_test, 1), 4) for g in ret}
    print(f"[reburn] Pixel-Saisons mit Rueckkehr: alle {res['share_pixel_seasons_with_return']}, "
          f"2023-2025 {res['share_pixel_seasons_with_return_test']}", flush=True)

    # Wiederverbrennung ueber Jahre: Anteil der Pixel-Saisons, deren Pixel in einem frueheren Jahr brannte
    first_year = {}
    reb = {1: 0, 5: 0, 10: 0}
    for key in u:
        p, y = divmod(int(key), 10000)
        if p in first_year:
            dy = y - first_year[p]
            for k in reb:
                reb[k] += dy <= k
        else:
            first_year[p] = y
    res["share_pixel_seasons_burned_in_earlier_year_within"] = {f"{k}y": round(v / n_season, 4) for k, v in reb.items()}
    print(f"[reburn] Pixel-Saisons mit Feuer schon in einem frueheren Jahr (innerhalb k Jahren nach dem ersten): "
          f"{res['share_pixel_seasons_burned_in_earlier_year_within']}", flush=True)

    # Ereignisse "erstes Feuer nach Tag t" (t = jeder Feuertag - 1 als Ersatz-Stichtag): wie viele sind Rueckkehr?
    prev = np.where(same, days[:-1], -10 ** 6)
    g_all = np.append(10 ** 6, days[1:] - prev)                           # Luecke zum vorherigen Feuertag des Pixels
    first_new = g_all > 7                                                 # Ereignis nach >= 7 feuerfreien Tagen
    ret100 = first_new & (g_all <= 100)
    res["events_after_7_fire_free_days"] = {"n": int(first_new.sum()),
                                            "share_return_within_100d": round(float(ret100.sum() / first_new.sum()), 4)}
    print(f"[reburn] Feuertage nach >= 7 feuerfreien Tagen: {first_new.sum():,}, davon Rueckkehr (Pixel brannte "
          f"in den 100 Tagen davor): {ret100.sum() / first_new.sum():.2%}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"[reburn] -> {out}", flush=True)


if __name__ == "__main__":
    main()
