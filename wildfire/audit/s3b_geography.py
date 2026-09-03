"""Stufe 3b: geographische Plausibilitaet und persistente Hotspots.

Nutzt die in Stufe 3 abgelegte Karte ``burn_count.npy`` (Feuertage je
Pixel ueber 26 Jahre) und beantwortet:

  * Welche Region deckt das Gitter tatsaechlich ab - ist es Kanada oder
    nur eine Bounding Box, die grosse Teile der USA einschliesst?
  * Gibt es Pixel mit physikalisch unplausibel vielen Feuertagen
    (Industrieanlagen, Gasfackeln, urbane Waermeinseln)?
  * Wie verteilen sich Feuertage ueber Breitengrad und Jahreszeit?
  * Wie gross ist der Anteil an Winterdetektionen, die in borealen
    Breiten kein Waldbrand sein koennen?

Erzeugt zusaetzlich zwei Abbildungen im Experimentverzeichnis.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wildfire.audit.common import assert_readonly, load_config, read_manifest, write_manifest

# Persistenz-Schwelle: mehr Feuertage als das an einem einzelnen 5-km-Pixel
# durch Waldbraende erklaerbar waere.
HOTSPOT_MIN_DAYS = 30
WINTER_MONTHS = ("12", "01", "02")


def pixel_lonlat(transform: list[float], rows: np.ndarray, cols: np.ndarray):
    lon = transform[2] + transform[0] * (cols + 0.5)
    lat = transform[5] + transform[4] * (rows + 0.5)
    return lon, lat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--scratch", default="/data/Wildfire/derived_data/audit_scratch")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)

    s3 = read_manifest(cfg, "s3_fire_labels")
    transform = s3["profile_example"]["transform"]
    bc = np.load(Path(args.scratch) / "burn_count.npy")
    total = int(bc.sum())

    # ------------------------------------------------ persistente Hotspots
    rr, cc = np.nonzero(bc >= HOTSPOT_MIN_DAYS)
    lon, lat = pixel_lonlat(transform, rr, cc)
    order = np.argsort(-bc[rr, cc])
    hotspots = [
        {
            "row": int(rr[i]),
            "col": int(cc[i]),
            "lon": float(lon[i]),
            "lat": float(lat[i]),
            "fire_days": int(bc[rr[i], cc[i]]),
        }
        for i in order[:40]
    ]
    hot_share = float(bc[rr, cc].sum() / max(1, total))

    # ------------------------------------------------ Breitenverteilung
    rows_any, _ = np.nonzero(bc > 0)
    _, lat_all = pixel_lonlat(transform, rows_any, np.zeros_like(rows_any))
    weights = bc[bc > 0]
    lat_w = np.repeat(lat_all, 1)  # eine Zeile je Pixel, Gewicht = Feuertage
    hist, edges = np.histogram(lat_w, bins=np.arange(40, 85, 2.5), weights=weights)
    lat_profile = {f"{edges[i]:.1f}-{edges[i+1]:.1f}": int(hist[i]) for i in range(len(hist))}

    # Anteil suedlich von 49degN (grob: USA-Anteil der Bounding Box)
    south = float(weights[lat_all < 49.0].sum() / max(1, total))
    south60 = float(weights[lat_all < 60.0].sum() / max(1, total))

    # ------------------------------------------------ Saisonalitaet
    per_day = s3["per_day"]
    per_month: Counter = Counter()
    for r in per_day:
        per_month[r["date"][5:7]] += r["n_fire_px"]
    winter = sum(per_month[m] for m in WINTER_MONTHS)

    # ------------------------------------------------ Abbildungen
    figs = Path(cfg["experiment_root"]) / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    im = ax.imshow(np.log1p(bc), cmap="inferno", interpolation="nearest")
    ax.set_title("Feuertage je Pixel, 2000-2025 (log1p)")
    ax.set_xlabel("Spalte"); ax.set_ylabel("Zeile")
    fig.colorbar(im, ax=ax, shrink=0.8, label="log(1 + Feuertage)")
    fig.tight_layout(); fig.savefig(figs / "burn_count_map.png", dpi=130); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].bar(list(per_month), [per_month[k] for k in sorted(per_month)])
    axs[0].set_title("Feuer-Pixeltage je Monat"); axs[0].set_xlabel("Monat")
    axs[1].barh([k for k in lat_profile], list(lat_profile.values()))
    axs[1].set_title("Feuer-Pixeltage je Breitenband"); axs[1].set_xlabel("Pixeltage")
    fig.tight_layout(); fig.savefig(figs / "fire_season_latitude.png", dpi=130); plt.close(fig)

    payload = {
        "grid_bounds_lonlat": s3["profile_example"],
        "total_fire_pixel_days": total,
        "hotspots": {
            "threshold_days": HOTSPOT_MIN_DAYS,
            "n_pixels_above_threshold": int(rr.size),
            "share_of_all_fire_pixel_days": hot_share,
            "top": hotspots,
        },
        "latitude_profile_fire_pixel_days": lat_profile,
        "share_south_of_49N": south,
        "share_south_of_60N": south60,
        "fire_pixel_days_per_month": {k: int(v) for k, v in sorted(per_month.items())},
        "winter_share": winter / max(1, total),
        "figures": [str(figs / "burn_count_map.png"), str(figs / "fire_season_latitude.png")],
    }
    print(f"[s3b] Hotspots >= {HOTSPOT_MIN_DAYS} Feuertage: {rr.size} Pixel, "
          f"{hot_share*100:.2f} % aller Feuer-Pixeltage")
    for h in hotspots[:10]:
        print(f"    ({h['row']:3d},{h['col']:4d}) lon={h['lon']:8.3f} lat={h['lat']:6.2f}  {h['fire_days']} Tage")
    print(f"[s3b] Anteil suedlich 49degN: {south*100:.1f} %, suedlich 60degN: {south60*100:.1f} %")
    print(f"[s3b] Winteranteil (Dez-Feb): {payload['winter_share']*100:.2f} %")
    write_manifest(cfg, "s3b_geography", args.config, payload)


if __name__ == "__main__":
    main()
