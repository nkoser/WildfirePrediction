"""Abbildungen zu risk_map: Pixel-Risiko F(tau) ueber Kanada mit den MODIS-Feuern danach.

Je Stichtag eine Teaser-Abbildung:
  oben    ganz Kanada, F(tau_main) auf Log-Skala; blau die Feuerpixel der naechsten
          tau_main Tage, schwarz die Feuer im Eingabefenster (vom Modell gesehen);
          Rahmen um die Zoom-Ausschnitte
  unten   je Zoom-Ausschnitt F(tau) fuer mehrere tau mit den Feuern bis tau
Dazu Kennzahlen je tau: Anteil der kuenftigen Feuerpixel in den riskantesten q %
der Landflaeche, getrennt nach "neu" (kein Feuer im 13 x 13-Umfeld im Fenster)
und "nahe bestehendem Feuer", sowie ROC-AUC und AP auf Pixelebene.

Laeuft auf CPU, liest nur die npz aus figures/risk_maps.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy.ndimage import maximum_filter  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from wildfire.audit.common import load_config, read_manifest  # noqa: E402
from wildfire.paper.surv_prepare import HALF  # noqa: E402

TOP_Q = (0.1, 0.5, 1.0, 5.0)
C_NEW = "#1f5fbf"        # Feuer nach dem Stichtag
C_OLD = "#111111"        # Feuer im Eingabefenster
C_SEA = "#eef1f4"
CMAP = "OrRd"


def grid_geo(cfg):
    t = read_manifest(cfg, "s3_fire_labels")["profile_example"]["transform"]
    return t[2], t[0], t[5], t[4]                     # lon0, dlon, lat0, dlat (< 0)


def to_lonlat(geo, r, c):
    lon0, dl, lat0, dla = geo
    return lon0 + dl * (np.asarray(c) + 0.5), lat0 + dla * (np.asarray(r) + 0.5)


def borders(path: Path, admin: str | None = "Canada"):
    """Ringe (lon, lat) aus einem Natural-Earth-GeoJSON."""
    g = json.loads(path.read_text())
    rings = []
    for f in g["features"]:
        if admin and f["properties"].get("admin", f["properties"].get("ADMIN")) != admin:
            continue
        geom = f["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for p in polys:
            rings += [np.asarray(r) for r in p]
    return rings


def capture(F, mask, first, tau, near):
    """Anteil der Feuerpixel (erstes Feuer bis tau) in den riskantesten q % der Landpixel; dazu AUC und AP."""
    f = F[mask]
    y = ((first > 0) & (first <= tau))[mask]
    nr = near[mask]
    order = np.argsort(-f, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    r = {"n_fire_px": int(y.sum()), "n_new": int((y & ~nr).sum()), "n_near": int((y & nr).sum()),
         "base_rate": float(y.mean())}
    for q in TOP_Q:
        top = rank < int(round(order.size * q / 100))
        for name, sel in (("all", y), ("new", y & ~nr), ("near", y & nr)):
            r[f"top{q}%_{name}"] = float(top[sel].mean()) if sel.any() else float("nan")
    if 0 < y.sum() < y.size:
        r["auc"], r["ap"] = float(roc_auc_score(y, f)), float(average_precision_score(y, f))
    return r


def auto_zoom(first, fire_win, mask, tau, geo, n=2, size_deg=(7.0, 4.0)):
    """n Ausschnitte (lon_min, lon_max, lat_min, lat_max) mit den meisten Feuerpixeln bis tau, ohne Ueberlappung."""
    lon0, dl, lat0, dla = geo
    w, h = int(size_deg[0] / dl), int(size_deg[1] / -dla)
    y = ((first > 0) & (first <= tau)).astype(np.float64)
    cs = np.pad(y, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    cnt = cs[h:, w:] - cs[:-h, w:] - cs[h:, :-w] + cs[:-h, :-w]       # Summe im Fenster ab (r, c)
    boxes = []
    for _ in range(n):
        r, c = np.unravel_index(np.argmax(cnt), cnt.shape)
        if cnt[r, c] <= 0:
            break
        boxes.append((lon0 + dl * c, lon0 + dl * (c + w), lat0 + dla * (r + h), lat0 + dla * r))
        cnt[max(0, r - h):r + h, max(0, c - w):c + w] = -1
    return boxes


def draw(ax, F, extent, norm, mask, first, fire_win, tau, geo, rings, box=None, ms=1.0, outline=False):
    """Risiko F auf Log-Skala, Feuer nur innerhalb der Maske: im Fenster schwarz gefuellt, danach blau
    (Punkte, oder mit outline als Umriss, damit das Risiko darunter sichtbar bleibt)."""
    ax.set_facecolor(C_SEA)
    img = np.where(mask, F, np.nan)
    im = ax.imshow(img, extent=extent, cmap=CMAP, norm=norm, interpolation="nearest", origin="upper")
    for ring in rings:
        ax.plot(ring[:, 0], ring[:, 1], color="#8a8f96", lw=0.3)
    old = fire_win & mask
    new = (first > 0) & (first <= tau) & mask
    lo, la = to_lonlat(geo, *np.nonzero(old))
    ax.scatter(lo, la, s=ms, c=C_OLD, marker="s", linewidths=0, zorder=3)
    if outline:
        ax.contour(new.astype(float), levels=[0.5], extent=extent, origin="upper", colors=C_NEW,
                   linewidths=0.8, zorder=4)
    else:
        lo, la = to_lonlat(geo, *np.nonzero(new))
        ax.scatter(lo, la, s=ms, c=C_NEW, marker="s", linewidths=0, zorder=4)
    if box is not None:
        ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3])
        ax.set_aspect(1 / np.cos(np.deg2rad(0.5 * (box[2] + box[3]))))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4); s.set_color("#666666")
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--glob", default="/data/Wildfire/figures/risk_maps/risk_*.npz")
    ap.add_argument("--tau-main", type=int, default=7)
    ap.add_argument("--zoom-taus", default="1,7,30")
    ap.add_argument("--zoom", default=None, help="lon_min,lon_max,lat_min,lat_max[;...], Standard: automatisch")
    ap.add_argument("--lat-max", type=float, default=70.0)
    ap.add_argument("--vmin", type=float, default=1e-4)
    args = ap.parse_args()
    cfg = load_config(args.config)
    exp = Path(cfg["experiment_root"])
    geo = grid_geo(cfg)
    rings = borders(exp / "artifacts" / "ne_50m_admin_1_states_provinces.geojson")
    zt = [int(t) for t in args.zoom_taus.split(",")]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
    stats = {}
    for fn in sorted(glob.glob(args.glob)):
        z = np.load(fn)
        date, mask, first, fire_win = str(z["date"]), z["mask"], z["first_fire"], z["fire_window"]
        Hg, Wg = mask.shape
        lon0, dl, lat0, dla = geo
        extent = (lon0, lon0 + dl * Wg, lat0 + dla * Hg, lat0)
        near = maximum_filter(fire_win, size=2 * HALF + 1)            # Feuer im 13 x 13-Umfeld im Fenster
        st = {int(t): capture(z[f"F_{t}"], mask, first, int(t), near) for t in z["taus"]}
        st["checkpoints"] = [str(c) for c in z["checkpoints"]]
        stats[Path(fn).stem] = st
        for t, r in ((t, r) for t, r in st.items() if t != "checkpoints"):
            print(f"[plot] {Path(fn).stem} tau {t:2d}: {r['n_fire_px']:5d} Feuerpx ({r['n_new']} neu), "
                  f"AUC {r.get('auc', float('nan')):.3f} AP {r.get('ap', float('nan')):.3f}  "
                  + "  ".join(f"top{q}%: {r[f'top{q}%_all']:.2f} (neu {r[f'top{q}%_new']:.2f})" for q in TOP_Q), flush=True)

        vmax = float(np.nanpercentile(z[f"F_{max(zt)}"], 99.9))
        norm = LogNorm(vmin=args.vmin, vmax=max(vmax, 10 * args.vmin))
        boxes = ([tuple(float(v) for v in b.split(",")) for b in args.zoom.split(";")] if args.zoom
                 else auto_zoom(first, fire_win, mask, max(zt), geo))
        rows_any = np.nonzero(mask.any(1))[0]
        cols_any = np.nonzero(mask.any(0))[0]
        lat_lo = lat0 + dla * (rows_any.max() + 1)
        lon_lo, lon_hi = lon0 + dl * cols_any.min(), lon0 + dl * (cols_any.max() + 1)

        fig = plt.figure(figsize=(7.0, 4.0 + 2.1 * len(boxes)), dpi=300)
        gs = fig.add_gridspec(1 + len(boxes), len(zt), height_ratios=[4.0] + [2.1] * len(boxes), hspace=0.1, wspace=0.03)
        ax0 = fig.add_subplot(gs[0, :])
        im = draw(ax0, z[f"F_{args.tau_main}"], extent, norm, mask, first, fire_win, args.tau_main, geo, rings,
                  box=(lon_lo, lon_hi, lat_lo, args.lat_max), ms=0.6)
        for i, b in enumerate(boxes):
            ax0.add_patch(Rectangle((b[0], b[2]), b[1] - b[0], b[3] - b[2], fill=False, lw=0.8, ec="#333333", zorder=5))
            ax0.text(b[1], b[3], f" {chr(65 + i)}", va="top", ha="left", fontsize=7, zorder=5)
        ax0.set_title(f"Risk of a fire within {args.tau_main} days, forecast date {date}", fontsize=7.5, loc="left")
        for i, b in enumerate(boxes):
            for j, t in enumerate(zt):
                ax = fig.add_subplot(gs[1 + i, j])
                draw(ax, z[f"F_{t}"], extent, norm, mask, first, fire_win, t, geo, rings, box=b, ms=3.0, outline=True)
                if i == 0:
                    ax.set_title(f"F(τ = {t} d)", fontsize=7.5)
                if j == 0:
                    ax.set_ylabel(f"{chr(65 + i)}  {abs(b[2]):.0f}–{abs(b[3]):.0f}°N, {abs(b[1]):.0f}–{abs(b[0]):.0f}°W",
                                  fontsize=6.5)
        cb = fig.colorbar(im, ax=fig.axes, orientation="horizontal", fraction=0.025, pad=0.02, aspect=50)
        cb.set_label("Contagion-STS, predicted probability of a fire in the pixel within τ days (raw, uncalibrated)", fontsize=6.5)
        cb.ax.tick_params(labelsize=6)
        h_new = plt.Line2D([], [], ls="", marker="s", ms=3, color=C_NEW, label="MODIS fire after the forecast date (≤ τ; outline in zooms)")
        h_old = plt.Line2D([], [], ls="", marker="s", ms=3, color=C_OLD, label="MODIS fire in the 30-day input window")
        ax0.legend(handles=[h_new, h_old], loc="lower left", fontsize=6, frameon=True, framealpha=0.9)
        out = Path(fn).with_suffix("")
        fig.savefig(f"{out}.png", bbox_inches="tight")
        fig.savefig(f"{out}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] {out}.png / .pdf", flush=True)
    (exp / "figures" / "risk_maps" / "capture_stats.json").write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
