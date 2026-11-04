"""Folienzeilen zum Ausbreitungs-Teaser: beobachtet | Modell 1 | Modell 2 ... je Ereignis, mit GEMEINSAMEN Beispielpixeln.

In teaser_spread.plot waehlt jede Abbildung ihre Beispielpixel a/b/c auf der eigenen Karte; nebeneinander
gestellte Panels zweier Modelle zeigen dann a/b/c an verschiedenen Orten. Hier werden die Pixel wie in
teaser_oracle_compare einmal auf der Karte des Bezugsmodells (--ref) gewaehlt und in allen Panels an
derselben Stelle markiert. Dazu eine eigene Legende mit Tagesklassen, Feuer im Eingabefenster, Land und Wasser.

Ausgabe (neuer Ordner, nichts wird ueberschrieben): figures/teaser_rows<tag>/row_<datum>.png, legend.png, pixels.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from wildfire.audit.common import load_config  # noqa: E402
from wildfire.paper.risk_map_plot import borders, grid_geo  # noqa: E402
from wildfire.paper.teaser_spread import (BIN_LABELS, BINS, C_LAND, C_WATER, C_WIN, COLORS, arrival_pred,  # noqa: E402
                                          box_slice, draw_arrival, load, pick_pixels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tags", default="teaser_lh,teaser_cgs")
    ap.add_argument("--ref", default="teaser_cgs")
    ap.add_argument("--events", default="/data/Wildfire/figures/teaser/teaser_v1_cgs.json")
    ap.add_argument("--p", type=float, default=0.05)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    exp = Path(cfg["experiment_root"])
    out = exp / "figures" / f"teaser_rows{args.tag}"
    if out.exists():
        raise SystemExit(f"ABBRUCH: {out} existiert")
    out.mkdir(parents=True)
    geo = grid_geo(cfg)
    rings = borders(exp / "artifacts" / "ne_50m_admin_1_states_provinces.geojson")
    events = [(d, tuple(b)) for d, b in json.loads(Path(args.events).read_text())["events"]]
    tags = args.tags.split(",")
    cmap, norm = ListedColormap(COLORS), BoundaryNorm(BINS, len(COLORS))
    lon0, dl, lat0, dla = geo
    info = {}
    for date, box in events:
        maps = {t: load(exp, t, date) for t in tags + ([args.ref] if args.ref not in tags else [])}
        ref = maps[args.ref]
        sl = box_slice(box, geo, ref["mask"].shape)
        extent = (lon0 + dl * sl[1].start, lon0 + dl * sl[1].stop, lat0 + dla * sl[0].stop, lat0 + dla * sl[0].start)
        pix = pick_pixels(ref, sl, args.p)
        obs = np.where(ref["first"][sl] > 0, ref["first"][sl], 31)
        fig, axes = plt.subplots(1, 1 + len(tags), figsize=(1.6 * (1 + len(tags)), 1.6), dpi=300,
                                 gridspec_kw={"wspace": 0.06})
        panels = [obs] + [arrival_pred(maps[t]["F"][sl], args.p) for t in tags]
        for ax, a in zip(axes, panels):
            draw_arrival(ax, a, ref, sl, extent, rings, cmap, norm, box)
            for k, (_, r, c) in enumerate(pix):
                x, y = extent[0] + dl * (c + 0.5), extent[3] + dla * (r + 0.5)
                ax.plot(x, y, marker="o", ms=3.2, mfc="white", mec="#1b3a5c", mew=0.7, zorder=6)
                ax.annotate("abc"[k], (x, y), xytext=(2.5, 2.5), textcoords="offset points", fontsize=6.5,
                            color="#1b3a5c", fontweight="bold", zorder=6)
        fig.savefig(out / f"row_{date}.png", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        info[date] = {"box": list(box), "ref": args.ref,
                      "pixels": {"abc"[k]: {"kind": n, "row": r, "col": c, "observed_day": int(obs[r, c]) if obs[r, c] <= 30 else None,
                                            "F8": {t: float(maps[t]["F"][sl][r, c, 7]) for t in tags},
                                            "F30": {t: float(maps[t]["F"][sl][r, c, 29]) for t in tags}}
                                 for k, (n, r, c) in enumerate(pix)}}
        print(f"[rows] {date}: " + json.dumps(info[date]["pixels"]), flush=True)

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(4.6, 0.95), dpi=300)
    cax = fig.add_axes([0.02, 0.62, 0.96, 0.18])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                      ticks=[0.5 * (BINS[k] + BINS[k + 1]) for k in range(len(BIN_LABELS))])
    cb.ax.set_xticklabels(BIN_LABELS, fontsize=7)
    cb.ax.xaxis.set_ticks_position("top")
    cb.set_label("first fire, days after the forecast date", fontsize=7, labelpad=2)
    cb.ax.xaxis.set_label_position("top")
    lax = fig.add_axes([0.0, 0.0, 1.0, 0.42])
    lax.set_axis_off()
    lax.legend(handles=[Patch(fc=C_WIN, ec="none", label="burning in the 30 input days"),
                        Patch(fc=C_LAND, ec="#bbbbbb", lw=0.4, label="land, no fire"),
                        Patch(fc=C_WATER, ec="none", label="water")],
               loc="center", ncol=3, fontsize=7, frameon=False, handlelength=1.2, columnspacing=1.4)
    fig.savefig(out / "legend.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    (out / "pixels.json").write_text(json.dumps(info, indent=1))
    print(f"[rows] -> {out}", flush=True)


if __name__ == "__main__":
    main()
