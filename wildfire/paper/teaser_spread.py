"""Teaser: beobachtete Feuerausbreitung gegen die vorhergesagte Ankunftszeit aus EINER monotonen Pixel-CIF.

Liest die npz von risk_map (mit --taus 1,...,30, also F(tau) fuer jeden Tag) und zeichnet je Ereignis
  (a) Eingabe     MODIS-Feuer im 30-Tage-Fenster bis zum Stichtag L (vom Modell gesehen)
  (b) beobachtet  Tag der ersten MODIS-Detektion nach L fuer Pixel, die im Fenster nicht brannten
  (c) vorhergesagt  Ankunftstag tau_p(x) = min{tau : F(tau | x) >= p}; wohldefiniert nur, weil F in tau
                  monoton ist (gestapelte Klassifikatoren pro Vorlaufzeit liefern das nicht);
                  Isochronen sind deshalb automatisch verschachtelt
  (d) Pixel-CIF   F(tau) an drei markierten Pixeln mit dem beobachteten Ankunftstag
(b) und (c) teilen eine diskrete Farbskala in Tagen nach L.

  python -m wildfire.paper.teaser_spread --config ... --tag teaser_cgs scan
      listet je Stichtag die Ausschnitte mit der staerksten Ausbreitung und Kennzahlen je Schwelle p
  python -m wildfire.paper.teaser_spread --config ... --tag teaser_cgs plot --event 2025-05-26:-106,-102.5,53.3,55.3 ...
      zeichnet die Abbildung (mehrere --event = mehrere Zeilen); ohne --event die besten Ausschnitte aus scan

Kennzahlen je Ausschnitt (nur Landpixel, die im Fenster nicht brannten): ROC-AUC von F(tau) fuer
"Ankunft bis tau", Spearman zwischen erwarteter Ankunft E[min(T, 30)] = sum_{t<30} S(t) und beobachteter
Ankunft (keine Ankunft = 31) ueber Pixel mit Feuer im 30-Tage-Horizont, und Trefferquote der Isochrone.
Wahrscheinlichkeiten roh wie in risk_map (nicht kalibriert); --p waehlt die Schwelle.
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
from scipy.ndimage import binary_dilation  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from wildfire.audit.common import load_config  # noqa: E402
from wildfire.paper.risk_map_plot import borders, grid_geo  # noqa: E402

BINS = (1, 2, 4, 8, 15, 31)                      # Tage nach L: 1 | 2-3 | 4-7 | 8-14 | 15-30
BIN_LABELS = ("1", "2–3", "4–7", "8–14", "15–30")
COLORS = ("#7a0403", "#c23b0a", "#ef7d1a", "#f9c74f", "#fbeeb0")
C_WIN = "#1d1d1f"                                # Feuer im Eingabefenster
C_LAND = "#f4f3ef"
C_WATER = "#dfe6ec"
BOX_DEG = (3.5, 2.0)                             # Ausschnitt lon x lat fuer scan


def load(exp: Path, tag: str, date: str):
    z = np.load(exp / "figures" / "risk_maps" / f"risk_{tag}_{date}.npz")
    taus = [int(t) for t in z["taus"]]
    if taus != list(range(1, len(taus) + 1)) or len(taus) < 30:
        raise SystemExit(f"ABBRUCH: {tag} {date} braucht --taus 1,...,30 in risk_map, hat {taus}")
    F = np.stack([z[f"F_{t}"] for t in taus], -1)                     # (H, W, 30)
    ign = np.stack([z[f"ign_{t}"] for t in taus], -1) if "ign_1" in z else None
    return dict(F=F, ign=ign, mask=z["mask"], win=z["fire_window"], first=z["first_fire"].astype(np.int16),
                date=str(z["date"]), ckpts=[str(c) for c in z["checkpoints"]])


def arrival_pred(F: np.ndarray, p: float) -> np.ndarray:
    """min{tau : F(tau) >= p}, 31 = nicht bis 30 Tage; F ist monoton, also ist das die erste Ueberschreitung."""
    hit = F >= p
    return np.where(hit.any(-1), hit.argmax(-1) + 1, 31).astype(np.int16)


def box_slice(box, geo, shape):
    lon0, dl, lat0, dla = geo
    c0, c1 = int((box[0] - lon0) / dl), int(np.ceil((box[1] - lon0) / dl))
    r0, r1 = int((box[3] - lat0) / dla), int(np.ceil((box[2] - lat0) / dla))
    return slice(max(0, r0), min(shape[0], r1)), slice(max(0, c0), min(shape[1], c1))


def box_stats(d, sl, ps):
    F, m = d["F"][sl], d["mask"][sl] & ~d["win"][sl]
    obs = np.where(d["first"][sl] > 0, d["first"][sl], 31)
    r = {"n_new_px": int(((obs <= 30) & m).sum()), "n_win_px": int((d["win"][sl] & d["mask"][sl]).sum())}
    for t in (1, 3, 7, 14, 30):
        y = (obs <= t)[m]
        if 0 < y.sum() < y.size:
            r[f"auc_{t}"] = float(roc_auc_score(y, F[..., t - 1][m]))
    erm = (1 - F[..., :29]).sum(-1) + 1                                 # E[min(T, 30)] = sum_{t=0}^{29} S(t)
    sel = m & (obs <= 30)
    if sel.sum() > 10:
        r["spearman_expected_vs_obs"] = float(spearmanr(erm[sel], obs[sel]).statistic)
    for p in ps:
        a = arrival_pred(F, p)
        for t in (7, 30):
            pred, true = (a <= t) & m, (obs <= t) & m
            r[f"p{p}_t{t}_recall"] = float((pred & true).sum() / max(true.sum(), 1))
            r[f"p{p}_t{t}_precision"] = float((pred & true).sum() / max(pred.sum(), 1))
    return r


def scan_boxes(d, geo, n=3):
    lon0, dl, lat0, dla = geo
    w, h = int(BOX_DEG[0] / dl), int(BOX_DEG[1] / -dla)
    y = ((d["first"] > 0) & ~d["win"] & d["mask"]).astype(np.float64)
    cs = np.pad(y, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    cnt = cs[h:, w:] - cs[:-h, w:] - cs[h:, :-w] + cs[:-h, :-w]
    out = []
    for _ in range(n):
        r, c = np.unravel_index(np.argmax(cnt), cnt.shape)
        if cnt[r, c] <= 0:
            break
        out.append((round(lon0 + dl * c, 2), round(lon0 + dl * (c + w), 2), round(lat0 + dla * (r + h), 2), round(lat0 + dla * r, 2)))
        cnt[max(0, r - h):r + h, max(0, c - w):c + w] = -1
    return out


def pick_pixels(d, sl, p):
    """Drei Pixel fuer (d): frueh erreicht (1-3 d), spaet erreicht (8-30 d), nahe am Feuer ohne Ankunft."""
    F, m = d["F"][sl], d["mask"][sl] & ~d["win"][sl]
    obs = np.where(d["first"][sl] > 0, d["first"][sl], 31)
    near = binary_dilation(d["win"][sl], iterations=6)
    groups = (("early", m & (obs <= 3)), ("late", m & (obs >= 8) & (obs <= 30)), ("none", m & near & (obs > 30)))
    out = []
    for name, sel in groups:
        rr, cc = np.nonzero(sel)
        if rr.size == 0:
            continue
        v = F[rr, cc, -1]
        i = np.argsort(v)[v.size // 2]                                  # Median von F(30) in der Gruppe
        out.append((name, int(rr[i]), int(cc[i])))
    return out


def draw_base(ax, d, sl, extent, rings):
    m = d["mask"][sl]
    ax.imshow(np.where(m, 1, 0), extent=extent, cmap=ListedColormap([C_WATER, C_LAND]), vmin=0, vmax=1,
              interpolation="nearest", origin="upper")
    for ring in rings:
        ax.plot(ring[:, 0], ring[:, 1], color="#9aa0a6", lw=0.4)


def draw_arrival(ax, a, d, sl, extent, rings, cmap, norm, box):
    draw_base(ax, d, sl, extent, rings)
    m = d["mask"][sl] & ~d["win"][sl]
    ax.imshow(np.ma.masked_where(~m | (a > 30), a), extent=extent, cmap=cmap, norm=norm, interpolation="nearest",
              origin="upper")
    ax.imshow(np.ma.masked_where(~(d["win"][sl] & d["mask"][sl]), np.ones(a.shape)), extent=extent,
              cmap=ListedColormap([C_WIN]), interpolation="nearest", origin="upper")
    finish(ax, box)


def finish(ax, box):
    ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3])
    ax.set_aspect(1 / np.cos(np.deg2rad(0.5 * (box[2] + box[3]))))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4); s.set_color("#666666")


def plot(exp, tag, events, p, geo, rings, out, model_name, isochrones):
    cmap, norm = ListedColormap(COLORS), BoundaryNorm(BINS, len(COLORS))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
    n = len(events)
    fig, axes = plt.subplots(n, 4, figsize=(7.0, 1.95 * n + 0.45), dpi=300,
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.95], "wspace": 0.06, "hspace": 0.12})
    axes = np.atleast_2d(axes)
    stats = {}
    for i, (date, box) in enumerate(events):
        d = load(exp, tag, date)
        sl = box_slice(box, geo, d["mask"].shape)
        lon0, dl, lat0, dla = geo
        extent = (lon0 + dl * sl[1].start, lon0 + dl * sl[1].stop, lat0 + dla * sl[0].stop, lat0 + dla * sl[0].start)
        obs = np.where(d["first"][sl] > 0, d["first"][sl], 31)
        a = arrival_pred(d["F"][sl], p)
        st = box_stats(d, sl, [p])
        stats[f"{date}:{box}"] = st
        print(f"[teaser] {tag} {date} {box}: " + json.dumps({k: round(v, 3) for k, v in st.items()}), flush=True)

        ax = axes[i, 0]
        draw_base(ax, d, sl, extent, rings)
        ax.imshow(np.ma.masked_where(~(d["win"][sl] & d["mask"][sl]), np.ones(obs.shape)), extent=extent,
                  cmap=ListedColormap([C_WIN]), interpolation="nearest", origin="upper")
        finish(ax, box)
        ax.set_ylabel(f"{date}\n{abs(box[2]):.1f}–{abs(box[3]):.1f}°N, {abs(box[1]):.1f}–{abs(box[0]):.1f}°W", fontsize=6.5)
        act = ((d["first"][sl] > 0) | d["win"][sl]).astype(float)       # Lage in Kanada, Ecke mit wenig Feuer
        hh, ww = act.shape
        h4, w4 = int(0.3 * hh), int(0.4 * ww)
        corners = {(0.0, 0.0): act[-h4:, :w4], (0.64, 0.0): act[-h4:, -w4:], (0.0, 0.74): act[:h4, :w4], (0.64, 0.74): act[:h4, -w4:]}
        cx, cy = min(corners, key=lambda k: corners[k].sum())
        ins = ax.inset_axes([cx, cy, 0.36, 0.26])
        for ring in rings:
            ins.plot(ring[:, 0], ring[:, 1], color="#8a8f96", lw=0.15)
        ins.add_patch(plt.Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2], fill=False, ec="#c0392b", lw=0.8))
        ins.set_xlim(-141, -52); ins.set_ylim(41, 70); ins.set_aspect(1.8)
        ins.set_facecolor("white"); ins.set_xticks([]); ins.set_yticks([])
        for s in ins.spines.values():
            s.set_linewidth(0.3); s.set_color("#888888")
        draw_arrival(axes[i, 1], obs, d, sl, extent, rings, cmap, norm, box)
        draw_arrival(axes[i, 2], a, d, sl, extent, rings, cmap, norm, box)
        if isochrones:
            for t, ls in ((7, "-"), (30, "--")):
                axes[i, 1].contour(((obs <= t) | d["win"][sl]).astype(float), levels=[0.5], extent=extent, origin="upper",
                                   colors="#2b2b2b", linewidths=0.5, linestyles=ls)
        pix = pick_pixels(d, sl, p)
        ax = axes[i, 3]
        tt = np.arange(0, 31)
        for k, (name, r, c) in enumerate(pix):
            lab = "abc"[k]
            x, y = extent[0] + dl * (c + 0.5), extent[3] + dla * (r + 0.5)
            for axm in (axes[i, 1], axes[i, 2]):
                axm.plot(x, y, marker="o", ms=3.2, mfc="white", mec="#1b3a5c", mew=0.7, zorder=6)
                axm.annotate(lab, (x, y), xytext=(2.5, 2.5), textcoords="offset points", fontsize=6.5, color="#1b3a5c",
                             fontweight="bold", zorder=6)
            Fx = np.concatenate([[0.0], d["F"][sl][r, c]])
            col = ("#1b3a5c", "#3f7fbf", "#8fa8c0")[k]
            ax.plot(tt, Fx, color=col, lw=1.1, label=f"{lab}: observed " + (f"day {obs[r, c]}" if obs[r, c] <= 30 else "none ≤ 30 d"))
            if obs[r, c] <= 30:
                ax.plot(obs[r, c], Fx[obs[r, c]], marker="v", ms=4, color=col, zorder=5)
        ax.axhline(p, color="#999999", lw=0.5, ls=":")
        ax.set_xlim(0, 30); ax.set_ylim(0, None)
        ax.set_xticks([0, 7, 14, 30]); ax.tick_params(labelsize=6)
        ax.yaxis.tick_right(); ax.yaxis.set_label_position("right")
        ax.set_ylabel("F(τ)", fontsize=6.5)
        ax.legend(fontsize=5.3, loc="upper left", frameon=True, framealpha=0.85, edgecolor="none", handlelength=1.2)
        if i == n - 1:
            ax.set_xlabel("τ (days after forecast date)", fontsize=6.5)
        for s in ("top",):
            ax.spines[s].set_visible(False)
        ax.set_box_aspect(1)
    for j, t in enumerate(("(a) Input window", "(b) Observed arrival", f"(c) Predicted, F(τ) ≥ {p:g}",
                           f"(d) Pixel CIF F(τ)")):
        axes[0, j].set_title(t, fontsize=7, loc="left")
    fig.canvas.draw()
    p0, p2 = axes[-1, 0].get_position(), axes[-1, 2].get_position()
    cax = fig.add_axes([p0.x0 + 0.1 * (p2.x1 - p0.x0), p0.y0 - 0.075 / n, 0.8 * (p2.x1 - p0.x0), 0.022 / n])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                      ticks=[0.5 * (BINS[k] + BINS[k + 1]) for k in range(len(BIN_LABELS))])
    cb.ax.set_xticklabels(BIN_LABELS, fontsize=6)
    cb.set_label("days after the forecast date (black: burning within the input window)", fontsize=6.5)
    fig.savefig(f"{out}.png", bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    plt.close(fig)
    Path(f"{out}.json").write_text(json.dumps({"tag": tag, "p": p, "events": [[e[0], list(e[1])] for e in events],
                                               "stats": stats}, indent=1))
    print(f"[teaser] {out}.png / .pdf", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", required=True, help="Tag der risk_map-npz (risk_<tag>_<datum>.npz)")
    ap.add_argument("--dates", default="", help="Stichtage, durch Leerzeichen oder Komma getrennt")
    ap.add_argument("--ps", default="0.02,0.05,0.1,0.2,0.3")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scan")
    pp = sub.add_parser("plot")
    pp.add_argument("--event", action="append", default=[], help="datum:lon_min,lon_max,lat_min,lat_max")
    pp.add_argument("--p", type=float, default=0.05)
    pp.add_argument("--model-name", default="Contagion-STS")
    pp.add_argument("--isochrones", action="store_true")
    pp.add_argument("--out-name", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    exp = Path(cfg["experiment_root"])
    geo = grid_geo(cfg)
    rings = borders(exp / "artifacts" / "ne_50m_admin_1_states_provinces.geojson")
    out_dir = exp / "figures" / "teaser"
    out_dir.mkdir(parents=True, exist_ok=True)
    ps = [float(v) for v in args.ps.split(",")]
    args.dates = args.dates.replace(",", " ").split()
    if args.cmd == "scan":
        res = {}
        for date in args.dates:
            d = load(exp, args.tag, date)
            m = d["mask"] & ~d["win"]
            near = binary_dilation(d["win"], iterations=6) & m
            q = {f"q{qq}": float(np.nanquantile(d["F"][..., -1][near], qq / 100)) for qq in (50, 90, 99)}
            print(f"[scan] {args.tag} {date}: F(30) nahe am Feuer (<= 6 px) {q}", flush=True)
            res[date] = {"F30_near": q, "boxes": {}}
            for box in scan_boxes(d, geo):
                st = box_stats(d, box_slice(box, geo, d["mask"].shape), ps)
                res[date]["boxes"][str(box)] = st
                print(f"[scan]   {box}: " + json.dumps({k: round(v, 3) for k, v in st.items()}), flush=True)
        f = out_dir / f"scan_{args.tag}.json"
        if f.exists():
            f = out_dir / f"scan_{args.tag}_{len(list(out_dir.glob(f'scan_{args.tag}*.json')))}.json"
        f.write_text(json.dumps(res, indent=1))
        print(f"[scan] -> {f}", flush=True)
        return
    events = []
    for e in args.event:
        date, b = e.split(":")
        events.append((date, tuple(float(v) for v in b.split(","))))
    if not events:
        for date in args.dates:
            events.append((date, scan_boxes(load(exp, args.tag, date), geo, n=1)[0]))
    name = args.out_name or f"teaser_{args.tag}_p{args.p:g}"
    out = out_dir / name
    if Path(f"{out}.png").exists():
        raise SystemExit(f"ABBRUCH: {out}.png existiert, --out-name aendern")
    plot(exp, args.tag, events, args.p, geo, rings, out, args.model_name, args.isochrones)


if __name__ == "__main__":
    main()
