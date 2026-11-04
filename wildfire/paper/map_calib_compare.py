"""Pixelkarten mehrerer Modelle, roh gegen kalibriert, an den Teaser-Ereignissen.

Liest die npz von risk_map (risk_<tag>_<datum>.npz, --taus 1,...,30) und je Modell die Hazards der
Mittelzelle auf den Val-Stichtagen 2020-2022 aus calib_eval (derived_data/calib_cache/<datei>.npz,
Schluessel val_centre). Darauf werden die drei Kalibrierer aus calib_eval angepasst und punktweise auf
die Karten-CIF angewandt; alle drei sind je tau monoton in F und gleich fuer alle tau, die Karten
bleiben also monoton in tau und die Rangfolge der Pixel je tau bleibt gleich (ROC-AUC unveraendert):

  raw    F wie trainiert
  curve  Lambda' = e^a Lambda^b mit Lambda = -log(1 - F)  (surv_eval.calib_fit, Survival-Likelihood)
  pit    F' = G(F), G = Verteilung der Val-PIT-Werte     (calib_eval.pit_fit, CiPOT-Prinzip)
  ks     F' = sigma(a logit F + b)                      (calib_eval.ks_fit, KSP-Prinzip)

Modelle ohne Cache (etwa das Wetter-Oracle, dessen Val-Eingaben fehlen) gehen nur roh ein.

Kennzahlen je Ereignis-Ausschnitt und fuer ganz Kanada, nur Landpixel, die im Eingabefenster nicht
brannten: ROC-AUC von F(tau), beobachtet / erwartet (O/E) und Brier fuer "Feuer bis tau", Trefferquote
und Praezision der Isochrone F(tau) >= p, je tau = 7 und 30.

Ausgabe (neue Ordner, nichts wird ueberschrieben):
  metrics/map_calib<tag>/summary.md, summary.json
  figures/map_calib<tag>/map_calib_<datum>.png/.pdf  Zeilen = Modelle, Spalten = beobachtet, roh, curve, pit, ks
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
from sklearn.metrics import roc_auc_score  # noqa: E402

from wildfire.audit.common import load_config, write_manifest  # noqa: E402
from wildfire.eval.metrics import cif_from_hazard  # noqa: E402
from wildfire.paper.calib_eval import ks_fit, ks_map, pit_apply, pit_fit  # noqa: E402
from wildfire.paper.risk_map_plot import borders, grid_geo  # noqa: E402
from wildfire.paper.surv_eval import calib_fit  # noqa: E402
from wildfire.paper.teaser_spread import (BIN_LABELS, BINS, COLORS, arrival_pred, box_slice, draw_arrival,  # noqa: E402
                                          load)

VARIANTS = ("raw", "curve", "pit", "ks")
TAUS = (7, 30)


def fit_calibrators(cache: Path, va: dict) -> dict:
    """Kalibrierer auf den Val-Stichtagen (Mittelzelle) wie calib_eval.analyse."""
    hv = np.load(cache)["val_centre"].astype(np.float64)
    ev, tv = va["ev_centre"].astype(bool), va["t_centre"].astype(int)
    Fv = cif_from_hazard(hv)
    a, b = calib_fit(hv, ev, tv)
    u, G = pit_fit(Fv, ev, tv)
    ka, kb = ks_fit(Fv, ev, tv)
    return {"curve": (a, b), "pit": (u, G), "ks": (ka, kb)}


def apply(F: np.ndarray, name: str, cal: dict) -> np.ndarray:
    if name == "raw":
        return F
    if name == "curve":
        a, b = cal["curve"]
        lam = -np.log1p(-np.clip(F, 0.0, 1 - 1e-7))
        return -np.expm1(-np.exp(a) * np.power(np.maximum(lam, 1e-300), b))
    if name == "pit":
        return pit_apply(F, *cal["pit"])
    return ks_map(F, *cal["ks"])


def region_stats(F: np.ndarray, obs: np.ndarray, m: np.ndarray, p: float) -> dict:
    """F (h, w, 30), obs = erster Feuertag (31 = keiner), m = bewertete Pixel."""
    r = {"n_px": int(m.sum())}
    a = arrival_pred(F, p)
    for t in TAUS:
        y, f = (obs <= t)[m], F[..., t - 1][m]
        r[f"obs_{t}"] = float(y.mean())
        r[f"pred_{t}"] = float(f.mean())
        r[f"o_e_{t}"] = float(y.mean() / max(f.mean(), 1e-12))
        r[f"brier_{t}"] = float(np.mean((f - y) ** 2))
        r[f"auc_{t}"] = float(roc_auc_score(y, f)) if 0 < y.sum() < y.size else None
        pred, true = (a <= t) & m, (obs <= t) & m
        r[f"recall_{t}"] = float((pred & true).sum() / max(true.sum(), 1))
        r[f"precision_{t}"] = float((pred & true).sum() / max(pred.sum(), 1))
        r[f"area_{t}"] = int(pred.sum())
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="+", required=True,
                    help="tag=Name[=calib_cache-Datei], z. B. teaser_lh=STS-LH=sts_teasermap_lh_sts")
    ap.add_argument("--events", default="/data/Wildfire/figures/teaser/teaser_v1_cgs.json")
    ap.add_argument("--surv-eval", default="surv_eval_test")
    ap.add_argument("--p", type=float, default=0.05)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    exp = Path(cfg["experiment_root"])
    out_m, out_f = exp / "metrics" / f"map_calib{args.tag}", exp / "figures" / f"map_calib{args.tag}"
    for o in (out_m, out_f):
        if o.exists():
            raise SystemExit(f"ABBRUCH: {o} existiert")
    geo = grid_geo(cfg)
    rings = borders(exp / "artifacts" / "ne_50m_admin_1_states_provinces.geojson")
    events = [(d, tuple(b)) for d, b in json.loads(Path(args.events).read_text())["events"]]
    va = dict(np.load(exp / "derived_data" / args.surv_eval / "val_meta.npz"))

    models, cals = [], {}
    for spec in args.models:
        parts = spec.split("=")
        tag, name = parts[0], parts[1]
        models.append((tag, name))
        if len(parts) > 2:
            cals[tag] = fit_calibrators(exp / "derived_data" / "calib_cache" / f"{parts[2]}.npz", va)
            c = cals[tag]
            print(f"[mcal] {name}: curve a={c['curve'][0]:.3f} b={c['curve'][1]:.3f}, ks a={c['ks'][0]:.3f} "
                  f"b={c['ks'][1]:.3f}, pit G(0.05)={np.interp(0.05, *c['pit']):.4f}", flush=True)

    out_m.mkdir(parents=True)
    out_f.mkdir(parents=True)
    cmap, norm = ListedColormap(COLORS), BoundaryNorm(BINS, len(COLORS))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
    res = {}
    for date, box in events:
        fig, axes = plt.subplots(len(models), 1 + len(VARIANTS), figsize=(1.55 * (1 + len(VARIANTS)), 1.5 * len(models) + 0.5),
                                 dpi=250, gridspec_kw={"wspace": 0.04, "hspace": 0.08})
        for i, (tag, name) in enumerate(models):
            d = load(exp, tag, date)
            sl = box_slice(box, geo, d["mask"].shape)
            lon0, dl, lat0, dla = geo
            extent = (lon0 + dl * sl[1].start, lon0 + dl * sl[1].stop, lat0 + dla * sl[0].stop, lat0 + dla * sl[0].start)
            obs_all = np.where(d["first"] > 0, d["first"], 31)
            m_all = d["mask"] & ~d["win"]
            obs = obs_all[sl]
            draw_arrival(axes[i, 0], obs, d, sl, extent, rings, cmap, norm, box)
            axes[i, 0].set_ylabel(name, fontsize=7)
            for j, v in enumerate(VARIANTS):
                ax = axes[i, 1 + j]
                if v != "raw" and tag not in cals:
                    ax.set_axis_off()
                    continue
                Fc = apply(d["F"], v, cals.get(tag))
                draw_arrival(ax, arrival_pred(Fc[sl], args.p), d, sl, extent, rings, cmap, norm, box)
                for reg, (FF, oo, mm) in {"event": (Fc[sl], obs, m_all[sl]), "canada": (Fc, obs_all, m_all)}.items():
                    st = region_stats(FF, oo, mm, args.p)
                    res.setdefault(date, {}).setdefault(reg, {}).setdefault(name, {})[v] = st
                e = res[date]["event"][name][v]
                print(f"[mcal] {date} {name:22s} {v:5s} AUC30 {e['auc_30']:.3f}  O/E7 {e['o_e_7']:.2f}  O/E30 "
                      f"{e['o_e_30']:.2f}  R/P7 {e['recall_7']:.2f}/{e['precision_7']:.2f}  R/P30 "
                      f"{e['recall_30']:.2f}/{e['precision_30']:.2f}", flush=True)
        for j, t in enumerate(("Observed", "Raw", "Curve", "PIT", "KS")):
            axes[0, j].set_title(t, fontsize=7.5)
        cax = fig.add_axes([0.2, 0.06, 0.6, 0.012])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                          ticks=[0.5 * (BINS[k] + BINS[k + 1]) for k in range(len(BIN_LABELS))])
        cb.ax.set_xticklabels(BIN_LABELS, fontsize=6)
        cb.set_label(f"days after the forecast date; predicted = first τ with F(τ) ≥ {args.p:g}; black: burning in the input window",
                     fontsize=6.5)
        fig.subplots_adjust(bottom=0.1)
        for ext in ("png", "pdf"):
            fig.savefig(out_f / f"map_calib_{date}.{ext}", bbox_inches="tight")
        plt.close(fig)

    cal_out = {tag: {"curve_ab": list(c["curve"]), "ks_ab": list(c["ks"]),
                     "pit_G": {"u": c["pit"][0][::100].tolist(), "G": c["pit"][1][::100].tolist()}} for tag, c in cals.items()}
    (out_m / "summary.json").write_text(json.dumps({"p": args.p, "events": [[d, list(b)] for d, b in events],
                                                    "calibrators": cal_out, "results": res}, indent=1))
    lines = [f"# Pixel maps: raw vs calibrated (threshold p = {args.p:g})", "",
             "Pixels: land, not burning in the 30 input days. AUC is the same for all variants (monotone map per τ).", ""]
    for date, box in events:
        for reg in ("event", "canada"):
            lines += [f"## {date} {'event region ' + str(list(box)) if reg == 'event' else 'all of Canada'}", "",
                      "| Model | Variant | AUC 7 / 30 | Obs 7 / 30 | Pred 7 / 30 | O/E 7 / 30 | Brier 7 / 30 | Recall 7 / 30 | Precision 7 / 30 |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for name, vs in res[date][reg].items():
                for v, s in vs.items():
                    f = lambda k, fmt: " / ".join("–" if s[f"{k}_{t}"] is None else format(s[f"{k}_{t}"], fmt) for t in TAUS)  # noqa: E731
                    lines.append(f"| {name} | {v} | {f('auc', '.3f')} | {f('obs', '.4f')} | {f('pred', '.4f')} | "
                                 f"{f('o_e', '.2f')} | {f('brier', '.4f')} | {f('recall', '.2f')} | {f('precision', '.2f')} |")
            lines.append("")
    (out_m / "summary.md").write_text("\n".join(lines))
    write_manifest(cfg, f"p22_map_calib{args.tag}", args.config, {
        "models": args.models, "events": args.events, "p": args.p, "outputs": [str(out_m), str(out_f)],
        "note": "Kalibrierer auf Val-Stichtagen 2020-2022 (Mittelzelle) angepasst, punktweise auf die Karten-CIF angewandt",
    })
    print(f"[mcal] {out_m / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
