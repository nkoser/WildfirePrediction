"""Black Swans: Risikokarten je Ereignis, Version 2 (Rueckmeldung zu black_swan_maps: die Log-Skala faerbte auch
0,1 % deutlich ein). Nur CPU.

Spalten je Kartentag (Zeilen wie black_swan_maps):
  1 beobachtet: MODIS-Detektionen (fire_index) in den naechsten 7 Tagen bzw. Tag 8-30, bisher verbrannt grau
  2 vorhergesagt: F(tau) des Mittelpixel-Kopfs, Ensemble, kalibriert, LINEARE Skala 0..1, 0 = weiss; gleiche
    Skala ueber alle Zeilen und Ereignisse
  3 relatives Risiko: F(tau) / Bezug, Log-Skala, 1 = weiss. Bezug = mittleres kalibriertes F(tau) desselben
    Ensembles auf den zufaelligen Test-Stichtagen von surv_eval_test (2023-2025) in derselben Region
    (+-5 Grad Breite, +-10 Grad Laenge um den Ereignis-Schwerpunkt), Kalendertag +-15 d, ohne das Ereignisjahr
  4 beobachtet - vorhergesagt: erstes Feuer am Mittelpixel bis tau minus F(tau), Mittel ueber 5 x 5 Kartenpixel
Laengengrad als Grad West (positiv). Ausgabe NEU: figures/black_swan_v2<tag>/, Manifest p40_black_swan_maps_v2<tag>.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from scipy.ndimage import uniform_filter  # noqa: E402

from wildfire.audit.common import assert_readonly, load_config, read_manifest, write_manifest  # noqa: E402
from wildfire.paper.black_swan_dense import TAUS  # noqa: E402
from wildfire.paper.black_swan_maps import MODEL_LABEL, NAMES, slug  # noqa: E402
from wildfire.paper.extreme_eval import year_of  # noqa: E402
from wildfire.paper.surv_eval import _cum_hazard  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
LAND, BURNED, NEXT7, NEXT30 = "#f0efec", "#c3c2b7", "#d03b3b", "#eda100"
SEQ0 = ["#ffffff", "#fde2cf", "#f6b08a", "#eb6834", "#c4401a", "#8a2a12"]          # weiss -> orange -> dunkelrot
DIV = ["#1c5cab", "#6da7ec", "#cde2fb", "#ffffff", "#f6c4c3", "#e66767", "#b02c2c"]  # 0 bzw. 1 = weiss


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--events", type=int, nargs="*", default=[16, 12, 1, 0, 2, 5, 30])
    ap.add_argument("--models", nargs="+", default=["lh_ref", "cg_ref"])
    ap.add_argument("--taus", type=int, nargs="+", default=[7, 30])
    ap.add_argument("--set-tag", default="", help="Tag von black_swan_dense (Menge und F_cal)")
    ap.add_argument("--ref-dlat", type=float, default=5.0)
    ap.add_argument("--ref-dlon", type=float, default=10.0)
    ap.add_argument("--ref-ddoy", type=int, default=15)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    fig_dir = exp / "figures" / f"black_swan_v2{args.tag}"
    if fig_dir.exists():
        raise SystemExit(f"ABBRUCH: {fig_dir} existiert schon - anderen --tag waehlen")

    plt.rcParams.update({"font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 9,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    font_used = Path(font_manager.findfont(font_manager.FontProperties(family=plt.rcParams["font.family"]))).name

    st = args.set_tag
    S = dict(np.load(exp / "derived_data" / f"black_swan{st}" / "set.npz"))
    sman = read_manifest(cfg, f"p40_black_swan_set{st}")
    info = {e["id"]: e for e in sman["events"]}
    kmap = int(sman["kmap"])
    calib = read_manifest(cfg, f"p40_black_swan_dense{st}")["calib_ab"]
    mdir = exp / "metrics" / f"black_swan_dense{st}"
    tr = json.loads((exp / "data_manifests" / "d0_masks.json").read_text())["grid"]["transform"]
    lat_of = lambda r: tr[5] + tr[4] * (np.asarray(r, float) + 0.5)  # noqa: E731
    lon_of = lambda c: tr[2] + tr[0] * (np.asarray(c, float) + 0.5)  # noqa: E731
    fi = np.load(exp / "derived_data" / "events" / "fire_index.npz", allow_pickle=False)
    W = int(fi["grid"][1])
    flat = np.repeat(fi["fire_pixels"].astype(np.int64), np.diff(fi["fire_offsets"]))
    fr, fc, fd = flat // W, flat % W, fi["fire_days"].astype(np.int64)
    dates = [str(d)[:10] for d in fi["dates"]]
    doy_of = lambda L: dt.date.fromisoformat(dates[L]).timetuple().tm_yday  # noqa: E731
    land = np.load(exp / "derived_data" / "masks" / "land.npy") & np.load(exp / "derived_data" / "masks" / "canada.npy")
    tm = dict(np.load(exp / "derived_data" / "surv_eval_test" / "eval_meta.npz"))
    t_lat, t_lon, t_yr, t_doy = lat_of(tm["row"]), lon_of(tm["col"]), year_of(tm["L_abs"]), tm["doy"]

    # Wertebereich der Vorhersage ueber alle Karten (gleiche lineare Skala ueber Ereignisse)
    seq = mcolors.LinearSegmentedColormap.from_list("seq0", SEQ0)
    seq.set_bad(SURFACE)
    div = mcolors.LinearSegmentedColormap.from_list("div", DIV)
    div.set_bad(SURFACE)
    fig_dir.mkdir(parents=True)
    written, notes = [], {}

    for name in args.models:
        Fp = mdir / f"F_cal_{name}_centre.npy"
        if not Fp.exists():
            print(f"[bsm2] fehlt: {Fp.name}", flush=True)
            continue
        F = np.load(Fp)
        ens = np.load(exp / "derived_data" / "calib_cache_ext" / f"ens_{name}.npz")
        a, b = calib[f"{name}_centre"]
        lam_t = np.exp(a) * np.power(np.maximum(_cum_hazard(ens["test_centre"].astype(np.float64)), 1e-300), b)
        for eid in args.events:
            if eid not in info:
                print(f"[bsm2] Ereignis {eid} nicht in der Menge", flush=True)
                continue
            e = info[eid]
            r0, r1, c0, c1 = e["map_box"]
            days = e["map_days"]
            if not days:
                continue
            ext = [tr[2] + tr[0] * c0, tr[2] + tr[0] * c1, tr[5] + tr[4] * r1, tr[5] + tr[4] * r0]
            aspect = 1.0 / np.cos(np.radians(0.5 * (ext[2] + ext[3])))
            yr = int(dates[days[0]][:4])
            in_box = (fr >= r0) & (fr < r1) & (fc >= c0) & (fc < c1)
            season0 = min(i for i, d in enumerate(dates) if d.startswith(str(yr)))
            reg = (np.abs(t_lat - e["lat"]) <= args.ref_dlat) & (np.abs(t_lon - e["lon"]) <= args.ref_dlon) & (t_yr != yr)
            for tau in args.taus:
                j = TAUS.index(tau)
                Ft = -np.expm1(-lam_t[:, tau - 1])
                nr, nc = (r1 - r0 + kmap - 1) // kmap, (c1 - c0 + kmap - 1) // kmap
                preds, diffs, rrs, refs = [], [], [], []
                for L in days:
                    m = (S["kind"] == 1) & (S["event"] == eid) & (S["L"] == L)
                    gi, gj = (S["row"][m] - r0) // kmap, (S["col"][m] - c0) // kmap
                    P = np.full((nr, nc), np.nan)
                    P[gi, gj] = F[m, j]
                    O = np.full((nr, nc), np.nan)
                    O[gi, gj] = (S["ev_centre"][m] & (S["t_centre"][m] <= tau)).astype(float)
                    D = O - P
                    ok = np.isfinite(D)
                    num = uniform_filter(np.where(ok, D, 0.0), 5, mode="constant")
                    den = uniform_filter(ok.astype(float), 5, mode="constant")
                    d0 = doy_of(L)
                    rm = reg & (np.abs(t_doy - d0) <= args.ref_ddoy)
                    ref = float(Ft[rm].mean()) if rm.sum() >= 50 else float("nan")
                    preds.append(P); diffs.append(np.where(ok, num / np.maximum(den, 1e-9), np.nan))
                    rrs.append(P / ref); refs.append({"date": dates[L], "ref": ref, "n_ref": int(rm.sum())})
                allp = np.concatenate([p[np.isfinite(p)] for p in preds])
                med, mx = float(np.median(allp)), float(allp.max())
                lr = np.log10(np.concatenate([r[np.isfinite(r) & (r > 0)] for r in rrs]))
                rl = float(np.clip(np.quantile(np.abs(lr), 0.99), 0.5, 3.0)) if lr.size else 1.0
                dl = max(float(np.nanquantile(np.abs(np.concatenate([d.ravel() for d in diffs])), 0.99)), 1e-3)
                fig, axs = plt.subplots(len(days), 4, figsize=(13.5, 2.9 * len(days) + 1.2), squeeze=False,
                                        constrained_layout=True)
                for k, L in enumerate(days):
                    obs = np.zeros((r1 - r0, c1 - c0), np.int8)
                    obs[~land[r0:r1, c0:c1]] = -1
                    sel = in_box & (fd >= season0) & (fd <= L)
                    obs[fr[sel] - r0, fc[sel] - c0] = 1
                    sel = in_box & (fd > L + 7) & (fd <= L + 30)
                    obs[fr[sel] - r0, fc[sel] - c0] = 3
                    sel = in_box & (fd > L) & (fd <= L + 7)
                    obs[fr[sel] - r0, fc[sel] - c0] = 2
                    cmap_o = mcolors.ListedColormap([SURFACE, LAND, BURNED, NEXT7, NEXT30])
                    axs[k, 0].imshow(obs + 1, cmap=cmap_o, vmin=-0.5, vmax=4.5, extent=ext, interpolation="nearest")
                    im1 = axs[k, 1].imshow(preds[k], cmap=seq, vmin=0.0, vmax=1.0, extent=ext, interpolation="nearest")
                    with np.errstate(divide="ignore", invalid="ignore"):
                        lrr = np.log10(rrs[k])
                    im2 = axs[k, 2].imshow(lrr, cmap=div, norm=mcolors.TwoSlopeNorm(0, -rl, rl), extent=ext,
                                           interpolation="nearest")
                    im3 = axs[k, 3].imshow(diffs[k], cmap=div, norm=mcolors.TwoSlopeNorm(0, -dl, dl), extent=ext,
                                           interpolation="nearest")
                    rel = L - e["start_day"]
                    lab = "Onset" if rel == 0 else f"Onset {'+' if rel > 0 else '−'} {abs(rel)} d"
                    axs[k, 0].set_ylabel(f"{lab}\n{dates[L]}\nLatitude (°N)", color=INK)
                    for a_ in axs[k]:
                        a_.set_aspect(aspect)
                        a_.tick_params(labelsize=7, length=2)
                        a_.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):g}"))
                        for sp in a_.spines.values():
                            sp.set_color(GRID)
                    for a_ in axs[k, 1:]:
                        a_.set_yticklabels([])
                    if k == len(days) - 1:
                        for a_ in axs[k]:
                            a_.set_xlabel("Longitude (°W)")
                axs[0, 0].set_title("Observed MODIS Detections", fontsize=10)
                axs[0, 1].set_title(f"Predicted {tau}-Day Fire Probability", fontsize=10)
                axs[0, 2].set_title("Relative Risk vs. Regional Test Mean", fontsize=10)
                axs[0, 3].set_title(f"Observed − Predicted ({tau} Days, 5×5 Mean)", fontsize=10)
                axs[-1, 0].legend(handles=[Patch(color=NEXT7, label="Fire in Next 7 Days"),
                                           Patch(color=NEXT30, label="Fire in Days 8–30"),
                                           Patch(color=BURNED, label="Burned Earlier This Season")],
                                  loc="upper center", bbox_to_anchor=(0.5, -0.32), frameon=False, fontsize=8)
                cb1 = fig.colorbar(im1, ax=axs[:, 1], location="bottom", shrink=0.9, pad=0.02)
                cb1.set_label(f"F{tau}, Centre-Pixel Head, Calibrated (Linear)")
                cb2 = fig.colorbar(im2, ax=axs[:, 2], location="bottom", shrink=0.9, pad=0.02)
                cb2.set_ticks([np.log10(v) for v in (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100)
                               if abs(np.log10(v)) <= rl + 1e-9])
                cb2.ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{10 ** v:g}×"))
                cb2.set_label("Relative Risk (Log Scale, 1× = White)")
                cb3 = fig.colorbar(im3, ax=axs[:, 3], location="bottom", shrink=0.9, pad=0.02)
                cb3.set_label("Red: More Fire Than Predicted")
                ttl = NAMES.get(eid, f"Event {eid}")
                ctrl = " (Training-Year Control)" if e.get("control") else ""
                fig.suptitle(f"{ttl}{ctrl}: Observed Fire vs. {MODEL_LABEL.get(name, name)} Ensemble\n"
                             f"Predicted F{tau} (Calibrated): Median {100 * med:.2g} %, Max {100 * mx:.2g} %. "
                             f"Relative Risk vs. Mean F{tau} on Random 2023–2025 Test Landmarks Within ±{args.ref_dlat:g}° Lat, "
                             f"±{args.ref_dlon:g}° Lon, ±{args.ref_ddoy} Days, Other Years", fontsize=10.5)
                base = fig_dir / f"bs_{eid}_{slug(ttl)}_f{tau}_{name}"
                for ext_ in ("png", "pdf"):
                    fig.savefig(f"{base}.{ext_}", dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
                plt.close(fig)
                written.append(str(base))
                notes[f"{eid}_{tau}_{name}"] = {"median": med, "max": mx, "refs": refs}
                print(f"[bsm2] {base.name}: F{tau} Median {med:.4f} Max {mx:.3f}; Bezug "
                      + ", ".join(f"{r['date']} {r['ref']:.4f} (n {r['n_ref']})" for r in refs), flush=True)
    write_manifest(cfg, f"p40_black_swan_maps_v2{args.tag}", args.config, {
        "figures": written, "models": args.models, "taus": args.taus, "font": font_used, "kmap": kmap,
        "set_tag": st, "reference": {"dlat": args.ref_dlat, "dlon": args.ref_dlon, "ddoy": args.ref_ddoy,
                                     "pool": "surv_eval_test eval, Ereignisjahr ausgeschlossen"}, "ranges": notes})


if __name__ == "__main__":
    main()
