"""Black Swans: Risikokarten je Ereignis, Version 3 = v2 (black_swan_maps_v2) zusaetzlich OHNE Kalibrierung. Nur CPU.

Spalten je Kartentag: beobachtet | vorhergesagt roh | vorhergesagt kalibriert | beobachtet - roh |
beobachtet - kalibriert. Roh = Ensemble der gespeicherten Hazards aus black_swan_score (score_<modell>.npz, Kopf-
Bias wie im Training, keine Nachkalibrierung); kalibriert = Lambda' = e^a Lambda^b mit a, b aus p40_black_swan_dense
(Val-Anpassung von calib_cache_ext/ens_<modell>). Vorhersage linear 0..1 (0 = weiss), Differenzen +-0.75, alle
Skalen gleich ueber Zeilen und Ereignisse. Dazu oe_by_event.csv: O/E@tau des Mittelpixels ueber die Kartenpixel,
vorher (Stichtag vor Beginn) und waehrend, roh und kalibriert.
Ausgabe NEU: figures/black_swan_v3<tag>/, Manifest p40_black_swan_maps_v3<tag>.
"""

from __future__ import annotations

import argparse
import csv
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
from wildfire.paper.black_swan_dense import TAUS, ens_F  # noqa: E402
from wildfire.paper.black_swan_maps import MODEL_LABEL, NAMES, slug  # noqa: E402
from wildfire.paper.black_swan_maps_v2 import BURNED, DIV, GRID, INK, LAND, MUTED, NEXT7, NEXT30, SEQ0, SURFACE  # noqa: E402

DLIM = 0.75


def smooth_diff(O, P):
    D = O - P
    ok = np.isfinite(D)
    num = uniform_filter(np.where(ok, D, 0.0), 5, mode="constant")
    den = uniform_filter(ok.astype(float), 5, mode="constant")
    return np.where(ok, num / np.maximum(den, 1e-9), np.nan)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--events", type=int, nargs="*", default=[16, 12, 1, 0, 2, 5, 30])
    ap.add_argument("--models", nargs="+", default=["lh_ref", "cg_ref"])
    ap.add_argument("--taus", type=int, nargs="+", default=[7, 30])
    ap.add_argument("--set-tag", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    fig_dir = exp / "figures" / f"black_swan_v3{args.tag}"
    if fig_dir.exists():
        raise SystemExit(f"ABBRUCH: {fig_dir} existiert schon - anderen --tag waehlen")

    plt.rcParams.update({"font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 9,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    font_used = Path(font_manager.findfont(font_manager.FontProperties(family=plt.rcParams["font.family"]))).name

    st = args.set_tag
    dd = exp / "derived_data" / f"black_swan{st}"
    S = dict(np.load(dd / "set.npz"))
    sman = read_manifest(cfg, f"p40_black_swan_set{st}")
    info = {e["id"]: e for e in sman["events"]}
    kmap = int(sman["kmap"])
    calib = read_manifest(cfg, f"p40_black_swan_dense{st}")["calib_ab"]
    tr = json.loads((exp / "data_manifests" / "d0_masks.json").read_text())["grid"]["transform"]
    fi = np.load(exp / "derived_data" / "events" / "fire_index.npz", allow_pickle=False)
    W = int(fi["grid"][1])
    flat = np.repeat(fi["fire_pixels"].astype(np.int64), np.diff(fi["fire_offsets"]))
    fr, fc, fd = flat // W, flat % W, fi["fire_days"].astype(np.int64)
    dates = [str(d)[:10] for d in fi["dates"]]
    land = np.load(exp / "derived_data" / "masks" / "land.npy") & np.load(exp / "derived_data" / "masks" / "canada.npy")
    seq = mcolors.LinearSegmentedColormap.from_list("seq0", SEQ0)
    seq.set_bad(SURFACE)
    div = mcolors.LinearSegmentedColormap.from_list("div", DIV)
    div.set_bad(SURFACE)
    fig_dir.mkdir(parents=True)
    written, oe_rows, ranges = [], [], {}
    mapm = S["kind"] == 1

    for name in args.models:
        sp = dd / f"score_{name}.npz"
        if not sp.exists():
            print(f"[bsm3] fehlt: {sp.name}", flush=True)
            continue
        lam = np.load(sp)["lam_centre"][:, mapm]                       # (seeds, n_map, taus)
        Fr_m, Fc_m = ens_F(lam, *calib[f"{name}_centre"])
        Fr, Fc = np.full((S["L"].size, len(TAUS)), np.nan), np.full((S["L"].size, len(TAUS)), np.nan)
        Fr[mapm], Fc[mapm] = Fr_m, Fc_m
        for eid in args.events:
            if eid not in info:
                print(f"[bsm3] Ereignis {eid} nicht in der Menge", flush=True)
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
            em = mapm & (S["event"] == eid)
            for tau in args.taus:
                j = TAUS.index(tau)
                obs_t = S["ev_centre"] & (S["t_centre"] <= tau)
                for w, wm in (("vorher", S["L"] < e["start_day"]), ("waehrend", S["L"] >= e["start_day"])):
                    m = em & wm
                    o = int(obs_t[m].sum())
                    oe_rows.append({"model": name, "event": eid, "name": NAMES.get(eid, ""), "tau": tau, "window": w,
                                    "n": int(m.sum()), "obs": o,
                                    "exp_raw": float(Fr[m, j].sum()), "exp_cal": float(Fc[m, j].sum()),
                                    "oe_raw": o / Fr[m, j].sum() if m.any() and Fr[m, j].sum() > 0 else float("nan"),
                                    "oe_cal": o / Fc[m, j].sum() if m.any() and Fc[m, j].sum() > 0 else float("nan")})
                nr, nc = (r1 - r0 + kmap - 1) // kmap, (c1 - c0 + kmap - 1) // kmap
                grids = []
                for L in days:
                    m = em & (S["L"] == L)
                    gi, gj = (S["row"][m] - r0) // kmap, (S["col"][m] - c0) // kmap
                    G = {}
                    for key, val in (("raw", Fr[m, j]), ("cal", Fc[m, j]), ("obs", obs_t[m].astype(float))):
                        A = np.full((nr, nc), np.nan)
                        A[gi, gj] = val
                        G[key] = A
                    G["d_raw"], G["d_cal"] = smooth_diff(G["obs"], G["raw"]), smooth_diff(G["obs"], G["cal"])
                    grids.append(G)
                st_ = {v: np.concatenate([g[v][np.isfinite(g[v])] for g in grids]) for v in ("raw", "cal")}
                rng = {v: (float(np.median(st_[v])), float(st_[v].max())) for v in st_}
                ranges[f"{eid}_{tau}_{name}"] = rng
                fig, axs = plt.subplots(len(days), 5, figsize=(16.5, 2.9 * len(days) + 1.2), squeeze=False,
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
                    im1 = axs[k, 1].imshow(grids[k]["raw"], cmap=seq, vmin=0, vmax=1, extent=ext, interpolation="nearest")
                    axs[k, 2].imshow(grids[k]["cal"], cmap=seq, vmin=0, vmax=1, extent=ext, interpolation="nearest")
                    norm = mcolors.TwoSlopeNorm(0, -DLIM, DLIM)
                    im3 = axs[k, 3].imshow(grids[k]["d_raw"], cmap=div, norm=norm, extent=ext, interpolation="nearest")
                    axs[k, 4].imshow(grids[k]["d_cal"], cmap=div, norm=norm, extent=ext, interpolation="nearest")
                    rel = L - e["start_day"]
                    lab = "Onset" if rel == 0 else f"Onset {'+' if rel > 0 else '−'} {abs(rel)} d"
                    axs[k, 0].set_ylabel(f"{lab}\n{dates[L]}\nLatitude (°N)", color=INK)
                    for a_ in axs[k]:
                        a_.set_aspect(aspect)
                        a_.tick_params(labelsize=7, length=2)
                        a_.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):g}"))
                        for sp_ in a_.spines.values():
                            sp_.set_color(GRID)
                    for a_ in axs[k, 1:]:
                        a_.set_yticklabels([])
                    if k == len(days) - 1:
                        for a_ in axs[k]:
                            a_.set_xlabel("Longitude (°W)")
                axs[0, 0].set_title("Observed MODIS Detections", fontsize=10)
                axs[0, 1].set_title(f"Predicted F{tau}, Raw", fontsize=10)
                axs[0, 2].set_title(f"Predicted F{tau}, Calibrated", fontsize=10)
                axs[0, 3].set_title("Observed − Raw (5×5 Mean)", fontsize=10)
                axs[0, 4].set_title("Observed − Calibrated (5×5 Mean)", fontsize=10)
                axs[-1, 0].legend(handles=[Patch(color=NEXT7, label="Fire in Next 7 Days"),
                                           Patch(color=NEXT30, label="Fire in Days 8–30"),
                                           Patch(color=BURNED, label="Burned Earlier This Season")],
                                  loc="upper center", bbox_to_anchor=(0.5, -0.32), frameon=False, fontsize=8)
                cb1 = fig.colorbar(im1, ax=axs[:, 1:3], location="bottom", shrink=0.8, pad=0.02)
                cb1.set_label(f"F{tau}, Centre-Pixel Head (Linear, Same Scale for Raw and Calibrated)")
                cb3 = fig.colorbar(im3, ax=axs[:, 3:5], location="bottom", shrink=0.8, pad=0.02)
                cb3.set_label(f"Observed Fire Within {tau} Days − Predicted F{tau}; Red: More Fire Than Predicted")
                oe = {w: next(r for r in oe_rows if r["model"] == name and r["event"] == eid and r["tau"] == tau
                              and r["window"] == w) for w in ("vorher", "waehrend")}
                f2 = lambda x: "–" if x != x else f"{x:.2f}"  # noqa: E731
                ttl = NAMES.get(eid, f"Event {eid}")
                ctrl = " (Training-Year Control)" if e.get("control") else ""
                fig.suptitle(
                    f"{ttl}{ctrl}: Observed Fire vs. {MODEL_LABEL.get(name, name)} Ensemble\n"
                    f"Raw (Training Head Bias, No Recalibration): Median {100 * rng['raw'][0]:.2g} %, Max {100 * rng['raw'][1]:.2g} %, "
                    f"O/E@{tau} Before / During Onset {f2(oe['vorher']['oe_raw'])} / {f2(oe['waehrend']['oe_raw'])}.  "
                    f"Calibrated: Median {100 * rng['cal'][0]:.2g} %, Max {100 * rng['cal'][1]:.2g} %, "
                    f"O/E@{tau} {f2(oe['vorher']['oe_cal'])} / {f2(oe['waehrend']['oe_cal'])}", fontsize=10.5)
                base = fig_dir / f"bs_{eid}_{slug(ttl)}_f{tau}_{name}"
                for ext_ in ("png", "pdf"):
                    fig.savefig(f"{base}.{ext_}", dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
                plt.close(fig)
                written.append(str(base))
                print(f"[bsm3] {base.name}: roh Median {rng['raw'][0]:.4f} Max {rng['raw'][1]:.3f}, kal. Median "
                      f"{rng['cal'][0]:.4f} Max {rng['cal'][1]:.3f}; O/E vorher {f2(oe['vorher']['oe_raw'])} / "
                      f"{f2(oe['vorher']['oe_cal'])}, waehrend {f2(oe['waehrend']['oe_raw'])} / {f2(oe['waehrend']['oe_cal'])}",
                      flush=True)
    with open(fig_dir / "oe_by_event.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(oe_rows[0]))
        w.writeheader(); w.writerows(oe_rows)
    write_manifest(cfg, f"p40_black_swan_maps_v3{args.tag}", args.config, {
        "figures": written, "models": args.models, "taus": args.taus, "font": font_used, "kmap": kmap,
        "set_tag": st, "raw": "score_<modell>.npz, Ensemble, Kopf-Bias wie im Training, keine Nachkalibrierung",
        "calib_ab": calib, "ranges": ranges, "oe": oe_rows})


if __name__ == "__main__":
    main()
