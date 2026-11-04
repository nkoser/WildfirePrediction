"""Black Swans: Risikokarten je Ereignis. Nur CPU.

Je Ereignis eine Abbildung, Zeilen = Kartentage aus black_swan_dense (Beginn + map_offsets), Spalten:
  1 beobachtet: MODIS-Detektionen (fire_index) in den naechsten 7 Tagen bzw. Tag 8-30, bisher in der Saison
    verbrannt grau
  2 vorhergesagt: F7 bzw. F30 des Mittelpixel-Kopfs je Kartenpixel (jedes kmap-te Pixel), Ensemble, kalibriert
    (F_cal_<modell>_centre.npy aus black_swan_dense --part eval), gleiche Log-Farbskala ueber alle Zeilen
  3 beobachtet - vorhergesagt: erstes Feuer am Mittelpixel bis tau (0/1) minus F(tau), Mittel ueber 5 x 5
    Kartenpixel; rot = mehr Feuer als vorhergesagt (Unterschaetzung)
Ausgabe NEU: figures/black_swan<tag>/bs_<id>_<slug>_f<tau>_<modell>.png/.pdf, Manifest p40_black_swan_maps<tag>.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.ndimage import uniform_filter  # noqa: E402

from wildfire.audit.common import assert_readonly, load_config, read_manifest, write_manifest  # noqa: E402
from wildfire.paper.black_swan_dense import TAUS  # noqa: E402

NAMES = {16: "Québec South, June 2023", 12: "Québec North, June 2023", 1: "Northeast BC 2023 (Donnie Creek Area)",
         0: "Southern NWT 2023 (Hay River Area)", 2: "Northeast BC 2024", 5: "Manitoba 2025",
         30: "Fort McMurray Area, May 2016", 3: "NWT 2014", 14: "Alberta, May 2023", 20: "Saskatchewan, May 2025"}
MODEL_LABEL = {"lh_ref": "STS-LH", "lh_mono": "STS-LH + Monotonicity", "cg_ref": "Contagion-STS",
               "cg_mono": "Contagion-STS + Monotonicity"}
# Farben: Referenzpalette des dataviz-Skills (sequenziell Blau, divergierend Blau/Rot mit grauer Mitte)
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
LAND, BURNED, NEXT7, NEXT30 = "#f0efec", "#c3c2b7", "#d03b3b", "#eda100"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV = ["#1c5cab", "#6da7ec", "#cde2fb", "#f0efec", "#f6c4c3", "#e66767", "#b02c2c"]


def slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s.lower()).strip("_").replace("__", "_")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--events", type=int, nargs="*", default=[16, 12, 1, 0, 2, 5, 30])
    ap.add_argument("--models", nargs="+", default=["lh_ref", "cg_ref"])
    ap.add_argument("--taus", type=int, nargs="+", default=[7, 30])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    fig_dir = exp / "figures" / f"black_swan{args.tag}"
    if fig_dir.exists():
        raise SystemExit(f"ABBRUCH: {fig_dir} existiert schon - anderen --tag waehlen")

    plt.rcParams.update({"font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 9,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    font_used = Path(font_manager.findfont(font_manager.FontProperties(family=plt.rcParams["font.family"]))).name
    print(f"[bsm] Schrift: {font_used}", flush=True)

    S = dict(np.load(exp / "derived_data" / f"black_swan{args.tag}" / "set.npz"))
    info = {e["id"]: e for e in read_manifest(cfg, f"p40_black_swan_set{args.tag}")["events"]}
    kmap = int(read_manifest(cfg, f"p40_black_swan_set{args.tag}")["kmap"])
    mdir = exp / "metrics" / f"black_swan_dense{args.tag}"
    tr = json.loads((exp / "data_manifests" / "d0_masks.json").read_text())["grid"]["transform"]
    fi = np.load(exp / "derived_data" / "events" / "fire_index.npz", allow_pickle=False)
    W = int(fi["grid"][1])
    flat = np.repeat(fi["fire_pixels"].astype(np.int64), np.diff(fi["fire_offsets"]))
    fr, fc, fd = flat // W, flat % W, fi["fire_days"].astype(np.int64)
    dates = [str(d)[:10] for d in fi["dates"]]
    land = np.load(exp / "derived_data" / "masks" / "land.npy") & np.load(exp / "derived_data" / "masks" / "canada.npy")
    fig_dir.mkdir(parents=True)
    written = []

    for name in args.models:
        Fp = mdir / f"F_cal_{name}_centre.npy"
        if not Fp.exists():
            print(f"[bsm] fehlt: {Fp.name}", flush=True)
            continue
        F = np.load(Fp)
        for eid in args.events:
            if eid not in info:
                print(f"[bsm] Ereignis {eid} nicht in der Menge", flush=True)
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
            for tau in args.taus:
                j = TAUS.index(tau)
                nr, nc = (r1 - r0 + kmap - 1) // kmap, (c1 - c0 + kmap - 1) // kmap
                preds, diffs = [], []
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
                    Ds = np.where(ok, num / np.maximum(den, 1e-9), np.nan)
                    preds.append(P); diffs.append(Ds)
                allp = np.concatenate([p[np.isfinite(p)] for p in preds])
                vmin, vmax = max(1e-4, float(np.quantile(allp, 0.02))), max(float(np.quantile(allp, 0.998)), 2e-4)
                dl = max(float(np.nanquantile(np.abs(np.concatenate([d.ravel() for d in diffs])), 0.99)), 1e-3)
                seq = mcolors.LinearSegmentedColormap.from_list("seq", SEQ)
                seq.set_bad(SURFACE)
                div = mcolors.LinearSegmentedColormap.from_list("div", DIV)
                div.set_bad(SURFACE)
                fig, axs = plt.subplots(len(days), 3, figsize=(10.5, 2.9 * len(days) + 0.9), squeeze=False,
                                        constrained_layout=True)
                for k, L in enumerate(days):
                    obs = np.zeros((r1 - r0, c1 - c0), np.int8)                     # 0 Land, 1 verbrannt, 2/3 naechste
                    obs[~land[r0:r1, c0:c1]] = -1
                    sel = in_box & (fd >= season0) & (fd <= L)
                    obs[fr[sel] - r0, fc[sel] - c0] = 1
                    sel = in_box & (fd > L + 7) & (fd <= L + 30)
                    obs[fr[sel] - r0, fc[sel] - c0] = 3
                    sel = in_box & (fd > L) & (fd <= L + 7)
                    obs[fr[sel] - r0, fc[sel] - c0] = 2
                    cmap_o = mcolors.ListedColormap([SURFACE, LAND, BURNED, NEXT7, NEXT30])
                    axs[k, 0].imshow(obs + 1, cmap=cmap_o, vmin=-0.5, vmax=4.5, extent=ext, interpolation="nearest")
                    im1 = axs[k, 1].imshow(preds[k], cmap=seq, norm=mcolors.LogNorm(vmin, vmax), extent=ext,
                                           interpolation="nearest")
                    im2 = axs[k, 2].imshow(diffs[k], cmap=div, norm=mcolors.TwoSlopeNorm(0, -dl, dl), extent=ext,
                                           interpolation="nearest")
                    rel = L - e["start_day"]
                    lab = "Onset" if rel == 0 else f"Onset {'+' if rel > 0 else '−'} {abs(rel)} d"
                    axs[k, 0].set_ylabel(f"{lab}\n{dates[L]}\nLatitude (°N)", color=INK)
                    for a in axs[k]:
                        a.set_aspect(aspect)
                        a.tick_params(labelsize=7, length=2)
                        for sp in a.spines.values():
                            sp.set_color(GRID)
                    for a in axs[k, 1:]:
                        a.set_yticklabels([])
                    if k == len(days) - 1:
                        for a in axs[k]:
                            a.set_xlabel("Longitude (°E)")
                axs[0, 0].set_title("Observed MODIS Detections", fontsize=10)
                axs[0, 1].set_title(f"Predicted {tau}-Day Fire Probability", fontsize=10)
                axs[0, 2].set_title(f"Observed − Predicted ({tau} Days, 5×5 Mean)", fontsize=10)
                axs[-1, 0].legend(handles=[Patch(color=NEXT7, label="Fire in Next 7 Days"),
                                           Patch(color=NEXT30, label="Fire in Days 8–30"),
                                           Patch(color=BURNED, label="Burned Earlier This Season")],
                                  loc="upper center", bbox_to_anchor=(0.5, -0.32), frameon=False, fontsize=8, ncol=1)
                cb = fig.colorbar(im1, ax=axs[:, 1], location="bottom", shrink=0.85, pad=0.02)
                cb.set_label(f"F{tau}, Centre-Pixel Head, Calibrated (Log Scale)")
                cb2 = fig.colorbar(im2, ax=axs[:, 2], location="bottom", shrink=0.85, pad=0.02)
                cb2.set_label("Red: More Fire Than Predicted")
                ttl = NAMES.get(eid, f"Event {eid}")
                ctrl = " (Training-Year Control)" if e.get("control") else ""
                fig.suptitle(f"{ttl}{ctrl}: Observed Fire vs. {MODEL_LABEL.get(name, name)} Ensemble", fontsize=12)
                base = fig_dir / f"bs_{eid}_{slug(ttl)}_f{tau}_{name}"
                for ext_ in ("png", "pdf"):
                    fig.savefig(f"{base}.{ext_}", dpi=200, facecolor=SURFACE)
                plt.close(fig)
                written.append(str(base))
                print(f"[bsm] {base}.png/.pdf", flush=True)
    write_manifest(cfg, f"p40_black_swan_maps{args.tag}", args.config, {
        "figures": written, "models": args.models, "taus": args.taus, "font": font_used, "kmap": kmap})


if __name__ == "__main__":
    main()
