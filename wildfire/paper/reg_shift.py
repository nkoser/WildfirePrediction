"""Warum schadet der regionale Saisonzustand (sts_extreme --ctx reg)? Nur CPU, kein Modell.

Vergleicht die 8 Regional-Merkmale (ctx_features.REG_NAMES) zwischen den Trainings-Stichtagen aus dem
Wuerfel (derived_data/paper_pixel, feuernah ausgewaehlt) und den zufaelligen Stichtagen von surv_eval_test
(Val 2020-2022, Test 2023-2025, je Jahr):
  V  Verteilung: Mittel, Quantile, z-Wert relativ zum Training (wie das Modell normiert), Anteil
     ausserhalb [min, max] und [p1, p99] des Trainings
  E  Beziehung zum Ziel: Rate "Feuer im Ausschnitt bis 30 Tage" je Dezil des Merkmals (Dezilgrenzen aus
     dem Training) im Training und in Val/Test. Gleiche Steigung = reine Verschiebung des Niveaus,
     andere Steigung = das Modell lernt im feuernahen Wuerfel eine andere Beziehung.
Ausgabe NEU: metrics/reg_shift<tag>/summary.md, deciles.csv, Manifest p37_reg_shift<tag>.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path

import numpy as np

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.data.dataset import CubeReader
from wildfire.paper.ctx_features import REG_NAMES
from wildfire.paper.extreme_eval import year_of
from wildfire.paper.sts_dctm import patch_events

TAU = 30


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ctx-dir", default="ctx_features")
    ap.add_argument("--cube-dir", default="paper_pixel")
    ap.add_argument("--surv-dir", default="surv_eval_test")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    out = exp / "metrics" / f"reg_shift{args.tag}"
    if out.exists():
        raise SystemExit(f"ABBRUCH: {out} existiert schon - anderen --tag waehlen")
    cdir, pdir, sd = (exp / "derived_data" / d for d in (args.ctx_dir, args.cube_dir, args.surv_dir))

    # ---- Training (Wuerfel-Stichtage)
    reg_tr = np.load(cdir / "train_reg.npy").astype(np.float64)
    ev = np.load(pdir / "train_event.npy", mmap_mode="r")
    tt = np.load(pdir / "train_t.npy", mmap_mode="r")
    e_tr, t_tr = patch_events(np.asarray(ev), np.asarray(tt))
    y_tr = (e_tr & (t_tr <= TAU)).astype(float)
    dates = CubeReader(exp).dates
    yr_tr = np.array([int(dates[int(L)][:4]) for L in np.load(pdir / "train_meta.npz")["L"]])
    print(f"[reg] Training {reg_tr.shape[0]:,} Stichtage, Rate@30 {y_tr.mean():.3f}", flush=True)

    sets = {"train": (reg_tr, y_tr, yr_tr)}
    for s, f in (("val", "val"), ("test", "eval")):
        meta = dict(np.load(sd / f"{f}_meta.npz"))
        r = np.load(cdir / f"surv_{f}_reg.npy").astype(np.float64)
        y = (meta["ev_patch"].astype(bool) & (meta["t_patch"].astype(int) <= TAU)).astype(float)
        yr = year_of(meta["L_abs"])
        sets[s] = (r, y, yr)
        for yy in sorted(set(yr.tolist())):
            m = yr == yy
            sets[f"{s}_{yy}"] = (r[m], y[m], yr[m])
        print(f"[reg] {s} {r.shape[0]:,} Stichtage, Rate@30 {y.mean():.3f}", flush=True)

    mu, sdv = reg_tr.mean(0), reg_tr.std(0) + 1e-6
    lo, hi = reg_tr.min(0), reg_tr.max(0)
    p1, p99 = np.percentile(reg_tr, [1, 99], axis=0)
    edges = np.percentile(reg_tr, np.arange(10, 100, 10), axis=0)      # (9, 8) Dezilgrenzen aus dem Training

    res, dec_rows = {}, []
    for sn, (r, y, _) in sets.items():
        z = (r - mu) / sdv
        d = {}
        for j, k in enumerate(REG_NAMES):
            q = np.percentile(r[:, j], [5, 50, 95])
            b = np.searchsorted(edges[:, j], r[:, j], side="right")
            rate = [float(y[b == i].mean()) if (b == i).any() else float("nan") for i in range(10)]
            cnt = [int((b == i).sum()) for i in range(10)]
            d[k] = {"mean": float(r[:, j].mean()), "p5": float(q[0]), "p50": float(q[1]), "p95": float(q[2]),
                    "z_mean": float(z[:, j].mean()), "z_p95": float(np.percentile(z[:, j], 95)),
                    "out_minmax": float(((r[:, j] < lo[j]) | (r[:, j] > hi[j])).mean()),
                    "out_p1p99": float(((r[:, j] < p1[j]) | (r[:, j] > p99[j])).mean()),
                    "dec_rate": rate, "dec_n": cnt}
            dec_rows += [{"menge": sn, "merkmal": k, "dezil": i, "n": cnt[i], "rate30": rate[i]} for i in range(10)]
        res[sn] = {"n": int(r.shape[0]), "rate30": float(y.mean()), "feat": d}

    out.mkdir(parents=True)
    with open(out / "deciles.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["menge", "merkmal", "dezil", "n", "rate30"])
        w.writeheader(); w.writerows(dec_rows)
    order = ["train", "val", "test"] + sorted(k for k in sets if "_" in k)
    md = ["# Regionaler Saisonzustand: Training (Wuerfel) gegen Val/Test (zufaellige Stichtage)", "",
          f"Rate@30 = Feuer im Ausschnitt bis {TAU} Tage. z = (x - Mittel_train) / Std_train, wie im Modell.", ""]
    md += ["## V Verteilung (z-Mittel / Anteil ausserhalb [min, max] des Trainings)", "",
           "| Menge | n | Rate@30 | " + " | ".join(REG_NAMES) + " |", "|---" * (3 + len(REG_NAMES)) + "|"]
    for sn in order:
        f = res[sn]["feat"]
        md.append(f"| {sn} | {res[sn]['n']:,} | {res[sn]['rate30']:.3f} | "
                  + " | ".join(f"{f[k]['z_mean']:+.2f} / {100 * f[k]['out_minmax']:.1f}%" for k in REG_NAMES) + " |")
    md += ["", "## E Rate@30 je Trainings-Dezil (D1 ... D10)", ""]
    for k in REG_NAMES:
        md += [f"### {k}", "", "| Menge | " + " | ".join(f"D{i + 1}" for i in range(10)) + " | D10/D1 |",
               "|---" * 12 + "|"]
        for sn in order:
            rt = res[sn]["feat"][k]["dec_rate"]
            ratio = rt[9] / rt[0] if rt[0] and rt[0] == rt[0] and rt[0] > 0 else float("nan")
            md.append(f"| {sn} | " + " | ".join("–" if v != v else f"{v:.3f}" for v in rt) + f" | {ratio:.1f} |")
        md.append("")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)
    write_manifest(cfg, f"p37_reg_shift{args.tag}", args.config, {
        "reg_names": REG_NAMES, "tau": TAU, "train_years": sorted(set(yr_tr.tolist())), "out": str(out),
        "edges": edges.tolist(), "results": res, "created": dt.datetime.utcnow().isoformat()})


if __name__ == "__main__":
    main()
