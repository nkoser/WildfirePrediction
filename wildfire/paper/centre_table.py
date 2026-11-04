"""Paper-Protokoll auf dem Mittelzellen-Label: U-Nets aus 99_paper_unet_centre neben den
Ausschnitt-Modellen, jeweils auf Mittelzelle und Ausschnitt bewertet.

  Ziel D    Mittelzelle: targets.centre_labels(d, X, "cache");  Ausschnitt: y.npy
  Ziel D+1  Mittelzelle: y_next_centre.npy;                    Ausschnitt: y_next_patch.npy

Fenster wie im Paper (Zieltag t, Eingabe [t - dt - 30, t - dt - 1]), im U-Net als
--end dt + 1 (D) bzw. --end dt (D+1). PR-AUC je Seed neu aus rec_*.npz; fehlende Laeufe
werden ausgelassen. Schreibt das Manifest p27_centre_table_<cache>.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.paper import targets
from wildfire.paper.metrics import paper_metrics

DTS = (1, 7, 15, 30)
REPORTED = {1: 0.718, 7: 0.598, 15: 0.551, 30: 0.481}     # 3D U-Net laut Paper, PR-AUC
KEYS = ("pr_auc", "f1", "precision", "recall", "ece", "brier")
# (Tag D/D+1, trainiert auf, Kanaele, Laufname)
RUNS = [("D", "centre", "paper55", "_centre_paperhp"), ("D", "centre", "no_fire", "_centre_paperhp_no_fire"),
        ("D", "patch", "paper55", "_paperhp"), ("D", "patch", "no_fire", "_paperhp_no_fire"),
        ("D+1", "centre", "paper55", "_nextc_paperhp"), ("D+1", "centre", "no_fire", "_nextc_paperhp_nofire"),
        ("D+1", "patch", "paper55", "_next_paperhp_w0"), ("D+1", "patch", "no_fire", "_next_paperhp_w0_nofire")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", default="cache_G")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    d = exp / "derived_data" / "paper" / f"{args.cache}_L61"
    X = np.load(d / "X.npy", mmap_mode="r")
    lab = {("D", "centre"): targets.centre_labels(d, X, "cache"), ("D", "patch"): targets.labels(d, "cache"),
           ("D+1", "centre"): np.load(d / "y_next_centre.npy").astype(int),
           ("D+1", "patch"): targets.labels(d, "next_patch")}

    rows = []
    for day, trained, cs, run in RUNS:
        p = exp / "predictions" / "paper" / f"rec_{args.cache}_cache{run}.npz"
        if not p.exists():
            print(f"[centre] fehlt: {p.name}", flush=True)
            continue
        z = np.load(p)
        for evl in ("centre", "patch"):
            yt = lab[(day, evl)][z["idx_test"]]
            r = {"day": day, "trained_on": trained, "eval_on": evl, "channels": cs, "source": p.name,
                 "base_rate_test": float(yt.mean()), "dt": {}}
            for dt_ in DTS:
                end = dt_ + 1 if day == "D" else dt_
                ks = sorted(k for k in z.files if k.startswith(f"{cs}_T30_end{end}_s") and k.endswith("_test"))
                if not ks:
                    continue
                per = [paper_metrics(z[k], yt) for k in ks]
                r["dt"][str(dt_)] = {"n_seeds": len(ks),
                                     "seed_mean": {m: float(np.mean([q[m] for q in per])) for m in KEYS},
                                     "seed_std": {m: float(np.std([q[m] for q in per])) for m in KEYS}}
            rows.append(r)

    print("[centre] PR-AUC, Mittel ueber Seeds (Streuung)   Paper-U-Net: "
          + " / ".join(f"{REPORTED[t]:.3f}" for t in DTS), flush=True)
    for r in rows:
        cells = "  ".join(f"{r['dt'][str(t)]['seed_mean']['pr_auc']:.3f}({r['dt'][str(t)]['seed_std']['pr_auc']:.3f})"
                          if str(t) in r["dt"] else "   -   " for t in DTS)
        print(f"[centre] {r['day']:4s} trainiert {r['trained_on']:6s} bewertet {r['eval_on']:6s} {r['channels']:8s} "
              f"Basisrate {r['base_rate_test']:.3f}  {cells}", flush=True)
    write_manifest(cfg, f"p27_centre_table_{args.cache}", args.config, {
        "cache": args.cache, "reported_pr_auc": REPORTED, "runs": RUNS, "rows": rows,
        "labels": {"D centre": "targets.centre_labels(d, X, 'cache')", "D patch": "y.npy",
                   "D+1 centre": "y_next_centre.npy", "D+1 patch": "y_next_patch.npy"},
    })


if __name__ == "__main__":
    main()
