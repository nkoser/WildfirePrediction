"""Gespeicherte U-Net-Vorhersagen (cache_G) auf dem Test-Zeitraum des FireMPC-Posters neu auswerten.

Poster (NeurIPS): Train 2000-2020, Val 2021-2022, Test 2023-2024 (2025 in Klammern).
README/JSON des Caches: Train 2000-2019, Val 2020-2022, Test 2023-2025.
Hier nur Neu-Auswertung derselben Vorhersagen: Test gesamt, 2023-2024, je Jahr.
Ausgabe NEU: metrics/poster_split_eval/{metrics.csv, summary.md}.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wildfire.audit.common import load_config
from wildfire.paper import targets
from wildfire.paper.metrics import paper_metrics

DTS = (1, 7, 15, 30)
RUNS = {"cache": ("rec_cache_G_cache_paperhp.npz", 1),          # Label des Caches, end = dt + 1
        "next_patch": ("rec_cache_G_cache_next_paperhp_w0.npz", 0)}
PAPER_UNET = {1: (71.42, 67.82), 7: (65.09, 59.84), 15: (62.31, 55.07), 30: (54.07, 44.85)}
SUBSETS = {"2023-2025": (2023, 2024, 2025), "2023-2024": (2023, 2024),
           "2023": (2023,), "2024": (2024,), "2025": (2025,)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    exp = Path(load_config(args.config)["experiment_root"])
    d = exp / "derived_data" / "paper" / "cache_G_L61"
    year = np.load(d / "meta.npy")[:, 0] // 10000
    out = exp / "metrics" / "poster_split_eval"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for target, (fn, off) in RUNS.items():
        z = np.load(exp / "predictions" / "paper" / fn)
        idx = z["idx_test"]
        y = targets.labels(d, target)[idx]
        yr = year[idx]
        for dt in DTS:
            keys = sorted(k for k in z.files if k.startswith(f"paper55_T30_end{dt + off}_s") and k.endswith("_test"))
            for name, years in SUBSETS.items():
                m = np.isin(yr, years)
                res = [paper_metrics(z[k][m], y[m]) for k in keys]
                rows.append((target, dt, name, int(m.sum()), float(y[m].mean()), len(keys),
                             100 * np.mean([r["f1"] for r in res]), 100 * np.mean([r["pr_auc"] for r in res]),
                             100 * np.mean([r["precision"] for r in res]), 100 * np.mean([r["recall"] for r in res]),
                             np.mean([r["roc_auc"] for r in res])))
    head = "target,dt,test_years,n,prevalence,seeds,f1,pr_auc,precision,recall,roc_auc"
    with open(out / "metrics.csv", "w") as f:
        f.write(head + "\n")
        for r in rows:
            f.write(",".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + "\n")
    md = ["| Ziel | Δt | Testjahre | n | Prävalenz | F1 | PR-AUC | P | R | ROC-AUC | Paper F1 / PR-AUC |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]:.3f} | {r[6]:.2f} | {r[7]:.2f} | {r[8]:.2f} | "
                  f"{r[9]:.2f} | {r[10]:.3f} | {PAPER_UNET[r[1]][0]} / {PAPER_UNET[r[1]][1]} |")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
