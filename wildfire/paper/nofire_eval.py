"""Paper-Protokoll mit und ohne Feuermaske als Eingabe (3D-U-Net, Ziel D+1).

Das MPCFire-Manuskript schreibt, MOD/MYD14A1 diene nur als Label und sei kein
Eingabekanal. Die veroeffentlichten Caches tragen die Maske aber als Kanal 0
(Audit s5), und unser Nachbau (_next_paperhp_w0) nutzt alle 55 Kanaele. Hier
stehen die Laeufe paper55 / no_fire / no_modis nebeneinander, je Delta t mit
Mittel und Streuung ueber die Seeds und dem Ensemble (Mittel der Seeds).

Liest nur vorhandene Vorhersagen (rec_<cache>_cache<run>.npz aus reconstruct)
und schreibt das Manifest p25_nofire_unet_<cache>.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.paper.metrics import paper_metrics

DTS = (1, 7, 15, 30)
REPORTED = {1: 0.718, 7: 0.598, 15: 0.551, 30: 0.481}     # 3D U-Net laut Paper, PR-AUC
KEYS = ("pr_auc", "f1", "precision", "recall", "ece", "brier")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--runs", nargs="+", default=["_next_paperhp_w0", "_next_paperhp_w0_nofire"])
    ap.add_argument("--target", default="next_patch")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    y = np.load(exp / "derived_data" / "paper" / f"{args.cache}_L61" / f"y_{args.target}.npy").astype(int)

    out = {}
    for run in args.runs:
        z = np.load(exp / "predictions" / "paper" / f"rec_{args.cache}_cache{run}.npz")
        yt = y[z["idx_test"]]
        csets = sorted({k.split("_T30_")[0] for k in z.files if "_T30_end" in k})
        for cs in csets:
            for dt_ in DTS:
                ks = sorted(k for k in z.files if k.startswith(f"{cs}_T30_end{dt_}_s") and k.endswith("_test"))
                if not ks:
                    continue
                per = [paper_metrics(z[k], yt) for k in ks]
                ens = paper_metrics(np.mean([z[k] for k in ks], 0), yt)
                r = {"run": run, "channels": cs, "dt": dt_, "n_seeds": len(ks), "ensemble": ens,
                     "seed_mean": {m: float(np.mean([p[m] for p in per])) for m in KEYS},
                     "seed_std": {m: float(np.std([p[m] for p in per])) for m in KEYS}}
                out[f"{cs}_dt{dt_}"] = r
                print(f"[nf] {cs:9s} dt={dt_:2d}  seeds {len(ks)}  PR-AUC {r['seed_mean']['pr_auc']:.3f}"
                      f"±{r['seed_std']['pr_auc']:.3f}  Ensemble PR-AUC {ens['pr_auc']:.3f}  F1 {ens['f1']:.3f}"
                      f"  ECE {ens['ece']:.3f}  Brier {ens['brier']:.3f}   Paper {REPORTED[dt_]:.3f}", flush=True)
    write_manifest(cfg, f"p25_nofire_unet_{args.cache}", args.config, {
        "cache": args.cache, "target": args.target, "runs": args.runs, "reported_pr_auc": REPORTED,
        "results": out,
    })


if __name__ == "__main__":
    main()
