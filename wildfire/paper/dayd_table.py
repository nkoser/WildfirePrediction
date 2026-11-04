"""Paper-Protokoll auf beiden Zielen, mit und ohne MODIS-Feuer als Eingabe.

Ziel D (Cache-Label, Feuer im Ausschnitt am letzten Cache-Tag) gegen D+1
(next_patch). Fenster wie im Paper: Zieltag t, Eingabe [t - dt - 30, t - dt - 1];
das U-Net speichert das als --end dt + 1 (Ziel D) bzw. --end dt (Ziel D+1).

  U-Net  rec_<cache>_cache<run>.npz (reconstruct), PR-AUC je Seed neu berechnet
  STS    p15-Manifeste (sts_dctm), Ausschnittskopf, Mittel und Streuung ueber Seeds

Liest nur vorhandene Ergebnisse; fehlende Laeufe werden ausgelassen. Schreibt das
Manifest p26_dayd_table_<cache>.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.paper import targets
from wildfire.paper.metrics import paper_metrics

DTS = (1, 7, 15, 30)
REPORTED = {1: 0.718, 7: 0.598, 15: 0.551, 30: 0.481}     # 3D U-Net laut Paper, PR-AUC
UNET = {"cache": {"paper55": "_paperhp", "no_fire": "_paperhp_no_fire", "no_modis": "_paperhp_no_modis"},
        "next_patch": {"paper55": "_next_paperhp_w0", "no_fire": "_next_paperhp_w0_nofire",
                       "no_modis": "_next_paperhp_w0_nomodis"}}
UNET_F1 = {"cache": {"paper55": "_paperhp_f1", "no_fire": "_paperhp_f1_no_fire", "no_modis": "_paperhp_f1_no_modis"},
           "next_patch": {"paper55": "_next_paperhp_f1", "no_fire": "_next_paperhp_f1_nofire",
                          "no_modis": "_next_paperhp_f1_nomodis"}}
UNET_F1B = {s: {"cache": {cs: f"_paperhp_{s}{'' if cs == 'paper55' else '_' + cs}"
                          for cs in ("paper55", "no_fire", "no_modis")},
                "next_patch": {"paper55": f"_next_paperhp_{s}", "no_fire": f"_next_paperhp_{s}_nofire",
                               "no_modis": f"_next_paperhp_{s}_nomodis"}}
            for s in ("f1w", "f1max")}           # 98: F1 bei 0,5 mit Warten / bester F1
STS = {"LH": "_d64_lh_ema_nll", "Contagion v2 (scratch)": "_d64_cg2_ema_nll"}
KEYS = ("pr_auc", "f1", "precision", "recall", "ece", "brier")


def unet_rows(exp: Path, cache: str, target: str, runs: dict = UNET, model: str = "U-Net") -> list[dict]:
    d = exp / "derived_data" / "paper" / f"{cache}_L61"
    y = targets.labels(d, target)
    rows = []
    for cs, run in runs[target].items():
        p = exp / "predictions" / "paper" / f"rec_{cache}_cache{run}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        yt = y[z["idx_test"]]
        r = {"model": model, "channels": cs, "target": target, "source": p.name, "dt": {}}
        for dt_ in DTS:
            end = dt_ + 1 - targets.shift(target)
            ks = sorted(k for k in z.files if k.startswith(f"{cs}_T30_end{end}_s") and k.endswith("_test"))
            if not ks:
                continue
            per = [paper_metrics(z[k], yt) for k in ks]
            r["dt"][str(dt_)] = {"n_seeds": len(ks),
                                 "seed_mean": {m: float(np.mean([q[m] for q in per])) for m in KEYS},
                                 "seed_std": {m: float(np.std([q[m] for q in per])) for m in KEYS}}
        rows.append(r)
    return rows


def sts_rows(exp: Path, cache: str, target: str) -> list[dict]:
    rows = []
    for model, tg in STS.items():
        for cs in ("paper55", "no_fire", "no_modis"):
            ctag = "" if cs == "paper55" else f"_{cs}"
            fs = [exp / "data_manifests" / f"p15_sts_dctm_{cache}{targets.tag(target)}{ctag}_cube{tg}_s{s}.json"
                  for s in range(3)]
            fs = [f for f in fs if f.exists()]
            if not fs:
                continue
            docs = [json.loads(f.read_text()) for f in fs]
            if any(dc.get("channels", "paper55") != cs or dc.get("target") != target for dc in docs):
                raise SystemExit(f"Manifest passt nicht zu {cs}/{target}: {[f.name for f in fs]}")
            r = {"model": model, "channels": cs, "target": target, "source": [f.name for f in fs], "dt": {}}
            for dt_ in DTS:
                per = [dc["results"][str(dt_)]["patch_head"]["per_seed"][0] for dc in docs]
                r["dt"][str(dt_)] = {"n_seeds": len(per),
                                     "seed_mean": {m: float(np.mean([q[m] for q in per])) for m in KEYS},
                                     "seed_std": {m: float(np.std([q[m] for q in per])) for m in KEYS}}
            rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", default="cache_G")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])

    rows = []
    for target in ("cache", "next_patch"):
        rows += (unet_rows(exp, args.cache, target) + unet_rows(exp, args.cache, target, UNET_F1, "U-Net (Val-F1)")
                 + unet_rows(exp, args.cache, target, UNET_F1B["f1w"], "U-Net (Val-F1, Warten)")
                 + unet_rows(exp, args.cache, target, UNET_F1B["f1max"], "U-Net (Val-F1max)")
                 + sts_rows(exp, args.cache, target))
    print("[dayd] PR-AUC, Mittel ueber Seeds (Streuung)   Paper-U-Net: "
          + " / ".join(f"{REPORTED[t]:.3f}" for t in DTS), flush=True)
    for r in rows:
        cells = "  ".join(f"{r['dt'][str(t)]['seed_mean']['pr_auc']:.3f}({r['dt'][str(t)]['seed_std']['pr_auc']:.3f})"
                          if str(t) in r["dt"] else "   -   " for t in DTS)
        n = max((v["n_seeds"] for v in r["dt"].values()), default=0)
        print(f"[dayd] {r['target']:10s} {r['model']:24s} {r['channels']:9s} n={n}  {cells}", flush=True)
    write_manifest(cfg, f"p26_dayd_table_{args.cache}", args.config, {
        "cache": args.cache, "reported_pr_auc": REPORTED, "unet_runs": UNET, "unet_f1_runs": UNET_F1, "unet_f1b_runs": UNET_F1B, "sts_tags": STS,
        "score_sts": "Ausschnittskopf F(dt + 1) bei Stichtag e = Zieltag - dt - 1", "rows": rows,
    })


if __name__ == "__main__":
    main()
