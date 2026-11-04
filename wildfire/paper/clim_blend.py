"""Bringt die Klimatologie bei langen Vorlaufzeiten etwas, das unsere Modelle nicht schon wissen?

Paper-Protokoll (Ziel: Feuer im Ausschnitt an D + 1), je dt:
  clim       nur die Klimatologie (climatology.py): day, season und beide per logistischer
             Regression auf dem Val-Split kombiniert. Zugleich die klassische Baseline "Klimatologie".
  model      Ensemble-Mittel der Modell-Scores F(dt + 1) (paper_metrics_ckpt --save-scores)
  blend      logistische Regression auf [logit model, logit day, logit season], angepasst auf dem
             Val-Split des Papers, ausgewertet auf dem Test-Split
Fuer blend - model: gepaartes Bootstrap ueber die Test-Stichproben (95 %-Intervall der PR-AUC-Differenz).
Manifest p25_clim_blend<tag>.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.paper.compare import DTS

EPS = 1e-4


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def fit_predict(Xv, yv, Xt):
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(Xv, yv)
    return lr.predict_proba(Xt)[:, 1], lr.coef_[0].tolist()


def paired_bootstrap(y, a, b, n=1000, seed=0):
    """95 %-Intervall von PR-AUC(b) - PR-AUC(a) ueber gezogene Test-Stichproben."""
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, y.size, y.size)
        if y[i].any():
            d.append(average_precision_score(y[i], b[i]) - average_precision_score(y[i], a[i]))
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--scores", required=True, help="Dateiname unter predictions/paper/")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    d = exp / "derived_data" / "paper" / f"{args.cache}_L61"
    y = np.load(d / "y_next_patch.npy").astype(int)
    c = np.load(d / "clim.npz")
    z = np.load(exp / "predictions" / "paper" / args.scores)
    iv, it = z["idx_val"], z["idx_test"]
    yv, yt = y[iv], y[it]
    C = np.stack([logit(c["day"]), logit(c["season"])], 1)

    res: dict = {"clim": {}}
    pc, coef = fit_predict(C[iv], yv, C[it])
    res["clim"] = {"day": float(average_precision_score(yt, c["day"][it])),
                   "season": float(average_precision_score(yt, c["season"][it])),
                   "both": float(average_precision_score(yt, pc)), "coef": coef}
    print(f"[blend] Klimatologie allein (gleich fuer alle dt): day {res['clim']['day']:.3f}  "
          f"season {res['clim']['season']:.3f}  beide {res['clim']['both']:.3f}", flush=True)

    models = sorted({k.split("|")[0] for k in z.files if "|" in k})
    for m in models:
        for dt_ in DTS:
            kv, kt = f"{m}|patch|{dt_}|val", f"{m}|patch|{dt_}|test"
            if kv not in z.files:
                continue
            sv, st = z[kv].mean(0), z[kt].mean(0)                       # Ensemble ueber die Seeds
            Xv = np.column_stack([logit(sv), C[iv]])
            Xt = np.column_stack([logit(st), C[it]])
            pb, cb = fit_predict(Xv, yv, Xt)
            seeds = [float(average_precision_score(yt, fit_predict(np.column_stack([logit(a), C[iv]]), yv,
                                                                   np.column_stack([logit(b), C[it]]))[0]))
                     for a, b in zip(z[kv], z[kt])]
            r = {"model": float(average_precision_score(yt, st)),
                 "model_per_seed": [float(average_precision_score(yt, b)) for b in z[kt]],
                 "blend": float(average_precision_score(yt, pb)), "blend_per_seed": seeds,
                 "blend_coef": cb, "delta_ci95": paired_bootstrap(yt, st, pb)}
            res.setdefault(m, {})[str(dt_)] = r
            print(f"[blend] {m:22s} dt={dt_:2d}  Modell {r['model']:.3f}  + Klimatologie {r['blend']:.3f}  "
                  f"(Differenz 95 %-KI {r['delta_ci95'][0]:+.3f} .. {r['delta_ci95'][1]:+.3f})  "
                  f"je Seed {np.mean(r['model_per_seed']):.3f} -> {np.mean(seeds):.3f}", flush=True)
    write_manifest(cfg, f"p25_clim_blend{args.tag}", args.config, {
        "cache": args.cache, "scores": args.scores, "target": "next_patch",
        "blend": "LogisticRegression(C=1) auf [logit F(dt+1) Ensemble, logit day, logit season], Val-Split -> Test-Split",
        "results": res,
    })


if __name__ == "__main__":
    main()
