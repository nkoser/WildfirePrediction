"""Einfache Vergleichsbasis nur aus Feuerhistorie und Kalendertag, CPU: Wie weit kommt man ohne Wetter und Brennstoff?

Merkmale je Stichtag aus dem Feuerkanal des 30-Tage-Fensters (Feuer = Wert > 0):
  Feuer-Pixeltage in den letzten 1 / 7 / 30 Tagen in Mitte, 3 x 3 und Ausschnitt (log1p), Tage seit dem
  letzten Feuer in Mitte, 3 x 3 und Ausschnitt (30 = keins), Kalendertag sin / cos.  -> 14 Merkmale
Modelle je Ziel: logistische Regression (standardisiert) und Gradient Boosting (HistGradientBoosting, Standard-
werte). Ziel = erstes Feuer bis tau Tage nach dem Stichtag, Mitte bzw. Ausschnitt, tau = dt + 1 (Paper-Protokoll)
und 30 (Survival-Auswertung). Training auf den Wuerfel-Stichtagen 2000-2019 (derived_data/paper_pixel, wie die
STS-Modelle; feuernah ausgewaehlt, die Achsenabschnitte sind deshalb verschoben, Rangmasse nicht).
Auswertung:
  paper  cache_G_L61, Test: Stichtag e = D - dt, Score P(Feuer bis dt + 1); PR-AUC / F1max gegen das Ziel D + 1
         (Mitte = Pixel (6, 6), Ausschnitt), wie sts_extreme.paper_eval
  surv   surv_eval_test, Test 2023-2025: AUC und PR-AUC fuer "Feuer bis 30 Tage" (Mitte, Ausschnitt); zum Vergleich
         dieselben Masse fuer die STS-Ensembles aus calib_cache_ext/ens_<name>.npz (F30 aus den Hazards)
Ausgabe NEU: metrics/fire_baseline<tag>/summary.md, results.json, Manifest p43_fire_baseline<tag>.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.data.dataset import CubeReader
from wildfire.paper import targets
from wildfire.paper.compare import DTS
from wildfire.paper.metrics import paper_metrics
from wildfire.paper.prepare import FIRE_CH
from wildfire.paper.sts_dctm import CENTRE, doy_of, patch_events
from wildfire.paper.surv_eval import _cum_hazard

TAUS = sorted({dt + 1 for dt in DTS} | {30})
UNET = {1: 0.691, 7: 0.607, 15: 0.558, 30: 0.515}


def features(fire: np.ndarray, doy: np.ndarray) -> np.ndarray:
    """fire (n, 30, 13, 13) bool, doy (n,) -> (n, 14) float32."""
    c, h = CENTRE, 1
    regs = (fire[:, :, c, c], fire[:, :, c - h:c + h + 1, c - h:c + h + 1].reshape(len(fire), fire.shape[1], -1),
            fire.reshape(len(fire), fire.shape[1], -1))
    out = []
    for r in regs:
        r = r if r.ndim == 3 else r[:, :, None]
        for k in (1, 7, 30):
            out.append(np.log1p(r[:, -k:].sum((1, 2))))
        anyd = r.any(2)                                                  # (n, 30)
        last = np.where(anyd.any(1), anyd.shape[1] - 1 - np.argmax(anyd[:, ::-1], 1), 30)
        out.append(last)
    ang = 2 * np.pi * np.asarray(doy, float) / 365.25
    out += [np.sin(ang), np.cos(ang)]
    return np.stack(out, 1).astype(np.float32)


def fire_of(X, rows, d0: int = 0, d1: int | None = None, chunk: int = 4000) -> np.ndarray:
    """Feuerkanal (n, T, 13, 13) bool aus einem (memmap-)Array, stueckweise."""
    out = []
    for a in range(0, len(rows), chunk):
        r = rows[a:a + chunk]
        out.append(np.asarray(X[r, d0:d1, FIRE_CH]) > 0)
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--cube-dir", default="paper_pixel")
    ap.add_argument("--surv-dir", default="surv_eval_test")
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--ens", nargs="*", default=["lh_ref", "cg_ref", "lh_mono", "cg_mono"])
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    out = exp / "metrics" / f"fire_baseline{args.tag}"
    if out.exists():
        raise SystemExit(f"ABBRUCH: {out} existiert schon - anderen --tag waehlen")
    t0 = time.time()

    # ---- Training: Wuerfel-Stichtage
    pdir = exp / "derived_data" / args.cube_dir
    Xtr = np.load(pdir / "train_X.npy", mmap_mode="r")
    n = Xtr.shape[0] if not args.limit_train else args.limit_train
    rows = np.arange(n)
    dates = CubeReader(exp).dates
    Lm = np.load(pdir / "train_meta.npz")["L"][:n]
    doy_tr = np.array([dt.date.fromisoformat(dates[int(L)][:10]).timetuple().tm_yday for L in Lm], np.float32)
    ftr = features(fire_of(Xtr, rows), doy_tr)
    ev = np.asarray(np.load(pdir / "train_event.npy", mmap_mode="r")[:n])
    tt = np.asarray(np.load(pdir / "train_t.npy", mmap_mode="r")[:n]).astype(np.int64)
    pe, pt = patch_events(ev, tt)
    ytr = {("centre", tau): ev[:, CENTRE, CENTRE] & (tt[:, CENTRE, CENTRE] <= tau) for tau in TAUS}
    ytr |= {("patch", tau): pe & (pt <= tau) for tau in TAUS}
    print(f"[fb] Training {n:,} Stichtage, Merkmale {ftr.shape} ({time.time() - t0:.0f} s)", flush=True)

    models = {}
    for (tg, tau), y in ytr.items():
        for kind in ("logit", "gbt"):
            m = (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)) if kind == "logit"
                 else HistGradientBoostingClassifier(random_state=0))
            m.fit(ftr, y.astype(int))
            models[(kind, tg, tau)] = m
        print(f"[fb]   {tg} tau {tau}: Rate {y.mean():.3f} ({time.time() - t0:.0f} s)", flush=True)

    res = {"paper": {}, "surv": {}}
    # ---- Paper-Protokoll
    cd = exp / "derived_data" / "paper" / f"{args.cache}_L61"
    Xp = np.load(cd / "X.npy", mmap_mode="r")
    LC = Xp.shape[1]
    te = np.where(np.load(cd / "split.npy") == "test")[0]
    y_pa = targets.labels(cd, "next_patch")[te]
    y_px = targets.fire_map(cd, Xp, "next_patch")[te][:, CENTRE, CENTRE].astype(int)
    doyD = doy_of(np.load(cd / "meta.npy")[:, 0])[te]
    for dt_ in DTS:
        e = LC - 1 - dt_
        f = features(fire_of(Xp, te, e - 29, e + 1), doyD - dt_)
        for kind in ("logit", "gbt"):
            pc = models[(kind, "centre", dt_ + 1)].predict_proba(f)[:, 1]
            pa = models[(kind, "patch", dt_ + 1)].predict_proba(f)[:, 1]
            res["paper"].setdefault(kind, {})[str(dt_)] = {"centre": paper_metrics(np.clip(pc, 1e-7, 1 - 1e-7), y_px),
                                                          "patch": paper_metrics(np.clip(pa, 1e-7, 1 - 1e-7), y_pa)}
        print(f"[fb] Paper dt {dt_}: Ausschnitt PR-AUC logit {res['paper']['logit'][str(dt_)]['patch']['pr_auc']:.3f} "
              f"gbt {res['paper']['gbt'][str(dt_)]['patch']['pr_auc']:.3f}  (U-Net {UNET[dt_]})", flush=True)

    # ---- Survival-Stichtage, Test
    sd = exp / "derived_data" / args.surv_dir
    meta = dict(np.load(sd / "eval_meta.npz"))
    Xs = np.load(sd / "eval_X.npy", mmap_mode="r")
    fs = features(fire_of(Xs, np.arange(meta["row"].size)), meta["doy"])
    ys = {"centre": meta["ev_centre"].astype(bool) & (meta["t_centre"] <= 30),
          "patch": meta["ev_patch"].astype(bool) & (meta["t_patch"] <= 30)}
    sc = {}
    for kind in ("logit", "gbt"):
        for tg in ("centre", "patch"):
            sc[(kind, tg)] = models[(kind, tg, 30)].predict_proba(fs)[:, 1]
    hz = exp / "derived_data" / "calib_cache_ext"
    for name in args.ens:
        p = hz / f"ens_{name}.npz"
        if p.exists():
            z = np.load(p)
            for tg in ("centre", "patch"):
                sc[(name, tg)] = -np.expm1(-_cum_hazard(z[f"test_{tg}"].astype(np.float64))[:, 29])
    for (kind, tg), s in sc.items():
        res["surv"].setdefault(kind, {})[tg] = {"auc30": float(roc_auc_score(ys[tg], s)),
                                                "pr30": float(average_precision_score(ys[tg], s))}

    # ---- Ausgaben
    out.mkdir(parents=True)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    md = ["# Vergleichsbasis nur Feuerhistorie + Kalendertag", "",
          "## Paper-Protokoll, Test cache_G_L61 (PR-AUC Mitte / Ausschnitt je dt)", "",
          "| Modell | " + " | ".join(f"Mitte dt {d}" for d in DTS) + " | " + " | ".join(f"Ausschnitt dt {d}" for d in DTS) + " |",
          "|---" * (1 + 2 * len(DTS)) + "|"]
    for kind in ("logit", "gbt"):
        r = res["paper"][kind]
        md.append(f"| {kind} | " + " | ".join(f"{r[str(d)]['centre']['pr_auc']:.3f}" for d in DTS) + " | "
                  + " | ".join(f"{r[str(d)]['patch']['pr_auc']:.3f}" for d in DTS) + " |")
    md.append("| U-Net-Nachbau (Ausschnitt) | – | – | – | – | " + " | ".join(f"{UNET[d]:.3f}" for d in DTS) + " |")
    md += ["", "## Survival-Stichtage Test 2023-2025: Feuer bis 30 Tage (AUC / PR-AUC)", "",
           "| Modell | Mitte AUC | Mitte PR | Ausschnitt AUC | Ausschnitt PR |", "|---|---|---|---|---|"]
    for kind, r in res["surv"].items():
        md.append(f"| {kind} | {r['centre']['auc30']:.3f} | {r['centre']['pr30']:.3f} | "
                  f"{r['patch']['auc30']:.3f} | {r['patch']['pr30']:.3f} |")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)
    write_manifest(cfg, f"p43_fire_baseline{args.tag}", args.config, {
        "taus": TAUS, "n_train": int(n), "features": "fire counts 1/7/30 d x centre/3x3/patch (log1p), days since "
        "last fire x 3, doy sin/cos", "results": res, "out": str(out)})


if __name__ == "__main__":
    main()
