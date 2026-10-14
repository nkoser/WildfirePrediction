"""Paper-Protokoll, Schritt 4: den FWI-HNM-Gefahrenwert nachbauen und an den Caches pruefen.

Definition laut Paper (Auszuege vom Nutzer):

  x_i(s,t)  Wert der CFFDRS-Komponente i am Pixeltag, i = FFMC, DMC, DC, ISI, BUI, FWI
  x~_i      = min(x_i / P99_i, 1),  P99_i = 99. Perzentil ueber das Trainingsset
  d_i       = max(0, (mu_i,Feuer - mu_i,kein Feuer) / sigma_i,kein Feuer)
  w_i       = d_i / sum_j d_j
  s         = sum_i w_i * x~_i  in [0, 1]

Laut README wird der Kandidatenpool am Median von s geteilt; harte Negative
(oben) und repraesentative (unten) zu gleichen Teilen.

Unsere Naeherung des Trainingssets: zufaellige Pixeltage der Trainingsjahre
2000-2019 aus dem Wuerfel (gueltige Zellen, 1. Mai bis 30. September; der
Oktober fehlt im Wuerfel), Feuer = MODIS-Feuermaske am selben Pixeltag. Das
Paper nutzt fuer Labels vermutlich FIRMS; das ist eine Naeherung.

Gemessen wird dann s fuer jede Stichprobe der vier Caches am Zentrumspixel und
Zieltag (letzter Schritt des Cache-Fensters, Audit s5).
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.data.dataset import CubeReader

FWI_CH = list(range(19, 25))                       # Wuerfel und Cache: dieselben Indizes
NAMES = ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]
CACHES = ("cache_A", "cache_G", "cache_H", "cache_Y")
Q = (5, 10, 25, 50, 75, 90, 95)


def score(x: np.ndarray, p99: np.ndarray, w: np.ndarray) -> np.ndarray:
    """x: (..., 6) Rohwerte -> s in [0, 1]."""
    return (np.minimum(x.astype(np.float64) / p99, 1.0) * w).sum(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train-years", nargs=2, type=int, default=[2000, 2019])
    ap.add_argument("--days-per-year", type=int, default=40)
    ap.add_argument("--pixel-frac", type=float, default=0.1)
    ap.add_argument("--season", nargs=2, default=["05-01", "09-30"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    rng = np.random.default_rng(int(cfg["seed"]))

    # ---------------------------------------------- Trainingsset (Pool)
    cube = CubeReader(exp)
    valid = np.load(exp / "derived_data" / "masks" / "valid.npy")
    vr, vc = np.nonzero(valid)
    xs, fs = [], []
    for year in range(args.train_years[0], args.train_years[1] + 1):
        info = cube.per_year[str(year)]
        days = [i for i, d in enumerate(info["days"]) if args.season[0] <= d[5:] <= args.season[1]]
        pick = np.sort(rng.choice(days, size=min(args.days_per_year, len(days)), replace=False))
        dyn, _ = cube._arrays(str(year))
        for li in pick:
            sl = np.asarray(dyn[li])                              # (H, W, 42)
            m = rng.random(vr.size) < args.pixel_frac
            r, c = vr[m], vc[m]
            xs.append(sl[r, c][:, FWI_CH])
            fs.append(sl[r, c, 0] > 0)
        print(f"[hnm] {year}: {len(pick)} Tage", flush=True)
    X = np.concatenate(xs).astype(np.float64)
    fire = np.concatenate(fs)
    print(f"[hnm] Pool: {X.shape[0]:,} Pixeltage, davon {fire.sum():,} mit Feuer "
          f"({100 * fire.mean():.3f} %)")

    p99 = np.percentile(X, 99, axis=0)
    p99 = np.where(p99 > 0, p99, 1.0)
    Xn = np.minimum(X / p99, 1.0)
    mu_f, mu_n = Xn[fire].mean(0), Xn[~fire].mean(0)
    sd_n = Xn[~fire].std(0)
    d = np.maximum(0.0, (mu_f - mu_n) / np.where(sd_n > 0, sd_n, 1.0))
    w = d / d.sum() if d.sum() > 0 else np.full(6, 1 / 6)
    s_pool = (Xn * w).sum(1)
    med = float(np.median(s_pool[~fire]))
    print("\n[hnm] Komponente   P99    mu_Feuer  mu_kein   sd_kein   d_i     w_i")
    for i, n in enumerate(NAMES):
        print(f"    {n:5s}  {p99[i]:7.1f}   {mu_f[i]:.3f}    {mu_n[i]:.3f}    {sd_n[i]:.3f}   "
              f"{d[i]:.3f}   {w[i]:.3f}")
    print(f"[hnm] Median von s ueber feuerfreie Pixeltage: {med:.4f}; "
          f"s bei Feuer im Median {np.median(s_pool[fire]):.4f}")

    # ---------------------------------------------- Caches
    out: dict = {}
    neg_s: dict = {}
    print("\n[hnm] s am Zieltag, Zentrumspixel   (Anteil ueber dem Pool-Median | Quantile 10/50/90)")
    for cache in CACHES:
        dd = exp / "derived_data" / "paper" / cache
        Xc = np.load(dd / "X.npy", mmap_mode="r")                 # (N, T, 55, 13, 13)
        y = np.load(dd / "y.npy")
        split = np.load(dd / "split.npy")
        sc = score(np.asarray(Xc[:, -1, 19:25, 6, 6]), p99, w)
        out[cache] = {}
        for sp in ("train", "val", "test"):
            for lab, name in ((1, "positive"), (0, "negative")):
                v = sc[(split == sp) & (y == lab)]
                q = np.percentile(v, Q)
                out[cache].setdefault(sp, {})[name] = {
                    "n": int(v.size), "share_above_pool_median": float((v > med).mean()),
                    "quantiles": {str(k): float(x) for k, x in zip(Q, q)}}
                if lab == 0:
                    neg_s[(cache, sp)] = v
                print(f"    {cache} {sp:5s} {name:8s} n={v.size:6d}  ueber Median "
                      f"{100 * (v > med).mean():5.1f} %  |  {q[1]:.3f} {q[3]:.3f} {q[5]:.3f}")

    # Verteilungsvergleich der Negativen: gleiche Ziehungsart sollte gleich aussehen
    ks = {}
    for sp in ("train", "test"):
        for a, b in (("cache_G", "cache_Y"), ("cache_A", "cache_Y"), ("cache_G", "cache_H"),
                     ("cache_A", "cache_G"), ("cache_H", "cache_Y")):
            r = ks_2samp(neg_s[(a, sp)], neg_s[(b, sp)])
            ks[f"{sp}:{a}|{b}"] = {"ks": float(r.statistic), "p": float(r.pvalue)}
    print("\n[hnm] KS-Abstand der Negativ-Verteilungen von s:")
    for k, v in ks.items():
        print(f"    {k:28s} KS {v['ks']:.3f}  p {v['p']:.1e}")

    write_manifest(cfg, "p3_hnm_score", args.config, {
        "definition": "x~=min(x/P99,1); d=max(0,(mu_fire-mu_nofire)/sd_nofire); w=d/sum d; s=sum w x~",
        "pool": {"train_years": args.train_years, "days_per_year": args.days_per_year,
                 "pixel_frac": args.pixel_frac, "season": args.season,
                 "n": int(X.shape[0]), "n_fire": int(fire.sum()),
                 "fire_definition": "MODIS-Feuermaske (Wuerfelkanal 0) am Pixeltag",
                 "caveat": "Wuerfel endet am 30.09.; Oktober fehlt im Pool. Paper-Labels vermutlich FIRMS."},
        "components": NAMES,
        "p99": p99.tolist(), "mu_fire": mu_f.tolist(), "mu_nofire": mu_n.tolist(),
        "sd_nofire": sd_n.tolist(), "d": d.tolist(), "w": w.tolist(),
        "pool_median_nofire": med,
        "caches": out, "ks_negatives": ks,
    })


if __name__ == "__main__":
    main()
