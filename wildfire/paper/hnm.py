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
from scipy.ndimage import maximum_filter
from scipy.stats import ks_2samp

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.data.dataset import CubeReader

FWI_CH = list(range(19, 25))                       # Wuerfel und Cache: dieselben Indizes
NAMES = ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]
CACHES = ("cache_A", "cache_G", "cache_H", "cache_Y")
Q = (5, 10, 25, 50, 75, 90, 95)


def calibrate(X: np.ndarray, fire: np.ndarray):
    """P99, d_i und Gewichte w_i laut Paper aus Rohwerten X (n, 6) und Feuer (n,)."""
    p99 = np.percentile(X, 99, axis=0)
    p99 = np.where(p99 > 0, p99, 1.0)
    Xn = np.minimum(X / p99, 1.0)
    sd_n = Xn[~fire].std(0)
    d = np.maximum(0.0, (Xn[fire].mean(0) - Xn[~fire].mean(0)) / np.where(sd_n > 0, sd_n, 1.0))
    w = d / d.sum() if d.sum() > 0 else np.full(X.shape[1], 1 / X.shape[1])
    return p99, d, w


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
    # Gezogen wird aus Land in Kanada (Obermenge unserer Gueltigkeitsmaske);
    # je Pixeltag wird mitgeschrieben, ob er in valid liegt und ob Feuer am
    # Pixel bzw. irgendwo im 13x13-Ausschnitt brennt (so sind die Cache-Labels
    # definiert, Audit s6). Daraus werden mehrere Lesarten von "Trainingsset"
    # und "Feuer" verglichen.
    cube = CubeReader(exp)
    mdir = exp / "derived_data" / "masks"
    valid = np.load(mdir / "valid.npy")
    pool_mask = np.load(mdir / "land.npy") & np.load(mdir / "canada.npy")
    vr, vc = np.nonzero(pool_mask)
    in_valid = valid[vr, vc]
    xs, fp, fpatch, iv = [], [], [], []
    for year in range(args.train_years[0], args.train_years[1] + 1):
        info = cube.per_year[str(year)]
        days = [i for i, d in enumerate(info["days"]) if args.season[0] <= d[5:] <= args.season[1]]
        pick = np.sort(rng.choice(days, size=min(args.days_per_year, len(days)), replace=False))
        dyn, _ = cube._arrays(str(year))
        for li in pick:
            sl = np.asarray(dyn[li])                              # (H, W, 42)
            f2d = sl[:, :, 0] > 0
            fpat = maximum_filter(f2d, size=13)                   # Feuer im 13x13-Ausschnitt
            m = rng.random(vr.size) < args.pixel_frac
            r, c = vr[m], vc[m]
            xs.append(sl[r, c][:, FWI_CH])
            fp.append(f2d[r, c]); fpatch.append(fpat[r, c]); iv.append(in_valid[m])
        print(f"[hnm] {year}: {len(pick)} Tage", flush=True)
    X_all = np.concatenate(xs).astype(np.float64)
    fire_px, fire_patch, is_valid = np.concatenate(fp), np.concatenate(fpatch), np.concatenate(iv)
    print(f"[hnm] Pool Land/Kanada: {X_all.shape[0]:,} Pixeltage; Feuer am Pixel "
          f"{fire_px.sum():,}, im Ausschnitt {fire_patch.sum():,}; davon in valid "
          f"{is_valid.sum():,}")

    variants = {
        "valid/pixel": (is_valid, fire_px),
        "valid/patch": (is_valid, fire_patch),
        "land_canada/pixel": (np.ones_like(is_valid), fire_px),
        "land_canada/patch": (np.ones_like(is_valid), fire_patch),
    }
    weights_by = {}
    for name, (sel, fire_v) in variants.items():
        p99_v, d_v, w_v = calibrate(X_all[sel], fire_v[sel])
        weights_by[name] = {"p99": p99_v.tolist(), "d": d_v.tolist(), "w": w_v.tolist()}

    # Standard fuer die Messung an den Caches: valid/pixel
    X, fire = X_all[is_valid], fire_px[is_valid]
    p99, d, w = calibrate(X, fire)
    Xn = np.minimum(X / p99, 1.0)
    mu_f, mu_n = Xn[fire].mean(0), Xn[~fire].mean(0)
    sd_n = Xn[~fire].std(0)
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
        raw = np.asarray(Xc[:, -1, 19:25, 6, 6]).astype(np.float64)
        tr = split == "train"
        p_c, d_c, w_c = calibrate(raw[tr], y[tr] == 1)
        weights_by[f"cache_train/{cache}"] = {"p99": p_c.tolist(), "d": d_c.tolist(), "w": w_c.tolist()}
        sc = score(raw, p99, w)
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
    print("\n[hnm] Gewichte je Lesart von Trainingsset und Feuer (Ziel laut Paper: FFMC 0,152)")
    print("    " + " " * 26 + "  ".join(f"{n:>5s}" for n in NAMES))
    for name, v in weights_by.items():
        print(f"    {name:26s}" + "  ".join(f"{x:5.3f}" for x in v["w"]))
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
        "weights_by_reading": weights_by, "target_ffmc_weight_paper": 0.152,
    })


if __name__ == "__main__":
    main()
