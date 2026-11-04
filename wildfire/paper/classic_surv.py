"""Klassische Vergleichsverfahren als zeitdiskrete Hazard-Modelle: FWI, FWI-Gruppe, logistische Regression, GBM.

Die klassischen Baselines der Literatur (logistische Regression, Gradient Boosting, der operationelle FWI-Index)
sagen meist eine Wahrscheinlichkeit je Horizont vorher. Damit sie mit denselben Survival-Metriken bewertet werden
koennen wie STS-DCTM, linear_tm und KM, werden sie hier als klassisches zeitdiskretes Hazard-Modell angepasst
(Person-Perioden-Daten, Singer & Willett): je Stichtag und Intervall [0,1], (1,7], (7,14], (14,30], (30,60], (60,100]
ein Bernoulli "Feuer in diesem Intervall, wenn bis zu seinem Beginn keins war"; innerhalb eines Intervalls ist der
Tages-Hazard konstant.

  fwi        nur FWI der Mittelzelle am Stichtag (+ Fehlend-Merkmal), logistisch mit Intervall-Dummies und
             FWI x Intervall
  fwi_group  FFMC, DMC, DC, ISI, BUI, FWI am Stichtag, sonst wie fwi
  lr         logistisch auf allen Merkmalen von linear_tm (57) + Intervall-Dummies (Proportional Odds je Intervall)
  gbm        HistGradientBoosting auf allen Merkmalen und dem Intervall-Index (Wechselwirkungen frei)

FWI und ISI: 255 an allen 30 Fenstertagen ist NoData (rund 4 % der Landzellen sind immer 255, p31_fwi255_check)
und wird fehlend gesetzt; einzelne 255 bleiben als Kappung stehen. DC und BUI bleiben unveraendert.

Daten wie linear_tm: Anpassung auf den zufaelligen Wuerfel-Stichtagen 2000-2019 (natuerliche Haeufigkeit),
Auswertung (a) auf den zufaelligen Test-Stichtagen 2023-2025 (surv_eval, voller Metriksatz) und (b) im Protokoll des
Papers (Score F(dt + 1) auf den Paper-Stichproben, roh und wie alle Modelle per Val-Split kalibriert; daneben die
im Paper berichtete PR-AUC des 3D U-Net). Ziele: Mittelzelle und Ausschnitt, je ein Modell.

  -> metrics/paper_classic/surv_eval<tag>.{json,npz}, paper_metrics<tag>.json   (nur CPU)

Alle Eingaben nur gelesen; bestehende Ausgaben werden nicht ueberschrieben (Abbruch).
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

from wildfire.audit.common import assert_readonly, load_config
from wildfire.data.dataset import CubeReader
from wildfire.eval.metrics import evaluate
from wildfire.paper import targets
from wildfire.paper.compare import DTS
from wildfire.paper.linear_tm import CH, NAMES
from wildfire.paper.metrics import paper_metrics
from wildfire.paper.paper_metrics_ckpt import calibrate
from wildfire.paper.pixel_stdctm import windows
from wildfire.paper.prepare import CHANNEL_NAMES, FIRE_CH
from wildfire.paper.region_eval_crn import nll_per_sample
from wildfire.paper.sts_dctm import doy_of

H = 100
C = 6
EDGES = (0, 1, 7, 14, 30, 60, 100)
NI = len(EDGES) - 1
FWI_GROUP = ("FFMC", "DMC", "DC", "ISI", "BUI", "FWI")
NODATA_255 = ("ISI", "FWI")
REPORTED_UNET = {1: 0.718, 7: 0.598, 15: 0.551, 30: 0.481}           # 3D U-Net laut Paper, PR-AUC (Ausschnitt)
MODELS = ("fwi", "fwi_group", "lr", "gbm")


# ---------------------------------------------------------------------------- Merkmale
def features(X, doy, rows=None, chunk=4096):
    """Wie linear_tm.features, dazu FWI-Gruppe am Stichtag; NoData (255 an allen Tagen) bei FWI und ISI -> NaN.

    -> (alle (n, len(NAMES)), fwi_group (n, 6))."""
    rows = np.arange(X.shape[0]) if rows is None else np.asarray(rows)
    gi = [CHANNEL_NAMES.index(c) for c in FWI_GROUP]
    out, grp = [], []
    for a in range(0, rows.size, chunk):
        r = rows[a:a + chunk]
        xw = np.asarray(X[r, :, :, C, C], np.float64)                       # (b, 30, 55) Mittelzelle
        nod = {c: (xw[:, :, CHANNEL_NAMES.index(c)] == 255).all(1) for c in NODATA_255}
        xc = xw[:, -7:]                                                       # (b, 7, 55)
        fire = np.asarray(X[r, :, FIRE_CH]) > 0                               # (b, 30, 13, 13)
        z = np.column_stack([xc[:, :, CH].mean(1), np.log1p(fire.sum((1, 2, 3))), fire[:, -7:, C, C].any(1)])
        g = xw[:, -1, gi].copy()
        for c, m in nod.items():
            z[m, CH.index(CHANNEL_NAMES.index(c))] = np.nan
            g[m, FWI_GROUP.index(c)] = np.nan
        out.append(z)
        grp.append(g)
    ang = 2 * np.pi * np.asarray(doy, np.float64) / 365.25
    return np.column_stack([np.concatenate(out), np.sin(ang), np.cos(ang)]), np.concatenate(grp)


def windows_features(X, rows, e, doy, chunk=2048):
    """Paper-Protokoll: Fenster mit letztem Tag e je Zeile aus dem 61-Tage-Cache."""
    za, zg = [], []
    for a in range(0, rows.size, chunk):
        r = rows[a:a + chunk]
        w = windows(X, r, np.full(r.size, e))
        z, g = features(w, doy[a:a + chunk])
        za.append(z)
        zg.append(g)
    return np.concatenate(za), np.concatenate(zg)


# ---------------------------------------------------------------------------- Person-Perioden
def person_period(n, ev, t):
    """Zeilen (Stichtag, Intervall) mit Risiko am Intervallbeginn, Ziel Feuer im Intervall."""
    i_s, i_j, y = [], [], []
    for j in range(NI):
        risk = t > EDGES[j]
        s = np.flatnonzero(risk)
        i_s.append(s)
        i_j.append(np.full(s.size, j))
        y.append(ev[s] & (t[s] <= EDGES[j + 1]))
    return np.concatenate(i_s), np.concatenate(i_j), np.concatenate(y).astype(int)


class Design:
    """Standardisierung, Imputation (Trainings-Median) und Fehlend-Merkmale; Design je Modellart."""

    def __init__(self, Z):
        self.med = np.nanmedian(Z, 0)
        self.med = np.where(np.isfinite(self.med), self.med, 0.0)
        Zi = np.where(np.isnan(Z), self.med, Z)
        self.mu, self.sd = Zi.mean(0), Zi.std(0)
        self.sd = np.where(self.sd > 0, self.sd, 1.0)
        self.miss = np.flatnonzero(np.isnan(Z).any(0))

    def std(self, Z):
        Zi = (np.where(np.isnan(Z), self.med, Z) - self.mu) / self.sd
        return np.column_stack([Zi, np.isnan(Z[:, self.miss]).astype(float)])


def onehot(j):
    return np.eye(NI)[j]


def design(kind, D, Z, j):
    x = D.std(Z)
    if kind in ("fwi", "fwi_group"):                                  # mit Intervall-Wechselwirkung
        oh = onehot(j)
        return np.column_stack([oh, x, (x[:, :, None] * oh[:, None, :]).reshape(len(j), -1)])
    return np.column_stack([onehot(j), x])


def fit_model(kind, Z, ev, t, args):
    s, j, y = person_period(len(Z), ev, t)
    if kind == "gbm":
        m = HistGradientBoostingClassifier(max_iter=args.gbm_iter, learning_rate=0.05, max_leaf_nodes=31,
                                           min_samples_leaf=200, l2_regularization=1.0, early_stopping=True,
                                           validation_fraction=0.1, random_state=args.seed)
        m.fit(np.column_stack([Z[s], j]), y)
        return m, None
    D = Design(Z)
    m = LogisticRegression(C=args.lr_c, max_iter=2000, fit_intercept=False)
    m.fit(design(kind, D, Z[s], j), y)
    return m, D


def hazards(kind, m, D, Z):
    """-> Tages-Hazards (n, H), innerhalb eines Intervalls konstant."""
    n = len(Z)
    p = np.empty((n, NI))
    for j in range(NI):
        jj = np.full(n, j)
        x = np.column_stack([Z, jj]) if kind == "gbm" else design(kind, D, Z, jj)
        p[:, j] = m.predict_proba(x)[:, 1]
    h = np.empty((n, H))
    for j in range(NI):
        ln = EDGES[j + 1] - EDGES[j]
        h[:, EDGES[j]:EDGES[j + 1]] = -np.expm1(np.log1p(-np.clip(p[:, [j]], 1e-12, 1 - 1e-12)) / ln)
    return h


def F_at(h, tau):
    return -np.expm1(np.log1p(-np.clip(h[:, :tau], 0, 1 - 1e-12)).sum(1))


def feats_of(kind, za, zg):
    return {"fwi": zg[:, [FWI_GROUP.index("FWI")]], "fwi_group": zg}.get(kind, za)


def compact(e):
    return {"nll": e["nll"], "ibs": e["integrated_brier"], "c_index": e["concordance_index"], "c_td": e["c_index_td"],
            "dcal_dev": e["d_cal_max_dev"], "dcal_p": e["d_cal_p"],
            **{f"{q}@{tau}": e["horizons"][str(tau)][key] for tau in (1, 7, 30, 100)
               for q, key in (("auc", "roc_auc"), ("pr", "pr_auc"), ("brier", "brier"), ("ece", "ece"))}}


# ---------------------------------------------------------------------------- Lauf
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cube-dir", default="paper_pixel")
    ap.add_argument("--surv-dir", default="surv_eval_test")
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--target", choices=targets.TARGETS, default="next_patch")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    ap.add_argument("--max-train", type=int, default=0, help="hoechstens so viele Trainings-Stichtage (0 = alle)")
    ap.add_argument("--max-test", type=int, default=0, help="Rauchtest: nur so viele Test-Stichtage bzw. Stichproben")
    ap.add_argument("--gbm-iter", type=int, default=400)
    ap.add_argument("--lr-c", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-paper", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    out_dir = exp / "metrics" / "paper_classic"
    outs = [out_dir / f"surv_eval{args.tag}.json", out_dir / f"surv_eval{args.tag}.npz",
            out_dir / f"paper_metrics{args.tag}.json"]
    for p in outs:
        if p.exists():
            raise SystemExit(f"ABBRUCH: {p} existiert schon - anderes --tag waehlen")
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    # ---- Anpassung: zufaellige Wuerfel-Stichtage (wie linear_tm)
    cdir = exp / "derived_data" / args.cube_dir
    meta = np.load(cdir / "train_meta.npz")
    rows = np.where(meta["kind"] == 1)[0]
    if args.max_train and rows.size > args.max_train:
        rows = np.sort(rng.choice(rows, args.max_train, replace=False))
    dates = CubeReader(exp).dates
    doy = np.array([dt.date.fromisoformat(dates[int(L)]).timetuple().tm_yday for L in meta["L"][rows]])
    Za_tr, Zg_tr = features(np.load(cdir / "train_X.npy", mmap_mode="r"), doy, rows)
    ev_px, t_px = np.load(cdir / "train_event.npy")[rows], np.load(cdir / "train_t.npy")[rows].astype(np.int64)
    y_tr = {"centre": (ev_px[:, C, C].astype(bool), t_px[:, C, C]),
            "patch": (ev_px.reshape(rows.size, -1).any(1), t_px.reshape(rows.size, -1).min(1))}
    print(f"[cls] Training: {rows.size:,} zufaellige Stichtage; FWI fehlend {np.isnan(Zg_tr[:, -1]).mean():.4f}; "
          f"Ereignisrate 100 d Mitte {y_tr['centre'][0].mean():.4f}, Ausschnitt {y_tr['patch'][0].mean():.4f} "
          f"({time.time() - t0:.0f} s)", flush=True)

    # ---- Test: zufaellige Stichtage der Testjahre
    sdir = exp / "derived_data" / args.surv_dir
    em = dict(np.load(sdir / "eval_meta.npz"))
    te = np.arange(len(em["row"]))
    if args.max_test:
        te = te[:args.max_test]
    Za_te, Zg_te = features(np.load(sdir / "eval_X.npy", mmap_mode="r"), em["doy"][te], te)
    y_te = {"centre": (em["ev_centre"][te].astype(bool), em["t_centre"][te].astype(int)),
            "patch": (em["ev_patch"][te].astype(bool), em["t_patch"][te].astype(int))}
    year = np.array([(dt.date(2000, 1, 1) + dt.timedelta(days=int(v))).year for v in em["L_abs"][te]])
    print(f"[cls] Test: {te.size:,} Stichtage ({time.time() - t0:.0f} s)", flush=True)

    # ---- Paper-Protokoll: Stichproben und Labels
    pp = None
    if not args.skip_paper:
        d = exp / "derived_data" / "paper" / f"{args.cache}_L61"
        Xp = np.load(d / "X.npy", mmap_mode="r")
        yl = targets.labels(d, args.target)
        split = np.load(d / "split.npy")
        doyD = doy_of(np.load(d / "meta.npy")[:, 0])
        fire_t = targets.fire_map(d, Xp, args.target)
        LC = Xp.shape[1]
        L = LC + targets.shift(args.target)
        idx = {s: np.where(split == s)[0] for s in ("val", "test")}
        ok = targets.ok_mask(d)
        if ok is not None:
            idx = {s: v[ok[v]] for s, v in idx.items()}
        if args.max_test:
            idx = {s: v[:args.max_test] for s, v in idx.items()}
        lab = {"patch": yl, "centre": fire_t[:, C, C].astype(int)}
        feats = {}
        for dt_ in DTS:
            e = L - dt_ - 2
            for s, r in idx.items():
                feats[(dt_, s)] = windows_features(Xp, r, e, doyD[r] - (LC - 1 - e))
        pp = (idx, lab, feats)
        print(f"[cls] Paper-Protokoll {args.cache} {args.target}: Val {idx['val'].size:,}, Test {idx['test'].size:,} "
              f"({time.time() - t0:.0f} s)", flush=True)

    res = {"args": vars(args), "n_train": int(rows.size), "n_test": int(te.size), "intervals": list(EDGES),
           "surv": {}, "surv_by_year": {}, "fit_s": {}}
    pres = {"args": vars(args), "reported_unet_pr_auc_patch": REPORTED_UNET, "results": {}}
    store = {"year": year}
    for kind in args.models:
        for target in ("centre", "patch"):
            t1 = time.time()
            m, D = fit_model(kind, feats_of(kind, Za_tr, Zg_tr), *y_tr[target], args)
            res["fit_s"][f"{kind}_{target}"] = round(time.time() - t1, 1)
            h = hazards(kind, m, D, feats_of(kind, Za_te, Zg_te))
            e, t = y_te[target]
            res["surv"].setdefault(kind, {})[target] = compact(evaluate(h, e, t))
            for y_ in np.unique(year):
                msk = year == y_
                res["surv_by_year"].setdefault(kind, {}).setdefault(target, {})[str(y_)] = compact(
                    evaluate(h[msk], e[msk], t[msk]))
            store[f"{kind}__{target}__nll"] = nll_per_sample(h, e, t).astype(np.float32)
            for tau in (1, 7, 14, 30, 60, 100):
                store[f"{kind}__{target}__F{tau}"] = F_at(h, tau).astype(np.float32)
            a = res["surv"][kind][target]
            print(f"[cls] {kind:9s} {target:6s} Test: NLL {a['nll']:.4f}  IBS {a['ibs']:.5f}  C {a['c_index']:.3f}  "
                  f"AUC@7 {a['auc@7']:.3f}  PR@7 {a['pr@7']:.3f}  PR@30 {a['pr@30']:.3f}  "
                  f"(Fit {res['fit_s'][f'{kind}_{target}']:.0f} s)", flush=True)
            if pp is None:
                continue
            idx, lab, feats = pp
            for dt_ in DTS:
                sc = {s: F_at(hazards(kind, m, D, feats_of(kind, *feats[(dt_, s)])), dt_ + 1) for s in idx}
                yv, yt = lab[target][idx["val"]], lab[target][idx["test"]]
                cal, how = calibrate(sc["val"], yv, sc["test"])
                r = {"raw": paper_metrics(sc["test"], yt), "cal": paper_metrics(cal, yt), "cal_choice": how}
                pres["results"].setdefault(kind, {}).setdefault(str(dt_), {})[target] = r
                rep = f"  (3D U-Net laut Paper {REPORTED_UNET[dt_]:.3f})" if target == "patch" else ""
                print(f"[cls]   Paper dt={dt_:2d} {target:6s} PR-AUC {r['raw']['pr_auc']:.3f}  "
                      f"cal: F1 {r['cal']['f1']:.3f} ECE {r['cal']['ece']:.3f} Brier {r['cal']['brier']:.4f}{rep}",
                      flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    outs[0].write_text(json.dumps(res, indent=1))
    np.savez_compressed(outs[1], idx=te, **store)
    if pp is not None:
        outs[2].write_text(json.dumps(pres, indent=1))
    print("[cls] " + "\n[cls] ".join(str(p) for p in outs), flush=True)


if __name__ == "__main__":
    main()
