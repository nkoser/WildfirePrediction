"""Verfahren aus der Feuer- und Survival-Literatur, an unser Protokoll angepasst (nur CPU).

Ergaenzt classic_surv (FWI, LogReg, GBM) um die Modellfamilien, mit denen Feuer bisher als Hazard bzw. Time-to-Event
modelliert wurde. Gleiche Merkmale (linear_tm, 57 + Saison), gleiche Daten, gleiche Auswertung wie classic_surv;
die Methoden werden nach ihrer Beschreibung in den Arbeiten nachgebaut, nicht reproduziert.

  lasso     L1-logistisches zeitdiskretes Hazard-Modell (Person-Perioden wie classic_surv), wie die taeglichen
            Auftretensmodelle fuer British Columbia (Nadeem et al. 2020); C per Holdout
  gam       additives logistisches Hazard-Modell: kubische B-Splines je Merkmal, L2-bestraft (Strafe per Holdout),
            im Sinne der GAM-Auftretensmodelle (Preisler et al. 2004; Woolford et al. 2021). Abweichung: keine
            Lage-Glaettung (Zeile/Spalte gibt es im Paper-Cache nicht im selben Gitter)
  cox       Cox-PH mit Breslow-Bindungen und Breslow-Basis-Hazard, Ridge (Morin et al. 2015; Tremblay et al. 2018)
  weibull   parametrisches Weibull-Modell (AFT = PH bei Weibull), intervallzensierte Likelihood (t-1, t]
  ipcw_gbm  Rezept aus Hegde et al. 2026 (WiDS 2026): je Horizont ein Gradient-Boosting-Klassifikator mit
            IPCW-Gewichten (KM der Zensierung), isotonische Kalibrierung je Horizont auf dem Holdout,
            Monotonie ueber die Horizonte per kumulativem Maximum
  xgb_aft   XGBoost-AFT (log-normal, intervallzensiert), Skala sigma per Holdout; nur wenn xgboost importierbar

Horizonte bzw. Intervalle wie classic_surv (0, 1, 7, 14, 30, 60, 100); Tages-Hazard innerhalb eines Intervalls konstant,
ausser bei cox, weibull, xgb_aft (tagesgenau). Holdout: 20 % der Trainings-Stichtage (fuer C, Strafe, sigma, Isotonie).

  -> metrics/paper_classic/lit_surv_eval<tag>.{json,npz}, lit_paper_metrics<tag>.json

Alle Eingaben nur gelesen; bestehende Ausgaben werden nicht ueberschrieben (Abbruch).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer

from wildfire.audit.common import assert_readonly, load_config
from wildfire.data.dataset import CubeReader
from wildfire.eval.metrics import evaluate
from wildfire.paper import targets
from wildfire.paper.classic_surv import (C, EDGES, NI, REPORTED_UNET, Design, F_at, compact, features, onehot,
                                         person_period, windows_features)
from wildfire.paper.compare import DTS
from wildfire.paper.metrics import paper_metrics
from wildfire.paper.paper_metrics_ckpt import calibrate
from wildfire.paper.region_eval_crn import nll_per_sample
from wildfire.paper.sts_dctm import doy_of

try:
    import xgboost
except ImportError:                                                     # optional, siehe 122_...sbatch
    xgboost = None

H = 100
MODELS = ("lasso", "gam", "cox", "weibull", "ipcw_gbm", "xgb_aft")
LASSO_C = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
GAM_C = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
AFT_SIGMA = (0.5, 1.0, 2.0, 4.0)
ISO_MIN = 1e-5


# ---------------------------------------------------------------------------- Hilfen
def interval_hazards(p):
    """Bedingte Intervall-Wahrscheinlichkeiten (n, NI) -> Tages-Hazards (n, H), je Intervall konstant."""
    h = np.empty((len(p), H))
    for j in range(NI):
        ln = EDGES[j + 1] - EDGES[j]
        h[:, EDGES[j]:EDGES[j + 1]] = -np.expm1(np.log1p(-np.clip(p[:, [j]], 1e-12, 1 - 1e-12)) / ln)
    return h


def logS_hazards(logS):
    """log S an den Tagen 0..H (n, H+1) -> Tages-Hazards (n, H)."""
    return np.clip(-np.expm1(np.diff(logS, axis=1)), 1e-12, 1 - 1e-12)


def pp_logloss(m, X, y):
    p = np.clip(m.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    return float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean())


def lbfgs(params, loss_fn, iters=200):
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=iters, tolerance_grad=1e-9, tolerance_change=1e-12,
                            history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return float(loss_fn())


# ---------------------------------------------------------------------------- Person-Perioden: lasso, gam
class PersonPeriod:
    """Logistisches zeitdiskretes Hazard-Modell mit Intervall-Dummies (Proportional Odds je Intervall)."""

    def __init__(self, kind, grid, args):
        self.kind, self.grid, self.args = kind, grid, args

    def basis(self, Z):
        x = self.D.std(Z)
        if self.kind == "lasso":
            return x
        k = self.D.mu.size
        return np.column_stack([self.spl.transform(x[:, :k]), x[:, k:]])        # Splines + Fehlend-Merkmale

    def design(self, B, j):
        return np.column_stack([onehot(j), B])

    def model(self, c):
        if self.kind == "lasso":
            return LogisticRegression(penalty="l1", solver="saga", C=c, max_iter=5000, tol=1e-4, fit_intercept=False)
        return LogisticRegression(C=c, max_iter=5000, fit_intercept=False)

    def fit(self, Z, ev, t, hold):
        self.D = Design(Z[~hold])
        if self.kind == "gam":
            self.spl = SplineTransformer(n_knots=5, degree=3, extrapolation="linear")
            self.spl.fit(self.D.std(Z[~hold])[:, :self.D.mu.size])
        B = self.basis(Z)
        tr, ho = np.flatnonzero(~hold), np.flatnonzero(hold)
        s_tr, j_tr, y_tr = person_period(tr.size, ev[tr], t[tr])
        s_ho, j_ho, y_ho = person_period(ho.size, ev[ho], t[ho])
        X_tr, X_ho = self.design(B[tr][s_tr], j_tr), self.design(B[ho][s_ho], j_ho)
        self.scores = {}
        for c in self.grid:
            self.scores[c] = pp_logloss(self.model(c).fit(X_tr, y_tr), X_ho, y_ho)
            print(f"[lit]   {self.kind} C={c:g}: Holdout-Logloss {self.scores[c]:.5f}", flush=True)
        self.c = min(self.scores, key=self.scores.get)
        s, j, y = person_period(len(Z), ev, t)                                    # Neuanpassung auf allen
        self.m = self.model(self.c).fit(self.design(B[s], j), y)
        return self

    def hazards(self, Z):
        B = self.basis(Z)
        p = np.column_stack([self.m.predict_proba(self.design(B, np.full(len(Z), j)))[:, 1] for j in range(NI)])
        return interval_hazards(p)

    def info(self):
        return {"C": self.c, "holdout_logloss": {str(k): v for k, v in self.scores.items()},
                "nonzero": int((np.abs(self.m.coef_) > 0).sum()), "n_coef": int(self.m.coef_.size)}


# ---------------------------------------------------------------------------- Cox-PH (Breslow)
class Cox:
    def __init__(self, args):
        self.l2 = args.cox_l2

    def fit(self, Z, ev, t, hold):
        self.D = Design(Z)
        X = torch.tensor(self.D.std(Z))
        E = torch.tensor(ev, dtype=torch.float64)
        T = torch.tensor(t, dtype=torch.long)
        d = torch.bincount(T[E > 0], minlength=H + 1).double()
        beta = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
        n = len(Z)

        def risk(eta):
            m = eta.max().detach()
            r = torch.zeros(H + 1, dtype=torch.float64).index_add(0, T, torch.exp(eta - m))
            return r.flip(0).cumsum(0).flip(0), m                                  # R[s] = sum_{t_i >= s} e^eta

        def loss():
            eta = X @ beta
            R, m = risk(eta)
            ll = (eta * E).sum() - (d * (torch.log(R.clamp_min(1e-300)) + m)).sum()
            return -ll / n + 0.5 * self.l2 * beta.pow(2).sum()
        self.loss = lbfgs([beta], loss, iters=500)
        with torch.no_grad():
            eta = X @ beta
            R, m = risk(eta)
            # Breslow, Skala e^(eta - m); je Intervall (EDGES) gepoolt, sonst ist der Basis-Hazard an Tagen ohne
            # Ereignis im Training exakt 0 (wie die Person-Perioden-Modelle: konstant je Intervall)
            dd, RR = d.numpy()[1:H + 1], R.numpy()[1:H + 1]
            self.dH0 = np.concatenate([np.full(EDGES[j + 1] - EDGES[j], dd[EDGES[j]:EDGES[j + 1]].sum()
                                               / max(RR[EDGES[j]:EDGES[j + 1]].sum(), 1e-300)) for j in range(NI)])
            self.m = float(m)
        self.beta = beta.detach().numpy()
        return self

    def hazards(self, Z):
        r = np.exp(np.clip(self.D.std(Z) @ self.beta - self.m, -700, 50))
        return np.clip(-np.expm1(-self.dH0[None] * r[:, None]), 1e-12, 1 - 1e-12)

    def info(self):
        return {"l2": self.l2, "train_loss": self.loss}


# ---------------------------------------------------------------------------- Weibull (intervallzensiert)
class Weibull:
    def __init__(self, args):
        self.l2 = args.cox_l2

    def logS(self, X, u):
        lam = self.b0 + X @ self.beta                                              # log Skala
        return -torch.exp(torch.exp(self.logk) * (torch.log(u.clamp_min(1e-12)) - lam)) * (u > 0)

    def fit(self, Z, ev, t, hold):
        self.D = Design(Z)
        X = torch.tensor(self.D.std(Z))
        E = torch.tensor(ev, dtype=torch.bool)
        T = torch.tensor(t, dtype=torch.float64)
        self.beta = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
        self.b0 = torch.tensor(np.log(1000.0), dtype=torch.float64, requires_grad=True)
        self.logk = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)

        def loss():
            s1, s0 = self.logS(X, T), self.logS(X, T - 1)
            ev_ll = s0 + torch.log(-torch.expm1((s1 - s0).clamp_max(-1e-12)))
            ll = torch.where(E, ev_ll, s1)
            return -ll.mean() + 0.5 * self.l2 * self.beta.pow(2).sum()
        self.loss = lbfgs([self.beta, self.b0, self.logk], loss, iters=500)
        return self

    def hazards(self, Z):
        X = torch.tensor(self.D.std(Z))
        u = torch.arange(H + 1, dtype=torch.float64)
        with torch.no_grad():
            lS = torch.stack([self.logS(X, torch.full((len(Z),), float(v), dtype=torch.float64)) for v in u], 1)
        return logS_hazards(lS.numpy())

    def info(self):
        return {"l2": self.l2, "shape_k": float(torch.exp(self.logk)), "train_loss": self.loss}


# ---------------------------------------------------------------------------- IPCW-GBM je Horizont (Hegde et al.)
def censoring_km(ev, t):
    """KM der Zensierungsverteilung G(s), s = 0..H (Zensierung = kein Ereignis bis t)."""
    c = np.bincount(t[~ev], minlength=H + 1)[:H + 1].astype(float)
    at_risk = np.bincount(t, minlength=H + 1)[:H + 1][::-1].cumsum()[::-1].astype(float)
    return np.cumprod(1 - np.divide(c, at_risk, out=np.zeros(H + 1), where=at_risk > 0))


def ipcw(ev, t, tau, G):
    """Label, Auswahl und Gewicht fuer den Horizont tau (vor tau Zensierte fallen heraus)."""
    y = ev & (t <= tau)
    keep = y | (t >= tau)
    w = 1.0 / np.clip(G[np.minimum(t, tau) - 1], 0.05, None)
    return y.astype(int), keep, w


class IpcwGBM:
    def __init__(self, args):
        self.args = args

    def fit(self, Z, ev, t, hold):
        a = self.args
        G = censoring_km(ev[~hold], t[~hold])
        self.models, self.iso = [], []
        for tau in EDGES[1:]:
            y, keep, w = ipcw(ev, t, tau, G)
            tr, ho = keep & ~hold, keep & hold
            m = HistGradientBoostingClassifier(max_iter=a.gbm_iter, learning_rate=0.05, max_leaf_nodes=31,
                                               min_samples_leaf=200, l2_regularization=1.0, early_stopping=True,
                                               validation_fraction=0.1, random_state=a.seed)
            m.fit(Z[tr], y[tr], sample_weight=w[tr])
            iso = IsotonicRegression(y_min=ISO_MIN, y_max=1.0 - ISO_MIN, out_of_bounds="clip")   # nie exakt 0
            iso.fit(m.predict_proba(Z[ho])[:, 1], y[ho], sample_weight=w[ho])
            self.models.append(m)
            self.iso.append(iso)
        self.G100 = float(G[H])
        return self

    def F_edges(self, Z):
        F = np.column_stack([iso.predict(m.predict_proba(Z)[:, 1]) for m, iso in zip(self.models, self.iso)])
        return np.maximum.accumulate(np.clip(F, 0, 1 - 1e-6), axis=1)              # Monotonie ueber Horizonte

    def hazards(self, Z):
        F = self.F_edges(Z)
        S = np.column_stack([np.ones(len(Z)), 1 - F])
        # wo das kumulative Maximum F flach macht, waere der Intervall-Hazard 0 (Ereignis dort -> NLL unendlich)
        p = np.maximum(1 - S[:, 1:] / np.clip(S[:, :-1], 1e-12, None), ISO_MIN)
        return interval_hazards(p)

    def info(self):
        return {"n_iter": [int(m.n_iter_) for m in self.models], "G_censor_100": self.G100}


# ---------------------------------------------------------------------------- XGBoost-AFT
class XgbAFT:
    def __init__(self, args):
        self.args = args

    def dmat(self, X, ev=None, t=None):
        d = xgboost.DMatrix(X)
        if ev is not None:
            d.set_float_info("label_lower_bound", np.where(ev, t - 1, t).astype(float))
            d.set_float_info("label_upper_bound", np.where(ev, t, np.inf).astype(float))
        return d

    def fit(self, Z, ev, t, hold):
        a = self.args
        self.D = Design(Z[~hold])
        X = self.D.std(Z)
        dtr, dho = self.dmat(X[~hold], ev[~hold], t[~hold]), self.dmat(X[hold], ev[hold], t[hold])
        self.scores, self.boosters = {}, {}
        for sg in AFT_SIGMA:
            p = {"objective": "survival:aft", "eval_metric": "aft-nloglik", "aft_loss_distribution": "normal",
                 "aft_loss_distribution_scale": sg, "tree_method": "hist", "learning_rate": 0.05, "max_depth": 6,
                 "min_child_weight": 10, "lambda": 1.0, "nthread": a.threads, "seed": a.seed}
            b = xgboost.train(p, dtr, num_boost_round=2000, evals=[(dho, "ho")], early_stopping_rounds=50,
                              verbose_eval=False)
            self.scores[sg] = float(b.best_score)
            self.boosters[sg] = b
            print(f"[lit]   xgb_aft sigma={sg:g}: Holdout-NLL {b.best_score:.5f} ({b.best_iteration + 1} Baeume)",
                  flush=True)
        self.sigma = min(self.scores, key=self.scores.get)
        self.b = self.boosters[self.sigma]
        return self

    def hazards(self, Z):
        mu = np.log(np.clip(self.b.predict(self.dmat(self.D.std(Z)),
                                           iteration_range=(0, self.b.best_iteration + 1)), 1e-12, None))
        u = np.arange(1, H + 1, dtype=float)
        lS = norm.logsf((np.log(u)[None] - mu[:, None]) / self.sigma)
        return logS_hazards(np.column_stack([np.zeros(len(Z)), lS]))

    def info(self):
        return {"sigma": self.sigma, "holdout_nll": {str(k): v for k, v in self.scores.items()},
                "n_trees": int(self.b.best_iteration + 1)}


def make(kind, args):
    if kind == "lasso":
        return PersonPeriod("lasso", LASSO_C, args)
    if kind == "gam":
        return PersonPeriod("gam", GAM_C, args)
    return {"cox": Cox, "weibull": Weibull, "ipcw_gbm": IpcwGBM, "xgb_aft": XgbAFT}[kind](args)


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
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--gbm-iter", type=int, default=400)
    ap.add_argument("--cox-l2", type=float, default=1e-4)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-paper", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    out_dir = exp / "metrics" / "paper_classic"
    outs = [out_dir / f"lit_surv_eval{args.tag}.json", out_dir / f"lit_surv_eval{args.tag}.npz",
            out_dir / f"lit_paper_metrics{args.tag}.json"]
    for p in outs:
        if p.exists():
            raise SystemExit(f"ABBRUCH: {p} existiert schon - anderes --tag waehlen")
    models = [k for k in args.models if k != "xgb_aft" or xgboost is not None]
    if len(models) < len(args.models):
        print("[lit] xgboost nicht importierbar -> xgb_aft entfaellt", flush=True)
    else:
        print(f"[lit] xgboost {getattr(xgboost, '__version__', '-')}", flush=True)
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    # ---- Anpassung: zufaellige Wuerfel-Stichtage (wie classic_surv)
    cdir = exp / "derived_data" / args.cube_dir
    meta = np.load(cdir / "train_meta.npz")
    rows = np.where(meta["kind"] == 1)[0]
    if args.max_train and rows.size > args.max_train:
        rows = np.sort(rng.choice(rows, args.max_train, replace=False))
    dates = CubeReader(exp).dates
    doy = np.array([dt.date.fromisoformat(dates[int(L)]).timetuple().tm_yday for L in meta["L"][rows]])
    Z_tr, _ = features(np.load(cdir / "train_X.npy", mmap_mode="r"), doy, rows)
    ev_px, t_px = np.load(cdir / "train_event.npy")[rows], np.load(cdir / "train_t.npy")[rows].astype(np.int64)
    y_tr = {"centre": (ev_px[:, C, C].astype(bool), t_px[:, C, C]),
            "patch": (ev_px.reshape(rows.size, -1).any(1), t_px.reshape(rows.size, -1).min(1))}
    hold = rng.random(rows.size) < args.holdout
    print(f"[lit] Training: {rows.size:,} Stichtage (Holdout {hold.sum():,}); t min {t_px.min()} max {t_px.max()}; "
          f"Ereignisrate Ausschnitt {y_tr['patch'][0].mean():.4f} ({time.time() - t0:.0f} s)", flush=True)

    # ---- Test: zufaellige Stichtage der Testjahre
    sdir = exp / "derived_data" / args.surv_dir
    em = dict(np.load(sdir / "eval_meta.npz"))
    te = np.arange(len(em["row"]))
    if args.max_test:
        te = rng.choice(te, min(args.max_test, te.size), replace=False)          # zufaellig, nicht die ersten
        te.sort()
    Z_te, _ = features(np.load(sdir / "eval_X.npy", mmap_mode="r"), em["doy"][te], te)
    y_te = {"centre": (em["ev_centre"][te].astype(bool), em["t_centre"][te].astype(int)),
            "patch": (em["ev_patch"][te].astype(bool), em["t_patch"][te].astype(int))}
    year = np.array([(dt.date(2000, 1, 1) + dt.timedelta(days=int(v))).year for v in em["L_abs"][te]])
    print(f"[lit] Test: {te.size:,} Stichtage ({time.time() - t0:.0f} s)", flush=True)

    # ---- Paper-Protokoll (wie classic_surv)
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
            idx = {s: np.sort(rng.choice(v, min(args.max_test, v.size), replace=False)) for s, v in idx.items()}
        lab = {"patch": yl, "centre": fire_t[:, C, C].astype(int)}
        feats = {}
        for dt_ in DTS:
            e = L - dt_ - 2
            for s, r in idx.items():
                feats[(dt_, s)] = windows_features(Xp, r, e, doyD[r] - (LC - 1 - e))[0]
        pp = (idx, lab, feats)
        print(f"[lit] Paper-Protokoll {args.cache} {args.target}: Val {idx['val'].size:,}, Test {idx['test'].size:,} "
              f"({time.time() - t0:.0f} s)", flush=True)

    res = {"args": vars(args), "n_train": int(rows.size), "n_holdout": int(hold.sum()), "n_test": int(te.size),
           "intervals": list(EDGES), "surv": {}, "surv_by_year": {}, "fit_s": {}, "fit_info": {}}
    pres = {"args": vars(args), "reported_unet_pr_auc_patch": REPORTED_UNET, "results": {}}
    store = {"year": year}
    for kind in models:
        for target in ("centre", "patch"):
            t1 = time.time()
            m = make(kind, args).fit(Z_tr, *y_tr[target], hold)
            res["fit_s"][f"{kind}_{target}"] = round(time.time() - t1, 1)
            res["fit_info"][f"{kind}_{target}"] = m.info()
            h = m.hazards(Z_te)
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
            print(f"[lit] {kind:8s} {target:6s} Test: NLL {a['nll']:.4f}  IBS {a['ibs']:.5f}  C {a['c_index']:.3f}  "
                  f"AUC@7 {a['auc@7']:.3f}  PR@7 {a['pr@7']:.3f}  PR@30 {a['pr@30']:.3f}  "
                  f"(Fit {res['fit_s'][f'{kind}_{target}']:.0f} s)  {res['fit_info'][f'{kind}_{target}']}",
                  flush=True)
            if pp is None:
                continue
            idx, lab, feats = pp
            for dt_ in DTS:
                sc = {s: F_at(m.hazards(feats[(dt_, s)]), dt_ + 1) for s in idx}
                yv, yt = lab[target][idx["val"]], lab[target][idx["test"]]
                cal, how = calibrate(sc["val"], yv, sc["test"])
                r = {"raw": paper_metrics(sc["test"], yt), "cal": paper_metrics(cal, yt), "cal_choice": how}
                pres["results"].setdefault(kind, {}).setdefault(str(dt_), {})[target] = r
                rep = f"  (3D U-Net laut Paper {REPORTED_UNET[dt_]:.3f})" if target == "patch" else ""
                print(f"[lit]   Paper dt={dt_:2d} {target:6s} PR-AUC {r['raw']['pr_auc']:.3f}  "
                      f"cal: F1 {r['cal']['f1']:.3f} ECE {r['cal']['ece']:.3f} Brier {r['cal']['brier']:.4f}{rep}",
                      flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    outs[0].write_text(json.dumps(res, indent=1, default=float))
    np.savez_compressed(outs[1], idx=te, **store)
    if pp is not None:
        outs[2].write_text(json.dumps(pres, indent=1, default=float))
    print("[lit] " + "\n[lit] ".join(str(p) for p in outs), flush=True)


if __name__ == "__main__":
    main()
