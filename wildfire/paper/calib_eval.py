"""Kalibrierung genauer ansehen: nach Gruppen, nachtraeglich ueber die PIT-Verteilung, konformal.

Auf denselben Stichtagen wie surv_eval (natuerliche Haeufigkeit; Kalibrierung auf den
Val-Jahren 2020-2022, Test 2023-2025), je Modell das Ensemble der Seeds:

  1  Gruppen      D-Calibration und beobachtet / erwartet bei 7, 30, 100 Tagen getrennt nach
                  Testjahr, Monat des Stichtags, Feuer im 30-Tage-Fenster und vier Laengsbaendern
                  des Rasters (Spalten-Quartile, West nach Ost). Ein kleines D-Cal im Mittel kann
                  Fehler in einzelnen Gruppen verdecken.
  2  Varianten    raw    das Modell wie trainiert
                  curve  Lambda' = e^a * Lambda^b (wie surv_eval --calibrate)
                  pit    PIT-Rekalibrierung nach dem Prinzip von CiPOT (Qi, Yu, Greiner, NeurIPS
                         2024): G = Verteilungsfunktion der Val-PIT-Werte (Ereignis an Tag t:
                         gleichverteilt auf [F(t-1), F(t)], kein Feuer bis H: auf [F(H), 1], wie
                         CiPOT in S auf [0, S(c)]), F'(tau) = G(F(tau)): eine globale Abbildung,
                         die die Kurven senkrecht umbeschriftet (CSD verschiebt waagerecht in der
                         Zeit). Hier stetig statt ueber R Gitterpunkte und bis zur Selbstkonsistenz
                         wiederholt (pit_fit); auf den Val-Jahren ist D-Cal danach flach.
                  ks     nach dem Prinzip von KSP (Park et al., NeurIPS 2025): F' = sigma(a logit F
                         + b), a und b minimieren die Kolmogorov-Smirnov-Abweichung der Val-PIT-
                         Verteilung von der Gleichverteilung (Zensierte wie bei pit).
  3  Konformal    Untere Schranke L(x) fuer die Zeit bis zum Feuer mit P(T >= L) >= 1 - alpha
                  (Candes, Lei, Ren 2023, Fall fester Zensur): Die Zensur bei H = 100 Tagen ist
                  fuer jeden Stichtag bekannt. Score s = F(t) bei Feuer an Tag t, +inf ohne
                  Feuer; s_hat = floor(alpha (n + 1))-kleinster Val-Score; L(x) = erster Tag
                  mit F(t) >= s_hat, sonst H + 1 ("nicht vor Tag 101"). Ohne Feuer ist T > H
                  >= L - 1, also immer abgedeckt. Garantie nur bei Austauschbarkeit von Val-
                  und Testjahren; deshalb Abdeckung je Gruppe. Zusaetzlich Mondrian je Monat
                  (eigenes s_hat je Monat) und die naive Schranke ohne Konformalisierung
                  (s_hat = alpha, das alpha-Quantil des Modells).

Die Hazards je Modell landen in derived_data/calib_cache/, damit spaetere Verfahren
(etwa weitere Kalibrierer) nicht neu rechnen muessen.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import time
from pathlib import Path

import numpy as np
import torch

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.eval.metrics import cif_from_hazard, d_calibration, evaluate
from wildfire.paper.reconstruct import cache_norm, to_device_once
from wildfire.paper.surv_eval import DAY0, H, calib_apply, calib_fit, compact, ensemble, hazards_pxst, hazards_sts

TARGETS = ("centre", "patch")
TAUS = (7, 30, 100)
MONTHS = (("apr_mai", 0, 152), ("jun", 152, 182), ("jul", 182, 213), ("aug", 213, 244), ("sep", 244, 367))


def groups(meta: dict, tg: str) -> dict[str, np.ndarray]:
    """Boolesche Masken je Gruppe; Laengsbaender aus den Spalten-Quartilen des jeweiligen Satzes."""
    year = np.array([(DAY0 + dt.timedelta(days=int(d))).year for d in meta["L_abs"]])
    doy, col = np.asarray(meta["doy"]), np.asarray(meta["col"])
    g = {"alle": np.ones(year.size, bool)}
    for y in np.unique(year):
        g[f"jahr_{y}"] = year == y
    for name, lo, hi in MONTHS:
        g[f"monat_{name}"] = (doy >= lo) & (doy < hi)
    fw = np.asarray(meta[f"fire_win_{tg}"]).astype(bool)
    g["feuer_im_fenster"], g["kein_feuer_im_fenster"] = fw, ~fw
    band = np.searchsorted(np.quantile(col, [0.25, 0.5, 0.75]), col, side="right")
    for b, name in enumerate(("west", "west_mitte", "ost_mitte", "ost")):
        g[f"band_{name}"] = band == b
    return g


def month_of(meta: dict) -> np.ndarray:
    doy = np.asarray(meta["doy"])
    m = np.zeros(doy.size, int)
    for k, (_, lo, hi) in enumerate(MONTHS):
        m[(doy >= lo) & (doy < hi)] = k
    return m


def hazard_from_F(F: np.ndarray) -> np.ndarray:
    S = 1.0 - np.clip(F, 0.0, 1.0)
    prev = np.concatenate([np.ones((S.shape[0], 1)), S[:, :-1]], 1)
    return np.clip(1.0 - S / np.maximum(prev, 1e-300), 1e-15, 1 - 1e-7)


def group_stats(F: np.ndarray, e: np.ndarray, t: np.ndarray, mask: np.ndarray) -> dict:
    """D-Cal und beobachtet / erwartet je Horizont in einer Gruppe; F = F(1..H)."""
    Fm, em, tm = F[mask], e[mask], t[mask]
    d = d_calibration(Fm, em, tm)
    out = {"n": int(mask.sum()), "n_events": int(em.sum()), "dcal_dev": d["d_cal_max_dev"],
           "dcal_p": d["d_cal_p"], "dcal_bins": d["d_cal_bins"]}
    for tau in TAUS:
        pred, obs = float(Fm[:, tau - 1].mean()), float((em & (tm <= tau)).mean())
        out.update({f"pred@{tau}": pred, f"obs@{tau}": obs, f"o_e@{tau}": obs / max(pred, 1e-12)})
    return out


def pit_map(F: np.ndarray, e: np.ndarray, t: np.ndarray, grid: int = 2001, chunk: int = 5000):
    """G(u) = Anteil der Val-PIT-Werte <= u (Intervall-PIT wie in d_calibration). -> (u, G)."""
    n = F.shape[0]
    idx = np.arange(n)
    Fp = np.concatenate([np.zeros((n, 1)), F], 1)                    # F(0..H)
    t = np.asarray(t, int)
    a = np.where(e, Fp[idx, np.maximum(t - 1, 0)], Fp[idx, -1])
    b = np.where(e, Fp[idx, t], 1.0)
    u = np.linspace(0.0, 1.0, grid)
    G = np.zeros(grid)
    for s in range(0, n, chunk):
        aa, ww = a[s:s + chunk, None], np.maximum(b[s:s + chunk] - a[s:s + chunk], 1e-12)[:, None]
        G += np.clip((u[None, :] - aa) / ww, 0.0, 1.0).sum(0)
    G /= n
    G[0], G[-1] = 0.0, 1.0
    return u, np.maximum.accumulate(G)


def pit_fit(F: np.ndarray, e: np.ndarray, t: np.ndarray, iters: int = 5):
    """PIT-Rekalibrierung bis zur Selbstkonsistenz: G einmal schaetzen, anwenden, auf den
    umgerechneten Kurven neu schaetzen. Noetig, weil die Ausschnitte ohne Feuer gleichverteilt
    auf [F(H), 1] angenommen werden und das nach einer nichtlinearen Umrechnung nicht mehr
    exakt gilt. -> (u, G) der zusammengesetzten Abbildung."""
    u, G = pit_map(F, e, t)
    Gc = G.copy()
    for _ in range(iters - 1):
        u, G = pit_map(np.interp(F, u, Gc), e, t)
        Gc = np.interp(Gc, u, G)
    return u, Gc


def pit_apply(F: np.ndarray, u: np.ndarray, G: np.ndarray) -> np.ndarray:
    return np.interp(F, u, G)


def ks_map(F: np.ndarray, a: float, b: float) -> np.ndarray:
    z = np.log(np.clip(F, 1e-9, 1 - 1e-9)) - np.log1p(-np.clip(F, 1e-9, 1 - 1e-9))
    return 1.0 / (1.0 + np.exp(-(a * z + b)))


def ks_fit(F: np.ndarray, e: np.ndarray, t: np.ndarray, n_max: int = 20000, seed: int = 0) -> tuple[float, float]:
    """KSP-Prinzip: a, b minimieren max |G(u) - u| der Val-PIT-Verteilung nach F' = sigma(a logit F + b)."""
    from scipy.optimize import minimize
    if F.shape[0] > n_max:                                          # Teilstichprobe: KS ist auf 20.000 schon scharf
        i = np.sort(np.random.default_rng(seed).choice(F.shape[0], n_max, replace=False))
        F, e, t = F[i], e[i], t[i]

    def ks(p):
        u, G = pit_map(ks_map(F, p[0], p[1]), e, t, grid=501)
        return float(np.abs(G - u).max())
    r = minimize(ks, x0=np.array([1.0, 0.0]), method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 300})
    return float(r.x[0]), float(r.x[1])


def conformal_threshold(F: np.ndarray, e: np.ndarray, t: np.ndarray, alpha: float) -> float:
    """s_hat = floor(alpha (n + 1))-kleinster Score; Score = F(t) bei Feuer an Tag t, sonst +inf."""
    s = np.where(e, F[np.arange(t.size), np.clip(t, 1, H) - 1], np.inf)
    k = int(np.floor(alpha * (s.size + 1)))
    return -np.inf if k < 1 else float(np.sort(s)[k - 1])


def lower_bound(F: np.ndarray, s_hat) -> np.ndarray:
    """L(x) = erster Tag t in 1..H mit F(t) >= s_hat, sonst H + 1; s_hat skalar oder je Zeile."""
    s_hat = np.broadcast_to(np.asarray(s_hat, float), (F.shape[0],))
    hit = F >= s_hat[:, None]
    L = np.where(hit.any(1), hit.argmax(1) + 1, H + 1)
    return np.where(np.isneginf(s_hat), 1, L)


def coverage(L: np.ndarray, e: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Abgedeckt, wenn T >= L: ohne Feuer bis H immer (T > H >= L - 1), sonst t >= L."""
    return ~e | (t >= L)


def conformal_report(Fv, ev, tv, Ft, et, tt, alpha, gmask: dict, m_val, m_test) -> dict:
    s_hat = conformal_threshold(Fv, ev, tv, alpha)
    L = lower_bound(Ft, s_hat)
    cov = coverage(L, et, tt)
    naive = coverage(lower_bound(Ft, alpha), et, tt)
    s_m = np.array([conformal_threshold(Fv[m_val == k], ev[m_val == k], tv[m_val == k], alpha)
                    for k in range(len(MONTHS))])
    L_m = lower_bound(Ft, s_m[m_test])
    cov_m = coverage(L_m, et, tt)
    info = L <= H
    return {
        "s_hat": s_hat, "s_hat_month": s_m.tolist(),
        "coverage": float(cov.mean()), "coverage_naive": float(naive.mean()), "coverage_mondrian": float(cov_m.mean()),
        "share_informative": float(info.mean()), "share_L_le_30": float((L <= 30).mean()),
        "median_L_informative": float(np.median(L[info])) if info.any() else None,
        "groups": {g: {"coverage": float(cov[m].mean()), "coverage_mondrian": float(cov_m[m].mean()),
                       "coverage_naive": float(naive[m].mean()), "share_informative": float(info[m].mean())}
                   for g, m in gmask.items()},
    }


def analyse(hv: np.ndarray, ht: np.ndarray, va: dict, te: dict, tg: str, alphas) -> dict:
    ev, tv = va[f"ev_{tg}"].astype(bool), va[f"t_{tg}"].astype(int)
    et, tt = te[f"ev_{tg}"].astype(bool), te[f"t_{tg}"].astype(int)
    Fv, Ft = cif_from_hazard(hv), cif_from_hazard(ht)
    gm = groups(te, tg)
    a, b = calib_fit(hv, ev, tv)
    u, G = pit_fit(Fv, ev, tv)
    ka, kb = ks_fit(Fv, ev, tv)
    variants = {"raw": Ft, "curve": cif_from_hazard(calib_apply(ht, a, b)), "pit": pit_apply(Ft, u, G), "ks": ks_map(Ft, ka, kb)}
    out = {"curve_ab": [a, b], "ks_ab": [ka, kb], "pit_G": {"u": u[::100].tolist(), "G": G[::100].tolist()}, "variants": {}}
    for name, F in variants.items():
        out["variants"][name] = {"overall": compact(evaluate(hazard_from_F(F), et, tt)),
                                 "groups": {g: group_stats(F, et, tt, m) for g, m in gm.items()}}
        o = out["variants"][name]["overall"]
        worst = max(out["variants"][name]["groups"].items(), key=lambda kv: kv[1]["dcal_dev"])
        print(f"[cal] {tg:6s} {name:5s} NLL {o['nll']:.4f}  D-Cal {o['dcal_dev']:.4f}  "
              f"schlechteste Gruppe {worst[0]} {worst[1]['dcal_dev']:.4f}", flush=True)
    m_val, m_test = month_of(va), month_of(te)
    out["conformal"] = {}
    for al in alphas:
        r = conformal_report(Fv, ev, tv, Ft, et, tt, al, gm, m_val, m_test)
        out["conformal"][str(al)] = r
        worst = min(r["groups"].items(), key=lambda kv: kv[1]["coverage"])
        print(f"[cal] {tg:6s} alpha {al:.2f}: Abdeckung {r['coverage']:.4f} (naiv {r['coverage_naive']:.4f}, "
              f"Mondrian {r['coverage_mondrian']:.4f}), informativ {r['share_informative']:.3f}, "
              f"schlechteste Gruppe {worst[0]} {worst[1]['coverage']:.4f}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="art:glob je Modell; art = pxst | sts...")
    ap.add_argument("--out-name", default="surv_eval_test")
    ap.add_argument("--cache-norm", default="cache_G")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.1])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = exp / "derived_data" / args.out_name
    te, va = dict(np.load(d / "eval_meta.npz")), dict(np.load(d / "val_meta.npz"))
    cache = exp / "derived_data" / "calib_cache"
    cache.mkdir(parents=True, exist_ok=True)
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / args.cache_norm / "norm_stats.npz")
    Xt = Xv = None
    res = {}
    for spec in args.models:
        kind, pattern = spec.split(":", 1)
        files = sorted(glob.glob(pattern))
        cp = cache / f"{kind}{args.tag}.npz"
        if cp.exists():
            z = np.load(cp)
            H_ = {(s, tg): z[f"{s}_{tg}"].astype(np.float64) for s in ("val", "test") for tg in TARGETS}
            print(f"[cal] {kind}: Hazards aus {cp}", flush=True)
        elif not files:
            print(f"[cal] {spec}: keine Checkpoints", flush=True)
            continue
        else:
            if Xt is None:
                Xt = to_device_once(np.load(d / "eval_X.npy"), device, "eval_X")
                Xv = to_device_once(np.load(d / "val_X.npy"), device, "val_X")
            fn = hazards_pxst if kind == "pxst" else hazards_sts
            acc = {(s, tg): [] for s in ("val", "test") for tg in TARGETS}
            for f in files:
                t0 = time.time()
                for s, X, meta in (("test", Xt, te), ("val", Xv, va)):
                    hc, hp = fn(f, X, meta["doy"], mean, std, device)
                    acc[(s, "centre")].append(hc); acc[(s, "patch")].append(hp)
                print(f"    {Path(f).name} in {time.time() - t0:.0f} s", flush=True)
            H_ = {k: ensemble(v) if len(v) > 1 else v[0] for k, v in acc.items()}
            np.savez(cp, **{f"{s}_{tg}": H_[(s, tg)].astype(np.float32) for s, tg in H_})
        res[kind] = {"files": [Path(f).name for f in files]}
        for tg in TARGETS:
            res[kind][tg] = analyse(H_[("val", tg)], H_[("test", tg)], va, te, tg, args.alphas)
    write_manifest(cfg, f"p22_calib_eval{args.tag}", args.config, {
        "models": args.models, "alphas": args.alphas, "results": res,
        "groups": "Testjahr, Monat des Stichtags, Feuer im 30-Tage-Fenster, Laengsbaender (Spalten-Quartile)",
        "variants": {"raw": "wie trainiert", "curve": "Lambda' = e^a Lambda^b, Survival-Likelihood auf Val",
                     "pit": "F' = G(F), G = Verteilung der Val-PIT-Werte (CiPOT-Prinzip, bis zur Selbstkonsistenz)",
                     "ks": "F' = sigma(a logit F + b), a und b minimieren die KS-Abweichung der Val-PIT-Werte (KSP-Prinzip)"},
        "conformal": "untere Schranke L mit P(T >= L) >= 1 - alpha, feste Zensur bei H (Candes, Lei, Ren 2023); "
                     "Kalibrierung auf den Val-Jahren, Mondrian je Monat, naiv = alpha-Quantil des Modells",
    })


if __name__ == "__main__":
    main()
