"""Extremes Feuerwetter und noch nie dagewesene Bedingungen: was taugt das Modell dort?

Anlass ist die Kritik aus der Brandschutz- und Evakuierungsforschung: Modelle, die aus vergangenen
Ereignissen lernen, erkennen beispiellose Lagen schlecht, und Extreme werden haeufiger. Getestet wird
auf denselben Stichtagen wie surv_eval / calib_eval (Val 2020-2022, Test 2023-2025, natuerliche
Haeufigkeit), aus den gespeicherten Hazards in derived_data/calib_cache/ (kein Modell wird neu
gerechnet; Kopf-Bias wie dort im Modus default, siehe load_sts).

  1  Eingaben   Feuerwetter der Trainingsjahre 2000-2019 an den Zellen der Stichtage (Mitte), aus dem
                Wuerfel: FFMC, DMC, DC, ISI, BUI, FWI, VPD, t2m_max. Je Jahr 2000-2025 Perzentile,
                Anteil bei 0 und 255 (uint8 ohne Skalierung, Kappung moeglich) und Anteil der Saison-
                tage ueber dem Trainingsmaximum derselben Zelle. Zeigt, ob 2023 in den Eingaben ueber-
                haupt extrem aussieht.
  2  Gruppen    Test-Stichtage nach dem Wert am Stichtag L in der Mitte: Perzentil-Baender der
                gepoolten Trainingsverteilung (< p50, p50-90, p90-95, p95-99, >= p99), lokal ueber p99
                bzw. ueber dem Maximum derselben Zelle in 2000-2019, und Testjahr x (>= p95). Je Gruppe
                beobachtet / erwartet, D-Cal, NLL, ROC-AUC / PR-AUC bei 7 und 30 Tagen.
  3  Saison     Nachkalibrierung waehrend der Saison auf der Val-Kalibrierung (curve):
                S'(tau) = S(tau)^c(L), c(L) = (O + k) / (E + k) aus den Stichtagen desselben Jahres mit
                L' in [L - W, L - lag - 1], zensiert bei L - lag - L' (bekannt sind die Feuer bis zum Tag
                L - lag); O = Feuer bis
                dahin, E = kumulierter Hazard bis Feuer bzw. Zensur. W und k werden auf den Val-Jahren
                gewaehlt. Orakel zum Vergleich: ein c je Testjahr aus dem ganzen Testjahr.

Die Zell-Klimatologie (Teil 1) landet in derived_data/extreme_eval/, damit weitere Laeufe den Wuerfel
nicht erneut lesen muessen.
"""

from __future__ import annotations

import argparse
import datetime as dt
import time
from pathlib import Path

import numpy as np

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.data.dataset import CubeReader
from wildfire.eval.metrics import auc_and_pr_at, cif_from_hazard, negative_log_likelihood
from wildfire.paper.calib_eval import TAUS, group_stats
from wildfire.paper.surv_eval import DAY0, H, _cum_hazard, calib_apply, calib_fit
from wildfire.paper.surv_prepare import HALF

TARGETS = ("centre", "patch")
# Wuerfel-Index = Cache-Index fuer die dynamischen Kanaele 0-25 (CACHE_FROM_CUBE)
WEATHER = {"FFMC": 19, "DMC": 20, "DC": 21, "ISI": 22, "BUI": 23, "FWI": 24, "VPD": 25, "t2m_max": 13}
TRAIN_YEARS = (2000, 2019)
BANDS = ((0.0, 0.5, "lt_p50"), (0.5, 0.9, "p50_90"), (0.9, 0.95, "p90_95"), (0.95, 0.99, "p95_99"),
         (0.99, 1.01, "ge_p99"))
GROUP_CH = ("FWI", "DC", "BUI", "ISI")


def year_of(L_abs: np.ndarray) -> np.ndarray:
    return np.array([(DAY0 + dt.timedelta(days=int(d))).year for d in L_abs])


def cell_climatology(cube: CubeReader, rr: np.ndarray, cc: np.ndarray, step: int) -> dict:
    """Ein Durchgang ueber den Wuerfel an den Zellen (rr, cc): Histogramme je Jahr und Kanal, je Zelle
    p99 und Maximum der Trainingsjahre, je Jahr Anteil der Zelltage ueber dem Zellmaximum."""
    chs = list(WEATHER.values())
    years = sorted(cube.per_year)
    hist = np.zeros((len(years), len(chs), 256), np.int64)
    train = [y for y in years if TRAIN_YEARS[0] <= int(y) <= TRAIN_YEARS[1]]
    n_tr = sum(len(range(0, int(cube.per_year[y]["T"]), step)) for y in train)
    assert train == years[:len(train)], "Trainingsjahre muessen vorn liegen"
    vals = np.empty((n_tr, rr.size, len(chs)), np.uint8)       # Zelltage der Trainingsjahre
    above = np.zeros((len(years), len(chs)), np.float64)       # Trainingsjahre: per Definition 0
    p99 = vmax = None
    k = 0
    for yi, y in enumerate(years):
        if y not in train and vmax is None:                   # Trainingsjahre fertig: Zellstatistik
            p99 = np.empty((rr.size, len(chs)), np.float32)
            for a in range(0, rr.size, 10000):                 # in Bloecken, sonst float64-Kopie zu gross
                p99[a:a + 10000] = np.percentile(vals[:, a:a + 10000], 99, axis=0)
            vmax = vals.max(0)
            del vals
        t0 = time.time()
        dyn, _ = cube._arrays(y)
        n = 0
        for d in range(0, dyn.shape[0], step):
            v = np.asarray(dyn[d, rr, cc][:, chs])              # (n_zellen, n_kanaele)
            for j in range(len(chs)):
                hist[yi, j] += np.bincount(v[:, j], minlength=256)
            if y in train:
                vals[k] = v
                k += 1
            else:
                above[yi] += (v > vmax).sum(0)
                n += rr.size
        above[yi] /= max(n, 1)
        print(f"[ext] Wuerfel {y}: {time.time() - t0:.0f} s", flush=True)
    return {"years": np.array([int(y) for y in years]), "hist": hist, "p99": p99, "vmax": vmax,
            "above_cell_max": above}


def hist_quantile(h: np.ndarray, q: float) -> int:
    c = np.cumsum(h)
    return int(np.searchsorted(c, q * c[-1]))


def year_table(cl: dict) -> dict:
    out = {}
    for yi, y in enumerate(cl["years"]):
        row = {}
        for j, name in enumerate(WEATHER):
            h = cl["hist"][yi, j]
            n = max(int(h.sum()), 1)
            row[name] = {"p50": hist_quantile(h, 0.5), "p95": hist_quantile(h, 0.95), "p99": hist_quantile(h, 0.99),
                         "max": int(np.nonzero(h)[0].max()) if h.any() else None,
                         "share_0": float(h[0] / n), "share_255": float(h[255] / n),
                         "share_above_cell_max": float(cl["above_cell_max"][yi, j])}
        out[str(int(y))] = row
    return out


def train_rank(cl: dict) -> list[np.ndarray]:
    """Je Kanal Perzentil-Rang eines uint8-Werts in der gepoolten Trainingsverteilung (Mittelrang bei Bindungen)."""
    tr = (cl["years"] >= TRAIN_YEARS[0]) & (cl["years"] <= TRAIN_YEARS[1])
    out = []
    for j in range(len(WEATHER)):
        h = cl["hist"][tr, j].sum(0).astype(np.float64)
        below = np.concatenate([[0.0], np.cumsum(h)[:-1]])
        out.append((below + 0.5 * h) / h.sum())
    return out


def landmark_values(cube: CubeReader, meta: dict) -> np.ndarray:
    """(n, n_kanaele) uint8 am Stichtag L in der Mitte, direkt aus dem Wuerfel (= letzter Tag des Fensters)."""
    chs = list(WEATHER.values())
    L, rr, cc = np.asarray(meta["L_cube"]), np.asarray(meta["row"]), np.asarray(meta["col"])
    out = np.zeros((L.size, len(chs)), np.uint8)
    for Lu in np.unique(L):
        m = L == Lu
        year, li = cube._pos[int(Lu)]
        dyn, _ = cube._arrays(year)
        out[m] = np.asarray(dyn[li, rr[m], cc[m]][:, chs])
    return out


def self_test(vals: np.ndarray, X_path: Path, n_chk: int = 500) -> float:
    """Wuerfelwerte am Stichtag gegen das gespeicherte Eingabefenster (letzter Tag, Mitte)."""
    X = np.load(X_path, mmap_mode="r")
    chk = np.linspace(0, X.shape[0] - 1, n_chk).astype(int)
    ref = np.stack([np.asarray(X[i, -1, list(WEATHER.values()), HALF, HALF]) for i in chk])
    return float((ref == vals[chk]).mean())


def groups(vals: np.ndarray, cell_idx: np.ndarray, cl: dict, ranks: list[np.ndarray], year: np.ndarray) -> dict:
    names = list(WEATHER)
    g = {"alle": np.ones(year.size, bool)}
    for ch in GROUP_CH:
        j = names.index(ch)
        r = ranks[j][vals[:, j]]
        for lo, hi, b in BANDS:
            g[f"{ch}_{b}"] = (r >= lo) & (r < hi)
        g[f"{ch}_ueber_zell_p99"] = vals[:, j] > cl["p99"][cell_idx, j]
        g[f"{ch}_ueber_zell_max"] = vals[:, j] > cl["vmax"][cell_idx, j]
        for y in np.unique(year):
            g[f"{ch}_ge_p95_jahr_{y}"] = (r >= 0.95) & (year == y)
    jf, jd = names.index("FWI"), names.index("DC")
    g["FWI_und_DC_ge_p95"] = (ranks[jf][vals[:, jf]] >= 0.95) & (ranks[jd][vals[:, jd]] >= 0.95)
    for y in np.unique(year):
        g[f"jahr_{y}"] = year == y
    return g


def stats(h: np.ndarray, e: np.ndarray, t: np.ndarray, m: np.ndarray) -> dict:
    F = cif_from_hazard(h[m])
    out = group_stats(F, e[m], t[m], np.ones(int(m.sum()), bool))
    out["nll"] = float(negative_log_likelihood(h[m], e[m], t[m])) if m.any() else None
    for tau in (7, 30):
        ok = m.sum() > 0 and 0 < (e[m] & (t[m] <= tau)).sum() < m.sum()
        auc, ap = auc_and_pr_at(h[m], e[m], t[m], tau, None, F) if ok else (None, None)
        out.update({f"auc@{tau}": auc, f"pr@{tau}": ap})
    out.pop("dcal_bins", None)
    return out


def season_scale(h: np.ndarray, e: np.ndarray, t: np.ndarray, L: np.ndarray, year: np.ndarray,
                 W: int, k: float, lag: int) -> np.ndarray:
    """c(L) je Stichtag aus den Stichtagen desselben Jahres, deren Ausgang bis L - lag teilweise bekannt ist."""
    Lam = _cum_hazard(h)                                       # Lambda(1..H)
    c = np.ones(L.size)
    for y in np.unique(year):
        iy = np.nonzero(year == y)[0]
        Ly = L[iy]
        for Lu in np.unique(Ly):
            src = iy[(Ly >= Lu - W) & (Ly <= Lu - lag - 1)]
            if src.size == 0:
                continue
            cens = np.minimum(Lu - lag - L[src], H)            # bekannte Tage 1..cens, cens >= 1
            hit = e[src] & (t[src] <= cens)
            stop = np.where(hit, t[src], cens).astype(int)
            O = float(hit.sum())
            E = float(Lam[src, stop - 1].sum())
            c[iy[Ly == Lu]] = (O + k) / (E + k)
    return c


def apply_scale(h: np.ndarray, c: np.ndarray) -> np.ndarray:
    """S' = S^c: der Hazard jedes Tages wird 1 - (1 - h)^c."""
    return np.clip(-np.expm1(c[:, None] * np.log1p(-np.clip(h, 1e-12, 1 - 1e-7))), 1e-15, 1 - 1e-7)


def year_oracle(h: np.ndarray, e: np.ndarray, t: np.ndarray, year: np.ndarray) -> np.ndarray:
    Lam = _cum_hazard(h)
    stop = np.clip(t.astype(int), 1, H)
    c = np.ones(year.size)
    for y in np.unique(year):
        m = year == y
        c[m] = (e[m] & (t[m] <= H)).sum() / max(Lam[m, stop[m] - 1].sum(), 1e-12)
    return c


def analyse(hv, ht, va, te, tg, g, yv, yt, grid_W, grid_k, lag) -> dict:
    ev, tv = va[f"ev_{tg}"].astype(bool), va[f"t_{tg}"].astype(int)
    et, tt = te[f"ev_{tg}"].astype(bool), te[f"t_{tg}"].astype(int)
    a, b = calib_fit(hv, ev, tv)
    hv_c, ht_c = calib_apply(hv, a, b), calib_apply(ht, a, b)
    # Saison-Nachkalibrierung: W, k nach Val-NLL
    sel = {}
    for W in grid_W:
        for k in grid_k:
            cv = season_scale(hv_c, ev, tv, np.asarray(va["L_abs"]), yv, W, k, lag)
            ct = season_scale(ht_c, et, tt, np.asarray(te["L_abs"]), yt, W, k, lag)
            hs = apply_scale(ht_c, ct)
            sel[f"W{W}_k{k:g}"] = {
                "val_nll": float(negative_log_likelihood(apply_scale(hv_c, cv), ev, tv)),
                "test_nll": float(negative_log_likelihood(hs, et, tt)),
                "test_c_q": np.quantile(ct, [0.05, 0.5, 0.95]).tolist(),
                "test_by_year": {str(y): {f"o_e@{tau}": stats(hs, et, tt, yt == y)[f"o_e@{tau}"] for tau in TAUS}
                                 for y in np.unique(yt)},
            }
    best = min(sel, key=lambda s: sel[s]["val_nll"])
    W_b, k_b = int(best.split("_")[0][1:]), float(best.split("_k")[1])
    c_season = season_scale(ht_c, et, tt, np.asarray(te["L_abs"]), yt, W_b, k_b, lag)
    c_oracle = year_oracle(ht_c, et, tt, yt)
    variants = {"raw": ht, "curve": ht_c, "season": apply_scale(ht_c, c_season), "oracle_year": apply_scale(ht_c, c_oracle)}
    out = {"curve_ab": [a, b], "season_grid": sel, "season_best": best,
           "oracle_c": {str(y): float(c_oracle[yt == y][0]) for y in np.unique(yt)}, "variants": {}}
    for name, h in variants.items():
        out["variants"][name] = {gn: stats(h, et, tt, m) for gn, m in g.items() if m.sum() >= 50}
        al = out["variants"][name]["alle"]
        print(f"[ext] {tg:6s} {name:11s} NLL {al['nll']:.4f}  O/E@30 {al['o_e@30']:.3f}  "
              + "  ".join(f"{gn} O/E@30 {out['variants'][name][gn]['o_e@30']:.3f}"
                          for gn in ("FWI_ge_p99", "DC_ueber_zell_max", "jahr_2023") if gn in out["variants"][name]),
              flush=True)
    out["group_sizes"] = {gn: {"n": int(m.sum()), "events": int(et[m].sum())} for gn, m in g.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--caches", nargs="+", required=True, help="name:pfad je Modell (npz aus calib_cache)")
    ap.add_argument("--out-name", default="surv_eval_test")
    ap.add_argument("--day-step", type=int, default=1, help="jeder n-te Saisontag fuer die Klimatologie")
    ap.add_argument("--season-W", type=int, nargs="+", default=[14, 30, 60])
    ap.add_argument("--season-k", type=float, nargs="+", default=[10, 50, 200, 1000])
    ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    d = exp / "derived_data" / args.out_name
    te, va = dict(np.load(d / "eval_meta.npz")), dict(np.load(d / "val_meta.npz"))
    cube = CubeReader(exp)

    # Teil 1: Klimatologie an allen Zellen der Val- und Test-Stichtage
    cells = np.unique(np.concatenate([te["row"] * cube.W + te["col"], va["row"] * cube.W + va["col"]]))
    rr, cc = cells // cube.W, cells % cube.W
    cdir = exp / "derived_data" / "extreme_eval"
    cdir.mkdir(parents=True, exist_ok=True)
    cpath = cdir / f"cell_clim_{args.out_name}_step{args.day_step}.npz"
    if cpath.exists():
        cl = dict(np.load(cpath))
        print(f"[ext] Klimatologie aus {cpath}", flush=True)
    else:
        print(f"[ext] {cells.size:,} Zellen, Wuerfel jeder {args.day_step}. Tag", flush=True)
        cl = cell_climatology(cube, rr, cc, args.day_step)
        np.savez(cpath, cells=cells, **cl)
    ranks = train_rank(cl)
    vt, vv = landmark_values(cube, te), landmark_values(cube, va)
    agree = self_test(vt, d / "eval_X.npy")
    print(f"[ext] Selbsttest Wuerfel gegen eval_X (letzter Tag, Mitte): {agree:.4f}", flush=True)
    if agree < 0.999:
        raise SystemExit("ABBRUCH: Stichtagswerte passen nicht zu eval_X")
    ci_t = np.searchsorted(cells, te["row"] * cube.W + te["col"])
    yt, yv = year_of(te["L_abs"]), year_of(va["L_abs"])
    g = groups(vt, ci_t, cl, ranks, yt)
    # Anteil der Stichtage in den Extremgruppen je Jahr, auch Val
    ci_v = np.searchsorted(cells, va["row"] * cube.W + va["col"])
    gv = groups(vv, ci_v, cl, ranks, yv)
    share = {"test": {gn: {str(y): float(m[yt == y].mean()) for y in np.unique(yt)} for gn, m in g.items()},
             "val": {gn: {str(y): float(m[yv == y].mean()) for y in np.unique(yv)} for gn, m in gv.items()
                     if not gn.startswith("jahr_") and "_jahr_" not in gn}}

    res = {}
    for spec in args.caches:
        name, path = spec.split(":", 1)
        if not Path(path).exists():
            print(f"[ext] {name}: {path} fehlt", flush=True)
            continue
        z = np.load(path)
        res[name] = {"cache": path}
        for tg in TARGETS:
            t0 = time.time()
            res[name][tg] = analyse(z[f"val_{tg}"].astype(np.float64), z[f"test_{tg}"].astype(np.float64),
                                    va, te, tg, g, yv, yt, args.season_W, args.season_k, args.lag)
            print(f"[ext] {name} {tg}: {time.time() - t0:.0f} s", flush=True)

    write_manifest(cfg, f"p31_extreme_eval{args.tag}", args.config, {
        "caches": args.caches, "day_step": args.day_step, "lag": args.lag,
        "season_grid": {"W": args.season_W, "k": args.season_k}, "self_test_agree": agree,
        "n_cells": int(cells.size), "climatology_file": str(cpath),
        "train_years": list(TRAIN_YEARS), "channels": WEATHER,
        "inputs_by_year": year_table(cl),
        "train_rank_thresholds": {n: {f"p{int(q * 100)}": int(np.searchsorted(ranks[j], q))
                                      for q in (0.5, 0.9, 0.95, 0.99)} for j, n in enumerate(WEATHER)},
        "group_share_by_year": share,
        "results": res,
        "notes": {
            "werte": "uint8 der Rohdaten ohne Skalierung; Raenge gegen die gepoolte Verteilung 2000-2019 an denselben Zellen",
            "bias": "Hazards aus calib_cache (load_sts, Kopf-Bias default); curve = Val-Kalibrierung Lambda' = e^a Lambda^b",
            "season": "S' = S^c(L), c = (O + k) / (E + k) aus Stichtagen desselben Jahres in [L - W, L - lag - 1], "
                      "zensiert bei L - lag - L'; W, k nach Val-NLL",
            "oracle_year": "c je Testjahr = O / E ueber das ganze Testjahr (nicht operativ, obere Schranke)",
        },
    })


if __name__ == "__main__":
    main()
