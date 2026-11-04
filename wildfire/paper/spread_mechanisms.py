"""Gibt es kooperative Zuendung und Feuer-Verstaerkung? Diagnose fuer CA- und Reaktions-Diffusions-Terme im Contagion.

Der Contagion-Kern nimmt an, dass jeder brennende Nachbar unabhaengig ansteckt (1 - h = prod (1 - k b)) und ein
Feuer nur abklingt (D(0) = 1, monoton). Hier direkt aus den MODIS-Feuerpixeln des Wuerfels (Kanal 0,
fire_index.npz), nur Trainingsjahre:

Kandidat am Tag t   Pixel ohne Feuer an den Tagen t-1 .. t-Q (--quiet, Standard 30 wie die Auswertemaske im
                    firemap_bench: unverbrannter Brennstoff) mit mindestens einem Feuerpixel am Vortag in der
                    5 x 5-Nachbarschaft; Vortag kalendarisch direkt vor t
Ergebnis y          Feuer am Tag t
n8 / n24            Zahl brennender Pixel am Vortag in der Moore-Nachbarschaft (3 x 3) bzw. 5 x 5 ohne Mitte
Groesse             Feuerpixel am Vortag im Umkreis von 10 px (Groesse der lokalen Front), Klassen
Alter               nur fuer n8 = 1: Tage seit Beginn der Episode des einzigen Nachbarn (Luecke <= 2 Tage)
FWI                 Wuerfelwert (uint8) am Kandidaten am Vortag, Klassen nach Terzilen aller Kandidaten

Tests (Ausgabe metrics/spread_mechanisms.json + .md, Manifest p70_spread_mechanisms):
  1. kooperativ    P(y | n8) gegen unabhaengig 1 - (1 - p)^n (ML-p) und Potenz 1 - exp(-lam n^gamma) (ML);
                   gamma > 1: Schwellen-/bistabile CA-Regel, gamma < 1: Saettigung. Je Groessen- und FWI-Klasse.
  2. Verstaerkung  P(y | n8 = 1, Alter des Nachbarn): steigt sie erst an, baut sich ein Feuer auf (Intensitaet).
  3. Groesse       P(y | n8 = 1, Groesse): grosse Fronten zuenden staerker (Rueckkopplung ueber die Front).
Nur Lesen.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial import cKDTree

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.data.dataset import CubeReader
from wildfire.paper.firemap_bench import split_years

FWI_DYN = 24                                   # Wuerfel-Index des FWI (Cache-Kanal 24)
SIZE_EDGES = (1, 3, 10, 30, 100, 10 ** 9)      # Feuerpixel am Vortag im Umkreis 10 px
AGE_MAX = 15


def offsets(r: int) -> np.ndarray:
    a = np.arange(-r, r + 1)
    dy, dx = np.meshgrid(a, a, indexing="ij")
    keep = (dy != 0) | (dx != 0)
    return np.stack([dy[keep], dx[keep]], 1)


def neighbour_counts(src: np.ndarray, H: int, W: int, off: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fuer alle Pixel mit >= 1 Quelle in der Nachbarschaft: (Pixel, Zahl der Quellen)."""
    r, c = src // W, src % W
    rr, cc = r[:, None] + off[None, :, 0], c[:, None] + off[None, :, 1]
    ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    return np.unique((rr * W + cc)[ok], return_counts=True)


def fit_curves(n: np.ndarray, k: np.ndarray, y: np.ndarray) -> dict:
    """n Nachbarzahl, k Kandidaten, y Zuendungen je n (n >= 1). ML fuer unabhaengig und Potenz."""
    n, k, y = n[k > 0].astype(float), k[k > 0].astype(float), y[k > 0].astype(float)
    if y.sum() < 20:
        return {}

    def nll(P):
        P = np.clip(P, 1e-12, 1 - 1e-12)
        return -float((y * np.log(P) + (k - y) * np.log1p(-P)).sum())

    ind = minimize_scalar(lambda q: nll(1 - (1 - q) ** n), bounds=(1e-6, 1 - 1e-6), method="bounded")
    pw = minimize(lambda th: nll(-np.expm1(-np.exp(th[0]) * n ** np.exp(th[1]))), x0=[np.log(ind.x), 0.0],
                  method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000})
    return {"p_indep": float(ind.x), "nll_indep": float(ind.fun), "lam": float(np.exp(pw.x[0])),
            "gamma": float(np.exp(pw.x[1])), "nll_power": float(pw.fun), "lr_stat": float(2 * (ind.fun - pw.fun)),
            "n_cand": int(k.sum()), "n_ign": int(y.sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--quiet", type=int, default=30)
    ap.add_argument("--gap", type=int, default=2, help="Episode: Luecke <= gap Tage")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    out = exp / "metrics" / f"spread_mechanisms_{args.split}.json"
    if out.exists():
        raise SystemExit(f"ABBRUCH: {out} existiert schon")
    cube = CubeReader(exp)
    H, W = cube.H, cube.W
    years = set(split_years(exp)[args.split])
    z = np.load(exp / "derived_data" / "events" / "fire_index.npz")
    pix = np.repeat(z["fire_pixels"], np.diff(z["fire_offsets"])).astype(np.int64)
    days = z["fire_days"].astype(np.int64)
    order = np.argsort(days, kind="stable")
    pix, days = pix[order], days[order]
    nd = len(cube.dates)
    bounds = np.searchsorted(days, np.arange(nd + 1))
    per_day = [pix[bounds[t]:bounds[t + 1]] for t in range(nd)]
    ordn = np.array([dt.date.fromisoformat(s).toordinal() for s in cube.dates], np.int64)
    print(f"[sm] {pix.size:,} Feuerpixel-Tage, {nd} Tage, Gitter {H} x {W}, Jahre {sorted(years)}", flush=True)

    off8, off24 = offsets(1), offsets(2)
    last = np.full(H * W, -10 ** 7, np.int64)              # letzter Feuertag (Ordinal)
    start = np.full(H * W, -10 ** 7, np.int64)             # Beginn der Episode
    NB, NS = 25, len(SIZE_EDGES) - 1
    cnt8 = np.zeros((9, NS, 256), np.int64)                 # n8, Groesse, FWI-Rohwert
    ign8 = np.zeros_like(cnt8)
    cnt24 = np.zeros((NB, NS), np.int64)
    ign24 = np.zeros_like(cnt24)
    cnt_age = np.zeros((AGE_MAX + 1, NS), np.int64)
    ign_age = np.zeros_like(cnt_age)
    for t in range(nd):
        cur = per_day[t]
        prev_ok = t > 0 and ordn[t] - ordn[t - 1] == 1
        y = int(cube.dates[t][:4])
        if prev_ok and y in years and per_day[t - 1].size and t in cube._pos:
            src = per_day[t - 1]
            cand, n24 = neighbour_counts(src, H, W, off24)
            fresh = last[cand] < ordn[t] - args.quiet       # keine Feuer an t-1 .. t-Q
            cand, n24 = cand[fresh], n24[fresh]
            if cand.size:
                c8, n8c = neighbour_counts(src, H, W, off8)
                n8 = np.zeros(cand.size, np.int64)
                pos = np.searchsorted(c8, cand)
                hit = (pos < c8.size) & (c8[np.minimum(pos, c8.size - 1)] == cand)
                n8[hit] = n8c[pos[hit]]
                yy = np.isin(cand, cur)
                tree = cKDTree(np.stack([src // W, src % W], 1))
                q = np.stack([cand // W, cand % W], 1)
                size = np.array([len(v) for v in tree.query_ball_point(q, 10.0)])
                sb = np.clip(np.searchsorted(SIZE_EDGES, size, side="right") - 1, 0, NS - 1)
                yr, li = cube._pos[t - 1] if (t - 1) in cube._pos else cube._pos[t]
                dyn, _ = cube._arrays(yr)
                srt = np.argsort(cand)
                fwi = np.empty(cand.size, np.int64)
                fwi[srt] = np.asarray(dyn[li][cand[srt] // W, cand[srt] % W, FWI_DYN])
                np.add.at(cnt8, (n8, sb, fwi), 1)
                np.add.at(ign8, (n8, sb, fwi), yy)
                np.add.at(cnt24, (np.minimum(n24, NB - 1), sb), 1)
                np.add.at(ign24, (np.minimum(n24, NB - 1), sb), yy)
                one = n8 == 1
                if one.any():
                    # der einzige Moore-Nachbar: Alter seiner Episode am Vortag
                    srcset = np.full(int(one.sum()), -1, np.int64)
                    for o in off8:
                        rr, cc = cand[one] // W + o[0], cand[one] % W + o[1]
                        ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
                        nb = np.where(ok, rr * W + cc, 0)
                        srcset = np.where(ok & np.isin(nb, src), nb, srcset)
                    age = np.clip(ordn[t - 1] - start[srcset], 0, AGE_MAX)
                    np.add.at(cnt_age, (age, sb[one]), 1)
                    np.add.at(ign_age, (age, sb[one]), yy[one])
        if cur.size:                                        # Episoden nach dem Tag t fortschreiben
            new_ep = last[cur] < ordn[t] - args.gap - 1
            start[cur[new_ep]] = ordn[t]
            last[cur] = ordn[t]
        if t % 500 == 0:
            print(f"    Tag {t}/{nd} {cube.dates[t]}: Kandidaten bisher {int(cnt8.sum()):,}", flush=True)

    # ---------------------------------------------------------------- Auswertung
    tot = cnt8.sum((1, 2))
    fwi_cdf = np.cumsum(cnt8[1:].sum((0, 1))) / max(cnt8[1:].sum(), 1)
    fwi_cut = [int(np.searchsorted(fwi_cdf, q)) for q in (1 / 3, 2 / 3)]
    fwi_bins = {"low": slice(0, fwi_cut[0] + 1), "mid": slice(fwi_cut[0] + 1, fwi_cut[1] + 1), "high": slice(fwi_cut[1] + 1, 256)}
    nn = np.arange(9)
    res = {"args": vars(args), "fwi_uint8_tercile_cuts": fwi_cut, "size_edges": list(SIZE_EDGES[:-1]),
           "n8": {"n": nn.tolist(), "cand": tot.tolist(), "ign": ign8.sum((1, 2)).tolist(),
                  "p": (ign8.sum((1, 2)) / np.maximum(tot, 1)).tolist()}}
    res["fit_all"] = fit_curves(nn[1:], tot[1:], ign8.sum((1, 2))[1:])
    res["fit_by_size"] = {f">={SIZE_EDGES[s]}": fit_curves(nn[1:], cnt8[1:, s].sum(-1), ign8[1:, s].sum(-1)) for s in range(NS)}
    res["fit_by_fwi"] = {k: fit_curves(nn[1:], cnt8[1:, :, sl].sum((1, 2)), ign8[1:, :, sl].sum((1, 2))) for k, sl in fwi_bins.items()}
    res["fit_by_size_fwi"] = {f">={SIZE_EDGES[s]}_{k}": fit_curves(nn[1:], cnt8[1:, s, sl].sum(-1), ign8[1:, s, sl].sum(-1))
                              for s in range(NS) for k, sl in fwi_bins.items()}
    n24 = np.arange(NB)
    res["n24"] = {"n": n24.tolist(), "cand": cnt24.sum(1).tolist(), "p": (ign24.sum(1) / np.maximum(cnt24.sum(1), 1)).tolist()}
    res["fit_n24"] = fit_curves(n24[1:NB - 1], cnt24.sum(1)[1:NB - 1], ign24.sum(1)[1:NB - 1])
    res["age_n8eq1"] = {"age": list(range(AGE_MAX + 1)), "cand": cnt_age.sum(1).tolist(),
                        "p": (ign_age.sum(1) / np.maximum(cnt_age.sum(1), 1)).tolist(),
                        "p_by_size": {f">={SIZE_EDGES[s]}": (ign_age[:, s] / np.maximum(cnt_age[:, s], 1)).tolist() for s in range(NS)},
                        "cand_by_size": {f">={SIZE_EDGES[s]}": cnt_age[:, s].tolist() for s in range(NS)}}
    c1, i1 = cnt8[1].sum(-1), ign8[1].sum(-1)
    res["size_n8eq1"] = {"size_from": list(SIZE_EDGES[:-1]), "cand": c1.tolist(), "p": (i1 / np.maximum(c1, 1)).tolist()}

    f = lambda x: f"{x:.4f}"                                                            # noqa: E731
    md = [f"# Zuendmechanismen aus MODIS ({args.split}, Jahre {min(years)}-{max(years)})", "",
          f"Kandidat: kein Feuer an den {args.quiet} Vortagen, Feuer am Vortag in 5 x 5; Ergebnis Feuer am Tag t.", "",
          "## 1. Kooperative Zuendung: P(Zuendung | n brennende Moore-Nachbarn)", "",
          "| n | Kandidaten | P beobachtet | P unabhaengig 1-(1-p)^n | P Potenz 1-exp(-lam n^g) |", "|---|---|---|---|---|"]
    fa = res["fit_all"]
    for n in range(1, 9):
        md.append(f"| {n} | {tot[n]:,} | {f(res['n8']['p'][n])} | {f(1 - (1 - fa['p_indep']) ** n)} | "
                  f"{f(-np.expm1(-fa['lam'] * n ** fa['gamma']))} |")
    md += ["", f"gamma = {fa['gamma']:.3f} (1 = unabhaengig), LR-Statistik {fa['lr_stat']:.1f} (chi2 1 FG)", "",
           "| Schicht | Kandidaten | Zuendungen | p unabh. | gamma | LR |", "|---|---|---|---|---|---|"]
    for grp in ("fit_by_size", "fit_by_fwi", "fit_by_size_fwi"):
        for k, v in res[grp].items():
            if v:
                md.append(f"| {grp[7:]} {k} | {v['n_cand']:,} | {v['n_ign']:,} | {f(v['p_indep'])} | {v['gamma']:.3f} | {v['lr_stat']:.1f} |")
    v = res["fit_n24"]
    md += ["", f"5 x 5 (n24): gamma = {v.get('gamma', float('nan')):.3f}, LR {v.get('lr_stat', float('nan')):.1f}", "",
           "## 2. Verstaerkung: P(Zuendung | genau ein Moore-Nachbar, Alter seiner Episode)", "",
           "| Alter (Tage) | Kandidaten | P | " + " | ".join(f"P Groesse >={s}" for s in SIZE_EDGES[:-1]) + " |",
           "|---" * (3 + NS) + "|"]
    for a in range(AGE_MAX + 1):
        md.append(f"| {a}{'+' if a == AGE_MAX else ''} | {res['age_n8eq1']['cand'][a]:,} | {f(res['age_n8eq1']['p'][a])} | "
                  + " | ".join(f(res["age_n8eq1"]["p_by_size"][f">={s}"][a]) for s in SIZE_EDGES[:-1]) + " |")
    md += ["", "## 3. Frontgroesse: P(Zuendung | genau ein Moore-Nachbar, Feuerpixel im Umkreis 10 px am Vortag)", "",
           "| Groesse ab | Kandidaten | P |", "|---|---|---|"]
    md += [f"| {s} | {c1[i]:,} | {f(res['size_n8eq1']['p'][i])} |" for i, s in enumerate(SIZE_EDGES[:-1])]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)
    write_manifest(cfg, f"p70_spread_mechanisms_{args.split}", args.config, {"out": str(out), "args": vars(args)})


if __name__ == "__main__":
    main()
