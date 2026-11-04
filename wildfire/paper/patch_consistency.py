"""Passen Pixel- und Ausschnittskopf zusammen? Und was bringt eine Clayton-Aggregation der Pixel?

Der Ausschnitt brennt, sobald irgendeines seiner Pixel brennt. Fuer jede
gemeinsame Verteilung gilt deshalb (Frechet-Schranken)

    1 - sum_i F_i(tau)  <=  S_patch(tau)  <=  min_i S_i(tau)

Der STS-DCTM hat getrennte Koepfe fuer Pixel und Ausschnitt; nichts erzwingt
diese Schranken. Gemessen wird, wie oft und wie stark sie verletzt sind, auf
den zufaelligen Stichtagen der Testjahre (surv_eval, natuerliche Haeufigkeit).

Zweitens, ohne neues Training: Gamma-Frailty mit Varianz theta ist dasselbe
wie die Clayton-Copula der Pixel-Survivals,

    S_R(tau) = (1 + sum_{i in R} (S_i(tau)^-theta - 1))^(-1/theta)

theta -> 0 ist Unabhaengigkeit (Produkt der Pixel), theta -> inf vollstaendige
Abhaengigkeit (Minimum). theta wird auf den Val-Stichtagen (2020-2022) per
Survival-NLL des Ausschnitts gewaehlt und auf den Test-Stichtagen mit dem
eigenen Ausschnittskopf verglichen. Liegt die Aggregation gleichauf, kann ein
Frailty-Kopf den Ausschnittskopf ersetzen und liefert jede Region (3x3, 7x7,
Polygone) konsistent aus einer Pixelkarte.
"""

from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import torch

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.eval.metrics import evaluate
from wildfire.paper.reconstruct import cache_norm, to_device_once
from wildfire.paper.sts_dctm import load_sts

H = 100
P = 13
THETAS = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, np.inf)
TAUS = (1, 7, 30, 100)


def clayton_log_s(ls_px: torch.Tensor, theta: float) -> torch.Tensor:
    """ls_px (B, n_px, T) log S_i -> (B, T) log S_R, float64.

    log S_R = -(1/theta) log(1 + sum_i expm1(-theta log S_i)); expm1/log1p halten die
    Differenzen bei S_i nahe 1 genau."""
    if theta == 0.0:
        return ls_px.sum(1)
    if np.isinf(theta):
        return ls_px.min(1).values
    return -torch.log1p(torch.expm1((-theta * ls_px).clamp_max(700.0)).sum(1)) / theta


def to_hazard(ls: np.ndarray) -> np.ndarray:
    """(n, H+1) log S(0..H) -> (n, H) diskrete Hazards."""
    return np.clip(-np.expm1(np.diff(ls, axis=1)), 1e-15, 1 - 1e-7)


@torch.no_grad()
def curves(net, X, doy, m_t, s_t, device, bs=64):
    """Ausschnittskopf und Clayton-Aggregate (n, H+1) je theta, dazu Verletzungen der Frechet-Schranken."""
    tau = torch.arange(0, H + 1, device=device, dtype=torch.float64)
    head, agg = [], {th: [] for th in THETAS}
    up, lo, up_mag = [], [], []
    for a in range(0, X.shape[0], bs):
        x = (X[a:a + bs].float() - m_t) / s_t
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            raw_px, raw_pa = net(x, torch.from_numpy(doy[a:a + bs]).to(device))
        b = raw_px.shape[0]
        ls_px = net.px_head.log_survival(raw_px.reshape(b * P * P, -1), tau).view(b, P * P, H + 1)
        ls_pa = net.pa_head.log_survival(raw_pa, tau)
        head.append(ls_pa.cpu().numpy())
        for th in THETAS:
            agg[th].append(clayton_log_s(ls_px, th).cpu().numpy())
        S_pa, S_min = ls_pa.exp(), ls_px.min(1).values.exp()
        F_sum = (-torch.expm1(ls_px)).sum(1)
        k = list(TAUS)
        up.append((S_pa[:, k] > S_min[:, k] + 1e-9).cpu().numpy())
        up_mag.append((S_pa[:, k] - S_min[:, k]).clamp_min(0).cpu().numpy())
        lo.append((S_pa[:, k] < 1 - F_sum[:, k] - 1e-9).cpu().numpy())
    cat = np.concatenate
    return cat(head), {th: cat(v) for th, v in agg.items()}, cat(up), cat(up_mag), cat(lo)


def compact(m: dict) -> dict:
    out = {"nll": m["nll"], "ibs": m["integrated_brier"], "c_td": m["c_index_td"], "dcal_dev": m["d_cal_max_dev"]}
    for k in TAUS:
        out.update({f"pr@{k}": m["horizons"][str(k)]["pr_auc"], f"ece@{k}": m["horizons"][str(k)]["ece"]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--surv-dir", default="surv_eval_test")
    ap.add_argument("--cache-norm", default="cache_G")
    C = "/data/Wildfire/checkpoints/paper/sts_cache_G_next_cube_d128"
    ap.add_argument("--models", nargs="+", default=[f"sts:{C}_s*_s*.pt", f"sts_lh:{C}_lh_s*_s*.pt",
                                                     f"sts_dctm_cloglog:{C}_dctm_cloglog_s*_s*.pt"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = exp / "derived_data" / args.surv_dir
    te, va = dict(np.load(sd / "eval_meta.npz")), dict(np.load(sd / "val_meta.npz"))
    y_te = (te["ev_patch"].astype(bool), te["t_patch"].astype(int))
    y_va = (va["ev_patch"].astype(bool), va["t_patch"].astype(int))
    Xt, Xv = to_device_once(np.load(sd / "eval_X.npy"), device, "eval_X"), to_device_once(np.load(sd / "val_X.npy"), device, "val_X")
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / args.cache_norm / "norm_stats.npz")
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]

    res = {}
    for spec in args.models:
        kind, pattern = spec.split(":", 1)
        for f in sorted(glob.glob(pattern)):
            t0 = time.time()
            net = load_sts(f, device)
            head_t, agg_t, up, up_mag, lo = curves(net, Xt, te["doy"], m_t, s_t, device)
            _, agg_v, *_ = curves(net, Xv, va["doy"], m_t, s_t, device)
            nll_v = {th: compact(evaluate(to_hazard(agg_v[th]), *y_va))["nll"] for th in THETAS}
            th_best = min(nll_v, key=nll_v.get)
            r = {"violations": {str(k): {"upper_share": float(up[:, j].mean()), "upper_mean_excess": float(up_mag[:, j].mean()),
                                         "upper_max_excess": float(up_mag[:, j].max()), "lower_share": float(lo[:, j].mean())}
                                for j, k in enumerate(TAUS)},
                 "any_upper_violation": float(up.any(1).mean()),
                 "val_nll_by_theta": {str(th): v for th, v in nll_v.items()}, "theta_val": float(th_best),
                 "test": {"patch_head": compact(evaluate(to_hazard(head_t), *y_te))}}
            for th in THETAS:
                r["test"][f"clayton_{th}"] = compact(evaluate(to_hazard(agg_t[th]), *y_te))
            res.setdefault(kind, {})[Path(f).stem] = r
            ph, cb = r["test"]["patch_head"], r["test"][f"clayton_{th_best}"]
            ind, com = r["test"]["clayton_0.0"], r["test"]["clayton_inf"]
            print(f"[pc] {Path(f).name} ({time.time() - t0:.0f} s): Schranke S_patch <= min S_i verletzt bei "
                  + ", ".join(f"tau {k}: {r['violations'][str(k)]['upper_share']:.1%}" for k in TAUS)
                  + f"; untere Schranke tau 100: {r['violations']['100']['lower_share']:.1%}", flush=True)
            for name, m in (("Kopf", ph), (f"Clayton {th_best:g}", cb), ("unabhaengig", ind), ("Minimum", com)):
                print(f"      {name:14s} NLL {m['nll']:.4f}  IBS {m['ibs']:.5f}  Ctd {m['c_td']:.3f}  D-Cal {m['dcal_dev']:.3f}  "
                      f"PR@7 {m['pr@7']:.3f}  PR@30 {m['pr@30']:.3f}  ECE@30 {m['ece@30']:.4f}", flush=True)

    summary = {}
    for kind, seeds in res.items():
        rows = list(seeds.values())
        variants = ["patch_head", "clayton_val"] + [f"clayton_{th}" for th in THETAS]
        summary[kind] = {"theta_val": [r["theta_val"] for r in rows],
                         "any_upper_violation": float(np.mean([r["any_upper_violation"] for r in rows])),
                         "upper_share": {str(k): float(np.mean([r["violations"][str(k)]["upper_share"] for r in rows])) for k in TAUS}}
        for v in variants:
            ms = [r["test"][f"clayton_{r['theta_val']}"] if v == "clayton_val" else r["test"][v] for r in rows]
            summary[kind][v] = {k: {"mean": float(np.mean([m[k] for m in ms])), "std": float(np.std([m[k] for m in ms]))}
                                for k in ms[0]}
        s = summary[kind]
        print(f"[pc] {kind}: Verletzung irgendwo {s['any_upper_violation']:.1%}; theta(val) {s['theta_val']}", flush=True)
        for v in variants:
            print(f"      {v:16s} NLL {s[v]['nll']['mean']:.4f}  IBS {s[v]['ibs']['mean']:.5f}  Ctd {s[v]['c_td']['mean']:.3f}  "
                  f"PR@7 {s[v]['pr@7']['mean']:.3f}  PR@30 {s[v]['pr@30']['mean']:.3f}", flush=True)
    write_manifest(cfg, f"p20_patch_consistency{args.tag}", args.config, {
        "landmarks": args.surv_dir, "models": args.models, "thetas": [str(t) for t in THETAS],
        "bounds": "1 - sum F_i <= S_patch <= min S_i (Frechet)",
        "aggregation": "Clayton / Gamma-Frailty: S_R = (1 + sum (S_i^-theta - 1))^(-1/theta), theta auf Val per NLL",
        "results": res, "summary": summary,
    })


if __name__ == "__main__":
    main()
