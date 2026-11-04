"""Auswertung der Ursachen-Zerlegung auf den zufaelligen Test-Stichtagen (surv_eval_test, 2023-2025).

  1. Zuordnung   Test-Ereignisse mit Label (cause_labels --part eval): trennt der Anteil
                 pi_ign = a(k) / h(k) Zuendungen von Ausbreitungen? ROC-AUC, mittleres pi je Klasse.
  2. Zuendung    Zielgroesse: erste NFDB-Zuendung (>= 1 ha) im 13x13-Ausschnitt bzw. in der Mitte innerhalb
                 von 100 Tagen, nur Stichtage in der NFDB-beobachteten Maske. Scores:
                   cg_ign    Contagion: nur der Zuendungs-Hazard, F_ign = 1 - prod (1 - a)
                   cg_total  Contagion: ganze Kurve (Ausschnitt exakt, Mitte Rollout)
                   lh        STS-LH: Ausschnittskopf bzw. Pixelkopf der Mitte
                 Metriken wie surv_eval (NLL, C^td, D-cal, PR-AUC je tau), gesamt und getrennt nach
                 Feuer im 30-Tage-Fenster ja / nein.
Alle Eingaben nur gelesen. Ausgabe NEU: metrics/paper_cause/cause_eval<tag>.json (bricht ab, wenn sie existiert).
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from wildfire.audit.common import assert_readonly, load_config
from wildfire.eval.metrics import evaluate
from wildfire.paper.cause_labels import SPREAD, nfdb_keys
from wildfire.paper.pixel_landmarks import pixel_events
from wildfire.paper.reconstruct import cache_norm
from wildfire.paper.region_eval import compact, hazards_from_logS
from wildfire.paper.sts_dctm import load_sts

H = 100
P = 13
HALF = P // 2
N = P * P


@torch.no_grad()
def contagion_scores(ck, X, doy, m_t, s_t, ev, tt, code, device, bs):
    """-> log S Zuendung / gesamt fuer Ausschnitt und Mitte (n, H+1), pi_ign und Klasse der Label-Ereignisse."""
    net = load_sts(ck, device)
    out = {"ign_patch": [], "tot_patch": [], "ign_centre": [], "tot_centre": []}
    pis, cls = [], []
    for a0 in range(0, X.shape[0], bs):
        x = (X[a0:a0 + bs].to(device).float() - m_t) / s_t
        d = torch.from_numpy(doy[a0:a0 + bs]).to(device)
        with torch.autocast(device_type=device.type, enabled=False):
            obs, a, k, logD = net.params(x, d)
            log_exo = net.exogenous(obs, a, k, logD)
            ls_px = net.rollout(log_exo, k, logD)                                 # (b, N, H+1)
        l_ign = torch.log1p(-a.clamp(max=1 - 1e-6))                               # (b, H, N)
        out["ign_patch"].append(F.pad(l_ign.sum(2).cumsum(1), (1, 0)).double().cpu().numpy())
        out["tot_patch"].append(F.pad(log_exo.sum(2).cumsum(1), (1, 0)).double().cpu().numpy())
        out["ign_centre"].append(F.pad(l_ign[:, :, N // 2].cumsum(1), (1, 0)).double().cpu().numpy())
        out["tot_centre"].append(ls_px[:, N // 2].double().cpu().numpy())
        # Zuordnung auf Label-Ereignissen
        b_ = slice(a0, a0 + x.shape[0])
        e, t, c = (torch.from_numpy(v[b_].reshape(-1, N)).to(device) for v in (ev, tt, code))
        lab = e.bool() & (((c >= 1) & (c <= 3)) | (c == SPREAD))
        if lab.any():
            bi, i = torch.nonzero(lab, as_tuple=True)
            kk = t[bi, i].long()
            l1mh = ls_px[bi, i, kk] - ls_px[bi, i, kk - 1]
            h = (-torch.expm1(l1mh)).clamp_min(1e-12)
            pis.append((a[bi, kk - 1, i] / h).clamp(max=1).cpu().numpy())
            cls.append(((c[bi, i] >= 1) & (c[bi, i] <= 3)).cpu().numpy())
    res = {k: np.concatenate(v) for k, v in out.items()}
    pi, y = (np.concatenate(pis), np.concatenate(cls)) if pis else (np.zeros(0), np.zeros(0, bool))
    return res, pi, y


@torch.no_grad()
def lh_scores(ck, X, doy, m_t, s_t, device, bs):
    net = load_sts(ck, device)
    pa, ce = [], []
    for a0 in range(0, X.shape[0], bs):
        x = (X[a0:a0 + bs].to(device).float() - m_t) / s_t
        d = torch.from_numpy(doy[a0:a0 + bs]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            raw_px, raw_pa = net(x, d)
        for lst, raw in ((pa, raw_pa), (ce, raw_px[:, HALF, HALF])):
            h = net.px_head.hazards(raw, H).double() if lst is ce else net.pa_head.hazards(raw, H).double()
            lst.append(F.pad(torch.log1p(-h.clamp(max=1 - 1e-7)).cumsum(1), (1, 0)).cpu().numpy())
    return {"tot_patch": np.concatenate(pa), "tot_centre": np.concatenate(ce)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--surv-dir", default="surv_eval_test")
    ap.add_argument("--cache-norm", default="cache_G")
    ap.add_argument("--models", nargs="+", default=[
        "cg:/data/Wildfire/checkpoints/paper_cause/sts_cache_G_next_cube_cause_lam*_s?.pt",
        "cg:/data/Wildfire/checkpoints/paper/sts_cache_G_next_cube_abl_cg_s?_s?.pt",
        "lh:/data/Wildfire/checkpoints/paper/sts_cache_G_next_cube_abl_lh_s?_s?.pt"])
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    out = exp / "metrics" / "paper_cause" / f"cause_eval{args.tag}.json"
    if out.exists():
        raise SystemExit(f"ABBRUCH: {out} existiert schon - anderes --tag waehlen")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = exp / "derived_data" / args.surv_dir
    me = dict(np.load(sd / "eval_meta.npz"))
    pe = dict(np.load(exp / "derived_data" / "region_eval" / "pixel_events.npz"))
    lab = dict(np.load(exp / "derived_data" / "cause_labels" / "eval_cause.npz"))
    assert (lab["row"] == me["row"]).all() and (lab["L_abs"] == me["L_abs"]).all(), "Stichtage passen nicht"
    observed = np.load(exp / "derived_data" / "masks" / "observed_nfdb.npy").astype(bool)
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(me["row"]), size=min(args.n, len(me["row"])), replace=False))
    idx = idx[observed[me["row"][idx], me["col"][idx]]]                       # nur NFDB-beobachtete Stichtage
    t0 = time.time()
    X = torch.from_numpy(np.ascontiguousarray(np.load(sd / "eval_X.npy", mmap_mode="r")[idx]))
    doy = me["doy"][idx].astype(np.float32)
    ev, tt, code = pe["ev"][idx], pe["t"][idx].astype(np.int64), lab["cause"][idx]
    fw = me["fire_win_patch"][idx].astype(bool)
    print(f"[ceval] {len(idx):,} Stichtage (NFDB-beobachtet) in {time.time() - t0:.0f} s", flush=True)

    # Zielgroesse: erste NFDB-Zuendung im Ausschnitt / in der Mitte
    W = observed.shape[1]
    nk, _ = nfdb_keys(exp / "derived_data" / "events" / "ignitions_nfdb_min1ha.csv", W)
    e_ig, t_ig = pixel_events(nk, me["L_abs"][idx].astype(np.int64), me["row"][idx], me["col"][idx],
                              observed.shape[0], W)
    targets = {"patch": (e_ig.reshape(len(idx), -1).any(1), np.clip(t_ig.reshape(len(idx), -1).min(1), 1, H)),
               "centre": (e_ig[:, HALF, HALF], np.clip(t_ig[:, HALF, HALF], 1, H))}
    res = {"n": int(len(idx)), "ignition_rate": {k: float(v[0].mean()) for k, v in targets.items()},
           "share_fire_in_window": float(fw.mean()), "models": {}, "attribution": {}}
    print(f"[ceval] NFDB-Zuendung bis 100 Tage: Ausschnitt {res['ignition_rate']['patch']:.4f}, "
          f"Mitte {res['ignition_rate']['centre']:.5f}", flush=True)
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / args.cache_norm / "norm_stats.npz")
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]

    def score(name, seed, curves):
        for key, ls in curves.items():
            kind, tg = key.split("_")
            e, t = targets[tg]
            for sub, m in (("all", slice(None)), ("fire", fw), ("nofire", ~fw)):
                r = compact(evaluate(hazards_from_logS(ls[m]), e[m], t[m]))
                res["models"].setdefault(name, {}).setdefault(f"{kind}_{tg}_{sub}", {})[seed] = r
            a = res["models"][name][f"{kind}_{tg}_nofire"][seed]
            print(f"[ceval] {name:28s} {seed} {key:11s} ohne Feuer im Fenster: NLL {a['nll']:.4f} "
                  f"C^td {a['c_td']:.3f} PR@7 {a['pr@7']:.3f} PR@30 {a['pr@30']:.3f}", flush=True)

    for spec in args.models:
        kind, pat = spec.split(":", 1)
        for ck in sorted(glob.glob(pat)):
            stem = Path(ck).stem
            name, seed = stem.rsplit("_s", 1)
            name = re.sub(r"_s\d+$", "", name)
            if kind == "cg":
                cur, pi, y = contagion_scores(ck, X, doy, m_t, s_t, ev, tt, code, device, args.bs)
                if y.size and y.any() and (~y).any():
                    res["attribution"].setdefault(name, {})[seed] = {
                        "auc": float(roc_auc_score(y, pi)), "n_ign": int(y.sum()), "n_spread": int((~y).sum()),
                        "mean_pi_ign": float(pi[y].mean()), "mean_pi_spread": float(pi[~y].mean())}
                    r = res["attribution"][name][seed]
                    print(f"[ceval] {name} {seed}: Zuordnung AUC {r['auc']:.3f}  pi_ign bei Zuendung "
                          f"{r['mean_pi_ign']:.3f}, bei Ausbreitung {r['mean_pi_spread']:.3f}", flush=True)
            else:
                cur = lh_scores(ck, X, doy, m_t, s_t, device, args.bs)
            score(name, seed, cur)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), **res}, indent=1))
    print(f"[ceval] {out}", flush=True)


if __name__ == "__main__":
    main()
