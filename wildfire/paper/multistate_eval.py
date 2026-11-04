"""Auswertung der Multi-State-Contagion auf den zufaelligen Test-Stichtagen (surv_eval_test, 2023-2025).

Zustand eines Pixels am Stichtag L aus dem Feuerkanal des 30-Tage-Fensters (dieselbe Information wie im Modell):
  B  brennend    Feuer in [L-6, L]
  O  erloschen   kein Feuer in [L-6, L], aber in [L-29, L-7]
  U  unverbrannt sonst
Ereignistyp des ersten Feuers nach L (Pixel-Ereignis aus region_eval/pixel_events.npz):
  cont  Pixel in B      ret  Pixel in O
  ign   Pixel in U, NFDB-Zuendung (eval_cause Code 1-3)      spr  Pixel in U, Ausbreitung (Code 4)
  sonst ohne Typ

  1. Zuordnung   Anteil der Ursache c am Hazard des Ereignistags pi_c = dF_c(k) / dF(k) (decompose_causes):
                 mittleres pi_c je wahrem Typ (Matrix) und ROC-AUC von pi_c fuer "Typ = c" gegen die anderen Typen
  2. Ursachen-CIF Mitte: "erstes Feuer bis tau mit Typ c" - ROC-AUC und PR-AUC von F_c(tau) gegen F(tau), tau 7/30
  3. Population  Mitte nach Zustand am Stichtag (U, O, B): NLL, C^td, PR@7/30 (evaluate) - wie gut trifft das
                 Modell die Rueckkehr-Population O?
Modelle: Multi-State (multistate.py) und ContagionSTS-Kontrollen (Rueckkehr = 0); Laden ueber multistate.load_any.
Ausgabe NEU: metrics/paper_multistate/ms_eval<tag>.json (bricht ab, wenn sie existiert).
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
from sklearn.metrics import average_precision_score, roc_auc_score

from wildfire.audit.common import assert_readonly, load_config
from wildfire.eval.metrics import evaluate
from wildfire.paper.cause_labels import SPREAD
from wildfire.paper.multistate import decompose_causes, load_any
from wildfire.paper.prepare import FIRE_CH
from wildfire.paper.reconstruct import cache_norm
from wildfire.paper.region_eval import compact, hazards_from_logS

H = 100
P = 13
N = P * P
C = N // 2
TYPES = ("ign", "spr", "cont", "ret")


def states(Xu8: np.ndarray) -> np.ndarray:
    """(n, 30, C, 13, 13) uint8 -> Zustand je Pixel (n, 169): 0 U, 1 O, 2 B."""
    f = Xu8[:, :, FIRE_CH] > 0                                                         # (n, 30, 13, 13)
    b = f[:, -7:].any(1)
    o = f[:, :-7].any(1) & ~b
    return np.where(b, 2, np.where(o, 1, 0)).reshape(len(Xu8), N).astype(np.int8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--surv-dir", default="surv_eval_test")
    ap.add_argument("--cache-norm", default="cache_G")
    ap.add_argument("--models", nargs="+", default=[
        "/data/Wildfire/checkpoints/paper_cause/sts_cache_G_next_cube_cause_ms_lam0_scratch_s?.pt",
        "/data/Wildfire/checkpoints/paper_cause/sts_cache_G_next_cube_cause_ms_lam1_scratch_s?.pt",
        "/data/Wildfire/checkpoints/paper_cause/sts_cache_G_next_cube_cause_lam0_scratch_s?.pt",
        "/data/Wildfire/checkpoints/paper_cause/sts_cache_G_next_cube_cause_lam1_scratch_s?.pt",
        "/data/Wildfire/checkpoints/paper/sts_cache_G_next_cube_abl_cg_scratch_s?_s?.pt"])
    ap.add_argument("--n", type=int, default=30_000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    out = exp / "metrics" / "paper_multistate" / f"ms_eval{args.tag}.json"
    if out.exists():
        raise SystemExit(f"ABBRUCH: {out} existiert schon - anderes --tag waehlen")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = exp / "derived_data" / args.surv_dir
    me = dict(np.load(sd / "eval_meta.npz"))
    pe = dict(np.load(exp / "derived_data" / "region_eval" / "pixel_events.npz"))
    lab = dict(np.load(exp / "derived_data" / "cause_labels" / "eval_cause.npz"))
    assert (lab["row"] == me["row"]).all() and (lab["L_abs"] == me["L_abs"]).all(), "Stichtage passen nicht"
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(me["row"]), size=min(args.n, len(me["row"])), replace=False))
    t0 = time.time()
    Xu8 = np.ascontiguousarray(np.load(sd / "eval_X.npy", mmap_mode="r")[idx])
    X = torch.from_numpy(Xu8)
    doy = me["doy"][idx].astype(np.float32)
    ev = pe["ev"][idx].reshape(len(idx), N).astype(bool)
    tt = np.clip(pe["t"][idx].reshape(len(idx), N).astype(np.int64), 1, H)
    code = lab["cause"][idx].reshape(len(idx), N)
    st = states(Xu8)
    typ = np.full(ev.shape, -1, np.int8)                                               # Index in TYPES
    typ[ev & (st == 2)] = 2
    typ[ev & (st == 1)] = 3
    typ[ev & (st == 0) & (code >= 1) & (code <= 3)] = 0
    typ[ev & (st == 0) & (code == SPREAD)] = 1
    res = {"n": int(len(idx)), "state_share": {s: float((st == i).mean()) for i, s in enumerate("UOB")},
           "event_type_counts": {t: int((typ == i).sum()) for i, t in enumerate(TYPES)},
           "events": int(ev.sum()), "models": {}}
    print(f"[mseval] {len(idx):,} Stichtage ({time.time() - t0:.0f} s); Zustaende {res['state_share']}; "
          f"Ereignisse {res['events']:,}, Typen {res['event_type_counts']}", flush=True)
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / args.cache_norm / "norm_stats.npz")
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]
    lab_ev = typ >= 0
    rows_ev, pix_ev = np.nonzero(lab_ev)
    k_ev = tt[rows_ev, pix_ev]

    files = [f for pat in args.models for f in sorted(glob.glob(pat))]
    for ck in files:
        stem = Path(ck).stem
        name = re.sub(r"(_s\d+)+$", "", stem)
        seed = re.search(r"_s(\d+)", stem[len(name):]).group(1)
        net = load_any(ck, device)
        t1 = time.time()
        pi = np.zeros((len(rows_ev), len(TYPES)))
        cen = {kk: [] for kk in ("ign", "spr", "cont", "ret", "logS")}
        for a0 in range(0, len(idx), args.bs):
            x = (X[a0:a0 + args.bs].to(device).float() - m_t) / s_t
            d = torch.from_numpy(doy[a0:a0 + args.bs]).to(device)
            with torch.autocast(device_type=device.type, enabled=False):
                r = decompose_causes(net, x, d)
            Fm = {kk: r[kk].double().cpu().numpy() for kk in TYPES}                    # (b, N, H+1) CIF je Ursache
            for kk in cen:
                cen[kk].append((r[kk] if kk == "logS" else r[kk]).double().cpu().numpy()[:, C])
            sel = (rows_ev >= a0) & (rows_ev < a0 + len(x))
            if sel.any():
                rr, pp, kk_ = rows_ev[sel] - a0, pix_ev[sel], k_ev[sel]
                dF = np.stack([Fm[c][rr, pp, kk_] - Fm[c][rr, pp, kk_ - 1] for c in TYPES], 1)
                pi[sel] = dF / np.clip(dF.sum(1, keepdims=True), 1e-300, None)
        cen = {kk: np.concatenate(v) for kk, v in cen.items()}
        m = {"attribution_matrix": {}, "attribution_auc": {}, "cause_cif_centre": {}, "population_centre": {}}
        y = typ[rows_ev, pix_ev]
        for i, t in enumerate(TYPES):
            if (y == i).any():
                m["attribution_matrix"][t] = {c: float(pi[y == i, j].mean()) for j, c in enumerate(TYPES)}
            if 0 < (y == i).sum() < len(y) and pi[:, i].std() > 0:
                m["attribution_auc"][t] = float(roc_auc_score(y == i, pi[:, i]))
        Ftot = 1 - np.exp(cen["logS"])
        for i, t in enumerate(TYPES):
            for tau in (7, 30):
                yy = (typ[:, C] == i) & (tt[:, C] <= tau)
                if 0 < yy.sum() < len(yy):
                    m["cause_cif_centre"][f"{t}@{tau}"] = {
                        "n": int(yy.sum()),
                        "auc_Fc": float(roc_auc_score(yy, cen[t][:, tau])), "pr_Fc": float(average_precision_score(yy, cen[t][:, tau])),
                        "auc_F": float(roc_auc_score(yy, Ftot[:, tau])), "pr_F": float(average_precision_score(yy, Ftot[:, tau])),
                        "mean_Fc": float(cen[t][:, tau].mean()), "obs_rate": float(yy.mean())}
        hz = hazards_from_logS(cen["logS"])
        for i, s in enumerate("UOB"):
            sel = st[:, C] == i
            if sel.sum() > 50 and ev[sel, C].any():
                m["population_centre"][s] = {"n": int(sel.sum()), **compact(evaluate(hz[sel], ev[sel, C], tt[sel, C]))}
        if getattr(net, "multistate", False):
            m["fuel_gate"] = float(torch.sigmoid(net.fuel).item())
        res["models"].setdefault(name, {})[seed] = m
        o = m["population_centre"].get("O", {})
        print(f"[mseval] {name} s{seed} ({time.time() - t1:.0f} s): Zuordnung AUC "
              + " ".join(f"{k}={v:.3f}" for k, v in m["attribution_auc"].items())
              + f"; Mitte in O: NLL {o.get('nll', float('nan')):.4f} C^td {o.get('c_td', float('nan')):.3f}"
              + (f"; g = {m['fuel_gate']:.3f}" if "fuel_gate" in m else ""), flush=True)
        for t, row in m["attribution_matrix"].items():
            print(f"[mseval]     wahrer Typ {t:4s}: " + "  ".join(f"pi_{c} {v:.3f}" for c, v in row.items()), flush=True)
        del net
        torch.cuda.empty_cache()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), **res}, indent=1))
    print(f"[mseval] {out}", flush=True)


if __name__ == "__main__":
    main()
