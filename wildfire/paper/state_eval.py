"""Auswertung nach Zustand am Stichtag (U/O/B) und nach Typ des ersten Feuers auf den Test-Stichtagen (surv_eval_test).

Zustaende und Typen aus state_labels.py (derived_data/state_labels/eval_<keys>.npz). Je Ziel (Mitte, Ausschnitt):
  Strata     alle, U, O, O mit letztem Feuer <= 30 Tage (im Eingabefenster sichtbar), O > 30 Tage, B
             Kurvenmodelle: C^td, NLL, IBS, PR/AUC@7/@30 (evaluate); Basislinien aus classic_surv / classic_lit
             (rohe Vorhersagen, nur NLL und F7/F30 gespeichert): NLL, PR/AUC@7/@30
  Typ-AUC    je Zustand: Positive = erstes Feuer des Typs bis tau, Negative = kein Feuer bis tau im selben Zustand;
             Score F(tau), tau = 7, 30. Gruppen: U ignition / spread (r = 3, dazu r = 1), O return, return <= 30 d,
             return > 30 d, B continuation
  Intervalle Block-Bootstrap wie horizon_landmarks (Jahr x Woche x 64 x 64-Kachel, 200 Ziehungen) fuer NLL, PR@30,
             AUC@30 der Strata und fuer die Typ-AUC@30, dazu gepaart lh - x
Kurvenmodelle als Seed-Ensemble (Mittel der Survival-Kurven), nachkalibriert wie surv_eval --calibrate auf den
Val-Stichtagen 2020-2022: lh, cg (scratch), lincg, utae, convlstm; dazu km und km_fire.
Laedt die lineare Contagion und U-TAE / ConvLSTM ueber die Lader von contagion_linear und vision_backbones.

  python -m wildfire.paper.state_eval --config ... [--max-test N --max-val M --label-tag _smoke --tag _smoke]
  -> metrics/paper_multistate/state_eval<tag>.json; bricht ab, wenn vorhanden.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch

from wildfire.paper import sts_dctm

_ORIG = sts_dctm.load_sts
from wildfire.paper import contagion_linear, multistate, vision_backbones  # noqa: E402
from wildfire.audit.common import assert_readonly, load_config, write_manifest  # noqa: E402
from wildfire.eval.metrics import _roc_pr, evaluate  # noqa: E402
from wildfire.paper.horizon_landmarks import block_ids, boot_weights, ci, run_net  # noqa: E402
from wildfire.paper.reconstruct import cache_norm, to_device_once  # noqa: E402
from wildfire.paper.state_labels import STATES, T_CODE  # noqa: E402
from wildfire.paper.surv_eval import calib_apply, calib_fit, compact, ensemble  # noqa: E402
from wildfire.paper.surv_train import km_hazard  # noqa: E402

DAY0 = dt.date(2000, 1, 1)
CK = "/data/Wildfire/checkpoints/paper/sts_cache_G_next_cube_abl_"
CURVES = {"lh": CK + "lh_s?_s?.pt", "cg": CK + "cg_scratch_s?_s?.pt", "lincg": CK + "lincg_s?_s?.pt",
          "utae": CK + "utae_s?_s?.pt", "convlstm": CK + "convlstm_s?_s?.pt"}
CLASSIC = {"gbm": ("surv_eval.npz", "gbm"), "ipcw_gbm": ("lit_surv_eval.npz", "ipcw_gbm"),
           "xgb_aft": ("lit_surv_eval.npz", "xgb_aft")}


def load_any(path, device="cpu"):
    """Erkennt U-TAE / ConvLSTM und die lineare Contagion am state dict, sonst sts_dctm.load_sts."""
    st = torch.load(path, map_location="cpu")["state"]
    if vision_backbones.backbone_of(st) is not None:
        return vision_backbones.load_any(path, device)
    if multistate.is_multistate(st):
        return multistate.load_any(path, device)
    if contagion_linear.is_linear(st):
        return contagion_linear.load_any(path, device)
    return _ORIG(path, device)


def roc_pr(y, s, w=None):
    y = np.asarray(y, float)
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan"), float("nan")
    return _roc_pr(y, np.asarray(s, np.float64), np.ones(y.size) if w is None else w)[:2]


def cif_at(h: np.ndarray, taus=(7, 30)) -> dict:
    ls = np.cumsum(np.log1p(-np.clip(h.astype(np.float64), 1e-15, 1 - 1e-7)), 1)
    return {f"F{t}": -np.expm1(ls[:, t - 1]) for t in taus}


def nll_per_sample(h, e, t) -> np.ndarray:
    hb = np.clip(np.asarray(h, np.float64), 1e-15, 1 - 1e-7)
    csum = np.cumsum(np.log1p(-hb), axis=1)
    idx = np.arange(len(e))
    prior = np.where(t > 1, csum[idx, np.maximum(t - 2, 0)], 0.0)
    return -np.where(e, np.log(hb[idx, t - 1]) + prior, csum[:, -1])


def strata(lab: dict, target: str) -> dict:
    st, ds = lab[f"state_{target}"], lab[f"days_since_{target}"]
    return {"all": np.ones(st.size, bool), "U": st == 0, "O": st == 1, "O_le30": (st == 1) & (ds <= 30),
            "O_gt30": (st == 1) & (ds > 30), "B": st == 2}


def type_groups(lab: dict, target: str) -> dict:
    st, ds = lab[f"state_{target}"], lab[f"days_since_{target}"]
    t1, t3 = lab[f"type_{target}_r1"], lab[f"type_{target}_r3"]
    return {"U_ignition_r3": (st == 0, t3 == T_CODE["ignition"]), "U_spread_r3": (st == 0, t3 == T_CODE["spread"]),
            "U_ignition_r1": (st == 0, t1 == T_CODE["ignition"]), "U_spread_r1": (st == 0, t1 == T_CODE["spread"]),
            "O_return": (st == 1, t3 == T_CODE["return"]),
            "O_return_le30": ((st == 1) & (ds <= 30), t3 == T_CODE["return"]),
            "O_return_gt30": ((st == 1) & (ds > 30), t3 == T_CODE["return"]),
            "B_continuation": (st == 2, t3 == T_CODE["continuation"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--surv-dir", default="surv_eval_test")
    ap.add_argument("--cache-norm", default="cache_G")
    ap.add_argument("--keys", nargs="+", default=["index"], choices=("index", "raw"),
                    help="Label-Saetze; bei mehreren je einer Ausgabe state_eval<tag>_<keys>.json auf denselben Vorhersagen")
    ap.add_argument("--hazards-npz", default=None,
                    help="gespeicherte kalibrierte Ensemble-Hazards (msv2_eval: <modell>__<ziel>, alle Test-Stichtage)")
    ap.add_argument("--hazards", nargs="*", default=[], help="Namen aus --hazards-npz, optional NAME=ALIAS")
    ap.add_argument("--label-tag", default="")
    ap.add_argument("--max-test", type=int, default=0, help="Rauchtest: nur die ersten N Test-Stichtage")
    ap.add_argument("--max-val", type=int, default=0)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="*", default=list(CURVES))
    ap.add_argument("--extra", nargs="*", default=[], help="weitere Kurvenmodelle NAME=GLOB (werden an --models angehaengt)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    keys = list(dict.fromkeys(args.keys))
    md = exp / "metrics" / "paper_multistate"
    outs = {k: md / (f"state_eval{args.tag}.json" if len(keys) == 1 else f"state_eval{args.tag}_{k}.json") for k in keys}
    for o in outs.values():
        if o.exists():
            raise SystemExit(f"ABBRUCH: {o} existiert")
    md.mkdir(parents=True, exist_ok=True)
    sts_dctm.load_sts = load_any                                     # run_net laedt ueber sts_dctm.load_sts
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = exp / "derived_data" / args.surv_dir
    ev, va, km = dict(np.load(sd / "eval_meta.npz")), dict(np.load(sd / "val_meta.npz")), dict(np.load(sd / "km_meta.npz"))
    n = args.max_test or ev["row"].size
    ev = {k: v[:n] for k, v in ev.items()}
    nv = args.max_val or va["row"].size
    va = {k: v[:nv] for k, v in va.items()}
    labs = {}
    for k in keys:                                                    # zuerst alle Labels pruefen, dann rechnen
        lab = dict(np.load(exp / "derived_data" / "state_labels" / f"eval_{k}{args.label_tag}.npz"))
        assert lab["row"].size >= n, f"Labels {k} kuerzer als die Test-Stichtage"
        lab = {kk: v[:n] for kk, v in lab.items()}
        assert np.array_equal(lab["row"], ev["row"]) and np.array_equal(lab["L_abs"], ev["L_abs"]), f"Labels {k} passen nicht"
        labs[k] = lab
    year = np.array([(DAY0 + dt.timedelta(days=int(d))).year for d in ev["L_abs"]])
    blocks = block_ids(ev, year)
    tg = {c: (ev[f"ev_{c}"].astype(bool), ev[f"t_{c}"].astype(int)) for c in ("centre", "patch")}
    tgv = {c: (va[f"ev_{c}"].astype(bool), va[f"t_{c}"].astype(int)) for c in ("centre", "patch")}
    print(f"[state] Test {n:,}, Val {nv:,}", flush=True)

    # ---------------- Vorhersagen: Kurvenmodelle (kalibrierte Ensembles), KM, klassische Basislinien
    H_: dict = {}
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / args.cache_norm / "norm_stats.npz")
    Xt = to_device_once(np.ascontiguousarray(np.load(sd / "eval_X.npy", mmap_mode="r")[:n]), device, "eval_X")
    Xv = to_device_once(np.ascontiguousarray(np.load(sd / "val_X.npy", mmap_mode="r")[:nv]), device, "val_X")
    calib: dict = {}
    curves = dict(CURVES)
    for spec in args.extra:
        nm, pat = spec.split("=", 1)
        curves[nm] = pat
    for name in list(args.models) + [s.split("=", 1)[0] for s in args.extra]:
        files = sorted(glob.glob(curves[name]))
        if not files:
            print(f"[state] {name}: keine Checkpoints", flush=True)
            continue
        T, V = {"centre": [], "patch": []}, {"centre": [], "patch": []}
        for f in files:
            t0 = time.time()
            rt, rv = run_net(f, Xt, ev["doy"], mean, std, device), run_net(f, Xv, va["doy"], mean, std, device)
            for c, k in (("centre", "hc"), ("patch", "hp")):
                T[c].append(rt[k]); V[c].append(rv[k])
            print(f"    {Path(f).name} in {time.time() - t0:.0f} s", flush=True)
        for c in ("centre", "patch"):
            a, b = calib_fit(ensemble(V[c]), *tgv[c])
            calib.setdefault(name, {})[c] = {"a": a, "b": b, "n_seeds": len(files)}
            H_[(name, c)] = calib_apply(ensemble(T[c]), a, b).astype(np.float32)
    if args.hazards:                                                  # gespeicherte kalibrierte Hazards (z. B. msv2_eval)
        z = np.load(args.hazards_npz)
        for spec in args.hazards:
            nm, alias = (spec.split("=", 1) + [spec])[:2]
            for c in ("centre", "patch"):
                arr = z[f"{nm}__{c}"]
                assert arr.shape[0] >= n, f"{nm}: nur {arr.shape[0]} Zeilen"
                H_[(alias, c)] = np.clip(arr[:n].astype(np.float32), 1e-15, 1 - 1e-7)
            curves[alias] = f"npz:{args.hazards_npz}:{nm}"
            print(f"[state] {alias}: Hazards aus {Path(args.hazards_npz).name} ({nm})", flush=True)
    for c in ("centre", "patch"):
        H_[("km", c)] = np.tile(km_hazard(km[f"ev_{c}"], km[f"t_{c}"]), (n, 1)).astype(np.float32)
        fk = km[f"fire_win_{c}"].astype(bool)
        kmf = {f: km_hazard(km[f"ev_{c}"][fk == f], km[f"t_{c}"][fk == f]) for f in (False, True)}
        H_[("km_fire", c)] = np.stack([kmf[bool(f)] for f in ev[f"fire_win_{c}"]]).astype(np.float32)
    P_: dict = {}                                                     # je Modell/Ziel: nll, F7, F30 je Stichtag
    for (name, c), h in H_.items():
        P_[(name, c)] = {"nll": nll_per_sample(h, *tg[c]), **cif_at(h)}
    for name, (fn, key) in CLASSIC.items():
        p = exp / "metrics" / "paper_classic" / fn
        if not p.exists():
            continue
        z = np.load(p)
        pos = {int(i): j for j, i in enumerate(z["idx"])}
        if not all(i in pos for i in range(n)) or f"{key}__patch__nll" not in z.files:
            print(f"[state] {name}: Stichtage passen nicht, entfaellt", flush=True)
            continue
        sel = np.array([pos[i] for i in range(n)])
        for c in ("centre", "patch"):
            P_[(name, c)] = {f: z[f"{key}__{c}__{f}"][sel].astype(np.float64) for f in ("nll", "F7", "F30")}
    models = sorted({m for m, _ in P_}, key=lambda m: (m not in curves, m))

    # ---------------- Strata und Typ-AUC mit Bootstrap
    for key in keys:
        lab, out = labs[key], outs[key]
        rng = np.random.default_rng(args.seed)                         # gleiche Ziehungen fuer jeden Label-Satz
        print(f"[state] ===== Labels {key} =====", flush=True)
        res: dict = {"calibration": calib, "strata": {}, "types": {}, "label_shares": {}}
        for c in ("centre", "patch"):
            e, t = tg[c]
            S = strata(lab, c)
            res["label_shares"][c] = {k: {"n": int(m.sum()), "n_events": int((e & m).sum())} for k, m in S.items()}
            for sname, m in S.items():
                if m.sum() < 50:
                    continue
                W = boot_weights(np.unique(blocks[m], return_inverse=True)[1], args.boot, rng)
                em, tm = e[m], t[m]
                row, bsv = {}, {}
                for mod in models:
                    ps = {k: v[m] for k, v in P_[(mod, c)].items()}
                    r = {"nll": float(ps["nll"].mean())}
                    for tau in (7, 30):                                     # beobachtet / erwartet bis tau
                        r[f"oe@{tau}"] = float((em & (tm <= tau)).sum() / max(ps[f"F{tau}"].sum(), 1e-12))
                    for tau in (7, 30):
                        y = em & (tm <= tau)
                        r[f"auc@{tau}"], r[f"pr@{tau}"] = roc_pr(y, ps[f"F{tau}"])
                    if (mod, c) in H_ and em.any():
                        cm = compact(evaluate(H_[(mod, c)][m], em, tm))
                        r.update({"c_td": cm["c_td"], "ibs": cm["ibs"]})
                    y30 = em & (tm <= 30)
                    bs = []
                    for w in W:
                        a30, p30 = roc_pr(y30, ps["F30"], w)
                        bs.append({"nll": float((w * ps["nll"]).sum() / w.sum()), "auc@30": a30, "pr@30": p30,
                                   "oe@30": float((w * y30).sum() / max((w * ps["F30"]).sum(), 1e-12))})
                    bsv[mod] = bs
                    row[mod] = {k: {"value": v, **(ci([b[k] for b in bs]) if k in bs[0] else {})} for k, v in r.items()}
                if "lh" in bsv:
                    for mod in bsv:
                        if mod != "lh":
                            row[mod]["lh_minus"] = {k: {"value": row["lh"][k]["value"] - row[mod][k]["value"],
                                                        **ci([bl[k] - bx[k] for bl, bx in zip(bsv["lh"], bsv[mod])])}
                                                    for k in ("nll", "auc@30", "pr@30", "oe@30")}
                res["strata"].setdefault(c, {})[sname] = {"n": int(m.sum()), "n_events": int(em.sum()),
                                                          "event_rate_30": float((em & (tm <= 30)).mean()), "models": row}
                print(f"[state] {c:6s} {sname:7s} n {int(m.sum()):6,} Ereignisse<=30 {int((em & (tm <= 30)).sum()):5,}: " +
                      "  ".join(f"{mod} Ctd {row[mod].get('c_td', {}).get('value', float('nan')):.3f} "
                                f"AUC30 {row[mod]['auc@30']['value']:.3f} PR30 {row[mod]['pr@30']['value']:.3f} "
                                f"NLL {row[mod]['nll']['value']:.4f} OE30 {row[mod]['oe@30']['value']:.2f}" for mod in models),
                      flush=True)
            for gname, (sm, tmask) in type_groups(lab, c).items():
                rowg, bsv = {}, {}
                for tau in (7, 30):
                    pos_ = sm & e & (t <= tau) & tmask
                    neg_ = sm & ~(e & (t <= tau))
                    use = pos_ | neg_
                    if pos_.sum() < 10:
                        continue
                    y = pos_[use]
                    W = boot_weights(np.unique(blocks[use], return_inverse=True)[1], args.boot, rng) if tau == 30 else None
                    for mod in models:
                        s = P_[(mod, c)][f"F{tau}"][use]
                        auc, pr = roc_pr(y, s)
                        d = rowg.setdefault(mod, {})
                        d[f"auc@{tau}"], d[f"pr@{tau}"] = {"value": auc}, {"value": pr}
                        if W is not None:
                            b = [roc_pr(y, s, w)[0] for w in W]
                            bsv[mod] = b
                            d["auc@30"].update(ci(b))
                    rowg["_n"] = rowg.get("_n", {}) | {f"pos@{tau}": int(pos_.sum()), f"neg@{tau}": int(neg_.sum()),
                                                       f"prevalence@{tau}": float(y.mean())}
                if "lh" in bsv:
                    for mod in bsv:
                        if mod != "lh":
                            rowg[mod]["lh_minus_auc@30"] = {"value": rowg["lh"]["auc@30"]["value"] - rowg[mod]["auc@30"]["value"],
                                                            **ci(np.array(bsv["lh"]) - np.array(bsv[mod]))}
                if rowg:
                    res["types"].setdefault(c, {})[gname] = rowg
                    print(f"[state] {c:6s} Typ {gname:15s} {rowg['_n']}: " + "  ".join(
                        f"{mod} AUC7 {rowg[mod].get('auc@7', {}).get('value', float('nan')):.3f} "
                        f"AUC30 {rowg[mod].get('auc@30', {}).get('value', float('nan')):.3f}" for mod in models
                        if mod in rowg), flush=True)
        meta = {"args": vars(args), "n_test": n, "n_val": nv, "models": models, "curves": curves, "classic": CLASSIC,
                "states": STATES, "note": "Kurvenmodelle kalibriert (Val 2020-2022); klassische Basislinien roh", **res}
        out.write_text(json.dumps(meta, indent=1, default=float))
        write_manifest(cfg, f"p40_state_eval{args.tag}" + ("" if len(keys) == 1 else f"_{key}"), args.config,
                       {"output": str(out), "n_test": n, "keys": key})
        print(f"[state] -> {out}", flush=True)


if __name__ == "__main__":
    main()
