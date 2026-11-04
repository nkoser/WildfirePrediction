"""Auswertung Multi-State v2 und Kontrollen: zufaellige Test-Stichtage (surv_eval + state_eval in einem) und
FireMPC-Protokoll. Modelle mit Zustandsmerkmalen (multistate_v2) bekommen die Merkmale aus state_features.

  surv   Test 2023-2025 (surv_eval_test), Kalibrierung Lambda' = e^a Lambda^b auf den Val-Stichtagen 2020-2022
         (Seed-Ensemble wie state_eval; dazu je Seed kalibriert):
           gesamt     Mitte und Ausschnitt, auch getrennt nach Feuer im 30-Tage-Fenster: compact(evaluate)
           Strata     U, O, O <= 30 d, O > 30 d, B (state_labels, 183-Tage-O): NLL, C^td, AUC/PR@30, O/E@30
           Typ-AUC    wie state_eval (Positive = erstes Feuer des Typs bis tau, Negative = kein Feuer im selben Zustand)
           Intervalle Block-Bootstrap (Jahr x Woche x 64er-Kachel), gepaart gegen --ref
           je Zustand --state-cal: zusaetzlich Kalibrierung getrennt je Zustand U/O/B (Merkmale), <name>_scal
           Zuordnung  Multi-State-Modelle, --n-attr Test-Stichtage: Anteil der Ursache am Ereignistag (decompose_causes)
                      je wahrem Pixel-Typ (state_features.train_types-Regel: B cont, O ret, U + Nachbar 7 x 7 spr, sonst ign)
  paper  FireMPC-Protokoll (paper/<cache>_L61): Score F(dt + 1) am Fensterende e = L - dt - 2, PR-AUC roh je Seed
         und als Seed-Ensemble (Mittel der Scores), Ausschnitt und Mitte, Test-Split; neben den Werten des Papers

  python -m wildfire.paper.msv2_eval surv|paper --config ... --models NAME=GLOB ... [--ref NAME] [--tag _x]
  -> metrics/paper_multistate/msv2_<teil><tag>.json (+ Hazards .npz bei surv); bricht ab, wenn vorhanden.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

from wildfire.paper import sts_dctm, state_eval
from wildfire.paper import multistate_v2 as msv2
from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.eval.metrics import evaluate
from wildfire.paper.horizon_landmarks import block_ids, boot_weights, ci
from wildfire.paper.multistate import decompose_causes
from wildfire.paper.reconstruct import cache_norm, to_device_once
from wildfire.paper.state_eval import cif_at, nll_per_sample, roc_pr, strata, type_groups
from wildfire.paper.state_features import TYPE_NAMES, train_types
from wildfire.paper.state_labels import raw_keys
from wildfire.paper.surv_eval import calib_apply, calib_fit, compact, ensemble

H, HALF, P = 100, 6, 13
DAY0 = dt.date(2000, 1, 1)
PAPER_REPORTED = {"patch": {1: 0.718, 7: 0.598, 15: 0.551, 30: 0.481}}       # 3D U-Net, FireMPC (Paper-Werte)


def load_net(f, device):
    if msv2.is_msv2(f):
        return msv2.load(f, device)
    return state_eval.load_any(f, device)


@torch.no_grad()
def hazards(net, X, doy, feats, mean, std, device, bs=64):
    """-> hc, hp (n, H) float32; feats (n, 169, NF) Tensor auf device oder None."""
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]
    use_f = msv2.needs_features(net)
    hc, hp = [], []
    for a in range(0, X.shape[0], bs):
        x = (X[a:a + bs].float() - m_t) / s_t
        if use_f:
            net.set_features(feats[a:a + bs].float())
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            raw_px, raw_pa = net(x, torch.from_numpy(doy[a:a + bs]).to(device))
        hc.append(net.px_head.hazards(raw_px[:, HALF, HALF].float(), H).float().cpu().numpy())
        hp.append(net.pa_head.hazards(raw_pa.float(), H).float().cpu().numpy())
    return np.concatenate(hc), np.concatenate(hp)


def state_calibrate(hv, ht, sv, st, yv, ab_all, min_events: int = 30):
    """Lambda' = e^a Lambda^b getrennt je Zustand (U/O/B), angepasst auf den Val-Stichtagen dieses Zustands;
    zu wenige Val-Ereignisse -> gemeinsame Abbildung ab_all. -> kalibrierte Test-Hazards, Parameter je Zustand."""
    out = np.empty_like(ht, dtype=np.float64)
    cal = {}
    for s, nm in enumerate(("U", "O", "B")):
        mv, mt = sv == s, st == s
        nev = int(yv[0][mv].sum())
        a, b = calib_fit(hv[mv], yv[0][mv], yv[1][mv]) if nev >= min_events else ab_all
        cal[nm] = {"a": a, "b": b, "n_val": int(mv.sum()), "n_val_events": nev, "own_fit": nev >= min_events}
        if mt.any():
            out[mt] = calib_apply(ht[mt], a, b)
    return out, cal


def model_files(specs):
    out = {}
    for s in specs:
        nm, pat = s.split("=", 1)
        files = sorted(glob.glob(pat))
        if files:
            out[nm] = files
        else:
            print(f"[msv2e] {nm}: keine Checkpoints ({pat})", flush=True)
    return out


def seed_of(f) -> str:
    return re.findall(r"_s(\d+)", Path(f).stem)[-1]


# ------------------------------------------------------------------ surv
def run_surv(args, cfg, exp, out):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    sd = exp / "derived_data" / args.surv_dir
    ev, va = dict(np.load(sd / "eval_meta.npz")), dict(np.load(sd / "val_meta.npz"))
    n, nv = args.max_test or ev["row"].size, args.max_val or va["row"].size
    ev, va = {k: v[:n] for k, v in ev.items()}, {k: v[:nv] for k, v in va.items()}
    lab = dict(np.load(exp / "derived_data" / "state_labels" / f"eval_index{args.label_tag}.npz"))
    lab = {k: v[:n] for k, v in lab.items()}
    assert np.array_equal(lab["row"], ev["row"]) and np.array_equal(lab["L_abs"], ev["L_abs"]), "Labels passen nicht"
    fd = exp / "derived_data" / "state_features"
    Ft = torch.from_numpy(np.ascontiguousarray(np.load(fd / f"eval{args.feat_tag}.npy", mmap_mode="r")[:n])).to(device)
    Fv = torch.from_numpy(np.ascontiguousarray(np.load(fd / f"val{args.feat_tag}.npy", mmap_mode="r")[:nv])).to(device)
    assert Ft.shape[0] == n and Fv.shape[0] == nv, "Merkmale kuerzer als die Stichtage"
    year = np.array([(DAY0 + dt.timedelta(days=int(d))).year for d in ev["L_abs"]])
    blocks = block_ids(ev, year)
    tg = {c: (ev[f"ev_{c}"].astype(bool), ev[f"t_{c}"].astype(int)) for c in ("centre", "patch")}
    tgv = {c: (va[f"ev_{c}"].astype(bool), va[f"t_{c}"].astype(int)) for c in ("centre", "patch")}
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / args.cache_norm / "norm_stats.npz")
    Xt = to_device_once(np.ascontiguousarray(np.load(sd / "eval_X.npy", mmap_mode="r")[:n]), device, "eval_X")
    Xv = to_device_once(np.ascontiguousarray(np.load(sd / "val_X.npy", mmap_mode="r")[:nv]), device, "val_X")
    print(f"[msv2e] Test {n:,}, Val {nv:,}", flush=True)

    res: dict = {"overall": {}, "calibration": {}, "strata": {}, "types": {}, "attribution": {}}
    H_: dict = {}
    files = model_files(args.models)
    attr_models = []

    def states_of(Fx):                                         # Zustand Mitte / Ausschnitt aus den Merkmalen (0 U, 1 O, 2 B)
        f = Fx.float()
        s = torch.where(f[..., 4] > 0.5, 2, torch.where(f[..., 3] > 0.5, 1, 0)).cpu().numpy()
        return {"centre": s[:, (P * P) // 2], "patch": s.max(1)}
    st_test, st_val = states_of(Ft), states_of(Fv)
    for name, fl in list(files.items()):
        T, V = {"centre": [], "patch": []}, {"centre": [], "patch": []}
        for f in fl:
            t0 = time.time()
            net = load_net(f, device)
            if getattr(net, "multistate", False):
                attr_models.append((name, f))
            rt, rv = hazards(net, Xt, ev["doy"], Ft, mean, std, device), hazards(net, Xv, va["doy"], Fv, mean, std, device)
            for j, c in enumerate(("centre", "patch")):
                T[c].append(rt[j]); V[c].append(rv[j])
                a, b = calib_fit(rv[j], *tgv[c])
                m = compact(evaluate(calib_apply(rt[j], a, b), *tg[c]))
                res["overall"].setdefault(name, {}).setdefault(c, {}).setdefault("seed_cal", {})[seed_of(f)] = m
            print(f"    {name} {Path(f).name} in {time.time() - t0:.0f} s; Ausschnitt kalibriert NLL "
                  f"{res['overall'][name]['patch']['seed_cal'][seed_of(f)]['nll']:.4f} "
                  f"C^td {res['overall'][name]['patch']['seed_cal'][seed_of(f)]['c_td']:.3f}", flush=True)
            del net
            torch.cuda.empty_cache()
        for c in ("centre", "patch"):
            a, b = calib_fit(ensemble(V[c]), *tgv[c])
            res["calibration"].setdefault(name, {})[c] = {"a": a, "b": b, "n_seeds": len(fl)}
            h = calib_apply(ensemble(T[c]), a, b).astype(np.float32)
            H_[(name, c)] = h
            e, t = tg[c]
            o = res["overall"][name][c]
            o["ens_cal"] = compact(evaluate(h, e, t))
            fw = ev[f"fire_win_{c}"].astype(bool)
            o["ens_cal_fire"] = compact(evaluate(h[fw], e[fw], t[fw]))
            o["ens_cal_nofire"] = compact(evaluate(h[~fw], e[~fw], t[~fw]))
            o["seed_mean"] = {k: float(np.mean([s[k] for s in o["seed_cal"].values()])) for k in o["ens_cal"]}
            o["seed_std"] = {k: float(np.std([s[k] for s in o["seed_cal"].values()])) for k in o["ens_cal"]}
            print(f"[msv2e] {name:10s} {c:6s} Ensemble kal.: NLL {o['ens_cal']['nll']:.4f} IBS {o['ens_cal']['ibs']:.5f} "
                  f"C^td {o['ens_cal']['c_td']:.3f} PR@7 {o['ens_cal']['pr@7']:.3f} PR@30 {o['ens_cal']['pr@30']:.3f}; "
                  f"Feuer im Fenster C^td {o['ens_cal_fire']['c_td']:.3f}, ohne {o['ens_cal_nofire']['c_td']:.3f}", flush=True)
            if name in args.state_cal:                         # Kontrolle: Kalibrierung je Zustand, ohne Neutraining
                hs, cal_s = state_calibrate(ensemble(V[c]), ensemble(T[c]), st_val[c], st_test[c], tgv[c], (a, b))
                H_[(name + "_scal", c)] = hs.astype(np.float32)
                res["calibration"].setdefault(name + "_scal", {})[c] = cal_s
                os_ = res["overall"].setdefault(name + "_scal", {}).setdefault(c, {})
                os_["ens_cal"] = compact(evaluate(hs, e, t))
                os_["ens_cal_fire"] = compact(evaluate(hs[fw], e[fw], t[fw]))
                os_["ens_cal_nofire"] = compact(evaluate(hs[~fw], e[~fw], t[~fw]))
                print(f"[msv2e] {name + '_scal':10s} {c:6s} je Zustand kal.: NLL {os_['ens_cal']['nll']:.4f} "
                      f"C^td {os_['ens_cal']['c_td']:.3f} PR@30 {os_['ens_cal']['pr@30']:.3f}; " + json.dumps(cal_s), flush=True)
        if name in args.state_cal:
            files[name + "_scal"] = fl

    # ---------------- Strata und Typ-AUC (wie state_eval), gepaart gegen --ref
    P_ = {k: {"nll": nll_per_sample(h, *tg[k[1]]), **cif_at(h)} for k, h in H_.items()}
    models = list(files)
    ref = args.ref if args.ref in models else None
    for c in ("centre", "patch"):
        e, t = tg[c]
        for sname, m in strata(lab, c).items():
            if m.sum() < 50:
                continue
            W = boot_weights(np.unique(blocks[m], return_inverse=True)[1], args.boot, rng)
            em, tm = e[m], t[m]
            y30 = em & (tm <= 30)
            row, bsv = {}, {}
            for mod in models:
                ps = {k: v[m] for k, v in P_[(mod, c)].items()}
                r = {"nll": float(ps["nll"].mean()),
                     "oe@30": float(y30.sum() / max(ps["F30"].sum(), 1e-12))}
                r["auc@30"], r["pr@30"] = roc_pr(y30, ps["F30"])
                if em.any():
                    r["c_td"] = compact(evaluate(H_[(mod, c)][m], em, tm))["c_td"]
                bs = []
                for w in W:
                    a30, p30 = roc_pr(y30, ps["F30"], w)
                    bs.append({"nll": float((w * ps["nll"]).sum() / w.sum()), "auc@30": a30, "pr@30": p30,
                               "oe@30": float((w * y30).sum() / max((w * ps["F30"]).sum(), 1e-12))})
                bsv[mod] = bs
                row[mod] = {k: {"value": v, **(ci([b[k] for b in bs]) if k in bs[0] else {})} for k, v in r.items()}
            if ref:
                for mod in models:
                    if mod != ref:
                        row[mod]["minus_ref"] = {k: {"value": row[mod][k]["value"] - row[ref][k]["value"],
                                                     **ci([bx[k] - br[k] for bx, br in zip(bsv[mod], bsv[ref])])}
                                                 for k in ("nll", "auc@30", "pr@30", "oe@30")}
            res["strata"].setdefault(c, {})[sname] = {"n": int(m.sum()), "n_events": int(em.sum()), "models": row}
            print(f"[msv2e] {c:6s} {sname:7s} n {int(m.sum()):6,}: " + "  ".join(
                f"{mod} NLL {row[mod]['nll']['value']:.4f} Ctd {row[mod].get('c_td', {}).get('value', float('nan')):.3f} "
                f"AUC30 {row[mod]['auc@30']['value']:.3f} OE30 {row[mod]['oe@30']['value']:.2f}" for mod in models), flush=True)
        for gname, (sm, tmask) in type_groups(lab, c).items():
            pos_ = sm & e & (t <= 30) & tmask
            neg_ = sm & ~(e & (t <= 30))
            use = pos_ | neg_
            if pos_.sum() < 10:
                continue
            y = pos_[use]
            W = boot_weights(np.unique(blocks[use], return_inverse=True)[1], args.boot, rng)
            rowg, bsv = {"_n": {"pos": int(pos_.sum()), "neg": int(neg_.sum())}}, {}
            for mod in models:
                s = P_[(mod, c)]["F30"][use]
                b = [roc_pr(y, s, w)[0] for w in W]
                bsv[mod] = np.array(b)
                rowg[mod] = {"auc@30": {"value": roc_pr(y, s)[0], **ci(b)}}
            if ref:
                for mod in models:
                    if mod != ref:
                        rowg[mod]["minus_ref"] = {"value": rowg[mod]["auc@30"]["value"] - rowg[ref]["auc@30"]["value"],
                                                  **ci(bsv[mod] - bsv[ref])}
            res["types"].setdefault(c, {})[gname] = rowg
            print(f"[msv2e] {c:6s} Typ {gname:15s} pos {rowg['_n']['pos']:5,}: " + "  ".join(
                f"{mod} {rowg[mod]['auc@30']['value']:.3f}" for mod in models), flush=True)

    # ---------------- Zuordnung der Ursachen (Multi-State-Modelle)
    if attr_models and args.n_attr > 0:
        from wildfire.data.dataset import CubeReader
        cube = CubeReader(exp)
        pe = dict(np.load(exp / "derived_data" / "region_eval" / "pixel_events.npz"))
        idx = np.sort(rng.choice(n, size=min(args.n_attr, n), replace=False))
        evp = pe["ev"][idx].reshape(len(idx), -1).astype(bool)
        ttp = np.clip(pe["t"][idx].reshape(len(idx), -1).astype(np.int64), 1, H)
        fe = Ft[idx].float().cpu().numpy()
        typ = train_types(raw_keys(cfg), ev["row"][idx], ev["col"][idx], ev["L_abs"][idx].astype(np.int64),
                          evp, ttp, fe, cube.H, cube.W)                        # 1 cont, 2 ret, 3 spr, 4 ign
        rows_ev, pix_ev = np.nonzero(typ > 0)
        k_ev = ttp[rows_ev, pix_ev]
        y = typ[rows_ev, pix_ev]
        order = ("cont", "ret", "spr", "ign")
        m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
        s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]
        Fi = Ft[idx]
        for name, f in attr_models:
            net = load_net(f, device)
            pi = np.zeros((len(rows_ev), 4))
            for a0 in range(0, len(idx), args.bs):
                x = (Xt[idx[a0:a0 + args.bs]].float() - m_t) / s_t
                if msv2.needs_features(net):
                    net.set_features(Fi[a0:a0 + args.bs].float())
                with torch.autocast(device_type=device.type, enabled=False):
                    r = decompose_causes(net, x, torch.from_numpy(ev["doy"][idx[a0:a0 + args.bs]]).to(device))
                sel = (rows_ev >= a0) & (rows_ev < a0 + len(x))
                if sel.any():
                    rr, pp, kk = rows_ev[sel] - a0, pix_ev[sel], k_ev[sel]
                    Fm = {c_: r[c_].double().cpu().numpy() for c_ in order}
                    dF = np.stack([Fm[c_][rr, pp, kk] - Fm[c_][rr, pp, kk - 1] for c_ in order], 1)
                    pi[sel] = dF / np.clip(dF.sum(1, keepdims=True), 1e-300, None)
            mat = {TYPE_NAMES[i]: {order[j]: float(pi[y == i, j].mean()) for j in range(4)}
                   for i in range(1, 5) if (y == i).any()}
            acc = float((pi.argmax(1) + 1 == y).mean())
            res["attribution"].setdefault(name, {})[seed_of(f)] = {
                "matrix": mat, "argmax_accuracy": acc, "n_events": int(len(y)),
                "type_counts": {TYPE_NAMES[i]: int((y == i).sum()) for i in range(1, 5)},
                **({"fuel_gate": float(torch.sigmoid(net.fuel).item())} if hasattr(net, "fuel") else {})}
            print(f"[msv2e] Zuordnung {name} s{seed_of(f)}: Trefferquote argmax {acc:.3f}", flush=True)
            for tn, rowm in mat.items():
                print(f"[msv2e]     wahr {tn:12s}: " + "  ".join(f"pi_{k} {v:.3f}" for k, v in rowm.items()), flush=True)
            del net
            torch.cuda.empty_cache()

    np.savez_compressed(out.with_suffix(".npz"), **{f"{m}__{c}": h.astype(np.float16) for (m, c), h in H_.items()})
    return res


# ------------------------------------------------------------------ paper
def run_paper(args, cfg, exp):
    from wildfire.paper import targets
    from wildfire.paper.compare import DTS
    from wildfire.paper.metrics import paper_metrics
    from wildfire.paper.pixel_stdctm import windows
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = exp / "derived_data" / "paper" / f"{args.cache}_L61"
    X = np.load(d / "X.npy")
    y = targets.labels(d, args.target)
    split = np.load(d / "split.npy")
    doyD = sts_dctm.doy_of(np.load(d / "meta.npy")[:, 0])
    fire_t = targets.fire_map(d, X, args.target)
    LC = X.shape[1]
    L = LC + targets.shift(args.target)
    idx = {s: np.where(split == s)[0] for s in ("val", "test")}
    ok = targets.ok_mask(d)
    if ok is not None:
        idx = {s: v[ok[v]] for s, v in idx.items()}
    lab = {"patch": y, "centre": fire_t[:, HALF, HALF].astype(int)}
    fz = np.load(exp / "derived_data" / "state_features" / f"paper{args.feat_tag}.npy", mmap_mode="r")
    info = json.loads((exp / "derived_data" / "state_features" / f"paper{args.feat_tag}.json").read_text())
    assert list(info["dts"]) == list(DTS), "Merkmale passen nicht zum Paper-Cache"
    if fz.shape[0] < X.shape[0]:                               # Rauchtest-Merkmale: nur die ersten Stichproben
        print(f"[msv2e] Merkmale nur fuer {fz.shape[0]} von {X.shape[0]} Stichproben (Rauchtest)", flush=True)
        idx = {s: v[v < fz.shape[0]] for s, v in idx.items()}
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / targets.base_cache(args.cache) / "norm_stats.npz")
    Xg = to_device_once(X, device)
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]

    @torch.no_grad()
    def scores(net, rows, j, dt_, bs=64):
        e = L - dt_ - 2
        k = torch.tensor([float(dt_ + 1)], device=device)
        sp, sc = [], []
        for a in range(0, rows.size, bs):
            r = rows[a:a + bs]
            x = (windows(Xg, r, np.full(r.size, e)).float() - m_t) / s_t
            if msv2.needs_features(net):
                net.set_features(torch.from_numpy(np.asarray(fz[r, j], np.float32)).to(device))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                raw_px, raw_pa = net(x, torch.from_numpy(doyD[r] - (LC - 1 - e)).to(device))
            sc.append((-torch.expm1(net.px_head.log_survival(raw_px[:, HALF, HALF].float(), k))[:, 0]).float().cpu().numpy())
            sp.append((-torch.expm1(net.pa_head.log_survival(raw_pa.float(), k))[:, 0]).float().cpu().numpy())
        return {"patch": np.concatenate(sp), "centre": np.concatenate(sc)}

    res: dict = {}
    for name, fl in model_files(args.models).items():
        S = {dt_: [] for dt_ in DTS}
        for f in fl:
            net = load_net(f, device)
            for j, dt_ in enumerate(DTS):
                S[dt_].append(scores(net, idx["test"], j, dt_))
            del net
            torch.cuda.empty_cache()
        for dt_ in DTS:
            for c in ("patch", "centre"):
                yt = lab[c][idx["test"]]
                per = [paper_metrics(s[c], yt)["pr_auc"] for s in S[dt_]]
                ens = paper_metrics(np.mean([s[c] for s in S[dt_]], 0), yt)["pr_auc"]
                res.setdefault(name, {}).setdefault(c, {})[str(dt_)] = {
                    "pr_auc_ens": float(ens), "pr_auc_seed_mean": float(np.mean(per)), "pr_auc_seed_std": float(np.std(per)),
                    **({"paper_reported_3d_unet": PAPER_REPORTED[c][dt_]} if c in PAPER_REPORTED else {})}
        print(f"[msv2e] {name:10s} Paper-Protokoll PR-AUC Ensemble Ausschnitt dt {'/'.join(map(str, DTS))}: "
              + "/".join(f"{res[name]['patch'][str(d_)]['pr_auc_ens']:.3f}" for d_ in DTS)
              + "  Mitte " + "/".join(f"{res[name]['centre'][str(d_)]['pr_auc_ens']:.3f}" for d_ in DTS)
              + "  (Paper 3D U-Net Ausschnitt " + "/".join(f"{PAPER_REPORTED['patch'][d_]:.3f}" for d_ in DTS) + ")",
              flush=True)
    return {"paper_protocol": res, "n_test": int(idx["test"].size)}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("surv", "paper"):
        raise SystemExit(__doc__)
    part = sys.argv[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="NAME=GLOB")
    ap.add_argument("--ref", default="lhb10")
    ap.add_argument("--state-cal", nargs="*", default=["lhb10", "lhf"],
                    help="Modelle, die zusaetzlich je Zustand kalibriert werden (<name>_scal)")
    ap.add_argument("--surv-dir", default="surv_eval_test")
    ap.add_argument("--cache-norm", default="cache_G")
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--target", default="next_patch")
    ap.add_argument("--feat-tag", default="")
    ap.add_argument("--label-tag", default="")
    ap.add_argument("--max-test", type=int, default=0)
    ap.add_argument("--max-val", type=int, default=0)
    ap.add_argument("--n-attr", type=int, default=30_000)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(sys.argv[2:])
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    out = exp / "metrics" / "paper_multistate" / f"msv2_{part}{args.tag}.json"
    if out.exists():
        raise SystemExit(f"ABBRUCH: {out} existiert")
    out.parent.mkdir(parents=True, exist_ok=True)
    res = run_surv(args, cfg, exp, out) if part == "surv" else run_paper(args, cfg, exp)
    out.write_text(json.dumps({"args": vars(args), **res}, indent=1, default=float))
    write_manifest(cfg, f"p41_msv2_{part}{args.tag}", args.config, {"output": str(out)})
    print(f"[msv2e] -> {out}", flush=True)


if __name__ == "__main__":
    main()
