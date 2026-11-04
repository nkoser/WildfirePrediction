"""Horizontkonsistenz und Aufschluesselung nach Testjahr auf den zufaelligen Test-Stichtagen (surv_eval_test).

Teil A, ein Modell je Horizont gegen eine Kurve F(tau):
  stack     die vier getrennt trainierten Einzelklassifikatoren bce2 / bce8 / bce16 / bce31 (gleiches STS-Rueckgrat,
            je Horizont ein eigenes Netz wie in den Arbeiten mit einem Modell je Vorlaufzeit), Seed s mit Seed s
            gestapelt; F(tau_k) sind die rohen Ausgaenge (BinaryHorizonHead.cif_raw)
  bce       der gemeinsame BCE-Kopf mit Ausgaengen an mehreren Horizonten (ein Netz), roh
  lh, cg    STS-LH und Contagion-STS (scratch): F(tau) aus den diskreten Hazards, per Konstruktion monoton
  Gemessen  Anteil der Stichtage mit F(tau_k) < F(tau_{k-1}) fuer irgendein k (streng und mit Marge 0,01), je Paar;
            ROC-AUC und PR-AUC fuer "Feuer bis tau" bei tau = 1 .. 31, auch an nicht trainierten Horizonten.
            Gestapelte Modelle dort linear in F zwischen den Stuetzstellen (0, 0), (tau_k, F_k) interpoliert - dieselbe
            Regel wie BinaryHorizonHead - bzw. "nearest": Ausgang des naechstgelegenen trainierten Horizonts.
            Ensemble = Mittel ueber die Seeds (Kurven: Mittel der Survival-Kurven; Stapel: Mittel der Ausgaenge).

Teil B, Testjahre 2023 / 2024 / 2025 einzeln:
  Modelle   lh, cg, lincg als Seed-Ensemble, nachkalibriert wie surv_eval --calibrate (Lambda' = e^a Lambda^b auf den
            Val-Stichtagen 2020-2022), dazu km / km_fire und die klassischen Basislinien aus classic_surv
            (metrics/paper_classic/surv_eval.npz: NLL je Stichtag und F an den Horizonten)
  Intervalle Block-Bootstrap: Block = (Jahr, Kalenderwoche des Stichtags, 64 x 64-Gitterkachel, etwa 290 km); Bloecke
            mit Zuruecklegen gezogen, als Gewichte an die gewichteten Metriken aus wildfire.eval.metrics; 95 %-Perzentil-
            intervalle, dazu gepaarte Differenzen lh - x auf denselben Ziehungen
  Kontrolle Die Gesamtwerte (alle Jahre) muessen die Manifeste p16_surv_eval_scr3 / _lincg wiedergeben.

Laeuft ueber den Wrapper, damit die lineare Contagion laedt:
  python -m wildfire.paper.contagion_linear wildfire.paper.horizon_landmarks --config ...
Ausgabe: metrics/paper_horizon/landmarks<tag>.json und .npz (Kurven je tau); bricht ab, wenn sie existieren.
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

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.eval.metrics import _roc_pr, antolini_c, cif_from_hazard, evaluate, \
    integrated_brier, negative_log_likelihood
from wildfire.models.surv_heads import BinaryHorizonHead
from wildfire.paper import sts_dctm
from wildfire.paper.reconstruct import cache_norm, to_device_once
from wildfire.paper.surv_eval import calib_apply, calib_fit, compact, ensemble
from wildfire.paper.surv_prepare import HALF
from wildfire.paper.surv_train import km_hazard

H = 100
TAUS = tuple(range(1, 32))
DAY0 = dt.date(2000, 1, 1)
CK = "/data/Wildfire/checkpoints/paper/sts_cache_G_next_cube_abl_"
CURVES = {"lh": CK + "lh_s?_s?.pt", "cg": CK + "cg_scratch_s?_s?.pt", "lincg": CK + "lincg_s?_s?.pt"}
STACK = {2: CK + "bce2_s?_s?.pt", 8: CK + "bce8_s?_s?.pt", 16: CK + "bce16_s?_s?.pt", 31: CK + "bce31_s?_s?.pt"}
JOINT = CK + "bce_s?_s?.pt"
CLASSIC = ("fwi", "fwi_group", "lr", "gbm")


@torch.no_grad()
def run_net(ck, X, doy, mean, std, device, bs=64):
    """-> dict mit hc, hp (n, H) float32 Hazards; bei BCE-Koepfen dazu rc, rp (n, K) rohe F an den Horizonten."""
    net = sts_dctm.load_sts(ck, device)
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]
    bce = isinstance(net.pa_head, BinaryHorizonHead)
    out = {"hc": [], "hp": [], "rc": [], "rp": []}
    for a in range(0, X.shape[0], bs):
        x = (X[a:a + bs].float() - m_t) / s_t
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            raw_px, raw_pa = net(x, torch.from_numpy(doy[a:a + bs]).to(device))
        if bce:
            out["rc"].append(net.px_head.cif_raw(raw_px[:, HALF, HALF].float()).float().cpu().numpy())
            out["rp"].append(net.pa_head.cif_raw(raw_pa.float()).float().cpu().numpy())
        else:
            out["hc"].append(net.px_head.hazards(raw_px[:, HALF, HALF], H).float().cpu().numpy())
            out["hp"].append(net.pa_head.hazards(raw_pa, H).float().cpu().numpy())
    res = {k: np.concatenate(v) for k, v in out.items() if v}
    if bce:
        res["taus"] = tuple(net.pa_head.taus)
    return res


def interp(F_knots: np.ndarray, knots, taus=TAUS) -> np.ndarray:
    """Linear in F zwischen (0, 0) und (tau_k, F_k); hinter dem letzten Knoten konstant. (n, K) -> (n, len(taus))."""
    kx = np.r_[0.0, np.asarray(knots, float)]
    Fk = np.concatenate([np.zeros((F_knots.shape[0], 1), F_knots.dtype), F_knots], 1)
    out = np.empty((F_knots.shape[0], len(taus)), np.float32)
    for j, t in enumerate(taus):
        i = min(int(np.searchsorted(kx, t)), len(kx) - 1)
        if kx[i] == t or i == len(kx) - 1 and t >= kx[-1]:
            out[:, j] = Fk[:, i]
        else:
            w = (t - kx[i - 1]) / (kx[i] - kx[i - 1])
            out[:, j] = (1 - w) * Fk[:, i - 1] + w * Fk[:, i]
    return out


def nearest(F_knots: np.ndarray, knots, taus=TAUS) -> np.ndarray:
    kn = np.asarray(knots)
    return np.stack([F_knots[:, int(np.argmin(np.abs(kn - t)))] for t in taus], 1).astype(np.float32)


def violations(F_knots: np.ndarray) -> dict:
    d = np.diff(F_knots, axis=1)
    return {"any": float((d < 0).any(1).mean()), "any_margin_0.01": float((d < -0.01).any(1).mean()),
            "per_pair": [float((d[:, k] < 0).mean()) for k in range(d.shape[1])],
            "mean_drop_if_violated": float(-d[d < 0].mean()) if (d < 0).any() else 0.0}


def per_tau(F: np.ndarray, e, t) -> dict:
    out = {"roc": [], "pr": []}
    w = np.ones(len(e))
    for j, tau in enumerate(TAUS):
        y = (e & (t <= tau)).astype(float)
        auc, ap, _ = _roc_pr(y, F[:, j].astype(np.float64), w)
        out["roc"].append(auc); out["pr"].append(ap)
    return out


def block_ids(meta, year) -> np.ndarray:
    week = (meta["L_abs"].astype(np.int64) // 7)
    tile = (meta["row"].astype(np.int64) // 64) * 1000 + meta["col"].astype(np.int64) // 64
    key = (year.astype(np.int64) * 10_000 + week) * 1_000_000 + tile
    return np.unique(key, return_inverse=True)[1]


def boot_weights(blocks: np.ndarray, B: int, rng) -> np.ndarray:
    nb = int(blocks.max()) + 1
    cnt = np.stack([np.bincount(rng.integers(0, nb, nb), minlength=nb) for _ in range(B)])   # (B, nb)
    return cnt[:, blocks].astype(np.float64)                                                  # (B, n)


def per_sample(h, F, e, t) -> dict:
    """NLL- und IBS-Beitrag je Stichtag; beide Metriken sind gewichtete Mittel dieser Beitraege (wie in
    wildfire.eval.metrics), der Bootstrap braucht dann nur noch Gewichte."""
    hb = np.clip(np.asarray(h, np.float64), 1e-15, 1 - 1e-7)          # wie negative_log_likelihood (EPS)
    csum = np.cumsum(np.log1p(-hb), axis=1)
    idx = np.arange(len(e))
    prior = np.where(t > 1, csum[idx, np.maximum(t - 2, 0)], 0.0)
    nll = -np.where(e, np.log(hb[idx, t - 1]) + prior, csum[:, -1])
    y = e[:, None] & (t[:, None] <= np.arange(1, h.shape[1] + 1)[None])
    ibs = ((F.astype(np.float64) - y) ** 2).mean(1)
    return {"nll": nll, "ibs": ibs, "F7": F[:, 6], "F30": F[:, 29]}


def metr(ps: dict, e, t, w=None) -> dict:
    w = np.ones(len(e)) if w is None else w
    out = {k: float((w * ps[k]).sum() / w.sum()) for k in ("nll", "ibs") if k in ps}
    for tau in (7, 30):
        out[f"pr@{tau}"] = _roc_pr((e & (t <= tau)).astype(float), ps[f"F{tau}"].astype(np.float64), w)[1]
    return out


def ci(vals) -> dict:
    v = np.asarray(vals, float)
    return {"lo": float(np.nanpercentile(v, 2.5)), "hi": float(np.nanpercentile(v, 97.5))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--surv-dir", default="surv_eval_test")
    ap.add_argument("--cache-norm", default="cache_G")
    ap.add_argument("--max-test", type=int, default=0, help="Rauchtest: zufaellige Teilmenge der Stichtage")
    ap.add_argument("--max-val", type=int, default=0)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip", nargs="*", default=[], help="Teile auslassen: stack joint lincg classic")
    ap.add_argument("--curves", nargs="+", default=None,
                    help="name=glob statt der Standard-Kurvenmodelle (lh, cg, lincg); gepaarte Differenzen gegen den Namen \"lh\"")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    out_dir = exp / "metrics" / "paper_horizon"
    out_dir.mkdir(parents=True, exist_ok=True)
    outs = [out_dir / f"landmarks{args.tag}.json", out_dir / f"landmarks{args.tag}.npz"]
    for p in outs:
        if p.exists():
            raise SystemExit(f"ABBRUCH: {p} existiert schon - anderes --tag waehlen")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    sd = exp / "derived_data" / args.surv_dir
    ev = dict(np.load(sd / "eval_meta.npz"))
    va = dict(np.load(sd / "val_meta.npz"))
    km = dict(np.load(sd / "km_meta.npz"))
    n_all = ev["row"].size
    te = np.sort(rng.choice(n_all, args.max_test, replace=False)) if 0 < args.max_test < n_all else np.arange(n_all)
    vi = (np.sort(rng.choice(va["row"].size, args.max_val, replace=False))
          if 0 < args.max_val < va["row"].size else np.arange(va["row"].size))
    evs = {k: v[te] for k, v in ev.items()}
    vas = {k: v[vi] for k, v in va.items()}
    year = np.array([(DAY0 + dt.timedelta(days=int(d))).year for d in evs["L_abs"]])
    tg = {"centre": (evs["ev_centre"].astype(bool), evs["t_centre"].astype(int)),
          "patch": (evs["ev_patch"].astype(bool), evs["t_patch"].astype(int))}
    tgv = {"centre": (vas["ev_centre"].astype(bool), vas["t_centre"].astype(int)),
           "patch": (vas["ev_patch"].astype(bool), vas["t_patch"].astype(int))}
    print(f"[hz] Test {te.size:,} Stichtage, Val {vi.size:,}; Jahre " +
          ", ".join(f"{y} {int((year == y).sum()):,}" for y in np.unique(year)), flush=True)

    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / args.cache_norm / "norm_stats.npz")
    Xall = np.load(sd / "eval_X.npy", mmap_mode="r")
    Xt = to_device_once(np.ascontiguousarray(Xall[te]) if te.size < n_all else np.load(sd / "eval_X.npy"), device, "eval_X")
    Vall = np.load(sd / "val_X.npy", mmap_mode="r")
    Xv = to_device_once(np.ascontiguousarray(Vall[vi]), device, "val_X")
    res: dict = {"horizon": {}, "by_year": {}, "overall_check": {}, "calibration": {}}
    store: dict = {"taus": np.array(TAUS), "idx": te}

    # ---------------- Kurvenmodelle (Test + Val, Seed-Ensemble, nachkalibriert)
    ens_h: dict = {}
    curves = dict(c.split("=", 1) for c in args.curves) if args.curves else CURVES
    for name, pattern in curves.items():
        if name in args.skip:
            continue
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"[hz] {name}: keine Checkpoints ({pattern})", flush=True)
            continue
        T, V = {"centre": [], "patch": []}, {"centre": [], "patch": []}
        for f in files:
            t0 = time.time()
            rt, rv = run_net(f, Xt, evs["doy"], mean, std, device), run_net(f, Xv, vas["doy"], mean, std, device)
            T["centre"].append(rt["hc"]); T["patch"].append(rt["hp"])
            V["centre"].append(rv["hc"]); V["patch"].append(rv["hp"])
            print(f"    {Path(f).name} in {time.time() - t0:.0f} s", flush=True)
        for target in ("centre", "patch"):
            hv, ht = ensemble(V[target]), ensemble(T[target])
            a, b = calib_fit(hv, *tgv[target])
            res["calibration"].setdefault(name, {})[target] = {"a": a, "b": b, "n_seeds": len(files)}
            hc = calib_apply(ht, a, b).astype(np.float32)
            ens_h[(name, target)] = hc
            res["overall_check"].setdefault(name, {})[target] = compact(evaluate(hc, *tg[target]))
            o = res["overall_check"][name][target]
            print(f"[hz] {name:6s} {target:6s} Ensemble kalibriert (alle Jahre): NLL {o['nll']:.4f} IBS {o['ibs']:.5f} "
                  f"Ctd {o['c_td']:.3f} PR@7 {o['pr@7']:.3f} PR@30 {o['pr@30']:.3f}", flush=True)
            # Horizonte: rohes Ensemble (Rangmetriken; Kalibrierung mit b > 0 aendert die Rangfolge je tau nicht)
            Fr = cif_from_hazard(ht.astype(np.float64))[:, [t - 1 for t in TAUS]].astype(np.float32)
            res["horizon"].setdefault(target, {})[f"{name}_ens"] = per_tau(Fr, *tg[target])
            for s, hs in enumerate(T[target]):
                Fs = cif_from_hazard(hs.astype(np.float64))[:, [t - 1 for t in TAUS]].astype(np.float32)
                res["horizon"][target][f"{name}_s{s}"] = per_tau(Fs, *tg[target])
            store[f"{name}_{target}_F"] = Fr.astype(np.float16)

    # ---------------- Stapel getrennter Einzelklassifikatoren
    if "stack" not in args.skip:
        knots = sorted(STACK)
        per = {k: sorted(glob.glob(p)) for k, p in STACK.items()}
        n_seed = min(len(v) for v in per.values())
        if n_seed == 0:
            print("[hz] Stapel: Checkpoints fehlen", flush=True)
        else:
            R = {"centre": [], "patch": []}
            for s in range(n_seed):
                cols = {"centre": [], "patch": []}
                for k in knots:
                    t0 = time.time()
                    r = run_net(per[k][s], Xt, evs["doy"], mean, std, device)
                    assert r["taus"] == (k,), f"{per[k][s]}: Horizonte {r['taus']} statt ({k},)"
                    cols["centre"].append(r["rc"][:, 0]); cols["patch"].append(r["rp"][:, 0])
                    print(f"    {Path(per[k][s]).name} in {time.time() - t0:.0f} s", flush=True)
                for target in ("centre", "patch"):
                    R[target].append(np.stack(cols[target], 1))
            for target in ("centre", "patch"):
                e, t = tg[target]
                hz = res["horizon"].setdefault(target, {})
                vio = [violations(Fk) for Fk in R[target]]
                res.setdefault("violations", {}).setdefault(target, {})["stack"] = {
                    "knots": knots, "per_seed": vio,
                    "any_mean": float(np.mean([v["any"] for v in vio])), "any_std": float(np.std([v["any"] for v in vio]))}
                for s, Fk in enumerate(R[target]):
                    hz[f"stack_interp_s{s}"] = per_tau(interp(Fk, knots), e, t)
                    hz[f"stack_nearest_s{s}"] = per_tau(nearest(Fk, knots), e, t)
                Fm = np.mean(R[target], 0)
                hz["stack_interp_ens"] = per_tau(interp(Fm, knots), e, t)
                hz["stack_nearest_ens"] = per_tau(nearest(Fm, knots), e, t)
                res["violations"][target]["stack_ens"] = violations(Fm)
                store[f"stack_{target}_knots"] = Fm.astype(np.float16)
                print(f"[hz] Stapel {target}: nicht monoton {np.mean([v['any'] for v in vio]):.4f} der Stichtage "
                      "(je Seed " + ", ".join(f"{v['any']:.4f}" for v in vio) + f"), Ensemble {violations(Fm)['any']:.4f}",
                      flush=True)

    # ---------------- gemeinsamer BCE-Kopf (ein Netz, mehrere Horizonte)
    if "joint" not in args.skip:
        files = sorted(glob.glob(JOINT))
        R = {"centre": [], "patch": []}
        kn = None
        for f in files:
            r = run_net(f, Xt, evs["doy"], mean, std, device)
            kn = list(r["taus"])
            R["centre"].append(r["rc"]); R["patch"].append(r["rp"])
        if files:
            for target in ("centre", "patch"):
                e, t = tg[target]
                vio = [violations(Fk) for Fk in R[target]]
                res.setdefault("violations", {}).setdefault(target, {})["joint_bce"] = {
                    "knots": kn, "per_seed": vio, "any_mean": float(np.mean([v["any"] for v in vio]))}
                for s, Fk in enumerate(R[target]):
                    res["horizon"].setdefault(target, {})[f"joint_bce_interp_s{s}"] = per_tau(interp(Fk, kn), e, t)
                res["horizon"][target]["joint_bce_interp_ens"] = per_tau(interp(np.mean(R[target], 0), kn), e, t)

    # ---------------- Testjahre mit Block-Bootstrap
    blocks_all = block_ids(evs, year)
    cls = None
    cp = exp / "metrics" / "paper_classic" / "surv_eval.npz"
    if "classic" not in args.skip and cp.exists():
        z = np.load(cp)
        pos = {int(i): k for k, i in enumerate(z["idx"])}
        if all(int(i) in pos for i in te):
            sel = np.array([pos[int(i)] for i in te])
            cls = {m: {target: {f: z[f"{m}__{target}__{f}"][sel] for f in ("nll", "F7", "F30")}
                       for target in ("centre", "patch")} for m in CLASSIC if f"{m}__patch__nll" in z.files}
        else:
            print("[hz] classic_surv: Stichtage passen nicht, Basislinien entfallen", flush=True)
    for target in ("centre", "patch"):
        e_all, t_all = tg[target]
        kmh = km_hazard(km["ev_" + target], km["t_" + target])
        fk = km["fire_win_" + target].astype(bool)
        kmf = {f: km_hazard(km["ev_" + target][fk == f], km["t_" + target][fk == f]) for f in (False, True)}
        for y in np.unique(year):
            m = year == y
            e, t = e_all[m], t_all[m]
            W = boot_weights(np.unique(blocks_all[m], return_inverse=True)[1], args.boot, rng)
            models = {k[0]: v[m] for k, v in ens_h.items() if k[1] == target}
            models["km"] = np.tile(kmh, (int(m.sum()), 1)).astype(np.float32)
            models["km_fire"] = np.stack([kmf[bool(f)] for f in evs["fire_win_" + target][m]]).astype(np.float32)
            pt, bs = {}, {}
            for k, h in models.items():
                F = cif_from_hazard(h.astype(np.float64)).astype(np.float32)
                ps = per_sample(h, F, e, t)
                pt[k] = {**metr(ps, e, t), "c_td": antolini_c(F, e, t)}
                ref = (negative_log_likelihood(h, e, t), integrated_brier(h, e, t, None, F))
                assert abs(ref[0] - pt[k]["nll"]) < 1e-6 and abs(ref[1] - pt[k]["ibs"]) < 1e-6, (k, ref, pt[k])
                bs[k] = [metr(ps, e, t, w) for w in W]
            for k in (cls or {}):
                ps = {f: cls[k][target][f][m] for f in ("nll", "F7", "F30")}
                pt[k] = metr(ps, e, t)
                bs[k] = [metr(ps, e, t, w) for w in W]
            yr = res["by_year"].setdefault(target, {}).setdefault(str(int(y)), {
                "n": int(m.sum()), "n_events": int(e.sum()), "n_blocks": int(np.unique(blocks_all[m]).size), "models": {}})
            for k in pt:
                yr["models"][k] = {mk: {"value": float(v), **(ci([b[mk] for b in bs[k]]) if mk in bs[k][0] else {})}
                                   for mk, v in pt[k].items()}
            if "lh" in bs:
                yr["paired_lh_minus"] = {k: {mk: {"value": float(pt["lh"][mk] - pt[k][mk]),
                                                  **ci([bl[mk] - bk[mk] for bl, bk in zip(bs["lh"], bs[k])])}
                                             for mk in bs[k][0]} for k in bs if k != "lh"}
            print(f"[hz] {target} {int(y)} (n {int(m.sum()):,}, Ereignisse {int(e.sum()):,}): " + "  ".join(
                f"{k} NLL {pt[k]['nll']:.4f} PR@30 {pt[k]['pr@30']:.3f}" for k in pt), flush=True)

    for target in res["horizon"]:
        row = res["horizon"][target]
        print(f"[hz] {target} PR-AUC je tau (1, 3, 7, 10, 14, 20, 30): " + " | ".join(
            f"{k} " + "/".join(f"{row[k]['pr'][t - 1]:.3f}" for t in (1, 3, 7, 10, 14, 20, 30))
            for k in row if k.endswith("_ens")), flush=True)
    np.savez_compressed(outs[1], **store)
    meta = {"args": vars(args), "n_test": int(te.size), "n_val": int(vi.size), "curves": curves, "stack": STACK,
            "joint": JOINT, "taus": list(TAUS),
            "block": "(Jahr, Woche L_abs // 7, Gitterkachel 64 x 64 px); Bloecke mit Zuruecklegen, Gewichte = Anzahl",
            "interp": "linear in F zwischen (0, 0), (tau_k, F_k); hinter dem letzten Knoten konstant", **res}
    outs[0].write_text(json.dumps(meta, indent=1, default=float))
    write_manifest(cfg, f"p30_horizon_landmarks{args.tag}", args.config, {"outputs": [str(p) for p in outs],
                                                                           "args": vars(args)})
    print(f"[hz] -> {outs[0]}", flush=True)


if __name__ == "__main__":
    main()
