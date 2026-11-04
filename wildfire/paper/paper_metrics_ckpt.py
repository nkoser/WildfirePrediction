"""Alle Paper-Metriken im Protokoll des Papers fuer unsere Survival-Modelle und den Nachbau.

Das Paper berichtet je dt Precision, Recall, F1 (Schwelle 0,5), PR-AUC, ECE,
MCE und Brier (metrics.paper_metrics, dazu ROC-AUC). PR-AUC und ROC-AUC haengen
nur an der Reihenfolge; F1, ECE, MCE und Brier an den Wahrscheinlichkeiten und
damit an der Basisrate der Paper-Stichproben.

  sts / pxst  Checkpoints aus dem Wuerfel-Vortraining: Score F(dt + 1) bei
              Stichtag e = Zieltag - dt - 1. Die Modelle sind auf andere
              Haeufigkeiten trainiert; fuer die wahrscheinlichkeitsabhaengigen
              Metriken wird der Score deshalb per Platt-Skalierung auf dem
              Val-Split des Papers umgerechnet (logistische Regression auf logit F).
              Zusaetzlich isoton (monotone Stufenfunktion, ebenfalls auf dem Val-Split):
              prueft, ob eine Kalibrierungsluecke nur an der Form der Platt-Kurve liegt.
              Isoton erzeugt Bindungen; PR-AUC und ROC-AUC deshalb aus raw/platt lesen.
              Ausschnitt: sts = Ausschnittskopf, pxst = Maximum ueber die Pixel.
  rec         Nachbau des Paper-Modells (42_paper_reconstruct), auf den Paper-
              Stichproben trainiert: Vorhersagen wie gespeichert (raw) und genauso
              kalibriert wie unsere Modelle.
  cal         Die faire Hauptvariante: je Modell, dt und Seed Platt oder isoton,
              ausgewaehlt per 5-facher Kreuzvalidierung innerhalb des Val-Splits
              (Brier), fuer alle Modelle nach derselben Regel.
  curve       Eine gemeinsame Abbildung fuer alle dt (wie die Kurven-Kalibrierung in
              surv_eval): Lambda' = e^a * Lambda^b auf F(dt + 1), a und b ueber alle dt
              zusammen auf dem Val-Split. Haelt F ueber die Horizonte monoton.
  Ensemble    Mittel der Scores der Seeds, dann wie oben.
"""

from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.models.surv_heads import BinaryHorizonHead
from wildfire.paper import targets
from wildfire.paper.compare import DTS, KEYS
from wildfire.paper.metrics import ASSUMPTIONS, paper_metrics
from wildfire.paper.pixel_stdctm import PixelSTDCTM, windows
from wildfire.paper.reconstruct import cache_norm, to_device_once
from wildfire.paper.sts_dctm import doy_of, load_sts

CENTRE = 6


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p) - np.log1p(-p)


def platt(s_val, y_val, s_test):
    lr = LogisticRegression(C=1e6).fit(logit(s_val)[:, None], y_val)
    return lr.predict_proba(logit(s_test)[:, None])[:, 1]


def isotonic(s_val, y_val, s_test):
    iso = IsotonicRegression(y_min=1e-6, y_max=1 - 1e-6, out_of_bounds="clip").fit(s_val, y_val)
    return iso.predict(s_test)


CALIBRATORS = {"platt": platt, "iso": isotonic}


def choose_calibration(s_val, y_val, k: int = 5, seed: int = 0) -> tuple[str, dict]:
    """Platt oder isoton per k-facher Kreuzvalidierung innerhalb des Val-Splits (Brier).

    Auf denselben Stichproben anpassen und vergleichen wuerde isoton bevorzugen, weil es
    sich an sie anschmiegt; deshalb je Fold auf k-1 Teilen anpassen, auf dem Rest messen.
    Folds geschichtet nach y. -> (Name, mittlerer Brier je Verfahren)."""
    rng = np.random.default_rng(seed)
    fold = np.empty(y_val.size, int)
    for c in (0, 1):
        i = rng.permutation(np.where(y_val == c)[0])
        fold[i] = np.arange(i.size) % k
    brier = {}
    for name, f in CALIBRATORS.items():
        se = [np.mean((f(s_val[fold != j], y_val[fold != j], s_val[fold == j]) - y_val[fold == j]) ** 2)
              for j in range(k)]
        brier[name] = float(np.mean(se))
    return min(brier, key=brier.get), brier


def calibrate(s_val, y_val, s_test) -> tuple[np.ndarray, str]:
    """Ausgewaehltes Verfahren (choose_calibration) auf dem ganzen Val-Split anpassen, auf Test anwenden."""
    name, _ = choose_calibration(s_val, y_val)
    return CALIBRATORS[name](s_val, y_val, s_test), name


def curve_map(F, a: float, b: float):
    """Gemeinsame Kalibrierung der ganzen Kurve: Lambda' = e^a * Lambda^b mit Lambda = -log(1 - F).

    Dieselbe Regel wie in surv_eval (calib_apply), hier auf F(dt + 1). Eine Abbildung fuer
    alle Horizonte, glatt und streng monoton: F(tau) bleibt in tau steigend."""
    lam = -np.log1p(-np.clip(F, 1e-12, 1 - 1e-12))
    return np.clip(-np.expm1(-np.exp(a) * np.power(np.maximum(lam, 1e-300), b)), 1e-7, 1 - 1e-7)


def curve_fit(vals: list, ys: list) -> tuple[float, float]:
    """a, b fuer curve_map: binaere Log-Likelihood ueber alle dt gemeinsam (Val-Split)."""
    from scipy.optimize import minimize
    F, y = np.concatenate(vals), np.concatenate(ys)

    def nll(p):
        q = curve_map(F, p[0], float(np.exp(p[1])))
        return -np.mean(y * np.log(q) + (1 - y) * np.log1p(-q))
    r = minimize(nll, x0=np.zeros(2), method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-9, "maxiter": 600})
    return float(r.x[0]), float(np.exp(r.x[1]))


def agg(ms: list[dict]) -> dict:
    return {k: {"mean": float(np.mean([m[k] for m in ms])), "std": float(np.std([m[k] for m in ms]))} for k in KEYS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--target", choices=targets.TARGETS, default="next_patch")
    ap.add_argument("--models", nargs="+", default=[
        "sts:/data/Wildfire/checkpoints/paper/sts_cache_G_next_cube_d128_s*_s*.pt",
        "pxst:/data/Wildfire/checkpoints/paper/pxst_cache_G_cubeP_s*_s*.pt"])
    ap.add_argument("--rec-run", default="", help="Lauf von 42_paper_reconstruct; leer = je nach Ziel")
    ap.add_argument("--subset", default="",
                    help="nur die Stichproben mit ok.npy dieses Caches (z. B. cache_G_ea): Originalmodelle auf der "
                         "Teilmenge der flaechentreuen Auswertung")
    ap.add_argument("--tag", default="")
    ap.add_argument("--save-scores", action="store_true",
                    help="Scores je Modell, dt, Split und Seed nach predictions/paper/scores_<cache><tag>.npz (fuer clim_blend)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rec_run = args.rec_run or ("_next_paperhp_w0" if args.target == "next_patch" else "_paperhp")

    d = exp / "derived_data" / "paper" / f"{args.cache}_L61"
    X = np.load(d / "X.npy")
    y = targets.labels(d, args.target)
    split = np.load(d / "split.npy")
    doyD = doy_of(np.load(d / "meta.npy")[:, 0])
    fire_t = targets.fire_map(d, X, args.target)
    LC = X.shape[1]
    L = LC + targets.shift(args.target)
    idx = {s: np.where(split == s)[0] for s in ("val", "test")}
    # Teilmenge: flaechentreuer Cache (ok.npy) bzw. --subset; die Originalmodelle laufen so auf denselben Stichproben
    ok = targets.ok_mask(exp / "derived_data" / "paper" / f"{args.subset}_L61") if args.subset else targets.ok_mask(d)
    if ok is not None:
        idx = {s: v[ok[v]] for s, v in idx.items()}
        print(f"[pm] Teilmenge {args.subset or args.cache} (alle 61 Fenstertage im Wuerfel): "
              f"Val {idx['val'].size}, Test {idx['test'].size}", flush=True)
    lab = {"patch": y, "centre": fire_t[:, CENTRE, CENTRE].astype(int)}
    if (d / "y_next_patch_orig.npy").exists() and args.target == "next_patch":   # flaechentreu: dazu die Paper-Labels
        lab["patchorig"] = np.load(d / "y_next_patch_orig.npy").astype(int)
        lab["centreorig"] = np.load(d / "y_next_centre_orig.npy").astype(int)
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / targets.base_cache(args.cache) / "norm_stats.npz")
    Xg = to_device_once(X, device)
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]

    @torch.no_grad()
    def scores(kind, net, rows, dt_, bs=64):
        """F(dt + 1) fuer Ausschnitt und Mitte, Stichtag e = Zieltag - dt - 1."""
        e = L - dt_ - 2
        k = torch.tensor([float(dt_ + 1)], device=device)
        sp, sc, rr_ = [], [], []
        for a in range(0, rows.size, bs):
            r = rows[a:a + bs]
            x = (windows(Xg, r, np.full(r.size, e)).float() - m_t) / s_t
            if kind.startswith("sts"):
                with_rate = getattr(net, "rate_head", None) is not None
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    outs = net(x, torch.from_numpy(doyD[r] - (LC - 1 - e)).to(device), rate=with_rate)
                raw_px, raw_pa = outs[0], outs[1]
                fc = -torch.expm1(net.px_head.log_survival(raw_px[:, CENTRE, CENTRE], k))[:, 0]
                fp = -torch.expm1(net.pa_head.log_survival(raw_pa, k))[:, 0]
                if with_rate:                                  # Rate-Kopf: P(Feuer an Tag dt + 1)
                    rr_.append(torch.sigmoid(outs[2][:, :, dt_].float()).cpu().numpy())
            else:
                raw = net(x)
                f = -torch.expm1(net.dctm.log_survival(raw.reshape(-1, raw.shape[-1]), k))[:, 0].view(len(r), -1)
                fc, fp = f[:, (13 * 13) // 2], f.max(1).values
            sp.append(fp.float().cpu().numpy()); sc.append(fc.float().cpu().numpy())
        out = {"patch": np.concatenate(sp), "centre": np.concatenate(sc)}
        if rr_:
            rt = np.clip(np.concatenate(rr_), 1e-7, 1 - 1e-7)
            out.update({"patch_rate": rt[:, 0], "centre_rate": rt[:, 1]})
        return out

    res: dict = {}

    def record(model, variant, dt_, tg, per_seed, ens=None):
        r = res.setdefault(model, {}).setdefault(variant, {}).setdefault(str(dt_), {})
        r[tg] = {"summary": agg(per_seed), "per_seed": per_seed}
        if ens is not None:
            r[tg]["ensemble"] = ens
        a = r[tg]["summary"]
        print(f"[pm] {model:5s} {variant:5s} dt={dt_:2d} {tg:6s} " + "  ".join(
            f"{k} {a[k]['mean']:.3f}" for k in ("precision", "recall", "f1", "pr_auc", "ece", "mce", "brier"))
            + (f"   Ensemble PR-AUC {ens['pr_auc']:.3f} F1 {ens['f1']:.3f} ECE {ens['ece']:.3f}" if ens else ""), flush=True)

    def record_all(model, dt_, tg, vals, tests, yv, yt):
        """raw, platt, iso und cal (Auswahl platt/iso per Kreuzvalidierung im Val-Split) je Seed und Ensemble."""
        ens_v, ens_t = np.mean(vals, 0), np.mean(tests, 0)
        record(model, "raw", dt_, tg, [paper_metrics(t, yt) for t in tests], paper_metrics(ens_t, yt))
        for name, f in CALIBRATORS.items():
            record(model, name, dt_, tg, [paper_metrics(f(v, yv, t), yt) for v, t in zip(vals, tests)],
                   paper_metrics(f(ens_v, yv, ens_t), yt))
        cal = [calibrate(v, yv, t) for v, t in zip(vals, tests)]
        cal_e = calibrate(ens_v, yv, ens_t)
        record(model, "cal", dt_, tg, [paper_metrics(p, yt) for p, _ in cal], paper_metrics(cal_e[0], yt))
        res[model]["cal"][str(dt_)][tg]["choice"] = {"per_seed": [n for _, n in cal], "ensemble": cal_e[1]}

    def record_curve(model, tg, V: dict, T: dict, yv, yt):
        """Eine gemeinsame Abbildung je Seed (und fuers Ensemble) fuer alle dt: V[dt], T[dt] = Liste je Seed."""
        dts = [d for d in DTS if d in V]
        n_seed = len(V[dts[0]])
        fits = [curve_fit([V[d][k] for d in dts], [yv] * len(dts)) for k in range(n_seed)]
        fit_e = curve_fit([np.mean(V[d], 0) for d in dts], [yv] * len(dts))
        for d in dts:
            record(model, "curve", d, tg, [paper_metrics(curve_map(T[d][k], *fits[k]), yt) for k in range(n_seed)],
                   paper_metrics(curve_map(np.mean(T[d], 0), *fit_e), yt))
            res[model]["curve"][str(d)][tg]["ab"] = {"per_seed": fits, "ensemble": fit_e}

    # ---- Survival-Checkpoints
    saved: dict = {}
    for spec in args.models:
        kind, pattern = spec.split(":", 1)
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"[pm] {spec}: keine Checkpoints", flush=True)
            continue
        S = {dt_: {s: [] for s in idx} for dt_ in DTS}
        for f in files:
            t0 = time.time()
            if kind.startswith("sts"):
                net = load_sts(f, device)
                for hd in (net.px_head, net.pa_head):       # bce: rohe Ausgaenge an den Horizonten (gestapelt)
                    if isinstance(hd, BinaryHorizonHead):
                        hd.monotone = False
            else:
                c = torch.load(f, map_location="cpu")
                net = PixelSTDCTM(degree=c["degree"], tau_max=c["tau_max"])
                net.load_state_dict(c["state"]); net.to(device).eval()
            for dt_ in DTS:
                for s in idx:
                    S[dt_][s].append(scores(kind, net, idx[s], dt_))
            print(f"    {Path(f).name} in {time.time() - t0:.0f} s", flush=True)
        if args.save_scores:
            for dt_ in DTS:
                for s in idx:
                    for key in S[dt_][s][0]:
                        saved[f"{kind}|{key}|{dt_}|{s}"] = np.stack([sd[key] for sd in S[dt_][s]])
        for dt_ in DTS:
            for key in S[dt_]["test"][0]:                     # patch, centre und ggf. patch_rate, centre_rate
                tg = key.split("_")[0]
                yv, yt = lab[tg][idx["val"]], lab[tg][idx["test"]]
                tests = [sd[key] for sd in S[dt_]["test"]]
                vals = [sd[key] for sd in S[dt_]["val"]]
                record_all(kind + ("_rate" if key.endswith("_rate") else ""), dt_, tg, vals, tests, yv, yt)
                if tg + "orig" in lab:                        # derselbe Score gegen das Label des Papers
                    yv, yt = lab[tg + "orig"][idx["val"]], lab[tg + "orig"][idx["test"]]
                    record_all(kind + ("_rate" if key.endswith("_rate") else ""), dt_, tg + "orig", vals, tests, yv, yt)
        for tg in ("patch", "centre"):
            record_curve(kind, tg, {d: [sd[tg] for sd in S[d]["val"]] for d in DTS},
                         {d: [sd[tg] for sd in S[d]["test"]] for d in DTS}, lab[tg][idx["val"]], lab[tg][idx["test"]])

    # ---- Nachbau des Paper-Modells
    z_path = exp / "predictions" / "paper" / f"rec_{args.cache}_cache{rec_run}.npz"
    if z_path.exists():
        z = np.load(z_path)
        pos = {int(i): n for n, i in enumerate(z["idx_test"])}
        sel = np.array([pos[int(i)] for i in idx["test"]])
        pos_v = {int(i): n for n, i in enumerate(z["idx_val"])}
        sel_v = np.array([pos_v[int(i)] for i in idx["val"]])
        yv, yt = lab["patch"][idx["val"]], lab["patch"][idx["test"]]
        V, T = {}, {}
        for dt_ in DTS:
            ks = sorted(k for k in z.files
                        if k.startswith(f"paper55_T30_end{dt_ + 1 - targets.shift(args.target)}_s") and k.endswith("_test"))
            if ks:
                V[dt_], T[dt_] = [z[k[:-5] + "_val"][sel_v] for k in ks], [z[k][sel] for k in ks]
                record_all("rec", dt_, "patch", V[dt_], T[dt_], yv, yt)
        if V and len({len(v) for v in V.values()}) == 1:     # gleiche Seeds je dt
            record_curve("rec", "patch", V, T, yv, yt)
    else:
        print(f"[pm] {z_path} fehlt", flush=True)

    if args.save_scores:
        out = exp / "predictions" / "paper" / f"scores_{args.cache}{targets.tag(args.target)}{args.tag}.npz"
        np.savez_compressed(out, idx_val=idx["val"], idx_test=idx["test"], **saved)
        print(f"[pm] Scores {out}", flush=True)
    write_manifest(cfg, f"p17_paper_metrics_{args.cache}{targets.tag(args.target)}{args.tag}", args.config, {
        "cache": args.cache, "target": args.target, "models": args.models, "rec_run": rec_run,
        "subset": {"source": args.subset or (args.cache if ok is not None else ""),
                   "n_val": int(idx["val"].size), "n_test": int(idx["test"].size),
                   "rule": "alle 61 Fenstertage im Wuerfel (1.4.-30.9.)" if ok is not None else "alle Stichproben"},
        "orig_targets": "patchorig/centreorig: Score des flaechentreuen Modells gegen die Labels des Papers "
                        "(13 x 13 Originalpixel an D+1)" if "patchorig" in lab else None,
        "score": "F(dt + 1) bei Stichtag e = Zieltag - dt - 1; Ausschnitt sts = Kopf, pxst = Maximum; "
                 "<art>_rate: Rate-Kopf P(Feuer an Tag dt + 1); bce-Kopf: rohe Ausgaenge (nicht monotonisiert)",
        "platt": "logistische Regression auf logit F, angepasst auf dem Val-Split des Papers",
        "iso": "isotone Regression auf F, angepasst auf dem Val-Split des Papers (Bindungen: Rangmetriken aus platt)",
        "cal": "je Modell, dt und Seed platt oder iso, gewaehlt per 5-facher geschichteter Kreuzvalidierung "
               "im Val-Split (Brier), dann auf dem ganzen Val-Split angepasst; gleiche Regel fuer alle Modelle "
               "einschliesslich des Nachbaus (rec)",
        "curve": "eine Abbildung Lambda' = e^a * Lambda^b (Lambda = -log(1 - F)) fuer alle dt gemeinsam, a und b per "
                 "binaerer Log-Likelihood ueber alle dt des Val-Splits; F bleibt ueber die Horizonte monoton",
        "metric_assumptions": ASSUMPTIONS, "results": res,
    })


if __name__ == "__main__":
    main()
