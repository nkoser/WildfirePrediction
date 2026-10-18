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
              Ausschnitt: sts = Ausschnittskopf, pxst = Maximum ueber die Pixel.
  rec         Nachbau des Paper-Modells (42_paper_reconstruct), auf den Paper-
              Stichproben trainiert: Vorhersagen wie gespeichert.
  Ensemble    Mittel der Scores der Seeds, dann wie oben.
"""

from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.paper import targets
from wildfire.paper.compare import DTS, KEYS
from wildfire.paper.metrics import ASSUMPTIONS, paper_metrics
from wildfire.paper.pixel_stdctm import PixelSTDCTM, windows
from wildfire.paper.reconstruct import cache_norm, to_device_once
from wildfire.paper.sts_dctm import STSDCTM, doy_of

CENTRE = 6


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p) - np.log1p(-p)


def platt(s_val, y_val, s_test):
    lr = LogisticRegression(C=1e6).fit(logit(s_val)[:, None], y_val)
    return lr.predict_proba(logit(s_test)[:, None])[:, 1]


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
    ap.add_argument("--tag", default="")
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
    lab = {"patch": y, "centre": fire_t[:, CENTRE, CENTRE].astype(int)}
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / args.cache / "norm_stats.npz")
    Xg = to_device_once(X, device)
    m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
    s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]

    @torch.no_grad()
    def scores(kind, net, rows, dt_, bs=64):
        """F(dt + 1) fuer Ausschnitt und Mitte, Stichtag e = Zieltag - dt - 1."""
        e = L - dt_ - 2
        k = torch.tensor([float(dt_ + 1)], device=device)
        sp, sc = [], []
        for a in range(0, rows.size, bs):
            r = rows[a:a + bs]
            x = (windows(Xg, r, np.full(r.size, e)).float() - m_t) / s_t
            if kind == "sts":
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    raw_px, raw_pa = net(x, torch.from_numpy(doyD[r] - (LC - 1 - e)).to(device))
                fc = -torch.expm1(net.px_head.log_survival(raw_px[:, CENTRE, CENTRE], k))[:, 0]
                fp = -torch.expm1(net.pa_head.log_survival(raw_pa, k))[:, 0]
            else:
                raw = net(x)
                f = -torch.expm1(net.dctm.log_survival(raw.reshape(-1, raw.shape[-1]), k))[:, 0].view(len(r), -1)
                fc, fp = f[:, (13 * 13) // 2], f.max(1).values
            sp.append(fp.float().cpu().numpy()); sc.append(fc.float().cpu().numpy())
        return {"patch": np.concatenate(sp), "centre": np.concatenate(sc)}

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

    # ---- Survival-Checkpoints
    for spec in args.models:
        kind, pattern = spec.split(":", 1)
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"[pm] {spec}: keine Checkpoints", flush=True)
            continue
        S = {dt_: {s: [] for s in idx} for dt_ in DTS}
        for f in files:
            t0 = time.time()
            c = torch.load(f, map_location="cpu")
            if kind == "sts":
                a_ = c.get("args", {})
                net = STSDCTM(d=a_.get("d", 96), heads=a_.get("heads", 4), depth=a_.get("depth", 4))
            else:
                net = PixelSTDCTM(degree=c["degree"], tau_max=c["tau_max"])
            net.load_state_dict(c["state"]); net.to(device).eval()
            for dt_ in DTS:
                for s in idx:
                    S[dt_][s].append(scores(kind, net, idx[s], dt_))
            print(f"    {Path(f).name} in {time.time() - t0:.0f} s", flush=True)
        for dt_ in DTS:
            for tg in ("patch", "centre"):
                yv, yt = lab[tg][idx["val"]], lab[tg][idx["test"]]
                tests = [sd[tg] for sd in S[dt_]["test"]]
                vals = [sd[tg] for sd in S[dt_]["val"]]
                ens_v, ens_t = np.mean(vals, 0), np.mean(tests, 0)
                record(kind, "raw", dt_, tg, [paper_metrics(t, yt) for t in tests], paper_metrics(ens_t, yt))
                record(kind, "platt", dt_, tg, [paper_metrics(platt(v, yv, t), yt) for v, t in zip(vals, tests)],
                       paper_metrics(platt(ens_v, yv, ens_t), yt))

    # ---- Nachbau des Paper-Modells
    z_path = exp / "predictions" / "paper" / f"rec_{args.cache}_cache{rec_run}.npz"
    if z_path.exists():
        z = np.load(z_path)
        pos = {int(i): n for n, i in enumerate(z["idx_test"])}
        sel = np.array([pos[int(i)] for i in idx["test"]])
        for dt_ in DTS:
            ks = sorted(k for k in z.files
                        if k.startswith(f"paper55_T30_end{dt_ + 1 - targets.shift(args.target)}_s") and k.endswith("_test"))
            if ks:
                yt = lab["patch"][idx["test"]]
                record("rec", "raw", dt_, "patch", [paper_metrics(z[k][sel], yt) for k in ks],
                       paper_metrics(np.mean([z[k][sel] for k in ks], 0), yt))
    else:
        print(f"[pm] {z_path} fehlt", flush=True)

    write_manifest(cfg, f"p17_paper_metrics_{args.cache}{targets.tag(args.target)}{args.tag}", args.config, {
        "cache": args.cache, "target": args.target, "models": args.models, "rec_run": rec_run,
        "score": "F(dt + 1) bei Stichtag e = Zieltag - dt - 1; Ausschnitt sts = Kopf, pxst = Maximum",
        "platt": "logistische Regression auf logit F, angepasst auf dem Val-Split des Papers",
        "metric_assumptions": ASSUMPTIONS, "results": res,
    })


if __name__ == "__main__":
    main()
