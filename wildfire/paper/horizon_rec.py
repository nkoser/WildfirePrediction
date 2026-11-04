"""Paper-Protokoll (MPCFire, Cache G, Ziel next_patch) an allen Vorlaufzeiten dt = 1 .. 30 statt nur an 1/7/15/30.

Das Paper trainiert je dt ein eigenes 3D-U-Net. Fuer eine Vorlaufzeit ohne eigenes Modell bleibt nur, eines der
vorhandenen Modelle zu nehmen oder zwischen zweien zu interpolieren. Hier:
  rec       3D-U-Nets wie 42_paper_reconstruct _next_paperhp_w0 (paper55, T 30, bs 512, lr 1e-4, bis 150 Epochen,
            Geduld 15, Auswahl nach Val-BCE), je trainiertem dt und Seed neu trainiert, diesmal mit gespeichertem
            Checkpoint (checkpoints/paper/horizon_rec<tag>/). Jedes Modell wird an JEDEM dt angewandt (Fenster endet
            dt + 1 Tage vor dem Zieltag) und liefert dort seine Wahrscheinlichkeit fuer "Feuer dt_m + 1 Tage spaeter"
              native   nur an seinem trainierten dt
              nearest  Modell mit dem naechstgelegenen trainierten dt (bei Gleichstand das kuerzere)
              interp   linear in p zwischen den beiden benachbarten Modellen, ausserhalb das naechste
  sts       STS-LH und Contagion-STS: F(dt + 1) aus EINEM Modell an jedem dt, wie paper_metrics_ckpt
PR-AUC ist rangbasiert, Platt aendert sie nicht; berichtet werden roh je Seed und als Ensemble (Mittel der Seeds).
Kontrolle: rec native an 1/7/15/30 muss etwa p17 (rec .669/.589/.554/.489, Seed-Mittel) wiedergeben.

  python -m wildfire.paper.horizon_rec --config ... [--epochs 2 --seeds 0 --tag _smoke]
Ausgabe metrics/paper_horizon/rec<tag>.{json,npz}; bricht ab, wenn sie existieren.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.paper import sts_dctm, targets
from wildfire.paper.channels import CHANNELS
from wildfire.paper.metrics import paper_metrics
from wildfire.paper.pixel_stdctm import windows
from wildfire.paper.reconstruct import UNet3D, cache_norm, fit, predict, to_device_once

TRAIN_DTS = (1, 7, 15, 30)
EVAL_DTS = tuple(range(1, 31))
CK = "/data/Wildfire/checkpoints/paper/sts_cache_G_next_cube_abl_"
STS = {"lh": CK + "lh_s?_s?.pt", "cg": CK + "cg_scratch_s?_s?.pt"}


def nearest_dt(dt_: int, trained) -> int:
    return min(trained, key=lambda m: (abs(m - dt_), m))


def interp_p(P: dict, dt_: int, trained) -> np.ndarray:
    """P[m] = Vorhersage des dt_m-Modells am Fenster von dt_; linear zwischen den Nachbarn."""
    lo = [m for m in trained if m <= dt_]
    hi = [m for m in trained if m >= dt_]
    if not lo or not hi:
        return P[nearest_dt(dt_, trained)]
    a, b = max(lo), min(hi)
    if a == b:
        return P[a]
    w = (dt_ - a) / (b - a)
    return (1 - w) * P[a] + w * P[b]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--load", action="store_true", help="gespeicherte U-Net-Checkpoints dieses --tag laden statt trainieren")
    ap.add_argument("--skip-sts", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = exp / "metrics" / "paper_horizon"
    out_dir.mkdir(parents=True, exist_ok=True)
    outs = [out_dir / f"rec{args.tag}.json", out_dir / f"rec{args.tag}.npz"]
    for p in outs:
        if p.exists():
            raise SystemExit(f"ABBRUCH: {p} existiert schon - anderes --tag waehlen")
    ck_dir = exp / "checkpoints" / "paper" / f"horizon_rec{args.tag}"
    if not args.load and ck_dir.exists() and any(ck_dir.iterdir()):
        raise SystemExit(f"ABBRUCH: {ck_dir} enthaelt schon Checkpoints - --load oder anderes --tag")
    ck_dir.mkdir(parents=True, exist_ok=True)

    target = "next_patch"
    d = exp / "derived_data" / "paper" / f"{args.cache}_L61"
    X = np.load(d / "X.npy")                                            # (N, 61, 55, 13, 13)
    y_train = np.load(d / f"y_{target}.npy").astype(int)                # wie reconstruct
    y = targets.labels(d, target)                                       # wie paper_metrics_ckpt
    assert (y_train == y).all()
    split = np.load(d / "split.npy")
    idx = {s: np.where(split == s)[0] for s in ("train", "val", "test")}
    ok = targets.ok_mask(d)
    ev_idx = {s: (idx[s][ok[idx[s]]] if ok is not None else idx[s]) for s in ("val", "test")}
    LC = X.shape[1]
    doyD = sts_dctm.doy_of(np.load(d / "meta.npy")[:, 0])
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / targets.base_cache(args.cache) / "norm_stats.npz")
    Xg = to_device_once(X, device)
    chs = CHANNELS["paper55"]
    yt = y[ev_idx["test"]]
    print(f"[hrec] {args.cache}: X {X.shape}, Test {yt.size} (positiv {yt.mean():.3f})", flush=True)

    # ---------------- 3D-U-Nets je trainiertem dt; jedes an jedem dt angewandt
    P = {s: {m: {} for m in TRAIN_DTS} for s in args.seeds}             # P[seed][m][dt] -> (n_test,)
    hist = {}
    for s in args.seeds:
        for m in TRAIN_DTS:
            t1 = LC - m
            path = ck_dir / f"rec_{args.cache}_dt{m}_s{s}.pt"
            t0_ = time.time()
            if args.load:
                net = UNet3D(len(chs)).to(device)
                net.load_state_dict(torch.load(path, map_location=device)["state"])
            else:
                net, h = fit(Xg, y_train, idx, chs, t1 - 30, t1, mean, std, device, s, epochs=args.epochs,
                             patience=args.patience, bs=args.bs, lr=args.lr, select="val_bce")
                torch.save({"state": net.state_dict(), "dt": m, "seed": s, "end": m, "T": 30, "history": h}, path)
                hist[f"dt{m}_s{s}"] = h
            for dt_ in EVAL_DTS:
                e1 = LC - dt_
                P[s][m][dt_] = predict(net, Xg, ev_idx["test"], chs, e1 - 30, e1, mean, std, device).astype(np.float32)
            nat = paper_metrics(P[s][m][m], yt)["pr_auc"]
            print(f"[hrec] U-Net dt {m:2d} Seed {s}: native PR-AUC {nat:.3f} ({time.time() - t0_:.0f} s)", flush=True)
            del net
            torch.cuda.empty_cache()

    res: dict = {}

    def rec(name, dt_, per_seed):
        r = res.setdefault(name, {})
        ps = [paper_metrics(p, yt)["pr_auc"] for p in per_seed]
        r[str(dt_)] = {"per_seed": ps, "mean": float(np.mean(ps)), "std": float(np.std(ps)),
                       "ensemble": paper_metrics(np.mean(per_seed, 0), yt)["pr_auc"]}

    for dt_ in EVAL_DTS:
        if dt_ in TRAIN_DTS:
            rec("rec_native", dt_, [P[s][dt_][dt_] for s in args.seeds])
        rec("rec_nearest", dt_, [P[s][nearest_dt(dt_, TRAIN_DTS)][dt_] for s in args.seeds])
        rec("rec_interp", dt_, [interp_p({m: P[s][m][dt_] for m in TRAIN_DTS}, dt_, TRAIN_DTS) for s in args.seeds])
        for m in TRAIN_DTS:                                             # jedes Einzelmodell ueber alle dt
            rec(f"rec_dt{m}", dt_, [P[s][m][dt_] for s in args.seeds])

    # ---------------- STS: F(dt + 1) aus einem Modell
    store = {"idx_test": ev_idx["test"], "y_test": yt, "dts": np.array(EVAL_DTS)}
    if not args.skip_sts:
        m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
        s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]
        L = LC + targets.shift(target)
        rows = ev_idx["test"]
        for name, pattern in STS.items():
            files = sorted(glob.glob(pattern))
            S = {dt_: [] for dt_ in EVAL_DTS}
            for f in files:
                net = sts_dctm.load_sts(f, device)
                with torch.no_grad():
                    for dt_ in EVAL_DTS:
                        e = L - dt_ - 2
                        k = torch.tensor([float(dt_ + 1)], device=device)
                        sp = []
                        for a in range(0, rows.size, 64):
                            r = rows[a:a + 64]
                            x = (windows(Xg, r, np.full(r.size, e)).float() - m_t) / s_t
                            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                                outs_ = net(x, torch.from_numpy(doyD[r] - (LC - 1 - e)).to(device))
                            sp.append((-torch.expm1(net.pa_head.log_survival(outs_[1], k))[:, 0]).float().cpu().numpy())
                        S[dt_].append(np.concatenate(sp))
                print(f"[hrec] {Path(f).name}: dt 1/7/15/30 PR-AUC " + "/".join(
                    f"{paper_metrics(S[dd][-1], yt)['pr_auc']:.3f}" for dd in TRAIN_DTS), flush=True)
            if files:
                for dt_ in EVAL_DTS:
                    rec(name, dt_, S[dt_])
                store[f"{name}_ens"] = np.stack([np.mean(S[dt_], 0) for dt_ in EVAL_DTS], 1).astype(np.float32)

    for s in args.seeds:
        for m in TRAIN_DTS:
            store[f"rec_dt{m}_s{s}"] = np.stack([P[s][m][dt_] for dt_ in EVAL_DTS], 1)
    show = (1, 3, 5, 7, 10, 12, 15, 20, 25, 30)
    for name, r in res.items():
        print(f"[hrec] {name:12s} Ensemble PR-AUC dt " + " ".join(
            f"{dd}:{r[str(dd)]['ensemble']:.3f}" for dd in show if str(dd) in r), flush=True)
    np.savez_compressed(outs[1], **store)
    meta = {"args": vars(args), "cache": args.cache, "target": target, "n_test": int(yt.size),
            "train_dts": list(TRAIN_DTS), "eval_dts": list(EVAL_DTS), "sts": STS, "checkpoints": str(ck_dir),
            "rule": "Zieltag fest, Fenster endet dt + 1 Tage davor (wie das Paper); rec_dt<m> = dt_m-Modell an diesem "
                    "Fenster; interp linear in p zwischen den benachbarten trainierten dt",
            "results": res, "history": hist}
    outs[0].write_text(json.dumps(meta, indent=1, default=float))
    write_manifest(cfg, f"p30_horizon_rec{args.tag}", args.config, {"outputs": [str(p) for p in outs], "args": vars(args)})
    print(f"[hrec] -> {outs[0]}", flush=True)


if __name__ == "__main__":
    main()
