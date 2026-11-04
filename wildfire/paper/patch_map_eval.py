"""Paper-Protokoll als Karte: passt die vorhergesagte 13 x 13-Karte zur beobachteten 13 x 13-Karte?

Die bisherigen Auswertungen (sts_dctm, pixel.py) fassen alle Pixel aller Stichproben in einer PR-AUC zusammen
("pixel"). Hier je Stichprobe die ganze Karte, dazu Aufschluesselungen, wo die Modelle sich unterscheiden.
Zwei Schritte (Ausgaben unter metrics/patch_maps<tag>/):
  scores  Karten (n, 13, 13) je dt fuer val und test, je Seed gemittelt -> scores_<modell>.npz
          persistence_px, unet3d_px  wie pixel.py (gleiche Funktionen, je dt und Seed neu trainiert, Ziel = Tag t)
          name=glob                  unsere Checkpoints (sts_dctm / sts_spread*), Score F(dt + 1) wie
                                     sts_dctm.paper_scores; Bias der sts_dctm-Koepfe wie im Training (joint_eval
                                     set_bias trained), sts_spread hat ihn im Checkpoint
  report  je Modell, dt und Ziel:
          Ziel "tag"   Feuer genau an Tag t (Paper-Label je Pixel, wie pixel.py)
          Ziel "kum"   Feuer an mindestens einem Tag e + 1 .. t (passt zur Survival-Frage "brennt bis t")
          PR-AUC Mitte / alle Pixel / je Ring (Chebyshev-Abstand 0..6 zur Mitte) / alle Pixel getrennt nach
          "brennt im Eingabefenster" (alt) und "neu"; Karte je Stichprobe mit Feuer: Schwelle q auf val (bester
          mittlerer IoU), dann IoU und Dice auf test, dazu IoU top-n (n = beobachtete Pixel).
Ziel D+1 (next_patch), cache_G, temporaler Split.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.paper import targets
from wildfire.paper.compare import CENTRE, DTS, window
from wildfire.paper.joint_eval import set_bias
from wildfire.paper.pixel import chunked, pixel_fire_fraction, unet_px_fit, unet_px_predict
from wildfire.paper.pixel_stdctm import windows
from wildfire.paper.prepare import FIRE_CH
from wildfire.paper.reconstruct import cache_norm, to_device_once
from wildfire.paper.sts_dctm import P, doy_of, load_sts
from wildfire.paper.sts_spread import load_spread

QS = (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
RING = np.maximum(*np.abs(np.mgrid[:P, :P] - CENTRE))                    # (13, 13) Chebyshev-Abstand 0..6


def load_data(exp: Path, cache: str, target: str):
    d = exp / "derived_data" / "paper" / f"{cache}_L61"
    X = np.load(d / "X.npy")
    split = np.load(d / "split.npy")
    fire = X[:, :, FIRE_CH] > 0
    if targets.shift(target):
        fire = np.concatenate([fire, targets.fire_map(d, X, target)[:, None]], 1)
    idx = {s: np.where(split == s)[0] for s in ("train", "val", "test")}
    return d, X, fire, idx, doy_of(np.load(d / "meta.npy")[:, 0])


# ----------------------------------------------------------------------------------------------- scores
def our_scores(paths, X, Xg, rows, L, dt_, doyD, m_t, s_t, device, cube_rates, bs=64):
    """Mittel ueber die Seeds von F(dt + 1) je Pixel, (n, 13, 13)."""
    LC = X.shape[1]
    e = L - dt_ - 2
    k = torch.tensor([float(dt_ + 1)], device=device)
    acc = 0.0
    for p in paths:
        a = torch.load(p, map_location="cpu").get("args", {})
        if "spread" in a:
            net = load_spread(p, device)
        else:
            net = load_sts(p, device)
            net._joint_args = a
            set_bias(net, "trained", cube_rates)
        out = []
        with torch.no_grad():
            for i in range(0, rows.size, bs):
                r = rows[i:i + bs]
                x = (windows(Xg, r, np.full(r.size, e)).to(device).float() - m_t) / s_t
                doy = torch.from_numpy(doyD[r] - (LC - 1 - e)).to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    raw_px = net(x, doy)[0]
                fp = -torch.expm1(net.px_head.log_survival(raw_px.reshape(-1, raw_px.shape[-1]).float(), k))[:, 0]
                out.append(fp.view(len(r), P, P).cpu().numpy())
        acc = acc + np.concatenate(out) / len(paths)
    return np.clip(acc, 1e-7, 1 - 1e-7)


# ----------------------------------------------------------------------------------------------- report
def ap(p, y):
    y = y.astype(int)
    return float(average_precision_score(y, p)) if 0 < y.sum() < y.size else float("nan")


def map_scores(S, Y, q):
    """Je Stichprobe mit mindestens einem Feuerpixel: IoU, Dice bei Schwelle q, IoU top-n."""
    has = Y.reshape(len(Y), -1).any(1)
    S, Y = S[has].reshape(has.sum(), -1), Y[has].reshape(has.sum(), -1)
    pred = S >= q
    inter, union = (pred & Y).sum(1), (pred | Y).sum(1)
    n = Y.sum(1)
    thr = np.take_along_axis(np.sort(S, 1)[:, ::-1], (n - 1)[:, None], 1)
    top = S >= thr
    return {"iou": float(np.mean(inter / np.maximum(union, 1))),
            "dice": float(np.mean(2 * inter / np.maximum(pred.sum(1) + n, 1))),
            "iou_topn": float(np.mean((top & Y).sum(1) / np.maximum((top | Y).sum(1), 1))),
            "n_maps": int(has.sum())}


def report(out: Path, fire, idx, L, rname: str = "summary") -> dict:
    files = sorted(out.glob("scores_*.npz"))
    res, md = {}, ["# Paper-Protokoll als 13 x 13-Karte (Ziel D+1, cache_G, temporaler Split)", "",
                   "Score: Survival-Modelle F(dt + 1) je Pixel, Pixel-U-Net / Persistenz P(Feuer an t) je Pixel. "
                   "Ziel tag = Feuer genau an t (Paper-Label), kum = Feuer an mindestens einem Tag e + 1 .. t. "
                   "Karte: Schwelle q auf val (bester mittlerer IoU der Stichproben mit Feuer), IoU / Dice auf test; "
                   "IoU top-n = die n wahrscheinlichsten Pixel (n = beobachtet), also Form ohne Flaechenfehler. "
                   "Ring = Chebyshev-Abstand zur Mitte. alt = Pixel brennt im Eingabefenster.", ""]
    for f in files:
        nm = f.stem.removeprefix("scores_")
        z = np.load(f)
        res[nm] = {}
        for dt_ in DTS:
            e = L - dt_ - 2
            for tg in ("tag", "kum"):
                Yv = {s: (fire[idx[s], L - 1] if tg == "tag" else fire[idx[s], e + 1:L].any(1)) for s in ("val", "test")}
                old = fire[idx["test"], e - 29:e + 1].any(1)
                Sv, St = z[f"val_{dt_}"], z[f"test_{dt_}"]
                Yt = Yv["test"]
                r = {"centre": ap(St[:, CENTRE, CENTRE], Yt[:, CENTRE, CENTRE]), "pixel": ap(St.ravel(), Yt.ravel()),
                     "rings": [ap(St[:, RING == k].ravel(), Yt[:, RING == k].ravel()) for k in range(CENTRE + 1)],
                     "old": ap(St[old], Yt[old]), "new": ap(St[~old], Yt[~old])}
                qb = max(QS, key=lambda q: map_scores(Sv, Yv["val"], q)["iou"])
                r.update({"q": qb, **map_scores(St, Yt, qb)})
                res[nm][f"{dt_}_{tg}"] = r
    for tg in ("tag", "kum"):
        md += [f"## Ziel {tg}", "", "| Modell | dt | PR-AUC Mitte | alle Pixel | alt | neu | Ring 0-1 / 2-3 / 4-5 / 6 | q | IoU | Dice | IoU top-n |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for dt_ in DTS:
            for nm in res:
                r = res[nm][f"{dt_}_{tg}"]
                rg = r["rings"]
                rr = " / ".join(f"{np.nanmean(rg[a:b]):.3f}" for a, b in ((0, 2), (2, 4), (4, 6), (6, 7)))
                md.append(f"| {nm} | {dt_} | {r['centre']:.3f} | {r['pixel']:.3f} | {r['old']:.3f} | {r['new']:.3f} | {rr} | "
                          f"{r['q']} | {r['iou']:.3f} | {r['dice']:.3f} | {r['iou_topn']:.3f} |")
        md.append("")
    (out / f"{rname}.md").write_text("\n".join(md) + "\n")
    (out / f"{rname}.json").write_text(json.dumps(res, indent=1))
    print("\n".join(md), flush=True)
    return res


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("stage", choices=["scores", "report"])
    ap_.add_argument("--config", required=True)
    ap_.add_argument("--cache", default="cache_G")
    ap_.add_argument("--target", choices=targets.TARGETS, default="next_patch")
    ap_.add_argument("--model", default="", help="persistence_px | unet3d_px | name=glob")
    ap_.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap_.add_argument("--cube-dir", default="paper_pixel")
    ap_.add_argument("--tag", default="")
    ap_.add_argument("--report-name", default="summary", help="report: Dateiname ohne Endung")
    args = ap_.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    out = exp / "metrics" / f"patch_maps{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d, X, fire, idx, doyD = load_data(exp, args.cache, args.target)
    L = X.shape[1] + targets.shift(args.target)
    print(f"[pm] {args.cache}: X {X.shape}, val {idx['val'].size}, test {idx['test'].size} ({time.time() - t0:.0f} s)", flush=True)
    if args.stage == "report":
        rep = out / f"{args.report_name}.md"
        if rep.exists():
            raise SystemExit(f"ABBRUCH: {rep} existiert")
        report(out, fire, idx, L, args.report_name)
        write_manifest(cfg, f"p66_patch_maps{args.tag}", args.config, {"cache": args.cache, "target": args.target,
                                                                          "outputs": [str(rep)]})
        return
    name, pattern = (args.model.split("=", 1) + [""])[:2]
    f = out / f"scores_{name}.npz"
    if f.exists():
        raise SystemExit(f"ABBRUCH: {f} existiert")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / targets.base_cache(args.cache) / "norm_stats.npz")
    Y = fire[:, L - 1].astype(np.int8)
    res = {}
    if name == "persistence_px":
        for dt_ in DTS:
            a, b = window(L, dt_)
            lr_ = LogisticRegression().fit(chunked(X, idx["train"], a, b, pixel_fire_fraction)[:, None], Y[idx["train"]].reshape(-1))
            for s in ("val", "test"):
                res[f"{s}_{dt_}"] = lr_.predict_proba(chunked(X, idx[s], a, b, pixel_fire_fraction)[:, None])[:, 1].reshape(-1, P, P)
    elif name == "unet3d_px":
        Xg = to_device_once(X, device)
        for dt_ in DTS:
            a, b = window(L, dt_)
            acc = {s: 0.0 for s in ("val", "test")}
            for seed in args.seeds:
                print(f"[pm] unet3d_px dt {dt_} seed {seed}", flush=True)
                net, _ = unet_px_fit(Xg, Y, idx, a, b, mean, std, device, seed)
                for s in acc:
                    acc[s] = acc[s] + unet_px_predict(net, Xg, idx[s], a, b, mean, std, device) / len(args.seeds)
            res.update({f"{s}_{dt_}": v for s, v in acc.items()})
    else:
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise SystemExit(f"ABBRUCH: keine Checkpoints fuer {name}: {pattern}")
        print(f"[pm] {name}: " + ", ".join(Path(p).name for p in paths), flush=True)
        ev_tr = np.load(exp / "derived_data" / args.cube_dir / "train_event.npy").astype(bool)
        cube_rates = (float(ev_tr.mean()), float(ev_tr.reshape(len(ev_tr), -1).any(1).mean()))
        Xg = to_device_once(X, device)
        m_t = torch.from_numpy(mean).float().to(device)[None, None, :, None, None]
        s_t = torch.from_numpy(std).float().to(device)[None, None, :, None, None]
        for dt_ in DTS:
            for s in ("val", "test"):
                res[f"{s}_{dt_}"] = our_scores(paths, X, Xg, idx[s], L, dt_, doyD, m_t, s_t, device, cube_rates)
            print(f"[pm] {name} dt {dt_}: test Mitte PR-AUC "
                  f"{ap(res[f'test_{dt_}'][:, CENTRE, CENTRE], fire[idx['test'], L - 1, CENTRE, CENTRE]):.3f}, alle Pixel "
                  f"{ap(res[f'test_{dt_}'].ravel(), fire[idx['test'], L - 1].ravel()):.3f} ({time.time() - t0:.0f} s)", flush=True)
    np.savez_compressed(f, **{k: v.astype(np.float32) for k, v in res.items()})
    write_manifest(cfg, f"p66_patch_maps_{name}{args.tag}", args.config, {"model": args.model, "seeds": args.seeds,
                                                                             "cache": args.cache, "target": args.target,
                                                                             "outputs": [str(f)]})


if __name__ == "__main__":
    main()
