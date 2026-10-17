"""Paper-Protokoll pro Pixel: jedes der 13 x 13 Pixel hat sein eigenes Label.

Das Paper-Label ist eine Zahl je Stichprobe (Feuer irgendwo im Ausschnitt am
Zieltag t). Hier wird statt dessen fuer jedes Pixel (r, c) des Ausschnitts
gelernt, ob genau dieses Pixel an t brennt:

    y(s, r, c) = Feuermaske (Cache-Kanal 0, MODIS) an Pixel (r, c) am Tag t

Das ist dieselbe Quelle wie das Paper-Label, nur nicht ueber den Ausschnitt
zusammengefasst. Stichproben, Splits, 55 Kanaele, T = 30 und das Verschiebe-
Protokoll (Fenster endet bei t - dt - 1) bleiben wie beim Paper.

  persistence_px  Anteil der Fenstertage mit Feuer an genau diesem Pixel,
                  logistische Regression auf TRAIN
  hgb_px          Gradient Boosting, eine Zeile je Pixel: die 5 x 55 = 275
                  Zusammenfassungen dieses Pixels (wie compare.summaries fuer
                  die Mittelzelle), ein Modell je dt
  unet3d_px       das 3D-U-Net des Nachbaus, aber ohne raeumliches Pooling:
                  Mittel nur ueber die Zeit, dann ein Logit je Pixel
                  (13 x 13-Karte), Kreuzentropie je Pixel, ein Modell je dt

Ausgewertet auf dem Test-Split dreifach:
  centre  nur die Mittelzelle (gleiches Label wie p5_compare_*_centre)
  pixel   alle 169 Pixel aller Test-Stichproben
  patch   Maximum der Pixel-Risiken gegen das Paper-Label
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.paper.channels import CHANNELS, has_fire, summary_columns
from wildfire.paper.channels import tag as channel_tag
from wildfire.paper.compare import CENTRE, DTS, KEYS, window
from wildfire.paper.metrics import ASSUMPTIONS, paper_metrics
from wildfire.paper.prepare import FIRE_CH
from wildfire.paper import targets
from wildfire.paper.reconstruct import UNet3D, cache_norm, loader, to_device_once

P = 13


def pixel_summaries(xw: np.ndarray) -> np.ndarray:
    """(n, T, 55, 13, 13) -> (n * 169, 275); Zeilen je Stichprobe, darin zeilenweise (r, c)."""
    s = xw.astype(np.float32).transpose(0, 3, 4, 1, 2).reshape(-1, xw.shape[1], xw.shape[2])
    return np.concatenate([s[:, -1], s[:, -7:].mean(1), s.mean(1), s.min(1), s.max(1)], 1)


def pixel_fire_fraction(xw: np.ndarray) -> np.ndarray:
    """(n, T, 55, 13, 13) -> (n * 169,) Anteil der Fenstertage mit Feuer am Pixel."""
    return (xw[:, :, FIRE_CH] > 0).mean(1).reshape(-1).astype(np.float64)


class UNet3DPixel(UNet3D):
    """3D-U-Net mit einem Logit je Pixel: Mittel ueber die Zeit, kein raeumliches Pooling."""

    def __init__(self, c_in: int, prior: float):
        super().__init__(c_in)
        nn.init.constant_(self.head.bias, float(np.log(prior / (1 - prior))))

    def forward(self, x):                                   # (B, C, T, H, W) -> (B, H, W)
        z = self.drop(self.decode(x).mean(dim=2))           # (B, w, H, W)
        return self.head(z.permute(0, 2, 3, 1)).squeeze(-1)


def map_metrics(maps_per_seed, Y_te: np.ndarray, y_te: np.ndarray) -> dict:
    """Risikokarten (n_test, 13, 13) je Seed -> Paper-Metriken fuer centre, pixel, patch."""
    out = {}
    for ev in ("centre", "pixel", "patch"):
        ms = []
        for m in maps_per_seed:
            if ev == "centre":
                ms.append(paper_metrics(m[:, CENTRE, CENTRE], Y_te[:, CENTRE, CENTRE]))
            elif ev == "pixel":
                ms.append(paper_metrics(m.reshape(-1), Y_te.reshape(-1)))
            else:
                ms.append(paper_metrics(m.max(axis=(1, 2)), y_te))
        out[ev] = {"summary": {k: {"mean": float(np.mean([x[k] for x in ms])),
                                   "std": float(np.std([x[k] for x in ms]))} for k in KEYS},
                   "per_seed": ms}
    return out


def chunked(X, rows, t0, t1, fn, bs=500):
    return np.concatenate([fn(np.asarray(X[rows[a:a + bs], t0:t1])) for a in range(0, rows.size, bs)])


@torch.no_grad()
def unet_px_predict(net, X, rows, t0, t1, mean, std, device, bs=128, chs=tuple(range(55))):
    net.eval()
    out = np.zeros((rows.size, P, P))
    where = {int(i): k for k, i in enumerate(rows)}
    for sel, xb in loader(X, rows, list(chs), t0, t1, mean, std, bs, device):
        pr = torch.sigmoid(net(xb)).double().cpu().numpy()
        for i, v in zip(sel, pr):
            out[where[int(i)]] = v
    return np.clip(out, 1e-7, 1 - 1e-7)


def unet_px_fit(X, Y, idx, t0, t1, mean, std, device, seed, epochs=40, patience=6, bs=64, lr=1e-3,
                chs=tuple(range(55))):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    prior = float(Y[idx["train"]].mean())
    net = UNet3DPixel(len(chs), prior).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    Yt = torch.from_numpy(Y.astype(np.float32)).to(device)
    yv = Y[idx["val"]].astype(float)
    best, state, bad, hist = np.inf, None, 0, []
    for ep in range(epochs):
        net.train()
        tl = tn = 0
        for sel, xb in loader(X, idx["train"], list(chs), t0, t1, mean, std, bs, device, rng):
            loss = F.binary_cross_entropy_with_logits(net(xb), Yt[sel])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tl += loss.item() * len(sel); tn += len(sel)
        pv = unet_px_predict(net, X, idx["val"], t0, t1, mean, std, device, chs=chs)
        vl = float(-np.mean(yv * np.log(pv) + (1 - yv) * np.log(1 - pv)))
        hist.append({"epoch": ep, "train_bce": tl / tn, "val_bce": vl})
        print(f"      Epoche {ep:2d}: train {tl / tn:.5f}  val BCE je Pixel {vl:.5f}", flush=True)
        if vl < best - 1e-5:
            best, bad = vl, 0
            state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state)
    return net, hist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--methods", nargs="+", default=["persistence_px", "hgb_px", "unet3d_px"])
    ap.add_argument("--target", choices=targets.TARGETS, default="cache",
                    help="cache: Feuer an D; next_patch: Feuer an D+1, Fenster einen Tag spaeter (targets.py)")
    ap.add_argument("--channels", choices=list(CHANNELS), default="paper55",
                    help="Eingabekanaele; das Label je Pixel bleibt die Feuermaske an t")
    args = ap.parse_args()
    chs = CHANNELS[args.channels]
    cols = summary_columns(args.channels)
    if "persistence_px" in args.methods and not has_fire(args.channels):
        print("[px] persistence_px uebersprungen: sie besteht nur aus der Feuermaske", flush=True)
        args.methods = [m for m in args.methods if m != "persistence_px"]
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t_load = time.time()
    d = exp / "derived_data" / "paper" / f"{args.cache}_L61"
    X = np.load(d / "X.npy")                                 # (N, 61, 55, 13, 13)
    y = targets.labels(d, args.target)
    split = np.load(d / "split.npy")
    L = X.shape[1] + targets.shift(args.target)          # effektive Laenge: Zieltag = Index L - 1
    idx = {s: np.where(split == s)[0] for s in ("train", "val", "test")}
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / args.cache / "norm_stats.npz")

    fire_t = targets.fire_map(d, X, args.target).astype(np.uint8)   # (N, 13, 13), Zieltag t
    vals = np.unique(fire_t)
    Y = (fire_t > 0).astype(np.int8)
    data_info = {
        "fire_mask_values_on_t": vals.tolist(),
        **{f"{s}_pixels": int(Y[idx[s]].size) for s in idx},
        **{f"{s}_fire_pixels": int(Y[idx[s]].sum()) for s in idx},
        **{f"{s}_samples_with_centre_fire": int(Y[idx[s], CENTRE, CENTRE].sum()) for s in idx},
        "paper_label_equals_patch_any_on_t": {s: float((Y[idx[s]].any((1, 2)) == y[idx[s]]).mean()) for s in idx},
    }
    print(f"[px] {args.cache}: X {X.shape} in {time.time() - t_load:.0f} s; {data_info}", flush=True)
    if not set(vals.tolist()) <= {0, 1}:
        print(f"[px] WARNUNG: Feuermaske hat Werte {vals.tolist()} - alles > 0 gilt als Feuer", flush=True)

    te = idx["test"]
    res: dict = {}
    # Netze lesen aus der GPU-Kopie, Merkmale fuer HGB und Persistenz aus dem RAM
    Xg = to_device_once(X, device) if "unet3d_px" in args.methods else X

    def record(method, dt_, maps_per_seed):
        """maps: (n_test, 13, 13) Risiko je Pixel."""
        out = map_metrics(maps_per_seed, Y[te], y[te])
        res.setdefault(method, {})[str(dt_)] = out
        print(f"[px] {method:15s} dt={dt_:2d}  " + "   ".join(
            f"{ev}: PR-AUC {out[ev]['summary']['pr_auc']['mean']:.3f} ROC {out[ev]['summary']['roc_auc']['mean']:.3f}"
            for ev in out), flush=True)

    for dt_ in DTS:
        t0, t1 = window(L, dt_)
        print(f"\n[px] dt = {dt_}: Fenster {t0}..{t1 - 1} (Zieltag = Index {L - 1})", flush=True)

        if "persistence_px" in args.methods:
            ftr = chunked(X, idx["train"], t0, t1, pixel_fire_fraction)
            lr_ = LogisticRegression().fit(ftr[:, None], Y[idx["train"]].reshape(-1))
            fte = chunked(X, te, t0, t1, pixel_fire_fraction)
            record("persistence_px", dt_, [lr_.predict_proba(fte[:, None])[:, 1].reshape(-1, P, P)])

        if "hgb_px" in args.methods:
            Ftr = chunked(X, idx["train"], t0, t1, lambda xw: pixel_summaries(xw)[:, cols])
            ytr = Y[idx["train"]].reshape(-1)
            print(f"    hgb_px: {Ftr.shape[0]:,} Trainingszeilen, {int(ytr.sum()):,} davon brennen", flush=True)
            maps = []
            for seed in args.seeds:
                h = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                                   min_samples_leaf=20, l2_regularization=1.0,
                                                   early_stopping=True, validation_fraction=0.15,
                                                   random_state=seed).fit(Ftr, ytr)
                p = np.concatenate([h.predict_proba(pixel_summaries(np.asarray(X[te[a:a + 500], t0:t1]))[:, cols])[:, 1]
                                    for a in range(0, te.size, 500)])
                maps.append(p.reshape(-1, P, P))
            del Ftr
            record("hgb_px", dt_, maps)

        if "unet3d_px" in args.methods:
            maps = []
            for seed in args.seeds:
                print(f"    unet3d_px seed {seed}", flush=True)
                net, _ = unet_px_fit(Xg, Y, idx, t0, t1, mean, std, device, seed, chs=chs)
                maps.append(unet_px_predict(net, Xg, te, t0, t1, mean, std, device, chs=chs))
            record("unet3d_px", dt_, maps)

    write_manifest(cfg, f"p9_pixel_{args.cache}{targets.tag(args.target)}{channel_tag(args.channels)}", args.config, {
        "cache": args.cache, "label": "Feuermaske (Kanal 0) je Pixel am Zieltag t",
        "target": args.target, "channels": args.channels, "channel_list": chs,
        "protocol": "Fenster [t - dt - 30, t - dt - 1], dt in {1, 7, 15, 30}",
        "evaluations": {"centre": "Mittelzelle gegen ihr Pixel-Label",
                        "pixel": "alle 169 Pixel aller Test-Stichproben",
                        "patch": "Maximum der Pixel-Risiken gegen das Paper-Label"},
        "data": data_info, "seeds": args.seeds, "metric_assumptions": ASSUMPTIONS,
        "results": res,
    })

    ref_path = exp / "data_manifests" / f"p5_compare_{args.cache}_centre.json"
    if ref_path.exists():
        ref = json.loads(ref_path.read_text())["results"]
        print(f"\n[px] {args.cache} TEST, Mittelzelle, PR-AUC: pro Stichprobe trainiert (p5) gegen pro Pixel")
        for dt_ in DTS:
            row = [f"dt={dt_:2d}"]
            row += [f"{m} {ref[m][str(dt_)]['summary']['pr_auc']['mean']:.3f}" for m in ("unet3d_dt", "hgb_dt") if m in ref]
            row += [f"{m} {res[m][str(dt_)]['centre']['summary']['pr_auc']['mean']:.3f}" for m in res]
            print("    " + "   ".join(row))


if __name__ == "__main__":
    main()
