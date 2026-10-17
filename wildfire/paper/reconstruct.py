"""Paper-Protokoll, Schritt 3: das Ergebnis des Papers rekonstruieren.

Das Paper berichtet fuer ein 3D-U-Net eine PR-AUC um 0,68. Mit allen 55
Cache-Kanaelen erreicht schon unser kleines Netz 0,99, ohne Feuermaske 0,93.
Das Paper muss also an mindestens einer Stelle anders vorgehen. Drei Stellen
sind zwischen Paper, README und Cache widerspruechlich; sie werden hier
vollstaendig gekreuzt:

  Kanaele      paper55  alle 55 wie im Cache (Kanal 0 = MODIS-Feuermaske)
               no_fire  54 Kanaele, Feuermaske entfernt (README: "not included
                        as a model input channel")
  Fenster T    30 Tage (Paper-Text) oder 10 Tage (README: "10-day backward
               window")
  Fensterende  end=0  letzter Schritt = Sample-Datum D (so liegt der Cache vor)
               end=1  letzter Schritt = D-1: echte Vorhersage fuer den naechsten
                      Tag, wenn das Label sich auf D bezieht (Audit s6)

Normalisierung standardmaessig mit den norm_stats.npz des Caches, so wie sie
mitgeliefert sind; Modell ein 3D-U-Net ueber (T, H, W) mit globalem Pooling
auf ein Logit, trainiert mit binaerer Kreuzentropie.

Das Paper selbst (Auszug, vom Nutzer geliefert): "Fuer jedes dt in {1, 7, 15,
30} werden der Zieltag t und seine Kennung Y_s,t festgehalten, und nur das
Eingabefenster wird rueckwaerts auf [t - dt - T, t - dt - 1] verschoben."
Mit t = Sample-Datum D endet das Fenster also bei D - (dt + 1):
--end 2 8 16 31 bei --T 30. Bei dt = 30 beginnt es bei D-60 - genau so weit
reicht der Cache (61 Tage); das bestaetigt T = 30.

Braucht das volle Fenster: 40_paper_prepare mit LOOKBACK=61 TAG=_L61.
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.paper import extra
from wildfire.paper.channels import CHANNELS
from wildfire.paper.metrics import ASSUMPTIONS, paper_metrics
from wildfire.paper.prepare import FIRE_CH, LANDCOVER_CH
from wildfire.paper.run import channel_stats


def block3d(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.SiLU(inplace=True),
        nn.Conv3d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.SiLU(inplace=True),
    )


class UNet3D(nn.Module):
    """Zwei Pooling-Stufen ueber (T, H, W), Skip-Verbindungen, globales Pooling."""

    def __init__(self, c_in: int, width: int = 32, dropout: float = 0.1):
        super().__init__()
        w = width
        self.enc1, self.enc2, self.bott = block3d(c_in, w), block3d(w, 2 * w), block3d(2 * w, 4 * w)
        self.dec2, self.dec1 = block3d(4 * w + 2 * w, 2 * w), block3d(2 * w + w, w)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(w, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, -0.693)           # logit(1/3)

    def decode(self, x):                                    # (B, C, T, H, W) -> (B, w, T, H, W)
        s1 = self.enc1(x)
        s2 = self.enc2(F.max_pool3d(s1, 2, ceil_mode=True))
        b = self.bott(F.max_pool3d(s2, 2, ceil_mode=True))
        d2 = self.dec2(torch.cat([F.interpolate(b, size=s2.shape[2:], mode="trilinear",
                                                align_corners=False), s2], 1))
        return self.dec1(torch.cat([F.interpolate(d2, size=s1.shape[2:], mode="trilinear",
                                                  align_corners=False), s1], 1))

    def forward(self, x):                                   # (B, C, T, H, W)
        return self.head(self.drop(self.decode(x).mean(dim=(2, 3, 4)))).squeeze(-1)


def cache_norm(path: Path, n_ch: int = 55):
    """norm_stats.npz des Caches auf 55 Kanaele bringen.

    Laut README sind Feuermaske und Landbedeckung von der Normierung
    ausgenommen. Steht die Datei mit 54 oder 53 Eintraegen da, werden sie den
    uebrigen Kanaelen der Reihe nach zugeordnet.
    """
    z = np.load(path, allow_pickle=True)
    mean, std = np.asarray(z["mean"], np.float64).ravel(), np.asarray(z["std"], np.float64).ravel()
    info = {k: (z[k].tolist() if z[k].ndim else z[k].item()) for k in z.files
            if k not in ("mean", "std")}
    info["len_mean"] = int(mean.size)
    full_m, full_s = np.zeros(n_ch), np.ones(n_ch)
    if mean.size == n_ch:
        full_m, full_s = mean, std
    else:
        others = [c for c in range(n_ch) if c not in ((FIRE_CH, LANDCOVER_CH) if mean.size == n_ch - 2
                                                      else (FIRE_CH,))]
        if len(others) != mean.size:
            raise SystemExit(f"norm_stats mit {mean.size} Eintraegen passt zu keiner Zuordnung")
        full_m[others], full_s[others] = mean, std
    full_s = np.where(full_s > 1e-8, full_s, 1.0)
    return full_m.astype(np.float32), full_s.astype(np.float32), info


def to_device_once(X: np.ndarray, device: torch.device, name: str = "X"):
    """Ganzes uint8-Array einmal auf die GPU; ohne GPU bleibt es ein numpy-Array.

    Ausschneiden und Kopieren je Batch aus dem RAM war sonst der Engpass: die GPU
    wartete die meiste Zeit (Batch 512: 2,5 min je Epoche statt 3 s).
    """
    if device.type != "cuda":
        return X
    t = time.time()
    Xt = torch.from_numpy(np.ascontiguousarray(X)).to(device)
    print(f"[gpu] {name} {tuple(Xt.shape)} ({Xt.numel() / 1e9:.1f} GB) auf {device} in {time.time() - t:.0f} s",
          flush=True)
    return Xt


def gather_windows(X, rows, starts, T: int):
    """(B, T, C, H, W): je Zeile rows[i] die Tage starts[i] .. starts[i] + T - 1.

    X ist ein numpy-Array (Ergebnis numpy) oder ein torch-Tensor (Ergebnis auf
    dessen Geraet, ein einziger Gather ohne Umweg ueber den RAM).
    """
    rows, starts = np.asarray(rows, np.int64), np.asarray(starts, np.int64)
    if isinstance(X, torch.Tensor):
        r = torch.from_numpy(rows).to(X.device)
        t = torch.from_numpy(starts).to(X.device)[:, None] + torch.arange(T, device=X.device)
        return X[r[:, None], t]
    return np.stack([X[r, s:s + T] for r, s in zip(rows, starts)])


def as_float(xb, device):
    """numpy- oder Tensor-Batch -> float-Tensor auf device."""
    return (xb if isinstance(xb, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(xb))).to(device).float()


def loader(X, idx, chs, t0, t1, mean, std, bs, device, rng=None):
    """X: numpy-Array im RAM oder torch-Tensor (dann wird direkt dort ausgeschnitten)."""
    order = rng.permutation(idx) if rng is not None else idx
    m = torch.from_numpy(mean[chs]).to(device)[None, :, None, None, None]
    s = torch.from_numpy(std[chs]).to(device)[None, :, None, None, None]
    on_tensor = isinstance(X, torch.Tensor)
    if on_tensor:
        chs_t = torch.as_tensor(list(chs), device=X.device)
    for a in range(0, order.size, bs):
        sel = np.sort(order[a:a + bs])
        if on_tensor:
            xb = X[torch.from_numpy(sel).to(X.device), t0:t1].index_select(2, chs_t).to(device).float()
        else:
            xb = torch.from_numpy(X[sel, t0:t1][:, :, chs]).to(device).float()   # (B, T, C, H, W)
        yield sel, (xb.permute(0, 2, 1, 3, 4) - m) / s                         # (B, C, T, H, W)


@torch.no_grad()
def predict(net, X, idx, chs, t0, t1, mean, std, device, bs=128):
    net.eval()
    p = np.zeros(idx.size)
    where = {int(i): k for k, i in enumerate(idx)}
    for sel, xb in loader(X, idx, chs, t0, t1, mean, std, bs, device):
        out = torch.sigmoid(net(xb)).double().cpu().numpy()
        for i, v in zip(sel, out):
            p[where[int(i)]] = v
    return np.clip(p, 1e-7, 1 - 1e-7)


def fit(X, y, idx, chs, t0, t1, mean, std, device, seed, epochs=40, patience=6, bs=64, lr=1e-3,
        select="val_bce"):
    """select: "val_bce" = bestes Modell nach Val-BCE mit Abbruch nach patience Epochen;
    "last" = feste Epochenzahl, das Modell nach der letzten Epoche."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = UNet3D(len(chs)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    yt = torch.from_numpy(y.astype(np.float32)).to(device)
    best, state, bad, hist = np.inf, None, 0, []
    for ep in range(epochs):
        net.train()
        tl = tn = 0
        for sel, xb in loader(X, idx["train"], chs, t0, t1, mean, std, bs, device, rng):
            loss = F.binary_cross_entropy_with_logits(net(xb), yt[sel])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tl += loss.item() * len(sel); tn += len(sel)
        pv = predict(net, X, idx["val"], chs, t0, t1, mean, std, device)
        yv = y[idx["val"]]
        vl = float(-np.mean(yv * np.log(pv) + (1 - yv) * np.log(1 - pv)))
        vm = paper_metrics(pv, yv)
        hist.append({"epoch": ep, "train_bce": tl / tn, "val_bce": vl, "val_pr_auc": vm["pr_auc"]})
        print(f"      Epoche {ep:2d}: train {tl / tn:.4f}  val BCE {vl:.4f}  val PR-AUC "
              f"{vm['pr_auc']:.3f}", flush=True)
        if select == "last":
            continue
        if vl < best - 1e-4:
            best, bad = vl, 0
            state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if select != "last":
        net.load_state_dict(state)
    return net, hist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", default="cache_G")
    ap.add_argument("--channels", nargs="+", default=list(CHANNELS))
    ap.add_argument("--T", type=int, nargs="+", default=[30, 10])
    ap.add_argument("--end", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--norm", choices=("cache", "ours"), default="cache")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--run-name", default="",
                    help="Suffix des Manifests, damit Laeufe sich nicht ueberschreiben")
    # Paper: Mini-Batch 512, Lernrate 1e-4 (Epochen nennt es nicht; Abbruch ueber Val-BCE)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--select", choices=["val_bce", "last"], default="val_bce",
                    help="Modellauswahl: bestes nach Val-BCE oder letzte Epoche (feste Epochenzahl)")
    ap.add_argument("--extra", choices=extra.MODES, default="none",
                    help="zusaetzliche Train-Stichproben aus 2000-2019 (paper_extra.py) anhaengen")
    ap.add_argument("--target", choices=["cache", "next_patch", "next_centre"], default="cache",
                    help="cache: /labels (Feuer im Ausschnitt an D); next_*: Feuer an D+1 (next_labels.py)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    d = exp / "derived_data" / "paper" / f"{args.cache}_L61"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t_load = time.time()
    X = np.load(d / "X.npy")                                 # (N, 61, 55, 13, 13)
    y = np.load(d / ("y.npy" if args.target == "cache" else f"y_{args.target}.npy")).astype(int)
    print(f"[rec] Ziel {args.target}: Basisrate {y.mean():.3f}", flush=True)
    split = np.load(d / "split.npy")
    if args.extra != "none":
        if args.target == "next_centre":
            raise SystemExit("ABBRUCH: --extra gibt es fuer cache und next_patch")
        X, y, split = extra.append(X, y, split, extra.load(exp, args.target, args.extra))
    idx = {s: np.where(split == s)[0] for s in ("train", "val", "test")}
    L = X.shape[1]
    print(f"[rec] {args.cache}: X {X.shape} in {time.time() - t_load:.0f} s auf {device}")

    if args.norm == "cache":
        mean, std, ninfo = cache_norm(Path(cfg["dataset_root"]) / args.cache / "norm_stats.npz")
    else:
        mean, std = channel_stats(X, idx["train"])
        ninfo = {"source": "train split, eigene Rechnung"}
    print(f"[rec] Normierung {args.norm}: {ninfo}")
    X = to_device_once(X, device)

    runs = []
    preds = {"idx_val": idx["val"], "idx_test": idx["test"]}
    for cset, T, end in itertools.product(args.channels, args.T, args.end):
        chs = CHANNELS[cset]
        t1 = L - end
        t0 = t1 - T
        tag = f"{cset}_T{T}_end{end}"
        print(f"\n[rec] {tag}: Tage D-{T - 1 + end} .. D-{end}, {len(chs)} Kanaele", flush=True)
        for seed in args.seeds:
            t_run = time.time()
            net, hist = fit(X, y, idx, chs, t0, t1, mean, std, device, seed,
                            epochs=args.epochs, patience=args.patience, bs=args.bs, lr=args.lr,
                            select=args.select)
            p = {s: predict(net, X, idx[s], chs, t0, t1, mean, std, device) for s in ("val", "test")}
            for s in ("val", "test"):
                preds[f"{tag}_s{seed}_{s}"] = p[s].astype(np.float32)
            res = {s: paper_metrics(p[s], y[idx[s]]) for s in ("val", "test")}
            t = res["test"]
            print(f"[rec] {tag} seed {seed} TEST  PR-AUC {t['pr_auc']:.3f}  ROC {t['roc_auc']:.3f}  "
                  f"F1 {t['f1']:.3f}  P {t['precision']:.3f}  R {t['recall']:.3f}  "
                  f"ECE {t['ece']:.3f}  Brier {t['brier']:.3f}  ({time.time() - t_run:.0f} s)",
                  flush=True)
            runs.append({"config": tag, "channels": cset, "T": T, "end": end, "seed": seed,
                         "history": hist, **res})

    print("\n[rec] Zusammenfassung TEST (PR-AUC, Ziel laut Paper ~0,68):")
    for r in sorted(runs, key=lambda r: r["test"]["pr_auc"]):
        print(f"    {r['config']:22s} seed {r['seed']}  PR-AUC {r['test']['pr_auc']:.3f}  "
              f"ROC {r['test']['roc_auc']:.3f}  F1 {r['test']['f1']:.3f}")

    pred_path = exp / "predictions" / "paper" / f"rec_{args.cache}_{args.norm}{args.run_name}.npz"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pred_path, **preds)
    write_manifest(cfg, f"p2_reconstruct_{args.cache}_{args.norm}{args.run_name}", args.config, {
        "predictions": str(pred_path),
        "cache": args.cache, "norm": args.norm, "norm_info": ninfo, "target": args.target, "extra": args.extra, "n_train": int(idx["train"].size),
        "model": "UNet3D, Breite 32, 2 Pooling-Stufen, globales Pooling, BCE",
        "training": {"bs": args.bs, "lr": args.lr, "max_epochs": args.epochs, "patience": args.patience,
                     "optimizer": "AdamW, weight_decay 1e-4", "selection": args.select},
        "metric_assumptions": ASSUMPTIONS, "target_pr_auc_paper": 0.68,
        "runs": runs,
    })


if __name__ == "__main__":
    main()
