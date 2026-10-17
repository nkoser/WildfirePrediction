"""Paper-Protokoll, Schritt 6: Methoden fair vergleichen, alle auf den Labels des Papers.

Gleich fuer alle Methoden: Stichproben, Splits und Labels aus dem Cache, alle
55 Kanaele, T = 30, Verschiebe-Protokoll (Fenster endet bei t - dt - 1),
binaere Kreuzentropie, Paper-Metriken auf dem Test-Split.

  M0  persistence  Anteil der Fenstertage mit Feuer im Ausschnitt, Platt auf TRAIN
  M1  unet3d_dt    3D-U-Net je dt (Vorhersagen aus 42_paper_reconstruct, _paperdt)
  M2  hgb_dt       Gradient Boosting je dt auf Zusammenfassungen der Mittelzelle
  M3  mlp_dt       MLP je dt, Rumpf wie unser B4, auf denselben Zusammenfassungen
  M4  mlp_multi    EIN MLP fuer alle dt, ein Logit je dt (Prinzip unseres
                   Survival-Kopfs: geteilter Rumpf, horizontweise Ausleserichtung)
  M5  unet3d_multi EIN 3D-U-Net fuer alle dt, ein Logit je dt

Zusammenfassungen je Kanal ueber das 30-Tage-Fenster der Mittelzelle: letzter
Wert, Mittel der letzten 7 Tage, Mittel, Minimum, Maximum (wie features.py,
aber aus den Cache-Kanaelen und ohne Sperre - das Paper sperrt nichts). Keine
Zusatzmerkmale wie Breite oder Kalendertag, damit alle Methoden genau die
Information des Caches sehen.

Mit --test-label centre wird nur der Test anders bewertet: positiv ist eine
Stichprobe, wenn die Feuermaske an t in der Mittelzelle 1 ist (Positive mit
Feuer nur bei Nachbarn zaehlen als 0). Training und Validierung bleiben beim
Paper-Label; dort brennt bei allen Positiven ohnehin die Mittelzelle.

Echte Survival-Ziele (erste Zuendung ueber die Zeit) lassen sich aus diesen
Labels nicht bilden: Es gibt je Stichprobe nur ein Label an einem Tag.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.paper.channels import CHANNELS, has_fire, masked_std, summary_columns
from wildfire.paper.channels import tag as channel_tag
from wildfire.paper.metrics import ASSUMPTIONS, paper_metrics
from wildfire.paper.prepare import FIRE_CH
from wildfire.paper import extra, targets
from wildfire.paper.reconstruct import UNet3D, as_float, cache_norm, gather_windows, to_device_once

DTS = (1, 7, 15, 30)
T = 30
CENTRE = 6
KEYS = ("f1", "pr_auc", "roc_auc", "precision", "recall", "ece", "mce", "brier")


def window(L: int, dt_: int) -> tuple[int, int]:
    t1 = L - (dt_ + 1)
    return t1 - T, t1


# ------------------------------------------------------------------ Merkmale
def summaries(X, idx, t0, t1):
    """(n, 5 * 55) Zusammenfassungen der Mittelzelle."""
    s = X[:, t0:t1, :, CENTRE, CENTRE][idx].astype(np.float32)      # (n, 30, 55)
    return np.concatenate([s[:, -1], s[:, -7:].mean(1), s.mean(1), s.min(1), s.max(1)], 1)


def fire_fraction(X, idx, t0, t1):
    return (X[idx, t0:t1, FIRE_CH] > 0).any(axis=(2, 3)).mean(1).astype(np.float64)


# ------------------------------------------------------------------ MLP
class MLP(nn.Module):
    """Rumpf wie unser B4: 3 x [Linear, LayerNorm, SiLU, Dropout], Breite 256."""

    def __init__(self, n_in: int, n_out: int, width: int = 256, depth: int = 3, p: float = 0.1):
        super().__init__()
        layers, c = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(c, width), nn.LayerNorm(width), nn.SiLU(), nn.Dropout(p)]
            c = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(c, n_out)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, -0.693)

    def forward(self, x):
        return self.head(self.body(x))


def fit_mlp(Xtr, ytr, ktr, Xva, yva, kva, n_out, seed, device, epochs=200, patience=15, bs=256):
    """Allgemein: Zeile i hat Ziel y[i] am Ausgang k[i] (k = 0 bei je-dt-Modellen)."""
    torch.manual_seed(seed)
    net = MLP(Xtr.shape[1], n_out).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    tX, ty, tk = (torch.from_numpy(a).to(device) for a in (Xtr, ytr.astype(np.float32), ktr))
    vX, vy, vk = (torch.from_numpy(a).to(device) for a in (Xva, yva.astype(np.float32), kva))
    g = torch.Generator(device="cpu").manual_seed(seed)
    best, state, bad = np.inf, None, 0
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(tX.shape[0], generator=g).to(device)
        for a in range(0, perm.numel(), bs):
            b = perm[a:a + bs]
            logit = net(tX[b]).gather(1, tk[b, None]).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logit, ty[b])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = F.binary_cross_entropy_with_logits(net(vX).gather(1, vk[:, None]).squeeze(1), vy).item()
        if vl < best - 1e-5:
            best, bad, state = vl, 0, {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state)
    return net


@torch.no_grad()
def mlp_predict(net, X, k, device):
    net.eval()
    out = net(torch.from_numpy(X).to(device)).gather(1, torch.from_numpy(k).to(device)[:, None])
    return torch.sigmoid(out.squeeze(1)).double().cpu().numpy().clip(1e-7, 1 - 1e-7)


# ------------------------------------------------------------------ 3D-U-Net, mehrere dt
class UNet3DMulti(UNet3D):
    def __init__(self, c_in: int, n_out: int, width: int = 32):
        super().__init__(c_in, width)
        self.head = nn.Linear(width, n_out)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, -0.693)

    def forward(self, x):                                  # -> (B, n_out)
        s1 = self.enc1(x)
        s2 = self.enc2(F.max_pool3d(s1, 2, ceil_mode=True))
        b = self.bott(F.max_pool3d(s2, 2, ceil_mode=True))
        d2 = self.dec2(torch.cat([F.interpolate(b, size=s2.shape[2:], mode="trilinear",
                                                align_corners=False), s2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=s1.shape[2:], mode="trilinear",
                                                align_corners=False), s1], 1))
        return self.head(self.drop(d1.mean(dim=(2, 3, 4))))


def pair_batch(X, rows, ks, L, mean, std, device):
    """Fenster je (Stichprobe, dt) einsammeln -> (B, 55, 30, 13, 13); X numpy oder GPU-Tensor."""
    starts = [window(L, DTS[k])[0] for k in ks]
    xb = as_float(gather_windows(X, rows, starts, T), device).permute(0, 2, 1, 3, 4)
    m = torch.from_numpy(mean).to(device)[None, :, None, None, None]
    s = torch.from_numpy(std).to(device)[None, :, None, None, None]
    return (xb - m) / s


def fit_unet_multi(X, y, tr, va, L, mean, std, device, seed, epochs=40, patience=6, bs=64):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = UNet3DMulti(55, len(DTS)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    pairs_tr = np.array([(r, k) for r in tr for k in range(len(DTS))])
    best, state, bad = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(pairs_tr))
        tl = tn = 0
        for a in range(0, order.size, bs):
            p = pairs_tr[order[a:a + bs]]
            xb = pair_batch(X, p[:, 0], p[:, 1], L, mean, std, device)
            kb = torch.from_numpy(p[:, 1]).to(device)
            yb = torch.from_numpy(y[p[:, 0]].astype(np.float32)).to(device)
            loss = F.binary_cross_entropy_with_logits(net(xb).gather(1, kb[:, None]).squeeze(1), yb)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
            tl += loss.item() * len(p); tn += len(p)
        pv = {k: unet_multi_predict(net, X, va, k, L, mean, std, device) for k in range(len(DTS))}
        yv = y[va]
        vl = float(np.mean([-np.mean(yv * np.log(pv[k]) + (1 - yv) * np.log(1 - pv[k]))
                            for k in pv]))
        print(f"      Epoche {ep:2d}: train {tl / tn:.4f}  val BCE (Mittel ueber dt) {vl:.4f}", flush=True)
        if vl < best - 1e-4:
            best, bad, state = vl, 0, {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state)
    return net


@torch.no_grad()
def unet_multi_predict(net, X, rows, k, L, mean, std, device, bs=128):
    net.eval()
    out = []
    for a in range(0, rows.size, bs):
        r = rows[a:a + bs]
        xb = pair_batch(X, r, np.full(r.size, k), L, mean, std, device)
        out.append(torch.sigmoid(net(xb)[:, k]).double().cpu().numpy())
    return np.concatenate(out).clip(1e-7, 1 - 1e-7)


# ------------------------------------------------------------------ Hauptteil
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--test-label", choices=["paper", "centre"], default="paper")
    ap.add_argument("--methods", nargs="+",
                    default=["persistence", "unet3d_dt", "hgb_dt", "mlp_dt", "mlp_multi", "unet3d_multi"])
    ap.add_argument("--channels", choices=list(CHANNELS), default="paper55",
                    help="Eingabekanaele; Labels bleiben die des Caches")
    ap.add_argument("--unet-run", default="_paperdt",
                    help="Lauf von 42_paper_reconstruct, dessen Vorhersagen unet3d_dt liest "
                         "(Datei rec_<cache>_cache<unet-run><Kanal-Tag>.npz)")
    ap.add_argument("--tag", default="", help="Zusatz am Manifestnamen")
    ap.add_argument("--extra", choices=extra.MODES, default="none",
                    help="zusaetzliche Train-Stichproben aus 2000-2019 (paper_extra.py) anhaengen")
    ap.add_argument("--target", choices=targets.TARGETS, default="cache",
                    help="cache: Feuer an D; next_patch: Feuer an D+1, Fenster einen Tag spaeter (targets.py)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t_load = time.time()
    d = exp / "derived_data" / "paper" / f"{args.cache}_L61"
    X = np.load(d / "X.npy")                                 # (N, 61, 55, 13, 13)
    y = targets.labels(d, args.target)
    split = np.load(d / "split.npy")
    if args.extra != "none":
        X, y, split = extra.append(X, y, split, extra.load(exp, args.target, args.extra))
    L = X.shape[1] + targets.shift(args.target)          # effektive Laenge: Zieltag = Index L - 1
    idx = {s: np.where(split == s)[0] for s in ("train", "val", "test")}
    print(f"[cmp] {args.cache}: X {X.shape} in {time.time() - t_load:.0f} s auf {device}")
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / args.cache / "norm_stats.npz")
    std = masked_std(std, args.channels)                      # gestrichene Kanaele -> Eingabe 0
    cols = summary_columns(args.channels)
    ctag = channel_tag(args.channels)
    print(f"[cmp] Kanalsatz {args.channels}: {len(CHANNELS[args.channels])} Kanaele", flush=True)

    res: dict = {}

    def record(method, dt_, per_seed):
        agg = {k: {"mean": float(np.mean([m[k] for m in per_seed])),
                   "std": float(np.std([m[k] for m in per_seed]))} for k in KEYS}
        res.setdefault(method, {})[str(dt_)] = {"summary": agg, "per_seed": per_seed}
        print(f"[cmp] {method:13s} dt={dt_:2d}  " + "  ".join(
            f"{k} {agg[k]['mean']:.3f}" for k in ("f1", "pr_auc", "roc_auc", "ece", "brier")), flush=True)

    yt = y[idx["test"]]
    if args.test_label == "centre":
        yt = targets.centre_labels(d, X, args.target)[idx["test"]]
        print(f"[cmp] Mittelzellen-Label auf dem Test: {yt.sum()} positiv von {yt.size}", flush=True)
    # Zusammenfassungen je dt, standardisiert mit TRAIN
    feats = {}
    for dt_ in DTS:
        t0, t1 = window(L, dt_)
        F_ = {s: summaries(X, idx[s], t0, t1)[:, cols] for s in idx}
        m_, s_ = F_["train"].mean(0), F_["train"].std(0) + 1e-6
        feats[dt_] = {s: ((F_[s] - m_) / s_).astype(np.float32) for s in idx}

    if "persistence" in args.methods and not has_fire(args.channels):
        print("[cmp] persistence uebersprungen: sie besteht nur aus der Feuermaske", flush=True)
    elif "persistence" in args.methods:
        for dt_ in DTS:
            t0, t1 = window(L, dt_)
            ff = {s: fire_fraction(X, idx[s], t0, t1) for s in ("train", "test")}
            pl = LogisticRegression().fit(ff["train"][:, None], y[idx["train"]])
            record("persistence", dt_, [paper_metrics(pl.predict_proba(ff["test"][:, None])[:, 1], yt)])

    if "unet3d_dt" in args.methods:
        z_path = exp / "predictions" / "paper" / f"rec_{args.cache}_cache{args.unet_run}{ctag}.npz"
        if z_path.exists():
            z = np.load(z_path)
            pos = {int(i): n for n, i in enumerate(z["idx_test"])}
            sel = np.array([pos[int(i)] for i in idx["test"]])
            for dt_ in DTS:
                ks = [k for k in z.files
                      if k.startswith(f"{args.channels}_T30_end{dt_ + 1 - targets.shift(args.target)}_s") and k.endswith("_test")]
                if not ks:
                    raise SystemExit(f"ABBRUCH: {z_path} hat keine Vorhersagen {args.channels}_T30_end{dt_ + 1 - targets.shift(args.target)}")
                record("unet3d_dt", dt_, [paper_metrics(z[k][sel], yt) for k in sorted(ks)])
        else:
            print(f"[cmp] WARNUNG: {z_path} fehlt - unet3d_dt uebersprungen")

    if "hgb_dt" in args.methods:
        for dt_ in DTS:
            per = []
            for seed in args.seeds:
                h = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                                   min_samples_leaf=20, l2_regularization=1.0,
                                                   early_stopping=True, validation_fraction=0.15,
                                                   random_state=seed).fit(feats[dt_]["train"], y[idx["train"]])
                per.append(paper_metrics(h.predict_proba(feats[dt_]["test"])[:, 1], yt))
            record("hgb_dt", dt_, per)

    if "mlp_dt" in args.methods:
        for dt_ in DTS:
            per = []
            for seed in args.seeds:
                net = fit_mlp(feats[dt_]["train"], y[idx["train"]], np.zeros(idx["train"].size, np.int64),
                              feats[dt_]["val"], y[idx["val"]], np.zeros(idx["val"].size, np.int64),
                              1, seed, device)
                per.append(paper_metrics(mlp_predict(net, feats[dt_]["test"],
                                                     np.zeros(yt.size, np.int64), device), yt))
            record("mlp_dt", dt_, per)

    if "mlp_multi" in args.methods:
        def stack(part):
            Xs = np.concatenate([feats[dt_][part] for dt_ in DTS])
            ys = np.concatenate([y[idx[part]]] * len(DTS))
            ks = np.concatenate([np.full(idx[part].size, k, np.int64) for k in range(len(DTS))])
            return Xs, ys, ks
        per = {dt_: [] for dt_ in DTS}
        for seed in args.seeds:
            net = fit_mlp(*stack("train"), *stack("val"), len(DTS), seed, device)
            for k, dt_ in enumerate(DTS):
                per[dt_].append(paper_metrics(mlp_predict(net, feats[dt_]["test"],
                                                          np.full(yt.size, k, np.int64), device), yt))
        for dt_ in DTS:
            record("mlp_multi", dt_, per[dt_])

    if "unet3d_multi" in args.methods:
        per = {dt_: [] for dt_ in DTS}
        Xg = to_device_once(X, device)
        for seed in args.seeds:
            print(f"    unet3d_multi seed {seed}", flush=True)
            net = fit_unet_multi(Xg, y, idx["train"], idx["val"], L, mean, std, device, seed)
            for k, dt_ in enumerate(DTS):
                per[dt_].append(paper_metrics(
                    unet_multi_predict(net, Xg, idx["test"], k, L, mean, std, device), yt))
        del Xg
        for dt_ in DTS:
            record("unet3d_multi", dt_, per[dt_])

    print(f"\n[cmp] {args.cache} TEST (Mittel ueber Seeds): " + "  ".join(KEYS))
    for dt_ in DTS:
        for mth in args.methods:
            if mth in res and str(dt_) in res[mth]:
                a = res[mth][str(dt_)]["summary"]
                print(f"    dt={dt_:2d} {mth:13s} " + "  ".join(f"{a[k]['mean']:.3f}" for k in KEYS))

    name = f"p5_compare_{args.cache}" + ("_centre" if args.test_label == "centre" else "") + targets.tag(args.target) + extra.tag(args.extra) + ctag + args.tag
    write_manifest(cfg, name, args.config, {
        "test_label": args.test_label, "target": args.target, "extra": args.extra,
        "cache": args.cache, "dts": list(DTS), "T": T, "seeds": args.seeds,
        "labels": "Cache-Labels (Paper-GT)", "channels": args.channels,
        "channel_list": CHANNELS[args.channels], "unet3d_dt_from": f"rec_{args.cache}_cache{args.unet_run}{ctag}.npz",
        "metric_assumptions": ASSUMPTIONS, "results": res,
    })


if __name__ == "__main__":
    main()
