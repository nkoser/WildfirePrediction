"""Paper-Protokoll, Schritt 2: Modelle trainieren und mit den Paper-Metriken auswerten.

Aufgabe wie im Paper: Fuer Ort s und Tag t die Wahrscheinlichkeit einer
Zuendung aus den T vorangehenden Tagen, binaere Klassifikation mit Sigmoid und
binaerer Kreuzentropie, Splits und 1:2-Verhaeltnis aus der Cache-JSON.

Modelle (die Modelle des Papers selbst liegen uns nicht vor):

  rule_fire   kein Lernen: Anteil der T Tage, an denen die Feuermaske irgendwo
              im 13x13-Ausschnitt brennt; auf TRAIN per Platt-Skalierung in
              eine Wahrscheinlichkeit uebersetzt. Misst, wie viel die
              Feuermaske im Eingabefenster allein verraet (Audit s6).
  logreg      Zeitreihenmodell, Eingabe X^TS in R^{T x C} am Zentrumspixel,
              flach, standardisiert.
  hgb         dieselbe Eingabe, Gradient Boosting.
  stnet       raeumlich-zeitlich, Eingabe X^ST in R^{T x C x 13 x 13}: geteilter
              Faltungs-Encoder je Tag, GRU ueber die Zeit, ein Logit.

Kanalsaetze:

  paper        alle 55 Kanaele - so wie im Cache und damit wie im Paper
  no_fire      ohne Kanal 0 (MODIS-Feuermaske)
  no_fire_bt   zusaetzlich ohne die Helligkeitstemperaturen (Kanaele 50-53),
               aus denen MODIS Feuer erkennt

Normalisierung: Mittel und Streuung je Kanal aus TRAIN; Feuermaske (0/1) und
Landbedeckung (Klasse) bleiben unnormiert, wie die norm_stats des Caches es
vorsehen. Die Landbedeckung wird durch 17 geteilt.
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
from sklearn.preprocessing import StandardScaler

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.paper.metrics import ASSUMPTIONS, paper_metrics
from wildfire.paper.prepare import BT_CH, CHANNEL_NAMES, FIRE_CH, LANDCOVER_CH

CHANNEL_SETS = {
    "paper": [c for c in range(55)],
    "no_fire": [c for c in range(55) if c != FIRE_CH],
    "no_fire_bt": [c for c in range(55) if c != FIRE_CH and c not in BT_CH],
}
CENTRE = 6


# ------------------------------------------------------------------ Normierung
def channel_stats(X: np.ndarray, idx: np.ndarray, chunk: int = 256):
    """Mittel und Streuung je Kanal ueber (N, T, H, W), nur auf den Zeilen idx."""
    C = X.shape[2]
    s, ss, n = np.zeros(C), np.zeros(C), 0
    for a in range(0, idx.size, chunk):
        b = X[idx[a:a + chunk]].astype(np.float64)          # (n, T, C, H, W)
        s += b.sum(axis=(0, 1, 3, 4))
        ss += (b ** 2).sum(axis=(0, 1, 3, 4))
        n += b.shape[0] * b.shape[1] * b.shape[3] * b.shape[4]
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean ** 2, 1e-12))
    mean[FIRE_CH], std[FIRE_CH] = 0.0, 1.0
    mean[LANDCOVER_CH], std[LANDCOVER_CH] = 0.0, 17.0
    return mean.astype(np.float32), std.astype(np.float32)


# ------------------------------------------------------------------ Modelle
class STNet(nn.Module):
    """Geteilter Faltungs-Encoder je Tag, GRU ueber die Zeit, ein Logit."""

    def __init__(self, c_in: int, width: int = 64, init_logit: float = -0.693):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(c_in, width, 3, padding=1), nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1, stride=2), nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.SiLU(),
        )
        self.gru = nn.GRU(width, width, batch_first=True)
        self.head = nn.Linear(width, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, init_logit)      # logit(1/3): Train-Anteil

    def forward(self, x):                                   # (B, T, C, H, W)
        B, T = x.shape[:2]
        z = self.enc(x.flatten(0, 1))                       # (B*T, w, h, w)
        z = z.mean(dim=(2, 3)).view(B, T, -1)               # (B, T, w)
        _, hT = self.gru(z)
        return self.head(hT[-1]).squeeze(-1)


def batches(X, idx, chs, mean, std, bs, device, shuffle=False, rng=None):
    order = rng.permutation(idx) if shuffle else idx
    m = torch.from_numpy(mean[chs]).to(device)[None, None, :, None, None]
    s = torch.from_numpy(std[chs]).to(device)[None, None, :, None, None]
    for a in range(0, order.size, bs):
        sel = np.sort(order[a:a + bs])
        xb = torch.from_numpy(X[sel][:, :, chs]).to(device).float()
        yield sel, (xb - m) / s


def train_stnet(X, y, tr, va, chs, mean, std, device, seed, max_epochs=40, patience=6,
                bs=128, lr=1e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = STNet(len(chs)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    yt = torch.from_numpy(y.astype(np.float32)).to(device)
    best, best_state, bad, hist = np.inf, None, 0, []
    for ep in range(max_epochs):
        net.train()
        tl, tn = 0.0, 0
        for sel, xb in batches(X, tr, chs, mean, std, bs, device, shuffle=True, rng=rng):
            loss = F.binary_cross_entropy_with_logits(net(xb), yt[sel])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tl += loss.item() * len(sel); tn += len(sel)
        pv = predict_stnet(net, X, va, chs, mean, std, device)
        vl = float(F.binary_cross_entropy(torch.from_numpy(pv), torch.from_numpy(
            y[va].astype(np.float64))).item())
        hist.append({"epoch": ep, "train_bce": tl / tn, "val_bce": vl})
        print(f"      Epoche {ep:2d}: train BCE {tl / tn:.4f}  val BCE {vl:.4f}", flush=True)
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return net, hist


@torch.no_grad()
def predict_stnet(net, X, idx, chs, mean, std, device, bs=256):
    net.eval()
    out = np.zeros(idx.size)
    pos = {int(i): k for k, i in enumerate(idx)}
    for sel, xb in batches(X, idx, chs, mean, std, bs, device):
        p = torch.sigmoid(net(xb)).double().cpu().numpy()
        for i, v in zip(sel, p):
            out[pos[int(i)]] = v
    return np.clip(out, 1e-7, 1 - 1e-7)


def centre_series(X, idx, chs):
    """X^TS: (n, T * c) am Zentrumspixel."""
    # erst das Zentrum als View (N, T, C), dann die Zeilen - sonst kopiert
    # X[idx] den ganzen Ausschnitt aller Kanaele (Gigabytes)
    return X[:, :, :, CENTRE, CENTRE][idx][:, :, chs].reshape(idx.size, -1).astype(np.float32)


def fire_fraction(X, idx):
    """Anteil der Tage mit Feuer irgendwo im Ausschnitt (Kanal 0)."""
    f = X[idx, :, FIRE_CH] > 0                                # (n, T, H, W)
    return f.any(axis=(2, 3)).mean(axis=1).astype(np.float64)


# ------------------------------------------------------------------ Hauptteil
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--sets", nargs="+", default=list(CHANNEL_SETS))
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    d = exp / "derived_data" / "paper" / args.cache
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    X = np.load(d / "X.npy")                                # in den RAM, ~7 GB
    y = np.load(d / "y.npy").astype(int)
    split = np.load(d / "split.npy")
    idx = {s: np.where(split == s)[0] for s in ("train", "val", "test")}
    print(f"[paper] {args.cache}: X {X.shape} in {time.time() - t0:.0f} s, "
          f"{ {s: (int(v.size), int(y[v].sum())) for s, v in idx.items()} } auf {device}")

    mean, std = channel_stats(X, idx["train"])
    results: dict = {}
    preds: dict = {}

    def record(cset, model, p_by_split, extra=None):
        res = {s: paper_metrics(p_by_split[s], y[idx[s]]) for s in ("val", "test")}
        if extra:
            res.update(extra)
        results.setdefault(cset, {})[model] = res
        for s in ("val", "test"):
            preds[f"{cset}__{model}__{s}"] = np.asarray(p_by_split[s], np.float32)
        t = res["test"]
        print(f"[paper] {args.cache} {cset:10s} {model:9s} TEST  P {t['precision']:.3f} "
              f"R {t['recall']:.3f} F1 {t['f1']:.3f} PR-AUC {t['pr_auc']:.3f} "
              f"ROC {t['roc_auc']:.3f} ECE {t['ece']:.3f} MCE {t['mce']:.3f} "
              f"Brier {t['brier']:.3f}", flush=True)

    # ---- Regel ohne Modell: nur mit Feuerkanal sinnvoll
    ff = {s: fire_fraction(X, idx[s]) for s in idx}
    platt = LogisticRegression().fit(ff["train"][:, None], y[idx["train"]])
    record("paper", "rule_fire", {s: platt.predict_proba(ff[s][:, None])[:, 1]
                                  for s in ("val", "test")},
           {"share_any_fire_in_window": {s: {
               "positive": float((ff[s][y[idx[s]] == 1] > 0).mean()),
               "negative": float((ff[s][y[idx[s]] == 0] > 0).mean())} for s in idx}})

    for cset in args.sets:
        chs = CHANNEL_SETS[cset]
        print(f"\n[paper] Kanalsatz {cset}: {len(chs)} Kanaele", flush=True)
        Xts = {s: centre_series(X, idx[s], chs) for s in idx}
        sc = StandardScaler().fit(Xts["train"])
        lr_ = LogisticRegression(max_iter=2000, C=0.1).fit(sc.transform(Xts["train"]),
                                                          y[idx["train"]])
        record(cset, "logreg", {s: lr_.predict_proba(sc.transform(Xts[s]))[:, 1]
                                for s in ("val", "test")})
        hgb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                             max_leaf_nodes=31, min_samples_leaf=20,
                                             l2_regularization=1.0, early_stopping=True,
                                             validation_fraction=0.15,
                                             random_state=int(cfg["seed"])).fit(
            Xts["train"], y[idx["train"]])
        record(cset, "hgb", {s: hgb.predict_proba(Xts[s])[:, 1] for s in ("val", "test")})

        # stnet: mehrere Seeds, berichtet wird das Mittel der Vorhersagen je Seed
        per_seed = []
        for seed in args.seeds:
            print(f"    stnet seed {seed}", flush=True)
            net, hist = train_stnet(X, y, idx["train"], idx["val"], chs, mean, std,
                                    device, seed)
            p = {s: predict_stnet(net, X, idx[s], chs, mean, std, device)
                 for s in ("val", "test")}
            per_seed.append({"seed": seed, "history": hist,
                             **{s: paper_metrics(p[s], y[idx[s]]) for s in ("val", "test")}})
            record(cset, f"stnet_s{seed}", p)
        keys = ("precision", "recall", "f1", "pr_auc", "roc_auc", "ece", "mce", "brier")
        results[cset]["stnet_summary"] = {
            s: {k: {"mean": float(np.mean([r[s][k] for r in per_seed])),
                    "std": float(np.std([r[s][k] for r in per_seed]))} for k in keys}
            for s in ("val", "test")}
        results[cset]["stnet_histories"] = [{"seed": r["seed"], "history": r["history"]}
                                            for r in per_seed]

    out = exp / "predictions" / "paper"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{args.cache}.npz", **preds,
                        y_val=y[idx["val"]], y_test=y[idx["test"]])
    write_manifest(cfg, f"p1_paper_{args.cache}", args.config, {
        "cache": args.cache, "lookback": int(X.shape[1]),
        "channel_sets": {k: [CHANNEL_NAMES[c] for c in v] for k, v in CHANNEL_SETS.items()},
        "metric_assumptions": ASSUMPTIONS,
        "split_sizes": {s: {"n": int(v.size), "pos": int(y[v].sum())} for s, v in idx.items()},
        "seeds": args.seeds,
        "results": results,
        "predictions": str(out / f"{args.cache}.npz"),
    })


if __name__ == "__main__":
    main()
