"""Paper-Protokoll, Schritt 7: unsere Methode (ST-DCTM-Architektur) im Setup des Papers.

Gleich wie in compare.py: Stichproben, Labels und Splits aus dem Cache, alle 55
Kanaele, 30-Tage-Fenster, Verschiebe-Protokoll (Fenster endet bei t - dt - 1),
Paper-Metriken auf dem Test.

Architektur (Vorschlag des Nutzers, 13 x 13 durch den Cache vorgegeben):

  1  Tages-Encoder: ein CNN, fuer alle Tage geteilt. Je Tag ein Token aus
     [Mittelzelle || gepoolter Nachbarschaft] plus Einbettung des Tagesabstands.
     Die 29 Vergangenheitstage bilden das Gedaechtnis.
  2  Abfrage: der letzte Fenstertag, gleich zusammengefasst.
  3  Ein Self-Attention-Block ueber das Gedaechtnis, dann zwei Cross-Attention-
     Bloecke, in denen die Abfrage das Gedaechtnis liest -> z.
  4  Kopf in Bernstein-Basis ueber den Vorlauf k = dt + 1:
         logit p(k | x) = sum_m theta_m(z) * B_m(k / K),   K = 31
     Auf den Paper-Labels (Feuer am Tag t, laufende Braende eingeschlossen) ist
     das die Wahrscheinlichkeit fuer Feuer an genau diesem Tag, keine Verteilungs-
     funktion; deshalb ohne Monotonie. Der echte Survival-DCTM-Kopf (monotone
     F(tau), zensierte Likelihood) gehoert auf die NFDB-Zuendungen.

Ein Modell fuer jeden Vorlauf. Training wahlweise nur mit den vier Paper-dt
(wie M4/M5) oder mit zufaelligem dt in 1..30 (nur mit diesem Kopf moeglich).
Ausgewertet an den Paper-dt und an Zwischenwerten.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wildfire.audit.common import assert_readonly, load_config, write_manifest
from wildfire.models.bernstein import bernstein_basis
from wildfire.paper.metrics import ASSUMPTIONS, paper_metrics
from wildfire.paper.reconstruct import cache_norm

T = 30
PAPER_DTS = (1, 7, 15, 30)
EXTRA_DTS = (3, 10, 20)
K_MAX = 31.0
CENTRE = 6
KEYS = ("f1", "pr_auc", "roc_auc", "precision", "recall", "ece", "mce", "brier")


class DayEncoder(nn.Module):
    """(B*T, C, 13, 13) -> (B*T, d): Mittelzelle || Mittel- und Max-Pooling."""

    def __init__(self, c_in: int, w: int = 64, d: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, w, 3, padding=1), nn.GroupNorm(8, w), nn.SiLU(),
            nn.Conv2d(w, w, 3, padding=1), nn.GroupNorm(8, w), nn.SiLU(),
            nn.Conv2d(w, w, 3, padding=1), nn.GroupNorm(8, w), nn.SiLU(),
        )
        self.proj = nn.Linear(3 * w, d)

    def forward(self, x):
        f = self.conv(x)
        centre = f[:, :, CENTRE, CENTRE]
        return self.proj(torch.cat([centre, f.mean((2, 3)), f.amax((2, 3))], 1))


class CrossBlock(nn.Module):
    def __init__(self, d: int, heads: int = 4, p: float = 0.1):
        super().__init__()
        self.att = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.SiLU(), nn.Dropout(p), nn.Linear(2 * d, d))

    def forward(self, q, mem):
        a, _ = self.att(self.n1(q), mem, mem, need_weights=False)
        q = q + a
        return q + self.ff(self.n2(q))


class STDCTM(nn.Module):
    def __init__(self, c_in: int = 55, d: int = 128, degree: int = 8):
        super().__init__()
        self.enc = DayEncoder(c_in, d=d)
        self.q_enc = DayEncoder(c_in, d=d)                  # eigene Gewichte fuer die Gegenwart
        self.time = nn.Embedding(T, d)                       # Tagesabstand zum letzten Fenstertag
        self.mem_block = nn.TransformerEncoderLayer(d, 4, 2 * d, 0.1, batch_first=True,
                                                    norm_first=True)
        self.cross = nn.ModuleList([CrossBlock(d) for _ in range(2)])
        self.out_norm = nn.LayerNorm(d)
        self.degree = degree
        self.head = nn.Linear(d, degree + 1)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, -0.693)           # logit(1/3) bei jedem Vorlauf

    def forward(self, x, k):
        """x: (B, T, C, 13, 13) normiert, k: (B,) Vorlauf in Tagen -> Logit (B,)."""
        B = x.shape[0]
        mem = self.enc(x[:, :-1].flatten(0, 1)).view(B, T - 1, -1)
        lag = torch.arange(T - 1, 0, -1, device=x.device)    # 29 ... 1 Tage vor dem letzten Fenstertag
        mem = self.mem_block(mem + self.time(lag)[None])
        q = (self.q_enc(x[:, -1]) + self.time.weight[0])[:, None]
        for blk in self.cross:
            q = blk(q, mem)
        theta = self.head(self.out_norm(q[:, 0]))            # (B, M+1)
        basis = bernstein_basis((k.float() / K_MAX).clamp(0, 1), self.degree)  # (B, M+1)
        return (theta * basis).sum(-1)


def batch(X, rows, dts, L, mean, std, device):
    xb = np.stack([X[r, L - d - 1 - T:L - d - 1] for r, d in zip(rows, dts)])   # (B, T, C, H, W)
    xb = torch.from_numpy(xb).to(device).float()
    m = torch.from_numpy(mean).to(device)[None, None, :, None, None]
    s = torch.from_numpy(std).to(device)[None, None, :, None, None]
    return (xb - m) / s


@torch.no_grad()
def predict(net, X, rows, dt, L, mean, std, device, bs=128):
    net.eval()
    out = []
    for a in range(0, rows.size, bs):
        r = rows[a:a + bs]
        xb = batch(X, r, np.full(r.size, dt), L, mean, std, device)
        k = torch.full((r.size,), dt + 1, device=device)
        out.append(torch.sigmoid(net(xb, k)).double().cpu().numpy())
    return np.concatenate(out).clip(1e-7, 1 - 1e-7)


def fit(X, y, idx, L, mean, std, device, seed, dt_mode, epochs=40, patience=6, bs=64, lr=5e-4):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = STDCTM().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    tr = idx["train"]
    best, state, bad, hist = np.inf, None, 0, []
    for ep in range(epochs):
        net.train()
        # je Epoche jede Trainingsstichprobe viermal, mit Paper-dt oder zufaelligem dt
        rows = np.repeat(tr, len(PAPER_DTS))
        dts = (np.tile(PAPER_DTS, tr.size) if dt_mode == "paper4"
               else rng.integers(1, 31, size=rows.size))
        order = rng.permutation(rows.size)
        tl = tn = 0
        for a in range(0, order.size, bs):
            o = order[a:a + bs]
            xb = batch(X, rows[o], dts[o], L, mean, std, device)
            kb = torch.from_numpy(dts[o] + 1).to(device)
            yb = torch.from_numpy(y[rows[o]].astype(np.float32)).to(device)
            loss = F.binary_cross_entropy_with_logits(net(xb, kb), yb)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
            tl += loss.item() * len(o); tn += len(o)
        yv = y[idx["val"]]
        vl = float(np.mean([-np.mean(yv * np.log(p) + (1 - yv) * np.log(1 - p))
                            for p in (predict(net, X, idx["val"], d, L, mean, std, device)
                                      for d in PAPER_DTS)]))
        hist.append({"epoch": ep, "train_bce": tl / tn, "val_bce": vl})
        print(f"      Epoche {ep:2d}: train {tl / tn:.4f}  val BCE (Paper-dt) {vl:.4f}", flush=True)
        if vl < best - 1e-4:
            best, bad, state = vl, 0, {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state)
    return net, hist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--dt-modes", nargs="+", default=["paper4", "all30"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    exp = Path(cfg["experiment_root"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    d = exp / "derived_data" / "paper" / f"{args.cache}_L61"
    X = np.load(d / "X.npy")
    y = np.load(d / "y.npy").astype(int)
    split = np.load(d / "split.npy")
    L = X.shape[1]
    idx = {s: np.where(split == s)[0] for s in ("train", "val", "test")}
    mean, std, _ = cache_norm(Path(cfg["dataset_root"]) / args.cache / "norm_stats.npz")
    n_par = sum(p.numel() for p in STDCTM().parameters())
    print(f"[st] {args.cache}: X {X.shape} in {time.time() - t0:.0f} s auf {device}, "
          f"Modell {n_par / 1e6:.2f} Mio. Parameter")

    yt = y[idx["test"]]
    res: dict = {}
    for mode in args.dt_modes:
        per = {dt: [] for dt in PAPER_DTS + EXTRA_DTS}
        hists = []
        for seed in args.seeds:
            print(f"\n[st] {mode} seed {seed}", flush=True)
            net, hist = fit(X, y, idx, L, mean, std, device, seed, mode)
            hists.append(hist)
            for dt in PAPER_DTS + EXTRA_DTS:
                per[dt].append(paper_metrics(predict(net, X, idx["test"], dt, L, mean, std, device), yt))
        res[mode] = {"histories": hists, "dt": {}}
        for dt in PAPER_DTS + EXTRA_DTS:
            agg = {k: {"mean": float(np.mean([m[k] for m in per[dt]])),
                       "std": float(np.std([m[k] for m in per[dt]]))} for k in KEYS}
            res[mode]["dt"][str(dt)] = {"summary": agg, "per_seed": per[dt],
                                        "trained_on_this_dt": dt in PAPER_DTS or mode == "all30"}
            print(f"[st] {mode:6s} dt={dt:2d}  " + "  ".join(
                f"{k} {agg[k]['mean']:.3f}" for k in ("f1", "pr_auc", "roc_auc", "ece", "brier")), flush=True)

    write_manifest(cfg, f"p6_stdctm_{args.cache}", args.config, {
        "cache": args.cache, "T": T, "paper_dts": list(PAPER_DTS), "extra_dts": list(EXTRA_DTS),
        "model": "STDCTM: Tages-CNN -> Gedaechtnis (29 Tokens) -> Self-Attention -> 2x Cross-Attention "
                 "-> Bernstein-Kopf ueber den Vorlauf (Grad 8, K = 31), BCE",
        "head_note": "auf Paper-Labels Wahrscheinlichkeit fuer Feuer genau am Tag t, nicht monoton",
        "seeds": args.seeds, "metric_assumptions": ASSUMPTIONS, "results": res,
    })


if __name__ == "__main__":
    main()
