"""B4: diskretes Survival-Modell auf Pixelmerkmalen, ohne raeumlichen Kontext.

Der Arbeitsplan begruendet diese Baseline damit, dass sie "den Nutzen des
raeumlichen Encoders isoliert". Sie schliesst zwei Luecken zwischen den
bisherigen Modellen:

    B1 -> B4   dieselben 219 Merkmale, aber EIN Kopf mit 30 Hazards und
               zensierter Likelihood statt vier zusammengeklebter
               Klassifikatoren. Die Differenz misst den Beitrag der
               Survival-FORMULIERUNG.

    B4 -> B2   dasselbe Ziel und dieselbe Likelihood, aber mit
               33x33-Umgebung statt eines Pixels. Die Differenz misst den
               Beitrag des raeumlichen KONTEXTS.

Ohne B4 waere ein Vorsprung von B2 nicht zuzuordnen - er koennte ebenso
gut aus der besseren Zielformulierung stammen wie aus der Nachbarschaft.

**Gewichte im Verlust.** Anders als B1 wird hier GEWICHTET trainiert. Der
gewichtete Verlust ist ein erwartungstreuer Schaetzer derselben NLL, die
am Ende berichtet wird - das Modell optimiert also unmittelbar die
Zielgroesse, und es braucht keine nachtraegliche Prior-Korrektur. Der Preis
ist eine kleinere effektive Stichprobe; bei 400 000 Zeilen ist das
vertretbar.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from wildfire.audit.common import (assert_readonly, landmark_horizon,
                                   load_config, write_manifest)
from wildfire.eval.metrics import evaluate
from wildfire.experiment import register
from wildfire.models.losses import discrete_survival_nll


class MLPSurvival(nn.Module):
    def __init__(self, n_in: int, horizon: int = 30, width: int = 256,
                 depth: int = 3, dropout: float = 0.1, init_bias: float = -7.0):
        super().__init__()
        layers, c = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(c, width), nn.LayerNorm(width), nn.SiLU(),
                       nn.Dropout(dropout)]
            c = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(c, horizon)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, init_bias)

    def forward(self, x):
        return self.head(self.body(x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="temporal")
    ap.add_argument("--horizon", type=int, default=None,
                    help="Standard: aus dem Landmark-Manifest")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--patience", type=int, default=5)
    args = ap.parse_args()
    cfg = load_config(args.config)
    assert_readonly(cfg)
    if args.horizon is None:
        args.horizon = landmark_horizon(cfg)
        print(f"[cfg] Horizont aus dem Landmark-Manifest: {args.horizon}")
    torch.manual_seed(int(cfg["seed"]))

    exp = Path(cfg["experiment_root"])
    fdir = exp / "derived_data" / "features"
    d = {}
    for part in ("train", "val", "test"):
        z = np.load(fdir / f"{args.split}_{part}.npz", allow_pickle=False)
        d[part] = {k: z[k] for k in z.files}
    names = [str(x) for x in d["train"]["names"]]

    # dieselben Ausschluesse wie bei B1: 'year' ist auf einem zeitlichen
    # Split eine Extrapolationsfalle
    cols = [i for i, n in enumerate(names) if n != "year"]
    Xtr = np.nan_to_num(d["train"]["X"][:, cols])
    mu, sd = Xtr.mean(0), np.maximum(Xtr.std(0), 1e-6)
    print(f"[B4] {len(cols)} Merkmale, train={Xtr.shape[0]}")

    def prep(part):
        X = (np.nan_to_num(d[part]["X"][:, cols]) - mu) / sd
        return (torch.tensor(X, dtype=torch.float32),
                torch.tensor(d[part]["event"], dtype=torch.float32),
                torch.tensor(d[part]["t_event"], dtype=torch.long),
                torch.tensor(d[part]["weight"], dtype=torch.float32))

    Xt, Et, Tt, Wt = prep("train")
    Xv, Ev, Tv, Wv = prep("val")
    Xs, Es, Ts, Ws = prep("test")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPSurvival(len(cols), args.horizon, args.width).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f"[B4] {n_par/1e3:.0f}k Parameter auf {dev}")

    def eval_on(X, E, T, W):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, X.shape[0], 65536):
                out.append(torch.sigmoid(model(X[i:i + 65536].to(dev))).cpu().numpy())
        return evaluate(np.concatenate(out), E.numpy().astype(bool),
                        T.numpy(), W.numpy())

    best, bad, hist = float("inf"), 0, []
    n = Xt.shape[0]
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        run = 0.0
        for i in range(0, n - args.batch_size + 1, args.batch_size):
            b = perm[i:i + args.batch_size]
            opt.zero_grad(set_to_none=True)
            loss = discrete_survival_nll(model(Xt[b].to(dev)), Et[b].to(dev),
                                         Tt[b].to(dev), Wt[b].to(dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run += loss.item()
        sched.step()
        m = eval_on(Xv, Ev, Tv, Wv)
        hz = m["horizons"]["30"]
        hist.append({"epoch": ep, "train_loss": run / max(1, n // args.batch_size),
                     "val_nll": m["nll"], "val_auc30": hz["roc_auc"],
                     "val_pr30": hz["pr_auc"], "val_ece30": hz["ece"]})
        print(f"[B4] Epoche {ep:2d}: Verlust {hist[-1]['train_loss']:.5f} | "
              f"val NLL {m['nll']:.5f} AUC@30 {hz['roc_auc']:.4f} "
              f"PR@30 {hz['pr_auc']:.4f} ECE {hz['ece']:.2e}", flush=True)
        if m["nll"] < best:
            best, bad = m["nll"], 0
            torch.save({"model": model.state_dict(), "mu": mu, "sd": sd,
                        "cols": cols, "epoch": ep},
                       exp / "checkpoints" / "b4_best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[B4] Early Stopping nach Epoche {ep}")
                break

    ck = torch.load(exp / "checkpoints" / "b4_best.pt", weights_only=False)
    model.load_state_dict(ck["model"])
    res = {"val": eval_on(Xv, Ev, Tv, Wv), "test": eval_on(Xs, Es, Ts, Ws)}
    for part, m in res.items():
        hz = m["horizons"]
        print(f"[B4] {part:4s} NLL={m['nll']:.5f} IBS={m['integrated_brier']:.3e} "
              f"AUC@30={hz['30']['roc_auc']:.4f} PR@30={hz['30']['pr_auc']:.4f} "
              f"ECE@30={hz['30']['ece']:.3e}")

    register(exp, f"b4_mlp_survival_{args.split}", "baseline", args.config,
             split=args.split, seed=int(cfg["seed"]),
             checkpoint_path=str(exp / "checkpoints" / "b4_best.pt"),
             metrics_path=str(exp / "data_manifests" / "m4_mlp_survival.json"),
             notes=f"{n_par} Parameter, gewichteter Verlust, ohne raeumlichen Kontext",
             extra={"test_auc_at_30": res["test"]["horizons"]["30"]["roc_auc"]})
    write_manifest(cfg, "m4_mlp_survival", args.config,
                   {"split": args.split, "n_features": len(cols),
                    "n_parameters": n_par, "epochs_run": len(hist),
                    "weighted_loss": True,
                    "rationale": "isoliert Survival-Formulierung (vs B1) "
                                 "und raeumlichen Kontext (vs B2)",
                    "history": hist, "results": res})


if __name__ == "__main__":
    main()
