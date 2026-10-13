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

**Zwei Koepfe im Vergleich.** Derselbe Rumpf, dieselben Merkmale, dieselbe
Ziehung - nur die Ausgabeschicht unterscheidet sich:

    discrete    H Logits, je einer pro Tag. Was B1 zusammengeklebt hat,
                lernt hier EIN Kopf gemeinsam mit zensierter Likelihood.
    bernstein   M+1 Koeffizienten einer monotonen Transformationsfunktion.
                Auswertbar bei JEDEM tau, nicht nur an Gitterpunkten.

Das ist Ablation 1 aus Phase 9 des Arbeitsplans: feste Horizonte gegen
diskreten Hazard gegen stetige Transformation. Weil sich nur der Kopf
unterscheidet, misst die Differenz genau ihn.

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
from wildfire.models.bernstein import BernsteinSurvivalHead, bernstein_nll
from wildfire.models.losses import discrete_survival_nll, init_hazard_logit


class MLPSurvival(nn.Module):
    def __init__(self, n_in: int, horizon: int = 100, width: int = 256,
                 depth: int = 3, dropout: float = 0.1, init_bias: float | None = None,
                 head: str = "discrete", degree: int = 12):
        super().__init__()
        layers, c = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(c, width), nn.LayerNorm(width), nn.SiLU(),
                       nn.Dropout(dropout)]
            c = width
        self.body = nn.Sequential(*layers)
        self.kind = head
        self.horizon = horizon
        if head == "bernstein":
            # Startwert steckt im Kopf selbst (F(H) = INIT_CIF_H bei Null-Ausgabe)
            self.bern = BernsteinSurvivalHead(degree=degree, tau_max=float(horizon))
            self.head = nn.Linear(c, self.bern.n_out)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        else:
            self.bern = None
            self.head = nn.Linear(c, horizon)
            nn.init.zeros_(self.head.weight)
            nn.init.constant_(self.head.bias, init_hazard_logit(horizon=horizon)
                              if init_bias is None else init_bias)

    def forward(self, x):
        return self.head(self.body(x))

    def hazards(self, x: torch.Tensor) -> torch.Tensor:
        """(N, H) diskrete Hazards - fuer beide Koepfe, damit die Metriken
        unveraendert bleiben und der Vergleich fair ist."""
        raw = self.forward(x)
        if self.kind == "discrete":
            return torch.sigmoid(raw)
        return self.bern.hazards(raw, self.horizon).clamp(0.0, 1.0 - 1e-7)


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
    ap.add_argument("--head", default="bernstein", choices=("discrete", "bernstein"))
    ap.add_argument("--degree", type=int, default=12)
    ap.add_argument("--eval-only", action="store_true",
                    help="vorhandenen Checkpoint auswerten statt neu zu trainieren")
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
    model = MLPSurvival(len(cols), args.horizon, args.width,
                        head=args.head, degree=args.degree).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    n_out = model.head.out_features
    print(f"[B4] Kopf '{args.head}' mit {n_out} Ausgaben, "
          f"{n_par/1e3:.0f}k Parameter auf {dev}")

    def eval_on(X, E, T, W):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, X.shape[0], 32768):
                out.append(model.hazards(X[i:i + 32768].to(dev))
                           .float().cpu().numpy().astype(np.float32))
        return evaluate(np.concatenate(out), E.numpy().astype(bool),
                        T.numpy(), W.numpy())

    best, bad, hist = float("inf"), 0, []
    n = Xt.shape[0]
    for ep in range(0 if args.eval_only else args.epochs):
        model.train()
        perm = torch.randperm(n)
        run = 0.0
        for i in range(0, n - args.batch_size + 1, args.batch_size):
            b = perm[i:i + args.batch_size]
            opt.zero_grad(set_to_none=True)
            raw = model(Xt[b].to(dev))
            if args.head == "bernstein":
                loss = bernstein_nll(model.bern, raw, Et[b].to(dev),
                                     Tt[b].to(dev), float(args.horizon),
                                     Wt[b].to(dev))
            else:
                loss = discrete_survival_nll(raw, Et[b].to(dev), Tt[b].to(dev),
                                             Wt[b].to(dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run += loss.item()
        sched.step()
        m = eval_on(Xv, Ev, Tv, Wv)
        hz = m["horizons"][str(min(30, args.horizon))]
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
                       exp / "checkpoints" / f"b4_{args.head}_best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[B4] Early Stopping nach Epoche {ep}")
                break

    ck = torch.load(exp / "checkpoints" / f"b4_{args.head}_best.pt",
                    weights_only=False)
    model.load_state_dict(ck["model"])
    res = {"val": eval_on(Xv, Ev, Tv, Wv), "test": eval_on(Xs, Es, Ts, Ws)}
    for part, m in res.items():
        hz = m["horizons"]
        k = str(min(30, args.horizon))
        print(f"[B4/{args.head}] {part:4s} NLL={m['nll']:.5f} "
              f"IBS={m['integrated_brier']:.3e} "
              f"AUC@30={hz[k]['roc_auc']:.4f} PR@30={hz[k]['pr_auc']:.4f} "
              f"ECE@30={hz[k]['ece']:.3e}")

    register(exp, f"b4_{args.head}_{args.split}", "baseline", args.config,
             split=args.split, seed=int(cfg["seed"]),
             checkpoint_path=str(exp / "checkpoints" / f"b4_{args.head}_best.pt"),
             metrics_path=str(exp / "data_manifests" / f"m4_{args.head}.json"),
             notes=(f"Kopf {args.head}, {n_out} Ausgaben, {n_par} Parameter, "
                    "gewichteter Verlust, ohne raeumlichen Kontext"),
             extra={"test_auc_at_30":
                    res["test"]["horizons"][str(min(30, args.horizon))]["roc_auc"]})
    write_manifest(cfg, f"m4_{args.head}", args.config,
                   {"split": args.split, "n_features": len(cols),
                    "head": args.head, "n_head_outputs": n_out,
                    "degree": args.degree if args.head == "bernstein" else None,
                    "horizon": args.horizon,
                    "n_parameters": n_par, "epochs_run": len(hist),
                    "weighted_loss": True,
                    "rationale": "isoliert Survival-Formulierung (vs B1) "
                                 "und raeumlichen Kontext (vs B2)",
                    "history": hist, "results": res})


if __name__ == "__main__":
    main()
