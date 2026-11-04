"""3D-U-Net des Papers mit dessen Abbruchregel: Early Stopping auf dem Validierungs-F1.

Das MPCFire-Manuskript: "fixed learning rate of 1e-4, binary cross-entropy on the
sampled FireMPC splits, mini-batch 512, each model trained from scratch, early
stopping on validation F1". reconstruct.py waehlt das Modell nach dem Val-BCE; hier
ist alles gleich (Netz, Daten, Fenster, AdamW, Gradienten-Clipping), nur Auswahl und
Abbruch laufen ueber F1 bei Schwelle 0,5 (wie paper_metrics). Gleiche Argumente wie
reconstruct.py; --select wird ignoriert, im Manifest steht "val_f1".

Die Schwelle und die Patience nennt das Paper nicht. Mit B 512 hat cache G nur 9
Schritte je Epoche; bei 0,5 bleibt der Val-F1 dann viele Epochen 0, und ohne Zusatz
endet jeder Lauf nach patience + 1 Epochen mit dem Modell aus Epoche 0 (97, Job 4702).
Zwei Lesarten dagegen:
  --f1-wait        die Patience zaehlt erst, wenn der Val-F1 einmal ueber 0 lag
  --f1-mode max    bester F1 ueber alle Schwellen (f1_max) statt F1 bei 0,5
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wildfire.models.ema import EMA
from wildfire.paper import reconstruct as rec
from wildfire.paper.metrics import paper_metrics

SELECTION = "val_f1 (Schwelle 0,5), Abbruch nach patience Epochen ohne Verbesserung"
F1_KEY = "f1"          # "f1_max" mit --f1-mode max
F1_WAIT = False        # --f1-wait


def fit_f1(X, y, idx, chs, t0, t1, mean, std, device, seed, epochs=40, patience=6, bs=64, lr=1e-3,
           select="val_f1", ema_decay=0.0):
    """Wie rec.fit, aber bestes Modell und Abbruch nach dem Val-F1 statt dem Val-BCE."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = rec.UNet3D(len(chs)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    ema = EMA(net, ema_decay) if ema_decay > 0 else None
    yt = torch.from_numpy(y.astype(np.float32)).to(device)
    best, state, bad, hist = -np.inf, None, 0, []
    for ep in range(epochs):
        net.train()
        tl = tn = 0
        for sel, xb in rec.loader(X, idx["train"], chs, t0, t1, mean, std, bs, device, rng):
            loss = F.binary_cross_entropy_with_logits(net(xb), yt[sel])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            if ema is not None:
                ema.update(net)
            tl += loss.item() * len(sel); tn += len(sel)
        live = None
        if ema is not None:                                 # Validierung mit den gemittelten Gewichten
            live = {k: v.detach().clone() for k, v in net.state_dict().items()}
            net.load_state_dict(ema.state(live))
        pv = rec.predict(net, X, idx["val"], chs, t0, t1, mean, std, device)
        cur = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if live is not None:
            net.load_state_dict(live)
        yv = y[idx["val"]]
        vl = float(-np.mean(yv * np.log(pv) + (1 - yv) * np.log(1 - pv)))
        vm = paper_metrics(pv, yv)
        hist.append({"epoch": ep, "train_bce": tl / tn, "val_bce": vl, "val_pr_auc": vm["pr_auc"],
                     "val_f1": vm["f1"], "val_f1_max": vm["f1_max"]})
        print(f"      Epoche {ep:2d}: train {tl / tn:.4f}  val BCE {vl:.4f}  val PR-AUC "
              f"{vm['pr_auc']:.3f}  val F1 {vm['f1']:.3f}  val F1max {vm['f1_max']:.3f}", flush=True)
        if vm[F1_KEY] > best + 1e-4:
            best, bad, state = vm[F1_KEY], 0, cur
        elif not (F1_WAIT and best <= 0):
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state)
    return net, hist


def write_manifest_f1(cfg, stage, config_path, payload):
    payload["training"]["selection"] = SELECTION
    return _write_manifest(cfg, stage, config_path, payload)


_write_manifest = rec.write_manifest

if __name__ == "__main__":
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--f1-mode", choices=["thr05", "max"], default="thr05")
    ap.add_argument("--f1-wait", action="store_true")
    own, rest = ap.parse_known_args()
    sys.argv = sys.argv[:1] + rest                       # Rest geht unveraendert an reconstruct.main
    if own.f1_mode == "max":
        F1_KEY = "f1_max"
        SELECTION = "val_f1_max (bester F1 ueber alle Schwellen), Abbruch nach patience Epochen ohne Verbesserung"
    if own.f1_wait:
        F1_WAIT = True
        SELECTION += "; Patience zaehlt erst ab dem ersten Val-F1 > 0"
    rec.fit = fit_f1
    rec.write_manifest = write_manifest_f1
    rec.main()
