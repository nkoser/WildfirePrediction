"""Horizont-BCE-Zusatzterm (paper/horizon_bce.py) fuer Contagion-STS und die Vision-Rumpfe U-TAE / ConvLSTM.

horizon_bce.py laesst nur --head lh mit dem STS-Rumpf zu. Dieser Wrapper nutzt denselben Verlust
(horizon_bce.head_loss: Intervall-NLL + beta * sum_tau BCE(F(tau), 1{T <= tau})) und erlaubt zusaetzlich
  --head contagion       F(tau) = 1 - S(tau) aus der Kurve des Contagion-Kopfs (CurveHead, Pixel und Ausschnitt)
  --backbone utae|convlstm  Rumpf aus vision_backbones.builder statt des STS-Transformers (nur STS-Koepfe)

Bestehende Module bleiben unveraendert; die Checkpoints laden wie bisher (Contagion direkt, Vision ueber den
Lade-Wrapper von vision_backbones).
  python -m wildfire.paper.horizon_bce_any train --hz-beta 1.0 [--hz-taus 1 7 14 30] [--backbone utae] <sts_dctm-Argumente>
"""

from __future__ import annotations

import argparse
import sys

from wildfire.paper import horizon_bce, sts_dctm


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] != "train":
        raise SystemExit(__doc__)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--hz-beta", type=float, required=True)
    ap.add_argument("--hz-taus", type=int, nargs="+", default=list(horizon_bce.CFG["taus"]))
    ap.add_argument("--backbone", choices=("sts", "utae", "convlstm"), default="sts")
    own, rest = ap.parse_known_args(sys.argv[2:])
    head = rest[rest.index("--head") + 1] if "--head" in rest else "dctm"
    if head not in ("lh", "contagion"):
        raise SystemExit("ABBRUCH: horizon_bce_any nur mit --head lh oder contagion")
    if own.backbone != "sts":
        if head != "lh":
            raise SystemExit("ABBRUCH: Vision-Rumpf nur mit --head lh")
        from wildfire.paper import vision_backbones
        sts_dctm.build_sts = vision_backbones.builder(own.backbone)
    horizon_bce.CFG.update(beta=own.hz_beta, taus=tuple(sorted(own.hz_taus)))
    print(f"[hzbce] Zusatzverlust beta {own.hz_beta} an tau {horizon_bce.CFG['taus']}, Kopf {head}, "
          f"Rumpf {own.backbone}", flush=True)
    sts_dctm.head_loss = horizon_bce.head_loss
    sys.argv = ["sts_dctm"] + rest
    sts_dctm.main()


if __name__ == "__main__":
    main()
